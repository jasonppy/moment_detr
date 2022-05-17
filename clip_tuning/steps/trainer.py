import time
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import qvh_dataset
from datasets.sampler import StatefulSampler
from model import clip_wrapper
import clip
from .utils import evaluate
from .trainer_utils import *
from .bert_adam import BertAdam
from apex.fp16_utils import *
from apex import amp
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)
class Trainer:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--exp_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--n_epochs", type=int)
        parser.add_argument("--n_print_steps", type=int)
        parser.add_argument("--n_val_steps", type=int)
        parser.add_argument("--take_snapshot_every_n_steps", type=int, default=None)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--warmup_fraction", type=float, default=0.1)
        parser.add_argument("--opt_level", type=str, default="O0", help="O0, O1, O2, O3. O0:fp32, O1:fp16+fp32 mixed, O2:almost fp16, O3:fp16")
        parser.add_argument('--model_version', type=str, default="ViT-B/32", choices=["ViT-B/32", "RN50x4"], help='Num parallel thread for video decoding')
        parser.add_argument("--logit_scale", type=float, default=0.07)
        parser.add_argument('--freeze_first_x', type=int, default=None, help="freeze the first x layers of CLIP")
        parser.add_argument("--num_positives", type=int, default=1)
        parser.add_argument('--in_video_negatives', type=int, default=30, help="how many negatives to sample within the same video")
        parser.add_argument('--in_batch_negatives', type=int, default=30, help="num negatives to sample within the batch, but not from the same video")
        parser.add_argument('--pyscenedetect_root', type=str, default="/saltpool0/data/pyp/vqhighlight")
        parser.add_argument('--val_gt_fn', type=str, default="/home/pyp/vqhighlight/data/highlight_val_release_exist.jsonl")
        parser.add_argument('--alpha', type=int, default=53, help="the threshold for the PySceneDetect")
    
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()
        self.progress, self.total_progress = setup_progress(self)
        self.model, self.trainables, self.indices, self.optim_states = self._setup_models()
        self.train_loader, self.valid_loader, self.train_sampler, self.train_data_length = self._setup_dataloader()
        self.total_num_updates = int(math.floor(self.train_data_length / self.args.batch_size))*self.args.n_epochs
        self.optimizer = self._setup_optimizer()
        self.model, self.optimizer = amp.initialize(models=self.model, optimizers=self.optimizer, opt_level=self.args.opt_level)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.scheduler = self._setup_scheduler()
        self.criterion = torch.nn.CrossEntropyLoss()
        logger.info(f"batch size: {self.args.batch_size}")
    
    def forward(self, batch):
        cur_batch = {
                "image": batch['image'].to(self.device),
                "vid": batch['vid'],
                "text": clip.tokenize(batch['text']).to(self.device),
                "gt_score": batch['gt_score'] if batch['gt_score'] == None else batch['gt_score'].to(self.device)
                }
        bsz, L, C, H, W = cur_batch["image"].shape
        cur_batch["image"] = cur_batch["image"].view(bsz*(self.args.in_video_negatives+self.args.num_positives), C, H, W) # we should have L = (self.args.in_video_negatives+self.args.num_positives)
        # print(f"(bsz, pos+in_vid_neg) = ({bsz}, {L})", )
        # print("train input shape", cur_batch['image'].shape)
        image_features, text_features = self.model(cur_batch["image"], cur_batch["text"]) # [bsz*(self.args.in_video_negatives+self.args.num_positives), D]
        # print("train output shape: ", image_features.shape)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # sample in batch negatives, and combine them with in video negatives
        image_features = image_features.view(bsz, L, image_features.shape[-1])
        # print(f"output from the model, (bsz, pos+in_vid_neg) = ({image_features.shape[0]}, {image_features.shape[1]})")
        neg_image_features = image_features.view(bsz*L, image_features.shape[-1])
        negative_pool = torch.arange(bsz*L)
        temp = torch.arange(bsz*L)
        if self.args.in_batch_negatives > 0:
            extended_image_features = []
            for i in range(bsz):
                cur_negative_pool = negative_pool[(temp < i*bsz) | (temp >= (i+1)*bsz)]
                neg_ind = np.random.choice(cur_negative_pool, size=self.args.in_batch_negatives, replace=False) if len(cur_negative_pool) >= self.args.in_batch_negatives else np.random.choice(cur_negative_pool, size=self.args.in_batch_negatives, replace=True)
                extended_image_features.append(torch.cat([image_features[i], neg_image_features[neg_ind]], dim=0))
            image_features = torch.stack(extended_image_features, dim=0) # [bsz, (num_positives + in_video_negatives + in_batch_negatives), D]
        else:
            raise RuntimeError("need to sample in batch negatives")
        assert (image_features.shape[0], image_features.shape[1], image_features.shape[2]) == (bsz, self.args.num_positives+self.args.in_video_negatives+self.args.in_batch_negatives, image_features.shape[-1]), f"shape of image features: {image_features.shape}"

        # get logits
        logit_scale = 2.718281828459045**self.args.logit_scale # 0.07 is the default, this means logit_scale ~ 1, TODO: might need to considering linearly ramp up logit_scale
        text_features = text_features.unsqueeze(1) # [bsz, 1, D]
        logits_per_text = logit_scale * (text_features * image_features).sum(2) # [bsz, (num_positives + in_video_negatives + in_batch_negatives)]
        num_negatives = logits_per_text.shape[1] - self.args.num_positives
        pos = logits_per_text[:, :self.args.num_positives].contiguous().view(-1,1) # [bsz * num_positives, 1]
        negs = logits_per_text[:, self.args.num_positives:].contiguous().repeat(1, self.args.num_positives).view(bsz*self.args.num_positives, num_negatives) # [bsz*num_positives, num_negatives]
        logits = torch.cat([pos, negs], dim=1) # [bsz*num_positives, 1 + num_negatives], the first column is the positive
        assert (logits.shape[0], logits.shape[1]) == (bsz*self.args.num_positives, 1+self.args.in_video_negatives+self.args.in_batch_negatives), f"shape of logits: {logits.shape}"
        loss = self.criterion(input=logits, target=torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)) # the criterion is the CrossEntropyLoss
        
        # TODO haven't implemented the regression loss

        # TODO
        weighted_loss = loss

        self.meters["matching_loss"].update(loss.item(), bsz)
        self.meters["weighted_loss"].update(weighted_loss.item(), bsz)
        self.writer.add_scalar("matching_loss", self.meters["matching_loss"].val, self.progress['num_updates'])
        self.writer.add_scalar('weighted_loss', self.meters["weighted_loss"].val, self.progress['num_updates'])
        
        return weighted_loss

    def train(self):
        flag = True
        step_per_epoch = int(self.train_data_length/self.args.batch_size)
        data_start_time = time.time()
        self.validate(hide_progress=True)
        while flag:
            for i, batch in enumerate(self.train_loader):
                data_end_time = time.time()
                self.model.train()
                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    self.validate_and_save()
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.scheduler.get_last_lr())

                self.writer.add_scalar("lr", cur_lr, self.progress['num_updates'])
                cur_step = self.progress['num_updates'] % step_per_epoch
                
                weighted_loss = self.forward(batch)

                with amp.scale_loss(weighted_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1.)
                # weighted_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.trainables, 1.)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])

                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.7f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                # validation and save models
                if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    self.validate_and_save()
                
                if self.args.take_snapshot_every_n_steps != None and (self.progress['num_updates'] - 1) % self.args.take_snapshot_every_n_steps == 0:
                    num = str((self.progress['num_updates'] - 1) // self.args.take_snapshot_every_n_steps)
                    save_path = os.path.join(self.args.exp_dir,f"snapshot_{num}.pth")
                    torch.save({"model": self.model.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict()}, save_path)
                    logger.info(f"take snapshot of the model at {save_path} at global step {self.progress['num_updates']}")

                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()

    def validate_and_save(self, libri=False, places=False):
        self.model.eval()
        acc = self.validate(hide_progress=True) # TODO change it to True once know the speed
        
        # acc = 0.1 # ignore validation, for debugging
        if acc > self.progress['best_acc']:
            self.progress['best_epoch'] = self.progress['epoch']
            self.progress['best_acc'] = acc
            save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
            torch.save(
                {
                    "model": self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "indices": self.train_sampler.state_dict(),
                },save_path
            )
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
        save_progress(self)
        save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        torch.save(
            {
                "model": self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "indices": self.train_sampler.state_dict(),
            },save_path
        )
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")

    def validate(self, hide_progress=True):
        self.model.eval()

        start_val_time = time.time()
        N_examples = self.valid_loader.dataset.__len__()

        # frame_counts = []
        with torch.no_grad():
            # get single modal representations
            qid_query_vid = []
            similarity = [] 
            for i, batch in enumerate(tqdm(self.valid_loader, disable=hide_progress)):
                self.model.eval()
                qid_query_vid.append((batch['qid'][0], batch['text'][0], batch['vid'][0]))
                cur_batch = {
                        "image": batch['image'].to(self.device), # [val_batch_size, L, C, H, W], and val_batch_size = 1 by default
                        "text": clip.tokenize(batch['text']).to(self.device)
                        }
                bsz, L, C, H, W = cur_batch["image"].shape
                cur_batch["image"] = cur_batch["image"].view(bsz*L, C, H, W)
                # print("test input image shape:", cur_batch["image"].shape)
                # print("test input text shape:", cur_batch["text"].shape)
                text_features = self.model.module.model.encode_text(cur_batch["text"])
                image_features = self.model.module.model.encode_image(cur_batch["image"])
                # print("test image features shape: ", image_features.shape)
                # print("text features shape: ", text_features.shape)
                
                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [L, D]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [1, D]
                # raise 

                # similarities
                # below only work when val_batch_size == 1
                assert self.args.val_batch_size == 1 and text_features.shape[0] == 1
                sims = (image_features*text_features).sum(1).cpu().numpy()# [L, 1] -> [L]
                similarity.append(sims) 
                # if len(sims) < 100:
                #     print(len(sims))
                # if i == 10:
                #     break

        all_numbers = evaluate(qid_query_vid, similarity, os.path.join(self.args.pyscenedetect_root, "scenedetect"+str(self.args.alpha)), self.valid_loader.dataset.data, self.args.val_framerate)

        logger.info("QVHighlight validation set results:")
        logger.info(f"MR-full-R1@0.5 {all_numbers['MR-full-R1@0.5']:.3f}  MR-full-R1@0.7 {all_numbers['MR-full-R1@0.7']:.3f}")
        logger.info(f"MR-full-mAP@0.5 {all_numbers['MR-full-mAP@0.5']:.3f}  MR-full-mAP@0.75 {all_numbers['MR-full-mAP@0.75']:.3f}  MR-full-mAP {all_numbers['MR-full-mAP']:.3f}")
        logger.info(f"validation time: {time.time() - start_val_time:.3f}")
        
        for key in all_numbers:
            self.writer.add_scalar(key, all_numbers[key], self.progress['num_updates'])
        
        acc = ((all_numbers['MR-full-R1@0.5'] + all_numbers['MR-full-R1@0.7'])/2 + all_numbers['MR-full-mAP'])/2
        return acc

    def _setup_meters(self):
        meters = {}
        meter_names = ['weighted_loss', "matching_loss", "regression_loss", 'data_time', 'train_time']
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
    
    def _setup_models(self):
        model, _ = clip.load(self.args.model_version, device="cuda")
        model.float()
        model = clip_wrapper.clip_wrapper(model)
        print(model)
        print_model_info(model)
        if self.args.validate:
            bundle = torch.load(os.path.join(self.args.exp_dir, "best_bundle.pth"))
            model.load_state_dict(bundle['model'])
            indices = None
            optim_states = None
            logger.info("Perform Validation")
        elif self.progress['num_updates'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "best_bundle.pth"))
            model.load_state_dict(bundle['model'])
            indices = bundle['indices']
            optim_states = bundle['optimizer']
            logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
        else:
            indices = None
            optim_states = None

        if self.args.freeze_first_x != None:
            freeze_names = ['model.visual.conv1.', 'model.visual.ln_pre.'] + [f'model.visual.transformer.resblocks.{i}.' for i in range(self.args.freeze_first_x)] + [f'model.transformer.resblocks.{i}.' for i in range(self.args.freeze_first_x)] + ['model.transformer.ln_final']
            for n, p in model.named_parameters():
                for fn in freeze_names:
                    if n.startswith(fn):
                        if p.requires_grad:
                            p.requires_grad = False
                            print(f"disable gradient of weights: {n}")
                            break

        trainables = [p for p in model.parameters() if p.requires_grad]
        model.to(self.device)

        return model, trainables, indices, optim_states

    def _setup_dataloader(self):
        train_dataset = qvh_dataset.ImageQueryDataset(self.args, split='train')
        val_dataset = qvh_dataset.ImageQueryDataset(self.args, split='val')
        train_sampler = StatefulSampler(len(train_dataset))
        if self.progress['num_updates'] > 1 and self.indices is not None:
            train_sampler.load_state_dict(self.indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate, drop_last=False)

        return train_loader, valid_loader, train_sampler, len(train_dataset)
    
    def _setup_optimizer(self):
        optimizer = torch.optim.AdamW(params=self.trainables, lr = self.args.lr)
        # optimizer = BertAdam(self.trainables, lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates)

        if self.progress['num_updates'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer.zero_grad()
        return optimizer
    
    def _setup_scheduler(self):
        def lr_lambda(current_step):
            num_warmup_steps = self.args.warmup_fraction * self.total_num_updates
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(self.total_num_updates - current_step) / float(max(1, self.total_num_updates - num_warmup_steps))
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)
        return scheduler

    def weight_loss(self, losses):
        raise RuntimeError
        weighted_loss = 0.
        if "cls_coarse_matching_loss" in losses:
            weighted_loss += losses['cls_coarse_matching_loss'] * self.args.cls_coarse_matching_weight
        if "feat_coarse_matching_loss" in losses:
            weighted_loss += losses['feat_coarse_matching_loss'] * self.args.feat_coarse_matching_weight
        if 'fine_matching_loss' in losses:
            weighted_loss += losses['fine_matching_loss'] * self.args.fine_matching_weight
        if 'caption_w2v2_loss' in losses:
            weighted_loss += losses['caption_w2v2_loss'].mean() * self.args.caption_w2v2_weight
        if 'libri_w2v2_loss' in losses:
            weighted_loss += losses['libri_w2v2_loss'].mean() * self.args.libri_w2v2_weight
        if 'caption_hubert_loss' in losses:
            weighted_loss += losses['caption_hubert_loss'].mean() * self.args.caption_hubert_weight
        if 'libri_hubert_loss' in losses:
            weighted_loss += losses['libri_hubert_loss'].mean() * self.args.libri_hubert_weight
        
        return weighted_loss
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

