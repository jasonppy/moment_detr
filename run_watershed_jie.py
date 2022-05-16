from standalone_eval.watershaed_jie import generate_tag_proposals_single_video
from standalone_eval.eval import eval_submission
from easydict import EasyDict
from utils.basic_utils import load_jsonl, save_jsonl, save_json, l2_normalize_np_array
from moment_detr.start_end_dataset import StartEndDataset
from torch.utils.data.dataloader import DataLoader
from utils.tensor_utils import pad_sequences_1d
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger(__name__)


def tag_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?
    query = torch.stack([e["model_inputs"]["query_feat"] for e in batch]) # (N, d)
    video, video_mask = pad_sequences_1d(
        [e["model_inputs"]["video_feat"] for e in batch],
        dtype=torch.float32,
        fixed_length=75
    )  # (N, L, d), (N, L) with 1 valid
    video_lengths = video_mask.sum(1)  # (N, )

    batched_data = dict(
        query=query,  # (N, d)
        video=video,  # (N, L, d)
        video_lengths=video_lengths,  # (N, )
    )
    return batch_meta, batched_data


def scale_min_max(tensor):
    min_tensor = torch.min(tensor, dim=1)[0].unsqueeze(1)
    max_tensor = torch.max(tensor, dim=1)[0].unsqueeze(1)
    tensor = (tensor - min_tensor) / max_tensor
    return tensor


class TAGMomentRetrieval(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_len = 2
        self.dataset = StartEndDataset(
            dset_name="hl",
            data_path=cfg.data_path, v_feat_dirs=cfg.v_feat_dirs,
            q_feat_dir=cfg.q_feat_dir, q_feat_type="pooler_output",
            max_v_l=75, data_ratio=cfg.data_ratio, ctx_mode="video",
            normalize_v=True, normalize_t=True, load_labels=False,
            clip_len=self.clip_len, max_windows=5
        )
        self.dataloader = DataLoader(
            self.dataset, collate_fn=tag_collate,
            batch_size=10, num_workers=8, shuffle=False
        )

    def compute_similarity_scores(self):
        iterator = tqdm(
            enumerate(self.dataloader),
            desc="compute similarity scores",
            total=len(self.dataloader)
        )
        device = torch.device("cuda")
        all_similarity_scores = []
        all_video_lengths = []
        all_meta = []
        for batch_idx, (batch_meta, batch_data) in iterator:
            query_feats = batch_data["query"].to(device)  # (N, d)
            video_feats = batch_data["video"].to(device)  # (N, L, d)
            video_lengths = batch_data["video_lengths"]  # (N, )
            scores = torch.einsum("bd,bld->bl", query_feats, video_feats)  # (N, L)
            scores = F.softmax(scores/0.07, dim=-1)  # scale [-1, 1] to [0, 1]
            scores = scale_min_max(scores)
            all_similarity_scores.append(scores.cpu().numpy())
            all_video_lengths.extend(video_lengths.numpy().tolist())
            all_meta.extend(batch_meta)
        all_similarity_scores = np.concatenate(all_similarity_scores, axis=0)
        return all_meta, all_similarity_scores, all_video_lengths

    def get_predictions(self, metas, scores, lengths):
        predictions = []
        iterator = tqdm(zip(metas, scores, lengths), total=len(metas), desc="get predictions")
        for meta, score, length in iterator:
            _pred_relevant_windows = generate_tag_proposals_single_video(
                score[:int(length)], min_len=1, max_len=75, bw=1, clip_length=self.clip_len)
            _pred_relevant_windows = [e.tolist() for e in _pred_relevant_windows[:10]]
            _pred = dict(
                qid=meta["qid"],
                vid=meta["vid"],
                pred_relevant_windows=_pred_relevant_windows,
                pred_saliency_scores=score[:int(length)].tolist()
            )
            predictions.append(_pred)
        return predictions

    def run_all(self):
        metas, scores, lengths = self.compute_similarity_scores()
        predictions = self.get_predictions(metas, scores, lengths)
        metrics = eval_submission(predictions, self.dataset.data)
        save_jsonl(predictions, self.cfg.save_path)
        save_metrics_path = self.cfg.save_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path)
        print(f"{metrics}")

def run():
    split = "val"
    cfg = EasyDict(
        data_path=f"/home/pyp/vqhighlight/data/highlight_val_release.jsonl",
        v_feat_dirs=["/saltpool0/data/pyp/vqhighlight/features/clip_features"],
        q_feat_dir="/saltpool0/data/pyp/vqhighlight/features/clip_text_features",
        save_path=f"clip_watershed.jsonl",
        data_ratio=0.05
    )
    tag_moment_retrieval = TAGMomentRetrieval(cfg)
    tag_moment_retrieval.run_all()


if __name__ == '__main__':
    run()
