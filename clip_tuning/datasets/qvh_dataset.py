import json
import random
import numpy as np
import os
import torch
import torch.nn.functional
import random
import time
import soundfile as sf
from torch.utils.data import Dataset
import h5py
import pickle
import torchvision.transforms as transforms
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class ImageQueryDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--jsonl_root", type=str, default="/home/pyp/vqhighlight/data")
        parser.add_argument("--image_root", type=str, default="/saltpool0/data/pyp/vqhighlight/image")
        parser.add_argument("--train_framerate", type=int, default=1)
        parser.add_argument("--val_framerate", type=int, default=1)
    
    def __init__(self, args, split):
        self.args = args
        self.split = split
        assert split in ["train", "test", "val"]
        if split == "train":
            self.data = load_jsonl(os.path.join(self.args.jsonl_root, f"highlight_{split}_release.jsonl"))
        else:
            self.data = load_jsonl(os.path.join(self.args.jsonl_root, f"highlight_{split}_release_exist.jsonl"))
        self.image_root = os.path.join(self.args.image_root, f"framerate{str(self.args.train_framerate)}") if "train" in split else os.path.join(self.args.image_root, f"framerate{str(self.args.val_framerate)}")
        if "train" not in split:
            self.image_transform = transforms.Compose(
                [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        else:
            self.image_transform = transforms.Compose(
                [transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]) # TODO might need to remove the randomhorizontalflip
        if "train" in split:
            self.new_data = []
            logger.info("Calculate positive range and negative range for all video-query pairs")
            self.positive_ranges = [] # in frames (int)
            self.negative_ranges = [] # in frames (int)
            time_s = time.time()
            total_count = []
            zero_count = 0
            full_count = 0
            for item in self.data:
                vid = item['vid']
                image_files = glob.glob(os.path.join(self.image_root, vid, "*.jpg"))
                total_range = list(range(len(image_files)))
                if len(total_range) > 120:
                    full_count += 1

                # print("total images: ", len(total_range))
                total_count.append(len(total_range))
                positive_range = []
                for s,e in item['relevant_windows']:
                    positive_range += list(range(min(int(s*self.args.train_framerate), len(total_range)), min(int(e*self.args.train_framerate), len(total_range))))
                # print("positive images", len(positive_range))
                negative_range = list(set(total_range).difference(set(positive_range)))
                # print("negative images", len(negative_range))
                if len(negative_range) == 0:
                    zero_count += 1
                if len(positive_range) > 0 and len(negative_range) > 0:
                    self.positive_ranges.append(positive_range)
                    self.negative_ranges.append(negative_range)
                    self.new_data.append(item)
            self.data = self.new_data
            # plt.figure()
            # sns.histplot(total_count)
            # plt.savefig("total_count.jpg")
            print("full cound: ", full_count)
            print("total data after filtering: ", len(self.data))
            print("total zero neg instance: ", zero_count)
            print("total instances: ", len(total_count))
            # print("total less than 150: ", small_total)
            logger.info(f"done, time elapsed {time.time() - time_s:.3f}")

    def __len__(self):
        return len(self.data)

    def _LoadImage(self, vid, inds):
        img_fns = [os.path.join(self.image_root, vid, f"{i}.jpg") for i in inds]
        # imgs = [cv2.imread(img_fn).transpose((2,0,1))/255 for img_fn in img_fns] # [C, H, W]
        imgs = torch.stack([self.image_transform(Image.open(img_fn).convert('RGB')) for img_fn in img_fns], dim=0)  # [num_positives, 3, 224 ,224]
        return imgs

    def __getitem__(self, index):
        vid = self.data[index]['vid']
        if "train" in self.split:
            p_range = self.positive_ranges[index]
            n_range = self.negative_ranges[index]
            while len(n_range) == 0:
                index = np.random.choice(list(range(len(self.data))), size=1)[0]
                vid = self.data[index]['vid']
                p_range = self.positive_ranges[index]
                n_range = self.negative_ranges[index]
            positive_ind = np.random.choice(p_range, size=self.args.num_positives, replace=False) if len(p_range) >= self.args.num_positives else np.random.choice(p_range, size=self.args.num_positives, replace=True)
            negative_ind = np.random.choice(n_range, size=self.args.in_video_negatives, replace=False) if len(n_range) >= self.args.in_video_negatives else np.random.choice(n_range, size=self.args.in_video_negatives, replace=True)
            positive_img = self._LoadImage(vid, positive_ind)
            negative_img = self._LoadImage(vid, negative_ind)
            if len(positive_img.shape) == 3:
                positive_img.unsqueeze(0)
            image = torch.cat([positive_img, negative_img], dim=0)
        else:
            num_img = len(list(glob.glob(os.path.join(self.image_root, vid, "*.jpg"))))
            img_fns = [os.path.join(self.image_root, vid, f"{i}.jpg") for i in range(num_img)]
            image = torch.stack([self.image_transform(Image.open(img_fn).convert('RGB')) for img_fn in img_fns], dim=0) # [total_img, 3, 224, 224]
            # print(image.shape)
        text = self.data[index]['query'].lower()
        
        return vid, image, text, self.data[index]['qid']
    
    def collate(self, batch):
        vals = list(zip(*batch))

        collated = {}
        collated['vid'] = vals[0]
        collated['image'] = torch.stack(vals[1], dim=0) # [bzs, num_positives + in_video_negatives, C,H,W] or [bzs, total_img, C,H,W]
        # if self.split == "val":
        #     print(collated['image'].shape)
        collated['text'] = vals[2]
        collated['qid'] = vals[3]
        collated['gt_score'] = None

        return collated