import torch
import torch.nn as nn
import torch.nn.functional as F

class clip_wrapper(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.model = clip
    def forward(self, image, text=None):
        if not self.training:
            image_features = self.model.encode_image(image) # [bsz*(self.args.in_video_negatives+self.args.num_positives), D]
            return image_features
        else:
            image_features = self.model.encode_image(image) # [bsz*(self.args.in_video_negatives+self.args.num_positives), D]
            text_features = self.model.encode_text(text) # [bsz, D]
        
        return image_features, text_features