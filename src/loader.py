import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class SyntheticSegDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.img_dir = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'masks')
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        img = np.array(img).astype(np.float32)/255.0
        mask = np.array(mask).astype(np.int64)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return img, mask
