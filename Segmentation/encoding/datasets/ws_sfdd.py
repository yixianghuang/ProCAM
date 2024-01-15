import os
import random
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


class WS_SFDD(data.Dataset):
    NUM_CLASS = 2
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=1024, crop_size=512):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

        self.norm = torch.FloatTensor([255, 192, 185]).view(1, 1, -1)
        # .png list
        if self.split == 'train':
            self.img_list = glob(os.path.join(self.root, 'train/img', '*.png'))      
            self.mask_list = glob(os.path.join(self.root, 'train/mask', '*.png'))
            self.img_list.sort()
            self.mask_list.sort()
            assert len(self.img_list) == len(self.mask_list)
        elif self.split == 'val' or self.split == 'test':
            self.img_list = glob(os.path.join(self.root, 'test/img', '*.png'))
            self.mask_list = glob(os.path.join(self.root, 'test/mask', '*.png'))
            self.img_list.sort()
            self.mask_list.sort()
            assert len(self.img_list) == len(self.mask_list)
        else:
            raise RuntimeError("Error mode! Please use 'train' or 'val'.")
        

    def make_pred(self, x):
        return x

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_dir = self.img_list[index]
        mask_dir = self.mask_list[index]
        img = np.asarray(Image.open(img_dir))
        mask = np.asarray(Image.open(mask_dir))
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        mask = mask.copy()
        mask[mask>0] = 1
        if self.mode == 'test':
            img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0).long()
            img = img / self.norm
            if self.transform is not None:
                img = self.transform(img.permute(2, 0, 1))
            return (img, mask), os.path.basename(img_dir)

        if self.mode == 'train':
            img, mask = rot_90(img=img, mask=mask)
            img, mask = horizontal_flip(img=img, mask=mask)
            img, mask = vertical_flip(img=img, mask=mask)
        # # Normalize
        img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
        img = img / self.norm
        if self.transform is not None:
            img = self.transform(img.permute(2, 0, 1))
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def horizontal_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return img.copy(), mask.copy()

def vertical_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[::-1, :, :]
        mask = mask[::-1, :]
    return img.copy(), mask.copy()

def rot_90(img, mask, p=0.5):
    if random.random() > p: 
        img = np.rot90(img, k=-1)
        mask = np.rot90(mask, k=-1)
    return img.copy(), mask.copy()