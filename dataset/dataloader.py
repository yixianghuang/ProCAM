import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from utils import imutils
import random


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class SeafogDataset(Dataset):

    def __init__(self, img_name_list_path, data_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.data_root = data_root
        self.split = img_name_list_path[:-4]
        img_name_list = os.path.join(data_root, img_name_list_path)
        self.ids = [id_.strip().split(' ') for id_ in open(img_name_list)]
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx][0] 
        img = np.asarray(imageio.imread(os.path.join(self.data_root, self.split, 'img', name+'.png')))
        
        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)
            
        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)
        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img}


class SeafogClassificationDataset(SeafogDataset):

    def __init__(self, img_name_list_path, data_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, data_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        
        out['label'] = torch.from_numpy(np.array([1,0] if self.ids[idx][1] == '0' else [0,1], dtype='float32'))

        return out


class SeafogClassificationPairDataset(SeafogDataset):

    def __init__(self, img_name_list_path, data_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, data_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.pos_ids = []
        self.neg_ids = []
        for i in range(len(self.ids)):
            if self.ids[i][1] == '1':
                self.pos_ids.append(i)
            else:
                self.neg_ids.append(i)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        pair_outs = {}

        # randomly select another image with a different label
        if self.ids[idx][1] == '1':
            pair_outs['pos_name'] = out['name']
            pair_outs['pos_img'] = out['img']
            another_idx = random.choice(self.neg_ids)
            another_out = super().__getitem__(another_idx)
            pair_outs['neg_name'] = another_out['name']
            pair_outs['neg_img'] = another_out['img']
        else:
            another_idx = random.choice(self.pos_ids)
            another_out = super().__getitem__(another_idx)
            pair_outs['pos_name'] = another_out['name']
            pair_outs['pos_img'] = another_out['img']
            pair_outs['neg_name'] = out['name']
            pair_outs['neg_img'] = out['img']
       
        pair_outs['pos_label'] = torch.from_numpy(np.array([0,1], dtype='float32'))
        pair_outs['neg_label'] = torch.from_numpy(np.array([1,0], dtype='float32'))

        return pair_outs


class SeafogClassificationDatasetMSF(SeafogDataset):

    def __init__(self, img_name_list_path, data_root,
                 img_normal=TorchvisionNormalize(),
                 resize_long=None,
                 scales=(1.0,)):
        super().__init__(img_name_list_path, data_root, img_normal=img_normal)
        self.scales = scales
        self.resize_long = resize_long

    def __getitem__(self, idx):
        name = self.ids[idx][0]
        img = np.asarray(imageio.imread(os.path.join(self.data_root, self.split, 'img', name+'.png')))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])
        
        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))

        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(np.array([1,0] if self.ids[idx][1] == '0' else [0,1], dtype='float32'))}
        return out

