import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel.scatter_gather import scatter
from torch.nn.parallel.data_parallel import DataParallel

from ..nn import JPU
from .. import dilated as resnet
from ..utils import batch_pix_accuracy, batch_intersection_union

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet', 'MultiEvalModule']

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, jpu=True, dilated=False, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root=None, **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.jpu_gate = jpu
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root) ##root=root
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False, dilated=dilated,
                                               deep_base=False, norm_layer=norm_layer, root=root)
            self.pretrained.load_state_dict(torch.load('./data/resnet101_coco-586e9e4e.pth'), strict=False)
        # elif backbone == 'resnet101':
        #     self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,
        #                                        deep_base=True, norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnest50':
            self.pretrained = resnet.resnest50(pretrained=True)
        elif backbone == 'resnest101':
            self.pretrained = resnet.resnest101(pretrained=True)
        elif backbone == 'res2net50':
            self.pretrained = resnet.res2net50_v1b(pretrained=True)
        elif backbone == 'res2net101':
            self.pretrained = resnet.res2net101_v1b(pretrained=True)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.backbone = backbone
        if self.jpu_gate:
            self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer, up_kwargs=up_kwargs)
        # self.conv1 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0) ##多加

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        # x = self.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu_gate:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        # self.base_size = module.base_size
        # self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        # print('MultiEvalModule: base_size {}, crop_size {}'. \
        #     format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        #for out in outputs:
        #    print('out.size()', out.size())
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert(batch == 1)
        # assert(len(self.scales) == 1)
        outputs = module_inference(self.module, image, self.flip)
        return outputs

def module_inference(module, image, flip=True):
    if flip:
        image = tta_image(image)
        output = module.evaluate(image)
        output = F.softmax(output, dim=1)
        output = verse_tta_pred(output)
    else:
        output = module.evaluate(image)
        output = F.softmax(output, dim=1)
    return output


def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)

def tta_image(img):
    assert(img.dim()==4)
    img_left = img.cpu().numpy() # 1, 3, h, w
    img_right = img_left[:, :, :, ::-1].copy()
    img_lr = np.concatenate([img_left, img_right], axis=0) # 2, 3, h, w
    img_td = img_lr[:, :, ::-1, :].copy()
    img_flip = np.concatenate([img_lr, img_td], axis=0) # 4, 3, h, w
    img_rot90 = np.rot90(img_flip, k=-1, axes=(2, 3)).copy()
    img_ = np.concatenate([img_flip, img_rot90], axis=0) # 8, 3, h, w
    img = torch.from_numpy(img_).to(img.get_device()) 
    return img

def verse_tta_pred(pred):
    pred_ = pred.cpu().numpy()
    pred_rot = np.rot90(pred_[4:, ...], k=1, axes=(2, 3)) # 4, 2, h, w
    pred_ = pred_[:4, ...] + pred_rot # 4, 2, h, w
    pred_top = pred_[2:, :, ::-1, :] # 2, 2, h, w
    pred_ = pred_[:2, ...] + pred_top # 2, 2, h, w
    pred_left = pred_[1, :, :, ::-1] # 1, 2 ,h, w
    pred_ = pred_[0, ...] + pred_left # 1, 2, h, w
    pred = torch.from_numpy(pred_).to(pred.get_device())
    return pred.unsqueeze(0) / 8
