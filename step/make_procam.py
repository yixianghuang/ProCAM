import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp

import dataset.dataloader
from utils import torchutils, imutils
import net.resnet50_cam
import cv2
import imageio
cudnn.enabled = True

from .par import PAR

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    refine_classifier = net.resnet50_cam.Refine_Classifier(2, args.feature_dim, args.momentum)
    refine_classifier.load_state_dict(torch.load(osp.join(args.procam_weight_dir,'refine_classifier_'+str(args.procam_num_epoches) + '.pth')))

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        refine_classifier.cuda()

        if args.par_refine:
            par = PAR(num_iter=20, dilations=[1,4,16,32,48,64])
            par = par.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]  
            size = pack['size']

            strided_up_size = imutils.get_strided_up_size(size, 16)

            # initial CAM
            # outputs = [model.forward(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 2 x w x h
            ## refined CAM
            outputs = [model.forward1(img[0].cuda(non_blocking=True),refine_classifier.classifier.weight) for img in pack['img']] # b x 2 x w x h

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            refined_cam = highres_cam
            if args.par_refine:
                img = pack['img'][0][0][0][None,...].cuda()
                refined_cam = refined_cam[None,...]
                refined_cam = par(img,refined_cam)
                refined_cam = refined_cam[0]
        
            refined_cam = refined_cam.cpu().numpy()
            refined_cam = np.pad(refined_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(pack['label'][0] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(refined_cam, axis=0)
            cls_labels = keys[cls_labels]
            cls_labels[cls_labels!=1] = 0
            cls_labels[cls_labels==1] = 255
            
            cls_labels = cv2.resize(cls_labels, (1024, 1024))

            imageio.imwrite(os.path.join(args.mask_dir, img_name + '.png'),
                            cls_labels.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(osp.join(args.procam_weight_dir,'res50_procam_'+str(args.procam_num_epoches) + '.pth')))
    model.eval()

    n_gpus = torch.cuda.device_count()

    data = dataset.dataloader.SeafogClassificationDatasetMSF(args.img_list, data_root=args.data_root, resize_long=(512,512), scales=args.cam_scales)
    data = torchutils.split_dataset(data, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, data, args), join=True)
    print(']')

    torch.cuda.empty_cache()