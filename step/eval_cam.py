import numpy as np
import os
from PIL import Image
from chainercv.evaluations import calc_semantic_segmentation_confusion
import torch.nn.functional as F
import torch
import argparse


def run(args):
    imgs = os.listdir(args.mask_dir)
    labels = []

    preds = []
    for id in imgs:
        label = np.array(Image.open(os.path.join(args.gt_dir, id)).convert("L"))
        label[np.where(label>0)] = 1
        pred = np.array(Image.open(os.path.join(args.mask_dir,id)).convert("L"))
        pred = torch.Tensor(pred)
        pred = F.interpolate(pred[None,None,...], label.shape, mode='bilinear', align_corners=False)[0][0].numpy()
        pred[np.where(pred>0)] = 1
        pred = pred.astype('uint8')
        labels.append(label)
        preds.append(pred)

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    FP = confusion.sum(axis=0) - np.diag(confusion)  
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    ACC = (TP+TN)/(TP+FP+FN+TN)
    PRECISION=TP/(TP+FP)
    RECALL=TP/(TP+FN)
    F1=2*PRECISION*RECALL/(PRECISION+RECALL)

    print({'iou': iou, 'miou': np.nanmean(iou)})
    print({'ACC':ACC, 'PRECISION':PRECISION, 'RECALL':RECALL, 'F1':F1})

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)