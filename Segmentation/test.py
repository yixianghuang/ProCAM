import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils import data
from tqdm import tqdm

import encoding.utils as utils
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import MultiEvalModule, get_model, get_segmentation_model
from option import Options

import torch.nn.functional as F

def intersectionAndUnionGPU(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    output = output[0]
    target = target[0].cuda()

    _, output = output.max(dim=1)

    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda(), area_output.cuda()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def test(args):
    # output folder
    if args.mode == 'test':
        outdir = args.save_folder
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        # transform.ToTensor(),
        transform.Resize((512,512)),
        transform.Normalize(
                    [0.485, 0.456, 0.406], # mean
                    [0.229, 0.224, 0.225] # std
                    )])      
                    
    # dataset
    testset = get_segmentation_dataset(args.dataset,
                                        root=args.data_root,
                                        split=args.split, 
                                        mode=args.mode,
                                        transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, 
                                **loader_kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset) ##unet
        # resuming checkpoint (best model)
        args.resume = os.path.join(args.model_root, args.best_name)
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model_dict = model.state_dict()
        checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items()
                               if k in model_dict.keys()}
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    #####
    scales = [0.5, 0.75, 1.0, 1.25, 1.5] if args.dataset == 'lip' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    if not args.ms:
        scales = [1.0]
    evaluator = MultiEvalModule(model, testset.NUM_CLASS, scales=scales, flip=args.flip).cuda()
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.NUM_CLASS)

    tbar = tqdm(test_data)
    miou_max = 0.
    pixacc_max = 0.
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    output_meter = AverageMeter()

    for i, (image, dst) in enumerate(tbar):
        if 'val' in args.mode:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)

                predict = predicts[0]
                predict = F.interpolate(predict, dst[0].shape[-2:], mode='bilinear', align_corners=True)
                predicts = [predict]

                inter, union, tgt, out = intersectionAndUnionGPU(predicts, dst, testset.NUM_CLASS)
                inter, union, tgt, out = inter.cpu().numpy(), union.cpu().numpy(), tgt.cpu().numpy(), out.cpu().numpy()
                intersection_meter.update(inter), union_meter.update(union), target_meter.update(tgt), output_meter.update(out)

                metric.update(dst, predicts)
                pixAcc, mIoU, IoU = metric.get()
                miou_max = mIoU
                pixacc_max = pixAcc

                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                img, targets = image
                outputs = evaluator.parallel_forward(img)
                predicts = [torch.max(F.interpolate(output, targets[0].shape[-2:], mode='bilinear', align_corners=True), 1)[1].cpu().numpy()
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                predict[predict == 1] = 255 
                predict = np.array(predict)[0]
                outname = os.path.splitext(impath)[0] + '.png'
                cv2.imwrite(os.path.join(outdir, outname), predict)

                
    if miou_max > 0:
        print('----------')
        print('Single or Multi-scal test:{}'.format(miou_max))
        print('Each class:{}, {}'.format(IoU, IoU.shape))
        print('PixAcc:{}'.format(pixacc_max))
        #################
        precision = intersection_meter.sum / (output_meter.sum + 1e-10)
        recall = intersection_meter.sum / (target_meter.sum + 1e-10)
        f1 = 2 * precision * recall / (precision + recall)
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        print('Precision:{}'.format(precision))
        print('Recall:{}'.format(recall))
        print('F1-score:{}'.format(f1))
        print('IoU:{}'.format(iou))
    else:
        print('----------')
        print('Getting Results')
if __name__ == "__main__":
    args = Options().parse()
    args.test_batch_size = torch.cuda.device_count()
    test(args)
    print('----------Test Done-------------')
