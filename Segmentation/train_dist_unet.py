import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transform
from torch.cuda.amp import GradScaler, autocast

from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from encoding.nn import dice_ce_loss
from encoding.utils import (AverageMeter, LR_Scheduler,
                            intersectionAndUnionGPU)
from option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = Options().parse()
    args.train_gpu = list(range(torch.cuda.device_count()))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # model folder
    model_root = args.model_root
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    seafog_weight = torch.FloatTensor([0.8000, 1.0000]) ## weights for each class

    criterion_seg = dice_ce_loss(ignore_index=2, weight=seafog_weight)

    model = get_segmentation_model(args.model, dataset=args.dataset, 
                                   criterion_seg=criterion_seg) ##unet

    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(model.nclass))
        logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)


    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        scaler = GradScaler(enabled=True)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        scaler = GradScaler()
        model = torch.nn.DataParallel(model.cuda())

    best_mIoU = 0.0
    # resuming checkpoint
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_mIoU = checkpoint['best_mIoU'] * 0.01
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    # clear start epoch if fine-tuning
    if args.ft:
        args.start_epoch = 0
    
    # data transforms
    input_transform = transform.Compose([
        # transform.ToTensor(),
        transform.Resize((512,512)),
        transform.Normalize(
                    [0.485, 0.456, 0.406], # mean
                    [0.229, 0.224, 0.225] # std
                    )])   
    # dataset
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}

    train_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='train', mode='train',
                                           **data_kwargs)
    val_data = get_segmentation_dataset(args.dataset, root=args.data_root, split='val', mode ='val',
                                           **data_kwargs)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                               shuffle=(train_sampler is None), num_workers=args.workers, 
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=args.workers, 
                                             pin_memory=True, sampler=val_sampler)

    scheduler = LR_Scheduler('poly', args.lr,
                             args.epochs, len(train_loader), 
                             warmup_epochs=3)
    

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if main_process():
            logger.info('>>>>>>>>>>>>>>>> Start One Epoch Training >>>>>>>>>>>>>>>>')
        train(train_loader, model, scaler, optimizer, epoch, scheduler, train_data.NUM_CLASS)

        if (epoch+1) % args.save_interval == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion_seg, train_data.NUM_CLASS)
            if main_process():
                filename = os.path.join(args.model_root, args.last_name)
                logger.info('Saving checkpoint to: ' + filename)
                logger.info('\n')
                torch.save({'epoch': epoch_log, 
                        'current_mIoU': mIoU_val * 100,
                        'state_dict': model.module.state_dict(), 
                        'optimizer': optimizer.state_dict()}, 
                        filename)  
                if mIoU_val >= best_mIoU:
                    best_mIoU = mIoU_val
                    filename = os.path.join(args.model_root, args.best_name)
                    logger.info('Saving checkpoint to: ' + filename)
                    logger.info('\n')
                    torch.save({'epoch': epoch_log, 
                            'best_mIoU': best_mIoU * 100,
                            'state_dict': model.module.state_dict(), 
                            'optimizer': optimizer.state_dict()}, 
                            filename)

def train(train_loader, model, scaler, optimizer, epoch, scheduler, nclass):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    aux_loss_meter = AverageMeter() ##
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        scheduler(optimizer, i, epoch)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            if args.aux: ##false
                output, main_loss, jpu_loss, aux_loss = model(input, target)
                loss = main_loss + 0.1 * jpu_loss + args.aux_weight * aux_loss
            else:
                output, main_loss = model(input, target) 
                loss = main_loss 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            if args.aux: ##false
                main_loss, aux_loss, jpu_loss, loss = main_loss.detach() * n, jpu_loss.detach() * n, aux_loss.detach() * n, loss.detach() * n  # not considering ignore pixels
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(jpu_loss), dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                main_loss, aux_loss, jpu_loss, loss = main_loss / n, aux_loss / n, jpu_loss / n, loss / n
            else:
                loss = loss.detach() * n  # not considering ignore pixels 
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                loss = loss / n

        intersection, union, target = intersectionAndUnionGPU(output, target, nclass)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        if args.aux:
            aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n) ##
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        lr = optimizer.param_groups[0]['lr']
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % 100 == 0 and main_process():
            logger.info('Epoch:[{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'LR:{lr:.5f} '
                        'Remain:{remain_time} '
                        'Loss:{loss_meter.val:.3f}({loss_meter.avg:.3f}) '
                        'Accuracy:{accuracy:.2f}'.format(epoch+1, args.epochs,
                                                         i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                         lr=lr,
                                                         remain_time=remain_time,
                                                         loss_meter=loss_meter,
                                                         accuracy=accuracy * 100))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> One Training Epoch Done >>>>>>>>>>>>>>>>')
        logger.info('Train epoch [{}/{}]: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f} '.format(epoch+1, args.epochs, 
                                                                                                               loss_meter.avg, 
                                                                                                               mIoU * 100, 
                                                                                                               mAcc * 100, 
                                                                                                               allAcc * 100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class[i]*100, accuracy_class[i]*100))
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        logger.info('\n')
    return loss_meter.avg, mIoU, mAcc, allAcc ##


def validate(val_loader, model, criterion, nclass):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)
        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, nclass)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 100 == 0) and main_process():
            logger.info('Evaluation: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
                        'Accuracy {accuracy:.2f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy * 100))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: Loss {:.3f} | mIoU {:.2f} | mAcc {:.2f} | allAcc {:.2f}.'.format(loss_meter.avg, mIoU*100, mAcc*100, allAcc*100))
        for i in range(nclass):
            logger.info('Class_{} Result: iou {:.2f} | accuracy {:.2f}.'.format(i, iou_class[i]*100, accuracy_class[i]*100))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        logger.info('\n')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
