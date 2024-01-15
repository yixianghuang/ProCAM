import argparse
import os

import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='unet',
                            help='model name (default: unet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--jpu', action='store_true', default=
                            False, help='JPU')
        parser.add_argument('--dilated', action='store_true', default=
                            False, help='dilation')
        parser.add_argument('--lateral', action='store_true', default=
                            False, help='employ FPN')
        parser.add_argument('--dataset', type=str, default='ws_sfdd',
                            help='dataset name (default: ws_sfdd)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=1024,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=512,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:3456',
                            help = 'distribute address')
        parser.add_argument('--world-size', type=int, default=1,
                            help='world size (default:1)')
        parser.add_argument('--rank', type=int, default=0,
                            help='rank (default:0)')
        parser.add_argument('--zoom-factor', type=int, default=8,
                            help='the output stride')
        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            default=True, help='Multiprocessing Distributed Training')
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        parser.add_argument('--save_interval', type=int, default=1)
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--manual-seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--split', default='val')
        parser.add_argument('--mode', default='val')
        parser.add_argument('--ms', action='store_true', default=False,
                            help='multi scale test')
        parser.add_argument('--flip', action='store_true', default=False,
                            help='flip & rot90 test')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        parser.add_argument('--best-name', type=str, default='best_model.pth',
                            help = 'The name of the best model')
        parser.add_argument('--last-name', type=str, default='last_model.pth',
                            help = 'The name of the last model')
        parser.add_argument('--save_folder', type=str, default='../data/workspace/seg_model/output_mask',
                            help = 'path to save images')
        # dataset and model directory
        parser.add_argument('--data_root', default='../data/WS_SFDD')
        parser.add_argument('--model_root', default='../data/workspace/seg_model')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'ws_sfdd': 80,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 32
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'ws_sfdd': 0.0001,
            }
            args.lr = lrs[args.dataset.lower()]
        # print(args)
        for i in vars(args).items():
            print("{} : {}".format(i[0], i[1]))
        return args
