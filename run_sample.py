import argparse
import os
import numpy as np
import os.path as osp

from utils import pyutils

import torch
import numpy as np
import random

import torch.backends.cudnn as cudnn

if __name__ == '__main__':

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    set_seed(0)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--data_root", default='./data/WS_SFDD', type=str,
                        help="Path to SeafogDataset")

    # Dataset
    parser.add_argument("--img_list", default="train.txt", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=10, type=int)
    parser.add_argument("--cam_learning_rate", default=0.01, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.2, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    # ProCAM
    parser.add_argument("--procam_num_epoches", default=20, type=int)
    parser.add_argument("--procam_learning_rate", default=0.01, type=float)
    parser.add_argument("--procam_loss_weight", default=1.0, type=float)
    parser.add_argument("--contrastive_loss_weight", default=0.1, type=float)
    parser.add_argument("--reg_loss_weight", default=1.0, type=float)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--cam_mask_thres", default=0.2, type=float)
    parser.add_argument("--activation_thres", default=0.9, type=float)
    parser.add_argument("--par_refine", type=str2bool, default=True)

    # Output Path
    parser.add_argument("--work_space", default="./data/workspace", type=str) # set your path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
    parser.add_argument("--mask_dir", default="procam_mask", type=str)
    parser.add_argument("--procam_weight_dir", default="procam_weight", type=str)
    parser.add_argument("--gt_dir", default='test/mask', type=str)
    
    # Step
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--train_procam_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_procam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)


    args = parser.parse_args()
    args.log_name = osp.join(args.work_space,args.log_name)
    args.cam_weights_name = osp.join(args.work_space,args.cam_weights_name)
    args.mask_dir = osp.join(args.work_space,args.mask_dir)
    args.procam_weight_dir = osp.join(args.work_space,args.procam_weight_dir)
    args.gt_dir = osp.join(args.data_root,args.gt_dir)

    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.procam_weight_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))


    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)
    
    
    if args.train_procam_pass is True:
        import step.train_procam

        timer = pyutils.Timer('step.train_procam:')
        step.train_procam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)
    
    if args.make_procam_pass is True:
        import step.make_procam

        timer = pyutils.Timer('step.make_procam:')
        step.make_procam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)