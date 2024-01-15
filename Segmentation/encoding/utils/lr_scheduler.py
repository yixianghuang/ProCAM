## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.gate = int(num_epochs * 0.5)
        self.t = int(num_epochs * 0.1)
        self.t_N = self.t * iters_per_epoch
        self.gate_iter = self.gate * iters_per_epoch
        self.poly_rec = 0.0
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

        self.iters = num_epochs*iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred=None):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9) #0.9
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'cycle':
            if epoch < self.gate:
                lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
                self.poly_rec = lr
            else:
                t = T - self.gate_iter
                lr = self.poly_rec + 0.6 * self.poly_rec * math.cos((t % self.t_N) / self.t_N * math.pi)            
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        # if epoch > self.epoch:
        #     print('\n=>Epoches %i, learning rate = %.8f, \
        #         previous best = %.4f' % (epoch, lr, best_pred))
        #     self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10 #10


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import numpy as np
    init_weights = [0.9, 0.1]
    depth = 19
    diag = np.concatenate(
                    (init_weights[0] * np.diag(np.ones(depth)), init_weights[1] * np.diag(np.ones(depth))),
                    axis=-2).astype(dtype=np.float32)
    diag = diag[np.newaxis, np.newaxis]
    print(diag.shape)
    diag = torch.tensor(diag).float().permute(3, 2, 0, 1)
    diag = nn.Parameter(diag)
    # print(diag)
    conv = nn.Conv2d(38, 19, 1, bias=False)
    conv.weight = diag
    print(conv.weight)