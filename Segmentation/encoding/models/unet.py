import torch
import torch.nn as nn

import torch.nn.functional as F


def get_unet(dataset='ws_sfdd', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = Unet(num_classes=datasets[dataset.lower()].NUM_CLASS, **kwargs)
    return model

class Unet(nn.Module):
    def __init__(
        self, 
        in_channels=3,
        num_classes=2,
        norm_layer=nn.BatchNorm2d,
        criterion_seg=None
    ):
        super(Unet, self).__init__()
        
        self.in_channels = in_channels
        self.nclass = num_classes
        self.norm_layer = norm_layer
        self.criterion_seg = criterion_seg
        self.base_size = 256
        self.crop_size = 256
        self.down1 = self.conv_stage(in_channels, 64)
        self.down2 = self.conv_stage(64, 128)
        self.down3 = self.conv_stage(128, 256)
        self.down4 = self.conv_stage(256, 512)
       
        self.center = self.conv_stage(512, 1024)
        
        self.up4 = self.conv_stage(1024, 512)
        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)
        
        self.trans4 = self.upsample(1024, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)
        
        self.conv_last = nn.Conv2d(64, num_classes, 3, 1, 1)
        
        self.max_pool = nn.MaxPool2d(2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, useBN=True):
        if useBN:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              self.norm_layer(dim_out),
              nn.ReLU(inplace=True),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              self.norm_layer(dim_out),
              nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.ReLU(inplace=True),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.ReLU(inplace=True)
            )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, y=None):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))
        
        out = self.center(self.max_pool(conv4_out))

        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))

        out = self.conv_last(out)

        if self.training:
            out = F.interpolate(out, y.shape[-2:], mode='bilinear', align_corners=True)
            loss = self.criterion_seg(out, y)
            return out.max(1)[1].detach(), loss
        else:
            return out.detach()
        
    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union