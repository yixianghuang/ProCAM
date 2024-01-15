## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss, MSELoss

from torch.autograd import Variable
from .dice import DiceLoss

torch_ver = torch.__version__[:3]

__all__ = ['SegmentationLosses', 'dice_ce_loss', 'BootstrappedCELoss', 'OhemCELoss', 'ProbOhemCrossEntropy2d', 'PyramidPooling', 'JPU', 'Mean', 'SeparableConv2d']

class dice_ce_loss(nn.Module):
    def __init__(self, ignore_index, weight):
        super(dice_ce_loss, self).__init__()
        self.ignore_index = ignore_index
        # self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.ce_loss = BootstrappedCELoss(weight=weight, ignore_index=ignore_index)
        
    def one_hot(self, y_pred, y_true):
        nclass = y_pred.size(1)
        y_one_hot = F.one_hot(y_true, 2).permute(0, 3, 1, 2)
        y_pred = y_pred.permute(1, 0, 2, 3).reshape(nclass, -1)
        y_one_hot = y_one_hot.permute(1, 0, 2, 3).reshape(nclass, -1).float()
        return F.softmax(y_pred, dim=0), y_one_hot
      
    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        i = torch.sum(y_true, dim=1)
        j = torch.sum(y_pred, dim=1)
        intersection = torch.sum(y_true * y_pred, dim=1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        y_pred, y_one_hot = self.one_hot(y_pred, y_true)
        loss = 1 - self.soft_dice_coeff(y_pred, y_one_hot)
        return loss

    def forward(self, y_pred, y_true):
        a = self.ce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b

class BootstrappedCELoss(nn.Module):
    def __init__(self, ignore_index, weight):
        super(BootstrappedCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.celoss = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        self.thresh = 0.5
   
    def single(self, predict, target):
        c, h, w = predict.size()
        k = h * w // 64
        predict = predict.permute(1,2,0).contiguous().view(-1, c)
        target = target.view(-1)
        loss_ = self.celoss(predict, target)
        sorted_loss = torch.sort(loss_, descending=True)[0]
        if sorted_loss[k] > self.thresh:
            loss = sorted_loss[sorted_loss > self.thresh]
        else: 
            loss = sorted_loss[:k]
        loss = torch.mean(loss)
        loss_ = torch.mean(loss_)
        return loss, loss_

    def forward(self, predict, target):
        batch = predict.size(0)
        BTloss = 0.0
        CEloss = 0.0
        for i in range(batch):
            loss, loss_ = self.single(predict[i], target[i])
            BTloss += loss
            CEloss += loss_
        return (BTloss * 0.4 + CEloss * 0.7) / float(batch)

class OhemCELoss(nn.Module):
    def __init__(self, weight):
        super(OhemCELoss, self).__init__()
        self.thresh = 0.9
        self.min_kept = 131072
        self.reduction = 'elementwise_mean'
        self.ignore_label = -1
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_label, reduction='none')

    def forward(self, predict, target, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1,) != self.ignore_label
        sort_prob, sort_indices = prob.contiguous().view(-1,)[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1,)
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.ignore_index = -1
        self.bceloss = BCELoss(weight, size_average)
        self.mseloss = MSELoss(reduce=True, size_average=True)

        # ADE20K
        self.weight = torch.FloatTensor([0.7658, 0.7803, 0.7880, 0.8024, 0.8126, 0.8156, 0.8210, 0.8455,
        0.8519, 0.8563, 0.8565, 0.8602, 0.8626, 0.8654, 0.8768, 0.8803, 0.8811,
        0.8847, 0.8832, 0.8838, 0.8855, 0.9008, 0.9050, 0.9065, 0.9100, 0.9111,
        0.9163, 0.9187, 0.9270, 0.9275, 0.9275, 0.9270, 0.9451, 0.9469, 0.9514,
        0.9505, 0.9592, 0.9626, 0.9641, 0.9657, 0.9647, 0.9687, 0.9701, 0.9733,
        0.9764, 0.9778, 0.9793, 0.9790, 0.9810, 0.9803, 0.9810, 0.9787, 0.9814,
        0.9851, 0.9882, 0.9844, 0.9845, 0.9866, 0.9896, 0.9888, 0.9895, 0.9906,
        0.9923, 1.0001, 0.9951, 0.9961, 0.9940, 0.9994, 0.9991, 1.0014, 1.0056,
        1.0080, 1.0047, 1.0058, 1.0171, 1.0147, 1.0196, 1.0284, 1.0231, 1.0269,
        1.0272, 1.0295, 1.0335, 1.0301, 1.0344, 1.0375, 1.0371, 1.0395, 1.0381,
        1.0424, 1.0399, 1.0405, 1.0429, 1.0489, 1.0458, 1.0483, 1.0508, 1.0502,
        1.0499, 1.0551, 1.0515, 1.0500, 1.0512, 1.0584, 1.0623, 1.0602, 1.0634,
        1.0675, 1.0591, 1.0624, 1.0628, 1.0593, 1.0683, 1.0674, 1.0685, 1.0734,
        1.0683, 1.0689, 1.0678, 1.0663, 1.0830, 1.0774, 1.0858, 1.0841, 1.0930,
        1.0860, 1.0947, 1.0825, 1.0923, 1.0856, 1.0929, 1.0953, 1.0885, 1.0916,
        1.0931, 1.0921, 1.0960, 1.0952, 1.1015, 1.1008, 1.0962, 1.0995, 1.1049,
        1.1070, 1.1039, 1.0988, 1.1183, 1.1316, 1.1273, 1.1413])

    def MSE_Loss(self, code):
        total_loss = 0
        for i in code:
            b, c, k = i.size()
            target = torch.eye(k, device=i.get_device()).float()
            matrix = torch.matmul(i.permute(0, 2, 1), i)
            loss = self.mseloss(matrix, target)
            total_loss += loss
        return total_loss

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            # loss3 = self.MSE_Loss(code)
            return loss1 + self.aux_weight * loss2 #+ self.aux_weight * loss3
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class ProbOhemCrossEntropy2d(Module):
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1, reduction='mean', thresh=0.7, min_kept=50000):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        # Pascal Context
        self.weight = torch.FloatTensor(
        [0.9752, 1.1304, 1.0394, 0.9790, 1.1678, 0.9746, 0.9684, 0.9843, 1.0779,
        1.0104, 0.8645, 0.9524, 0.9868, 0.9177, 0.8935, 0.9964, 0.9434, 0.9809,
        1.1404, 0.9986, 1.1305, 1.0130, 0.9012, 1.0040, 0.9556, 0.9000, 1.0835,
        1.1341, 0.8632, 0.8645, 0.9675, 1.1404, 1.1137, 0.9552, 0.9701, 1.4059,
        0.8564, 1.1347, 1.0534, 0.9957, 0.9114, 1.0241, 0.9884, 1.0245, 1.0236,
        1.1080, 0.8488, 1.0122, 0.9343, 0.9747, 1.0404, 0.9482, 0.8689, 1.1267,
        0.9776, 0.8640, 0.9030, 0.9728, 1.0239]
        )
        self.criterion = CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, weight=self.weight)

    def forward(self, *inputs):
        pred, pred_, target = inputs
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = torch.sort(mask_prob)
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target) + self.aux_weight * self.criterion(pred_, target)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat

class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)
