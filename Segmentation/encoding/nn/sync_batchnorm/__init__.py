# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback