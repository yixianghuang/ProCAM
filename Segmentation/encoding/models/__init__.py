from .model_zoo import get_model
from .model_store import get_model_file
from .base import *

from .deeplabv3 import *
from .unet import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn, 
        'unet': get_unet,
        'deeplabv3': get_deeplab,
    }
    return models[name.lower()](**kwargs)
