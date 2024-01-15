from .base import *
from .ws_sfdd import WS_SFDD


datasets = {
    'ws_sfdd': WS_SFDD,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
