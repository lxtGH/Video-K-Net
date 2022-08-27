from mmcv.utils import Registry
from mmcv.cnn import build_model_from_cfg as build

TRACKERS = Registry('tracker')


def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)
