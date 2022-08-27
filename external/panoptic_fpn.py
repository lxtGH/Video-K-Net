from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class PanopticFPN(TwoStageDetector):
    """Implementation of `Panoptic FPN <https://arxiv.org/abs/1901.02446>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(PanopticFPN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_semantic)
                or (hasattr(self, 'semantic_head')
                    and self.semantic_head is not None))
