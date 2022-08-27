import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss, build_neck
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


@HEADS.register_module()
class SemanticHead(FusedSemanticHead):
    """Semantic segmentation head that can be used in panoptic segmentation.

    Args:
        semantic_decoder (dict): Config dict of decoder.
            It usually is a neck, like semantic FPN.
        in_channels (int, optional): Input channels. Defaults to 256.
        num_classes (int, optional):  Number of semantic classes including
            the background. Defaults to 183.
        ignore_label (int, optional): Labels to be ignored. Defaults to 255.
        loss_seg (dict, optional): Config dict of loss.
            Defaults to `dict(type='CrossEntropyLoss', use_sigmoid=False, \
            loss_weight=1.0)`.
        conv_cfg (dict, optional): Config of convolutional layers.
            Defaults to None.
        norm_cfg (dict, optional): Config of normalization layers.
            Defaults to None.
    """

    def __init__(self,
                 semantic_decoder,
                 in_channels=256,
                 num_classes=183,
                 ignore_label=255,
                 pred_stride=4,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None):
        super(FusedSemanticHead, self).__init__()
        self.semantic_decoder = build_neck(semantic_decoder)
        self.conv_logits = nn.Conv2d(in_channels, num_classes, 1)
        self.loss_seg = build_loss(loss_seg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.pred_stride = pred_stride
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

    def init_weights(self):
        kaiming_init(self.conv_logits)

    @auto_fp16()
    def forward(self, feats):
        x = self.semantic_decoder(feats)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        mask_pred = F.interpolate(
            mask_pred,
            scale_factor=self.pred_stride,
            mode='bilinear',
            align_corners=False)
        labels = labels.squeeze(1).long()
        loss_sem_seg = self.loss_seg.loss_weight * F.cross_entropy(
            mask_pred,
            labels,
            reduction='mean',
            ignore_index=self.ignore_label)
        # loss_semantic_seg = self.loss_seg(
        #     mask_pred, labels, ignore_index=self.ignore_label)
        return dict(loss_sem_seg=loss_sem_seg)

    def get_semantic_seg(self, seg_preds, ori_shape, img_shape_withoutpad):
        """Obtain semantic segmentation map for panoptic segmentation.

        Args:
            seg_preds (torch.Tensor): Segmentation prediction
            ori_shape (tuple[int]): Input image shape with padding.
            img_shape_withoutpad (tuple[int]): Original image shape before
                without padding.
        Returns:
            list[list[np.ndarray]]: The decoded segmentation masks.
                The first dimension is the number of classes.
                The second dimension is the number of masks of a similar class.
        """
        # only surport 1 batch
        seg_preds = F.interpolate(
            seg_preds,
            scale_factor=self.pred_stride,
            mode='bilinear',
            align_corners=False)
        seg_preds = seg_preds[:, :, 0:img_shape_withoutpad[0],
                              0:img_shape_withoutpad[1]]
        # seg_masks = F.softmax(seg_preds, 1)
        # seg_masks = F.interpolate(
        #     seg_masks,
        #     size=ori_shape[0:2],
        #     mode='bilinear',
        #     align_corners=False)
        seg_results = F.interpolate(
            seg_preds,
            size=ori_shape[0:2],
            mode='bilinear',
            align_corners=False)
        return seg_results
