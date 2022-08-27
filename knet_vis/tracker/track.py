import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
from mmdet.utils import get_root_logger
from mmdet.models import build_head

from knet_vis.det.utils import sem2ins_masks


@DETECTORS.register_module()
class KNetTrack(TwoStageDetector):

    def __init__(self,
                 *args,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 thing_label_in_seg=0,
                 direct_tracker=False,
                 tracker_num=1,
                 tracker=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        self.roi_head = None # init roi_head with None
        super().__init__(*args, **kwargs, train_cfg=train_cfg, test_cfg=test_cfg)
        assert self.with_rpn, 'KNet does not support external proposals'
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.direct_tracker = direct_tracker
        self.tracker_num = tracker_num
        if tracker is not None:
            rcnn_train_cfg = train_cfg.tracker if train_cfg is not None else None
            tracker.update(train_cfg=rcnn_train_cfg)
            tracker.update(test_cfg=test_cfg.tracker)
            self.tracker = build_head(tracker)
            if self.tracker_num > 1:
                self.tracker_extra = nn.ModuleList(
                    [build_head(tracker) for _ in range(tracker_num - 1)]
                )
        logger = get_root_logger()
        logger.info(f'Model: \n{self}')


    def gt_transform(self, img_metas, gt_masks, gt_labels, gt_semantic_seg):
        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

            if gt_semantic_seg is not None:
                # gt_semantic seg is padded by 255 and
                # zero indicating the first class
                sem_labels, sem_seg = sem2ins_masks(
                    gt_semantic_seg[i],
                    num_thing_classes=self.num_thing_classes)
                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(
                        mask_tensor.new_zeros(
                            (mask_tensor.size(0), assign_H, assign_W))
                    )
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0]
                    )
                gt_sem_cls.append(sem_labels)

            else:
                gt_sem_seg = None
                gt_sem_cls = None

            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W))
                )
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0]
                )
        return gt_masks_tensor, gt_sem_seg, gt_sem_cls

    def ref_gt_transform(self, ref_img_metas, ref_gt_masks, ref_gt_labels, ref_gt_semantic_seg=None ):
        # gt_masks and gt_semantic_seg are not padded when forming batch
        ref_gt_masks_tensor = []
        assert ref_gt_semantic_seg is None
        ref_gt_sem_seg = None
        ref_gt_sem_cls = None
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = ref_img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for bs_i, gt_mask_frame in enumerate(ref_gt_masks):
            batch_cur_gt_masks_tensor = []
            for i, gt_mask in enumerate(gt_mask_frame):
                mask_tensor = gt_mask.to_tensor(torch.float, ref_gt_labels[bs_i].device)
                if gt_mask.width != pad_W or gt_mask.height != pad_H:
                    pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                    mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

                if mask_tensor.shape[0] == 0:
                    batch_cur_gt_masks_tensor.append(
                        mask_tensor.new_zeros(
                            (mask_tensor.size(0), assign_H, assign_W))
                    )
                else:
                    batch_cur_gt_masks_tensor.append(
                        F.interpolate(
                            mask_tensor[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0]
                    )
            ref_gt_masks_tensor.append(batch_cur_gt_masks_tensor)

        return ref_gt_masks_tensor, ref_gt_sem_seg, ref_gt_sem_cls


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      gt_instance_ids=None,
                      # references
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_gt_instance_ids=None,
                      **kwargs):

        super(TwoStageDetector, self).forward_train(img, img_metas)
        assert proposals is None, 'KNet does not support external proposals'
        assert gt_masks is not None

        ref_gt_masks, ref_gt_sem_seg, ref_gt_sem_cls  = \
            self.ref_gt_transform(img_metas, ref_gt_masks, ref_gt_labels, ref_gt_semantic_seg=None)
        bs, num_frame, _, h, w = ref_img.size()
        x = self.extract_feat(ref_img.reshape(bs * num_frame, _, h, w))

        losses = dict()

        rpn_losses, proposal_feats, x_feats, mask_preds, cls_scores = \
            self.rpn_head.forward_train(x, img_metas, ref_img_metas, ref_gt_masks, ref_gt_labels,
                                        ref_gt_instance_ids, ref_gt_sem_seg, ref_gt_sem_cls)
        losses.update(rpn_losses)

        if self.roi_head is not None:
            roi_losses, features = self.roi_head.forward_train(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                ref_img_metas,
                ref_gt_masks,
                ref_gt_labels,
                gt_bboxes_ignore=ref_gt_bboxes_ignore,
                gt_bboxes=ref_gt_bboxes,
                gt_sem_seg=ref_gt_sem_seg,
                gt_sem_cls=ref_gt_sem_cls,
                imgs_whwh=None)
            losses.update(roi_losses)

        if self.direct_tracker:
            proposal_feats = self.rpn_head.init_kernels.weight.clone()
            proposal_feats = proposal_feats[None].expand(bs, *proposal_feats.size())
            if mask_preds.shape[0] == bs * num_frame:
                mask_preds = mask_preds.reshape((bs, num_frame, *mask_preds.size()[1:]))
                x_feats = x_feats.reshape((bs, num_frame, *x_feats.size()[1:]))
            else:
                assert mask_preds.size()[:2] == (bs, num_frame)
                assert x_feats.size()[:2] == (bs, num_frame)

            tracker_losses, features = self.tracker.forward_train(
                x=x_feats,
                ref_img_metas=ref_img_metas,
                cls_scores=None,
                masks=mask_preds,
                obj_feats=proposal_feats,
                ref_gt_masks=ref_gt_masks,
                ref_gt_labels=ref_gt_labels,
                ref_gt_instance_ids=ref_gt_instance_ids,
            )
            if self.tracker_num > 1:
                for i in range(self.tracker_num - 1):
                    _tracker_losses, features = self.tracker_extra[i].forward_train(
                        x=features['x_feats'],
                        ref_img_metas=ref_img_metas,
                        cls_scores=None,
                        masks=features['masks'],
                        obj_feats=features['obj_feats'],
                        ref_gt_masks=ref_gt_masks,
                        ref_gt_labels=ref_gt_labels,
                        ref_gt_instance_ids=ref_gt_instance_ids,
                    )
                    for key, value in _tracker_losses.items():
                        tracker_losses[f'extra_m{i}_{key}'] = value
        else:
            tracker_losses, _ = self.tracker.forward_train(
                x=features['x_feats'],
                ref_img_metas=ref_img_metas,
                cls_scores=features['cls_scores'],
                masks=features['masks'],
                obj_feats=features['obj_feats'],
                ref_gt_masks=ref_gt_masks,
                ref_gt_labels=ref_gt_labels,
                ref_gt_instance_ids=ref_gt_instance_ids,
            )

        losses.update(tracker_losses)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            kwargs['ref_img_metas'] = kwargs['ref_img_metas'][0]
            kwargs['ref_img'] = kwargs['ref_img'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, imgs, img_metas, **kwargs):
        ref_img = kwargs['ref_img']
        ref_img_metas = kwargs['ref_img_metas']
        # Step 1 extract features and get masks
        bs, num_frame, _, h, w = ref_img.size()
        x = self.extract_feat(ref_img.reshape(bs * num_frame, _, h, w))

        proposal_feats, x_feats, mask_preds, cls_scores, seg_preds = \
            self.rpn_head.simple_test_rpn(x, img_metas, ref_img_metas)

        if self.roi_head is not None:
            segm_results_single_frame, features = self.roi_head.simple_test(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas,
                ref_img_metas,
                imgs_whwh=None,
                rescale=True
            )

        if self.direct_tracker:
            proposal_feats = self.rpn_head.init_kernels.weight.clone()
            proposal_feats = proposal_feats[None].expand(bs, *proposal_feats.size())
            if mask_preds.shape[0] == bs * num_frame:
                mask_preds = mask_preds.reshape((bs, num_frame, *mask_preds.size()[1:]))
                x_feats = x_feats.reshape((bs, num_frame, *x_feats.size()[1:]))
            else:
                assert mask_preds.size()[:2] == (bs, num_frame)
                assert x_feats.size()[:2] == (bs, num_frame)
            segm_results, features = self.tracker.simple_test(
                x=x_feats,
                img_metas=img_metas,
                ref_img_metas=ref_img_metas,
                cls_scores=None,
                masks=mask_preds,
                obj_feats=proposal_feats,
            )
            if self.tracker_num > 1:
                for i in range(self.tracker_num - 1):
                    segm_results, features = self.tracker_extra[i].simple_test(
                        x=features['x_feats'],
                        img_metas=img_metas,
                        ref_img_metas=ref_img_metas,
                        cls_scores=None,
                        masks=features['masks'],
                        obj_feats=features['obj_feats'],
                    )
        else:
            segm_results, _ = self.tracker.simple_test(
                x=features['x_feats'],
                img_metas=img_metas,
                ref_img_metas=ref_img_metas,
                cls_scores=features['cls_scores'],
                masks=features['masks'],
                obj_feats=features['obj_feats'],
            )

        return segm_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_feats, proposal_feats, dummy_img_metas)
        return roi_outs

    def init_weights(self):
        super().init_weights()
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            assert self.tracker.init_cfg is None
            self.tracker.init_cfg = copy.deepcopy(self.init_cfg)
            self.tracker.init_cfg['prefix']='roi_head'
            self.tracker.init_weights()
            if self.tracker_num > 1:
                for _ in range(self.tracker_num - 1):
                    assert self.tracker_extra[_].init_cfg is None
                    self.tracker_extra[_].init_cfg = copy.deepcopy(self.init_cfg)
                    self.tracker_extra[_].init_cfg['prefix'] = 'roi_head'
                    self.tracker_extra[_].init_weights()
