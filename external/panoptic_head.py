import torch
from mmdet.core import bbox2result
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import StandardRoIHead


class PanopticTestMixin(object):

    def simple_test_semantic(self, x, img_metas):
        segm_feature_pred = self.semantic_head(x)
        semantic_seg_results = []
        for i, img_meta in enumerate(img_metas):
            semantic_seg_results.append(
                self.semantic_head.get_semantic_seg(segm_feature_pred[i:i + 1],
                                                    img_meta['ori_shape'],
                                                    img_meta['img_shape'])[0])

        return semantic_seg_results

    def generate_panoptic(self, det_bboxes, det_labels, mask_preds, sem_seg,
                          img_metas, merge_cfg):
        panoptic_results = []
        for i in range(len(img_metas)):
            panoptic_results.append(
                merge_stuff_thing(det_bboxes[i], det_labels[i], mask_preds[i],
                                  sem_seg[i], merge_cfg))
        return panoptic_results


@HEADS.register_module()
class PanopticHead(StandardRoIHead, PanopticTestMixin):
    """Panoptic Segmentation Head for Panoptic Seg."""

    def __init__(self, *args, semantic_head, **kwargs):
        super(PanopticHead, self).__init__(*args, **kwargs)
        self.semantic_head = build_head(semantic_head)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained)
        if self.with_semantic:
            self.semantic_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        if self.with_semantic:
            for i in range(gt_semantic_seg.shape[0]):
                gt_semantic_seg[i, :, img_metas[i]['img_shape']
                                [0]:, :] = self.semantic_head.ignore_label
                gt_semantic_seg[i, :, :, img_metas[i]['img_shape']
                                [1]:] = self.semantic_head.ignore_label
            seg_preds = self.semantic_head(x)
            seg_losses = self.semantic_head.loss(seg_preds, gt_semantic_seg)
            losses.update(seg_losses)

        return losses

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        raise NotImplementedError('PanopticHead does not support async test')

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            mask_preds = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            segm_results = mask2result(mask_preds, det_labels,
                                       self.mask_head.num_classes)

            if self.with_semantic:
                sem_seg = self.simple_test_semantic(x, img_metas)
                panoptic_results = self.generate_panoptic(
                    det_bboxes, det_labels, mask_preds, sem_seg, img_metas,
                    self.test_cfg.merge_stuff_thing)
                return list(zip(bbox_results, segm_results, panoptic_results))
            return list(zip(bbox_results, segm_results))


def mask2result(mask_preds, labels, num_classes):
    cls_segms = []
    for batch_id, mask_pred in enumerate(mask_preds):
        if isinstance(mask_pred, list):
            cls_segms.append(mask_pred)
            continue
        cls_segms.append([[] for _ in range(num_classes)])
        N = mask_preds[batch_id].shape[0]
        for i in range(N):
            cls_segms[batch_id][labels[batch_id][i]].append(
                mask_pred[i].detach().cpu().numpy())
    return cls_segms


def merge_stuff_thing(det_bboxes,
                      det_labels,
                      mask_preds,
                      sem_seg,
                      merge_cfg=None):
    """Merge stuff and thing segmentation maps.

    This function is modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/panoptic_fpn.py#L183  # noqa

    Args:
        det_bboxes  (torch.Tensor): Bounding boxes in shape (n, 5).
        det_labels (torch.Tensor): Labels of bounding boxes in shape (n, ).
        mask_preds (torch.Tensor): Mask prediction in the original image size.
        sem_seg (torch.Tensor): Semantic segmentation prediction in the original
            image size.
        merge_cfg (dict): The config dict containing merge hyper-parameters.
    """
    sem_seg = sem_seg.argmax(dim=0)
    box_scores = det_bboxes[:, -1]
    panoptic_seg = torch.zeros_like(sem_seg, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-box_scores)

    current_segment_id = 0
    segments_info = []

    if isinstance(mask_preds, list):
        instance_masks = None
    else:
        instance_masks = mask_preds.to(
            dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = box_scores[inst_id].item()
        if score < merge_cfg.instance_score_thr:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > merge_cfg.iou_thr:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append({
            'id': current_segment_id,
            'isthing': True,
            'score': score,
            'category_id': det_labels[inst_id].item(),
            'instance_id': inst_id.item(),
        })

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(sem_seg).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (sem_seg == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < merge_cfg.stuff_max_area:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append({
            'id': current_segment_id,
            'isthing': False,
            'category_id': semantic_label,
            'area': mask_area,
        })

    return panoptic_seg.cpu().numpy(), segments_info
