import warnings
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import BaseDetector
from mmdet.models.builder import build_head, build_neck, build_backbone
from knet.det.utils import sem2ins_masks, sem2ins_masks_cityscapes


@DETECTORS.register_module()
class VideoKNetFuseROITrack(BaseDetector):
    """
        Simple Extension of KNet to Video KNet by the implementation of VPSFuse Net.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 track_head=None,
                 extra_neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cityscapes=False,
                 **kwargs):
        super(VideoKNetFuseROITrack, self).__init__(init_cfg)

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if extra_neck is not None:
            self.extra_neck = build_neck(extra_neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        if track_head is not None:
            self.track_head = build_head(track_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.ignore_label = ignore_label
        self.cityscapes = cityscapes  # whether to train the cityscape panoptic segmentation

    def preprocess_gt_masks(self, img_metas, gt_masks, gt_labels, gt_semantic_seg):
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
                # gt_semantic seg is padded by zero when forming a batch
                # need to convert them from 0 to ignore
                gt_semantic_seg[
                i, :, img_metas[i]['img_shape'][0]:, :] = self.ignore_label
                gt_semantic_seg[
                i, :, :, img_metas[i]['img_shape'][1]:] = self.ignore_label
                if self.cityscapes:
                    sem_labels, sem_seg = sem2ins_masks_cityscapes(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=self.num_thing_classes)
                else:
                    sem_labels, sem_seg = sem2ins_masks(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=self.num_thing_classes,
                        thing_label_in_seg=self.thing_label_in_seg)

                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(
                        mask_tensor.new_zeros(
                            (mask_tensor.size(0), assign_H, assign_W)))
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0])
                gt_sem_cls.append(sem_labels)
            else:
                gt_sem_seg = None
                gt_sem_cls = None

            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),  # downsample to 1/4 resolution
                        mode='bilinear',
                        align_corners=False)[0])

        return gt_masks_tensor, gt_sem_cls, gt_sem_seg

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_instance_ids=None,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_labels=None,
                      ref_gt_bboxes=None,
                      ref_gt_masks=None,
                      ref_gt_semantic_seg=None,
                      ref_gt_instance_ids=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN-like network in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

            # This is for video only:
            ref_img (Tensor): of shape (N, 2, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                2 denotes there is two reference images for each input image.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): The list only has one Tensor. The
                Tensor contains ground truth bboxes for each reference image
                with shape (num_all_ref_gts, 5) in
                [ref_img_id, tl_x, tl_y, br_x, br_y] format. The ref_img_id
                start from 0, and denotes the id of reference image for each
                key image.

            ref_gt_labels (list[Tensor]): The list only has one Tensor. The
                Tensor contains class indices corresponding to each reference
                box with shape (num_all_ref_gts, 2) in
                [ref_img_id, class_indice].

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        assert proposals is None, 'KNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None
        assert gt_instance_ids is not None

        # preprocess the reference images
        ref_img = ref_img.squeeze(1)  # (b,3,h,w)
        ref_masks_gt = []
        for ref_gt_mask in ref_gt_masks:
            ref_masks_gt.append(ref_gt_mask[0])

        ref_labels_gt = []
        for ref_gt_label in ref_gt_labels:
            ref_labels_gt.append(ref_gt_label[:, 1].long())
        ref_gt_labels = ref_labels_gt

        ref_semantic_seg_gt = ref_gt_semantic_seg.squeeze(1)

        ref_gt_instance_id_list = []
        for ref_gt_instance_id in ref_gt_instance_ids:
            ref_gt_instance_id_list.append(ref_gt_instance_id[:,1].long())

        ref_img_metas_new = []
        for ref_img_meta in ref_img_metas:
            ref_img_meta[0]['batch_input_shape'] = batch_input_shape
            ref_img_metas_new.append(ref_img_meta[0])

        gt_pids_list =[]
        for i in range(len(ref_gt_instance_id_list)):
            ref_ids = ref_gt_instance_id_list[i].cpu().data.numpy().tolist()
            gt_ids = gt_instance_ids[i].cpu().data.numpy().tolist()
            gt_pids = [ref_ids.index(i) + 1 if i in ref_ids else 0 for i in gt_ids]
            gt_pids_list.append(torch.LongTensor([gt_pids]).to(img.device)[0])
        gt_pids = gt_pids_list

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks, gt_sem_cls, gt_sem_seg = self.preprocess_gt_masks(img_metas, gt_masks, gt_labels, gt_semantic_seg)

        ref_gt_masks, ref_gt_sem_cls, ref_gt_sem_seg = self.preprocess_gt_masks(ref_img_metas_new,
                                                                    ref_masks_gt, ref_labels_gt, ref_semantic_seg_gt)

        x = self.extract_feat(img)
        x_ref = self.extract_feat(ref_img)
        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,
                                                  gt_labels, gt_sem_seg,
                                                  gt_sem_cls)

        ref_rpn_results = self.rpn_head.forward_train(x_ref, ref_img_metas_new, ref_gt_masks,
                                                  ref_labels_gt, ref_gt_sem_seg,
                                                  ref_gt_sem_cls)

        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results

        (ref_rpn_losses, ref_proposal_feats, ref_x_feats, ref_mask_preds,
         ref_cls_scores) = ref_rpn_results

        losses, sample_results, object_feats, mask_preds = self.roi_head.forward_train(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_labels,
            gt_pids=gt_pids,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            imgs_whwh=None)

        ref_losses, ref_sample_results, ref_object_feats, ref_mask_preds = self.roi_head.forward_train(
            ref_x_feats,
            ref_proposal_feats,
            ref_mask_preds,
            ref_cls_scores,
            ref_img_metas,
            ref_gt_masks,
            ref_gt_labels,
            gt_bboxes=ref_gt_bboxes,
            gt_bboxes_ignore=ref_gt_bboxes_ignore,
            gt_sem_seg=ref_gt_sem_seg,
            gt_sem_cls=ref_gt_sem_cls,
            imgs_whwh=None)
        proposals_nums = [self.roi_head.num_proposals] * img.size()[0]
        ref_proposals_nums = proposals_nums

        thing_mask_preds, ref_thing_mask_preds = self.pack_things_masks(mask_preds, ref_mask_preds)
        match_score = self.track_head(x, x_ref, thing_mask_preds, ref_thing_mask_preds, proposals_nums, ref_proposals_nums)

        track_loss = self.track_head.loss(match_score, sample_results)

        # format the loss
        ref_rpn_losses = self.add_ref_rpn_loss(ref_rpn_losses)
        ref_losses = self.add_ref_rpn_loss(ref_losses)

        losses.update(ref_rpn_losses)
        losses.update(ref_losses)
        losses.update(track_loss)
        losses.update(rpn_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False, ref_img=None):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        if ref_img is not None:
            ref_img = ref_img[0]
        # whether is the first frame for such clips
        assert 'city' in img_metas[0]['filename'] and 'iid' in img_metas[0]
        iid = img_metas[0]['iid']
        fid = iid % 10000
        is_first = (fid == 1)

        # for current frame
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results

        if not is_first:
            ref_x = self.extract_feat(ref_img)
            ref_rpn_results = self.rpn_head.simple_test_rpn(ref_x, img_metas)
            (ref_proposal_feats, ref_x_feats, ref_mask_preds, ref_cls_scores,
             ref_seg_preds) = ref_rpn_results
            x_fuse = self.combine(ref_x_feats + x_feats)

        cur_segm_results, cur_object_query = self.roi_head.simple_test(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            imgs_whwh=None,
            rescale=rescale)

        bbox_result, segm_result, panoptic_result = cur_segm_results[0]

        panoptic_seg, segments_info = panoptic_result

        cur_results, sseg_results = self.pack_stuff_things_result(panoptic_seg, segments_info)

        if is_first:
            self.track_query = cur_object_query

        if not is_first:
            track_seg_results = self.track_roi_head.simple_test(
                    x_fuse,
                    self.track_query,
                    ref_mask_preds,
                    ref_cls_scores,
                    img_metas,
                    imgs_whwh=None,
                    rescale=rescale
            )
            bbox_result, segm_result, panoptic_result = track_seg_results[0]
            track_panoptic_seg, track_segments_info = panoptic_result
            track_results, ref_sseg_results = self.pack_stuff_things_result(track_panoptic_seg, track_segments_info)

            # update the tracking query
            self.track_query = cur_object_query

        if is_first:
            self.tracker.reset_all()
            init_track_results = self.tracker.init_track(cur_results)
            track_maps = self.generate_track_id_maps(init_track_results, panoptic_seg)

        elif not is_first:
            results = self.tracker.step(cur_results, track_results)
            track_maps = self.generate_track_id_maps(results, panoptic_seg)

        return cur_segm_results, track_maps, sseg_results

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
        roi_outs = self.roi_head.forward_dummy(x_feats, proposal_feats,
                                               dummy_img_metas)
        return roi_outs

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass

    def pack_things_object(self, object_feats, ref_object_feats):
        object_feats, ref_object_feats = object_feats.squeeze(-1).squeeze(-1), ref_object_feats.squeeze(-1).squeeze(-1)
        thing_object_feats = torch.split(object_feats, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        ref_thing_object_feats = torch.split(ref_object_feats, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        return thing_object_feats, ref_thing_object_feats

    def pack_things_masks(self, mask_pred, ref_mask_pred):
        thing_mask_pred = torch.split(mask_pred, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        ref_thing_thing_mask_pred= torch.split(ref_mask_pred, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        return thing_mask_pred, ref_thing_thing_mask_pred

    def add_track_loss(self, loss_dict):
        track_loss ={}
        for k,v in loss_dict.items():
            track_loss[str(k)+"_track"] = v
        return track_loss

    def add_ref_rpn_loss(self, loss_dict):
        ref_rpn_loss = {}
        for k,v in loss_dict.items():
            ref_rpn_loss[str(k) +"_ref"] = v
        return ref_rpn_loss

    def pack_stuff_things_result(self, panoptic_seg, segments_info):
        results = {}
        masks = []
        scores = []
        semantic_seg = np.zeros(panoptic_seg.shape)
        for segment in segments_info:
            if segment['isthing'] == True:
                thing_mask = panoptic_seg == segment["id"]
                masks.append(thing_mask)
                scores.append(segment["score"])
                # for things to shift the labels
                # (n - c)
                semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"] + 11
            else:
                # for stuff (0- n-1)
                semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"] - 1

        results["masks"] = np.array(masks)  # (N)
        results["scores"] = np.array(scores)  # (N,H,W)

        return results, semantic_seg

    def generate_track_id_maps(self, track_results, panopitc_seg_maps):
        final_id_maps = np.zeros(panopitc_seg_maps.shape)
        # assert len(things_mask_results) == len(track_results)
        for track in track_results:
            id = track["tracking_id"]
            mask = track["mask"]
            final_id_maps[mask] = id
        return final_id_maps