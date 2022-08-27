import warnings
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import BaseDetector
from mmdet.models.builder import build_head, build_neck, build_backbone, build_roi_extractor
from mmdet.core import build_assigner, build_sampler
from knet.video.qdtrack.builder import build_tracker
from knet.det.utils import sem2ins_masks, sem2ins_masks_cityscapes, sem2ins_masks_kitti_step
from unitrack.mask import tensor_mask2box
from unitrack.utils.mask import mask2box, batch_mask2boxlist, bboxlist2roi

@DETECTORS.register_module()
class VideoKNetQuansiTrackROIGTBox(BaseDetector):
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
                 track_localization_fpn=None,
                 tracker=None,
                 train_cfg=None,
                 test_cfg=None,
                 track_train_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cityscapes=False,
                 kitti_step=False,
                 freeze_detector=False,
                 semantic_filter=False,
                 # linking parameters
                 link_previous=False,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 **kwargs):
        super(VideoKNetQuansiTrackROIGTBox, self).__init__(init_cfg)

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
            self.track_train_cfg = track_train_cfg
            self.track_head = build_head(track_head)
            self.init_track_assigner_sampler()
            if track_localization_fpn is not None:
                self.track_localization_fpn = build_neck(track_localization_fpn)

            self.track_roi_extractor = build_roi_extractor(
                bbox_roi_extractor)

        if tracker is not None:
            self.tracker_cfg = tracker

        if freeze_detector:
           self._freeze_detector()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_proposals = self.rpn_head.num_proposals
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.ignore_label = ignore_label
        self.cityscapes = cityscapes  # whether to train the cityscape panoptic segmentation
        self.kitti_step = kitti_step  # whether to train the kitti step panoptic segmentation

        self.semantic_filter = semantic_filter
        self.link_previous = link_previous

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def _freeze_detector(self):

        self.detector = [
            self.rpn_head, self.roi_head
        ]
        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""

        self.track_roi_assigner = build_assigner(
            self.track_train_cfg.assigner)
        self.track_share_assigner = False

        self.track_roi_sampler = build_sampler(
            self.track_train_cfg.sampler, context=self)
        self.track_share_sampler = False

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
                        label_shift=self.num_thing_classes,
                        thing_label_in_seg=list(range(self.num_stuff_classes,
                                                      self.num_thing_classes + self.num_stuff_classes))
                    )
                elif self.kitti_step:
                    sem_labels, sem_seg = sem2ins_masks_kitti_step(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=2,
                        thing_label_in_seg=(11, 13))
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
        img_h, img_w = batch_input_shape
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

        # prepare the gt_match_indices
        gt_pids_list = []
        for i in range(len(ref_gt_instance_id_list)):
            ref_ids = ref_gt_instance_id_list[i].cpu().data.numpy().tolist()
            gt_ids = gt_instance_ids[i].cpu().data.numpy().tolist()
            gt_pids = [ref_ids.index(i) if i in ref_ids else -1 for i in gt_ids]
            gt_pids_list.append(torch.LongTensor([gt_pids]).to(img.device)[0])

        gt_match_indices = gt_pids_list

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks, gt_sem_cls, gt_sem_seg = self.preprocess_gt_masks(img_metas, gt_masks, gt_labels, gt_semantic_seg)

        ref_gt_masks, ref_gt_sem_cls, ref_gt_sem_seg = self.preprocess_gt_masks(ref_img_metas_new,
                                                                    ref_masks_gt, ref_gt_labels, ref_semantic_seg_gt)

        x = self.extract_feat(img)
        x_ref = self.extract_feat(ref_img)

        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,
                                                  gt_labels, gt_sem_seg,
                                                  gt_sem_cls)

        # simple forward to get the reference results
        self.rpn_head.eval()
        ref_rpn_results = self.rpn_head.simple_test_rpn(x_ref, ref_img_metas_new)
        self.rpn_head.train()

        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results

        (ref_proposal_feats, ref_x_feats, ref_mask_preds,
         ref_cls_scores, ref_seg_preds) = ref_rpn_results

        ref_obj_feats,  ref_cls_scores, ref_mask_preds, ref_scaled_mask_preds = self.roi_head.simple_test_mask_preds(
            ref_x_feats,
            ref_proposal_feats,
            ref_mask_preds,
            ref_cls_scores,
            ref_img_metas_new,
           )

        if self.link_previous:
            losses, object_feats, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.forward_train_with_previous(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas,
                gt_masks,
                gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_bboxes=gt_bboxes,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls,
                imgs_whwh=None,
                previous_obj_feats=ref_obj_feats,
                previous_mask_preds=ref_scaled_mask_preds,
                previous_x_feats=ref_x_feats,
            )
        else:
            # forward to get the current results
            losses, object_feats, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.forward_train(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas,
                gt_masks,
                gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_bboxes=gt_bboxes,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls,
                imgs_whwh=None)

        # ===== Tracking Part -==== #
        # assign both key frame and reference frame tracking targets
        key_sampling_results, ref_sampling_results = [], []
        num_imgs = len(img_metas)

        for i in range(num_imgs):
            assign_result = self.track_roi_assigner.assign(
                scaled_mask_preds[i][:self.num_proposals].detach(), cls_scores[i][:self.num_proposals, :self.num_thing_classes].detach(),
                gt_masks[i], gt_labels[i], img_meta=img_metas[i])
            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                mask_preds[i][:self.num_proposals].detach(),
                gt_masks[i])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.track_roi_assigner.assign(
                ref_scaled_mask_preds[i][:self.num_proposals].detach(), ref_cls_scores[i][:self.num_proposals, :self.num_thing_classes].detach(),
                ref_gt_masks[i], ref_gt_labels[i], img_meta=ref_img_metas_new[i])
            ref_sampling_result = self.track_roi_sampler.sample(
                ref_assign_result,
                ref_mask_preds[i][:self.num_proposals].detach(),
                ref_gt_masks[i])
            ref_sampling_results.append(ref_sampling_result)

        # roi feature embeddings
        key_masks = [res.pos_gt_masks for res in key_sampling_results]
        for i in range(len(key_masks)):
            key_masks[i] = F.interpolate(key_masks[i].unsqueeze(0),
                                        size=(img_h, img_w), mode="bilinear", align_corners=False).squeeze(0)
            key_masks[i] = (key_masks[i].sigmoid() > 0.5).float()

        key_feats = self._track_forward(x, key_masks)

        # roi feature embeddings
        ref_masks = [res.pos_gt_masks for res in ref_sampling_results]
        for i in range(len(ref_masks)):
            ref_masks[i] = F.interpolate(ref_masks[i].unsqueeze(0),
                                        size=(img_h, img_w), mode="bilinear", align_corners=False).squeeze(0)
            ref_masks[i] = (ref_masks[i].sigmoid() > 0.5).float()

        ref_feats = self._track_forward(x_ref, ref_masks)

        match_feats = self.track_head.match(key_feats, ref_feats,
                                            key_sampling_results,
                                            ref_sampling_results)

        asso_targets = self.track_head.get_track_targets(
            gt_match_indices, key_sampling_results, ref_sampling_results)
        loss_track = self.track_head.loss(*match_feats, *asso_targets)

        losses.update(loss_track)
        losses.update(rpn_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False, ref_img=None, **kwargs):
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

        # whether is the first frame for such clips
        # assert 'city' in img_metas[0]['filename'] and 'iid' in img_metas[0]
        if "city" in img_metas[0]['filename']:
            iid = img_metas[0]['iid']
            fid = iid % 10000
            is_first = (fid == 1)
        elif "motchallenge" in img_metas[0]['filename']:
            iid = kwargs['img_id'][0].item()
            fid = iid % 10000
            is_first = (fid == 1)
            if is_first:
                print("First detected on {}".format(fid))
        else:
            iid = kwargs['img_id'][0].item()
            fid = iid % 10000
            is_first = (fid == 0)

        if is_first:
            self.init_tracker()
            self.obj_feats_memory = None
            self.x_feats_memory = None
            self.mask_preds_memory = None

        # for current frame
        x = self.extract_feat(img)
        # current frame inference
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results

        if self.link_previous:
            cur_segm_results, query_output, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.simple_test_with_previous(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas,
                previous_obj_feats=self.obj_feats_memory,
                previous_mask_preds=self.mask_preds_memory,
                previous_x_feats=self.x_feats_memory,
            )
            self.obj_feats_memory = query_output
            self.x_feats_memory = x_feats
            self.mask_preds_memory = scaled_mask_preds
        else:
            cur_segm_results, query_output, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.simple_test(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas)

        # for tracking part
        _, segm_result, mask_preds, panoptic_result, _ = cur_segm_results[0]
        panoptic_seg, segments_info = panoptic_result

        if self.semantic_filter:
            seg_preds = torch.nn.functional.interpolate(seg_preds, panoptic_seg.shape, mode='bilinear', align_corners=False)
            seg_preds = seg_preds.sigmoid()
            seg_out = seg_preds.argmax(1)
            semantic_thing = (seg_out < self.num_thing_classes).to(dtype=torch.float32)
        else:
            semantic_thing = 1.

        # get sorted tracking thing ids, labels, masks, score for tracking
        things_index_for_tracking, things_labels_for_tracking, thing_masks_for_tracking, things_score_for_tracking = \
            self.get_things_id_for_tracking(panoptic_seg, segments_info)
        things_labels_for_tracking = torch.Tensor(things_labels_for_tracking).to(cls_scores.device).long()
        if len(things_labels_for_tracking) > 0:
            things_bbox_for_tracking = torch.zeros((len(things_score_for_tracking), 5),
                                                   dtype=torch.float, device=x_feats.device)
            things_bbox_for_tracking[:, 4] = torch.tensor(things_score_for_tracking,
                                                          device=things_bbox_for_tracking.device)

            thing_masks_for_tracking_final = []
            for mask in thing_masks_for_tracking:
                thing_masks_for_tracking_final.append(torch.Tensor(mask).unsqueeze(0).to(
                    x_feats.device).float())
            thing_masks_for_tracking_final = torch.cat(thing_masks_for_tracking_final, 0)
            thing_masks_for_tracking = thing_masks_for_tracking_final
            thing_masks_for_tracking_with_semantic_filter = thing_masks_for_tracking_final * semantic_thing

        if len(things_labels_for_tracking) == 0:
            track_feats = None
        else:
            # tracking embedding features
            track_feats = self._track_forward(x, thing_masks_for_tracking_with_semantic_filter)

        if track_feats is not None:
            # assert len(things_id_for_tracking) == len(things_labels_for_tracking)
            things_bbox_for_tracking[:, :4] = torch.tensor(tensor_mask2box(thing_masks_for_tracking_with_semantic_filter),
                                                           device=things_bbox_for_tracking.device)
            bboxes, labels, ids = self.tracker.match(
                bboxes=things_bbox_for_tracking,
                labels=things_labels_for_tracking,
                track_feats=track_feats,
                frame_id=fid)
            ids = ids + 1
            ids[ids == -1] = 0
        else:
            ids = []

        track_maps = self.generate_track_id_maps(ids, thing_masks_for_tracking, panoptic_seg)

        semantic_map = self.get_semantic_seg(panoptic_seg, segments_info)

        from scripts.visualizer import trackmap2rgb, cityscapes_cat2rgb, draw_bbox_on_img
        vis_tracker = trackmap2rgb(track_maps)
        vis_sem = cityscapes_cat2rgb(semantic_map)
        if len(things_labels_for_tracking):
            vis_tracker = draw_bbox_on_img(vis_tracker, things_bbox_for_tracking.cpu().numpy())

        # Visualization end
        return semantic_map, track_maps, None, vis_sem, vis_tracker


    def _track_forward(self, x, mask_pred):
        """Track head forward function used in both training and testing.
        We use mask pooling to get the fine grain features"""
        if not self.training:
            mask_pred = [mask_pred]
        bbox_list = batch_mask2boxlist(mask_pred)
        track_rois = bboxlist2roi(bbox_list)
        track_rois = track_rois.clamp(min=0.0)
        track_feats = self.track_roi_extractor(x[:self.track_roi_extractor.num_inputs], track_rois)
        track_feats = self.track_head(track_feats)

        return track_feats

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

    def get_things_id_for_tracking(self, panoptic_seg, seg_infos):
        idxs = []
        labels = []
        masks = []
        score = []
        for segment in seg_infos:
            if segment['isthing'] == True:
                thing_mask = panoptic_seg == segment["id"]
                masks.append(thing_mask)
                idxs.append(segment["instance_id"])
                labels.append(segment['category_id'])
                score.append(segment['score'])
        return idxs, labels, masks, score


    def pack_things_object(self, object_feats, ref_object_feats):
        object_feats, ref_object_feats = object_feats.squeeze(-1).squeeze(-1), ref_object_feats.squeeze(-1).squeeze(-1)
        thing_object_feats = torch.split(object_feats, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        ref_thing_object_feats = torch.split(ref_object_feats, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        return thing_object_feats, ref_thing_object_feats

    def pack_things_masks(self, mask_pred, ref_mask_pred):
        thing_mask_pred = torch.split(mask_pred, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        ref_thing_thing_mask_pred = torch.split(ref_mask_pred, [self.roi_head.num_proposals, self.num_stuff_classes], dim=1)[0]
        return thing_mask_pred, ref_thing_thing_mask_pred

    def get_semantic_seg(self, panoptic_seg, segments_info):
        results = {}
        masks = []
        scores = []
        kitti_step2cityscpaes = [11, 13]
        semantic_seg = np.zeros(panoptic_seg.shape)
        for segment in segments_info:
            if segment['isthing'] == True:
                if self.kitti_step:
                    cat_cur = kitti_step2cityscpaes[segment["category_id"]]
                    semantic_seg[panoptic_seg == segment["id"]] = cat_cur
                else:
                    semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"] + 11
            else:
                # for stuff (0- n-1)
                if self.kitti_step:
                    cat_cur = segment["category_id"]
                    cat_cur -= 1
                    offset = 0
                    for thing_id in kitti_step2cityscpaes:
                        if cat_cur + offset >= thing_id:
                            offset += 1
                    cat_cur += offset
                    semantic_seg[panoptic_seg == segment["id"]] = cat_cur
                else:
                    semantic_seg[panoptic_seg == segment["id"]] = segment["category_id"] - 1
        return semantic_seg

    def generate_track_id_maps(self, ids, masks, panopitc_seg_maps):
        final_id_maps = np.zeros(panopitc_seg_maps.shape)
        if len(ids) == 0:
            return final_id_maps
        # assert len(things_mask_results) == len(track_results)
        masks = masks.bool()
        for i, id in enumerate(ids):
            mask = masks[i].cpu().numpy()
            final_id_maps[mask] = id
        return final_id_maps