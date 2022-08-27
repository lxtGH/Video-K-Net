"""
    This file implements several tracking heads
"""
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmdet.models.builder import HEADS, build_head, build_loss, build_roi_extractor
from mmdet.models.losses import accuracy
from mmdet.core import multi_apply, bbox2roi
from knet.video.qdtrack.track import cal_similarity
from unitrack.utils.mask import mask2box, batch_mask2boxlist, bboxlist2roi


@HEADS.register_module()
class QueryTrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 num_fcs=2,
                 in_channels=256,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 dynamic=True,
                 loss_match=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):

        super(QueryTrackHead, self).__init__()
        self.in_channels = in_channels
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            out_channels = (in_channels
                           if i < num_fcs - 1  else fc_out_channels)
            fc = nn.Linear(in_channels, out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic = dynamic
        assert self.dynamic == True, "Naive tracking embedding head must be dynamic"
        #### modification
        self.loss_match = build_loss(loss_match)

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1).to(torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1).to(torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta), dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert (len(self.match_coeff) == 3)
            return (match_ll +
                    self.match_coeff[0] * torch.log(bbox_scores) +
                    self.match_coeff[1] * bbox_ious +
                    self.match_coeff[2] * label_delta)

    def forward(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        assert len(x_n) == len(ref_x_n)  # ==> the batch size should be the same.
        b, N, d = x.size()
        x = x.reshape(b*N, d)
        ref_x = ref_x.reshape(b*N, d)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros(m, 1).to(torch.cuda.current_device())

                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)

        return match_score

    def loss(self,
             match_score,
             sampling_results):
        losses = dict()
        n = len(match_score)
        x_n = [s.size(0) for s in match_score]
        ids, id_weights = self.get_targets(sampling_results)
        ids = torch.split(ids, x_n, dim=0)
        id_weights = torch.split(id_weights, x_n, dim=0)
        loss_match = 0.0
        match_acc = 0.0
        n_total = 0

        for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze()
            if len(valid_idx.size()) == 0:
                continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            loss_match_per_batch = self.loss_match(score, cur_ids, cur_weights)
            match_acc += accuracy(
                torch.index_select(score, 0, valid_idx),
                torch.index_select(cur_ids, 0, valid_idx)) * n_valid
            loss_match += loss_match_per_batch
        if loss_match == 0.0:
            losses['loss_match'] = ids[0].sum() * 0
        else:
            losses['loss_match'] = loss_match / n
        return losses

    def get_targets(self,
                    sampling_results,
                    concat=True,
                    ):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_pid_list = [res.pos_gt_pids for res in sampling_results]
        ids, id_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_pid_list)
        if concat:
            ids = torch.cat(ids, 0)
            id_weights = torch.cat(id_weights, 0)

        return ids, id_weights

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask, pos_gt_pid_list):

        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg

        ids = pos_mask.new_zeros((num_samples,), dtype=torch.long)
        ids_weights = pos_mask.new_zeros((num_samples,))
        if num_pos > 0:
            ids[pos_inds] = pos_gt_pid_list
            ids_weights[pos_inds] = 1.0

        if num_neg > 0:
            ids_weights[neg_inds] = 0.0

        return ids, ids_weights


@HEADS.register_module()
class TrackHeadWithROIAlign(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 num_fcs=2,
                 in_channels=256,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 dynamic=True,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 loss_match=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):

        super(TrackHeadWithROIAlign, self).__init__()
        assert bbox_roi_extractor is not None
        self.in_channels = in_channels
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        self.fcs = nn.ModuleList()

        for i in range(num_fcs):
            out_channels = (in_channels
                           if i < num_fcs - 1  else fc_out_channels)
            fc = nn.Linear(in_channels, out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic = dynamic
        assert self.dynamic == True, "Naive tracking embedding head must be dynamic"

        self.bbox_roi_extractor = build_roi_extractor(
                bbox_roi_extractor)
        #### modification
        self.loss_match = build_loss(loss_match)

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1).to(torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1).to(torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta), dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert (len(self.match_coeff) == 3)
            return (match_ll +
                    self.match_coeff[0] * torch.log(bbox_scores) +
                    self.match_coeff[1] * bbox_ious +
                    self.match_coeff[2] * label_delta)

    def forward(self, x, ref_x, mask_pred, ref_mask_pred, x_n, ref_x_n):
        """
        Args:
            x: backbone feature of current frame
            ref_x: backbone feature of reference frame
            mask_pred: mask prediction of current frame
            ref_mask_pred: reference mask prediction
            x_n: number of proposal
            ref_x_n:  number of proposal in ref frame

        Returns:

        """
        # print("mask shape ",mask_pred.shape)
        bbox_pred = batch_mask2boxlist(mask_pred)
        ref_bbox_pred = batch_mask2boxlist(ref_mask_pred)

        # rois = bboxlist2roi(bbox_pred)
        # ref_rois = bboxlist2roi(ref_bbox_pred)

        x = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        ref_x = self.bbox_roi_extractor(
                ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching

        b, N, d = x.size()
        x = x.reshape(b*N, d)
        ref_x = ref_x.reshape(b*N, d)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros(m, 1).to(torch.cuda.current_device())

                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)

        return match_score

    def loss(self,
             match_score,
             sampling_results):
        losses = dict()
        n = len(match_score)
        x_n = [s.size(0) for s in match_score]
        ids, id_weights = self.get_targets(sampling_results)
        ids = torch.split(ids, x_n, dim=0)
        id_weights = torch.split(id_weights, x_n, dim=0)
        loss_match = torch.zeros(0).to(torch.cuda.current_device())
        match_acc = 0.
        n_total = 0

        for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze()
            if len(valid_idx.size()) == 0:
                continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            loss_match += self.loss_match(
                score, cur_ids, cur_weights)
            match_acc += accuracy(
                torch.index_select(score, 0, valid_idx),
                torch.index_select(cur_ids, 0, valid_idx)) * n_valid
        losses['loss_match'] = loss_match / n
        if n_total > 0:
            losses['match_acc'] = match_acc / n_total
        return losses

    def get_targets(self,
                    sampling_results,
                    concat=True,
                    ):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_pid_list = [res.pos_gt_pids for res in sampling_results]
        ids, id_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_pid_list)
        if concat:
            ids = torch.cat(ids, 0)
            id_weights = torch.cat(id_weights, 0)

        return ids, id_weights

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask, pos_gt_pid_list):

        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg

        ids = pos_mask.new_zeros((num_samples,), dtype=torch.long)
        ids_weights = pos_mask.new_zeros((num_samples,))
        if num_pos > 0:
            ids[pos_inds] = pos_gt_pid_list
            ids_weights[pos_inds] = 1.0

        if num_neg > 0:
            ids_weights[neg_inds] = 0.0

        return ids, ids_weights


@HEADS.register_module()
class QuasiDenseMaskEmbedHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 embed_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 softmax_temp=-1,
                 loss_track=dict(
                     type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                 loss_track_aux=dict(
                     type='L2Loss',
                     sample_ratio=3,
                     margin=0.3,
                     loss_weight=1.0,
                     hard_mining=True)):
        super(QuasiDenseMaskEmbedHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)
        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)

        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        return x

    def get_track_targets(self, gt_match_indices, key_sampling_results,
                          ref_sampling_results):
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(gt_match_indices,
                                                       key_sampling_results,
                                                       ref_sampling_results):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_masks.size(0), ref_res.pos_masks.size(0)),
                dtype=torch.int)
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (_match_indices.view(
                -1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)).int()
            targets[:, :pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds, key_sampling_results,
              ref_sampling_results):

        num_key_rois = [res.pos_masks.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        num_ref_rois = [res.pos_masks.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method='dot_product',
                temperature=self.softmax_temp)
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(
                    key_embed, ref_embed, method='cosine')
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum())
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]


@HEADS.register_module()
class QuasiDenseMaskEmbedHeadGTMask(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 embed_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 softmax_temp=-1,
                 loss_track=dict(
                     type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                 loss_track_aux=dict(
                     type='L2Loss',
                     sample_ratio=3,
                     margin=0.3,
                     loss_weight=1.0,
                     hard_mining=True)):
        super(QuasiDenseMaskEmbedHeadGTMask, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)
        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)

        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        return x

    def get_track_targets(self, gt_match_indices, key_sampling_results,
                          ref_sampling_results):
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(gt_match_indices,
                                                       key_sampling_results,
                                                       ref_sampling_results):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_masks.size(0), ref_res.pos_masks.size(0)),
                dtype=torch.int)
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (_match_indices.view(
                -1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)).int()
            targets[:, :pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds, key_sampling_results,
              ref_sampling_results):
        num_key_rois = [res.pos_masks.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        num_ref_rois = [res.pos_masks.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method='dot_product',
                temperature=self.softmax_temp)
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(
                    key_embed, ref_embed, method='cosine')
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum())
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]