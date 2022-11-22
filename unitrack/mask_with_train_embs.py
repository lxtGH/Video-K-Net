###################################################################
# File Name: mask.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri Jan 29 15:16:53 2021
###################################################################
import time

import numpy as np
import torch
import torch.nn.functional as F

from unitrack.utils.box import *
from unitrack.utils.mask import *
from .basetrack import *

from unitrack.model import AppearanceModel

class AssociationTrackerWithTrainedEmbed(object):
    def __init__(self, opt):
        self.opt = opt
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = opt.track_buffer
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        # self.app_model = AppearanceModel(opt).to(opt.device)
        # self.app_model.eval()

        if not self.opt.asso_with_motion:
            self.opt.motion_lambda = 1
            self.opt.motion_gated = False

    def extract_emb(self, img, obs):
        raise NotImplementedError

    def prepare_obs(self, img, img0, obs, embs=None):
        raise NotImplementedError

    def update(self, img, img0, obs, embs=None):
        torch.cuda.empty_cache()
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        t1 = time.time()
        detections = self.prepare_obs(img, img0, obs, embs=None)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        tracks = joint_stracks(tracked_stracks, self.lost_stracks)
        dists, recons_ftrk = matching.center_emb_distance(tracks, detections)
        if self.opt.use_kalman:
            # Predict the current location with KF
            STrack.multi_predict(tracks)
            dists = matching.fuse_motion(self.kalman_filter, dists, tracks, detections,
                                         lambda_=self.opt.motion_lambda, gate=self.opt.motion_gated)
        if obs.shape[1] == 6:
            dists = matching.category_gate(dists, tracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if self.opt.use_kalman:
            '''(optional) Step 3: Second association, with IOU'''
            tracks = [tracks[i] for i in u_track if tracks[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(tracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

            for itracked, idet in matches:
                track = tracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=self.opt.confirm_iou_thres)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        for it in u_track:
            track = tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks, ioudist=self.opt.dup_iou_thres)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    def reset_all(self, ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0


class MaskAssociationTracker(AssociationTrackerWithTrainedEmbed):
    def __init__(self, opt):
        super(MaskAssociationTracker, self).__init__(opt)

    def extract_emb(self, img, obs, embs):
        img = img.to(self.opt.device).float()
        obs = obs.to(self.opt.device).float()
        embs = embs.to(self.opt.device).float().unsqueeze(-1)
        # print(img.shape)
        # print("obs", obs.shape)
        # print("embs", embs.shape)
        # exit()
        # obs = F.interpolate(obs.unsqueeze(1), size=(h,w), mode='nearest')
        # template_scale = np.prod(self.opt.feat_size)
        embs_list = []
        for emb in embs:
            # obfeat = ob
            embs_list.append(emb.cpu())
            # scale = ob.sum()
            # if scale > 0:
            #     if scale > self.opt.max_mask_area:
            #         scale_factor = np.sqrt(self.opt.max_mask_area/scale.item())
            #     else:
            #         scale_factor = 1
            #     norm_obfeat = F.interpolate(obfeat, scale_factor=scale_factor, mode='bilinear')
            #     norm_mask = F.interpolate(ob.unsqueeze(1), scale_factor=scale_factor, mode='nearest')
            #     emb = norm_obfeat[:,:, norm_mask.squeeze(0).squeeze(0).ge(0.5)]
            #     embs.append(emb.cpu())
            # else:
            #     embs.append(torch.randn(d, template_scale))
        return obs, embs_list

    def prepare_obs(self, img, img0, obs, embs=None):
        ''' Step 1: Network forward, get detections & embeddings'''
        if obs.shape[0] > 0:
            if embs is not None:
                masks, embs = self.extract_emb(img, obs, embs)
            boxes = mask2box(masks)
            keep_idx = remove_duplicated_box(boxes, iou_th=0.7)
            boxes, masks, obs = boxes[keep_idx], masks[keep_idx], obs[keep_idx]
            embs = [embs[k] for k in keep_idx]
            detections = [STrack(tlbr_to_tlwh(tlbrs), 1, f, self.buffer_size, mask, ac=True) \
                    for (tlbrs,mask,f) in zip(boxes, obs, embs)]
        else:
            detections = []
        return detections


