"""
This is a simple mask based tracker
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by Xiangtai Li

"""
# coding: utf-8
import torch
from scipy.optimize import linear_sum_assignment
from .util import generalized_box_iou, masks_to_boxes
import copy


class SimpleMaskTracker(object):
    def __init__(self, score_thresh, max_age=32):
        self.score_thresh = score_thresh
        self.max_age = max_age
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        self.reset_all()

    def reset_all(self):
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()

    def init_track(self, results):

        scores = results["scores"] # (n,)
        masks = results["masks"]  # (n,h,w)

        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["mask"] = masks[idx]
                obj["tracking_id"] = self.id_count
                obj['active'] = 1
                obj['age'] = 1
                ret.append(obj)
                ret_dict[idx] = obj

        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    def step(self, output_results, track_results):
        """
        Args:
            output_results: Current Frame Output including the tracked results
        Returns:
        """
        scores = output_results["scores"]  # (n,h,w)
        bboxes = output_results["masks"]  # (n,h,w)
        # track_bboxes = track_results["masks"]  # (m,h,w)

        results = list()
        results_dict = dict()

        # tracks = list()
        # for idx in range(scores.shape[0]):
        #     if idx in self.tracks_dict and idx < len(track_bboxes):
        #         self.tracks_dict[idx]["mask"] = track_bboxes[idx]
        #
        #     if scores[idx] >= self.score_thresh:
        #         obj = dict()
        #         obj["score"] = float(scores[idx])
        #         obj["mask"] = bboxes[idx]
        #         results.append(obj)
        #         results_dict[idx] = obj

        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)

        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box = masks_to_boxes(torch.stack([torch.tensor(obj['mask']) for obj in results], dim=0))  # N x h * w
            track_box = masks_to_boxes(torch.stack([torch.tensor(obj['mask']) for obj in tracks], dim=0))  # M x h * w
            cost_bbox = 1.0 - generalized_box_iou(det_box, track_box)  # N x M

            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[], []]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > 1.2:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            # handle the matched tracks
            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)

        curent_track = ret

        # handle the remaining tracks
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ret.append(track)
                ret_unmatched_tracks.append(track)

        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return curent_track
