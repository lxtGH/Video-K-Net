from typing import Sequence, Tuple
import collections

import numpy as np

from .STQ import STQuality


class DSTQuality(STQuality):
    def __init__(
            self,
            num_classes: int,
            things_list: Sequence[int],
            ignore_label: int,
            label_bit_shift: int,
            offset: int,
            depth_threshold: Tuple[float] = (1.25, 1.1),
            name: str = 'dstq'
    ):
        super().__init__(
            num_classes=num_classes,
            things_list=things_list,
            ignore_label=ignore_label,
            label_bit_shift=label_bit_shift,
            offset=offset
        )
        if not (isinstance(depth_threshold, tuple) or
                isinstance(depth_threshold, list)):
            raise TypeError('The type of depth_threshold must be tuple or list.')
        if not depth_threshold:
            raise ValueError('depth_threshold must be non-empty.')
        self._depth_threshold = tuple(depth_threshold)
        self._depth_total_counts = collections.OrderedDict()
        self._depth_inlier_counts = []
        for _ in range(len(self._depth_threshold)):
            self._depth_inlier_counts.append(collections.OrderedDict())

    def update_state(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            d_true: np.ndarray,
            d_pred: np.ndarray,
            sequence_id: int = 0
    ):
        """Accumulates the depth-aware segmentation and tracking quality statistics.
        Args:
          y_true: The ground-truth panoptic label map for a particular video frame
            (defined as semantic_map * max_instances_per_category + instance_map).
          y_pred: The predicted panoptic label map for a particular video frame
            (defined as semantic_map * max_instances_per_category + instance_map).
          d_true: The ground-truth depth map for this video frame.
          d_pred: The predicted depth map for this video frame.
          sequence_id: The optional ID of the sequence the frames belong to. When no
            sequence is given, all frames are considered to belong to the same
            sequence (default: 0).
        """
        super().update_state(y_true, y_pred, sequence_id)
        # Valid depth labels contain positive values.
        d_valid_mask = d_true > 0
        d_valid_total = np.sum(d_valid_mask.astype(int))
        # Valid depth prediction is expected to contain positive values.
        # TODO : very wrong implementation because it is hackable
        d_valid_mask = np.logical_and(d_valid_mask, d_pred > 0)
        d_valid_true = d_true[d_valid_mask]
        d_valid_pred = d_pred[d_valid_mask]
        inlier_error = np.maximum(d_valid_pred / d_valid_true,
                                  d_valid_true / d_valid_pred)
        # For each threshold, count the number of inliers.
        for threshold_index, threshold in enumerate(self._depth_threshold):
            num_inliers = np.sum((inlier_error <= threshold).astype(int))
            inlier_counts = self._depth_inlier_counts[threshold_index]
            inlier_counts[sequence_id] = (inlier_counts.get(sequence_id, 0) + int(num_inliers))
        # Update the total counts of the depth labels.
        self._depth_total_counts[sequence_id] = (
                self._depth_total_counts.get(sequence_id, 0) + int(d_valid_total))

    def result(self):
        """Computes the depth-aware segmentation and tracking quality.
        Returns:
          A dictionary containing:
            - 'STQ': The total STQ score.
            - 'AQ': The total association quality (AQ) score.
            - 'IoU': The total mean IoU.
            - 'STQ_per_seq': A list of the STQ score per sequence.
            - 'AQ_per_seq': A list of the AQ score per sequence.
            - 'IoU_per_seq': A list of mean IoU per sequence.
            - 'Id_per_seq': A list of sequence Ids to map list index to sequence.
            - 'Length_per_seq': A list of the length of each sequence.
            - 'DSTQ': The total DSTQ score.
            - 'DSTQ@thres': The total DSTQ score for threshold thres
            - 'DSTQ_per_seq@thres': A list of DSTQ score per sequence for thres.
            - 'DQ': The total DQ score.
            - 'DQ@thres': The total DQ score for threshold thres.
            - 'DQ_per_seq@thres': A list of DQ score per sequence for thres.
        """
        # Gather the results for STQ.
        stq_results = super().result()
        # Collect results for depth quality per sequecne and threshold.
        dq_per_seq_at_threshold = {}
        dq_at_threshold = {}
        for threshold_index, threshold in enumerate(self._depth_threshold):
            dq_per_seq_at_threshold[threshold] = [0] * len(self._ground_truth)
            total_count = 0
            inlier_count = 0
            # Follow the order of computing STQ by enumerating _ground_truth.
            for index, sequence_id in enumerate(self._ground_truth):
                sequence_inlier = self._depth_inlier_counts[threshold_index][sequence_id]
                sequence_total = self._depth_total_counts[sequence_id]
                if sequence_total > 0:
                    dq_per_seq_at_threshold[threshold][
                        index] = sequence_inlier / sequence_total
                total_count += sequence_total
                inlier_count += sequence_inlier
            if total_count == 0:
                dq_at_threshold[threshold] = 0
            else:
                dq_at_threshold[threshold] = inlier_count / total_count
        # Compute DQ as the geometric mean of DQ's at different thresholds.
        dq = 1
        for _, threshold in enumerate(self._depth_threshold):
            dq *= dq_at_threshold[threshold]
        dq = dq ** (1 / len(self._depth_threshold))
        dq_results = {}
        dq_results['DQ'] = dq
        for _, threshold in enumerate(self._depth_threshold):
            dq_results['DQ@{}'.format(threshold)] = dq_at_threshold[threshold]
            dq_results['DQ_per_seq@{}'.format(
                threshold)] = dq_per_seq_at_threshold[threshold]
        # Combine STQ and DQ to get DSTQ.
        dstq_results = {}
        dstq_results['DSTQ'] = (stq_results['STQ'] ** 2 * dq) ** (1 / 3)
        for _, threshold in enumerate(self._depth_threshold):
            dstq_results['DSTQ@{}'.format(threshold)] = (stq_results['STQ'] ** 2 * dq_at_threshold[
                                                            threshold]) ** (1 / 3)
            dstq_results['DSTQ_per_seq@{}'.format(threshold)] = [
                (stq_result ** 2 * dq_result) ** (1 / 3) for stq_result, dq_result in zip(
                    stq_results['STQ_per_seq'], dq_per_seq_at_threshold[threshold])
            ]
        # Merge all the results.
        dstq_results.update(stq_results)
        dstq_results.update(dq_results)
        return dstq_results

    def reset_states(self):
        """Resets all states that accumulated data."""
        super().reset_states()
        self._depth_total_counts = collections.OrderedDict()
        self._depth_inlier_counts = []
        for _ in range(len(self._depth_threshold)):
            self._depth_inlier_counts.append(collections.OrderedDict())
