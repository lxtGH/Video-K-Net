# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.core import bbox2result

def outs2results(bboxes=None,
                 labels=None,
                 masks=None,
                 ids=None,
                 num_classes=None,
                 **kwargs):
    """Convert tracking/detection results to a list of numpy arrays.
    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        masks (torch.Tensor | np.ndarray): shape (n, h, w)
        ids (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, not including background class
    Returns:
        dict[str : list(ndarray) | list[list[np.ndarray]]]: tracking/detection
        results of each class. It may contain keys as belows:
        - bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.
        - mask_results (list[list[np.ndarray]]): Each outer list denotes masks
            of one category. Each inner list denotes one mask belonging to
            the category. Each mask has shape (h, w).
    """
    assert labels is not None
    assert num_classes is not None

    results = dict()

    if ids is not None:
        valid_inds = ids > -1
        ids = ids[valid_inds]
        labels = labels[valid_inds]

    if bboxes is not None:
        if ids is not None:
            bboxes = bboxes[valid_inds]
            if bboxes.shape[0] == 0:
                bbox_results = [
                    np.zeros((0, 6), dtype=np.float32)
                    for i in range(num_classes)
                ]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate(
                        (ids[labels == i, None], bboxes[labels == i, :]),
                        axis=1) for i in range(num_classes)
                ]
        else:
            bbox_results = bbox2result(bboxes, labels, num_classes)
        results['bbox_results'] = bbox_results

    if masks is not None:
        if ids is not None:
            masks = masks[valid_inds]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks_results = [[] for _ in range(num_classes)]
        for i in range(bboxes.shape[0]):
            masks_results[labels[i]].append(masks[i])
        results['mask_results'] = masks_results

    return results