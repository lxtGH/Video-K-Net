from typing import List

import torch
import torch.nn.functional as F
from mmdet.utils import get_root_logger


def sem2ins_masks(gt_sem_seg,
                  ignore_label=255,
                  label_shift=80,
                  thing_label_in_seg=0):
    classes = torch.unique(gt_sem_seg)
    ins_masks = []
    ins_labels = []
    for i in classes:
        # skip ignore class 255 and "special thing class" in semantic seg
        if i == ignore_label or i == thing_label_in_seg:
            continue
        ins_labels.append(i)
        ins_masks.append(gt_sem_seg == i)
    # 0 is the special thing class in semantic seg, so we also shift it by 1
    # Thus, 0-79 is foreground classes of things (similar in instance seg)
    # 80-151 is foreground classes of stuffs (shifted by the original index)
    if len(ins_labels) > 0:
        ins_labels = torch.stack(ins_labels) + label_shift - 1
        ins_masks = torch.cat(ins_masks)
    else:
        ins_labels = gt_sem_seg.new_zeros(size=[0])
        ins_masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return ins_labels.long(), ins_masks.float()


def sem2ins_masks_cityscapes(gt_sem_seg,
                             ignore_label=255,
                             label_shift=8,
                             thing_label_in_seg=list(range(11, 19))):
    """
        Shift the cityscapes semantic labels to instance labels and masks.
    """
    # assert label range from 0-18 (255)
    classes = torch.unique(gt_sem_seg)
    ins_masks = []
    ins_labels = []
    for i in classes:
        # skip ignore class 255 and "special thing class" in semantic seg
        if i == ignore_label or i in thing_label_in_seg:
            continue
        ins_labels.append(i)
        ins_masks.append(gt_sem_seg == i)
    # For cityscapes, 0-7 is foreground classes of things (similar in instance seg)
    # 8-18 is foreground classes of stuffs (shifted by the original index)
    if len(ins_labels) > 0:
        ins_labels = torch.stack(ins_labels) + label_shift
        ins_masks = torch.cat(ins_masks)
    else:
        ins_labels = gt_sem_seg.new_zeros(size=[0])
        ins_masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return ins_labels.long(), ins_masks.float()


def sem2ins_masks_kitti_step(gt_sem_seg,
                             ignore_label=255,
                             label_shift=2,
                             thing_label_in_seg=(11,13)):
    """
        Shift the cityscapes semantic labels to instance labels and masks.
    """
    # assert label range from 0-18 (255)
    classes = torch.unique(gt_sem_seg)
    ins_masks = []
    ins_labels = []
    for i in classes:
        # skip ignore class 255 and "special thing class" in semantic seg
        if i == ignore_label or i in thing_label_in_seg:
            continue
        offset = 0
        for thing_label in thing_label_in_seg:
            if i > thing_label:
                offset -= 1
        ins_labels.append(i + offset)
        ins_masks.append(gt_sem_seg == i)
    # For cityscapes, 0-7 is foreground classes of things (similar in instance seg)
    # 8-18 is foreground classes of stuffs (shifted by the original index)
    if len(ins_labels) > 0:
        ins_labels = torch.stack(ins_labels) + label_shift
        ins_masks = torch.cat(ins_masks)
    else:
        ins_labels = gt_sem_seg.new_zeros(size=[0])
        ins_masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return ins_labels.long(), ins_masks.float()