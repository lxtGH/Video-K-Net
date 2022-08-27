import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import ProgressBar

import torch.nn.functional as F

from tools.utils.DSTQ import DSTQuality
from tools.utils.STQ import STQuality


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of DSTQ')
    parser.add_argument('result_path')
    parser.add_argument('--gt-path', default='data/kitti-step')
    parser.add_argument('--split', default='val')
    parser.add_argument(
        '--depth',
        action='store_true',
        help='eval depth')
    parser.add_argument('--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def updater(pred_ins_name,
            pred_cls_name,
            pred_dep_name,
            gt_pan_seq_name,
            gt_dep_seq_name,
            updater_obj,
            seq_id):
    pred_ins = mmcv.imread(pred_ins_name, flag='unchanged').astype(np.int32)
    pred_cls = mmcv.imread(pred_cls_name, flag='unchanged').astype(np.int32)
    pred_dep = mmcv.imread(pred_dep_name, flag='unchanged').astype(np.float32) if pred_dep_name is not None else None

    gt_pan = mmcv.imread(gt_pan_seq_name, flag='color', channel_order='rgb')
    gt_cls = gt_pan[..., 0].astype(np.int32)
    gt_ins = gt_pan[..., 1].astype(np.int32) * 256 + gt_pan[..., 2].astype(np.int32)
    gt_dep = mmcv.imread(gt_dep_seq_name, flag='unchanged').astype(np.float32) if gt_dep_seq_name is not None else None
    if pred_dep is not None:
        pred_dep = F.interpolate(torch.from_numpy(pred_dep)[None][None], size=gt_dep.shape)[0][0].numpy()

    valid_mask_seg = gt_cls != 255

    pred_masked_ps = pred_cls[valid_mask_seg] * (2 ** 16) + pred_ins[valid_mask_seg]
    gt_masked_ps = gt_cls[valid_mask_seg] * (2 ** 16) + gt_ins[valid_mask_seg]

    if pred_dep_name is not None:
        valid_mask_dep = gt_dep > 0.

        pred_masked_depth = pred_dep[valid_mask_dep]
        gt_masked_depth = gt_dep[valid_mask_dep]

        updater_obj.update_state(gt_masked_ps, pred_masked_ps, gt_masked_depth, pred_masked_depth, seq_id)
    else:
        updater_obj.update_state(gt_masked_ps, pred_masked_ps, seq_id)


def eval_dstq(result_dir, gt_dir, seq_ids, with_depth=True):
    if with_depth:
        dstq_obj = DSTQuality(
            num_classes=19,
            things_list=list(range(11, 19)),
            ignore_label=255,
            label_bit_shift=16,
            offset=2 ** 16 * 256,
            depth_threshold=(1.25,),
        )
    else:
        dstq_obj = STQuality(
            num_classes=19,
            things_list=list(range(11, 19)),
            ignore_label=255,
            label_bit_shift=16,
            offset=2 ** 16 * 256,
        )

    gt_names = list(mmcv.scandir(gt_dir))
    gt_pan_names = sorted(list(filter(lambda x: 'panoptic' in x, gt_names)))
    if with_depth:
        gt_dep_names = sorted(list(filter(lambda x: 'depth' in x, gt_names)))
    else:
        gt_dep_names = None

    for seq_id in seq_ids:
        pred_name_panoptic = list(mmcv.scandir(os.path.join(result_dir, 'panoptic', str(seq_id))))
        pred_ins_names = sorted(list(filter(lambda x: 'ins' in x, pred_name_panoptic)))
        pred_cls_names = sorted(list(filter(lambda x: 'cat' in x, pred_name_panoptic)))
        if with_depth:
            pred_name_depth = list(mmcv.scandir(os.path.join(result_dir, 'depth', str(seq_id))))
            pred_dep_names = sorted(pred_name_depth)
        else:
            pred_dep_names = [None] * len(pred_ins_names)
        gt_pan_seq_names = list(filter(lambda x: x.startswith('{:06d}'.format(seq_id)), gt_pan_names))
        if with_depth:
            gt_dep_seq_names = list(filter(lambda x: x.startswith('{:06d}'.format(seq_id)), gt_dep_names))
        else:
            gt_dep_seq_names = [None] * len(gt_pan_seq_names)
        prog_bar = ProgressBar(len(pred_ins_names))
        for pred_ins_name, pred_cls_name, pred_dep_name, gt_pan_seq_name, gt_dep_seq_name in zip(
                pred_ins_names, pred_cls_names, pred_dep_names, gt_pan_seq_names, gt_dep_seq_names
        ):
            prog_bar.update()
            updater(
                os.path.join(result_dir, 'panoptic', str(seq_id), pred_ins_name),
                os.path.join(result_dir, 'panoptic', str(seq_id), pred_cls_name),
                os.path.join(result_dir, 'depth', str(seq_id), pred_dep_name) if pred_dep_name is not None else None,
                os.path.join(gt_dir, gt_pan_seq_name),
                os.path.join(gt_dir, gt_dep_seq_name) if gt_dep_seq_name is not None else None,
                dstq_obj,
                seq_id
            )
    result = dstq_obj.result()
    print(result)


if __name__ == '__main__':
    args = parse_args()
    result_path = args.result_path
    gt_path = args.gt_path
    split = args.split
    eval_dstq(result_path, os.path.join(gt_path, 'video_sequence', split), [2, 6, 7, 8, 10, 13, 14, 16, 18], args.depth)
