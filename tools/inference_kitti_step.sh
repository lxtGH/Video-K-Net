#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
LOG=$3

# configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2.py logger/models/video_knet_vis/video_knet_step_quansi_r50.pth logger/results/kitti_step_merge_joint_semantic_filter
# configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2.py logger/models/video_knet_vis/video_knet_step_quansi_r50.pth logger/results/kitti_step_semantic_filter

# --cfg-options data.test.split=val model.roi_head.merge_joint=True model.semantic_filter=True
# --cfg-options data.test.split=val model.roi_head.merge_joint=False model.semantic_filter=True

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test_dvps.py $CONFIG $CHECKPOINT --eval dummy --show-dir $LOG ${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/eval_dstq_step.py $LOG
