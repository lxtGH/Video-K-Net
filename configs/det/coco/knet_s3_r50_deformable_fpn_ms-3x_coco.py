_base_ = [
    '../_base_/models/knet_s3_r50_deformable_fpn.py',
    '../common/mstrain_3x_coco_instance.py'
]

model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,),

)