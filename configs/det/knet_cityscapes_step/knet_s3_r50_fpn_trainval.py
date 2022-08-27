_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_citystep_s3_r50_fpn.py',
    '../_base_/datasets/cityscapes_step_trainval.py',
]


num_proposals = 100
load_from = "work_dirs/city_step/r50_joint_baseline_init_ce_loss/latest.pth"

work_dir = 'logger/blackhole'

runner = dict(type='EpochBasedRunner', max_epochs=8)

model = dict(
    type='KNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
            type='KernelIterHead',
            merge_joint=True,),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=num_proposals,
            mask_thr=0.5,
            stuff_score_thr=0.05,
            merge_stuff_thing=dict(
                overlap_thr=0.6,
                iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3)))
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[7, ],
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
