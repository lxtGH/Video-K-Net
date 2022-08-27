_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_citystep_s3_r50_fpn.py',
    '../_base_/datasets/cityscapes_step.py',
]


num_proposals = 100
load_from = "/mnt/lustre/lixiangtai/pretrained/video_knet_vis/knet_swin_b_city.pth"

work_dir = 'logger/blackhole'

runner = dict(type='EpochBasedRunner', max_epochs=8)

model = dict(
    type='KNet',
    backbone=dict(
        _delete_=True,
        type='SwinTransformerDIY',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        type='KernelIterHead',
        merge_joint=True,
    ),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
)
