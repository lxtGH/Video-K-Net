_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_track_r50_deformablefpn.py',
    '../_base_/datasets/youtubevis_2019.py',
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerDIY',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)

dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = '2019'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_track=True),
    dict(
        type='SeqResize',
        multiscale_mode='value',
        share_params=True,
        img_scale=[(288,1e6), (320,1e6), (352,1e6), (392,1e6), (416,1e6), (448,1e6), (480,1e6), (512,1e6)],
        keep_ratio=True
    ),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_instance_ids'],
        reject_empty=True,
        num_ref_imgs=5,
    ),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2019_train.json',
        img_prefix=data_root + 'train/JPEGImages',
        ref_img_sampler=dict(
            num_ref_imgs=5,
            frame_range=[-2, 2],
            filter_key_img=False,
            method='uniform'),
        pipeline=train_pipeline
    )),
)