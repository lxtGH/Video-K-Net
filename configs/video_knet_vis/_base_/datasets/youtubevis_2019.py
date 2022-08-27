# dataset settings
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

test_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='MultiScaleFlipAugVideo',
         img_scale=(640, 360),
         flip=False,
         transforms=[
             dict(type='SeqResize'),
             dict(type='SeqNormalize', **img_norm_cfg),
             dict(type='SeqPad', size_divisor=32),
             dict(
                 type='VideoCollect',
                 keys=['img'],
                 reject_empty=False,
                 num_ref_imgs=0,  # 0 means do not apply check
             ),
             dict(type='ConcatVideoReferences'),
             dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
         ])
]

dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = '2019'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
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
    ),
    val=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2019_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        ref_img_sampler=None,
        load_all_frames=True,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2019_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        ref_img_sampler=None,
        load_all_frames=True,
        pipeline=test_pipeline
    )
)
