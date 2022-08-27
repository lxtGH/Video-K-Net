_base_ = '../_base_/default_runtime.py'
# dataset settings
dataset_type = 'CocoPanopticDatasetCustom'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=dict(
                ins_ann=data_root + 'annotations/panoptic_train2017_thing_only_coco.json',
                panoptic_ann=data_root + 'annotations/panoptic_train2017.json'),
            img_prefix=data_root + 'train2017/',
            seg_prefix=data_root + 'panoptic_stuff_train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/instances_val2017.json',
            panoptic_ann=data_root + 'annotations/panoptic_val2017.json'),
        seg_prefix=data_root + 'panoptic_val2017/',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/instances_val2017.json',
            panoptic_ann=data_root + 'annotations/panoptic_val2017.json'),
        seg_prefix=data_root + 'panoptic_val2017/',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

evaluation = dict(metric=['segm', 'panoptic'], interval=5)

checkpoint_config = dict(interval=5)

# optimizer
# this is different from the original 1x schedule that use SGD
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[42, 48])
runner = dict(type='EpochBasedRunner', max_epochs=50)
