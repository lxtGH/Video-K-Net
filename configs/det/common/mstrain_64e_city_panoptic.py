_base_ = '../_base_/default_runtime.py'
# dataset settings
dataset_type = 'CityscapesPanopticDataset'
data_root = 'data/cityscapes/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='Resize', img_scale=[(512, 1024), (2048, 4096)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 2048)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=dict(
                ins_ann=data_root + 'annotations/instancesonly_filtered_gtFine_train.json',
                panoptic_ann=data_root + 'annotations/cityscapes_panoptic_train.json'
            ),
            img_prefix=data_root + 'leftImg8bit/train/',
            seg_prefix=data_root + 'gtFine/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root +'annotations/instancesonly_filtered_gtFine_val.json',
            panoptic_ann=data_root + "annotations/cityscapes_panoptic_val.json"
        ),
        img_prefix=data_root + 'leftImg8bit/val/',
        seg_prefix=data_root + 'gtFine/cityscapes_panoptic_val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/instancesonly_filtered_gtFine_val.json',
            panoptic_ann=data_root + "annotations/cityscapes_panoptic_val.json"
        ),
        img_prefix=data_root + 'leftImg8bit/val/',
        seg_prefix=data_root + 'gtFine/cityscapes_panoptic_val',
        pipeline=test_pipeline))

evaluation = dict(metric=['panoptic'])

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
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8)  # actual epoch = 8 * 8 = 64
