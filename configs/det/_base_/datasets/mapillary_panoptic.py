dataset_type = 'MapillaryPanopticDataset'
data_root = 'data/mapillary/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=[(1024, 4096), (2048, 4096)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
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
        img_scale=(2048, 4096),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
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
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/coco/training.json',
            panoptic_ann=data_root + 'annotations/panoptic_train.json'
        ),
        img_prefix=data_root + 'training/images',
        seg_prefix=data_root + 'training/panoptic_stuff_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/coco/validation.json',
            panoptic_ann=data_root + 'annotations/panoptic_val.json'),
        seg_prefix=data_root + 'validation/panoptic',
        img_prefix=data_root + 'validation/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/coco/validation.json',
            panoptic_ann=data_root + 'annotations/panoptic_val.json'),
        seg_prefix=data_root + 'validation/panoptic',
        img_prefix=data_root + 'validation/images',
        pipeline=test_pipeline))

evaluation = dict(metric=['segm', 'panoptic'])
