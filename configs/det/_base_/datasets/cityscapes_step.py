dataset_type = 'CityscapesSTEP'
data_root = 'data/cityscapes'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsInstanceMasks', cherry=[11, 13]),
    dict(type='KNetInsAdapterCherryPick', stuff_nums=11, cherry=[11, 13]),
    dict(type='Resize', img_scale=(1024, 2048), ratio_range=[0.5, 2.0], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=(1024, 2048)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadFutureMMDet', size_divisor=32, pad_val=dict(img=0, masks=0, seg=255)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks', 'gt_labels', 'gt_semantic_seg'],
         meta_keys=('ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg')
         ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img'],
                 meta_keys=[
                     'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                     'flip_direction', 'img_norm_cfg'
                 ]),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            test_mode=False,
            pipeline=train_pipeline
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        test_mode=True,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        test_mode=True,
        pipeline=test_pipeline
    )
)

evaluation = dict()
