dataset_type = 'KITTISTEPDVPSDataset'
data_root = 'data/kitti-step'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

# The kitti dataset contains 1226 x 370 and 1241 x 376
# 384 x 1248 is the minimum size that is 32-divisible
train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='LoadMultiAnnotationsDirect', with_depth=False, divisor=-1, cherry_pick=True, cherry=[11, 13]),
    dict(type='SeqResizeWithDepth', img_scale=(384, 1248), ratio_range=[0.5, 2.0], keep_ratio=True),
    dict(type='SeqFlipWithDepth', flip_ratio=0.5),
    dict(type='SeqRandomCropWithDepth', crop_size=(384, 1248), share_params=True),
    dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadImgDirect'),
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
                     'flip_direction', 'img_norm_cfg', 'ori_filename', "filename"
                 ]),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            ref_seq_index=None,
            test_mode=False,
            pipeline=train_pipeline,
            with_depth=False,
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_seq_index=None,
        test_mode=True,
        pipeline=test_pipeline,
        with_depth=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_seq_index=None,
        test_mode=True,
        pipeline=test_pipeline,
        with_depth=False,
    )
)

evaluation = dict()
