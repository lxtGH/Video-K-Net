dataset_type = 'CityscapesVPSDataset'
data_root = 'data/cityscapes_vps/'
dataset_type_test = "CityscapesPanopticDataset"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='SeqResize', img_scale=[(512, 1024), (2048, 4096)], multiscale_mode='range', keep_ratio=True),
    dict(type='SeqRandomFlip',  share_params=True, flip_ratio=0.5),
    dict(type='SeqRandomCrop',  crop_size=(1024, 1024), share_params=True),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', "gt_instance_ids"]),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]


test_pipeline = [
    dict(type='LoadRefImageFromFile'),

    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2048, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'ref_img']),
            dict(type='Collect', keys=['img', 'ref_img']),
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
            ann_file=dict(ins_ann=data_root +'instances_train_city_vps_rle.json',
                          panoptic_ann=data_root + 'panoptic_im_train_city_vps.json'
                          ),
            img_prefix=data_root + 'train/img/',
            seg_prefix=data_root + 'train/labelmap/',
            pipeline=train_pipeline,
            offsets=[-1,+1])),
    val=dict(
        type=dataset_type_test,
        ann_file=dict(ins_ann=data_root + 'instances_val_city_vps_rle.json',
                      panoptic_ann=data_root + 'panoptic_gt_val_city_vps.json',
                      vps=True
                      ),
        img_prefix=data_root + 'val/img/',
        seg_prefix=data_root + 'val/panoptic_video/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type_test,
        ann_file=dict(ins_ann=data_root + 'instances_val_city_vps_rle.json',
                      panoptic_ann=data_root + 'panoptic_gt_val_city_vps.json',
                      vps=True
                      ),
        img_prefix=data_root + 'val/img_all/',     # img for validation
        ref_prefix=data_root + 'val/img_all/',  # ref_images
        nframes_span_test=30,
        pipeline=test_pipeline))

evaluation = dict(metric=['panoptic'])