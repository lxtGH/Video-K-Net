_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_kitti_step_s3_r50_fpn.py',
    '../_base_/datasets/kitti_step_vps.py',
]

load_from = None

num_stages = 3
conv_kernel_size = 1
num_thing_classes = 2
num_stuff_classes = 17
num_classes = num_thing_classes + num_stuff_classes

model = dict(
    type="VideoKNetQuansiEmbedFCJointTrain",
    cityscapes=False,
    kitti_step=True,
    link_previous=True,
    mask_assign_stride=2,
    num_thing_classes=num_thing_classes,
    num_stuff_classes=num_stuff_classes,
    ignore_label=255,
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
        with_cp=False
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    rpn_head=dict(
        loss_seg=dict(
                _delete_=True,
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
        feat_downsample_stride=4,
    ),
    # add video_knet_vis roi head
    track_head=dict(
        type='QuasiDenseMaskEmbedHeadGTMask',
        num_convs=0,
        num_fcs=2,
        roi_feat_size=1,
        in_channels=256,
        fc_out_channels=256,
        embed_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
        loss_track_aux=dict(
            type='L2Loss',
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            loss_weight=1.0),
    ),
    # add tracker config
    tracker=dict(
        type='QuasiDenseEmbedTracker',
        init_score_thr=0.35,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=5,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'
    ),
    # roi head
    roi_head=dict(
        type='VideoKernelIterHead',
        num_stages=num_stages,
        num_thing_classes=2,
        num_stuff_classes=17,
        with_track=True,
        merge_joint=True,
        mask_head=[
            dict(
                type='VideoKernelUpdateHead',
                num_classes=num_classes,
                previous='placeholder',
                previous_link="update_dynamic_cov",
                previous_type="update",
                num_thing_classes=num_thing_classes,
                num_stuff_classes=num_stuff_classes,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=4,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
            ) for _ in range(num_stages)
        ]
    ),
    track_train_cfg=dict(
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
            mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
        sampler=dict(type='MaskPseudoSampler'),),
    bbox_roi_extractor=None
)



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

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
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_instance_ids',]),
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
                 keys=['img', 'img_id', 'seq_id'],
                 meta_keys=[
                     'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                     'flip_direction', 'img_norm_cfg', 'ori_filename'
                 ]),
        ])
]

runner = dict(type='EpochBasedRunner', max_epochs=12)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[9, 11])

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            split='train',
            ref_seq_index=[-2, -1, 1, 2],
            test_mode=False,
            pipeline=train_pipeline
        )),
    test=dict(
        ref_seq_index=None,
        test_mode=True,
        pipeline=test_pipeline,
        split='val',
    )
)

find_unused_parameters=True