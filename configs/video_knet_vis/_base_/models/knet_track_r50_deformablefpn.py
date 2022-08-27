num_stages = 3
num_proposals = 100
conv_kernel_size = 1
model = dict(
    type='KNetTrack',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='MSDeformAttnPixelDecoder',
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='ReLU'),
        return_one_list=True,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        init_cfg=None),
    rpn_head=dict(
        type='ConvKernelHeadVideo',
        conv_kernel_size=conv_kernel_size,
        feat_downsample_stride=2,
        feat_refine_stride=1,
        feat_refine=False,
        use_binary=True,
        num_loc_convs=1,
        num_seg_convs=1,
        conv_normal_init=True,
        localization_fpn=dict(
            type='SemanticFPNWrapper',
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            upsample_times=2,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            cat_coors=False,
            cat_coors_level=3,
            fuse_by_cat=False,
            return_list=False,
            num_aux_convs=1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        ),
        num_proposals=num_proposals,
        proposal_feats_with_obj=True,
        xavier_init_kernel=False,
        kernel_init_std=1,
        num_cls_fcs=1,
        in_channels=256,
        num_classes=40,
        feat_transform_cfg=None,
        loss_seg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=4.0)),
    roi_head=dict(
        type='KernelIterHeadVideo',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        num_thing_classes=40,
        num_stuff_classes=0,
        mask_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=40,
                num_thing_classes=40,
                num_stuff_classes=0,
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
                mask_upsample_stride=2,
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
                    loss_weight=2.0)
            ) for _ in range(num_stages)
        ]),
    tracker=dict(
        type="KernelFrameIterHeadVideo",
        num_proposals=num_proposals,
        num_stages=3,
        assign_stages=2,
        proposal_feature_channel=256,
        stage_loss_weights=(1., 1., 1.),
        num_thing_classes=40,
        num_stuff_classes=0,
        mask_head=dict(
            type='KernelUpdateHeadVideo',
            num_proposals=num_proposals,
            num_classes=40,
            num_thing_classes=40,
            num_stuff_classes=0,
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
            mask_upsample_stride=2,
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
                loss_weight=2.0)
        ),

    ),
    # training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)
            ),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)
                ),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ],
        tracker=dict(
            assigner=dict(
                type='MaskHungarianAssignerVideo',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0,
                               pred_act=True)
            ),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1)
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=10,
            mask_thr=0.5,
            merge_stuff_thing=dict(
                iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3
            )
        ),
        tracker=dict(
            max_per_img=10,
            mask_thr=0.5,
            merge_stuff_thing=dict(
                iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3
            ),
        ),
    )
)

custom_imports = dict(
    imports=[
        'knet_vis.det.knet',
        'knet_vis.det.kernel_head',
        'knet_vis.det.kernel_iter_head',
        'knet_vis.det.kernel_update_head',
        'knet_vis.det.semantic_fpn_wrapper',
        'knet_vis.kernel_updator',
        'knet.det.msdeformattn_decoder',
        'knet_vis.det.mask_hungarian_assigner',
        'knet_vis.det.mask_pseudo_sampler',
        'knet_vis.tracker.track',
        'knet_vis.tracker.kernel_head',
        'knet_vis.tracker.kernel_iter_head',
        'knet_vis.tracker.kernel_frame_iter_head',
        'knet_vis.tracker.mask_hungarian_assigner',
        'knet_vis.tracker.kernel_update_head',
        'swin.swin_transformer',
        'mmtrack.datasets.youtube_vis_dataset',
        'mmtrack.pipelines',
    ],
    allow_failed_imports=False
)
