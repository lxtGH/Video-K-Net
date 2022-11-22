num_stages = 3
num_proposals = 100
conv_kernel_size = 1

model = dict(
    type='KNet',
    cityscapes=False,
    kitti_step=True,
    num_thing_classes=2,
    num_stuff_classes=17,
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
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4
    ),
    rpn_head=dict(
        type='ConvKernelHead',
        num_classes=19,
        num_thing_classes=2,
        num_stuff_classes=17,
        cat_stuff_mask=True,
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
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        num_proposals=num_proposals,
        proposal_feats_with_obj=True,
        xavier_init_kernel=False,
        kernel_init_std=1,
        num_cls_fcs=1,
        in_channels=256,
        feat_transform_cfg=None,
        loss_rank=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.1),
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
        type='KernelIterHead',
        num_thing_classes=2,
        num_stuff_classes=17,
        do_panoptic=True,
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        mask_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=19,
                num_thing_classes=2,
                num_stuff_classes=17,
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
                    conv_cfg=dict(type='Conv2d'),
                    act_cfg=None
                ),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_rank=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1
                ),
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0
                ),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=4.0
                ),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0))
            for _ in range(num_stages)
        ]
    ),
    # training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1)

            for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=num_proposals,
            mask_thr=0.5,
            stuff_score_thr=0.05,
            merge_stuff_thing=dict(
                overlap_thr=0.6,
                iou_thr=0.5,
                stuff_max_area=4096,
                instance_score_thr=0.25
            )
        )
    )
)

custom_imports = dict(
    imports=[
        'knet.det.kernel_head',
        'knet.det.kernel_iter_head',
        'knet.det.kernel_update_head',
        'knet.det.semantic_fpn_wrapper',
        'knet.det.dice_loss',
        'knet.det.mask_hungarian_assigner',
        'knet.det.mask_pseudo_sampler',
        'knet.kernel_updator',
        'knet.cross_entropy_loss',
        'swin.swin_transformer',
        'swin.mix_transformer',
        'swin.DetectRS',
        'swin.swin_transformer_rfp',
        'external.cityscapes_step',
        'external.kitti_step_dvps',
        'external.dataset.dvps_pipelines.transforms',
        'external.dataset.dvps_pipelines.loading',
        'external.dataset.dvps_pipelines.tricks',
        'external.dataset.pipelines.formatting',
        # 'knet.video.knet_track',
        # 'knet.video.knet_track_head',
        'knet.video.track_heads',
        'knet.video.kernel_head',
        'knet.video.kernel_iter_head',
        'knet.video.kernel_update_head',
        'knet.video.knet_uni_track',
        'knet.video.knet_quansi_dense',
        # 'knet.video.knet_quansi_dense_roi',
        'knet.video.knet_quansi_dense_roi_gt_box',
        'knet.video.knet_quansi_dense_embed_fc',
        'knet.video.knet_quansi_dense_embed_fc_joint_train',
        # 'knet.video.knet_quansi_dense_embed_fc_with_appearance',
        'knet.video.knet_quansi_dense_roi_gt_box_joint_train',
        # 'knet.video.knet_quansi_dense_embed_fc_toy_exp',
        'knet.video.qdtrack.losses.l2_loss',
        'knet.video.qdtrack.losses.multipos_cross_entropy_loss',
        'knet.video.qdtrack.trackers.quasi_dense_embed_tracker',
    ],
    allow_failed_imports=False
)
