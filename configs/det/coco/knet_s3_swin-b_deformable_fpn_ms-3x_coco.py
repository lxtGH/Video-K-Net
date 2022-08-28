_base_ = [
    '../_base_/models/knet_s3_r50_deformable_fpn.py',
    '../common/mstrain_3x_coco_instance.py'
]

model = dict(
    pretrained='/mnt/lustre/lixiangtai/pretrained/swin/swin_base_patch4_window7_224_22k.pth',
    backbone=dict(
        _delete_=True,
        type='SwinTransformerDIY',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False),
    neck=dict(in_channels=[128, 256, 512, 1024])
)
