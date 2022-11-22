_base_ = [
    '../_base_/schedules/schedule_8e.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_track_r50.py',
    '../_base_/datasets/youtubevis_2019.py',
]

model = dict(
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
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)