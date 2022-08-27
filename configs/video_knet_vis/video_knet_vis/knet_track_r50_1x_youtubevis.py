_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_track_r50.py',
    '../_base_/datasets/youtubevis_2019.py',
]

_load_from = '/home/lxt/pretrained_models/video_knet/video_knet_vis_r50_3x.pth'

model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint=_load_from,
        map_location='cpu',
        prefix=None,
    ),
)
