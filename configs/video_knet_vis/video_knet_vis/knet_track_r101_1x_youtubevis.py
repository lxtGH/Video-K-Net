_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_track_r50.py',
    '../_base_/datasets/youtubevis_2019.py',
]


model = dict(

    backbone=dict(
        type='ResNet',
        depth=101,),

    init_cfg=dict(
        type='Pretrained',
        checkpoint=_load_from,
        map_location='cpu',
        prefix=None,
    ),
)
