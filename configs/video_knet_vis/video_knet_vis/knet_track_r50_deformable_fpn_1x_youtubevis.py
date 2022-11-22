_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_track_r50_deformablefpn.py',
    '../_base_/datasets/youtubevis_2019.py',
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,)
