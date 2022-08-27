_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_track_r50_deformablefpn.py',
    '../_base_/datasets/youtubevis_2019.py',
]
_load_from ="/mnt/lustre/lixiangtai/project/Knet/work_dirs/coco/knet_r50_deformable_fpn_3x_instance_seg/latest.pth"

model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint=_load_from,
        map_location='cpu',
        prefix=None,
    ),
)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,)
