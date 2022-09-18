# Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation (CVPR-2022, oral)


[Xiangtai Li](https://lxtgh.github.io/),
[Wenwei Zhang](https://zhangwenwei.cn/),
[Jiangmiao Pang](https://oceanpang.github.io/),
[Kai Chen](https://chenkai.site/), 
[Guangliang Cheng](https://scholar.google.com/citations?user=FToOC-wAAAAJ),
[Yunhai Tong](https://scholar.google.com/citations?user=T4gqdPkAAAAJ&hl=zh-CN),
[Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/).

We introduce Video K-Net, a simple, strong, and unified framework for fully end-to-end video panoptic segmentation. The method is built upon K-Net, a method of unifying image segmentation via a group of learnable kernels.

This project will contain the training and test code of Video K-Net for both VPS(Video Panoptic Segmetnation), VSS(Video Semantic Segmentation), VIS(Video Instance Segmentation).


### Environment and DataSet Preparation 

- MIM >= 0.1.1
- MMCV-full >= v1.3.8
- MMDetection == v2.18.0
- timm
- scipy
- panopticapi


See the DATASET.md


### [VPS] KITTI-STEP

1. First pretrain K-Net on Cityscapes datasset.

```bash
# train cityscapes step panoptic segmentation models
sh ./tools/slurm_train.sh $PARTITION knet_step configs/det/knet_cityscapes_step/knet_s3_r50_fpn.py $WORK_DIR --no-validate
```

2. Then train the Video K-Net on KITTI-STEP. We have provided the pretrained models from Cityscapes of Video K-Net.

For slurm users:

```bash
# train Video K-Net on KITTI-step using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_step configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $WORK_DIR --no-validate
```

```bash
# train Video K-Net on KITTI-step using Swin-base
GPUS=16 GPUS_PER_NODE=8 sh ./tools/slurm_train.sh $PARTITION video_knet_step configs/det/video_knet_kitti_step/video_knet_s3_swinb_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $WORK_DIR --no-validate
```

For Local machine:

```bash
# train Video K-Net on KITTI-step with 8 GPUs
sh ./tools/dist_train.sh video_knet_step configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py 8 $WORK_DIR --no-validate
```


3. Testing and Demo.

We provide both VPQ and STQ metrics to evaluate VPS models. 

```bash
# test locally 
sh ./tools/dist_step_test.sh configs/det/knet_cityscapes_ste/knet_s3_r50_fpn.py $MODEL_DIR 
```

We also dump the colored images for debug.

```bash
# eval STEP STQ
python tools/eval_dstq_step.py result_path gt_path
```

```bash
# eval STEP VPQ
python tools/eval_dvpq_step.py result_path gt_path
```

### [VIS] YouTube-VIS-2019

1. First Download the pre-trained Image K-Net instance segmentation models. All the models are pretrained on COCO which is
a common. You can also pre-train it by yourself.

For slurm users:

```bash
# train K-Net instance segmentation models on COCO using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION knet_instance configs/det/coco/knet_s3_r50_fpn_ms-3x_coco.py $WORK_DIR 
```

2. Then train the video K-Net in a clip-wised manner. 


```bash
# train Video K-Net VIS models using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_vis configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_youtubevis.py $WORK_DIR 
```

3. To evaluate the results of Video K-Net on VIS. Dump the prediction results for submission to the server. 



### [VSS] VSPW

To be released soon since we are planning to release a stronger baseline.



## Visualization Results

### Short term segmentation and tracking results on Cityscapes VPS dataset.

images(left), Video K-Net(middle), Ground Truth 
![Alt Text](./figs/cityscapes_vps_video_1_20220318131729.gif)

![Alt Text](./figs/cityscapes_vps_video_2_20220318132943.gif)

### Long term segmentation and tracking results on STEP dataset.

![Alt Text](./figs/step_video_1_20220318133227.gif)

![Alt Text](./figs/step_video_2_20220318133423.gif)

## Related Project

K-Net: Unified Segmentation: Our Image baseline (https://github.com/ZwwWayne/K-Net)

PolyphonicFormer: A Unified Framework For Panoptic Segmentation + Depth Estimation (winner of ICCV-2021 BMTT workshop)
(https://github.com/HarborYuan/PolyphonicFormer)


