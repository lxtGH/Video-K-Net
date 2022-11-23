# Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation (CVPR-2022, oral) [Sides](./slides/Video-KNet-cvpr-slides-10-25-version.pptx) [Poster](./slides/cvpr22_poster_lxt_zww_pjm.pdf), [Video](https://www.youtube.com/watch?v=LIEyp_czu20&t=3s)

[Xiangtai Li](https://lxtgh.github.io/),
[Wenwei Zhang](https://zhangwenwei.cn/),
[Jiangmiao Pang](https://oceanpang.github.io/),
[Kai Chen](https://chenkai.site/), 
[Guangliang Cheng](https://scholar.google.com/citations?user=FToOC-wAAAAJ),
[Yunhai Tong](https://scholar.google.com/citations?user=T4gqdPkAAAAJ&hl=zh-CN),
[Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/).

We introduce Video K-Net, a simple, strong, and unified framework for fully end-to-end dense video segmentation. 

The method is built upon K-Net, a method of unifying image segmentation via a group of learnable kernels.

This project contains the training and testing code of Video K-Net for both VPS (Video Panoptic Segmentation), 
VSS(Video Semantic Segmentation), VIS(Video Instance Segmentation).

To the best of our knowledge, our Video K-Net is the first open-sourced method that supports three different video segmentation tasks (VIS, VPS, VSS) for Video Scene Understanding.


## News! Video K-Net also supports [VIP-Seg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) dataset(CVPR-2022). It also achieves the new state-of-the-art result.



### Environment and DataSet Preparation 
Our codebase is based on MMDetection and MMSegmentation. Parts of the code is borrowed from MMtracking and UniTrack.

- MIM >= 0.1.1
- MMCV-full >= v1.3.8
- MMDetection == v2.18.0
- timm
- scipy
- panopticapi

See the [DATASET.md](https://github.com/lxtGH/Video-K-Net/blob/main/DATASET.md)

knet folder contains the Video K-Net for VPS.

knet_vis folder contains the Video K-Net for VIS.



### Pretrained CKPTs and Trained Models

We provide the pretrained models for VPS and VIS.

Baidu Yun Link: [here](https://pan.baidu.com/s/12dIinkAF3o60fcAoggVhjQ)  Code:i034

The pretrained models are provided to train the Video K-Net.

The trained models are also provided for play and test.

To Do:
One Drive Link: [here]()

### [VPS] KITTI-STEP

1. First pretrain K-Net on Cityscapes-STEP datasset. As shown in original STEP paper(Appendix Part) and our own EXP results, this step is very important to improve the segmentation performance.
You can also use our trained model for verification.

Cityscape-STEP follows the format of STEP: 17 stuff classes and 2 thing classes. 

```bash
# train cityscapes step panoptic segmentation models
sh ./tools/slurm_train.sh $PARTITION knet_step configs/det/knet_cityscapes_step/knet_s3_r50_fpn.py $WORK_DIR --no-validate
```

2. Then train the Video K-Net on KITTI-STEP. We have provided the pretrained models from Cityscapes of Video K-Net.

For slurm users:

```bash
# train Video K-Net on KITTI-step using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_step configs/det/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $WORK_DIR --no-validate --load-from /path_to_knet_step_city_r50
```

```bash
# train Video K-Net on KITTI-step using Swin-base
GPUS=16 GPUS_PER_NODE=8 sh ./tools/slurm_train.sh $PARTITION video_knet_step configs/det/video_knet_kitti_step/video_knet_s3_swinb_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $WORK_DIR --no-validate --load-from /path_to_knet_step_city_r50
```

Our models are trained with two V100 machines. 

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

#### Toy Video K-Net 

As shown in the paper, we also provide toy video K-Net in knet/video/knet_quansi_dense_embed_fc_toy_exp.py. 
You use the K-Net pre-trained on image-level KITTI-STEP without tracking.


### [VIS] YouTube-VIS-2019

1. First Download the pre-trained Image K-Net instance segmentation models. All the models are pretrained on COCO which is
a common. You can also pretrain it by yourself. We also provide the config for pretraining.

For slurm users:

```bash
# train K-Net instance segmentation models on COCO using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION knet_instance configs/det/coco/knet_s3_r50_fpn_ms-3x_coco.py $WORK_DIR 
```

2. Then train the video K-Net in a clip-wised manner. 

```bash
# train Video K-Net VIS models using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_vis configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_youtubevis.py $WORK_DIR --load-from /path_to_knet_instance_coco
```

3. To evaluate the results of Video K-Net on VIS. Dump the prediction results for submission to the conda server. 

```bash
# test Video K-Net VIS models using R-50
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_vis configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_youtubevis.py $WORK_DIR 
```


### [VPS] VIP-Seg

1. First Download the pre-trained Image K-Net panoptic segmentation models. All the models are pretrained on COCO which is
a common step following VIP-Seg. You can also pretrain it by yourself. We also provide the config for pretraining.
```bash
# train K-Net on COCO Panoptic Segmetnation
GPUS=8 sh ./tools/slurm_train.sh $PARTITION knet_coco configs/det/coco/knet_s3_r50_fpn_ms-3x_coco-panoptic.py $WORK_DIR 
```

2. Train the Video K-Net on the VIP-Seg dataset. 
```bash
# train Video K-Net on VIP-Seg
GPUS=8 sh ./tools/slurm_train.sh $PARTITION video_knet_vis configs/det/video_knet_vipseg/video_knet_s3_r50_rpn_vipseg_mask_embed_link_ffn_joint_train.py $WORK_DIR --load-from /path/knet_coco_pretrained_r50
```

3. Test the Video K-Net on VIP-Seg val dataset.
```bash
# test locally on VIP-Seg
sh ./tools/dist_step_test.sh configs/det/video_knet_vipseg/video_knet_s3_r50_rpn_vipseg_mask_embed_link_ffn_joint_train.py $MODEL_DIR 
```

We also dump the colored images for debug.

```bash
# eval STEP STQ
python tools/eval_dstq_vipseg.py result_path gt_path
```

```bash
# eval STEP VPQ
python tools/eval_dvpq_vipseg.py result_path gt_path
```


## Visualization Results


### Results on KITTI-STEP DataSet



### Results on VIP-Seg DataSet



### Results on YouTube-VIS DataSet



### Short term segmentation and tracking results on Cityscapes VPS dataset.

images(left), Video K-Net(middle), Ground Truth 
![Alt Text](./figs/cityscapes_vps_video_1_20220318131729.gif)

![Alt Text](./figs/cityscapes_vps_video_2_20220318132943.gif)

### Long term segmentation and tracking results on STEP dataset.

![Alt Text](./figs/step_video_1_20220318133227.gif)

![Alt Text](./figs/step_video_2_20220318133423.gif)


## Related Project and Acknowledgement

NIPS-2021, K-Net: Unified Segmentation: Our Image baseline (https://github.com/ZwwWayne/K-Net)

ECCV-2022, PolyphonicFormer: A Unified Framework For Panoptic Segmentation + Depth Estimation (winner of ICCV-2021 BMTT workshop)
(https://github.com/HarborYuan/PolyphonicFormer)


