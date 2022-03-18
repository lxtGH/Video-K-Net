# Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation (CVPR-2022)


[Xiangtai Li](https://lxtgh.github.io/),
[Wenwei Zhang](https://zhangwenwei.cn/),
[Jiangmiao Pang](https://oceanpang.github.io/),
[Kai Chen](https://chenkai.site/), 
[Guangliang Cheng](https://scholar.google.com/citations?user=FToOC-wAAAAJ),
[Yunhai Tong](https://eecs.pku.edu.cn/info/1475/9689.htm),
[Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/).

We introduce Video K-Net, a simple, strong, and unified framework for fully end-to-end video panoptic segmentation. The method is built upon K-Net, a method of unifying image segmentation via a group of learnable kernels.


This project will contain the training and test code of Video K-Net for both VPS(Video Panoptic Segmetnation), VSS(Video Semantic Segmentation), VIS(Video Instance Segmentation).
Since this project is quit complex than most segmentation repo, I will try my best to make it easy to use.
Moreover, different datasets may contain the different training/test settings with different pre-trained model and different hypre-parameters.
Please stay tuned!



## Demo

### Short term segmentation tracking results on STEP dataset

images(left), Video K-Net(middle), Ground Truth 
![Alt Text](./figs/cityscapes_vps_video_1_20220318131729.gif)

![Alt Text](./figs/cityscapes_vps_video_2_20220318132943.gif)

### Long term segmentation and tracking results on Cityscapes VPS dataset.

![Alt Text](./figs/step_video_1_20220318133227.gif)

![Alt Text](./figs/step_video_2_20220318133423.gif)





## Related Project

K-Net: Unified Segmentation (https://github.com/ZwwWayne/K-Net)


PolyphonicFormer: Unified Panoptic Segmentation + Depth Estimation (winner of ICCV-2021 BMTT workshop)
(https://github.com/HarborYuan/PolyphonicFormer)
