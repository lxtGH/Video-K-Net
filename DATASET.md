Please prepare the data structure as the following instruction:

The final dataset folder should be like this. 
```
root 
├── data
│   ├──  kitti-step
│   ├──  coco
│   ├──  VIPSeg
│   ├──  youtube_vis_2019
│   ├──  cityscapes
```

### [VPS] KITTI-STEP

Download the KITTI-STEP from the official website. 

Then run the scripts in scripts/kitti_step_prepare.py.
You will get such format.
You can get the our pre-process format in https://huggingface.co/LXT/VideoK-Net/tree/main

```
├── kitti-step
│   ├──  video_sequence
│   │   ├── train
            ├──00018_000331_leftImg8bit.png
            ├──000018_000331_panoptic.png
            ├──****
│   │   ├── val
│   │   ├── test 
```


### [VPS] VIPSeg

Download the origin dataset from the official repo.\
Following official repo, we use resized videos for training and evaluation (The short size of the input is set to 720 while the ratio is keeped).

```
├── VIPSeg
│   ├──  images
│   │   ├── 1241_qYvEuwrSiXc
        │      ├──*.jpg
│   ├──  panomasks 
│   │   ├── 1241_qYvEuwrSiXc
        │      ├──*.png
│   ├──  panomasksRGB 
```


### [VIS] Youtube-VIS-2019
We use pre-processed json file according to mmtracking codebase.
see the "tools/dataset/youtubevis2coco.py"

```
├── youtube_vis_2019
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
│   │   ├── youtube_vis_2019_train.json
│   │   ├── youtube_vis_2019_valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video floders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video floders
```


### [VSS] VSPW

To do


### [VPS] Cityscapes 

For Cityscape-VPS and Cityscape-DVPS, we suggest the follower to see
The model of Video K-Net will not be released due to the Patent ISSUE and INTERNAL USEAGE. 

You can find our related works. ECCV-2022, PolyphonicFormer: A Unified Framework For Panoptic Segmentation + Depth Estimation (winner of ICCV-2021 BMTT workshop)
(https://github.com/HarborYuan/PolyphonicFormer)



## Image DataSet For Pretraining K-Net

### COCO dataset

COCO is most common datatsets. It contains 80 thing classes and 54 stuff classes.

The dataset format is the same as origin [Detectron2](https://github.com/facebookresearch/detectron2)
including panoptic segmentation preparation [scirpts](https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py).

Then the final folder is like this:
```
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   ├── train2017
│   ├── val2017
│   ├── panoptic_{train,val}2017/  # png annotations
```

### Cityscapes dataset

Cityscapes dataset is a high-resolution road-scene dataset which contains 19 classes. 
(8 thing classes and 11 stuff classes). 2975 images for training, 500 images for validation and 1525 images for testing.

Preparing cityscape dataset has three steps:

1, Convert segmentation id map(origin label id maps) to trainId maps (id ranges: 0-18 for training) using 
the official scripts [repo](https://github.com/mcordts/cityscapesScripts)

2, The run python dataset/prepare_cityscapes.py to generate the COCO-like annotations. 
This annotations can be used for Instance Segmentation training.

using csCreateTrainIdLabelImgs.py

and put the instancesonly_filtered_gtFine_train.json into annotations folder


3, For Panoptic Segmenation dataset, to generate the json file 

using csCreatePanopticImgs.py 

or you can download the our transformed .json and .png files via link: () and put the 
json file into annotations folder. 

Then the final folder is like this:

```
├── cityscapes
│   ├── annotations
│   │   ├── instancesonly_filtered_gtFine_train.json # coco instance annotation file(COCO format)
│   │   ├── instancesonly_filtered_gtFine_val.json
│   │   ├── cityscapes_panoptic_train.json  # panoptic json file 
│   │   ├── cityscapes_panoptic_val.json  
│   ├── leftImg8bit
│   ├── gtFine
│   │   ├──cityscapes_panoptic_{train,val}/  # png annotations
│   │   
```
