#!/bin/bash

DATALOC=${DATALOC:-`realpath ../datasets`}
LOGLOC=${LOGLOC:-`realpath ../logger`}
IMG=${IMG:-"harbory/openmmlab:eccv-2022"}

docker run --gpus all -it --rm --ipc=host --net=host \
  --mount src=$(pwd),target=/data,type=bind \
  --mount src=$DATALOC,target=/data/data,type=bind \
  --mount src=$LOGLOC,target=/data/logger,type=bind \
  $IMG
