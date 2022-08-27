#!/bin/bash

DATALOC=${DATALOC:-~/datasets}
LOGLOC=${LOGLOC:-~/logger}
IMG=${IMG:-"harbory/openmmlab:latest"}

docker run --gpus all -it --rm --ipc=host --net=host -v $(pwd):/data -v $DATALOC:/data/data -v $LOGLOC:/data/logger $IMG
