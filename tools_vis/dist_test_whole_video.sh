#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-$((29500 + $RANDOM % 29))}

if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_whole_video.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_whole_video.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
fi
