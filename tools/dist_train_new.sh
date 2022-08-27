#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-$((29500 + $RANDOM % 29))}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run  --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_new.py $CONFIG --launcher pytorch ${@:3}
