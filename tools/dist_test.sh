#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
   $(dirname "$0")/test.py $CONFIG $(dirname $1)/$CHECKPOINT --format_only --options save_dir=$(dirname $1)/test --launcher pytorch ${@:4}
