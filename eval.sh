#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

python tools/test.py $CONFIG $(dirname $1)/$CHECKPOINT --eval \
--options save_dir=$(dirname $1)/test
