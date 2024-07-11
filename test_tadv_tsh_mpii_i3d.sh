#!/bin/bash

PROJ_ROOT="/home/marco/dev/tadv"
SCRIPT="${PROJ_ROOT}/BEAR/benchmark/BEAR-Standard/tools/test.py"

SOURCE=$1
TARGET=$2
TIME=$3

CONFIG="${PROJ_ROOT}/out/tadv/$SOURCE-$TARGET/$TIME/tadv_mm_config.py"
CHKPT="${PROJ_ROOT}/out/tadv/$SOURCE-$TARGET/$TIME/latest.pth"

export PYTHONPATH="$(dirname $SCRIPT)/..":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$4

python $SCRIPT $CONFIG $CHKPT --eval top_k_accuracy
