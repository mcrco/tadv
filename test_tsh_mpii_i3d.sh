#!/bin/bash

PROJ_ROOT="/home/marco/dev/diffusion_ar"
SCRIPT="${PROJ_ROOT}/BEAR/benchmark/BEAR-Standard/tools/test.py"

SOURCE=$1
TARGET=$2

CONFIG="${PROJ_ROOT}/out/$SOURCE-$TARGET/i3d_convnext_video_3d_8x8x1_50e.py"
CHKPT="${PROJ_ROOT}/out/$SOURCE-$TARGET/latest.pth"

export PYTHONPATH="$(dirname $SCRIPT)/..":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$3

python $SCRIPT $CONFIG $CHKPT --eval top_k_accuracy
