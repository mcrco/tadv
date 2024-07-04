#!/bin/bash

PROJ_ROOT="/home/marco/dev/tadv"
SCRIPT="${PROJ_ROOT}/BEAR/benchmark/BEAR-Standard/tools/train.py"
CONFIG="${PROJ_ROOT}/custom_mm_configs/tadv_mm_config.py"
SOURCE=$1
TARGET=$2

if [ "$SOURCE" = "mpii" ]; then
    DATA_ROOT="$PROJ_ROOT/BEAR/datasets/MPII-Cooking/frames/"
    TRAIN_FILE="$PROJ_ROOT/BEAR/benchmark/BEAR-UDA/data/toyota_smarthome_mpii_cooking/mpii_cooking_da_train.csv"
fi
if [ "$SOURCE" = "tsh" ]; then
    DATA_ROOT="$PROJ_ROOT/BEAR/datasets/ToyotaSmarthome/frames/"
    TRAIN_FILE="$PROJ_ROOT/BEAR/benchmark/BEAR-UDA/data/toyota_smarthome_mpii_cooking/toyota_smarthome_da_train.csv"
fi
if [ "$TARGET" = "mpii" ]; then
    DATA_ROOT_VAL="$PROJ_ROOT/BEAR/datasets/MPII-Cooking/frames/"
    TEST_FILE="$PROJ_ROOT/BEAR/benchmark/BEAR-UDA/data/toyota_smarthome_mpii_cooking/mpii_cooking_da_test.csv"
fi
if [ "$TARGET" = "tsh" ]; then
    DATA_ROOT_VAL="$PROJ_ROOT/BEAR/datasets/ToyotaSmarthome/frames/"
    TEST_FILE="$PROJ_ROOT/BEAR/benchmark/BEAR-UDA/data/toyota_smarthome_mpii_cooking/toyota_smarthome_da_test.csv"
fi

WORK_DIR="${PROJ_ROOT}/out/tadv/$SOURCE-$TARGET"


export PYTHONPATH="$(dirname $SCRIPT)/..":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$3

python $SCRIPT $CONFIG \
    --validate \
    --cfg-options data.videos_per_gpu=10 \
    data_root=$DATA_ROOT \
    data.train.data_prefix=$DATA_ROOT \
    ann_file_train=$TRAIN_FILE \
    data.train.ann_file=$TRAIN_FILE \
    data_root_val=$DATA_ROOT_VAL \
    data.test.data_prefix=$DATA_ROOT_VAL \
    ann_file_test=$TEST_FILE \
    data.test.ann_file=$TEST_FILE \
	work_dir=$WORK_DIR \
