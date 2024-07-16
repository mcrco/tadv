#!/bin/sh

CUDA_VISIBLE_DEVICES=$2 python train_video.py --blip_caption_path captions/hmdb51_captions.json --exp_name $1 --check_val_every_n_epoch 5 --max_epochs 50 --cls_head rogerio #--debug
