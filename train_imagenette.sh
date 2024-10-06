#!/bin/sh

CUDA_VISIBLE_DEVICES=$2 python3 train_video_imagenette.py --exp_name $1 --check_val_every_n_epoch 5 --max_epochs 200 --cls_head neehar --freeze_backbone 1 --log_ca --text_conditioning class_names --batch_size 32 --diffusion_batch_size 8 #--debug
