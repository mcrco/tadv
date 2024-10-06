#!/bin/sh

CUDA_VISIBLE_DEVICES=$2 python3 train_video.py --text_conditioning class_names --exp_name $1 --check_val_every_n_epoch 5 --max_epochs 500 --cls_head rogerio --freeze_backbone 1 --log_ca --batch_size 16 --diffusion_batch_size 8 #--use_only_attn #--debug #--strategy ddp_find_unused_parameters_true
