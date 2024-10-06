#!/bin/sh

export TERM=xterm-256color
rsync -a --info=progress2 --exclude='BEAR/' --exclude='out/' --exclude='.git/' --exclude='data/hmdb-small/' \
    --exclude='data/hmdb51/avi' --exclude='data/hmdb51/frames' \
    --exclude='data/hmdb51/videos' --exclude='data/imagenette2/' --exclude='checkpoints/' \
    --exclude='wandb/' --exclude='ca_test'  -e \
    'ssh -p 42178' ./ \
    root@36.225.131.3:/tadv/
