#!/bin/sh

export TERM=xterm-256color
rsync -a --info=progress2 --exclude='BEAR/' --exclude='out/' --exclude='.git/' --exclude='data/hmdb-small/avi' \
    --exclude='data/hmdb51/avi' --exclude='data/hmdb-small/frames' --exclude='data/hmdb51/frames' --exclude='checkpoints/' -e \
    'ssh -p 10045' ./ \
    root@69.55.141.149:/tadv/
