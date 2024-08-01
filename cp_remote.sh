#!/bin/sh

export TERM=xterm-256color
rsync -a --info=progress2 --exclude='BEAR/' --exclude='out/' --exclude='.git/' --exclude='data/hmdb-small/' \
    --exclude='data/hmdb51/avi' --exclude='data/hmdb51/frames' --exclude='checkpoints/' -e \
    'ssh -p 51797' ./ \
    root@47.186.63.233:/tadv/
