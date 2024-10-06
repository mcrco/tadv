#!/bin/sh

export TERM=xterm-256color
rsync -a --info=progress2 --exclude='BEAR/' --exclude='out/' --exclude='.git/' --exclude='data/' \
    --exclude='checkpoints/' -e \
    'ssh -p 31967' ./ \
    root@79.116.69.38:/tadv/
