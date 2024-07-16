#!/bin/sh

export TERM=xterm-256color
rsync -a --info=progress2 --exclude='BEAR/' --exclude='out/' --exclude='.git/' --exclude='data/hmdb-small/' --exclude='data/hmdb51/avi' --exclude='data/hmdb51/frames' -e 'ssh -p 40163' ./ root@71.36.176.34:/tadv/
