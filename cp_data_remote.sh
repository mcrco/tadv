#!/bin/sh

export TERM=xterm-256color
rsync -a --info=progress2  -e \
    'ssh -p 14937' ./data/hmdb51/ \
    root@184.145.31.176:/tadv/data/hmdb51
