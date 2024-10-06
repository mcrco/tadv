#!/bin/sh
export TERM=xterm-256color
rsync -a --info=progress2  -e \
    'ssh -p 41196' ./TADP/ \
    root@36.225.131.3:/tadv/TADP/
