
#!/bin/sh
export TERM=xterm-256color
rsync -a --info=progress2  -e \
    'ssh -p 51379' ./TADP/tadv_heads.py \
    root@47.186.63.233:/tadv/TADP/tadv_heads.py
