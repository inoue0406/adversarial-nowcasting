#!/bin/bash

lightweight_gan \
  --data ../data/data_alljapan_image/ \
  --name run_lwgan_tstrun_201208 \
  --batch-size 16 \
  --gradient-accumulate-every 4 \
  --num-train-steps 200000
