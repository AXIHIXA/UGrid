#!/bin/bash

python generate.py \
    --dataset_root "../../Downloads/SynDat_1025/train" \
    --shape "curve" \
    --image_size 1025 \
    --start_index 1 \
    --num_instances 16000
