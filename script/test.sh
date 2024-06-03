#!/bin/bash


python test.py \
    --checkpoint_root "./var/checkpoint" \
    --load_experiment "22" \
    --load_epoch 300 \
    --dataset_root "../../Downloads/SynDat1025" \
    --num_workers 8 \
    --batch_size 1 \
    --seed 9590589012167207234
