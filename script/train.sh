#!/bin/bash

python train.py \
    --structure "unet" \
    --downsampling_policy "lerp" \
    --upsampling_policy "lerp" \
    --num_iterations 16 \
    --relative_tolerance 1e-6 \
    --initialize_x0 "random" \
    --num_mg_layers 6 \
    --num_mg_pre_smoothing 2 \
    --num_mg_post_smoothing 2 \
    --activation "none" \
    --initialize_trainable_parameters "default" \
    --optimizer "adam" \
    --scheduler "step" "50" "0.1" \
    --initial_lr 1e-3 \
    --lambda_1 1 \
    --lambda_2 1 \
    --start_epoch 0 \
    --max_epoch 1000 \
    --save_every 1 \
    --evaluate_every 1 \
    --checkpoint_root "./checkpoint" \
    --dataset_root "../../Downloads/SynDat1025" \
    --num_workers 8 \
    --batch_size 4 \
    --seed 9590589012167207234
