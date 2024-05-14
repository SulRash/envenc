#!/bin/bash

python main_envpool.py \
    --env_id "Pong-v5" \
    --exp_name "big_cnn_run_1" \
    --total_timesteps 10000000 \
    --num_envs 64 \
    --num_minibatches 8 \
    --num_steps 128 \
    --clip_coef 0.2 \
    --vf_coef 1 \
    --learning_rate 8e-4 \
    --network "bigcnn" --use_vlm \
    --track
