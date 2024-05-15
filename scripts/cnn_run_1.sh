python main_envpool.py \
    --env_id "Pong-v5" \
    --exp_name "big_cnn_run_1" \
    --total_timesteps 2000000 \
    --num_envs 64 \
    --learning_rate 2.5e-3 \
    --network "cnn" --use_vlm \
    --track