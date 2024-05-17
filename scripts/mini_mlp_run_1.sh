python main_envpool.py \
    --env_id "Pong-v5" \
    --exp_name "mini_mlp_run1.sh" \
    --total_timesteps 1000000 \
    --num_envs 64 \
    --learning_rate 4e-4 \
    --network "mlp" \
    --track