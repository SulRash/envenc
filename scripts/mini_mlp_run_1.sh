python main_envpool.py \
    --env_id "Pong-v5" \
    --exp_name "mini_mlp_run1.sh" \
    --total_timesteps 2000000 \
    --num_envs 64 \
    --learning_rate 1.2e-3 \
    --network "mlp" --use_vlm \
    #--track