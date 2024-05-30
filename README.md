# EnvEnc
Repository for environment encoder.

![Stage 1 Training Procedure](https://github.com/SulRash/envenc/blob/main/assets/stage_1.png)
![Stage 2 Training Procedure](https://github.com/SulRash/envenc/blob/main/assets/stage_2.png)

The idea is to train an RL agent to play games given a multimodal model's embeddings, instead of the actual environment. I believe embeddings are a more universal representation, and if learned on it might make it easier for an agent to generalize to unseen environments.

# Running Everything

Run the following to install dependencies

```
apt-get install swig
conda env create -f conda_env.yml
pip install flash-attn --no-build-isolation
```

# Progress

- [x] Setup RL training code
- [x] Setup code to extract VLM embeddings from Gym environment frames.
- [x] Train RL agent on embeddings instead of game frames.
- [x] Improve efficiency of generating VLM embeddings through caching and batching.
- [x] Setup code to autotune hyperparameters for RL agent.
- [ ] Train autoencoder on Atari game frame VLM embeddings.
- [ ] Replace embeddings input to RL agent with autoencoder dense representation input.
- [ ] Train RL agent on autoencoder dense representation.
- [ ] Show improved performance / generalizability

Big thanks to [CleanRL](https://github.com/vwxyzjn/cleanrl) for providing the basis of the RL training code and hyperparameter tuning code, made my life a lot easier :)
