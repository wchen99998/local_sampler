export WANDB_API_KEY="d1f4c3ab36181e9a905a174c1cd756806ecdecc1"
# export WANDB_MODE="offline"

CUDA_VISIBLE_DEVICES=3 python main.py \
    target.gmm_n_mixes=9 \
    wandb.name="GMM9" \
    base.initial_sigma=1.5 \
    training.timesteps_for_loss=16 \
    training.num_timesteps=16 \
    optimiser.learning_rate=1e-4