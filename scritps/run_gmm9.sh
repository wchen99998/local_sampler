export WANDB_API_KEY="d1f4c3ab36181e9a905a174c1cd756806ecdecc1"
# export WANDB_MODE="offline"

CUDA_VISIBLE_DEVICES=3 python main.py \
    target.gmm_n_mixes=9 \
    wandb.name="GMM9" \
    base.initial_sigma=1.5