import hydra
from omegaconf import DictConfig

import torch

from utils import seed_everything
from target_dists.mogs import GMM, MultivariateGaussian
from model import TimeVelocityField

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = GMM(
        dim=cfg.target.input_dim, 
        n_mixes=cfg.target.gmm_n_mixes, 
        loc_scaling=40, 
        log_var_scaling=1,
        seed=cfg.seed,
        device=device,
    )
    base = MultivariateGaussian(
        dim=cfg.target.input_dim,
        sigma=cfg.base.initial_sigma,
        device=device,
    )
    v_theta = TimeVelocityField(
        input_dim=cfg.target.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
    ).to(device)
    optimizer = torch.optim.Adam(
        v_theta.parameters(), lr=cfg.optimiser.learning_rate
    )

    num_timesteps = cfg.training.num_timesteps
    num_epochs = cfg.training.num_epochs
    batches_per_epoch = cfg.training.batches_per_epoch
    timesteps_for_loss = cfg.training.timesteps_for_loss
    batch_size = cfg.training.batch_size
    eps = 1e-5
    timesteps = torch.linspace(eps, 1.0, num_timesteps).to(device)
    dt = (1 - eps) / num_timesteps
    delta_t = 0.01
    for epoch in range(num_epochs):

        total_epoch_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            v_theta.train()

            loss = torch.tensor(0.0, device=device)

            # We select only #timesteps_for_loss timesteps randomly for loss calculation to fit in memory
            is_selected_timestep = torch.zeros(
                num_timesteps, dtype=torch.bool
            ).scatter_(0, torch.randperm(num_timesteps)[:timesteps_for_loss], True)

            x_t = base.sample((batch_size * 2,)).to(device) # half for training; half for importance weight
            for i in range(num_timesteps):
                t = timesteps[i].expand(x_t.shape[0], 1)    # (B*2, 1)
                with torch.no_grad():
                    v_t = v_theta(x_t, t)

                if timesteps[i] <= 1 - delta_t and is_selected_timestep[i]:
                    # Compute loss only for selected timesteps
                    t_ = torch.rand((batch_size, 1, 1), device=device) # (B, 1, 1)
                    z_ts = x_t[:batch_size, :].unsqueeze(1)  # training samples (B, 1, D)
                    z_te = x_t[batch_size:, :].unsqueeze(0)  # importance weight samples (1, B, D)
                    z_t = (1 - t_) * z_ts + t_ * z_te  # (B, B, D)

                    t_ = t_ * delta_t + t[:t_.size(0), :]  # (B, 1, 1)
                    t_ = t_.expand(batch_size, batch_size, 1)  # (B, B, 1)
                    v_t = v_theta(z_t.reshape(-1, cfg.target.input_dim), t_.reshape(-1, 1))  # (B*B, D)
                    v_t = v_t.reshape(batch_size, batch_size, cfg.target.input_dim)  # (B, B, D)
                    loss = ((v_t - (z_te - z_ts))**2).sum(dim=-1)

                    weight = None
                    print(loss.shape)
                    exit()

                x_t = x_t + v_t * dt



@hydra.main(config_path="./configs", config_name="gmm.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    run(cfg)

if __name__ == "__main__":
    main()