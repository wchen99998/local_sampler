import os
import hydra
import wandb
import tempfile
from omegaconf import DictConfig
import matplotlib.pyplot as plt

import torch

from utils import seed_everything
from target_dists.mogs import GMM9, GMM40, MultivariateGaussian, plot_MoG
from model import TimeVelocityField

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.wandb.name == "GMM40":
        target = GMM40(
            dim=cfg.target.input_dim, 
            n_mixes=cfg.target.gmm_n_mixes, 
            loc_scaling=40, 
            log_var_scaling=1,
            seed=cfg.seed,
            device=device,
        )  
    elif cfg.wandb.name == "GMM9":
        target = GMM9(
            dim=cfg.target.input_dim, 
            scale=3, 
            std=.35,
            device=device,
        )
    else:
        raise NotImplementedError
    
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


    wandb.init(
        project=cfg.wandb.project,
        # config=cfg,
        name=f'{cfg.wandb.name}_isfm',
        # mode="online",
    )

    def diff_log_density(x):
        # log p_1 - log p_0
        return target.log_prob(x) - base.log_prob(x)

    num_timesteps = cfg.training.num_timesteps
    num_epochs = cfg.training.num_epochs
    batches_per_epoch = cfg.training.batches_per_epoch
    timesteps_for_loss = cfg.training.timesteps_for_loss
    batch_size = cfg.training.batch_size
    eps = 1e-5
    for epoch in range(num_epochs):

        total_epoch_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            v_theta.train()

            x_1 = base.sample((batch_size * 2,)).to(device) # half for training; half for importance weight
            t = torch.rand(batch_size, 1).to(device)    # (B, 1)

            z_ts = x_1[:batch_size, :]  # training samples (B, D)
            z_te = x_1[batch_size:, :]  # importance weight samples (B, D)
            z_t = (1 - t) * z_ts + t * z_te  # (B, D)
            
            weights = diff_log_density(z_te).softmax(dim=-1)
            loss = ((v_theta(z_t, t) - (z_te - z_ts))**2).mean(dim=-1)
            loss = (loss * weights).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            wandb.log({f"train/loss": loss.item()})
            # print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{batches_per_epoch}], Loss: {loss.item():.4f}")

        v_theta.eval()
        avg_loss = total_epoch_loss / batches_per_epoch
        wandb.log({f"train/avg_epoch_loss": avg_loss})
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {avg_loss:.4f}")

        if epoch % 50 == 0:
            initial_samples = base.sample((2048,)).to(device)
            samples = generate_samples_with_euler_method(v_theta, initial_samples, ts=torch.linspace(0, 1, cfg.sampling.num_timesteps)).detach().cpu()  # (N, D)
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                fig = plot_MoG(
                        log_prob_function=target.log_prob,
                        samples=samples, 
                        name=cfg.wandb.name,
                    )
                fig.savefig(os.path.join(tmpdir, "samples.png"))
                wandb.log({f"samples": wandb.Image(
                    os.path.join(tmpdir, "samples.png")
                )})
                # wandb.log({f"samples": wandb.Image(fig)})
                plt.close(fig)

def generate_samples_with_euler_method(v_theta, initial_samples, ts):
    """
    Generate samples using the Euler method.
        t = 0 -> t = 1 : noise -> data
    """
    device = initial_samples.device
    samples = initial_samples
    t_prev = ts[:-1]
    t_next = ts[1:]

    for t_p, t_n in zip(t_prev, t_next):
        t = torch.ones(samples.size(0), device=device).unsqueeze(1) * t_p
        with torch.no_grad():
            samples = samples + v_theta(samples, t) * (t_n - t_p)
 
    return samples

@hydra.main(config_path="./configs", config_name="gmm.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    run(cfg)

if __name__ == "__main__":
    main()