import os
import tempfile

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
import flax.nnx as nnx
from jax import random as jrandom
from omegaconf import DictConfig

from model import TimeVelocityField
from target_dists.mogs import GMM40, GMM9, MultivariateGaussian, plot_MoG
from utils import seed_everything


def run(cfg):
    key = seed_everything(cfg.seed)

    if cfg.wandb.name == "GMM40":
        target = GMM40(
            dim=cfg.target.input_dim,
            n_mixes=cfg.target.gmm_n_mixes,
            loc_scaling=40,
            log_var_scaling=1,
            seed=cfg.seed,
        )
    elif cfg.wandb.name == "GMM9":
        target = GMM9(
            dim=cfg.target.input_dim,
            scale=3,
            std=0.35,
            seed=cfg.seed,
        )
    else:
        raise NotImplementedError

    base = MultivariateGaussian(
        dim=cfg.target.input_dim,
        sigma=cfg.base.initial_sigma,
    )

    key, model_key = jrandom.split(key)
    rngs = nnx.Rngs(model_key)
    v_theta = TimeVelocityField(
        input_dim=cfg.target.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(v_theta, optax.adam(cfg.optimiser.learning_rate), wrt=nnx.Param)

    wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}_isfm",
    )

    def diff_log_density(x: jax.Array) -> jax.Array:
        return target.log_prob(x) - base.log_prob(x)

    num_epochs = cfg.training.num_epochs
    batches_per_epoch = cfg.training.batches_per_epoch
    batch_size = cfg.training.batch_size

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            key, batch_key = jrandom.split(key)
            k_samples, k_t = jrandom.split(batch_key)

            x_1 = base.sample(k_samples, (batch_size * 2,))
            t = jrandom.uniform(k_t, (batch_size, 1))

            z_ts = x_1[:batch_size]
            z_te = x_1[batch_size:]
            z_t = (1.0 - t) * z_ts + t * z_te
            weights = jax.nn.softmax(diff_log_density(z_te), axis=0)

            def loss_fn(model: TimeVelocityField) -> jax.Array:
                prediction = model(z_t, t)
                residual = prediction - (z_te - z_ts)
                per_example = jnp.mean(residual**2, axis=-1)
                return jnp.sum(per_example * weights)

            loss_value, grads = nnx.value_and_grad(loss_fn)(v_theta)
            optimizer.update(v_theta, grads)

            loss_float = float(jax.device_get(loss_value))
            total_epoch_loss += loss_float

        avg_loss = total_epoch_loss / batches_per_epoch
        wandb.log({f"train/avg_epoch_loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Avg Loss: {avg_loss:.4f}")

        if epoch % 50 == 0:
            key, sample_key = jrandom.split(key)
            initial_samples = base.sample(sample_key, (2048,))
            ts = jnp.linspace(0.0, 1.0, cfg.sampling.num_timesteps)
            samples = generate_samples_with_euler_method(v_theta, initial_samples, ts)
            with tempfile.TemporaryDirectory() as tmpdir:
                fig = plot_MoG(
                    log_prob_function=target.log_prob,
                    samples=samples,
                    name=cfg.wandb.name,
                )
                fig.savefig(os.path.join(tmpdir, "samples.png"))
                wandb.log({f"samples": wandb.Image(os.path.join(tmpdir, "samples.png"))})
                plt.close(fig)


def generate_samples_with_euler_method(
    v_theta: TimeVelocityField, initial_samples: jax.Array, ts: jax.Array
) -> jax.Array:
    def step(samples, t_pair):
        t_p, t_n = t_pair
        t = jnp.full((samples.shape[0], 1), t_p)
        velocity = v_theta(samples, t)
        samples = samples + velocity * (t_n - t_p)
        return samples, samples

    t_pairs = jnp.stack([ts[:-1], ts[1:]], axis=-1)
    final_samples, _ = jax.lax.scan(step, initial_samples, t_pairs)
    return final_samples


@hydra.main(config_path="./configs", config_name="gmm.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
