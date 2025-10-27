import os
import tempfile

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
import flax.nnx as nnx
from jax import lax, random as jrandom
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
    optimizer = nnx.Optimizer(v_theta, optax.adamw(cfg.optimiser.learning_rate), wrt=nnx.Param)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
    )

    def diff_log_density(x: jax.Array) -> jax.Array:
        return target.log_prob(x) - base.log_prob(x)

    num_timesteps = cfg.training.num_timesteps
    num_epochs = cfg.training.num_epochs
    batches_per_epoch = cfg.training.batches_per_epoch
    timesteps_for_loss = cfg.training.timesteps_for_loss
    batch_size = cfg.training.batch_size
    eps = 1e-5
    timesteps = jnp.linspace(eps, 1.0, num_timesteps)
    dt = (1.0 - eps) / num_timesteps
    delta_t = dt

    def batch_loss(model: TimeVelocityField, batch_key: jax.Array) -> jax.Array:
        k_sample, k_select, k_loop = jrandom.split(batch_key, 3)
        x_t = base.sample(k_sample, (batch_size * 2,))

        selected_indices = jrandom.permutation(k_select, num_timesteps)[:timesteps_for_loss]
        is_selected = jnp.zeros((num_timesteps,), dtype=bool).at[selected_indices].set(True)

        def body(i, carry):
            samples, loss_acc, num_loss, inner_key = carry
            t_scalar = timesteps[i]
            t = jnp.full((samples.shape[0], 1), t_scalar)
            velocity = lax.stop_gradient(model(samples, t))

            def loss_branch(branch_carry):
                current_samples, current_loss, current_num_loss, key_inner = branch_carry
                key_u, key_inner = jrandom.split(key_inner)
                u = jrandom.uniform(key_u, (batch_size, 1))
                z_ts = current_samples[:batch_size]
                z_te = current_samples[batch_size:]
                z_t = (1.0 - u) * z_ts + u * z_te
                t_ = u * delta_t + t[:batch_size]
                weights = jax.nn.softmax(delta_t * diff_log_density(z_te), axis=0)
                prediction = model(z_t, t_)
                residual = prediction - (z_te - z_ts)
                per_example = jnp.mean(residual**2, axis=-1)
                weighted_loss = jnp.sum(per_example * weights)
                return current_samples, current_loss + weighted_loss, current_num_loss + 1, key_inner

            samples, loss_acc, num_loss, inner_key = lax.cond(
                (t_scalar <= 1.0 - delta_t) & is_selected[i],
                loss_branch,
                lambda x: x,
                (samples, loss_acc, num_loss, inner_key),
            )

            samples = samples + velocity * dt
            return samples, loss_acc, num_loss, inner_key

        init_carry = (
            x_t,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            k_loop,
        )
        samples, loss_total, loss_count, _ = lax.fori_loop(0, num_timesteps, body, init_carry)
        loss_count = jnp.maximum(loss_count, 1)
        return loss_total / loss_count

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            key, batch_key = jrandom.split(key)

            def loss_fn(model: TimeVelocityField) -> jax.Array:
                return batch_loss(model, batch_key)

            loss_value, grads = nnx.value_and_grad(loss_fn)(v_theta)
            optimizer.update(v_theta, grads)

            loss_float = float(jax.device_get(loss_value))
            total_epoch_loss += loss_float
            wandb.log({"train/loss": loss_float})
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Batch [{batch_idx + 1}/{batches_per_epoch}], "
                f"Loss: {loss_float:.4f}"
            )

        avg_loss = total_epoch_loss / batches_per_epoch
        wandb.log({"train/avg_epoch_loss": avg_loss})
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
                wandb.log({"samples": wandb.Image(os.path.join(tmpdir, "samples.png"))})
                plt.close(fig)


def generate_samples_with_euler_method(
    v_theta: TimeVelocityField, initial_samples: jax.Array, ts: jax.Array
) -> jax.Array:
    """Generate samples with forward Euler integration over provided timesteps."""

    def step(samples, t_pair):
        t_p, t_n = t_pair
        t = jnp.full((samples.shape[0], 1), t_p)
        velocity = v_theta(samples, t)
        samples = samples + velocity * (t_n - t_p)
        return samples, samples

    t_pairs = jnp.stack([ts[:-1], ts[1:]], axis=-1)
    final_samples, _ = lax.scan(step, initial_samples, t_pairs)
    return final_samples


@hydra.main(config_path="./configs", config_name="gmm.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
