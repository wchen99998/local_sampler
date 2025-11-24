from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp

from locally_tilted_sampler import (
    FlowDimensions,
    TrainingConfig,
    GaussianDensity,
    make_gmm9,
    make_gmm40,
    plot_density,
    train_locally_tilted_sampler,
)


def build_densities(target_name: str):
    target_name = target_name.lower()
    if target_name == "gmm9":
        prior = GaussianDensity.from_mean_cov([0.0, 0.0], [[1.5**2, 0.0], [0.0, 1.5**2]])
        target = make_gmm9(scale=3.0, std=0.35)
    elif target_name == "gmm40":
        prior = GaussianDensity.from_mean_cov([0.0, 0.0], [[20.0**2, 0.0], [0.0, 20.0**2]])
        target = make_gmm40(log_var_scaling=1.0)
    else:
        raise ValueError(f"unknown target: {target_name}")
    return prior, target


def parse_args():
    p = argparse.ArgumentParser(description="Run Locally Tilted Sampler training (JAX/NNX).")
    p.add_argument("--target", choices=["gmm9", "gmm40"], default="gmm9")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--nsample", type=int, default=1024)
    p.add_argument("--nbatch", type=int, default=256)
    p.add_argument("--nstep", type=int, default=64)
    p.add_argument("--nstep-per-grid", type=int, default=16, dest="nstep_per_grid")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jump-kernel-std", type=float, default=0.0, dest="jump_kernel_std")
    p.add_argument("--t-end", type=float, default=1.0, dest="t_end")
    p.add_argument("--plot-target", type=Path, default=None, dest="plot_path")
    return p.parse_args()


def main():
    args = parse_args()
    prior, target = build_densities(args.target)

    config = TrainingConfig(
        nstep=args.nstep,
        nstep_per_grid=args.nstep_per_grid,
        lr=args.lr,
        epochs=args.epochs,
        nsample=args.nsample,
        nbatch=args.nbatch,
        seed=args.seed,
        jump_kernel_std=args.jump_kernel_std,
        t_end=args.t_end,
    )

    result = train_locally_tilted_sampler(
        FlowDimensions(dim=2, hidden=args.hidden, depth=args.depth),
        prior,
        target,
        config,
    )

    print(f"trained {len(result.flows)} flows, losses: {[round(l, 4) for l in result.loss_log]}")
    print("final sample mean:", jnp.array(result.final_samples).mean(axis=0))

    if args.plot_path is not None:
        fig, _ = plot_density(target.log_prob, bounds=(-5, 5), n_points=200)
        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_path)
        print(f"saved target density plot to {args.plot_path}")


if __name__ == "__main__":
    main()
