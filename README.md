# Locally Tilted Sampler (JAX/Flax)

Segment flow matching for importance-resampled trajectories, implemented with Flax/JAX instead of PyTorch.

## Quick start
- Install deps with uv: `uv sync`
- Run a tiny sanity check (small numbers to keep it fast) using Flax NNX:
```bash
python - <<'PY'
from locally_tilted_sampler import (
    FlowDimensions,
    TrainingConfig,
    make_gmm9,
    GaussianDensity,
    train_locally_tilted_sampler,
    plot_density,
)
prior = GaussianDensity.from_mean_cov([0.0, 0.0], [[1.5**2, 0.0], [0.0, 1.5**2]])
target = make_gmm9(scale=3.0, std=0.35)
config = TrainingConfig(
    time_slices=16,
    solver_substeps=8,
    max_updates=50,
    train_samples=1024,
    train_batch_size=256,
    seed=0,
    use_ancestral_pairs=True,  # set False to pair targets with random parents
)
result = train_locally_tilted_sampler(
    FlowDimensions(dim=2, hidden=128, depth=4),
    prior,
    target,
    config,
)
print("trained flows:", len(result.flows))
print("final sample mean:", result.final_samples.mean(axis=0))
fig, _ = plot_density(target.log_prob, bounds=(-5, 5), n_points=100)
fig.savefig("target_density.png")
PY
```

## What’s here
- `locally_tilted_sampler/densities.py`: Gaussian and mixture targets (`make_gmm9`, `make_gmm40`) with `log_prob`/`sample`.
- `locally_tilted_sampler/flow.py`: Flax NNX MLP velocity field mirroring the original Torch architecture.
- `locally_tilted_sampler/segment_flow_matching.py`: segment flow matching trainer, importance resampling, and flow propagation utilities.

Stick to functional usage: create densities, define `FlowDimensions`, set a `TrainingConfig`, and call `train_locally_tilted_sampler`. The trainer returns learned flow parameters plus the propagated sample set.
