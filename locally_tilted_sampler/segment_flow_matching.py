from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from .flow import FlowDimensions, FlowMLP
from .utils import print_parameter_counts

Array = jnp.ndarray


class Density(Protocol):
    def log_prob(self, x: Array) -> Array: ...
    def sample(self, key: jax.Array, shape: Tuple[int, ...]) -> Array: ...


@dataclass(frozen=True)
class TrainingConfig:
    time_slices: int = 256
    solver_substeps: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 1000
    train_samples: int = 20000
    train_batch_size: int = 1024
    threshold: float = 1e-5
    seed: int = 0
    t_end: float = 1.0
    dtmax: float | None = None


@dataclass(frozen=True)
class TrainResult:
    flows: List[FlowMLP]
    time_points: List[float]
    loss_log: List[float]
    final_samples: Array


def importance_sample_batch(
    key: jax.Array, xbatch: Array, target: Density, prior: Density, delta_t: float
) -> Tuple[Array, Array]:
    log_prob_diff = target.log_prob(xbatch) - prior.log_prob(xbatch)
    weights = jax.nn.softmax(delta_t * log_prob_diff)
    indices = jax.random.choice(key, xbatch.shape[0], shape=(xbatch.shape[0],), p=weights)
    return xbatch[indices], indices


def compute_weighted_loss(
    flow: FlowMLP, key: jax.Array, x_ts: Array, x_te: Array
) -> Array:
    key_t, key_noise = jax.random.split(key)
    t = jax.random.uniform(key_t, (x_ts.shape[0], 1), dtype=x_ts.dtype)
    x_start = x_ts
    x_t = (1.0 - t) * x_start + t * x_te
    v_t = flow(x_t, t)
    diff = v_t - (x_te - x_start)
    return jnp.mean(jnp.sum(diff * diff, axis=-1))


def make_train_step():
    @nnx.jit
    def train_step(flow: FlowMLP, optimizer: nnx.Optimizer, key: jax.Array, x_ts: Array, x_te: Array):
        def loss_fn(model: FlowMLP):
            return compute_weighted_loss(model, key, x_ts, x_te)

        loss, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, nnx.Param)
        )(flow)
        optimizer.update(flow, grads)
        return flow, optimizer, loss

    return train_step


def init_flow(flow_dims: FlowDimensions, key: jax.Array, lr: float, weight_decay: float) -> Tuple[FlowMLP, nnx.Optimizer]:
    flow = FlowMLP(flow_dims, rngs=nnx.Rngs(key))
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    optimizer = nnx.Optimizer(flow, tx, wrt=nnx.Param)
    return flow, optimizer


def apply_single_flow(
    key: jax.Array,
    flow: FlowMLP,
    samples: Array,
    solver_substeps: int,
) -> Array:
    samples = samples
    ts = jnp.linspace(0.0, 1.0, solver_substeps, dtype=samples.dtype)
    x = samples
    for t_prev, t_next in zip(ts[:-1], ts[1:]):
        t_val = jnp.full((x.shape[0], 1), t_prev, dtype=x.dtype)
        v = flow(x, t_val)
        x = x + v * (t_next - t_prev)
    return x


def propagate_flow_sequence(
    key: jax.Array,
    flows: Sequence[FlowMLP],
    samples: Array,
    solver_substeps: int,
    return_all: bool = False,
) -> Array | List[Array]:
    history: List[Array] = [samples] if return_all else []
    x = samples
    for flow in flows:
        key, subkey = jax.random.split(key)
        x = apply_single_flow(subkey, flow, x, solver_substeps)
        if return_all:
            history.append(x)
    return history if return_all else x


def train_locally_tilted_sampler(
    flow_dims: FlowDimensions,
    prior: Density,
    target: Density,
    config: TrainingConfig,
) -> TrainResult:
    if not (0.0 < config.t_end <= 1.0):
        raise ValueError("t_end must be in (0, 1].")

    dtmax = 1.0 / config.time_slices if config.dtmax is None else config.dtmax
    key = jax.random.PRNGKey(config.seed)

    key, sample_key = jax.random.split(key)
    samples = prior.sample(sample_key, (config.train_samples,))

    flows: List[FlowMLP] = []
    time_points: List[float] = []
    loss_log: List[float] = []
    printed_params = False

    t = 0.0
    epsilon = 1e-6

    while t < config.t_end - epsilon:
        delta_t = float(min(dtmax, config.t_end - t))
        key, init_key = jax.random.split(key)
        flow, optimizer = init_flow(flow_dims, init_key, config.lr, config.weight_decay)
        train_step = make_train_step()

        if not printed_params:
            print_parameter_counts(flow, module_name="Flow", max_depth=3)
            printed_params = True

        loss_val = jnp.inf
        for _ in range(config.epochs):
            key, batch_key, is_key, loss_key = jax.random.split(key, 4)
            idx = jax.random.choice(
                batch_key, config.train_samples, (config.train_batch_size,), replace=False
            )
            xbatch = samples[idx]
            x_is, parents = importance_sample_batch(is_key, xbatch, target, prior, delta_t)
            x_parents = xbatch[parents]
            flow, optimizer, loss_val = train_step(flow, optimizer, loss_key, x_parents, x_is)
            if float(loss_val) < config.threshold:
                break

        flows.append(nnx.clone(flow))
        loss_log.append(float(loss_val))
        t += delta_t
        time_points.append(t)

        key, prop_key = jax.random.split(key)
        samples = propagate_flow_sequence(
            prop_key,
            flows,
            samples,
            config.solver_substeps,
            return_all=False,
        )

    return TrainResult(
        flows=flows,
        time_points=time_points,
        loss_log=loss_log,
        final_samples=samples,
    )
