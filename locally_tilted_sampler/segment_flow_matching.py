from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Protocol, Sequence, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from .flow import FlowDimensions, FlowMLP
from .utils import LiveLossPlot, print_parameter_counts

Array = jnp.ndarray
DEFAULT_LOG_EVERY = 50


class Density(Protocol):
    def log_prob(self, x: Array) -> Array: ...
    def sample(self, key: jax.Array, shape: Tuple[int, ...]) -> Array: ...


@dataclass(frozen=True)
class TrainingConfig:
    time_slices: int = 256
    solver_substeps: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_updates: int = 1000
    train_samples: int = 20000
    train_batch_size: int = 1024
    threshold: float = 1e-5
    seed: int = 0
    t_end: float = 1.0
    dtmax: float | None = None
    use_ancestral_pairs: bool = True
    use_stratified_coupling: bool = False
    random_walk_std: float = 0.0
    random_walk_steps: int = 0
    loss_callback: callable | None = None
    optimizer: str = "adamw"  # choices: adamw, adam, muon
    optimizer_kwargs: Mapping[str, float] = field(default_factory=dict)
    log_every: int = DEFAULT_LOG_EVERY


@dataclass(frozen=True)
class TrainResult:
    flows: List[FlowMLP]
    time_points: List[float]
    loss_log: List[float]
    final_samples: Array


def make_importance_sampler(
    target_log_prob: Callable[[Array], Array],
    prior_log_prob: Callable[[Array], Array],
):
    @nnx.jit(static_argnums=2)
    def _fn(key: jax.Array, xbatch: Array, delta_t: float) -> Tuple[Array, Array]:
        log_prob_diff = target_log_prob(xbatch) - prior_log_prob(xbatch)
        weights = jax.nn.softmax(delta_t * log_prob_diff)
        indices = jax.random.choice(key, xbatch.shape[0], shape=(xbatch.shape[0],), p=weights)
        return xbatch[indices], indices

    return _fn


def make_ancestral_pair_sampler(
    target_log_prob: Callable[[Array], Array],
    prior_log_prob: Callable[[Array], Array],
    use_stratified: bool,
):
    @nnx.jit(static_argnums=2)
    def _fn(key: jax.Array, xbatch: Array, delta_t: float) -> Tuple[Array, Array]:
        log_prob_diff = target_log_prob(xbatch) - prior_log_prob(xbatch)
        weights = jax.nn.softmax(delta_t * log_prob_diff)
        key_resample, key_cpl = jax.random.split(key)
        if use_stratified:
            parent_idx, child_idx = stratified_coupling(key_cpl, xbatch, weights)
            return xbatch[child_idx], parent_idx
        idx = jax.random.choice(
            key_resample, xbatch.shape[0], shape=(xbatch.shape[0],), p=weights
        )
        return xbatch[idx], idx

    return _fn


def apply_random_walk(key: jax.Array, x: Array, steps: int, step_std: float) -> Array:
    if steps <= 0 or step_std <= 0.0:
        return x
    scale = step_std * jnp.sqrt(float(steps))
    noise = jax.random.normal(key, x.shape) * scale
    return x + noise


def make_random_pair_sampler(
    importance_sampler: Callable[[jax.Array, Array, float], Tuple[Array, Array]]
):
    @nnx.jit(static_argnums=2)
    def _fn(key: jax.Array, xbatch: Array, delta_t: float) -> Tuple[Array, Array]:
        """Return (child, parent) pairs with parents as the current batch order.

        Endpoints are importance-resampled; parents stay aligned to ``xbatch`` positions.
        """
        x_is, _ = importance_sampler(key, xbatch, delta_t)
        parent_idx = jnp.arange(xbatch.shape[0], dtype=jnp.int32)
        return x_is, parent_idx

    return _fn


def stratified_coupling(key: jax.Array, x: Array, w: Array) -> Tuple[Array, Array]:
    """Stratified resampling that keeps parents and children monotone along a random ray.

    Args:
        key: PRNG key.
        x: Points to couple, shape (N, dim).
        w: Non-negative weights for ``x``; they are normalized internally.

    Returns:
        parent_idx: Indices of ``x`` sorted along the random projection.
        child_idx: Coupled resampled indices, matched to ``parent_idx`` order.
    """

    n = x.shape[0]
    key_r, key_u = jax.random.split(key)

    direction = jax.random.normal(key_r, (x.shape[1],))
    direction = direction / (jnp.linalg.norm(direction) + 1e-12)

    order = jnp.argsort(x @ direction)
    weights = jnp.asarray(w)[order]
    weights = weights / (jnp.sum(weights) + 1e-12)

    cdf = jnp.cumsum(weights)
    u0 = jax.random.uniform(key_u, ())
    u = (jnp.arange(n, dtype=weights.dtype) + u0) / float(n)

    child_pos = jnp.searchsorted(cdf, u, side="left")
    child_idx = order[child_pos]
    parent_idx = order
    return parent_idx, child_idx


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


def _build_optimizer(name: str, lr: float, weight_decay: float, **kwargs):
    name = name.lower()
    wd = kwargs.pop("weight_decay", weight_decay)
    if name == "adamw":
        return optax.adamw(learning_rate=lr, weight_decay=wd, **kwargs)
    if name == "adam":
        return optax.adam(learning_rate=lr, **kwargs)
    if name == "muon":
        return optax.contrib.muon(learning_rate=lr, weight_decay=wd, **kwargs)
    raise ValueError(f"Unknown optimizer '{name}'. Expected one of ['adamw', 'adam', 'muon'].")


def init_flow(flow_dims: FlowDimensions, key: jax.Array, lr: float, weight_decay: float, optimizer_name: str, optimizer_kwargs: Mapping[str, float] | None = None) -> Tuple[FlowMLP, nnx.Optimizer]:
    flow = FlowMLP(flow_dims, rngs=nnx.Rngs(key))
    tx = _build_optimizer(optimizer_name, lr=lr, weight_decay=weight_decay, **(optimizer_kwargs or {}))
    optimizer = nnx.Optimizer(flow, tx, wrt=nnx.Param)
    return flow, optimizer


@nnx.jit(static_argnums=3)
def apply_single_flow(
    key: jax.Array,
    flow: FlowMLP,
    samples: Array,
    solver_substeps: int,
) -> Array:
    del key  # unused but kept for API consistency
    ts = jnp.linspace(0.0, 1.0, solver_substeps, dtype=samples.dtype)
    t_prev = ts[:-1]
    dt = ts[1:] - ts[:-1]

    def body(x, inputs):
        t_curr, dt_curr = inputs
        t_val = jnp.full((x.shape[0], 1), t_curr, dtype=x.dtype)
        v = flow(x, t_val)
        x_new = x + v * dt_curr
        return x_new, None

    x_final, _ = jax.lax.scan(body, samples, (t_prev, dt))
    return x_final


def propagate_flow_sequence(
    key: jax.Array,
    flows: Sequence[FlowMLP],
    samples: Array,
    solver_substeps: int,
    return_all: bool = False,
    random_walk_steps: int = 0,
    random_walk_std: float = 0.0,
) -> Array | List[Array]:
    use_random_walk = random_walk_steps > 0 and random_walk_std > 0.0

    x = samples
    if use_random_walk:
        key, rw_key = jax.random.split(key)
        x = apply_random_walk(rw_key, x, random_walk_steps, random_walk_std)

    history: List[Array] = [x] if return_all else []
    for idx, flow in enumerate(flows):
        key, subkey = jax.random.split(key)
        x = apply_single_flow(subkey, flow, x, solver_substeps)
        if return_all:
            history.append(x)
        if use_random_walk and idx < len(flows) - 1:
            key, rw_key = jax.random.split(key)
            x = apply_random_walk(rw_key, x, random_walk_steps, random_walk_std)

    return history if return_all else x


def train_locally_tilted_sampler(
    flow_dims: FlowDimensions,
    prior: Density,
    target: Density,
    config: TrainingConfig,
) -> TrainResult:
    if not (0.0 < config.t_end <= 1.0):
        raise ValueError("t_end must be in (0, 1].")

    steps_per_epoch = max(
        1, (config.train_samples + config.train_batch_size - 1) // config.train_batch_size
    )
    total_batch_slots = steps_per_epoch * config.train_batch_size

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

    def build_epoch_batches(rng: jax.Array) -> Tuple[Array, jax.Array]:
        perm_key, pad_key, next_key = jax.random.split(rng, 3)
        perm = jax.random.permutation(perm_key, config.train_samples)
        if total_batch_slots > config.train_samples:
            pad = jax.random.choice(
                pad_key,
                config.train_samples,
                shape=(total_batch_slots - config.train_samples,),
                replace=False,
            )
            perm = jnp.concatenate([perm, pad], axis=0)
        batches = perm.reshape((steps_per_epoch, config.train_batch_size))
        return batches, next_key

    def refresh_training_samples(
        rng: jax.Array, current_flows: Sequence[FlowMLP]
    ) -> Tuple[Array, jax.Array]:
        sample_key, prop_key, next_key = jax.random.split(rng, 3)
        new_samples = prior.sample(sample_key, (config.train_samples,))
        new_samples = propagate_flow_sequence(
            prop_key,
            current_flows,
            new_samples,
            config.solver_substeps,
            return_all=False,
            random_walk_steps=config.random_walk_steps,
            random_walk_std=config.random_walk_std,
        )
        return new_samples, next_key

    # Build jitted samplers once.
    importance_sampler = make_importance_sampler(target.log_prob, prior.log_prob)
    ancestral_pair_sampler = make_ancestral_pair_sampler(
        target.log_prob, prior.log_prob, config.use_stratified_coupling
    )
    random_pair_sampler = make_random_pair_sampler(importance_sampler)

    while t < config.t_end - epsilon:
        delta_t = float(min(dtmax, config.t_end - t))
        key, init_key = jax.random.split(key)
        flow, optimizer = init_flow(
            flow_dims,
            init_key,
            config.lr,
            config.weight_decay,
            config.optimizer,
            config.optimizer_kwargs,
        )
        train_step = make_train_step()

        if not printed_params:
            print_parameter_counts(flow, module_name="Flow", max_depth=3)
            printed_params = True
        print(f"[time slice start] t={t:.4f}, delta_t={delta_t:.4f}, samples={samples.shape[0]}")

        epoch_batches, key = build_epoch_batches(key)
        loss_val = jnp.inf
        for step in range(config.max_updates):
            epoch_step = step % steps_per_epoch
            if step > 0 and epoch_step == 0:
                samples, key = refresh_training_samples(key, flows)
                epoch_batches, key = build_epoch_batches(key)
            if config.random_walk_steps > 0 and config.random_walk_std > 0.0:
                key, is_key, loss_key, rw_key = jax.random.split(key, 4)
            else:
                key, is_key, loss_key = jax.random.split(key, 3)
                rw_key = None
            xbatch = samples[epoch_batches[epoch_step]]
            if rw_key is not None:
                xbatch = apply_random_walk(
                    rw_key, xbatch, config.random_walk_steps, config.random_walk_std
                )
            if config.use_ancestral_pairs:
                x_is, parents = ancestral_pair_sampler(is_key, xbatch, delta_t)
            else:
                x_is, parents = random_pair_sampler(is_key, xbatch, delta_t)
            x_parents = xbatch[parents]
            flow, optimizer, loss_val = train_step(flow, optimizer, loss_key, x_parents, x_is)
            if step % config.log_every == 0 and config.loss_callback is None:
                print(f"  step {step:04d}  loss={float(loss_val):.6f}")
            if config.loss_callback is not None and step % config.log_every == 0:
                config.loss_callback(
                    len(flows),  # time slice index
                    step,
                    float(loss_val),
                    float(t),
                    float(delta_t),
                )
            if float(loss_val) < config.threshold:
                break

        flows.append(nnx.clone(flow))
        loss_log.append(float(loss_val))
        t += delta_t
        time_points.append(t)
        print(f"[time slice end]  t={t:.4f}, loss={float(loss_val):.6f}")
        if config.loss_callback is not None:
            config.loss_callback(
                len(flows) - 1,
                step,
                float(loss_val),
                float(t),
                float(delta_t),
            )

        samples, key = refresh_training_samples(key, flows)

    return TrainResult(
        flows=flows,
        time_points=time_points,
        loss_log=loss_log,
        final_samples=samples,
    )
