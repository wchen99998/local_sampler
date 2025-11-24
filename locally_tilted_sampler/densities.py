from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import math

Array = jnp.ndarray

LOG2PI = math.log(2.0 * math.pi)


def _as_array(x: Array | Sequence[float]) -> Array:
    return jnp.asarray(x, dtype=jnp.float32)


@dataclass(frozen=True)
class GaussianDensity:
    mean: Array
    cov: Array
    precision: Array
    log_norm: Array

    @classmethod
    def from_mean_cov(cls, mean: Sequence[float], cov: Sequence[Sequence[float]]) -> "GaussianDensity":
        m = _as_array(mean)
        c = _as_array(cov)
        precision = jnp.linalg.inv(c)
        _, logdet = jnp.linalg.slogdet(c)
        log_norm = -0.5 * (m.shape[-1] * LOG2PI + logdet)
        return cls(mean=m, cov=c, precision=precision, log_norm=log_norm)

    def log_prob(self, x: Array) -> Array:
        diff = x - self.mean
        mahal = jnp.einsum("...i,ij,...j->...", diff, self.precision, diff)
        return self.log_norm - 0.5 * mahal

    def sample(self, key: jax.Array, shape: Tuple[int, ...]) -> Array:
        return jax.random.multivariate_normal(key, self.mean, self.cov, shape)


@dataclass(frozen=True)
class GaussianMixture:
    means: Array
    logits: Array
    std: float

    def log_prob(self, x: Array) -> Array:
        x = _as_array(x)
        std2 = self.std ** 2
        diff = x[:, None, :] - self.means[None, :, :]
        sq_norm = jnp.sum(diff * diff, axis=-1)
        log_comp = -0.5 * (sq_norm / std2 + diff.shape[-1] * jnp.log(2.0 * jnp.pi * std2))
        return jsp.special.logsumexp(self.logits + log_comp, axis=1)

    def sample(self, key: jax.Array, shape: Tuple[int, ...]) -> Array:
        key_comp, key_noise = jax.random.split(key)
        idx = jax.random.categorical(key_comp, self.logits, shape=shape)
        eps = jax.random.normal(key_noise, shape + (self.means.shape[-1],))
        return self.means[idx] + self.std * eps


def make_gmm9(scale: float = 4.0, std: float = 1.0) -> GaussianMixture:
    means = jnp.array(
        [
            [-scale, -scale],
            [-scale, scale],
            [scale, scale],
            [scale, -scale],
            [-scale, 0.0],
            [scale, 0.0],
            [0.0, -scale],
            [0.0, scale],
            [0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    logits = jnp.zeros((means.shape[0],), dtype=jnp.float32)
    return GaussianMixture(means=means, logits=logits, std=float(std))


def make_gmm40(log_var_scaling: float = 1.0) -> GaussianMixture:
    means = jnp.array(
        [
            [-0.2995, 21.4577],
            [-32.9218, -29.4376],
            [-15.4062, 10.7263],
            [-0.7925, 31.7156],
            [-3.5498, 10.5845],
            [-12.0885, -7.8626],
            [-38.2139, -26.4913],
            [-16.4889, 1.4817],
            [15.8134, 24.0009],
            [-27.1176, -17.4185],
            [14.5287, 33.2155],
            [-8.232, 29.9325],
            [-6.4473, 4.2326],
            [36.219, -37.1068],
            [-25.1815, -10.1266],
            [-15.592, 34.56],
            [-25.9272, -18.4133],
            [-27.9456, -37.4624],
            [-23.3496, 34.3839],
            [17.8487, 19.3869],
            [2.1037, -20.5073],
            [6.7674, -37.3478],
            [-28.9026, -20.6212],
            [25.2375, 23.4529],
            [-17.7398, -1.4433],
            [25.5824, 39.7653],
            [15.8753, 5.4037],
            [26.8195, -23.5521],
            [7.4538, -31.0122],
            [-27.7234, -20.6633],
            [18.0989, 16.0864],
            [-23.6941, 12.0843],
            [21.9589, -5.0487],
            [1.5273, 9.2682],
            [24.8151, 38.4078],
            [-30.8249, -14.6588],
            [15.7204, 33.142],
            [34.8083, 35.2943],
            [7.9606, -34.7833],
            [3.6797, -25.0242],
        ],
        dtype=jnp.float32,
    )
    logits = jnp.zeros((means.shape[0],), dtype=jnp.float32)
    std = float(jax.nn.softplus(log_var_scaling))
    return GaussianMixture(means=means, logits=logits, std=std)


def evaluate_on_grid(
    log_prob_fn: Callable[[Array], Array],
    bounds: Tuple[float, float] = (-5.0, 5.0),
    n_points: int = 200,
) -> Tuple[Array, Array, Array]:
    xs = jnp.linspace(bounds[0], bounds[1], n_points)
    grid = jnp.stack(jnp.meshgrid(xs, xs, indexing="xy"), axis=-1).reshape(-1, 2)
    logp = log_prob_fn(grid).reshape(n_points, n_points)
    return xs, xs, logp


def plot_density(
    log_prob_fn: Callable[[Array], Array],
    bounds: Tuple[float, float] = (-5.0, 5.0),
    n_points: int = 200,
    contour_levels: int | None = None,
    to_prob: bool = True,
    samples: Array | None = None,
    sample_kwargs: dict | None = None,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - soft dependency
        raise ImportError("matplotlib is required for plotting densities") from exc

    xs, ys, logp = evaluate_on_grid(log_prob_fn, bounds=bounds, n_points=n_points)
    values = jnp.exp(logp) if to_prob else logp
    fig, ax = plt.subplots(figsize=(5, 4))
    if contour_levels is None:
        contour_levels = 30
    cs = ax.contourf(xs, ys, values, levels=contour_levels)
    fig.colorbar(cs, ax=ax, label="density" if to_prob else "log-density")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Density")
    if samples is not None:
        kw = {"s": 4, "c": "r", "alpha": 0.6}
        if sample_kwargs:
            kw.update(sample_kwargs)
        smp = jnp.asarray(samples)
        ax.scatter(smp[:, 0], smp[:, 1], **kw)
    return fig, ax
