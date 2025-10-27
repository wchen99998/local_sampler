from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

Array = jax.Array


def _gaussian_log_prob(x: Array, loc: Array, scale_diag: Array) -> Array:
    """Compute log N(x | loc, diag(scale_diag^2)) elementwise."""
    diff = (x - loc) / scale_diag
    log_det = 2.0 * jnp.log(scale_diag).sum(axis=-1)
    dim = x.shape[-1]
    norm = -0.5 * (dim * jnp.log(2 * jnp.pi) + log_det)
    return norm - 0.5 * jnp.sum(diff**2, axis=-1)


@dataclass
class GMM40:
    dim: int
    n_mixes: int
    loc_scaling: float
    log_var_scaling: float = 1.0
    seed: int = 0
    n_test_set_samples: int = 1000

    def __post_init__(self):
        key = jax.random.PRNGKey(self.seed)
        key_mean, key = jax.random.split(key)

        self.locs = (
            (jax.random.uniform(key_mean, (self.n_mixes, self.dim)) - 0.5)
            * 2.0
            * self.loc_scaling
        )
        log_var = jnp.ones((self.n_mixes, self.dim)) * self.log_var_scaling
        self.scale_diag = jax.nn.softplus(log_var)
        self.logits = jnp.zeros((self.n_mixes,))
        self._key = key

    @property
    def test_set(self) -> Array:
        key, self._key = jax.random.split(self._key)
        return self.sample(key, (self.n_test_set_samples,))

    def log_prob(self, x: Array) -> Array:
        x = jnp.atleast_2d(x)
        diff = x[:, None, :] - self.locs[None, :, :]
        scale = self.scale_diag[None, :, :]
        log_components = _gaussian_log_prob(diff, jnp.zeros_like(diff), scale)
        log_weights = jax.nn.log_softmax(self.logits)
        return jax.scipy.special.logsumexp(log_weights + log_components, axis=1)

    def sample(self, key: Array, shape: Sequence[int] = ()) -> Array:
        key_comp, key_noise = jax.random.split(key)
        comps = jax.random.categorical(key_comp, self.logits, shape=shape)
        locs = self.locs[comps]
        scale = self.scale_diag[comps]
        noise = jax.random.normal(key_noise, shape + (self.dim,))
        return locs + noise * scale


@dataclass
class GMM9:
    dim: int = 2
    scale: float = 4.0
    std: float = 1.0
    seed: int = 0
    n_test_set_samples: int = 1000

    def __post_init__(self):
        self.n_mixes = 9
        self.dimensionality = 2
        self.locs = jnp.array(
            [
                [-self.scale, -self.scale],
                [-self.scale, self.scale],
                [self.scale, self.scale],
                [self.scale, -self.scale],
                [-self.scale, 0.0],
                [self.scale, 0.0],
                [0.0, -self.scale],
                [0.0, self.scale],
                [0.0, 0.0],
            ]
        )
        self.scale_diag = jnp.ones((self.n_mixes, self.dimensionality)) * self.std
        self.logits = jnp.zeros((self.n_mixes,))
        self._key = jax.random.PRNGKey(self.seed)

    @property
    def test_set(self) -> Array:
        key, self._key = jax.random.split(self._key)
        return self.sample(key, (self.n_test_set_samples,))

    def log_prob(self, x: Array) -> Array:
        x = jnp.atleast_2d(x)
        diff = x[:, None, :] - self.locs[None, :, :]
        scale = self.scale_diag[None, :, :]
        log_components = _gaussian_log_prob(diff, jnp.zeros_like(diff), scale)
        log_weights = jax.nn.log_softmax(self.logits)
        return jax.scipy.special.logsumexp(log_weights + log_components, axis=1)

    def sample(self, key: Array, shape: Sequence[int] = ()) -> Array:
        key_comp, key_noise = jax.random.split(key)
        comps = jax.random.categorical(key_comp, self.logits, shape=shape)
        locs = self.locs[comps]
        scale = self.scale_diag[comps]
        noise = jax.random.normal(key_noise, shape + (self.dimensionality,))
        return locs + noise * scale


@dataclass
class MultivariateGaussian:
    dim: int = 2
    sigma: float | Array = 1.0

    def __post_init__(self):
        sigma = jnp.asarray(self.sigma)
        if sigma.ndim == 0:
            sigma = jnp.ones((self.dim,)) * sigma
        if sigma.shape[0] != self.dim:
            raise ValueError(f"Sigma shape {sigma.shape} does not match dimension {self.dim}.")
        self.scale_diag = sigma

    def log_prob(self, x: Array) -> Array:
        x = jnp.atleast_2d(x)
        return _gaussian_log_prob(x, jnp.zeros(self.dim), self.scale_diag)

    def sample(self, key: Array, shape: Sequence[int]) -> Array:
        noise = jax.random.normal(key, shape + (self.dim,))
        return noise * self.scale_diag


def plot_contours(
    log_prob_func: Callable[[Array], Array],
    samples: Array | None = None,
    ax=None,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    xy_ticks: Tuple[float, float] = (-40.0, 40.0),
    grid_width_n_points: int = 20,
    n_contour_levels: int | None = None,
    log_prob_min: float = -1000.0,
    plot_marginal_dims: Tuple[int, int] = (0, 1),
    s: float = 2.0,
    alpha: float = 0.6,
    title: str | None = None,
    plt_show: bool = True,
    xy_tick: bool = True,
):
    """Plot contours of a log probability function defined on 2D space."""
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.figure

    x_points_dim1 = np.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)), dtype=np.float32)
    log_p_x = np.array(log_prob_func(jnp.asarray(x_points)))
    log_p_x = np.clip(log_p_x, log_prob_min, None)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))

    grid_x = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points))
    grid_y = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points))
    if n_contour_levels:
        ax.contour(grid_x, grid_y, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(grid_x, grid_y, log_p_x)

    if samples is not None:
        samples_np = np.asarray(samples)
        samples_np = np.clip(samples_np, bounds[0], bounds[1])
        ax.scatter(
            samples_np[:, plot_marginal_dims[0]],
            samples_np[:, plot_marginal_dims[1]],
            s=s,
            alpha=alpha,
        )
        if xy_tick:
            ax.set_xticks([xy_ticks[0], 0.0, xy_ticks[1]])
            ax.set_yticks([xy_ticks[0], 0.0, xy_ticks[1]])
        ax.tick_params(axis="both", which="major", labelsize=15)
    if title:
        ax.set_title(title)
        ax.title.set_fontsize(40)
    if plt_show:
        plt.show()
    return fig


def plot_heat(log_prob_function: Callable[[Array], Array], samples: Array, size: float = 4.5):
    w = 100
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    heat_score = np.exp(np.array(log_prob_function(jnp.asarray(grid))))
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(heat_score.reshape(w, w), extent=(-size, size, -size, size), origin="lower")
    samples_np = np.asarray(samples)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], c="r", s=1)
    return fig


def plot_MoG(
    log_prob_function: Callable[[Array], Array],
    samples: Array,
    name: str = "GMM40",
    save_path: str | None = None,
    save_name: str | None = None,
    title: str | None = None,
):
    if name == "GMM40":
        fig = plot_contours(
            log_prob_function,
            samples=samples,
            bounds=(-56, 56),
            n_contour_levels=50,
            grid_width_n_points=200,
            title=title,
            plt_show=False,
        )
    elif name == "GMM9":
        fig = plot_heat(log_prob_function, samples=samples)
    else:
        raise NotImplementedError

    if save_path and save_name:
        fig.savefig(f"{save_path}/{save_name}")
    return fig
