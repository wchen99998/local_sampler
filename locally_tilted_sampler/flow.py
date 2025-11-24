from __future__ import annotations

from dataclasses import dataclass
import math

import flax.nnx as nnx
import jax.numpy as jnp


@dataclass(frozen=True)
class FlowDimensions:
    dim: int
    hidden: int
    depth: int
    time_embed_dim: int | None = None
    use_residual: bool = True


class FlowMLP(nnx.Module):
    def __init__(self, dims: FlowDimensions, *, rngs: nnx.Rngs):
        self.dims = dims
        time_dim = dims.time_embed_dim if dims.time_embed_dim is not None else dims.hidden
        self.time_embed_dim = time_dim
        width = dims.hidden

        self.input = nnx.Linear(dims.dim + time_dim, width, rngs=rngs)
        self.ff1 = nnx.List(nnx.Linear(width, width, rngs=rngs) for _ in range(dims.depth))
        self.ff2 = nnx.List(nnx.Linear(width, width, rngs=rngs) for _ in range(dims.depth))
        self.norm1 = nnx.List(nnx.RMSNorm(width, rngs=rngs) for _ in range(dims.depth))
        self.norm2 = nnx.List(nnx.RMSNorm(width, rngs=rngs) for _ in range(dims.depth))
        self.output_norm = nnx.RMSNorm(width, rngs=rngs)
        self.output = nnx.Linear(width, dims.dim, rngs=rngs)
        self.residual_scale: float = (
            1.0 / math.sqrt(dims.depth) if dims.use_residual and dims.depth > 0 else 0.0
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
        h = jnp.concatenate([x, t_embed], axis=-1)
        h = self.input(h)
        for l1, l2, n1, n2 in zip(self.ff1, self.ff2, self.norm1, self.norm2):
            residual = h
            h = n1(h)
            h = nnx.silu(h)
            h = l1(h)
            h = n2(h)
            h = nnx.silu(h)
            h = l2(h)
            h = residual + self.residual_scale * h
        h = self.output_norm(nnx.silu(h))
        return self.output(h)


def sinusoidal_time_embedding(t: jnp.ndarray, embed_dim: int) -> jnp.ndarray:
    """Standard sinusoidal positional encoding for scalar t."""
    half = embed_dim // 2
    # frequencies: exp(-2i/emb_dim * ln(10000))
    freq = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half, dtype=t.dtype) / jnp.maximum(half - 1, 1)
    )
    angles = t * freq
    sin = jnp.sin(angles)
    cos = jnp.cos(angles)
    emb = jnp.concatenate([sin, cos], axis=-1)
    if emb.shape[-1] < embed_dim:
        # Pad if embed_dim is odd.
        emb = jnp.pad(emb, ((0, 0), (0, embed_dim - emb.shape[-1])))
    return emb
