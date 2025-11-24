from __future__ import annotations

from dataclasses import dataclass

import flax.nnx as nnx
import jax.numpy as jnp


@dataclass(frozen=True)
class FlowDimensions:
    dim: int
    hidden: int
    depth: int


class FlowMLP(nnx.Module):
    def __init__(self, dims: FlowDimensions, *, rngs: nnx.Rngs):
        self.dims = dims
        self.input = nnx.Linear(dims.dim + 1, dims.hidden, rngs=rngs)
        self.hidden = nnx.List(
            nnx.Linear(dims.hidden, dims.hidden, rngs=rngs) for _ in range(dims.depth)
        )
        self.output = nnx.Linear(dims.hidden, dims.dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        h = jnp.concatenate([x, t], axis=-1)
        h = self.input(h)
        for layer in self.hidden:
            h = nnx.sigmoid(h)
            h = layer(h)
        h = nnx.sigmoid(h)
        return self.output(h)
