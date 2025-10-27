import jax
import jax.numpy as jnp
import flax.nnx as nnx


class TimeVelocityField(nnx.Module):
    """Simple MLP that models the time-dependent velocity field."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3, *, rngs: nnx.Rngs):
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.layers = nnx.List()
        in_features = input_dim + 1  # concat x and t
        for _ in range(depth):
            self.layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        self.output_layer = nnx.Linear(hidden_dim, input_dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Forward pass that expects x and t with matching leading dims."""
        h = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            h = jax.nn.sigmoid(layer(h))
        return self.output_layer(h)
