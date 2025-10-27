import random

import numpy as np
import jax
from jax import random as jrandom


def seed_everything(seed: int) -> jax.Array:
    """Seed Python, NumPy, and return a base JAX PRNGKey."""
    random.seed(seed)
    np.random.seed(seed)
    return jrandom.PRNGKey(seed)
