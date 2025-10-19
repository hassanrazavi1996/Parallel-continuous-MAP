import jax
import jax.numpy as jnp
from typing import Optional, Callable, Any
from jax import config

config.update("jax_enable_x64", True)


def euler(f: Callable, dt: float, x: jnp.ndarray, t: Optional[float]) -> jnp.ndarray:

    dx = f(x, t)* dt
    x_next = x + dx
    
    return x_next
