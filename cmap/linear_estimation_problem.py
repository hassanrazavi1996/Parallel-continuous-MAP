from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
from typing import NamedTuple
from typing import Callable


class Estimation(NamedTuple):

    F: Callable
    H: Callable
    c: Callable
    r: Callable
    L: Callable
    W: Callable
    R: Callable
    y: jnp.ndarray
    P0: jnp.ndarray
    m0: jnp.ndarray
    T: float
