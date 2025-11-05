from jax import config 
config.update("jax_enable_x64",True)


import jax.numpy as jnp
from typing import NamedTuple
from typing import callable


class Estimation(NamedTuple):
      F:callable
      H:callable
      c:callable
      r:callable
      L:callable
      W:callable
      R:callable
      y:jnp.ndarray
      P0:jnp.ndarray
      x0:jnp.ndarray





