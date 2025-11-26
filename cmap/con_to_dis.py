import jax.debug
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

# general function
def f_convert(t, dt, nsteps, x_dis):

    t = jnp.asarray(t)
    k = (t / (dt - 1e-15)).astype(jnp.int32)
    k = jnp.clip(k, 0, nsteps - 1)
    x_vals = jnp.take(x_dis, k, axis=0)
    x_vals = jnp.where(jnp.ndim(t) == 0, x_vals, x_vals.T)

    return x_vals
