import jax.debug
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

# general function
# def f_convert(t, dt, nsteps, x_dis):

#     t = jnp.asarray(t)
#     k = jnp.floor(t / dt + 1e-9).astype(jnp.int32)
#     k = jnp.clip(k, 0, nsteps - 1)
#     x_vals = x_dis[k]
#     x_vals = jnp.where(jnp.ndim(t) == 0, x_vals, x_vals.T)

#     return x_vals


def f_convert(t, dt, nsteps, x_dis):
    t = jnp.asarray(t)
    nsteps = x_dis.shape[0]

    ratio = t / dt + 1e-9
    ratio = jnp.clip(ratio, 0, nsteps - 1)
    k = jnp.floor(ratio).astype(jnp.int32)

    x_vals = x_dis[k]
    return x_vals if t.ndim == 0 else jnp.moveaxis(x_vals, 0, -1)
