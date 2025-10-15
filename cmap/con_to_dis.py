import jax.debug
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


def y(t,dt,nsteps,DY_jax0):
   
        t = jnp.asarray(t)
        k = (t / (dt- 1e-10) ).astype(jnp.int32)
        k = jnp.clip(k, 0, nsteps - 1)
        y_vals = jnp.take(DY_jax0, k, axis=0)
        y_vals = jnp.where(jnp.ndim(t) == 0, y_vals, y_vals.T)
         
        return y_vals


def y_reverse(t,tf,dt,nsteps,DY_jax0):
    
        t = jnp.asarray(t)
        k = (t / (dt+ 1e-10) ).astype(jnp.int32)
        k = jnp.clip(k, 0, nsteps - 1)
        y_vals = jnp.take(DY_jax0[::-1], k, axis=0)
        y_vals = jnp.where(jnp.ndim(t) == 0, y_vals, y_vals.T)
         

        return y_vals

