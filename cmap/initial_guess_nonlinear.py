from jax import config
config.update("jax_enable_x64",True)

import jax.numpy as jnp
from jax import lax


def intial_guess(x0, steps):
    x = jnp.full((steps, len(x0)), 0.1)
    u = jnp.full((steps, len(x0)), 0.0)
    x = x.at[-1, :].set(x0)
    return x, u


def simulate(x0,f,steps,dt):
    x_seq=jnp.zeros((steps, x0.shape[0]))

    def step(x,t):
        x_new=x+f(x)*dt
        return x_new, x_new
    
    Ts=jnp.arange(steps)*dt
    _,x_seq=lax.scan(step, x0,Ts)

    x_seq = jnp.vstack([x0, x_seq])[::-1]
    return x_seq
