from jax import config
config.update("jax_enable_x64",True)

import jax.numpy as jnp
from jax import lax


def intial_guess(x0, steps):
    x = jnp.full((steps+1, len(x0)), 0.1)
    # u = jnp.full((steps, len(x0)), 0.2)
    u = jnp.tile(jnp.array([0.0,0.0,0.01,0.01,0.0]), (steps, 1))

    x = x.at[-1, :].set(x0)
    return x, u


def simulate(x0, f, u_seq, steps, dt):
    
    def step(x, t_u):
        t, u = t_u
        x_new = x + ( f(x) + u ) * dt
        return x_new, x_new

    Ts = jnp.arange(steps) * dt
    _, x_seq = lax.scan(step, x0, (Ts, u_seq))
    x_seq = jnp.vstack([x0, x_seq])[::-1]
    return x_seq