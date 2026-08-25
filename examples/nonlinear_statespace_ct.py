from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp


f = lambda x: jnp.array(
    [
        x[2],  # dp_x/dt = vx
        x[3],  # dp_y/dt = vy
        -x[4] * x[3],  # dv_x/dt = -omega * vy
        x[4] * x[2],  # dv_y/dt =  omega * vx
        0.0,  # domega/dt = 0
    ],
    dtype=jnp.float64,
)

h = lambda x: jnp.array(
    [jnp.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-8), jnp.arctan2(x[1], x[0])]
)
