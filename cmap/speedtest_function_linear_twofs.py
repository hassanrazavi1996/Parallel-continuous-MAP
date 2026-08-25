import jax
from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp



from cmap.clqt_jax import parBackwardPass
from cmap.clqt_jax import parFwdBwdPass



def clqt_par_speedtest_linear_tfs(clqt, blocks, steps, t0, dt):
    
    C_init=200
    (
        Kx_list_jax_par,
        d_list_jax_par,
        S_list_jax_par,
        v_list_jax_par,
    ) = parBackwardPass(clqt, blocks, steps, t0, dt)
    u_list_jax_par, x_list_jax_par = parFwdBwdPass(
        clqt, C_init, Kx_list_jax_par, d_list_jax_par,S_list_jax_par, v_list_jax_par ,blocks, steps, dt, t0
    )

    return u_list_jax_par, x_list_jax_par
