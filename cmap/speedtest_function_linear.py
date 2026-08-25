import jax
from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp


from cmap.clqt_jax import seqBackwardPass
from cmap.clqt_jax import seqForwardPass
from cmap.clqt_jax import parBackwardPass
from cmap.clqt_jax import parForwardPass


def clqt_seq_speedtest_linear(clqt, steps, t0, dt, sigma, mu,diffeq_solver):
    S_list_jax_seq, v_list_jax_seq, Kx_list_jax_seq, d_list_jax_seq = seqBackwardPass(
        clqt, steps, dt, t0, sigma, mu, diffeq_solver
    )
    phi0 = jnp.linalg.solve(S_list_jax_seq[0], v_list_jax_seq[0])
    x_list_jax_seq, u_list_jax_seq = seqForwardPass(
        clqt, dt, t0, phi0, Kx_list_jax_seq, d_list_jax_seq, diffeq_solver, u_zoh=False
    )

    return x_list_jax_seq, u_list_jax_seq


def clqt_par_speedtest_linear(clqt, blocks, steps, t0, dt, diffeq_solver):
    (
        Kx_list_jax_par,
        d_list_jax_par,
        S_list_jax_par,
        v_list_jax_par,
    ) = parBackwardPass(clqt, blocks, steps, t0, dt, diffeq_solver)
    phi0 = jnp.linalg.solve(S_list_jax_par[0], v_list_jax_par[0])
    x_list_jax_par, u_list_jax_par = parForwardPass(
        clqt, phi0, Kx_list_jax_par, d_list_jax_par, blocks, steps, dt, t0, diffeq_solver, u_zoh=False
    )

    return x_list_jax_par, u_list_jax_par
