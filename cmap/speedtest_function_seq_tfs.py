import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp


from cmap.clqt_jax import seqBackwardPass
from cmap.clqt_jax import seqFwdBwdPass
from cmap.clqt_jax import combine_seqFwdBwdPass

def clqt_seq_speedtest_linear_tfs_seq(clqt, steps, t0, dt, sigma, mu, diffeq_solver):
    A_init_m=jnp.zeros_like(sigma)
    C_init_m=500*jnp.eye(mu.shape[0])
    b_init_m=jnp.zeros(mu.shape[0])

    S_list_jax_seq, v_list_jax_seq, Kx_list_jax_seq, d_list_jax_seq = seqBackwardPass(
        clqt, steps, dt, t0, sigma, mu
    )

    A_seq,b_seq,C_seq=seqFwdBwdPass(clqt,steps,dt,t0,A_init_m,b_init_m,C_init_m,diffeq_solver)
    u_fwdbwd_seq,x_fwdbwd_seq=combine_seqFwdBwdPass(S_list_jax_seq, Kx_list_jax_seq, v_list_jax_seq, d_list_jax_seq, A_seq, b_seq, C_seq)

    return x_fwdbwd_seq, u_fwdbwd_seq
