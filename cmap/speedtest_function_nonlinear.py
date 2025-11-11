import jax
from jax import config
config.update("jax_enable_x64",True)

import jax.numpy as jnp
from jax import lax

from cmap.nonlinear_iclqt import seq_iterate
from cmap.nonlinear_iclqt import par_iterate


def clqt_seq_speedtest_nonlinear(clqt,steps,blocks,f,h,u,x,t0,niter):
    
    def step(carry,_):
        u, x = carry
        
        u_seq, x_seq = seq_iterate(clqt, steps, blocks, f, h, u, x, t0)

        u_seq_new = u_seq
        x_seq_new = x_seq

        return (u_seq_new, x_seq_new), (u_seq_new, x_seq_new)

    init_carry = (u, x)
    _, (u_seq_final, x_seq_final) = lax.scan(step, init_carry,None, length=niter)

    return u_seq_final, x_seq_final


def clqt_par_speedtest_nonlinear(clqt,steps,blocks,f,h,u,x,t0,niter):

    def step(carry,_):

        u, x = carry
        
        u_par, x_par = par_iterate(clqt, steps, blocks, f, h, u, x, t0)

        u_par_new = u_par
        x_par_new = x_par

        return (u_par_new, x_par_new), (u_par_new, x_par_new)

    init_carry = (u, x)
    _, (u_par_final, x_par_final) = lax.scan(step, init_carry,None, length=niter)

    return u_par_final , x_par_final
