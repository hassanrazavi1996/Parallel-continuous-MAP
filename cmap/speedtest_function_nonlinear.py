import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax

from cmap.nonlinear_iclqt import seq_iterate
from cmap.nonlinear_iclqt import par_iterate


def clqt_seq_speedtest_nonlinear(clqt, steps, blocks, f, h, x, t0, niter, diffeq_solver):

    def step(carry, _):
        x = carry

        x_seq = seq_iterate(clqt, steps, blocks, f, h, x, t0,diffeq_solver)

        x_seq_new = x_seq

        return (x_seq_new), (x_seq_new)

    init_carry = x
    _, (x_seq_final) = lax.scan(step, init_carry, None, length=niter)

    return x_seq_final


def clqt_par_speedtest_nonlinear(clqt, steps, blocks, f, h, x, t0, niter, diffeq_solver):

    def step(carry, _):

        x = carry

        x_par = par_iterate(clqt, steps, blocks, f, h, x, t0, diffeq_solver)

        x_par_new = x_par

        return (x_par_new), (x_par_new)

    init_carry = x
    _, (x_par_final) = lax.scan(step, init_carry, None, length=niter)

    return x_par_final
