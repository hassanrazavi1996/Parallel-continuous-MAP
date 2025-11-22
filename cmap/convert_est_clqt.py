from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
from cmap.linear_estimation_problem import Estimation
from cmap.con_to_dis import f_convert


def est_to_clqt(est: Estimation, steps_all):

    T = est.T
    F = lambda t: -est.F(T - t)
    H = lambda t: est.H(T - t)
    c = lambda t: -est.c(T - t)
    r = lambda t: est.r(T - t)
    Q_est = lambda t: est.L(t) @ est.W(t) @ est.L(t).T
    Q = lambda t: Q_est(T - t)
    R = lambda t: est.R(t)
    y = est.y

    y_rev = lambda t: f_convert(t, T / steps_all, steps_all, y[::-1])

    ST = jnp.linalg.solve(est.P0, jnp.eye(len(est.x0)))
    vT = jnp.linalg.solve(est.P0, est.x0)

    return F, H, c, r, Q, R, T, ST, vT, y_rev
