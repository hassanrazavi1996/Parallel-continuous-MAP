import jax
import jax.numpy as jnp
import math
import cmap.clqt_jax as cmap
from jax import random, lax
import numpy as np
from cmap.Linear_model_data import make_cv_data

from jax import config

config.update("jax_enable_x64", True)


###########################################################################
#
# Example linear model (of Wiener velocity type)
#
###########################################################################


def getCLQT(ocp: cmap):

    # ######################
    T = 50.0
    #########################
    q=0.2
    v=0.001
    W = lambda t: q * jnp.eye(2)
    H_rev = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = lambda t: 0.001 * jnp.eye(2)
    Sigma = lambda t: 0.01 * jnp.eye(4)

    F_rev = lambda t: -jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
])
    L_rev = lambda t: -jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = lambda t: L_rev(t) @ W(t) @ L_rev(t).T
    c_rev = lambda t: -jnp.zeros((4,))

    r_rev = lambda t: -jnp.zeros((2,))
    Q_rev = lambda t: Q(T - t)

    R_rev = lambda t: R(T - t)

    mu = jnp.array([5.0, 5.0, 0.0, 0.0])

    steps=5000
    dt=T/steps

    _, X, y, Y, y_rev = make_cv_data(mu, Sigma(0),steps,dt,q,v,seed=123)

    clqt = cmap.CLQT(mu, F_rev, Sigma, Q_rev, R_rev, c_rev, H_rev, y_rev, T)

    return clqt, mu
