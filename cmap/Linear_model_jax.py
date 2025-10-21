from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import cmap.clqt_jax as cmap
from cmap.Linear_model_data import make_cv_data
from cmap.con_to_dis import y_reverse


###########################################################################
#
# Example linear model (of Wiener velocity type)
#
###########################################################################


def getCLQT(ocp: cmap, steps):

    # ######################
    T = 50.0
    #########################
    q = 0.2
    v = 0.001
    p0 = 0.01
    W = lambda t: q * jnp.eye(2)
    H_rev = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = lambda t: v * jnp.eye(2)
    P0 = lambda t: p0 * jnp.eye(4)

    F_rev = lambda t: -jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    L_rev = lambda t: -jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = lambda t: L_rev(t) @ W(t) @ L_rev(t).T

    c_rev = lambda t: -jnp.zeros((4,))

    r_rev = lambda t: -jnp.zeros((2,))
    Q_rev = lambda t: Q(T - t)

    R_rev = lambda t: R(T - t)

    x0 = jnp.array([5.0, 5.0, 0.0, 0.0])

    mu = jnp.linalg.solve(P0(0), x0)

    dt = T / steps
    Sigma = lambda t: (1 / p0) * jnp.eye(4)

    X, y_discrete, y_discrete_rev = make_cv_data(x0, P0(0), steps, dt, q, v, seed=123)
    y_rev = lambda t: y_reverse(t, T, dt, steps, y_discrete)
    clqt = cmap.CLQT(
        mu, F_rev, Sigma, Q_rev, R_rev, c_rev, H_rev, y_rev, r_rev, T, L_rev, W
    )

    return clqt, x0, P0, q, v
