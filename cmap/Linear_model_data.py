from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp


def make_cv_data(m0, P0, nsteps=5000, dt=0.01, q=0.2, r=0.01, seed=None):

    rng = np.random.default_rng(seed)

    F = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    L = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    T = np.zeros(nsteps)
    X = np.zeros((nsteps, 4))
    Y = np.zeros((nsteps, 2))
    DY = np.zeros((nsteps, 2))
    X[0] = m0

    x = rng.multivariate_normal(m0, P0)
    y = np.zeros(2)
    t = 0.0

    for k in range(0, nsteps):

        dw = rng.standard_normal(2) * np.sqrt(dt)
        dx = (F @ x) * dt + (L @ (np.sqrt(q) * dw))
        x = x + dx

        dv = rng.standard_normal(2) * np.sqrt(r * dt)
        dy = (H @ x) * dt + dv
        y = y + dy
        t += dt
        if k == nsteps - 1:
            DY[k] = dy
            Y[k] = y
        else:
            X[k + 1] = x
            DY[k] = dy
            Y[k] = y
            T[k + 1] = t

    DY_jax0 = jnp.array(DY,dtype=jnp.float64) / dt
    DY_jax1 = jnp.flip(DY_jax0, axis=0)
    return X,DY_jax0,DY_jax1

