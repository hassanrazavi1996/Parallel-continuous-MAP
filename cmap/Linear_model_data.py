import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from scipy import signal


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
    tf=np.ceil(T[-1])

    DY_jax0 = jnp.array(DY) / dt
    # DY_jax = jnp.flip(DY_jax0, axis=0)
    def y(t):
        dt_sim = dt

        if jnp.array(t).ndim == 1:
            t_a = jnp.array(t)
            k = jnp.floor(t_a / dt_sim).astype(jnp.int32)
            k = jnp.clip(k, 0, nsteps - 1)
            y_vals = jax.vmap(
                lambda idx: jax.lax.dynamic_index_in_dim(DY_jax0, idx, keepdims=False))(k)

            y_vals = y_vals.T

            return y_vals.reshape(2, len(t))
        elif jnp.array(t).ndim == 0:
            t_a = jnp.array(t)

            k = jnp.floor(t_a / dt_sim).astype(jnp.int32)

            k = jnp.clip(k, 0, nsteps - 1)
            y_vals = DY_jax0[k]

            return y_vals.reshape(2,)
        else:
            t_a = jnp.asarray(t)

            k = jnp.floor(t_a / dt_sim).astype(jnp.int32)
            k = jnp.clip(k, 0, nsteps - 1).T

            y_vals = jnp.take(DY_jax0, k, axis=0)
            return y_vals.T

    def y_rev(t):
        dt_sim = dt
        if jnp.array(t).ndim == 1:
            t_a = jnp.array(t)
            k = jnp.ceil((tf-t_a) / dt_sim).astype(jnp.int32)
            k = jnp.clip(k, 0, nsteps - 1)

            y_vals = jax.vmap(
                lambda idx: jax.lax.dynamic_index_in_dim(DY_jax0, idx, keepdims=False))(k)

            y_vals = y_vals.T

            return y_vals.reshape(2, len(t))
        elif jnp.array(t).ndim == 0:
            t_a = jnp.array(t)

            k = jnp.ceil((tf-t_a) / dt_sim).astype(jnp.int32)

            k = jnp.clip(k, 0, nsteps - 1)
            y_vals = DY_jax0[k]

            return y_vals.reshape(2,)
        else:
            t_a = jnp.asarray(t)

            k = jnp.ceil((tf-t_a) / dt_sim).astype(jnp.int32)
            k = jnp.clip(k, 0, nsteps - 1).T

            y_vals = jnp.take(DY_jax0, k, axis=0)
            return y_vals.T

    return T, X, y, Y, y_rev
