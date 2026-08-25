from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from cmap.con_to_dis import f_convert

import numpy as np
import jax.random


def test_ConToDis():

    T = 50.0
    dt = 10e-5

    N = int(T / dt)
    d = 5

    mean = jnp.ones(d)
    cov = 10 * jnp.eye(d)

    key = jax.random.key(123)
    y_dis = jax.random.multivariate_normal(key, mean, cov, shape=(N,))
    y_dis_rev = y_dis[::-1]

    y_s = np.zeros((N, d))
    y_rev = np.zeros((N, d))

    k = 0
    for t in jnp.arange(0.0, T, dt):
        y_s[k] = f_convert(t, dt, N, y_dis)
        k = k + 1
    k = 0
    for t in jnp.arange(0.0, T, dt):
        y_rev[k] = f_convert(t, dt, N, y_dis_rev)
        k = k + 1

    assert jnp.allclose(y_s, y_rev[::-1])
    assert jnp.allclose(y_s, y_dis)
    assert jnp.allclose(y_rev, y_dis_rev)
