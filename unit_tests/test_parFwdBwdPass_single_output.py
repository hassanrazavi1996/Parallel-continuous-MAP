import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from cmap.clqt_jax import CLQT, parFwdBwdPass


def make_zero_ocp(n):
    F = lambda t: jnp.zeros((n, n))
    Q = lambda t: jnp.zeros((n, n))
    H = lambda t: jnp.zeros((1, n))
    R = lambda t: jnp.eye(1)
    c = lambda t: jnp.zeros((n,))
    r = lambda t: jnp.zeros((1,))
    y = lambda t: jnp.zeros((1,))
    ST = jnp.eye(n)
    vT = jnp.zeros((n,))
    T = 1.0
    return CLQT(vT=vT, F=F, ST=ST, Q=Q, R=R, c=c, H=H, y=y, r=r, T=T)


def test_parFwdBwdPass_single_step_output():
    n = 2
    m = 1
    blocks = 1
    steps = 1
    dt = 0.1
    t0 = 0.0

    ocp = make_zero_ocp(n)
    x0 = jnp.array([1.5, -0.5])
    S0 = jnp.zeros((n, n))

    steps_total = blocks * steps

    K = jnp.zeros((steps_total, m, n))
    d = jnp.ones((steps_total, m)) * 2.0

    S = jnp.zeros((steps_total + 1, n, n))
    v = jnp.zeros((steps_total + 1, n))

    u_out, x_out, As_all, bs_all, Cs_all = parFwdBwdPass(
        ocp=ocp,
        x0=x0,
        S0=S0,
        K=K,
        d=d,
        S=S,
        v=v,
        blocks=blocks,
        steps=steps,
        dt=dt,
        t0=t0,
    )

    expected_x = jnp.stack([x0, x0])
    expected_u = d
    expected_bs = jnp.stack([x0, x0])

    assert u_out.shape == expected_u.shape
    assert x_out.shape == expected_x.shape

    assert jnp.allclose(u_out, expected_u, atol=1e-12)
    assert jnp.allclose(x_out, expected_x, atol=1e-12)

    assert As_all.shape == (steps_total + 1, n, n)
    assert bs_all.shape == (steps_total + 1, n)
    assert Cs_all.shape == (steps_total + 1, n, n)

    assert jnp.allclose(As_all, jnp.zeros_like(As_all), atol=1e-12)
    assert jnp.allclose(bs_all, expected_bs, atol=1e-12)
    assert jnp.allclose(Cs_all, jnp.zeros_like(Cs_all), atol=1e-12)
