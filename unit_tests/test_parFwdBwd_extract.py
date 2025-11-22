import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from cmap.clqt_jax import CLQT, parFwdBwdPass_extract


def make_trivial_ocp(n):
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


def test_parFwdBwdPass_extract_trivial():
    n = 4
    m = 2
    blocks = 2
    steps = 3
    dt = 0.1
    t0 = 0.0

    ocp = make_trivial_ocp(n)

    steps_total = blocks * steps

    As = jnp.stack([jnp.eye(n) for _ in range(blocks + 1)])
    bs = jnp.stack([jnp.zeros((n,)) for _ in range(blocks + 1)])
    Cs = jnp.stack([jnp.zeros((n, n)) for _ in range(blocks + 1)])
    etas = jnp.stack([jnp.zeros((n,)) for _ in range(blocks + 1)])
    Js = jnp.stack([jnp.zeros((n, n)) for _ in range(blocks + 1)])
    elems = (As, bs, Cs, etas, Js)

    S = jnp.stack([jnp.zeros((n, n)) for _ in range(steps_total + 1)])
    v = jnp.stack([jnp.zeros((n,)) for _ in range(steps_total + 1)])

    K = jnp.zeros((steps_total, m, n))
    d = jnp.ones((steps_total, m)) * 2.0

    u_out, x_out, As_all, bs_all, Cs_all = parFwdBwdPass_extract(
        ocp=ocp, K=K, d=d, S=S, v=v, elems=elems, steps=steps, dt=dt, t0=t0
    )

    expected_x = jnp.zeros((steps_total + 1, n))
    expected_u = d

    assert u_out.shape == expected_u.shape
    assert x_out.shape == expected_x.shape

    assert jnp.allclose(u_out, expected_u, atol=1e-12)
    assert jnp.allclose(x_out, expected_x, atol=1e-12)

    assert As_all.shape == (steps_total + 1, n, n)
    assert bs_all.shape == (steps_total + 1, n)
    assert Cs_all.shape == (steps_total + 1, n, n)

    I = jnp.eye(n)
    expected_As = jnp.stack([I for _ in range(steps_total + 1)])
    expected_bs = jnp.stack([jnp.zeros((n,)) for _ in range(steps_total + 1)])
    expected_Cs = jnp.stack([jnp.zeros((n, n)) for _ in range(steps_total + 1)])

    assert jnp.allclose(As_all, expected_As, atol=1e-12), "As_all mismatch"
    assert jnp.allclose(bs_all, expected_bs, atol=1e-12), "bs_all mismatch"
    assert jnp.allclose(Cs_all, expected_Cs, atol=1e-12), "Cs_all mismatch"
