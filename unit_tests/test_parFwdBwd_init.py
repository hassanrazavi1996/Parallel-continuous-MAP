import jax.numpy as jnp
from cmap.clqt_jax import CLQT, parFwdBwd_init

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

def test_parFwdBwd_init_trivial_system():
    n = 3
    blocks = 2
    steps = 4
    dt = 0.1
    t0 = 0.0
    ocp = make_zero_ocp(n)

    As, bs, Cs, etas, Js = parFwdBwd_init(ocp, blocks, steps, t0, dt,diffeq_solver='heun')

    assert As.shape == (blocks + 1, n, n)
    assert bs.shape == (blocks + 1, n)
    assert Cs.shape == (blocks + 1, n, n)
    assert etas.shape == (blocks + 1, n)
    assert Js.shape == (blocks + 1, n, n)

    I = jnp.eye(n)
    Zm = jnp.zeros((n, n))
    zv = jnp.zeros((n,))

    assert jnp.allclose(As[:-1], jnp.stack([I] * blocks))
    assert jnp.allclose(bs[:-1], jnp.stack([zv] * blocks))
    assert jnp.allclose(Cs[:-1], jnp.stack([Zm] * blocks))
    assert jnp.allclose(etas[:-1], jnp.stack([zv] * blocks))
    assert jnp.allclose(Js[:-1], jnp.stack([Zm] * blocks))

    assert jnp.allclose(As[-1], Zm)      
    assert jnp.allclose(bs[-1], zv)
    assert jnp.allclose(Cs[-1], Zm)
    assert jnp.allclose(etas[-1], ocp.vT)
    assert jnp.allclose(Js[-1], ocp.ST)
