from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.clqt_jax import pack_abcej, unpack_abcej

def make_demo(n):
    A = jnp.arange(n * n, dtype=jnp.float64).reshape(n, n) + 1.0
    C = jnp.arange(n * n, dtype=jnp.float64).reshape(n, n) + 10.0
    J = jnp.arange(n * n, dtype=jnp.float64).reshape(n, n) + 20.0
    b = jnp.arange(n, dtype=jnp.float64) + 100.0
    eta = jnp.arange(n, dtype=jnp.float64) + 200.0
    return A, b, C, eta, J

def test_pack_unpack_abcej_scalar():
    n = 4
    A, b, C, eta, J = make_demo(n)

    packed = pack_abcej(A, b, C, eta, J)
    assert packed.shape == (n, 3 * n + 2)

    A2, b2, C2, eta2, J2 = unpack_abcej(packed)

    assert A2.shape == A.shape
    assert C2.shape == C.shape
    assert J2.shape == J.shape
    assert b2.shape == b.shape
    assert eta2.shape == eta.shape

    assert jnp.allclose(A2, A)
    assert jnp.allclose(b2, b)
    assert jnp.allclose(C2, C)
    assert jnp.allclose(eta2, eta)
    assert jnp.allclose(J2, J)

def test_pack_unpack_abcej_batched():
    n = 3
    B = 2
    A, b, C, eta, J = make_demo(n)
    As = jnp.stack([A, A + 1.0])
    bs = jnp.stack([b, b + 2.0])
    Cs = jnp.stack([C, C + 3.0])
    etas = jnp.stack([eta, eta + 4.0])
    Js = jnp.stack([J, J + 5.0])

    packed = pack_abcej(As, bs, Cs, etas, Js)
    assert packed.shape == (B, n, 3 * n + 2)

    A2, b2, C2, eta2, J2 = unpack_abcej(packed)

    assert A2.shape == As.shape
    assert C2.shape == Cs.shape
    assert J2.shape == Js.shape
    assert b2.shape == bs.shape
    assert eta2.shape == etas.shape

    assert jnp.allclose(A2, As)
    assert jnp.allclose(b2, bs)
    assert jnp.allclose(C2, Cs)
    assert jnp.allclose(eta2, etas)
    assert jnp.allclose(J2, Js)