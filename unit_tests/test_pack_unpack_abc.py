from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.clqt_jax import pack_abc, unpack_abc


def make_demo(n):
    A = jnp.arange(n * n, dtype=jnp.float64).reshape(n, n) + 1.0
    C = jnp.arange(n * n, dtype=jnp.float64).reshape(n, n) + 10.0
    b = jnp.arange(n, dtype=jnp.float64) + 100.0
    return A, b, C


def test_pack_unpack_abc_scalar():
    n = 4
    A, b, C = make_demo(n)

    packed = pack_abc(A, b, C)
    assert packed.shape == (n, 2 * n + 1)

    A2, b2, C2 = unpack_abc(packed)

    assert A2.shape == A.shape
    assert C2.shape == C.shape
    assert b2.shape == b.shape

    assert jnp.allclose(A2, A)
    assert jnp.allclose(b2, b)
    assert jnp.allclose(C2, C)


def test_pack_unpack_abc_batched():
    n = 3
    B = 2
    A, b, C = make_demo(n)
    As = jnp.stack([A, A + 1.0])
    bs = jnp.stack([b, b + 2.0])
    Cs = jnp.stack([C, C + 3.0])

    packed = pack_abc(As, bs, Cs)
    assert packed.shape == (B, n, 2 * n + 1)

    A2, b2, C2 = unpack_abc(packed)

    assert A2.shape == As.shape
    assert C2.shape == Cs.shape
    assert b2.shape == bs.shape

    assert jnp.allclose(A2, As)
    assert jnp.allclose(b2, bs)
    assert jnp.allclose(C2, Cs)
