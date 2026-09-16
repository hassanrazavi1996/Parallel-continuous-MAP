from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
from cmap.clqt_jax import combine_abcej_forward


def test_combine_abcej_forward_Cij_infinite():
    # small, well-conditioned setup
    Aij = jnp.array([[2.0, 0.5], [-0.3, 1.0]])
    bij = jnp.array([0.2, -0.1])
    etaij = jnp.array([0.4, 0.6])
    Jij = jnp.array([[0.7, 0.0], [0.0, 0.9]])

    Ajk = jnp.array([[1.0, 0.1], [0.0, 1.2]])
    bjk = jnp.array([0.3, -0.2])
    etajk = jnp.array([0.5, 0.7])
    Jjk = jnp.array([[1.5, 0.0], [0.0, 1.1]])
    Cjk = jnp.array([[0.2, 0.0], [0.0, 0.3]])

    C_base = jnp.array([[1.0, 0.2], [0.2, 1.3]])
    alpha = 1e8
    Cij = alpha * C_base

    out = combine_abcej_forward(
        (Aij, bij, Cij, etaij, Jij), (Ajk, bjk, Cjk, etajk, Jjk)
    )

    Jjk_inv = jnp.linalg.inv(Jjk)
    Aik_exp = jnp.zeros_like(Aij)
    bik_exp = Ajk @ Jjk_inv @ etajk + bjk
    Cik_exp = Ajk @ Jjk_inv @ Ajk.T + Cjk
    etaik_exp = etaij
    Jik_exp = Jij

    for got, exp in zip(out, (Aik_exp, bik_exp, Cik_exp, etaik_exp, Jik_exp)):
        assert jnp.allclose(got, exp, atol=1e-7, rtol=1e-7)
