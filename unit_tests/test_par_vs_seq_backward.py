from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.clqt_jax import CLQT, seqBackwardPass, parBackwardPass

import jax
def make_test_ocp():
    nx = 4
    nz = 2
    q = 4.0
    v_meas = 1e-2
    W = lambda t: q * jnp.eye(2)
    H = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = lambda t: v_meas * jnp.eye(2)
    P0 = 0.01 * jnp.eye(nx)
    F = lambda t: jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    L = lambda t: jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = lambda t: L(t) @ W(t) @ L(t).T
    c = lambda t: jnp.zeros((nx,))
    r = lambda t: jnp.zeros((nz,))

    cx = jnp.array(
        [
            5.7770,
            -2.6692,
            1.1187,
            0.1379,
            0.5718,
            1.1214,
            0.2998,
            0.3325,
            0.7451,
            0.2117,
            0.6595,
            0.0401,
            -0.2995,
        ]
    )
    cy = jnp.array(
        [
            4.3266,
            -1.4584,
            -1.2457,
            1.1804,
            0.2035,
            0.5123,
            1.0588,
            0.2616,
            -0.6286,
            -0.3802,
            0.2750,
            -0.0070,
            -0.0022,
        ]
    )

    y = lambda t: jnp.array(
        [
            cx[0]
            + cx[1] * jnp.cos(2.0 * jnp.pi * t / 50.0)
            + cx[2] * jnp.sin(2.0 * jnp.pi * t / 50.0)
            + cx[3] * jnp.cos(4.0 * jnp.pi * t / 50.0)
            + cx[4] * jnp.sin(4.0 * jnp.pi * t / 50.0)
            + cx[5] * jnp.cos(6.0 * jnp.pi * t / 50.0)
            + cx[6] * jnp.sin(6.0 * jnp.pi * t / 50.0)
            + cx[7] * jnp.cos(8.0 * jnp.pi * t / 50.0)
            + cx[8] * jnp.sin(8.0 * jnp.pi * t / 50.0)
            + cx[9] * jnp.cos(10.0 * jnp.pi * t / 50.0)
            + cx[10] * jnp.sin(10.0 * jnp.pi * t / 50.0)
            + cx[11] * jnp.cos(12.0 * jnp.pi * t / 50.0)
            + cx[12] * jnp.sin(12.0 * jnp.pi * t / 50.0),
            cy[0]
            + cy[1] * jnp.cos(2.0 * jnp.pi * t / 50.0)
            + cy[2] * jnp.sin(2.0 * jnp.pi * t / 50.0)
            + cy[3] * jnp.cos(4.0 * jnp.pi * t / 50.0)
            + cy[4] * jnp.sin(4.0 * jnp.pi * t / 50.0)
            + cy[5] * jnp.cos(6.0 * jnp.pi * t / 50.0)
            + cy[6] * jnp.sin(6.0 * jnp.pi * t / 50.0)
            + cy[7] * jnp.cos(8.0 * jnp.pi * t / 50.0)
            + cy[8] * jnp.sin(8.0 * jnp.pi * t / 50.0)
            + cy[9] * jnp.cos(10.0 * jnp.pi * t / 50.0)
            + cy[10] * jnp.sin(10.0 * jnp.pi * t / 50.0)
            + cy[11] * jnp.cos(12.0 * jnp.pi * t / 50.0)
            + cy[12] * jnp.sin(12.0 * jnp.pi * t / 50.0),
        ]
    )
    ST = jnp.eye(nx)
    vT = jnp.zeros((nx,))
    T = 0.025
    return CLQT(vT, F, ST, Q, R, c, H, y, r, T)


def test_seq_vs_par_backward_pass_equal():
    ocp = make_test_ocp()
    steps = 17
    blocks = 3000
    steps_all = steps * blocks
    dt = ocp.T / steps_all
    t0 = 0.0

    S_seq, v_seq, Kx_seq, d_seq = seqBackwardPass(
        ocp, steps_all, dt, t0, ocp.ST, ocp.vT
    )

    Kx_par, d_par, S_par, v_par = parBackwardPass(ocp, blocks, steps, t0, dt)

    assert (
        S_seq.shape == S_par.shape
    ), f"S shape mismatch {S_seq.shape} != {S_par.shape}"
    assert (
        v_seq.shape == v_par.shape
    ), f"v shape mismatch {v_seq.shape} != {v_par.shape}"
    assert (
        Kx_seq.shape == Kx_par.shape
    ), f"Kx shape mismatch {Kx_seq.shape} != {Kx_par.shape}"
    assert (
        d_seq.shape == d_par.shape
    ), f"d shape mismatch {d_seq.shape} != {d_par.shape}"

    
    assert jnp.allclose(S_seq, S_par, atol=1e-9)
    assert jnp.allclose(v_seq, v_par, atol=1e-9)
    assert jnp.allclose(Kx_seq, Kx_par, atol=1e-9)
    assert jnp.allclose(d_seq, d_par, atol=1e-9)
