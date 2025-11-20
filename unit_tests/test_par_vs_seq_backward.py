from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.clqt_jax import CLQT, seqBackwardPass, parBackwardPass

def make_test_ocp():
    nx = 4
    nz=2
    q = 4.0
    v_meas = 1e-2
    W = lambda t: q * jnp.eye(2)
    H = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0]])
    R = lambda t: v_meas * jnp.eye(2)
    P0 = 0.01 * jnp.eye(nx)
    F = lambda t: jnp.array([[0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
    L = lambda t: jnp.array([[0.0, 0.0],
                             [0.0, 0.0],
                             [1.0, 0.0],
                             [0.0, 1.0]])
    Q = lambda t: L(t) @ W(t) @ L(t).T
    c = lambda t: jnp.zeros((nx,))
    r = lambda t: jnp.zeros((nz,))
    y = lambda t: jnp.zeros((nz,))
    ST = jnp.eye(nx)
    vT = jnp.zeros((nx,))
    T = 0.025
    return CLQT(vT, F, ST, Q, R, c, H, y, r, T)

def test_seq_vs_par_backward_pass_equal():
    ocp = make_test_ocp()
    steps = 17
    blocks = 300
    steps_all = steps * blocks
    dt = ocp.T / steps_all
    t0 = 0.0

    S_seq, v_seq, Kx_seq, d_seq = seqBackwardPass(ocp, steps_all, dt, t0, ocp.ST, ocp.vT)

    Kx_par, d_par, S_par, v_par = parBackwardPass(ocp, blocks, steps, t0, dt)

    assert S_seq.shape == S_par.shape, f"S shape mismatch {S_seq.shape} != {S_par.shape}"
    assert v_seq.shape == v_par.shape, f"v shape mismatch {v_seq.shape} != {v_par.shape}"
    assert Kx_seq.shape == Kx_par.shape, f"Kx shape mismatch {Kx_seq.shape} != {Kx_par.shape}"
    assert d_seq.shape == d_par.shape, f"d shape mismatch {d_seq.shape} != {d_par.shape}"

    assert jnp.allclose(S_seq, S_par, atol=1e-5)
    assert jnp.allclose(v_seq, v_par, atol=1e-5)
    assert jnp.allclose(Kx_seq, Kx_par, atol=1e-5)
    assert jnp.allclose(d_seq, d_par, atol=1e-5)