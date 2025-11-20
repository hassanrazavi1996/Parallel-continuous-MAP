from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.clqt_jax import CLQT, seqBackwardPass, parBackwardPass, seqForwardPass, parForwardPass

def make_test_ocp():
    nx = 4
    nz = 2
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

def test_seq_vs_par_forward_pass_equal():
    ocp = make_test_ocp()
    steps = 17
    blocks = 300
    steps_all = steps * blocks
    dt = ocp.T / steps_all
    t0 = 0.0

    S_seq, v_seq, Kx_seq, d_seq = seqBackwardPass(ocp, steps_all, dt, t0, ocp.ST, ocp.vT)
    Kx_par, d_par, S_par, v_par = parBackwardPass(ocp, blocks, steps, t0, dt)

    assert Kx_seq.shape == Kx_par.shape
    assert d_seq.shape == d_par.shape

    phi0 = jnp.linalg.solve(S_seq[0], v_seq[0])

    x_seq, u_seq = seqForwardPass(ocp, dt, t0, phi0, Kx_seq, d_seq, u_zoh=False)

    u_par, x_par = parForwardPass(ocp, phi0, Kx_par, d_par, blocks, steps, dt, t0, u_zoh=False)

    assert x_seq.shape == x_par.shape, f"x shape mismatch {x_seq.shape} vs {x_par.shape}"
    assert u_seq.shape == u_par.shape, f"u shape mismatch {u_seq.shape} vs {u_par.shape}"

    assert jnp.allclose(x_seq, x_par, atol=1e-7, rtol=1e-7)
    assert jnp.allclose(u_seq, u_par, atol=1e-7, rtol=1e-7)