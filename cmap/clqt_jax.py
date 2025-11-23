import jax
from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap
from cmap.diffeq_jax import euler
from typing import NamedTuple
from typing import Callable


def pack_abcej(A, b, C, eta, J):
    b = jnp.expand_dims(b, axis=-1)
    eta = jnp.expand_dims(eta, axis=-1)
    return jnp.concatenate([A, C, J, b, eta], axis=-1)


def unpack_abcej(x):
    n = x.shape[-2]
    A = x[..., :n]
    C = x[..., n : (2 * n)]
    J = x[..., (2 * n) : (3 * n)]
    b = x[..., 3 * n]
    eta = x[..., 3 * n + 1]
    return A, b, C, eta, J


def pack_Psiphi(Psi, phi):
    phi = jnp.expand_dims(phi, axis=-1)
    return jnp.concatenate((Psi, phi), axis=-1)


def unpack_Psiphi(x):
    n = x.shape[-2]
    Psi = x[..., :n]
    phi = x[..., n]
    return Psi, phi


def pack_abc(A, b, C):
    b = jnp.expand_dims(b, axis=-1)
    return jnp.concatenate((A, C, b), axis=-1)


def unpack_abc(x):
    n = x.shape[-2]
    A = x[..., :n]
    C = x[..., n : (2 * n)]
    b = x[..., 2 * n]
    return A, b, C


def pack_Sv(S, v):
    v = jnp.expand_dims(v, axis=-1)
    return jnp.concatenate((S, v), axis=-1)


def unpack_Sv(x):
    n = x.shape[-2]
    S = x[..., :n]
    v = x[..., n]
    return S, v


##############################################################################
# Continuous-time Sequential / parallel LQT
##############################################################################


class CLQT(NamedTuple):

    vT: jnp.ndarray
    F: Callable
    ST: jnp.ndarray
    Q: Callable
    R: Callable
    c: Callable
    H: Callable
    y: Callable
    r: Callable
    T: float


###########################################################################
# Sequential computation of gains, value functions, states, and controls
###########################################################################


def riccati_ode_f(ocp: CLQT, x, t):

    S, v = unpack_Sv(x)

    Q = ocp.Q(t)
    F = ocp.F(t)
    H = ocp.H(t)
    R = ocp.R(t)
    c = ocp.c(t)
    r = ocp.r(t)
    y = ocp.y(t)

    I = jnp.eye(R.shape[0])
    R_inv = jlinalg.solve(R, I)

    dS = -F.T @ S - S @ F + S @ Q @ S - H.T @ R_inv @ H
    dv = -H.T @ R_inv @ y + H.T @ R_inv @ r - F.T @ v + S @ Q @ v + S @ c

    dx = pack_Sv(dS, dv)

    return dx


def seqBackwardPass(ocp: CLQT, steps, dt, t0, S, v):

    Q = ocp.Q(0)
    Kx = jnp.zeros((Q.shape[-1], Q.shape[-2]))
    d = jnp.zeros((Q.shape[-1],))
    Ts = dt * jnp.arange(steps)

    def step(carry, t):

        f = lambda x, t: riccati_ode_f(ocp, x, t)

        S, v, _, _ = carry

        x = pack_Sv(S, v)
        x = euler(f, -dt, x, t + dt)
        S, v = unpack_Sv(x)

        S = 0.5 * (S + S.T)
        Q = ocp.Q(t)

        Kx = Q @ S
        d = Q @ v

        return (S, v, Kx, d), (S, v, Kx, d)

    _, (Ss, vs, Kxs, ds) = lax.scan(
        f=step, init=(S, v, Kx, d), xs=(t0 + Ts), reverse=True
    )

    Ss = jnp.concatenate([Ss, S[None, ...]], axis=0)
    vs = jnp.concatenate([vs, v[None, ...]], axis=0)

    return Ss, vs, Kxs, ds


def seqForwardPass(ocp: CLQT, dt, t_start, x0, Kx_block, d_block, u_zoh=False):

    steps = Kx_block.shape[0]
    times = t_start + dt * jnp.arange(steps)

    def step(x, t_Kd):

        t, Kx, d = t_Kd
        u = -Kx @ x + d

        def dynamics_zoh(x, t):
            return ocp.F(t) @ x + u + ocp.c(t)

        def dynamics_no_zoh(x, t):
            return ocp.F(t) @ x + (-Kx @ x + d) + ocp.c(t)

        f = lambda x, t: lax.cond(
            u_zoh, lambda _: dynamics_zoh(x, t), lambda _: dynamics_no_zoh(x, t), None
        )

        x_next = euler(f, dt, x, t)
        return x_next, (u, x_next)

    scan_inputs = (times, Kx_block, d_block)
    _, (us, xs_next) = lax.scan(step, x0, scan_inputs, reverse=False)
    xs = jnp.concatenate([x0[None, :], xs_next], axis=0)

    return xs, us


def FwdBwdPass_odes(ocp: CLQT, x, t):

    A, b, C = unpack_abc(x)

    F = ocp.F(t)
    H = ocp.H(t)
    R = ocp.R(t)
    Q = ocp.Q(t)
    r = ocp.r(t)
    c = ocp.c(t)
    y = ocp.y(t)

    I = jnp.eye(R.shape[0])
    R_inv = jlinalg.solve(R, I)

    dA = F @ A - C @ H.T @ R_inv @ H @ A
    db = C @ H.T @ R_inv @ (y - r) - C @ H.T @ R_inv @ H @ b + F @ b + c
    dC = -C @ H.T @ R_inv @ H @ C + Q + F @ C + C @ F.T

    dx = pack_abc(dA, db, dC)

    return dx


def seqFwdBwdPass(ocp: CLQT, steps, dt, t0, A0, b0, C0):

    def step(carry, t):

        A, b, C = carry
        f = lambda x, t: FwdBwdPass_odes(ocp, x, t)

        x = pack_abc(A, b, C)
        x = euler(f, dt, x, t)
        A, b, C = unpack_abc(x)

        C = 0.5 * (C + C.T)

        return (A, b, C), (A, b, C)

    Ts = dt * jnp.arange(steps)
    _, (As, bs, Cs) = lax.scan(f=step, init=(A0, b0, C0), xs=(t0 + Ts),reverse=False)

    As = jnp.concatenate([A0[None, ...], As], axis=0)
    bs = jnp.concatenate([b0[None, ...], bs], axis=0)
    Cs = jnp.concatenate([C0[None, ...], Cs], axis=0)

    return As, bs, Cs


def combine_seqFwdBwdPass(S, K, v, d, A, b, C):

    d_x = C.shape[-1]
    I = jnp.eye(d_x)

    def solve_x(S_i, v_i, b_i, C_i):

        x = jlinalg.solve(I + C_i @ S_i, b_i + C_i @ v_i)
        return x

    xs = vmap(solve_x)(S, v, b, C)
    us = -jnp.einsum("bij,bj->bi", K, xs[:-1]) + d

    return us, xs


###########################################################################
# Parallel computation of gains and value functions backwards
############################################################################


def bwpass_bw_ode_f(ocp: CLQT, x, t):

    A, b, C, eta, J = unpack_abcej(x)

    F = ocp.F(t)
    H = ocp.H(t)
    y = ocp.y(t)
    c = ocp.c(t)
    r = ocp.r(t)
    R = ocp.R(t)
    Q = ocp.Q(t)

    I = jnp.eye(R.shape[0])
    R_inv = jlinalg.solve(R, I)

    dA = A @ Q @ J - A @ F
    db = -A @ Q @ eta - A @ c
    dC = -A @ Q @ A.T
    deta = -H.T @ R_inv @ y + H.T @ R_inv @ r + J @ Q @ eta - F.T @ eta + J @ c
    dJ = -H.T @ R_inv @ H + J @ Q @ J - J @ F - F.T @ J

    dx = pack_abcej(dA, db, dC, deta, dJ)

    return dx


def parBackwardPass_init(ocp: CLQT, blocks, steps, t0, dt):

    elems = []

    dim = ocp.ST.shape[0]

    A0 = jnp.eye(dim)
    b0 = jnp.zeros((dim,))
    C0 = jnp.zeros((dim, dim))
    eta0 = jnp.zeros((dim,))
    J0 = jnp.zeros((dim, dim))

    Ts = jnp.arange(steps) * dt

    def step_backward(carry, t):

        f = lambda x, t: bwpass_bw_ode_f(ocp, x, t)

        A, b, C, eta, J = carry

        x = pack_abcej(A, b, C, eta, J)
        x = euler(f, -dt, x, t + dt)
        A, b, C, eta, J = unpack_abcej(x)

        C = 0.5 * (C + C.T)
        J = 0.5 * (J + J.T)

        return (A, b, C, eta, J), (A, b, C, eta, J)

    def single_pass(A0, b0, C0, eta0, J0, _t0):

        _, (As, bs, Cs, etas, Js) = lax.scan(
            f=step_backward, init=(A0, b0, C0, eta0, J0), xs=(_t0 + Ts), reverse=True
        )
        return As[0], bs[0], Cs[0], etas[0], Js[0]

    t0s = t0 + jnp.arange(blocks) * steps * dt

    (A_blocks, b_blocks, C_blocks, eta_blocks, J_blocks) = vmap(
        single_pass, in_axes=(None, None, None, None, None, 0)
    )(A0, b0, C0, eta0, J0, t0s)

    AT = jnp.zeros_like(ocp.ST)
    bT = b0
    CT = C0
    etaT = ocp.vT
    JT = ocp.ST

    As = jnp.concatenate([A_blocks, AT[None]], axis=0)
    bs = jnp.concatenate([b_blocks, bT[None]], axis=0)
    Cs = jnp.concatenate([C_blocks, CT[None]], axis=0)
    etas = jnp.concatenate([eta_blocks, etaT[None]], axis=0)
    Js = jnp.concatenate([J_blocks, JT[None]], axis=0)

    elems = (As, bs, Cs, etas, Js)

    return elems


def parBackwardPass_extract(ocp: CLQT, elems, steps, dt, t0):

    As, bs, Cs, etas, Js = elems
    blocks = Js.shape[0] - 1

    t0s = t0 + jnp.arange(blocks) * steps * dt

    J_blocks = Js[1:]
    eta_blocks = etas[1:]

    (Ss, vs, Kxs, ds) = vmap(seqBackwardPass, in_axes=(None, None, None, 0, 0, 0))(
        ocp, steps, dt, t0s, J_blocks, eta_blocks
    )

    Ss = Ss[:, 1:, :, :]
    vs = vs[:, 1:, :]

    Ss = Ss.reshape((-1,) + Ss.shape[-2:])
    vs = vs.reshape((-1,) + vs.shape[-1:])
    Kxs = Kxs.reshape((-1,) + Kxs.shape[-2:])
    ds = ds.reshape((-1,) + ds.shape[-1:])

    S0 = Js[0]
    v0 = etas[0]

    Ss = jnp.concatenate([S0[None], Ss], axis=0)
    vs = jnp.concatenate([v0[None], vs], axis=0)

    return Kxs, ds, Ss, vs


def combine_abcej(elem1, elem2):

    Ajk, bjk, Cjk, etajk, Jjk = elem1
    Aij, bij, Cij, etaij, Jij = elem2

    I = jnp.eye(Aij.shape[0])
    Aik = jnp.dot(Ajk, jlinalg.solve(I + jnp.dot(Cij, Jjk), Aij))
    bik = (
        jnp.dot(Ajk, jlinalg.solve(I + jnp.dot(Cij, Jjk), bij + jnp.dot(Cij, etajk)))
        + bjk
    )
    Cik = jnp.dot(Ajk, jlinalg.solve(I + jnp.dot(Cij, Jjk), jnp.dot(Cij, Ajk.T))) + Cjk
    etaik = (
        jnp.dot(Aij.T, jlinalg.solve(I + jnp.dot(Jjk, Cij), etajk - jnp.dot(Jjk, bij)))
        + etaij
    )
    Jik = jnp.dot(Aij.T, jlinalg.solve(I + jnp.dot(Jjk, Cij), jnp.dot(Jjk, Aij))) + Jij
    return Aik, bik, Cik, etaik, Jik


def par_bwd_pass_scan(elems):
    return lax.associative_scan(vmap(combine_abcej), elems, reverse=True)


def parBackwardPass(ocp: CLQT, blocks, steps, t0, dt):

    elems = parBackwardPass_init(ocp, blocks, steps, t0, dt)
    elems = par_bwd_pass_scan(elems)
    return parBackwardPass_extract(ocp, elems, steps, dt, t0)


###########################################################################
# Parallel computation of states and controls forward
###########################################################################


def fwpass_fw_ode_f(ocp: CLQT, x, t, Kx_d):

    Kx, d = Kx_d

    Psi, phi = unpack_Psiphi(x)

    tF = ocp.F(t) - Kx
    tc = ocp.c(t) + d
    dPsi = tF @ Psi
    dphi = tF @ phi + tc
    dx = pack_Psiphi(dPsi, dphi)

    return dx


def parForwardPass_init(ocp: CLQT, x0, Kx, d, blocks, steps, dt, t0):

    elems = []
    Psi0 = jnp.eye((ocp.F(0)).shape[0])
    phi0 = x0

    elems.append((Psi0, phi0))

    def step_forward(carry, input):

        t, K, d = input
        Psi, phi = carry
        f = lambda x, t: fwpass_fw_ode_f(ocp, x, t, (K, d))
        x = pack_Psiphi(Psi, phi)
        x = euler(f, dt, x, t)
        Psi, phi = unpack_Psiphi(x)

        return (Psi, phi), (Psi, phi)

    Ts = jnp.arange(steps) * dt
    Kx = Kx.reshape((blocks, steps, Kx.shape[-2], Kx.shape[-1]))
    d = d.reshape((blocks, steps, d.shape[-1]))

    Psi0_b = jnp.eye((ocp.F(0)).shape[0])
    phi0_b = jnp.zeros_like(x0)

    def single_pass(Psi0, phi0, elem):

        t_s, Kx, d = elem

        _, (Psi_n, phi_n) = lax.scan(step_forward, (Psi0, phi0), (t_s + Ts, Kx, d))

        Psis = jnp.concatenate([Psi_n, Psi0[None, ...]], axis=0)[:-1]
        phis = jnp.concatenate([phi_n, phi0[None, ...]], axis=0)[:-1]

        return (Psis[-1], phis[-1]), (Psis[-1], phis[-1])

    t0s = t0 + jnp.arange(blocks) * steps * dt
    _, (Psi, phi) = vmap(single_pass, in_axes=(None, None, 0))(
        Psi0_b, phi0_b, (t0s, Kx, d)
    )

    Psiss = jnp.concatenate([jnp.expand_dims(Psi0, axis=0), Psi], axis=0)
    phiss = jnp.concatenate([jnp.expand_dims(phi0, axis=0), phi], axis=0)

    elems = (Psiss, phiss)

    return elems


def parForwardPass_extract(ocp: CLQT, Kxs, ds, elems, steps, dt, t0, u_zoh):

    (Psis, phis) = elems
    blocks_n = phis.shape[0] - 1
    state_dim = phis.shape[-1]
    control_dim = ds.shape[-1]

    t0s = t0 + jnp.arange(blocks_n) * steps * dt
    Kxs = Kxs.reshape((blocks_n, steps, control_dim, state_dim))
    ds = ds.reshape((blocks_n, steps, control_dim))

    (xs_blocks, us_blocks) = vmap(seqForwardPass, in_axes=(None, None, 0, 0, 0, 0))(
        ocp, dt, t0s, phis[:-1], Kxs, ds
    )

    xs = xs_blocks[:, :-1, :].reshape(-1, state_dim)
    us = us_blocks.reshape(-1, control_dim)
    xs = jnp.concatenate([xs, phis[-1][None, :]], axis=0)

    return us, xs


def combine_fc(elem1, elem2):

    Fij, cij = elem1
    Fjk, cjk = elem2

    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_fwd_pass_scan(elems):
    return lax.associative_scan(vmap(combine_fc), elems, reverse=False)


def parForwardPass(ocp: CLQT, x0, Kx, d, blocks, steps, dt, t0, u_zoh=False):

    elems = parForwardPass_init(ocp, x0, Kx, d, blocks, steps, dt=dt, t0=t0)
    elems = par_fwd_pass_scan(elems)
    return parForwardPass_extract(ocp, Kx, d, elems, steps, dt=dt, t0=t0, u_zoh=u_zoh)


###########################################################################
# Parallel computation of (backward and) forward value functions
###########################################################################

def bwpass_fw_ode_f(ocp: CLQT, x, t):

    A, b, C, eta, J = unpack_abcej(x)

    F = ocp.F(t)
    H = ocp.H(t)
    y = ocp.y(t)
    c = ocp.c(t)
    r = ocp.r(t)
    R = ocp.R(t)
    Q = ocp.Q(t)

    I = jnp.eye(R.shape[0])
    R_inv = jlinalg.solve(R, I)

    dA = F @ A - C @ H.T @ R_inv @ H @ A
    db = C @ H.T @ R_inv @ (y - r) - C @ H.T @ R_inv @ H @ b + F @ b + c
    dC = -C @ H.T @ R_inv @ H @ C + Q + F @ C + C @ F.T

    deta = A.T @ H.T @ R_inv @ (y - r) - A.T @ H.T @ R_inv @ H @ b
    dJ   = A.T @ H.T @ R_inv @ H @ A

    dx = pack_abcej(dA, db, dC, deta, dJ)

    return dx

def parFwdBwd_init(ocp: CLQT,blocks, steps, t0, dt):

    elems = []

    dim = ocp.ST.shape[0]

    A0 = jnp.eye(dim)
    b0 = jnp.zeros((dim,))
    C0 = jnp.zeros((dim, dim))
    eta0 = jnp.zeros((dim,))
    J0 = jnp.zeros((dim, dim))

    Ts = jnp.arange(0,steps) * dt

    def step_forward(carry, t):

        f = lambda x, t: bwpass_fw_ode_f(ocp, x, t)

        A, b, C, eta, J = carry

        x = pack_abcej(A, b, C, eta, J)
        x = euler(f, dt, x, t)
        A, b, C, eta, J = unpack_abcej(x)

        C = 0.5 * (C + C.T)
        J = 0.5 * (J + J.T)

        return (A, b, C, eta, J), (A, b, C, eta, J)

    def single_pass(A0, b0, C0, eta0, J0, _t0):

        _, (As, bs, Cs, etas, Js) = lax.scan(
            f=step_forward, init=(A0, b0, C0, eta0, J0), xs=(_t0 + Ts), reverse=False
        )
        return As[-1,:,:], bs[-1,:] , Cs[-1,:,:], etas[-1,:] , Js[-1,:,:]

    t0s = t0 + jnp.arange(0,blocks) * steps * dt

    (A_blocks, b_blocks, C_blocks, eta_blocks, J_blocks) = vmap(
        single_pass, in_axes=(None, None, None, None, None, 0)
    )(A0, b0, C0, eta0, J0, t0s)

    AT = jnp.zeros_like(ocp.ST)
    bT = b0
    CT = C0
    etaT = ocp.vT
    JT = ocp.ST

    As = jnp.concatenate([A_blocks, AT[None]], axis=0)
    bs = jnp.concatenate([b_blocks, bT[None]], axis=0)
    Cs = jnp.concatenate([C_blocks, CT[None]], axis=0)
    etas = jnp.concatenate([eta_blocks, etaT[None]], axis=0)
    Js = jnp.concatenate([J_blocks, JT[None]], axis=0)

    elems = (As, bs, Cs, etas, Js)


    return elems


def parFwdBwdPass_init(ocp: CLQT, x0, blocks, steps, dt, t0):

    (A_blocks, b_blocks, C_blocks, eta_blocks, J_blocks) = parFwdBwd_init(
        ocp, blocks, steps, t0, dt
    )

    elems = (A_blocks, b_blocks, C_blocks, eta_blocks, J_blocks)

    return elems


def parFwdBwdPass_extract(ocp: CLQT, K, d, S, v, elems, steps, dt, t0):

    (As, bs, Cs, etas, Js) = elems
    blocks = As.shape[0] - 1

    t0s = t0 + jnp.arange(blocks) * steps * dt

    A_blocks = As[:-1]
    b_blocks = bs[:-1]
    C_blocks = Cs[:-1]

    A_n, b_n, C_n = vmap(seqFwdBwdPass, in_axes=(None, None, None, 0, 0, 0, 0))(
        ocp, steps, dt, t0s, A_blocks, b_blocks, C_blocks
    )

    AT = A_n[-1, -1, :, :]
    bT = b_n[-1, -1, :]
    CT = C_n[-1, -1, :, :]

    As_all = A_n[:, :-1, :, :]
    bs_all = b_n[:, :-1, :]
    Cs_all = C_n[:, :-1, :, :]

    Ac = jnp.reshape(As_all, (-1, As_all.shape[-2], As_all.shape[-1]))
    bc = jnp.reshape(bs_all, (-1, bs_all.shape[-1]))
    Cc = jnp.reshape(Cs_all, (-1, Cs_all.shape[-2], Cs_all.shape[-1]))

    As_all = jnp.concatenate([ Ac,AT[None, ...]], axis=0)
    bs_all = jnp.concatenate([ bc,bT[None, ...]], axis=0)
    Cs_all = jnp.concatenate([ Cc,CT[None, ...]], axis=0)


    u, x = combine_seqFwdBwdPass(S, K, v, d, As_all, bs_all, Cs_all)

    return u, x, As_all, bs_all, Cs_all


def par_fwdbwd_pass_scan(elems):
    return lax.associative_scan(vmap(combine_abcej), elems, reverse=False)


def parFwdBwdPass(ocp: CLQT, x0, K, d, S, v, blocks, steps, dt, t0):
    
    elems = parFwdBwdPass_init(ocp, x0, blocks, steps, dt, t0)

    A_blocks,b_blocks,C_blocks,eta_blocks,J_blocks=elems

    dim = ocp.F(0).shape[0]


    A0 = jnp.zeros((dim,dim))
    b0 =  jnp.zeros((dim,))
    C0 = jnp.zeros((dim,dim))
    eta0 = jnp.zeros((dim,))
    J0 = jnp.zeros((dim, dim))

    As = jnp.concatenate([A0[None], A_blocks[:-1]], axis=0)
    bs = jnp.concatenate([b0[None], b_blocks[:-1]], axis=0)
    Cs = jnp.concatenate([C0[None], C_blocks[:-1]], axis=0)
    etas = jnp.concatenate([eta0[None], eta_blocks[:-1]], axis=0)
    Js = jnp.concatenate([J0[None], J_blocks[:-1]], axis=0)

    elems=As,bs,Cs,etas,Js
    elems = par_fwdbwd_pass_scan(elems)
    return parFwdBwdPass_extract(ocp, K, d, S, v, elems, steps, dt, t0)
