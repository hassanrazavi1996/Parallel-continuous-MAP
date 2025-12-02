import jax
from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap
from cmap.diffeq_jax import euler
from typing import NamedTuple
from typing import Callable

def pack_abcej(A, b, C, m, P):
    b = jnp.expand_dims(b, axis=-1)
    m = jnp.expand_dims(m, axis=-1)
    return jnp.concatenate([A, C, P, b, m], axis=-1)


def unpack_abcej(x):
    n = x.shape[-2]
    A = x[..., :n]
    C = x[..., n : (2 * n)]
    P = x[..., (2 * n) : (3 * n)]
    b = x[..., 3 * n]
    m = x[..., 3 * n + 1]
    return A, b, C, m, P


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


def pack_Pm(P, m):
    m = jnp.expand_dims(m, axis=-1)
    return jnp.concatenate((P, m), axis=-1)


def unpack_Pm(x):
    n = x.shape[-2]
    P = x[..., :n]
    m = x[..., n]
    return P, m


##############################################################################
# Continuous-time Kalman filter and RTS smoother in reverse
##############################################################################


class KFRTS_R(NamedTuple):

    F: Callable
    H: Callable
    c: Callable
    r: Callable
    Q: Callable
    R: Callable
    y: jnp.ndarray
    P0: jnp.ndarray
    m0: jnp.ndarray
    T: float


##################################################################
# Sequential Continuous Kalman filter and RTS smoother in Backward 
##################################################################

def kalman_filtering_bacward_riccati_ode_f(fs: KFRTS_R, x, t):

    P, m = unpack_Pm(x)

    Q = fs.Q(t)
    F = fs.F(t)
    H = fs.H(t)
    R = fs.R(t)
    c = fs.c(t)
    r = fs.r(t)
    y = fs.y(t)

    I = jnp.eye(R.shape[0])
    R_inv = jlinalg.solve(R, I)

    dP = -P @ F.T - F @ P + Q  - P @ H.T @ R_inv @ H @ P
    dm = -F @ m  - c + P @ H.T @ R_inv @ ( y - H @ m - r)

    dx = pack_Pm(dP, dm)

    return dx

def seqKalmanBackward(fs: KFRTS_R, steps, dt, t0, P0, m0):

    Kx = jnp.zeros_like(fs.Q(0))
    d = jnp.zeros((fs.Q(0).shape[-1],))
    Ts = dt * jnp.arange(steps)

    def step(carry, t):

        f = lambda x, t: kalman_filtering_bacward_riccati_ode_f(fs, x, t)

        P, m, _, _ = carry

        x = pack_Pm(P, m)
        x = euler(f, -dt, x, t + dt)
        P, m = unpack_Pm(x)

        P = 0.5 * (P + P.T)
        Q = fs.Q(t)

        Kx = Q @ jlinalg.solve(P,jnp.eye(Q.shape[0]))
        d  = Q @ jlinalg.solve(P,m)

        return (P, m, Kx, d), (P, m, Kx, d)

    _, (Ps, ms, Kxs, ds) = lax.scan(
        f=step, init=(P0, m0, Kx, d), xs=(t0 + Ts), reverse=True
    )

    Ps = jnp.concatenate([Ps, P0[None, ...]], axis=0)
    ms = jnp.concatenate([ms, m0[None, ...]], axis=0)

    return Ps, ms, Kxs, ds


def seqRTSForward(fs: KFRTS_R, dt, t_start, x0, Kx_block, d_block):

    steps = Kx_block.shape[0]
    times = t_start + dt * jnp.arange(steps)

    def step(x, t_Kd):

        t, Kx, d = t_Kd
        f = lambda x, t: fs.F(t) @ x + (-Kx @ x + d) + fs.c(t)
        u = -Kx @ x + d

        x_next = euler(f, dt, x, t)

        return x_next, (u, x_next)

    scan_inputs = (times, Kx_block, d_block)
    _, (us, xs_next) = lax.scan(step, x0, scan_inputs, reverse=False)
    xs = jnp.concatenate([x0[None, :], xs_next], axis=0)

    return xs, us

########################################################
# Parallel Continuous Kalman filter in Backward 
########################################################

def parallel_Kalman_bw_ode_f(fs: KFRTS_R, x, t):

    A, b, C, m, P = unpack_abcej(x)

    F = fs.F(t)
    H = fs.H(t)
    y = fs.y(t)
    c = fs.c(t)
    r = fs.r(t)
    R = fs.R(t)
    Q = fs.Q(t)

    I = jnp.eye(R.shape[0])
    R_inv = jlinalg.solve(R, I)

    dA = jlinalg.solve(P, (A @ Q).T).T + A @ F
    db = -A @ Q @ jlinalg.solve(P, m)  + A @ c
    dC = -A @ Q @ A.T
    dm =  P @ H.T @ R_inv @ (y - r - H @ m) - F @ m - c
    dP = -P @ H.T @ R_inv @ H @ P + Q  - P @ F.T - F @ P

    dx = pack_abcej(dA, db, dC, dm, dP)

    return dx

def parKalmanBackward_init(fs: KFRTS_R, blocks, steps, t0, dt):

    elems = []

    dim = fs.P0.shape[0]

    A0 = jnp.eye(dim)
    b0 = jnp.zeros((dim,))
    C0 = jnp.zeros((dim, dim))
    m0 = jnp.zeros((dim,))
    P0 = jnp.zeros((dim, dim))

    Ts = jnp.arange(steps) * dt

    def step_backward(carry, t):

        f = lambda x, t: parallel_Kalman_bw_ode_f(fs, x, t)

        A, b, C, m, P = carry

        x = pack_abcej(A, b, C, m, P)
        x = euler(f, -dt, x, t + dt)
        A, b, C, m, P = unpack_abcej(x)

        C = 0.5 * (C + C.T)
        P = 0.5 * (P + P.T)

        return (A, b, C, m, P), (A, b, C, m, P)

    def single_pass(A0, b0, C0, m0, P0, _t0):

        _, (As, bs, Cs, ms, Ps) = lax.scan(
            f=step_backward, init=(A0, b0, C0, m0, P0), xs=(_t0 + Ts), reverse=True
        )
        return As[0], bs[0], Cs[0], ms[0], Ps[0]

    t0s = t0 + jnp.arange(blocks) * steps * dt

    (A_blocks, b_blocks, C_blocks, m_blocks, P_blocks) = vmap(
        single_pass, in_axes=(None, None, None, None, None, 0)
    )(A0, b0, C0, m0, P0, t0s)

    AT = jnp.zeros_like(fs.P0)
    bT = b0
    CT = C0
    mT = fs.m0
    PT = fs.P0

    As = jnp.concatenate([A_blocks, AT[None]], axis=0)
    bs = jnp.concatenate([b_blocks, bT[None]], axis=0)
    Cs = jnp.concatenate([C_blocks, CT[None]], axis=0)
    ms = jnp.concatenate([m_blocks, mT[None]], axis=0)
    Ps = jnp.concatenate([P_blocks, PT[None]], axis=0)

    Ps = jnp.concatenate([P_blocks, PT[None]], axis=0)

    vs = jlinalg.solve(Ps, ms[..., None]).squeeze(-1)

    I = jnp.eye(Ps.shape[-1])[None, :, :].repeat(Ps.shape[0], axis=0)
    Ss = jlinalg.solve(Ps, I)
    elems = (As, bs, Cs, vs, Ss)

    return elems


# def parKalmanBackward_extract(fs: KFRTS_R, elems, steps, dt, t0):

#     As, bs, Cs, ms, Ps = elems
#     blocks = Ps.shape[0] - 1

#     t0s = t0 + jnp.arange(blocks) * steps * dt

#     P_blocks = Ps[1:]
#     m_blocks = ms[1:]

#     (Ps, ms, Kxs, ds) = vmap(seqKalmanBackward, in_axes=(None, None, None, 0, 0, 0))(
#         fs, steps, dt, t0s, P_blocks, m_blocks
#     )

#     Ps = Ps[:, 1:, :, :]
#     ms = ms[:, 1:, :]

#     Ps = Ps.reshape((-1,) + Ps.shape[-2:])
#     ms = ms.reshape((-1,) + ms.shape[-1:])
#     Kxs = Kxs.reshape((-1,) + Kxs.shape[-2:])
#     ds = ds.reshape((-1,) + ds.shape[-1:])

#     P0 = Ps[0]
#     m0 = ms[0]

#     Ps = jnp.concatenate([P0[None], Ps], axis=0)
#     ms = jnp.concatenate([m0[None], ms], axis=0)

#     return Kxs, ds, Ps, ms


# def combine_abcej_backward(elem1, elem2):
#     Ajk, bjk, Cjk, mjk, Pjk = elem1
#     Aij, bij, Cij, mij, Pij = elem2

#     I = jnp.eye(Aij.shape[0])
#     Aik = jnp.dot(jnp.dot(Ajk,Pjk), jlinalg.solve(Pjk + Cij, jnp.dot(Pjk,Aij)))
#     bik = bjk + jnp.dot(Ajk, jnp.dot(jlinalg.solve(Pjk + Cij, jnp.dot(Pjk, bij + jnp.dot(Cij, jlinalg.solve(Pij, mjk)))), jnp.eye(bij.shape[0])))
#     Cik = jnp.dot(Ajk, jnp.dot(jlinalg.solve((Cij +  Pjk),Pjk),jnp.dot(Cij,Ajk.T) ) )+ Cjk
#     mik = mij + jnp.dot(
#     Pij,
#     jnp.dot(
#         Aij.T,
#         jlinalg.solve(
#             jnp.dot(Aij, jnp.dot(Pij, Aij.T)) + Pjk + Cij,
#             mjk - bij - jnp.dot(Aij, mij),
#         ),
#     ),)


#     Pik = Pij - jnp.dot(Pij,
#          jnp.dot(Aij.T,
#             jlinalg.solve(
#                 Pjk + Cij + jnp.dot(Aij, jnp.dot(Pij, Aij.T)),
#                 jnp.dot(Aij, Pij)
#             )
#          )
#       )
#     elems = (Aik, bik ,Cik ,mik ,Pik)

#     return elems


def parKalmanBackward_extract(ocp: KFRTS_R, elems, steps, dt, t0):

    As, bs, Cs, etas, Js = elems
    blocks = Js.shape[0] - 1

    t0s = t0 + jnp.arange(blocks) * steps * dt

    J_blocks = Js[1:]
    eta_blocks = etas[1:]

    (Ss, vs, Kxs, ds) = vmap(seqKalmanBackward, in_axes=(None, None, None, 0, 0, 0))(
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



def combine_abcej_backward(elem1, elem2):

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
    return lax.associative_scan(vmap(combine_abcej_backward), elems, reverse=True)


def parKalmanBackward(fs: KFRTS_R, blocks, steps, t0, dt):
   
    elems = parKalmanBackward_init(fs, blocks, steps, t0, dt)
    elems = par_bwd_pass_scan(elems)
    return parKalmanBackward_extract(fs, elems, steps, dt, t0)

#######################################################################################################
# Parallel Continuous RTS smoother in Forward (This exactly as same as the forwardpass in the clqt_jax)
#######################################################################################################















