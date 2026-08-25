import jax
from jax import config

config.update("jax_enable_x64", True)

from cmap.diffeq_jax import euler
import jax.numpy as jnp
import numpy as np
from jax import debug
from jax.experimental import host_callback as hcb


def pack_Pm(P, m):

    m = jnp.expand_dims(m, axis=-1)
    return jnp.concatenate((P, m), axis=-1)


def unpack_Pm(x):

    n = x.shape[-2]
    P = x[..., :n]
    m = x[..., n]
    return P, m




def odes(F, L, W, H, R, y, x, t):
        
    P, m = unpack_Pm(x)
    Ft = F(t)
    Lt = L(t)
    Wt = W(t)
    Ht = H(t)
    Rt = R(t)
    yt = y(t)


    Qt = Lt @ Wt @ Lt.T

    I = jnp.eye(Rt.shape[0])
    R_inv = jnp.linalg.solve(Rt, I)

    # dP = Ft @ P + P @ Ft.T + Qt - P @ Ht.T @ R_inv @ Ht @ P
    # dm = Ft @ m + P @ Ht.T @ R_inv @ (yt - Ht @ m)
    dP = -Ft @ P - P @ Ft.T - Qt + P @ Ht.T @ R_inv @ Ht @ P
    dm = -Ft @ m - P @ Ht.T @ R_inv @ (yt - Ht @ m)
    dx = pack_Pm(dP, dm)

    return dx


def kalman_bucy_filter(F, L, W, H, R, y, steps, dt, t0, P0, m0):

    Ts = jnp.arange(steps)*dt

    def body(carry, t):

        P, m = carry
        x = pack_Pm(P, m)
        f=lambda x, t: odes(F, L, W, H, R, y, x, t)
        x = euler(f, -dt, x, t+dt)

        P, m = unpack_Pm(x)
        P = 0.5 * (P + P.T)
        return (P, m), (P, m)

    _, (P, m) = jax.lax.scan(f=body, init=(P0, m0), xs=(t0 + Ts),reverse=True)

    Ps = jnp.concatenate([P,P0[None, ...]], axis=0)
    ms = jnp.concatenate([m,m0[None, ...]], axis=0)

    return Ps, ms

def continuous_rts_smoother(Ps_f, ms_f, Fs, Ls, Qs, t_eval, dt):

    n_steps = len(t_eval)
    n = ms_f.shape[1]

    def f_smoother(x, t, P, m):
        P_next, m_next = unpack_Pm(x)

        F = Fs(t)
        L = Ls(t)
        Q = Qs(t)

        G = F + L @ Q @ L.T @ jnp.linalg.solve(P, jnp.eye(n))

        dm = F @ m_next + L @ Q @ L.T @ jnp.linalg.solve(P, jnp.eye(n)) @ (m_next - m)
        dP = G @ P_next + P_next @ G.T - L @ Q @ L.T

        return pack_Pm(dP, dm)

    def body(carry, i):
        P_next, m_next = carry
        P = Ps_f[i]
        m = ms_f[i]
        t = t_eval[i]

        x_next = pack_Pm(P_next, m_next)
        f = lambda x, t: f_smoother(x, t, P, m)
        x_prev = euler(f, -dt, x_next, t + dt)

        P_prev, m_prev = unpack_Pm(x_prev)
        P_prev = 0.5 * (P_prev + P_prev.T)
        return (P_prev, m_prev), (P_prev, m_prev)

    init = (Ps_f[-1], ms_f[-1])
    t_s = jnp.arange(0, n_steps )

    _, (Ps_s, ms_s) = jax.lax.scan(f=body, init=init, xs=t_s, reverse=True)

    Ps_s = jnp.concatenate([Ps_s, Ps_f[-1][None, ...]], axis=0)
    ms_s = jnp.concatenate([ms_s, ms_f[-1][None, ...]], axis=0)

    return ms_s, Ps_s
