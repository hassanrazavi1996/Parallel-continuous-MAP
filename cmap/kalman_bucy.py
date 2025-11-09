import jax
from jax import config

config.update("jax_enable_x64", True)

from cmap.diffeq_jax import euler
import jax.numpy as jnp
import numpy as np


def pack_Pm(P, m):

    m = jnp.expand_dims(m, axis=-1)
    return jnp.concatenate((P, m), axis=-1)


def unpack_Pm(x):

    n = x.shape[-2]
    P = x[..., :n]
    m = x[..., n]
    return P, m


def kalman_bucy_filter(F, L, W, H, R, y, steps, dt, t0, P0, m0):

    Ts = jnp.arange(1, steps+1, dtype=jnp.float64) * dt

    def odes(F, L, W, H, R, y, x, t):
        P, m = unpack_Pm(x)

        F = F(t)
        L = L(t)
        W = W(t)
        H = H(t)
        R = R(t)
        y = y(t)

        Q = L @ W @ L.T

        I = jnp.eye(R.shape[0])
        R_inv = jnp.linalg.solve(R, I)

        dP = F @ P + P @ F.T + Q - P @ H.T @ R_inv @ H @ P
        dm = F @ m + P @ H.T @ R_inv @ (y - H @ m)

        dx = pack_Pm(dP, dm)

        return dx

    def body(carry, t):
        f = lambda x, t_: odes(F, L, W, H, R, y, x, t_)
        P, m = carry
        x = pack_Pm(P, m)
        x = euler(f, dt, x, t)
        P, m = unpack_Pm(x)
        P = 0.5 * (P + P.T)
        return (P, m), (P, m)

    _, (P, m) = jax.lax.scan(f=body, init=(P0, m0), xs=(t0 + Ts))

    Ps = jnp.concatenate([P0[None, ...], P], axis=0)
    ms = jnp.concatenate([m0[None, ...], m], axis=0)

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
    t_s = jnp.arange(n_steps)

    _, (Ps_s, ms_s) = jax.lax.scan(f=body, init=init, xs=t_s, reverse=True)

    Ps_s = jnp.concatenate([Ps_s, Ps_f[-1][None, ...]], axis=0)
    ms_s = jnp.concatenate([ms_s, ms_f[-1][None, ...]], axis=0)

    return ms_s, Ps_s
