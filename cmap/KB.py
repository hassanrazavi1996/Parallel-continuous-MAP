import numpy as np
import jax
import jax.numpy as jnp
from cmap.diffeq_jax import euler

from jax import config

config.update("jax_enable_x64", True)


def pack_Pm(P,m):
    
    m = jnp.expand_dims(m,axis=-1)
    return jnp.concatenate((P, m), axis=-1)

def unpack_Pm(x):
   
    n = x.shape[-2]
    P = x[..., :n]
    m = x[..., n]
    return P, m




def kalman_bucy_filter(F, L, W, H, R,y, steps, dt, t0, P0, m0):
    
    Ts = jnp.arange(steps, dtype=m0.dtype) * dt


    def odes(F, L, W, H, R,y,x,t):
        P,m = unpack_Pm(x)

       
        F = F(t)
        L = L(t)
        W = W(t)
        H = H(t)
        R = R(t)
        y = y(t)


        Q = L @ W @ L.T
        
        I = jnp.eye(R.shape[0])          
        R_inv = jnp.linalg.solve(R, I)

        
        dP = F @ P + P @ F.T +  Q  - P @ H.T @ R_inv @ H @ P
        dm = F @ m + P @ H.T @ R_inv @ (y - H @ m)

        dx = pack_Pm(dP,dm)

        return dx
    
    def body(carry, t):
        f = lambda x, t: odes(F, L, W, H, R,y,x,t)
        P, m= carry
        x = pack_Pm(P, m)
        x = euler(f, dt, x, t)
        P, m = unpack_Pm(x)
        P= 0.5 * (P + P.T) 
        return (P, m), (P, m)

    _, (P,m) = jax.lax.scan(f=body,init=(P0, m0),xs=(t0 + Ts))

    Ps = jnp.concatenate([P0[None, ...], P], axis=0)  
    ms = jnp.concatenate([m0[None, ...], m], axis=0)  


    return Ps, ms 





def continuous_rts_smoother(Ps_f, ms_f, Fs, Ls, Qs, t_eval, dt):
    
    n_steps = len(t_eval)
    n = ms_f.shape[1]

    Ps_s = np.zeros_like(Ps_f)
    ms_s = np.zeros_like(ms_f)

    Ps_s[-1] = Ps_f[-1]
    ms_s[-1] = ms_f[-1]

    for k in reversed(range(n_steps)):
        print(k)
        P = Ps_f[k]
        m = ms_f[k]

        t = t_eval[k]
        F = Fs(t)
        L = Ls(t)
        Q = Qs(t)

        G = F + L @ Q @ L.T @ jnp.linalg.inv(P)

        dm = F @ ms_s[k+1] + L @ Q @ L.T @ jnp.linalg.inv(P) @ (ms_s[k+1] - m)
        dP = G @ Ps_s[k+1] + Ps_s[k+1] @ G.T - L @ Q @ L.T

        ms_s[k] = ms_s[k+1] - dm * dt
        Ps_s[k] = Ps_s[k+1] - dP * dt


    return ms_s, Ps_s



