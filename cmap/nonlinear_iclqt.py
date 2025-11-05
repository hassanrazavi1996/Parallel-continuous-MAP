import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.con_to_dis import f_convert
from cmap.clqt_jax import seqBackwardPass
from cmap.clqt_jax import seqForwardPass

from cmap.clqt_jax import parBackwardPass
from cmap.clqt_jax import parForwardPass



def linearize(clqt,steps,f,h,u,x):

   dt=clqt.T/steps 
   x_con=lambda t:f_convert(t, dt, steps, x)
   u_con=lambda t:f_convert(t, dt, steps, u)

   Fx_rev = lambda x: jax.jacfwd(lambda x: f(x))(x)

  
   Fu_rev = lambda u : jnp.eye(u.shape[-1])
   Hx_rev = lambda x: jax.jacfwd(lambda x: h(x))(x)
   
   
   clqt = clqt._replace(
    F=lambda t: -Fx_rev(x_con(t)),
    c=lambda t: -f(x_con(t)) + Fx_rev(x_con(t)) @ x_con(t)+ Fu_rev(u_con(t)) @ u_con(t),
    H=lambda t: Hx_rev(x_con(t)),
    r=lambda t: h(x_con(t))  - Hx_rev(x_con(t)) @ x_con(t)) 
   
   return clqt
   

def seq_iterate(clqt,steps,blocks,f,h,u,x):
   
   clqt=linearize(clqt,steps*blocks,f,h,u,x)
   dt=clqt.T/(steps*blocks)
   t0=0.0
   vT = clqt.vT
   T  = clqt.T
   ST = clqt.ST

   
   S_seq, v_seq, Kx_seq, d_seq = seqBackwardPass(clqt,steps*blocks,dt,t0,ST,vT)
   phi0= jnp.linalg.solve(S_seq[0] ,v_seq[0])
   x_seq,u_seq = seqForwardPass(clqt, dt, T, phi0, Kx_seq, d_seq,u_zoh=False)

   return u_seq , x_seq 



def par_iterate(clqt,steps,blocks,f,h,u,x,P0):
   
   clqt=linearize(clqt,steps*blocks,f,h,u,x)
   dt=clqt.T/(steps*blocks)
   t0=0.0

   
   
   clqt=clqt._replace(mu=jnp.linalg.solve(P0,x[-1]),
                      Sigma=lambda t: jnp.linalg.solve(P0,jnp.eye(len(x[0]))))

   Kx_par, d_par, S_par, v_par = parBackwardPass(clqt,blocks,steps,t0,dt)
   phi0= jnp.linalg.solve(S_par[0] ,v_par[0])
   u_par,x_par =parForwardPass(clqt, phi0, Kx_par, d_par, blocks, steps,dt,t0,u_zoh=False)

   return u_par , x_par
   
   

   







   


   


