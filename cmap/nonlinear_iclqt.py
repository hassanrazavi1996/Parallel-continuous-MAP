import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.con_to_dis import f_convert
from cmap.clqt_jax import seqBackwardPass
from cmap.clqt_jax import seqForwardPass

from cmap.clqt_jax import parBackwardPass
from cmap.clqt_jax import parForwardPass



def linearize(clqt,steps,f,h,x):

   dt=clqt.T/steps 
   x_con=lambda t:f_convert(t, dt, steps, x)

   Fx_rev = lambda x: jax.jacfwd(f)(x)
   Hx_rev = lambda x: jax.jacfwd(h)(x)
   
   
   clqt = clqt._replace(
    F=lambda t: -Fx_rev(x_con(t)),
    c=lambda t: -f(x_con(t)) + Fx_rev(x_con(t)) @ x_con(t),
    H=lambda t: Hx_rev(x_con(t)),
    r=lambda t: h(x_con(t)) - Hx_rev(x_con(t)) @ x_con(t)) 
   
   return clqt,x_con
   

def seq_iterate(clqt,steps,blocks,f,h,x,t0):
   steps_all=steps*blocks
   clqt,_=linearize(clqt,steps_all,f,h,x)
   dt=clqt.T/(steps*blocks)
   vT = clqt.vT
   ST = clqt.ST
   
   S_seq, v_seq, Kx_seq, d_seq = seqBackwardPass(clqt,steps_all,dt,t0,ST,vT)
   phi0= jnp.linalg.solve(S_seq[0] ,v_seq[0])
   x_seq,u_seq = seqForwardPass(clqt, dt, t0, phi0, Kx_seq, d_seq,u_zoh=False)
   
   return x_seq 



def par_iterate(clqt,steps,blocks,f,h,x,t0):
   
   clqt,_=linearize(clqt,steps*blocks,f,h,x)
   dt=clqt.T/(steps*blocks)

   Kx_par, d_par, S_par, v_par = parBackwardPass(clqt,blocks,steps,t0,dt)
   phi0= jnp.linalg.solve(S_par[0] ,v_par[0])
   u_par,x_par =parForwardPass(clqt, phi0, Kx_par, d_par, blocks, steps,dt,t0,u_zoh=False)

   return  x_par
   
   

   







   


   


