import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from cmap.con_to_dis import f_convert
from cmap.clqt_jax import seqBackwardPass
from cmap.clqt_jax import seqForwardPass

from cmap.clqt_jax import parBackwardPass
from cmap.clqt_jax import parForwardPass



def linearize(clqt,steps,u,x):
    

   dt=clqt.T/steps 
   x_con=lambda t:f_convert(t, dt, steps, x)
   u_con=lambda t:f_convert(t, dt, steps, u)
   T=clqt.T

 
   f = lambda x , t: jnp.array([
        x[2],        # dp_x/dt = vx
        x[3],        # dp_y/dt = vy
        -x[4]*x[3],  # dv_x/dt = -omega * vy
        x[4]*x[2],  # dv_y/dt =  omega * vx
        0.0         # domega/dt = 0
          ], dtype=jnp.float64)

   h = lambda x , t: jnp.array([
    jnp.sqrt(x[0]**2 + x[1]**2 + 1e-8),
    jnp.where(
        (x[0]**2 + x[1]**2) > 1e-8,
        jnp.arctan2(x[1], x[0]),
        0.0,
         ),], dtype=jnp.float64)  
   

   Fx_rev = lambda x, t: jax.jacfwd(lambda x: f(x, t))(x)
   Fu_rev = lambda u : jnp.eye(u.shape[-1])
   Hx_rev = lambda x, t: jax.jacfwd(lambda x: h(x, t))(x)
   
  

   clqt = clqt._replace(
    F=lambda t: -Fx_rev(x_con(T-t),T-t),
    c=lambda t: -f(x_con(T-t), T-t) + Fx_rev(x_con(T-t),T-t) @ x_con(T-t) +Fu_rev(u_con(T-t)) @ u_con(T-t),
    H=lambda t: Hx_rev(x_con(T-t), T-t),
    r=lambda t: h(x_con(T-t), T-t) - Hx_rev(x_con(T-t),T-t) @ x_con(T-t)) 
   
   return clqt
   

def seq_iterate(clqt,steps,blocks,u,x,P0):
   
   clqt=linearize(clqt,steps*blocks,u,x)
   dt=clqt.T/(steps*blocks)
   t0=0.0
   
   clqt=clqt._replace(mu=jnp.linalg.solve(P0,x[-1]),
                      Sigma=lambda t: jnp.linalg.solve(P0,jnp.eye(len(x[0]))))

   
   sigma=clqt.Sigma(0)

   
   mu = clqt.mu
   T  = clqt.T


   S_seq, v_seq, Kx_seq, d_seq = seqBackwardPass(clqt,steps*blocks,dt,t0,sigma,mu)
   phi0= jnp.linalg.solve(S_seq[0] ,v_seq[0])
   x_seq,u_seq = seqForwardPass(clqt, dt, T, phi0, Kx_seq, d_seq,u_zoh=False)

   u=u_seq
   x=x_seq

   return u , x



def par_iterate(clqt,steps,blocks,u,x,P0):
   
   clqt=linearize(clqt,steps*blocks,u,x)
   dt=clqt.T/(steps*blocks)
   t0=0.0
   
   clqt=clqt._replace(mu=jnp.linalg.solve(P0,x[-1]),
                      Sigma=lambda t: jnp.linalg.solve(P0,jnp.eye(len(x[0]))))



   Kx_par, d_par, S_par, v_par = parBackwardPass(clqt,blocks,steps,t0,dt)
   phi0= jnp.linalg.solve(S_par[0] ,v_par[0])
   u_par,x_par =parForwardPass(clqt, phi0, Kx_par, d_par, blocks, steps,dt,t0,u_zoh=False)

   u=u_par
   x=x_par

   return u , x
   
   

   







   


   


