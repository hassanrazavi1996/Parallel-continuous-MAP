import jax
import jax.numpy as jnp
from con_to_dis import f_convert
from clqt_jax import seqBackwardPass
from clqt_jax import seqForwardPass

from clqt_jax import parBackwardPass
from clqt_jax import parForwardPass



def linearize(clqt,steps,u,x):
    

   dt=clqt.T/steps 
   x_con=lambda t:f_convert(t, dt, steps, x)
   u_con=lambda t:f_convert(t, dt, steps, u)


   f = lambda x, t: -jnp.array([
    x[2],                                        
    x[3],                                        
    -(x[4] + 0.2 * jnp.sin(0.5 * -t)) * x[3],    
    (x[4] + 0.2 * jnp.sin(0.5 * -t)) * x[2],     
    0.2 * 0.5 * jnp.cos(0.5 * -t)])

   h = lambda x, t: -jnp.array([
    jnp.sqrt(x[0]**2 + x[1]**2),       
    jnp.arctan2(x[1], x[0])])     


    

   Fx_rev = lambda x, t: jax.jacfwd(lambda x: f(x, t))(x)
   #Fu_rev = lambda u : jnp.eye(len(u))
   Hx_rev = lambda x, t: jax.jacfwd(lambda x: h(x, t))(x)

   clqt = clqt._replace(
    F=lambda t: Fx_rev(t),
    c=lambda t: f(x_con, t) - Fx_rev(t) @ x_con(t),
    H=lambda t: Hx_rev,
    r=lambda t: h(x_con, t) - Hx_rev(t) @ x_con(t)) 
   
   return clqt
   

def seq_iterate(clqt,steps,blocks,u,x):
   
   clqt=linearize(clqt,steps,u,x)
   dt=clqt.T/(steps*blocks)
   t0=0
   sigma=clqt.Sigma
   mu=clqt.mu
   T=clqt.T


   S_seq, v_seq,Kx_seq, d_seq = seqBackwardPass(clqt,steps*blocks,dt,t0,sigma,mu)
   phi0= jnp.linalg.solve(S_seq[0] ,v_seq[0])
   x_seq,u_seq =seqForwardPass(clqt, dt, T, phi0, Kx_seq, d_seq,u_zoh=False)

   u=u_seq
   x=x_seq

   return u,x

def par_iterate(clqt,steps,blocks,u,x):
   
   clqt=linearize(clqt,steps,u,x)
   dt=clqt.T/(steps*blocks)
   t0=0
   sigma=clqt.Sigma
   mu=clqt.mu
   T=clqt.T


   S_seq, v_seq,Kx_seq, d_seq = parBackwardPass(clqt,steps*blocks,dt,t0,sigma,mu)
   phi0= jnp.linalg.solve(S_seq[0] ,v_seq[0])
   x_seq,u_seq =parForwardPass(clqt, dt, T, phi0, Kx_seq, d_seq,u_zoh=False)

   u=u_seq
   x=x_seq

   return u,x
   
   

   







   


   


