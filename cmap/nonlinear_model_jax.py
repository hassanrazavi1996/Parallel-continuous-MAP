import jax
import jax.numpy as jnp

import cmap.clqt_jax as clqt_p
from cmap.con_to_dis import y_reverse
from cmap.con_to_dis import y
from cmap.nonlinear_model_data import make_ct_data





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
Fu_rev = lambda u : jnp.eye(len(u))
Hx_rev = lambda x, t: jax.jacfwd(lambda x: h(x, t))(x)


def intial_guess(x0,steps):

    x=jnp.zeros((steps-1,len(x0)))
    u=jnp.zeros((steps,len(x0)))

    x=x.at[0,:].set(x0)


    return x,u




def getCLQT_nonlinear(steps):
        

    T=50.0    

    q = 0.2
    v = 0.001
    p0 = 0.01    
        
    R=lambda t: jnp.array([[10.0 ,0],[0, 0.01]])  
    P0 = lambda t: p0 * jnp.eye(4)

    W = lambda t: q * jnp.eye(2)


    L_rev = lambda t: -jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])



    Q = lambda t: L_rev(t) @ W(t) @ L_rev(t).T

    c_rev = lambda t: -jnp.zeros((4,))

    r_rev = lambda t: -jnp.zeros((2,))
    Q_rev = lambda t: Q(T - t)

    R_rev = lambda t: R(T - t)

    x0 = jnp.array([5.0, 5.0, 0.0, 0.0])

    mu = jnp.linalg.solve(P0(0), x0)

    Sigma = lambda t: (1 / p0) * jnp.eye(4)
 

    dt=T/steps

    Sigma = lambda t: (1 / p0) * jnp.eye(4)
    

    _,y_discrete=make_ct_data(x0,P0,nsteps=5000,dt=0.01,q=0.1,r_range=10.0,r_bearing=0.01,seed=None)
    y_rev = lambda t: y_reverse(t, T, dt, steps, y_discrete)




    clqt =  clqt_p.CLQT(mu, Fx_rev, Sigma, Q_rev, R_rev, c_rev, Hx_rev, y_rev, r_rev, T, L_rev, W)

    return clqt, x0, P0, q, v







