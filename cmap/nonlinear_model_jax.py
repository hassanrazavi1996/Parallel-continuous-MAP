import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


import cmap.clqt_jax as clqt_p
from cmap.con_to_dis import y_reverse
from cmap.con_to_dis import y
from cmap.nonlinear_model_data import make_ct_data
from jax import lax


def intial_guess(x0, steps):
    x = jnp.full((steps+1, len(x0)), 0.1)
    u = jnp.full((steps+1, len(x0)), 0.1)
    x = x.at[-1, :].set(x0)
    return x, u


def simulate(x0,f,nsteps,dt):
    x=jnp.zeros((nsteps+1, x0.shape[0]))

    def step(x,t):
        x_new=x+f(x,t)*dt
        return x_new, x_new
    
    Ts=jnp.arange(nsteps+1)*dt
    _,x_seq=lax.scan(step, x0,Ts)

    x_seq = jnp.vstack([x0, x_seq])
    return x_seq


def getCLQT_nonlinear(steps):
        

    T=50.0    


    pos_std = 1e-20
    vel_std = 1e-20
    omega_std = 1e-4

    sigma_v=0.005
    sigma_omega=0.02    

    r_range=1.0
    r_bearing=0.01
        
    R=lambda t: jnp.array([[r_range**2 ,0],[0, r_bearing**2]])  
    P0 = lambda t: jnp.diag(jnp.array([pos_std**2, pos_std**2, vel_std**2, vel_std**2, omega_std**2]))


    W = lambda t: jnp.eye(3)


    L_rev = lambda t: jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [sigma_v, 0.0, 0.0],
        [0.0, sigma_v, 0.0],
        [0.0, 0.0, sigma_omega]
    ])



    Q = lambda t: L_rev(t) @ W(t) @ L_rev(t).T

    c_rev = lambda t: -jnp.zeros((5,))

    r_rev = lambda t: -jnp.zeros((2,))
    Q_rev = lambda t: Q(T - t)

    R_rev = lambda t: R(T - t)

    x0 = jnp.array([0.001, 0.001, 0.0, 0.3, jnp.deg2rad(0.0)])

    mu = jnp.linalg.solve(P0(0), x0)

 

    dt=T/steps

    Sigma = lambda t: jnp.linalg.solve(P0(0),jnp.eye(5))
    Fx_rev = lambda x, t: jnp.eye(5)
    Hx_rev = lambda x, t: jnp.array([[1.0, 0.0, 0.0, 0.0,0.0],
                                     [0.0, 1.0, 0.0, 0.0,0.0]])

    _,y_discrete=make_ct_data(x0, P0(0),steps,dt,sigma_v,sigma_omega,r_range,r_bearing,seed=123)
    y_rev = lambda t: y_reverse(t, T, dt, steps, y_discrete)


    clqt =  clqt_p.CLQT(mu, Fx_rev, Sigma, Q_rev, R_rev, c_rev, Hx_rev, y_rev, r_rev, T, L_rev, W)

    return clqt, x0, P0,sigma_v,sigma_omega,r_range,r_bearing







