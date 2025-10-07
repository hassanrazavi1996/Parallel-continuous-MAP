
import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from scipy import signal



def make_cv_data( m0, P0, nsteps=5000, dt=0.01, q=0.2, r=0.01, seed=None):
    """
    Continuous-time Constant Velocity (4D) data generator,
    similar style to make_benes_data.

    Args
    ----
    nsteps : int
        number of time steps
    dt : float
        time increment
    m0 : (4,) ndarray
        initial mean state [p_x, p_y, v_x, v_y]
    P0 : (4,4) ndarray
        initial covariance (for sampling initial x)
    q : float
        process noise spectral density (scalar acceleration intensity)
    r : float
        measurement noise spectral density (scalar, same for x/y)
    seed : int or None
        RNG seed

    Returns
    -------
    T : (nsteps,) array       time grid
    X : (nsteps,4) array      true states
    DY: (nsteps,2) array      measurement increments
    Y : (nsteps,2) array      integrated measurements
    """
    rng = np.random.default_rng(seed)

    # Continuous CV matrices
    F = np.array([[0,0,1,0],
                  [0,0,0,1],
                  [0,0,0,0],
                  [0,0,0,0]])
    L = np.array([[0,0],
                  [0,0],
                  [1,0],
                  [0,1]])           # acceleration noise enters velocities
    H = np.array([[1,0,0,0],
                  [0,1,0,0]])

    T = np.zeros(nsteps)
    X = np.zeros((nsteps,4))
    Y = np.zeros((nsteps,2))
    DY= np.zeros((nsteps,2))
    X[0]=m0
    # initial state sample
    x = rng.multivariate_normal(m0, P0)
    y = np.zeros(2)
    t = 0.0

    for k in range(0,nsteps):
        # process noise increment  (Euler-Maruyama)
        dw = rng.standard_normal(2) * np.sqrt(dt)
        dx = (F @ x) * dt + (L @ (np.sqrt(q) * dw))
        x = x + dx

        # measurement noise increment
        dv = rng.standard_normal(2) * np.sqrt(r * dt)
        dy = (H @ x) * dt + dv
        y  = y + dy
        t += dt
        if  k==nsteps-1:
           DY[k] = dy
           Y[k]  = y
        else:
           X[k+1]  = x
           DY[k] = dy
           Y[k]  = y
           T[k+1]  = t     




    DY_jax0= jnp.array(DY)/dt
    DY_jax = jnp.flip(DY_jax0 , axis= 0)
    def ydot_of_t(t):    
            dt_sim=dt
            print(t / dt_sim)
            if jnp.array(t).ndim ==1:
                 t_a=jnp.array(t)
                 k = jnp.ceil(t_a / dt_sim).astype(jnp.int32)
                 k = jnp.clip(k, 0, nsteps-1)
                 y_vals = jax.vmap(lambda idx: jax.lax.dynamic_index_in_dim(DY_jax0, idx, keepdims=False))(k)
                   
                 y_vals = y_vals.T  
 
                 return y_vals.reshape(2, len(t))  
            elif jnp.array(t).ndim == 0:
                 t_a=jnp.array(t)

                 k = jnp.ceil(t_a/ dt_sim).astype(jnp.int32)

                 k = jnp.clip(k, 0, nsteps-1)
                 y_vals=DY_jax0[k]

                 return y_vals.reshape(2,)       
            else:
                 t_a=jnp.asarray(t)
                 
                 k =jnp.ceil(t_a / dt_sim).astype(jnp.int32)
                 k = jnp.clip(k, 0, nsteps-1).T

                 y_vals  = jnp.take(DY_jax0, k, axis=0) 
                 return y_vals.T


    def ydot_of_t2(t):    
            dt_sim=dt
            if jnp.array(t).ndim ==1:
                 t_a=jnp.array(t)
                 k = (t_a / dt_sim).astype(jnp.int32)
                 k = jnp.clip(k, 0, nsteps-1)
                 y_vals = jax.vmap(lambda idx: jax.lax.dynamic_index_in_dim(DY_jax, idx, keepdims=False))(k)
                   
                 y_vals = y_vals.T  
 
                 return y_vals.reshape(2, len(t))  
            elif jnp.array(t).ndim == 0:
                 t_a=jnp.array(t)

                 k = (t_a/ dt_sim).astype(jnp.int32)

                 k = jnp.clip(k, 0, nsteps-1)
                 y_vals=DY_jax[k]

                 return y_vals.reshape(2,)       
            else:
                 t_a=jnp.asarray(t)
                 
                 k =(t_a / dt_sim).astype(jnp.int32)
                 k = jnp.clip(k, 0, nsteps-1).T

                 y_vals  = jnp.take(DY_jax, k, axis=0) 
                 return y_vals.T
    
    
   

   
    return T, X,ydot_of_t, Y,ydot_of_t2
