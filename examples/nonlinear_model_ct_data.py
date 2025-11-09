from jax import config 
import jax.numpy as jnp
config.update("jax_enable_x64", True)
import jax
import numpy as np

def make_ct_data(m0, P0, f, h, L, nsteps, dt, sigma_v, sigma_omega, r_range, r_bearing, seed=None):



    rng = np.random.default_rng(seed)
    X = np.zeros((nsteps, 5))
    Y = np.zeros((nsteps, 2))
    DY = np.zeros((nsteps, 2))
    T = np.zeros(nsteps)

    x = rng.multivariate_normal(m0, P0)
    X[0] = x
    z = np.zeros(2)
    t = 0.0


    for k in range(0, nsteps):

        dB = rng.standard_normal(2) * np.sqrt(dt)
        w = L @ dB

        dx=f(x)
        
        
        x = x + dt * dx + w

        d_eta = np.array([
            rng.normal(0, r_range  *np.sqrt( dt)),
            rng.normal(0, r_bearing*np.sqrt( dt))
        ])

        dz = h(x) * dt + d_eta
        z = z + dz

        t += dt

       
        if k == nsteps - 1:
         Y[k] = z
         DY[k] = dz
        else:
         X[k+1] = x
         Y[k] = z
         DY[k] = dz
         T[k+1] = t

    X_jax = jnp.array(X, dtype=jnp.float64)
    DY_jax = jnp.array(DY, dtype=jnp.float64) / dt

    return X_jax, DY_jax






