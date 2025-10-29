from jax import config 
import jax.numpy as jnp
config.update("jax_enable_x64", True)

import numpy as np

def make_ct_data(m0,P0,nsteps=5000,dt=0.01,q=0.1,r_range=10.0,r_bearing=0.01,seed=None):
    rng = np.random.default_rng(seed)
    X = np.zeros((nsteps, 5))
    Y = np.zeros((nsteps, 2))
    DY = np.zeros((nsteps, 2))
    T = np.zeros(nsteps)

    x = rng.multivariate_normal(m0, P0)
    X[0] = x
    z = np.zeros(2)
    t = 0.0
    sqrt_q = np.sqrt(q)

    for k in range(1, nsteps):
        w = rng.standard_normal(3) * sqrt_q * np.sqrt(dt)

        px, py, vx, vy, omega = x

        dp_x = vx
        dp_y = vy
        dv_x = -omega * vy + w[0]
        dv_y =  omega * vx + w[1]
        domega = w[2]

        x = x + dt * np.array([dp_x, dp_y, dv_x, dv_y, domega])

        h = np.array([
            np.sqrt(x[0]**2 + x[1]**2),
            np.arctan2(x[1], x[0])
        ])

        d_eta = np.array([
            rng.normal(0, np.sqrt(r_range * dt)),
            rng.normal(0, np.sqrt(r_bearing * dt))
        ])

        dz = h * dt + d_eta
        z = z + dz

        X[k] = x
        Y[k] = z
        DY[k] = dz
        t += dt
        T[k] = t

    X_jax = jnp.array(X, dtype=jnp.float64)
    Y_jax = jnp.array(Y, dtype=jnp.float64)
    DY_jax = jnp.array(DY, dtype=jnp.float64)/dt

    return  X_jax, DY_jax






