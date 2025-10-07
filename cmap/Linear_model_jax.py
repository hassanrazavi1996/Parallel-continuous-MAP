
import jax
import jax.numpy as jnp
import math
import cmap.clqt_jax as cmap
from jax import random, lax
import numpy as np
from cmap.Linear_model_data import make_cv_data

from jax import config

config.update("jax_enable_x64", True)


###########################################################################
#
# Example linear model (of Wiener velocity type)
#
###########################################################################


  

def getCLQT(ocp:cmap):
        

        # ######################
        T=50.0
        #########################
        
        W= lambda t:0.2 * jnp.eye(2)
        H_rev =  lambda t: jnp.array([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0]])
        R  = lambda t: 0.01 * jnp.eye(2)
        Sigma = lambda t: 1 * jnp.eye(4)

        F_rev =  lambda t: -jnp.array([[0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]])
        L_rev =  lambda t: -jnp.array([[0.0,0.0],
                                [0.0,0.0],
                                [1.0,0.0],
                                [0.0,1.0]])
        Q= lambda t: L_rev(t) @ W(t) @ L_rev(t).T
        c_rev =  lambda t: -jnp.zeros((4,))
        
        r_rev = lambda t: -jnp.zeros((2,))
        Q_rev=lambda t: Q(T-t)

        
        R_rev=lambda t: R(T-t)
    
     
        mu=jnp.array([5.0,5.0,0.0,0.0])


        # _, X, y, Y,y_rev=make_cv_data(mu,Sigma(0) ,seed=123)


        cx = jnp.array([5.7770, -2.6692, 1.1187, 0.1379, 0.5718, 1.1214,
                0.2998, 0.3325, 0.7451, 0.2117, 0.6595, 0.0401, -0.2995])
        cy = jnp.array([4.3266, -1.4584, -1.2457, 1.1804, 0.2035, 0.5123,
                1.0588, 0.2616, -0.6286, -0.3802, 0.2750, -0.0070, -0.0022])

        y = lambda t: jnp.array([cx[0] +
             cx[1] * jnp.cos(2.0 * jnp.pi * t / 50.0) + cx[2] * jnp.sin(2.0 * jnp.pi * t / 50.0) +
               cx[3] * jnp.cos(4.0 * jnp.pi * t / 50.0) + cx[4] * jnp.sin(4.0 * jnp.pi * t / 50.0) +
               cx[5] * jnp.cos(6.0 * jnp.pi * t / 50.0) + cx[6] * jnp.sin(6.0 * jnp.pi * t / 50.0) +
               cx[7] * jnp.cos(8.0 * jnp.pi * t / 50.0) + cx[8] * jnp.sin(8.0 * jnp.pi * t / 50.0) +
               cx[9] * jnp.cos(10.0 * jnp.pi * t / 50.0) + cx[10] * jnp.sin(10.0 * jnp.pi * t / 50.0) +
               cx[11] * jnp.cos(12.0 * jnp.pi * t / 50.0) + cx[12] * jnp.sin(12.0 * jnp.pi * t / 50.0),
               cy[0] +
               cy[1] * jnp.cos(2.0 * jnp.pi * t / 50.0) + cy[2] * jnp.sin(2.0 * jnp.pi * t / 50.0) +
               cy[3] * jnp.cos(4.0 * jnp.pi * t / 50.0) + cy[4] * jnp.sin(4.0 * jnp.pi * t / 50.0) +
              cy[5] * jnp.cos(6.0 * jnp.pi * t / 50.0) + cy[6] * jnp.sin(6.0 * jnp.pi * t / 50.0) +
              cy[7] * jnp.cos(8.0 * jnp.pi * t / 50.0) + cy[8] * jnp.sin(8.0 * jnp.pi * t / 50.0) +
              cy[9] * jnp.cos(10.0 * jnp.pi * t / 50.0) + cy[10] * jnp.sin(10.0 * jnp.pi * t / 50.0) +
              cy[11] * jnp.cos(12.0 * jnp.pi * t / 50.0) + cy[12] * jnp.sin(12.0 * jnp.pi * t / 50.0)
                     ])
        
        y_rev=lambda t:y(T-t)
        
        clqt = cmap.CLQT(mu,F_rev, Sigma, Q_rev, R_rev, c_rev, H_rev , y_rev, T)

        return clqt, mu