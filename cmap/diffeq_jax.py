import jax
import jax.numpy as jnp
from typing import Optional, Callable, Any
from jax import config

config.update("jax_enable_x64", True)


# def euler(f: Callable, dt: float, x: jnp.ndarray, t: Optional[float]) -> jnp.ndarray:
#     """
#     Forward Euler integrator for ODEs dx/dt = f(x,t)

#     Parameters:
#         f: Function for RHS of ODE, signature f(x,t).
#         dt: Time step (scalar).
#         x: State vector (jax.numpy array).
#         t: Current time.

#     Returns:
#         Updated state vector x after one Euler step.
#     """
#     dx = f(x, t) * dt
#     x_next = x + dx
#     return x_next

def euler(f: Callable, dt: float, x: jnp.ndarray, t: Optional[float] ) -> jnp.ndarray:
    
    
    dx1 = f(x, t) * dt
    dx2 = f(x + 0.5 * dx1, t + dt / 2) * dt
    dx3 = f(x + 0.5 * dx2, t + dt / 2) * dt
    dx4 = f(x + dx3, t + dt) * dt
       

    x_next = x + (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6.0

    return x_next