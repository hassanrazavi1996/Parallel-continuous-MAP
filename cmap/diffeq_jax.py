from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp


from typing import Optional, Callable, Any


def euler(f: Callable, dt: float, x: jnp.ndarray, t: Optional[float]) -> jnp.ndarray:

    dx = f(x, t) * dt
    x_next = x + dx

    return x_next

def heun(f: Callable, dt: float, x: jnp.ndarray, t: Optional[float]) -> jnp.ndarray:

    k1 = f(x, t)
    x_euler = x + dt * k1
    k2 = f(x_euler, t + dt)

    x_next = x + (dt / 2.0) * (k1 + k2)

    return x_next


def rk4(f: Callable, dt: float, x: jnp.ndarray, t: Optional[float]) -> jnp.ndarray:

    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_next
