import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import time
import pandas as pd
import matplotlib.pyplot as plt

from cmap.speedtest_function_nonlinear import clqt_seq_speedtest_nonlinear
from cmap.speedtest_function_nonlinear import clqt_par_speedtest_nonlinear
from cmap.linear_estimation_problem import Estimation
from cmap.clqt_jax import CLQT
from cmap.convert_est_clqt import est_to_clqt
from cmap.initial_guess_nonlinear import intial_guess
from cmap.initial_guess_nonlinear import simulate

from nonlinear_model_ct_data import make_ct_data
from nonlinear_statespace_ct import f, h

jax.config.update("jax_platform_name", "cuda")


######
blocks = jnp.logspace(2, 5, 8, base=10).astype(jnp.int32)
n = 10

par_time_means = []
seq_time_means = []

par_time_samples = []
seq_time_samples = []
######

T = 1.0

pos_std = 1e-1
vel_std = 1e-1
omega_std = 2e-1

sigma_v = 5e-3
sigma_omega = 0.02

r_range = 0.005
r_bearing = 0.001

R = lambda t: jnp.array([[r_range**2, 0], [0, r_bearing**2]])
P0 = jnp.diag(jnp.array([pos_std**2, pos_std**2, vel_std**2, vel_std**2, omega_std**2]))
W = lambda t: jnp.eye(3)
L = lambda t:  jnp.array([
        [ 0.0, 0.0,0.0],
        [ 0.0, 0.0,0.0],
        [sigma_v,0.0,0.0],
        [0.0,sigma_v,0.0],
        [0.0,0.0,sigma_omega]
    ])

Q = lambda t: L(t) @ W(t) @ L(t).T
c = lambda t: jnp.zeros((5,))
r = lambda t: jnp.zeros((2,))
m0 = jnp.array([5.0, 5.0, 0.0, 0.3, jnp.deg2rad(0.0)])
vT = jnp.linalg.solve(P0, m0)


ST = jnp.linalg.solve(P0, jnp.eye(5))
F = lambda t: jnp.eye(5)
H = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
t0 = 0.0

niter = 10


for i in range(0, len(blocks)):

    block = int(blocks[i])

    steps_all = n * block

    dt = T / steps_all

    _, y_discrete = make_ct_data(
        m0,
        P0,
        f,
        h,
        L(0),
        steps_all,
        dt,
        sigma_v,
        sigma_omega,
        r_range,
        r_bearing,
        seed=123,
    )
    est = Estimation(F, H, c, r, L, W, R, y_discrete, P0, m0, T)

    F_cl, H_cl, c_cl, r_cl, Q_cl, R_cl, T_cl, ST_cl, vT_cl, y_cl = est_to_clqt(
        est, steps_all
    )
    clqt = CLQT(vT_cl, F_cl, ST_cl, Q_cl, R_cl, c_cl, H_cl, y_cl, r_cl, T_cl)

    x, u = intial_guess(m0, steps_all)
    dt = T / steps_all

    seq_jit = lambda x, t0: clqt_seq_speedtest_nonlinear(
        clqt, n, block, f, h, x, t0, niter,diffeq_solver="euler"
    )
    jit_fun1 = jax.jit(seq_jit)
    _ = jit_fun1(x, t0)

    par_jit = lambda x, t0: clqt_par_speedtest_nonlinear(
        clqt, n, block, f, h, x, t0, niter,diffeq_solver="euler"
    )
    jit_fun2 = jax.jit(par_jit)
    _ = jit_fun2(x, t0)

    jax.block_until_ready(jit_fun1(x, t0))  
    jax.block_until_ready(jit_fun2(x, t0))

    par_time_array = []
    seq_time_array = []

    for _ in range(20):
        start_time = time.time()
        result1 = jit_fun1(x, t0)
        jax.block_until_ready(result1)
        end_time = time.time()
        seq_time = end_time - start_time

        start_time = time.time()
        result2 = jit_fun2(x, t0)
        jax.block_until_ready(result2)
        end_time = time.time()
        par_time = end_time - start_time

        par_time_array.append(par_time)
        seq_time_array.append(seq_time)

    par_time_means.append(jnp.mean(jnp.array(par_time_array)))
    seq_time_means.append(jnp.mean(jnp.array(seq_time_array)))

    par_time_samples.append(par_time_array)
    seq_time_samples.append(seq_time_array)


par_time_means_arr = jnp.array(par_time_means)
seq_time_means_arr = jnp.array(seq_time_means)


par_time_var_arr = jnp.var(jnp.array(par_time_samples), axis=1, ddof=1)
seq_time_var_arr = jnp.var(jnp.array(seq_time_samples), axis=1, ddof=1)

df_all_samples_par = pd.DataFrame(par_time_samples)
df_all_samples_seq = pd.DataFrame(seq_time_samples)

df_mean_par = pd.DataFrame(par_time_means_arr)
df_mean_seq = pd.DataFrame(seq_time_means_arr)




df_mean_par.to_csv("runtime_nonlinear_ct/par_time_nonlinear_ct_euler.csv")
df_mean_seq.to_csv("runtime_nonlinear_ct/seq_time_nonlinear_ct_euler.csv")

df_all_samples_par.to_csv("runtime_nonlinear_ct/par_all_samples_nonlinear_ct_euler.csv")
df_all_samples_seq.to_csv("runtime_nonlinear_ct/seq_all_samples_nonlinear_ct_euler.csv")


from scipy import stats

n_samp = 20  # reps per block size
tval = stats.t.ppf(0.975, df=n_samp - 1)

par_time_ci_arr = tval * jnp.sqrt(par_time_var_arr) / jnp.sqrt(n_samp)
seq_time_ci_arr = tval * jnp.sqrt(seq_time_var_arr) / jnp.sqrt(n_samp)


plt.plot(blocks, par_time_means_arr, label="Parallel method", marker="o")
plt.fill_between(
    blocks,
    par_time_means_arr - par_time_ci_arr,
    par_time_means_arr + par_time_ci_arr,
    alpha=0.2,
)

plt.plot(
    blocks,
    seq_time_means_arr,
    label="Sequential Method",
    linestyle="--",
    marker="x",
)
plt.fill_between(
    blocks,
    seq_time_means_arr - seq_time_ci_arr,
    seq_time_means_arr + seq_time_ci_arr,
    alpha=0.2,
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Blocks")
plt.ylabel("Runtime (s)")
plt.legend()
plt.show()

plt.savefig("runtime_nonlinear_ct/runtime_nonlinear_ct_euler.png", dpi=150, bbox_inches="tight")