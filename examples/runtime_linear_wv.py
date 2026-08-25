import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import jax

import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import pandas as pd
import time

config.update("jax_enable_x64", True)



from linear_model_wv_data import make_wv_data
from cmap.linear_estimation_problem import Estimation
from cmap.convert_est_clqt import est_to_clqt
from cmap.clqt_jax import CLQT
from cmap.speedtest_function_linear import clqt_seq_speedtest_linear
from cmap.speedtest_function_linear import clqt_par_speedtest_linear
from cmap.linear_estimation_problem import Estimation
from cmap.diffeq_jax import euler

jax.config.update("jax_platform_name", "cuda")


blocks = jnp.logspace(2, 5, 8, base=10, dtype=jnp.int32)
n = 10


T = 5.0
q = 4
v = 0.01
p0 = 0.01

t0 = 0.0

W = lambda t: q * jnp.eye(2)
H = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
R = lambda t: v * jnp.eye(2)
P0 = p0 * jnp.eye(4)

F = lambda t: jnp.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)

L = lambda t: jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
Q = lambda t: L(t) @ W(t) @ L(t).T

c = lambda t: jnp.zeros((4,))
r = lambda t: jnp.zeros((2,))
m0 = jnp.array([5.0, 5.0, 0.0, 0.0])


par_time_means = []
seq_time_means = []

par_time_samples = []
seq_time_samples = []


for i in range(0, len(blocks)):
    block = blocks[i]

    steps_all = n * blocks[i]

    dt = T / steps_all
    _, y_discrete, _ = make_wv_data(
        m0, P0, F(0), L(0), H(0), steps_all, dt, q, v, t0, seed=123
    )

    est = Estimation(F, H, c, r, L, W, R, y_discrete, P0, m0, T)

    F_cl, H_cl, c_cl, r_cl, Q_cl, R_cl, T_cl, ST_cl, vT_cl, y_cl = est_to_clqt(
        est, steps_all
    )
    clqt = CLQT(vT_cl, F_cl, ST_cl, Q_cl, R_cl, c_cl, H_cl, y_cl, r_cl, T_cl)

    t0 = 0.0
    ST = clqt.ST
    vT = clqt.vT

    par_time_array = []
    seq_time_array = []
    dt = clqt.T / steps_all


    seq_jit = lambda t0, dt, ST, vT: clqt_seq_speedtest_linear(
        clqt, steps_all, t0, dt, ST, vT, method="euler"
    )
    jit_fun1 = jax.jit(seq_jit)
    _, _ = jit_fun1(t0, dt, ST, vT)

    par_jit = lambda t0, dt: clqt_par_speedtest_linear(clqt, block, n, t0, dt,method="euler")
    jit_fun2 = jax.jit(par_jit)
    _, _ = jit_fun2(t0, dt)

    for _ in range(5):
        start_time = time.time()
        _, _ = jit_fun1(t0, dt, ST, vT)
        end_time = time.time()
        seq_time = end_time - start_time

        start_time = time.time()
        _, _ = jit_fun2(t0, dt)
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



df_mean_par = pd.DataFrame(par_time_means_arr)
df_mean_seq = pd.DataFrame(seq_time_means_arr)


df_mean_par.to_csv("par_time_linear_wv_euler.csv")
df_mean_seq.to_csv("seq_time_linear_wv_euler.csv")


from scipy import stats

n_samp = 5  # reps per block size
tval = stats.t.ppf(0.975, df=n_samp - 1)

par_time_ci_arr = tval * jnp.sqrt(par_time_var_arr) / jnp.sqrt(n_samp)
seq_time_ci_arr = tval * jnp.sqrt(seq_time_var_arr) / jnp.sqrt(n_samp)


plt.plot(blocks, par_time_means_arr, label="Parallel method", marker="o")
plt.plot(
    blocks,
    seq_time_means_arr,
    label="Sequential Method",
    linestyle="--",
    marker="x",
)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
