import jax
import jax.numpy as jnp
from jax import config


import time
config.update("jax_enable_x64", True)

from cmap.clqt_jax import CLQT
from cmap.Linear_model_jax import getCLQT 
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd

from cmap.speedtest_function_linear import clqt_seq_speedtest_linear
from cmap.speedtest_function_linear import clqt_par_speedtest_linear

jax.config.update("jax_platform_name", "cuda")


blocks = jnp.logspace(2, 5, 5, base=10, dtype=jnp.int32)
n =10 


par_time_means = []
seq_time_means = []


for i in range(0,len(blocks)):
    block=blocks[i]


    steps_all=n*blocks[i]
    clqt, x0,P0,q,v = getCLQT(CLQT,steps_all)
    t0=0.0
    sigma=clqt.Sigma(0)
    mu=clqt.mu

    par_time_array = []
    seq_time_array = []
    dt = clqt.T / steps_all



    seq_jit=lambda t0,dt,sigma,mu: clqt_seq_speedtest_linear(clqt,steps_all,t0,dt,sigma,mu)
    jit_fun1 = jax.jit(seq_jit)
    _,_=jit_fun1(t0,dt,sigma,mu)
    
    par_jit=lambda t0,dt: clqt_par_speedtest_linear(clqt,block,n,t0,dt)
    jit_fun2 = jax.jit(par_jit)
    _,_=jit_fun2(t0,dt)


    for _ in range(10):
        start_time = time.time()
        _, _ = jit_fun1(t0, dt, sigma, mu)
        end_time = time.time()
        seq_time=end_time-start_time

        start_time = time.time()


        start_time = time.time()
        _, _ = jit_fun2(t0, dt)
        end_time = time.time()
        par_time=end_time-start_time


        par_time_array.append(par_time)
        seq_time_array.append(seq_time)

    par_time_means.append(jnp.mean(jnp.array(par_time_array)))
    seq_time_means.append(jnp.mean(jnp.array(seq_time_array)))



par_time_means_arr = jnp.array(par_time_means)
seq_time_means_arr = jnp.array(seq_time_means)


df_mean_par = pd.DataFrame(par_time_means_arr)
df_mean_seq = pd.DataFrame(seq_time_means_arr)




df_mean_par.to_csv("par_time_linearcase.csv")
df_mean_seq.to_csv("seq_time_linearcase.csv")


plt.plot(blocks, par_time_means_arr, label='Parallel Backward Pass', marker='o')
plt.plot(blocks, seq_time_means_arr, label='Sequential Backward Pass', linestyle='--', marker='x')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()



    



