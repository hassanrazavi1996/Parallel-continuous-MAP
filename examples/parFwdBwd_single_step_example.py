import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from cmap.clqt_jax import CLQT, seqBackwardPass, parFwdBwdPass
from cmap.linear_estimation_problem import Estimation
from cmap.convert_est_clqt import est_to_clqt
from linear_model_wv_data import make_wv_data
from cmap.clqt_jax import seqFwdBwdPass,combine_seqFwdBwdPass


def make_con_linear_wv_clqt(steps_all, dt):
    q = 0.2
    v_meas = 0.001
    p0 = 0.01
    T = 1.0
    t0 = 0.0

    F = lambda t: jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    L = lambda t: jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    W = lambda t: q * jnp.eye(2)
    H = lambda t: jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = lambda t: v_meas * jnp.eye(2)
    c = lambda t: jnp.zeros((4,))
    r = lambda t: jnp.zeros((2,))
    m0 = jnp.array([5.0, 5.0, 0.0, 0.0])
    P0 = p0 * jnp.eye(4)

    _, y_discrete, _ = make_wv_data(m0, P0, F(0), L(0), H(0), steps_all, dt, q, v_meas, t0, seed=123)

    est = Estimation(F, H, c, r, L, W, R, y_discrete, P0, m0, T)
    F_cl, H_cl, c_cl, r_cl, Q_cl, R_cl, T_cl, ST_cl, vT_cl, y_cl = est_to_clqt(est, steps_all)
    clqt = CLQT(vT=vT_cl, F=F_cl, ST=ST_cl, Q=Q_cl, R=R_cl, c=c_cl, H=H_cl, y=y_cl, r=r_cl, T=T_cl)

    return clqt, m0


def main():
    blocks = 2
    steps = 2
    steps_all = blocks * steps
    t0 = 0.0
    dt = 1.0 / steps_all

    ocp, x0 = make_con_linear_wv_clqt(steps_all, dt)
    S0 = ocp.ST

    S_seq, v_seq, K_seq, d_seq = seqBackwardPass(ocp, steps_all, dt, t0, ocp.ST, ocp.vT)
    phi0= jnp.linalg.solve(S_seq[0] ,v_seq[0])

    u_out_par, x_out_par, As_all_par, bs_all_par, Cs_all_par = parFwdBwdPass(
        ocp=ocp,
        x0=phi0,
        K=K_seq,
        d=d_seq,
        S=S_seq,
        v=v_seq,
        blocks=blocks,
        steps=steps,
        dt=dt,
        t0=t0,
    )

    A_all_seq,b_all_seq,C_all_seq = seqFwdBwdPass(ocp,steps_all,dt,t0,jnp.zeros((4,4)),phi0,jnp.zeros((4,4)))
    u_out_seq,x_out_seq = combine_seqFwdBwdPass(S_seq,K_seq,v_seq,d_seq,A_all_seq,b_all_seq,C_all_seq)



    print("States x_out_seq:\n", x_out_par,"States x_out_par:\n",x_out_seq)
    
    print("A difference):\n", A_all_seq-As_all_par)
    print("b difference\n", b_all_seq-bs_all_par)
    print("C difference:\n", C_all_seq-Cs_all_par)


if __name__ == "__main__":
    main()
