import jax
import jax.numpy as jnp
from cmap.nonlinear_iclqt import linearize
from cmap.clqt_jax import CLQT

from jax import config
config.update("jax_enable_x64",True)

"Here is a testing for the Jacobian F and H that whether they"
"are being computed at the nominal trajectory correctly or not"


f = lambda x: jnp.array([x[0]**2+1,x[1]**2+1])
h = lambda x: jnp.array([jnp.sin(x[0])+2,jnp.cos(x[1])+2])


vT= 1
F=lambda t: 1
ST= 1
Q= lambda t: 1
R= lambda t: 1
c= lambda t: 1
H= lambda t: 1
y= lambda t: 1
r= lambda t: 1
T= 1

clqt=CLQT(vT,F,ST,Q,R,c,H,y,r,T)
# u=jnp.array([[2.0,3.0,4.0],[0.4,0.5,0.6]]).T
x=jnp.array([[1.0,1.0,1.0],[0.4,0.5,0.6]]).T

def test_linearize():
    steps=len(x)
    clqt_new,x_con=linearize(clqt,steps,f,h,x)

    Fx=lambda x: jnp.array([[-2*x[0],0.0],[0.0,-2*x[1]]])
    Hx=lambda x: jnp.array([[jnp.cos(x[0]),0.0],[0.0,-jnp.sin(x[1])]])
    
    t_r=0.2
    
    c= -f(x_con(t_r))-Fx(x_con(t_r))@x_con(t_r)
    r=  h(x_con(t_r))-Hx(x_con(t_r))@x_con(t_r)
    
    assert jnp.allclose(Fx(x_con(t_r)),clqt_new.F(t_r))
    assert jnp.allclose(Hx(x_con(t_r)),clqt_new.H(t_r))
    assert jnp.allclose(c,clqt_new.c(t_r))
    assert jnp.allclose(r,clqt_new.r(t_r))
