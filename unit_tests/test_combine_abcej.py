import jax.numpy as jnp
from cmap.clqt_jax import combine_abcej

def make_elem(A, b, C, eta, J):
    return (jnp.array(A), jnp.array(b), jnp.array(C), jnp.array(eta), jnp.array(J))

def test_combine_abcej_identity():
    n = 2
    I = jnp.eye(n)
    Z = jnp.zeros((n, n))
    z = jnp.zeros((n,))
    ident = make_elem(I, z, Z, z, Z)

    elem = make_elem([[2.0, 0.0], [0.0, 3.0]], [1.0, -1.0], [[0.5, 0.0],[0.0, 0.25]], [0.2, -0.1], [[0.1,0.0],[0.0,0.2]])

    left = combine_abcej(ident, elem)
    right = combine_abcej(elem, ident)

    for a,b in zip(left, elem):
        assert jnp.allclose(a, b)
    for a,b in zip(right, elem):
        assert jnp.allclose(a, b)

def test_combine_abcej_composition_and_associativity():
    A1 = jnp.array([[2.0, 0.0],[0.0, 1.0]])
    b1 = jnp.array([1.0, 0.0])
    C1 = jnp.zeros((2,2))
    eta1 = jnp.zeros((2,))
    J1 = jnp.zeros((2,2))

    A2 = jnp.array([[1.0, 1.0],[0.0, 1.0]])
    b2 = jnp.array([0.0, 2.0])
    C2 = jnp.zeros((2,2))
    eta2 = jnp.zeros((2,))
    J2 = jnp.zeros((2,2))

    elem1 = (A1, b1, C1, eta1, J1)
    elem2 = (A2, b2, C2, eta2, J2)

    A_expected = A1 @ A2  
    b_expected = A1 @ b2 + b1

    A_out, b_out, C_out, eta_out, J_out = combine_abcej(elem1, elem2)
    assert jnp.allclose(A_out, A_expected)
    assert jnp.allclose(b_out, b_expected)
    assert jnp.allclose(C_out, C1 + A1 @ C2 @ A1.T)  
    assert jnp.allclose(eta_out, eta2)
    assert jnp.allclose(J_out, J2)

    elem3 = make_elem([[1.,0.],[0.,1.]], [0.5, -0.5], jnp.zeros((2,2)), jnp.zeros((2,)), jnp.zeros((2,2)))

    left = combine_abcej(combine_abcej(elem1, elem2), elem3)
    right = combine_abcej(elem1, combine_abcej(elem2, elem3))
    for a,b in zip(left, right):
        assert jnp.allclose(a, b)