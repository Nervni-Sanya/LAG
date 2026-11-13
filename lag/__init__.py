"""LAG - Lie Algebra Geometry (PyTorch). Mathematical specification: MATH_SPEC.md.

Modules (MATH_SPEC.md section in brackets):
  lag.algebra   bases, hat/vee maps, projection onto su(N)              [1]
  lag.exp       closed-form exponentials and dexp^L                     [2]
  lag.jacobian  log|det J| of a step x -> exp(v(x)) . x                 [3]
  lag.heads     equivariant connection heads                            [4]
  lag.flows     Lie-Euler and RKMK4 steps                               [3.2, 3.3]
  lag.groups    generic MatrixLieGroup core + SO3, SU2, SU3, SE3        [10, step 6]
"""
from lag.algebra import C, T2, T3, proj_su, se3_ad, se3_hat, se3_vee_first_order, so3_hat, so3_vee, su_hat, su_vee
from lag.exp import dexpL_generic, dexpL_se3, dexpL_so3, exp_vjp_normal, se3_exp, so3_exp, su2_exp, su3_exp
from lag.jacobian import logabsdet_step
from lag.heads import cross_dual_su3, eps_dual_su2, head_adjoint, head_fundamental, head_loops, wedge_connection
from lag.flows import lie_euler, rkmk4
from lag.groups import SE3, SO3, SU2, SU3, ExpFn, LieGroupLike, MatrixLieGroup

__version__ = "0.1.0"

__all__ = [
    # algebra
    "C", "T2", "T3", "so3_hat", "so3_vee", "su_hat", "su_vee", "proj_su", "se3_hat", "se3_vee_first_order", "se3_ad",
    # exp
    "so3_exp", "su2_exp", "se3_exp", "su3_exp", "dexpL_so3", "dexpL_se3", "dexpL_generic", "exp_vjp_normal",
    # jacobian
    "logabsdet_step",
    # heads
    "eps_dual_su2", "cross_dual_su3", "head_fundamental", "head_adjoint", "head_loops", "wedge_connection",
    # flows
    "lie_euler", "rkmk4",
    # groups
    "MatrixLieGroup", "LieGroupLike", "ExpFn", "SO3", "SU2", "SU3", "SE3",
]
