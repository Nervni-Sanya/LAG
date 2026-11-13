"""Tests of the closed-form exponentials of lag.exp against torch.linalg.matrix_exp (MATH_SPEC.md section 2).

Covers su3_exp across norm scales and its first and second derivatives at generic, near-degenerate, exactly
degenerate and zero X (where the matrix_exp fallback takes over), and so3_exp / su2_exp / se3_exp around the
series / closed-form switch (th^2 = 0.25) and at large angles.
All tensors are float64 / complex128 with explicit dtypes; every test is seeded with a local generator.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from lag.algebra import T2, T3, proj_su, se3_hat, so3_hat, su_hat
from lag.exp import se3_exp, so3_exp, su2_exp, su3_exp

F64 = torch.float64
C128 = torch.complex128
mexp = torch.linalg.matrix_exp


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = F64) -> Tensor:
    return torch.randn(*shape, generator=_gen(seed), dtype=dtype)


# ---------------------------------------------------------------------------------------------
# 1. su3_exp vs matrix_exp across norm scales
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1e-3, 1e-2, 1e-1, 1.0, 3.0])
def test_su3_exp_vs_matrix_exp_norm_scales(scale):
    """Batch of random su(3) elements of Frobenius norm exactly `scale`.

    Tolerance 1e-14: the measured error of the unmutated code is <= 2e-15 at every scale (a few ulp of
    matrix_exp itself), so 1e-14 leaves a factor 5 of margin. These random elements are generic, so they take
    the closed-form path: this checks the closed form itself across four decades of norm.
    """
    w = _randn(200, 8, seed=1)
    X = su_hat(w, T3)
    X = scale * X / X.abs().pow(2).sum((-1, -2), keepdim=True).sqrt()
    E = su3_exp(X)
    assert E.dtype == C128
    err = float((E - mexp(X)).abs().max())
    assert err < 1e-14, f"scale {scale}: max |su3_exp - matrix_exp| = {err:.3e}"
    # the small-norm series and the closed form must also be unitary with det 1
    eye = torch.eye(3, dtype=C128)
    assert float((E.mH @ E - eye).abs().max()) < 1e-14


# ---------------------------------------------------------------------------------------------
# 2. first derivatives at exactly degenerate X (gradient guard)
# ---------------------------------------------------------------------------------------------
def _rand_su3_element(seed: int) -> Tensor:
    return mexp(su_hat(_randn(8, seed=seed), T3))


def _grad(f, X: Tensor, Cw: Tensor) -> Tensor:
    Xr = X.clone().requires_grad_(True)
    (f(Xr) * Cw).sum().real.backward()
    return Xr.grad


@pytest.mark.parametrize("sign", [+1, -1])
def test_su3_exp_gradient_at_exactly_degenerate_X(sign):
    """X = g diag(i a, i a, -2 i a) g^H: two equal eigenvalues, cos(3 phi) = -sign up to rounding.

    The gradient of Re sum(E * C) through su3_exp must be finite and equal to the one through matrix_exp,
    both projected onto su(3) (su3_exp is only specified on su(3)); tolerance 1e-12 (the gradients are O(1)).
    These elements go to the matrix_exp fallback. The reference closed form gives NaN here (arccos'(+-1) = inf
    times a zero upstream gradient) whenever the clamp is hit exactly, which the case g = 1 always is.
    """
    cases = [torch.eye(3, dtype=C128)] + [_rand_su3_element(100 + i) for i in range(40)]
    assert len(cases) >= 20
    gen = _gen(7)
    for i, g in enumerate(cases):
        a = sign * (0.4 + 0.05 * i)
        D = torch.diag(torch.tensor([1j * a, 1j * a, -2j * a], dtype=C128))
        X = g @ D @ g.mH
        Cw = torch.randn(3, 3, generator=gen, dtype=C128)
        ga = _grad(su3_exp, X, Cw)
        gm = _grad(mexp, X, Cw)
        assert torch.isfinite(ga).all(), f"case {i}, a = {a}: non-finite gradient"
        diff = float((proj_su(ga) - proj_su(gm)).abs().max())
        assert diff < 1e-12, f"case {i}, a = {a}: |grad su3_exp - grad matrix_exp| = {diff:.3e}"
        assert float(proj_su(gm).abs().max()) > 1e-3        # the compared gradients are not trivially zero


# ---------------------------------------------------------------------------------------------
# 2b. second derivatives: correct at generic X, never silently wrong at degenerate X
# ---------------------------------------------------------------------------------------------
def _hvp(f, X: Tensor, Cw: Tensor, V: Tensor) -> Tensor:
    """Hessian-vector product of Re sum(f(X) * Cw) in direction V, projected onto su(3)."""
    Xr = X.clone().requires_grad_(True)
    g = torch.autograd.grad((f(Xr) * Cw).sum().real, Xr, create_graph=True)[0]
    return proj_su(torch.autograd.grad((g.conj() * V).real.sum(), Xr)[0])


def test_su3_exp_hessian_generic_matches_matrix_exp():
    """Double backward through su3_exp at generic X equals the one through matrix_exp (measured 1.4e-15)."""
    gen = _gen(11)
    for _ in range(10):
        X = su_hat(torch.randn(8, generator=gen, dtype=F64), T3)
        Cw = torch.randn(3, 3, generator=gen, dtype=C128)
        V = proj_su(torch.randn(3, 3, generator=gen, dtype=C128))
        err = float((_hvp(su3_exp, X, Cw, V) - _hvp(mexp, X, Cw, V)).abs().max())
        assert err < 1e-12, f"generic HVP error {err:.3e}"


@pytest.mark.parametrize("sign", [+1, -1])
def test_su3_exp_hessian_at_degenerate_X_matches_matrix_exp(sign):
    """At exactly degenerate X = g diag(i a, i a, -2 i a) g^H (random g in SU(3), both signs of cos 3 phi) su3_exp
    falls back to matrix_exp, so Hessian-vector products are finite and equal to those through matrix_exp;
    tolerance 1e-12. Before the fallback they were NaN in about half of these cases (23 of 41 per sign), and a
    guard that merely detaches the arccos path gave finite errors up to ~2 here."""
    gen = _gen(13)
    cases = [torch.eye(3, dtype=C128)] + [_rand_su3_element(300 + i) for i in range(40)]
    for i, g in enumerate(cases):
        a = sign * (0.4 + 0.05 * i)
        X = g @ torch.diag(torch.tensor([1j * a, 1j * a, -2j * a], dtype=C128)) @ g.mH
        Cw = torch.randn(3, 3, generator=gen, dtype=C128)
        V = proj_su(torch.randn(3, 3, generator=gen, dtype=C128))
        h = _hvp(su3_exp, X, Cw, V)
        assert torch.isfinite(h).all(), f"case {i}, a = {a}: non-finite Hessian-vector product"
        err = float((h - _hvp(mexp, X, Cw, V)).abs().max())
        assert err < 1e-12, f"case {i}, a = {a}: Hessian-vector product differs from matrix_exp by {err:.3e}"


def test_su3_exp_gradgradcheck_including_degenerate():
    """Finite-difference first and second derivatives (gradcheck, gradgradcheck) at generic, near-degenerate
    (gap 1e-4), exactly degenerate (1.7 T_8) and zero inputs. From the reviewed fallback patch."""
    f = lambda w: torch.view_as_real(su3_exp(su_hat(w, T3)))
    gen = _gen(0)
    points = [torch.randn(8, generator=gen, dtype=F64),
              torch.tensor([0, 0, 1e-4, 0, 0, 0, 0, 1.7], dtype=F64),
              torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.7], dtype=F64),
              torch.zeros(8, dtype=F64)]
    for w0 in points:
        w = w0.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(f, (w,), eps=1e-6, atol=1e-6, rtol=1e-5)
        assert torch.autograd.gradgradcheck(f, (w,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_su3_exp_hessian_finite_at_every_gap():
    """Hessian-vector products against matrix_exp for eigenvalue gaps from 1e-1 down to exactly 0: never NaN,
    relative error below 1e-9. From the reviewed fallback patch."""
    gen = _gen(1)

    def hvp(expf, w, Cw, v):
        w = w.clone().requires_grad_(True)
        g = torch.autograd.grad((expf(su_hat(w, T3)) * Cw).sum().real, w, create_graph=True)[0]
        return torch.autograd.grad((g * v).sum(), w)[0]
    for gap in (1e-1, 1e-3, 1e-5, 1e-7, 1e-8, 1e-10, 0.0):
        for _ in range(10):
            w = torch.zeros(8, dtype=F64)
            w[7] = 1.7
            w[2] = gap
            Cw = torch.randn(3, 3, generator=gen, dtype=C128)
            v = torch.randn(8, generator=gen, dtype=F64)
            ha, hm = hvp(su3_exp, w, Cw, v), hvp(mexp, w, Cw, v)
            assert torch.isfinite(ha).all(), f"gap {gap}: non-finite Hessian-vector product"
            assert float((ha - hm).abs().max() / hm.abs().max()) < 1e-9, f"gap {gap}"


# ---------------------------------------------------------------------------------------------
# 3. so3_exp / su2_exp / se3_exp around the series cutoff and at large angles
# ---------------------------------------------------------------------------------------------
# th^2 = 0.25 (th = 0.5) switches _coeffs between its power series (K = 12 terms) and the closed form;
# angles straddle that cutoff, then reach several turns. tests/test_groups.py covers |w| ~ 1e-4 and ~ 2.
_ANGLES = [0.0, 0.3, 0.45, 0.499, 0.5, 0.501, 0.55, 1.0, math.pi, 6.0, 15.0]


def _directions(n: int, seed: int) -> Tensor:
    u = _randn(n, 3, seed=seed)
    return u / u.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("th", _ANGLES)
def test_so3_su2_se3_exp_vs_matrix_exp_around_cutoff(th):
    """Tolerance 1e-13: entries are O(1) (the se3 translation part grows at most like the angle, <= 15), so
    round-off is a few 1e-16 .. 1e-14 relative to matrix_exp; a series cut to K = 3 terms has error ~ 4e-7
    near th = 0.5 and a cutoff that is too large breaks the series at th = 6."""
    u = _directions(30, seed=int(th * 1000) + 3)
    w = th * u
    assert float((so3_exp(w) - mexp(so3_hat(w))).abs().max()) < 1e-13, "so3"
    assert float((su2_exp(w) - mexp(su_hat(w, T2))).abs().max()) < 1e-13, "su2"
    rho = _randn(30, 3, seed=int(th * 1000) + 5)
    xi = torch.cat([w, rho], -1)
    assert float((se3_exp(xi) - mexp(se3_hat(xi))).abs().max()) < 1e-13, "se3"
