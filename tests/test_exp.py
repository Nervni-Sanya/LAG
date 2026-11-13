"""Tests of the closed-form exponentials of lag.exp against torch.linalg.matrix_exp (MATH_SPEC.md section 2).

Covers su3_exp across norm scales, its first derivatives at exactly degenerate X (the gradient guard) and its
second derivatives (correct at generic X, NaN rather than silently wrong at degenerate X), and
so3_exp / su2_exp / se3_exp around the series / closed-form switch (th^2 = 0.25) and at large angles.
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
    matrix_exp itself), so 1e-14 leaves a factor 5 of margin. A Taylor cutoff that is too large
    (taylor_cut = 1e-3, K = 4 terms) has truncation error ~ scale^5 / 120 ~ 1e-12 at scale 1e-2 and is caught.
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
    """X = g diag(i a, i a, -2 i a) g^H: two equal eigenvalues, cos(3 phi) = -sign (+-1 after clamping).

    The gradient of Re sum(E * C) through su3_exp must be finite and equal to the one through matrix_exp,
    both projected onto su(3) (su3_exp is only specified on su(3)). The measured difference of the unmutated
    code is <= 4e-15, tolerance 1e-12 (the gradients are O(1)). Without the guard arccos'(+-1) = inf times a
    zero upstream gradient gives NaN in the cases where the clamp is hit exactly; the case g = 1 always is.
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
def test_su3_exp_hessian_at_degenerate_X_is_nan_or_correct(sign):
    """Second derivatives at exactly degenerate X are unsupported (su3_exp docstring): they must come out NaN, as
    in the reference lagmath.py, never as a finite wrong number. Finite cases are near-degenerate after rounding and
    match matrix_exp to <= 3.5e-8 (eigenvalue-gap amplification); tolerance 1e-6. A guard that merely detaches the
    arccos path gives finite errors up to ~2 here and fails."""
    gen = _gen(13)
    cases = [torch.eye(3, dtype=C128)] + [_rand_su3_element(300 + i) for i in range(40)]
    nan_cases = 0
    for i, g in enumerate(cases):
        a = sign * (0.4 + 0.05 * i)
        X = g @ torch.diag(torch.tensor([1j * a, 1j * a, -2j * a], dtype=C128)) @ g.mH
        Cw = torch.randn(3, 3, generator=gen, dtype=C128)
        V = proj_su(torch.randn(3, 3, generator=gen, dtype=C128))
        h = _hvp(su3_exp, X, Cw, V)
        if not torch.isfinite(h).all():
            nan_cases += 1
            continue
        err = float((h - _hvp(mexp, X, Cw, V)).abs().max())
        assert err < 1e-6, f"case {i}, a = {a}: finite but wrong Hessian-vector product (error {err:.3e})"
    assert nan_cases >= 1, "g = 1 hits |cos 3 phi| = 1 exactly; the unsupported case must be reported as NaN"


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
