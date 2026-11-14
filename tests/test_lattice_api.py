"""Tests of the lag.lattice pieces that verify_lattice.py does not cover directly.

- mask: the even-L check of MATH_SPEC_LATTICE.md section 11, and the sweep partition of section 3.2;
- invert_masked_layer: bit-for-bit equal to the fixed-point loop of verify_lattice.py (section 6);
- the Bessel forms of section 7 (new code, no reference implementation): against the Weyl integration of
  exact_logZ_2d at the points of exact_bessel_checks.json, and against Bessel-function identities;
- the package exports.
"""
from __future__ import annotations

import math

import pytest
import torch

import lag
from lag import lattice
from lag.lattice import (GaugeGroup, bessel_i, exact_logZ_2d, exact_logZ_2d_su2_bessel, haar, head,
                         invert_masked_layer, mask, masked_layer, staples, su2_coefficient_bessel,
                         su3_trivial_coefficient_bessel, validate_mask)

F64 = torch.float64


def _field(B: int, D: int, L: int, N: int) -> torch.Tensor:
    return haar(B * D * L ** D, N).reshape(B, D, *[L] * D, N, N)


def _coeff(n_loops: int, scale: float, seed: int):
    """The random kernel of verify_lattice.rand_coeff (without its float64 default-dtype assumption)."""
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(4 * n_loops, 2 * n_loops, generator=g, dtype=F64) / 2
    b1 = torch.randn(4 * n_loops, generator=g, dtype=F64)

    def coeff(t):
        n = t.shape[-1]
        o = scale * torch.tanh(torch.cat([t.real, t.imag], -1) @ W1[:4 * n, :2 * n].T + b1[:4 * n])
        return torch.complex(o[..., :n], o[..., n:2 * n]), torch.complex(o[..., 2 * n:3 * n], o[..., 3 * n:])
    return coeff


# ------------------------------------------------------------------ masks
@pytest.mark.parametrize("L", [1, 3, 5])
def test_mask_rejects_odd_L(L):
    """Section 3.2: M(mu, p) is an independent set only for even L, so mask() and every caller refuse odd L."""
    with pytest.raises(ValueError, match="even"):
        mask(L, 2, 0, 0)
    torch.manual_seed(0)
    G, U, cf = GaugeGroup(2), _field(1, 2, L, 2), _coeff(2, 0.25, 0)
    for call in (lambda: masked_layer(U, 0, 0, cf, G), lambda: masked_layer(U, 1, 1, cf, G, logdet=False),
                 lambda: validate_mask(U, 0, 0, cf, G), lambda: invert_masked_layer(U, 0, 0, cf, G)):
        with pytest.raises(ValueError, match="even"):
            call()


@pytest.mark.parametrize("D,L", [(2, 2), (2, 4), (2, 6), (3, 4), (4, 2)])
def test_masks_partition_every_direction(D, L):
    """Section 3.2: for each mu, M(mu, 0) and M(mu, 1) split the mu-links in halves, so a sweep of the 2D sub-layers
    updates every link exactly once, a fraction 1/(2D) of all links per sub-layer."""
    for mu in range(D):
        m0, m1 = mask(L, D, mu, 0), mask(L, D, mu, 1)
        assert m0.shape == (L,) * D and m0.dtype == torch.bool
        assert bool((m0 ^ m1).all())
        assert int(m0.sum()) == int(m1.sum()) == L ** D // 2


def test_mask_has_no_two_links_on_one_plaquette():
    """Section 3.2: two mu-links share a plaquette only at x and x +- nu, whose transverse parities differ."""
    D, L = 3, 4
    for mu in range(D):
        for p in (0, 1):
            m = mask(L, D, mu, p)
            for nu in range(D):
                if nu != mu:
                    assert not bool((m & torch.roll(m, 1, dims=nu)).any())


# ------------------------------------------------------------------ inverse (section 6)
@pytest.mark.parametrize("N", [2, 3])
def test_invert_masked_layer_matches_reference_loop(N):
    """invert_masked_layer is the loop of verify_lattice.sec_inverse: same iterates, same count, bit for bit.

    The input field is recovered to 1e-12 and the frozen links of the output are passed through unchanged.
    """
    torch.manual_seed(3 + N)
    G, D, L, mu, p = GaugeGroup(N), 2, 4, 1, 0
    U = _field(2, D, L, N)
    cf = _coeff(2, 0.1, 8)
    Uout, ld, sg, nm = masked_layer(U, mu, p, cf, G)
    idx = mask(L, D, mu, p).reshape(-1).nonzero().squeeze(-1)
    V = L ** D
    # the reference loop, verbatim from verify_lattice.py
    Ya = Uout[:, mu].reshape(2, V, N, N)[:, idx]; S = staples(Uout, mu).reshape(2, V, -1, N, N)[:, idx]
    X = Ya.clone(); it = 0; res = 1.0
    while res > 1e-14 and it < 200:
        Xn = G.exp(-G.vee(head(X.unsqueeze(-3) @ S, cf))) @ Ya; res = float((Xn - X).abs().max()); X = Xn; it += 1
    Uin, it2, res2 = invert_masked_layer(Uout, mu, p, cf, G)
    assert it2 == it and res2 == res and 0 < it < 200 and res <= 1e-14
    assert torch.equal(Uin[:, mu].reshape(2, V, N, N)[:, idx], X)
    frozen = torch.ones(D, V, dtype=torch.bool)
    frozen[mu, idx] = False
    assert torch.equal(Uin.reshape(2, D, V, N, N)[:, frozen], Uout.reshape(2, D, V, N, N)[:, frozen])
    assert float((Uin - U).abs().max()) < 1e-12
    assert float(nm.max()) < 1                                      # Proposition 3: q < 1


def test_invert_masked_layer_reports_unconverged_iteration():
    """With max_iter too small the last change stays above tol, which is how non-convergence is reported."""
    torch.manual_seed(11)
    G = GaugeGroup(3)
    Uout = masked_layer(_field(1, 2, 4, 3), 0, 1, _coeff(2, 0.1, 8), G, logdet=False)
    _, it, res = invert_masked_layer(Uout, 0, 1, _coeff(2, 0.1, 8), G, max_iter=2)
    assert it == 2 and res > 1e-14


# ------------------------------------------------------------------ Bessel forms (section 7)
def test_bessel_i_identities():
    """I_n(0) = delta_n0, I_-n = I_n, the recurrence I_{n-1} - I_{n+1} = (2n/x) I_n, and I_0, I_1 against torch."""
    assert bessel_i(0, 0.0) == 1.0 and bessel_i(3, 0.0) == 0.0
    for x in (1e-3, 0.5, 2 / 3, 1.0, 2.0, 3.0, 6.0, 12.0):
        assert bessel_i(-4, x) == bessel_i(4, x)
        for n in range(1, 20):
            lhs, rhs = bessel_i(n - 1, x) - bessel_i(n + 1, x), 2 * n / x * bessel_i(n, x)
            assert abs(lhs - rhs) <= 1e-13 * abs(rhs) + 1e-300
        t = torch.tensor(x, dtype=F64)
        assert abs(bessel_i(0, x) / float(torch.special.i0(t)) - 1) < 1e-14
        assert abs(bessel_i(1, x) / float(torch.special.i1(t)) - 1) < 1e-14


def test_bessel_i_against_scipy():
    sp = pytest.importorskip("scipy.special")
    worst = max(abs(bessel_i(n, x) / float(sp.iv(n, x)) - 1)
                for n in range(40) for x in (0.01, 0.5, 2 / 3, 1.0, 2.0, 6.0, 12.0, 30.0) if sp.iv(n, x) > 1e-290)
    assert worst < 1e-14


# Points of exact_bessel_checks.json; the recorded |Bessel - Weyl| is at most 3.8e-14 (SU(2), beta = 3, V = 1, where
# exact_logZ_2d keeps spins up to 8) and 2.2e-15 (SU(3)).
@pytest.mark.parametrize("beta", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("V", [1, 4, 16])
def test_su2_bessel_form_matches_weyl_integration(beta, V):
    assert abs(exact_logZ_2d_su2_bessel(beta, V) - exact_logZ_2d(beta, V, 2)) < 1e-13


@pytest.mark.parametrize("beta", [2.0, 3.0, 6.0])
def test_su3_bessel_determinant_matches_weyl_integration(beta):
    """a_0 = sum_q det[I_{i-j+q}(beta/3)] is the single-plaquette integral: exact_logZ_2d with V = 1, pmax = 0."""
    assert abs(math.log(su3_trivial_coefficient_bessel(beta)) - exact_logZ_2d(beta, 1, 3, pmax=0)) < 1e-14


def test_bessel_forms_reference_values():
    """MATH_SPEC.md section 7: log Z = 1.2358900916 for one SU(3) variable at beta = 6 (Weyl formula, 600 x 600 grid);
    the U(3) term q = 0 alone is not enough; SU(2) spin-0 coefficient 2 I_1(beta)/beta; truncations."""
    assert abs(math.log(su3_trivial_coefficient_bessel(6.0)) - 1.2358900916) < 1e-9
    assert su3_trivial_coefficient_bessel(6.0, qmax=0) < 0.8 * su3_trivial_coefficient_bessel(6.0)
    assert abs(su2_coefficient_bessel(0, 2.0) - 2 * bessel_i(1, 2.0) / 2.0) == 0
    # kmax = 0 keeps the trivial representation only, as exact_logZ_2d with pmax = 0
    assert abs(exact_logZ_2d_su2_bessel(2.0, 4, kmax=0) - exact_logZ_2d(2.0, 4, 2, pmax=0)) < 1e-13
    # V -> large: log Z / V -> log a_0
    assert abs(exact_logZ_2d_su2_bessel(2.0, 10 ** 6) / 10 ** 6 - math.log(su2_coefficient_bessel(0, 2.0))) < 1e-15


# ------------------------------------------------------------------ exports
def test_lattice_exports():
    assert "lattice" in lag.__all__ and lag.lattice is lattice
    for name in lattice.__all__:
        assert getattr(lattice, name) is not None
    assert len(lattice.__all__) == len(set(lattice.__all__)) == 20
