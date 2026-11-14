"""Exact reference for 2D lattice gauge theory on a torus (MATH_SPEC_LATTICE.md section 7).

Character expansion exp((beta/N) Re tr U) = sum_r d_r a_r chi_r(U), a_r = (1/d_r) int dU conj(chi_r(U)) exp((beta/N) Re tr U).
Integrating out the links of a torus with V plaquettes (Euler characteristic 0) gives Z = sum_r a_r^V, so

    <Re tr P>/N = (1/V) d/dbeta log Z,    ESS of the Haar prior = Z(beta)^2 / Z(2 beta).

`exact_logZ_2d` (moved verbatim from the reference laglattice.py; type hints and docstring added) computes the a_r by
numerical Weyl integration on the maximal torus, for SU(2) and SU(3).

The Bessel forms of section 7 are new code: the reference files record their checks (exact_bessel_checks.json) but
not their implementation, so they are written here from the formulas of section 7,

    SU(2):  a_k = 2 I_{k+1}(beta) / beta   (spin k/2, dimension k + 1),
    SU(3):  a_0 = sum_{q in Z} det[ I_{i-j+q}(beta/3) ]_{i,j=1..3},

with I_n the modified Bessel function of the first kind (`bessel_i`, power series). They are independent of the Weyl
integration and agree with exact_logZ_2d to 4e-14 at the points of exact_bessel_checks.json (tests/test_lattice_api.py).
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "exact_logZ_2d",
    "bessel_i", "su2_coefficient_bessel", "exact_logZ_2d_su2_bessel", "su3_trivial_coefficient_bessel",
]


# ------------------------------------------------------------------ Weyl integration (reference)
def exact_logZ_2d(beta: float, V: int, N: int, pmax: int = 8, n: int | None = None) -> float:
    """log Z of 2D SU(N) Wilson lattice gauge theory on a periodic lattice with V plaquettes (normalised Haar links):
    Z = sum_r c_r^V, c_r = (1/d_r) int dU conj(chi_r(U)) exp((beta/N) Re tr U). N = 2 or 3.

    Representations kept: SU(2) spins k/2 with k <= 2 pmax; SU(3) highest weights (p, q) with p + q <= pmax
    (pmax = 0 keeps only the trivial representation). The class-function integrals use n midpoints of the SU(2)
    class angle (default 4000) or an n x n grid of the SU(3) maximal torus (default 400); both converge spectrally.
    Returns V log c_max + log sum_r (c_r / c_max)^V as a float.
    """
    if N == 2:
        n = n or 4000
        th = (torch.arange(n, dtype=torch.float64) + 0.5) * math.pi / n
        wgt = (2 / math.pi) * torch.sin(th) ** 2 * (math.pi / n) * torch.exp((beta / 2) * 2 * torch.cos(th))
        cs = [float((wgt * torch.sin((k + 1) * th) / torch.sin(th)).sum()) / (k + 1) for k in range(0, 2 * pmax + 1)]
    else:
        n = n or 400
        t = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
        t1, t2 = torch.meshgrid(t, t, indexing="ij")
        z = [torch.exp(1j * t1), torch.exp(1j * t2), torch.exp(-1j * (t1 + t2))]
        alt = lambda l: torch.linalg.det(torch.stack([torch.stack([zk ** lj for zk in z], -1) for lj in l], -2))
        rho = alt((2, 1, 0))
        f = torch.exp((beta / 3) * sum(zk.real for zk in z))
        norm = 6 * n * n
        cs = []
        for pp in range(pmax + 1):
            for q in range(pmax + 1 - pp):
                dr = (pp + 1) * (q + 1) * (pp + q + 2) / 2
                cs.append(float((alt((pp + q + 2, q + 1, 0)).conj() * rho * f).sum().real) / norm / dr)
    cs = torch.tensor(cs, dtype=torch.float64)
    c0 = cs.max()
    return float(V * torch.log(c0) + torch.log(((cs / c0).abs() ** V * torch.sign(cs) ** V).sum()))


# ------------------------------------------------------------------ Bessel forms (new code, section 7 formulas)
_SERIES_EPS = 1e-17           # relative size of the last term kept in a series


def bessel_i(n: int, x: float) -> float:
    """Modified Bessel function of the first kind I_n(x) for integer order n (I_{-n} = I_n) and real x.

    Power series I_n(x) = sum_{m>=0} (x/2)^(2m+n) / (m! (m+n)!). Its terms share one sign, so there is no
    cancellation; it is summed (math.fsum) until a term falls below 1e-17 of the partial sum, which happens only
    after the terms have started to decrease geometrically.
    """
    n = abs(n)
    h = 0.5 * x
    term = 1.0
    for j in range(1, n + 1):                     # (x/2)^n / n!
        term *= h / j
    terms = [term]
    total = term
    h2 = h * h
    m = 0
    while abs(term) > _SERIES_EPS * abs(total):
        m += 1
        term *= h2 / (m * (m + n))
        terms.append(term)
        total += term
    return math.fsum(terms)


def su2_coefficient_bessel(k: int, beta: float) -> float:
    """a_k(beta) = 2 I_{k+1}(beta) / beta (section 7): the SU(2) character coefficient of spin k/2 (dimension k + 1)
    for the plaquette weight exp((beta/2) Re tr U). beta > 0."""
    return 2 * bessel_i(k + 1, beta) / beta


def exact_logZ_2d_su2_bessel(beta: float, V: int, kmax: int | None = None) -> float:
    """log Z of 2D SU(2) Wilson lattice gauge theory on a torus with V plaquettes from the Bessel form of section 7:
    Z = sum_{k>=0} a_k^V, a_k = 2 I_{k+1}(beta) / beta. beta > 0.

    The sum runs over k = 0..kmax; by default it stops once (a_k / a_0)^V < 1e-17 (a_k decreases with k). Returns
    V log a_0 + log sum_k (a_k / a_0)^V as a float. exact_logZ_2d keeps k <= 2 pmax (16 by default), which is visible
    only at small V and beta of a few units (3.8e-14 at V = 1, beta = 3).
    """
    a0 = su2_coefficient_bessel(0, beta)
    terms = [1.0]
    k = 0
    while (k < kmax) if kmax is not None else (terms[-1] > _SERIES_EPS):
        k += 1
        terms.append((su2_coefficient_bessel(k, beta) / a0) ** V)
    return V * math.log(a0) + math.log(math.fsum(terms))


def su3_trivial_coefficient_bessel(beta: float, qmax: int | None = None) -> float:
    """a_0(beta) = int dU exp((beta/3) Re tr U) over SU(3) with normalised Haar measure, from the Bessel-determinant
    form of section 7: a_0 = sum_{q in Z} det[ I_{i-j+q}(beta/3) ]_{i,j=1..3}.

    a_0 is the coefficient of the trivial representation: the single-plaquette (single-variable) partition function,
    and the limit of Z^(1/V) on large tori. The sum runs over |q| <= qmax; by default it stops once the q and -q terms
    together fall below 1e-17 of the sum.
    """
    x = beta / 3

    def det_q(q: int) -> float:
        M = [[bessel_i(i - j + q, x) for j in range(3)] for i in range(3)]
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

    terms = [det_q(0)]
    q = 0
    while (q < qmax) if qmax is not None else (q == 0 or abs(terms[-1]) + abs(terms[-2]) > _SERIES_EPS * abs(math.fsum(terms))):
        q += 1
        terms += [det_q(q), det_q(-q)]
    return math.fsum(terms)
