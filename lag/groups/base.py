"""Generic matrix Lie group core (MATH_SPEC.md sections 1 and 2.3; build order section 10, step 6).

A group is defined by ONE of three data (three constructors):

  1. generators:            ``MatrixLieGroup(generators, name)`` or ``MatrixLieGroup.from_generators``,
                            a basis of the Lie algebra in a faithful matrix representation, shape (d, N, N);
  2. invariant tensors:     ``MatrixLieGroup.from_invariants(N, tensors=..., commuting=...)``,
                            the stabiliser algebra of the given tensors in gl(N);
  3. structure constants:   ``MatrixLieGroup.from_structure_constants(f)``,
                            the adjoint representation, (ad_a)[c, b] = f[a, b, c].

Everything else (hat/vee, Frobenius-orthogonal projection, bracket, ad, exp, dexp^L, action,
closure validation) is derived group-agnostically from the generators.  Closed forms (for
example those of ``lag.exp`` for SO(3), SU(2), SU(3), SE(3)) plug in as backend overrides via
``closed_form_exp`` / ``closed_form_dexpL``; see ``lag.groups.builtin``.

Usage example::

    import torch
    from lag.groups import MatrixLieGroup

    # G2 from its invariant 3-form  phi = e123 + e145 + e167 + e246 - e257 - e347 - e356
    phi = torch.zeros(7, 7, 7, dtype=torch.float64)
    for (i, j, k), s in {(1, 2, 3): 1, (1, 4, 5): 1, (1, 6, 7): 1, (2, 4, 6): 1,
                         (2, 5, 7): -1, (3, 4, 7): -1, (3, 5, 6): -1}.items():
        for p, sign in [((i, j, k), 1), ((j, k, i), 1), ((k, i, j), 1),
                        ((j, i, k), -1), ((i, k, j), -1), ((k, j, i), -1)]:
            phi[p[0] - 1, p[1] - 1, p[2] - 1] = s * sign
    G2 = MatrixLieGroup.from_invariants(7, tensors=(phi,), name="G2")    # dim 14
    g = G2.exp(0.1 * torch.randn(14, dtype=torch.float64))               # element of G2 in GL(7)

    # E8 (or any algebra) from structure constants f: (248, 248, 248)
    E8 = MatrixLieGroup.from_structure_constants(f, name="E8")
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from lag.exp import dexpL_generic

__all__ = ["ExpFn", "LieGroupLike", "MatrixLieGroup"]

ExpFn = Callable[["MatrixLieGroup", Tensor], Tensor]
"""Signature of the closed_form_exp / closed_form_dexpL backends: (group, w) -> tensor."""


@runtime_checkable
class LieGroupLike(Protocol):
    """What ``lag.flows`` needs from a group: the action and the Lie bracket in coordinates.

    ``MatrixLieGroup`` satisfies it; non-matrix implementations can too.
    """

    def act(self, w: Tensor, z: Tensor) -> Tensor:
        """Action of exp(w) on z: w (..., d), z (..., N) -> (..., N)."""
        ...

    def bracket(self, a: Tensor, b: Tensor) -> Tensor:
        """Lie bracket in coordinates: a, b (..., d) -> (..., d)."""
        ...


class MatrixLieGroup:
    """Matrix Lie group defined by a basis of its Lie algebra in a faithful representation.

    Generators have shape (d, N, N) and may be real or complex (real Frobenius inner product).
    Computations run in the generators' precision (float64 / complex128 for the built-in constructors);
    coordinates are real tensors of that precision (real coordinates are cast automatically for complex
    generators). float32 is not supported yet (MATH_SPEC.md section 9, item 6).
    The constructor checks shapes and linear independence; closure is checked on demand with
    closure_residual(). The alternative constructors call cls(generators, name, **kw), so a subclass
    with a different __init__ signature (such as the built-in SO3, SU2, SU3, SE3) cannot use them.
    """

    def __init__(
        self,
        generators: Tensor,
        name: str = "",
        structure_constants: Tensor | None = None,
        exp_backend: str = "pade",
        closed_form_exp: ExpFn | None = None,
        closed_form_dexpL: ExpFn | None = None,
    ) -> None:
        """Build the group from generators (d, N, N); optional structure constants (d, d, d)."""
        if exp_backend not in ("pade", "spectral_skew"):
            raise ValueError(f"exp_backend must be 'pade' or 'spectral_skew', got {exp_backend!r}")
        if generators.dim() != 3 or generators.shape[-1] != generators.shape[-2]:
            raise ValueError(f"generators must have shape (d, N, N), got {tuple(generators.shape)}")
        if exp_backend == "spectral_skew" and \
                float((generators + generators.mH).abs().max()) > 1e-10 * float(generators.abs().max()):
            raise ValueError("exp_backend='spectral_skew' needs real skew-symmetric or anti-Hermitian generators")
        self.G = generators                      # (d, N, N)
        self.dim, self.N, self.name = generators.shape[0], generators.shape[-1], name
        self._Gf = generators.reshape(self.dim, -1)
        # Gram matrix of the real Frobenius inner product <A, B> = Re tr(A^H B)
        try:
            self._gram_inv = torch.linalg.inv((self._Gf.conj() @ self._Gf.T).real)
        except torch.linalg.LinAlgError as err:
            raise ValueError("generators are linearly dependent (singular Gram matrix); with "
                             "from_structure_constants the algebra must be centreless, e.g. semisimple") from err
        self._f = structure_constants
        self.exp_backend, self.closed_form_exp = exp_backend, closed_form_exp
        self.closed_form_dexpL = closed_form_dexpL

    # ---- three constructors -------------------------------------------------------------
    @classmethod
    def from_invariants(cls, N: int, tensors: tuple[Tensor, ...] = (), commuting: tuple[Tensor, ...] = (),
                        name: str = "", tol: float = 1e-9, **kw) -> MatrixLieGroup:
        """Stabiliser algebra of the given tensors (and centraliser of given matrices) in gl(N).

        Tensors are covariant (multilinear forms: metrics, symplectic forms, the G2 3-form): A is kept iff
        sum_s T(..., A v_s, ...) = 0, i.e. T(g u, g v, ...) = T(u, v, ...). An invariant vector v therefore
        gives g^T v = v, which equals g v = v only for orthogonal groups. Each matrix P in `commuting` adds
        [A, P] = 0. The result is a real subalgebra of gl(N, R) with a Frobenius-orthonormal basis; a trace
        condition cannot be expressed (build SU(n) with from_generators). `tol` is the relative singular-value
        cutoff of the rank decision. Memory grows like N^4 times the tensor size, so use this for small N
        (an algebra such as E8 comes from from_structure_constants).
        """
        E = torch.eye(N * N, dtype=torch.float64).reshape(N * N, N, N)
        rows = []
        for T in tensors:
            r = 0
            for s in range(T.dim()):             # sum over slots of T(..., A v_s, ...)
                r = r + torch.einsum("...j,bji->b...i", T.movedim(s, -1), E).movedim(-1, s + 1)
            rows.append(r.reshape(N * N, -1))
        for Pm in commuting:
            rows.append((E @ Pm - Pm @ E).reshape(N * N, -1))
        M = torch.cat(rows, 1)
        U, S, _ = torch.linalg.svd(M, full_matrices=True)
        rank = int((S > tol * S.max()).sum())
        return cls(U[:, rank:].T.reshape(-1, N, N).contiguous(), name, **kw)

    @classmethod
    def from_structure_constants(cls, f: Tensor, name: str = "", **kw) -> MatrixLieGroup:
        """Adjoint representation: (ad_a)[c, b] = f[a, b, c].

        f[a, b, c] are the structure constants, [e_a, e_b] = f_abc e_c (e.g. E8: shape (248, 248, 248)). The
        adjoint representation is faithful only for centreless algebras (e.g. semisimple ones). f is stored
        as given; closure_residual() detects an f that violates the Jacobi identity.
        """
        return cls(f.permute(0, 2, 1).contiguous(), name, structure_constants=f, **kw)

    @classmethod
    def from_generators(cls, generators: Tensor, name: str = "", **kw) -> MatrixLieGroup:
        """Third constructor: generators (d, N, N) given directly."""
        return cls(generators, name, **kw)

    # ---- algebra ------------------------------------------------------------------------
    def hat(self, w: Tensor) -> Tensor:
        """Coordinates to matrix: w (..., d) -> (..., N, N); real w is cast for complex generators."""
        if self.G.is_complex() and not w.is_complex():
            w = w.to(self.G.dtype)
        return torch.einsum("...a,aij->...ij", w, self.G)

    def vee(self, X: Tensor) -> Tensor:
        """Frobenius-orthogonal coordinates: X (..., N, N) -> (..., d), real or complex generators."""
        Gf = self._Gf
        if Gf.is_complex() and not X.is_complex():
            X = X.to(Gf.dtype)
        elif X.is_complex() and not Gf.is_complex():
            X = X.real
        return (X.reshape(*X.shape[:-2], -1) @ Gf.conj().T).real @ self._gram_inv

    def project(self, X: Tensor) -> Tensor:
        """Frobenius-orthogonal projection onto g: X (..., N, N) -> (..., N, N)."""
        return self.hat(self.vee(X))          # Frobenius-orthogonal projection onto g

    @property
    def f(self) -> Tensor:
        """Structure constants f[a, b, c] of shape (d, d, d); computed lazily from the generators.

        f[a] = vee([G_a, G_b]) one a at a time: O(d N^2) memory instead of O(d^2 N^2) (E8-sized algebras).
        """
        if self._f is None:
            self._f = torch.stack([self.vee(self.G[a] @ self.G - self.G @ self.G[a]) for a in range(self.dim)])
        return self._f

    def bracket(self, a: Tensor, b: Tensor) -> Tensor:
        """Lie bracket in coordinates: a, b (..., d) -> (..., d)."""
        return torch.einsum("...a,...b,abc->...c", a, b, self.f)

    def closure_residual(self) -> float:
        """Max-abs failure of [G_a, G_b] to lie in span(G); about 0 for a valid algebra.

        Evaluated one a at a time: O(d N^2) memory instead of O(d^2 N^2) (E8-sized algebras).
        """
        res = []
        for a in range(self.dim):
            Ca = self.G[a] @ self.G - self.G @ self.G[a]          # [G_a, G_b] for all b
            res.append((Ca - self.project(Ca)).abs().max())
        return float(torch.stack(res).max())

    def ad(self, w: Tensor) -> Tensor:
        """Matrix of ad_w = [w, .] in coordinates: w (..., d) -> (..., d, d); MATH_SPEC section 1."""
        return torch.einsum("...a,abc->...cb", w, self.f)

    # ---- group --------------------------------------------------------------------------
    def exp(self, w: Tensor) -> Tensor:
        """Group exponential exp(hat w): w (..., d) -> (..., N, N).

        Uses closed_form_exp if given; otherwise torch.linalg.matrix_exp ("pade") or an eigendecomposition
        ("spectral_skew", only for real skew-symmetric or anti-Hermitian generators; checked in __init__).
        """
        if self.closed_form_exp is not None:
            return self.closed_form_exp(self, w)
        X = self.hat(w)
        if self.exp_backend == "spectral_skew":                  # X real skew or anti-Hermitian  =>  iX Hermitian
            lam, U = torch.linalg.eigh(1j * X)
            E = (U * torch.exp(-1j * lam).unsqueeze(-2)) @ U.mH
            return E if X.is_complex() else E.real
        return torch.linalg.matrix_exp(X)

    def dexpL(self, w: Tensor) -> Tensor:
        """Left-trivialised differential of exp: w (..., d) -> (..., d, d); MATH_SPEC sections 1, 2.3."""
        if self.closed_form_dexpL is not None:
            return self.closed_form_dexpL(self, w)
        return dexpL_generic(self.ad(w))

    def act(self, w: Tensor, z: Tensor) -> Tensor:
        """Action of exp(w) on vectors: w (..., d), z (..., N) -> (..., N)."""
        return (self.exp(w) @ z.unsqueeze(-1)).squeeze(-1)

    def __repr__(self) -> str:
        """E.g. MatrixLieGroup(name='G2', dim=14, N=7, dtype=torch.float64)."""
        return (f"{type(self).__name__}(name={self.name!r}, dim={self.dim}, N={self.N}, "
                f"dtype={self.G.dtype})")
