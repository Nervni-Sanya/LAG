"""Lie groups: generic pluggable core and built-in groups (MATH_SPEC.md sections 1, 2, 10).

Plug in a custom group by generators, invariant tensors or structure constants::

    from lag.groups import MatrixLieGroup
    G2 = MatrixLieGroup.from_invariants(7, tensors=(phi,), name="G2")
    E8 = MatrixLieGroup.from_structure_constants(f, name="E8")
    K = MatrixLieGroup.from_generators(gens, name="custom", closed_form_exp=my_exp)
"""
from __future__ import annotations

from lag.groups.base import ExpFn, LieGroupLike, MatrixLieGroup
from lag.groups.builtin import SE3, SO3, SU2, SU3

__all__ = ["MatrixLieGroup", "LieGroupLike", "ExpFn", "SO3", "SU2", "SU3", "SE3"]
