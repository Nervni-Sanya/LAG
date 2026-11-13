"""Flows on Lie groups (MATH_SPEC.md sections 3.2 and 3.3; build order section 10, step 4).

They depend only on ``group.act`` and ``group.bracket`` (see ``LieGroupLike``).
"""
from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from lag.groups.base import LieGroupLike

__all__ = ["lie_euler", "rkmk4"]


# ---- flows: depend only on group.act and group.bracket ----------------------------------
def lie_euler(group: LieGroupLike, field: Callable[[Tensor], Tensor], z: Tensor, h: float | Tensor) -> Tensor:
    """Lie-Euler step z -> exp(h * field(z)) . z; z (..., N) -> (..., N); MATH_SPEC 3.2."""
    return group.act(h * field(z), z)


def rkmk4(group: LieGroupLike, field: Callable[[Tensor], Tensor], z: Tensor, h: float | Tensor) -> Tensor:
    """RKMK4 step z -> exp(v) . z, v the composite increment of stages k1..k4; MATH_SPEC 3.3."""
    k1 = h * field(z)
    k2 = h * field(group.act(k1 / 2, z))
    k3 = h * field(group.act(k2 / 2 - group.bracket(k1, k2) / 8, z))
    k4 = h * field(group.act(k3, z))
    return group.act((k1 + 2 * k2 + 2 * k3 + k4) / 6 - group.bracket(k1, k4) / 12, z)
