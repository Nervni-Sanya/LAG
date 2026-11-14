"""lag.lattice - multi-link lattice gauge layers. Mathematical specification: MATH_SPEC_LATTICE.md.

Modules (MATH_SPEC_LATTICE.md section in brackets; build order of section 11):
  lag.lattice.field     field layout, GaugeGroup, Haar links, shifts, staples, the loop head, plaquettes,
                        Wilson action, gauge transformations                                   [1, 3.1]
  lag.lattice.masks     the masks M(mu, p) (even L only) and the numerical mask validator      [3.2, 3.3]
  lag.lattice.layers    masked sub-layer with batched Dv and its total log|det J|;
                        fixed-point inverse                                                    [2, 6]
  lag.lattice.spectral  eigen-angle log-determinant of single-loop kernels                     [5]
  lag.lattice.exact     exact log Z of 2D lattice gauge theory on a torus; Bessel forms        [7]

Field layout: U has shape (B, D, L, ..., L, N, N) (batch, direction, D periodic lattice axes, colour), complex128.
"""
from lag.lattice.field import (C, CoeffFn, GaugeGroup, gauge_transform, haar, head, plaquettes, shift, staples,
                               wilson_action)
from lag.lattice.masks import mask, validate_mask
from lag.lattice.layers import invert_masked_layer, masked_layer
from lag.lattice.spectral import logdet_spectral
from lag.lattice.exact import (bessel_i, exact_logZ_2d, exact_logZ_2d_su2_bessel, su2_coefficient_bessel,
                               su3_trivial_coefficient_bessel)

__all__ = [
    # field
    "C", "CoeffFn", "GaugeGroup", "haar", "shift", "staples", "head", "gauge_transform", "plaquettes", "wilson_action",
    # masks
    "mask", "validate_mask",
    # layers
    "masked_layer", "invert_masked_layer",
    # spectral
    "logdet_spectral",
    # exact
    "exact_logZ_2d", "bessel_i", "su2_coefficient_bessel", "exact_logZ_2d_su2_bessel", "su3_trivial_coefficient_bessel",
]
