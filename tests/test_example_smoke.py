"""Smoke tests of examples/su3_single_variable_flow.py (MATH_SPEC.md section 7; build order section 10, step 5).

Fast checks only: the exact reference values, the Haar sampler, the example's section-3.1 wiring against a
brute-force Jacobian, and a few training steps on a small batch. The full section-7 run (350 Adam steps, about
6 minutes on a CPU) is not part of the default suite. Run it with

    python examples/su3_single_variable_flow.py

or, through pytest, with the environment variable LAG_FULL_FLOW=1 (test_full_run_acceptance).
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from lag.exp import su3_exp
from lag.groups import SU3
from lag.heads import head_loops

_PATH = Path(__file__).resolve().parents[1] / "examples" / "su3_single_variable_flow.py"
_spec = importlib.util.spec_from_file_location("su3_single_variable_flow", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)

F64 = torch.float64
C128 = torch.complex128


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_weyl_reference_matches_spec():
    """MATH_SPEC.md section 7: log Z = 1.23589, <Re tr U>/3 = 0.4225, Haar-prior ESS = 11.2 %."""
    ref = ex.weyl_reference()
    assert abs(ref["weyl_norm"] - 6.0) < 1e-12                       # mean of |Delta|^2 on the torus = 3!
    assert abs(ref["log_Z"] - 1.23589) < 5e-6
    assert abs(ref["mean_re_tr_3"] - 0.4225) < 5e-5
    assert abs(ref["ess_prior"] - 0.112) < 5e-4
    coarse = ex.weyl_reference(n=300)                                 # spectral convergence of the grid
    assert abs(coarse["log_Z"] - ref["log_Z"]) < 1e-12


def test_haar_su3_sampler():
    """Unitary, det 1, and Haar moments of SU(3): E tr U = 0, E |tr U|^2 = 1, E (tr U)^3 = 1."""
    n = 40000
    U = ex.haar_su3(n, _gen(1))
    assert U.shape == (n, 3, 3) and U.dtype == C128
    assert float((U.mH @ U - torch.eye(3, dtype=C128)).abs().max()) < 1e-13
    assert float((torch.linalg.det(U) - 1).abs().max()) < 1e-13
    t = U.diagonal(dim1=-2, dim2=-1).sum(-1)
    tol = 5 / math.sqrt(n)                                            # 5 standard errors (all variances <= ~6)
    assert abs(complex(t.mean())) < tol
    assert abs(float((t.abs() ** 2).mean()) - 1) < 2 * tol
    assert abs(complex((t ** 3).mean()) - 1) < 3 * tol


def test_parameter_count_and_identity_at_initialisation():
    """7704 parameters (section 7); zero-initialised output layers make the untrained flow the identity."""
    torch.manual_seed(0)
    flow = ex.LoopFlow()
    assert ex.n_parameters(flow) == 7704
    U0 = ex.haar_su3(64, _gen(2))
    U, logdet, signs, norms = flow(U0, create_graph=False, with_norms=True)
    assert torch.allclose(U, U0, rtol=0, atol=1e-15)
    assert float(logdet.abs().max()) < 1e-14
    assert bool((signs > 0).all())
    assert float(norms.max()) == 0.0


def _randomise(layer: nn.Module, seed: int, scale: float) -> None:
    g = _gen(seed)
    with torch.no_grad():
        for p in layer.net[-1].parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g, dtype=F64))


def test_layer_logdet_matches_bruteforce_jacobian():
    """The example's log|det J| (section 3.1 with Dv of section 3.5) equals the 8 x 8 Jacobian of the layer.

    Brute force: J = d/dxi vee(Phi(exp(hat xi) U) Phi(U)^H) at xi = 0 (right-trivialised coordinates); det Ad = 1.
    The random field is large on purpose (||Dv||_2 of 2 to 4), so both signs of det J are compared. A transposed
    Dv or a missing dexpL changes log|det J| by more than 2 here.
    """
    torch.manual_seed(3)
    layer = ex.LoopLayer()
    _randomise(layer, seed=4, scale=0.5)
    G = SU3()
    U0 = ex.haar_su3(6, _gen(5))
    U1, sign, logabsdet, Dv = layer(U0, G, create_graph=False)

    def phi(U: Tensor) -> Tensor:
        return su3_exp(head_loops(U, layer.coeff)) @ U

    assert float((U1 - phi(U0).detach()).abs().max()) < 1e-14
    for i in range(U0.shape[0]):
        Ui, Pi = U0[i], phi(U0[i]).detach()
        J = torch.autograd.functional.jacobian(lambda xi: G.vee(phi(su3_exp(G.hat(xi)) @ Ui) @ Pi.mH),
                                               torch.zeros(8, dtype=F64))
        s_ref, l_ref = torch.linalg.slogdet(J)
        assert float(sign[i]) == float(s_ref)
        assert abs(float(logabsdet[i] - l_ref)) < 1e-10
    assert float(torch.linalg.matrix_norm(Dv, 2).max()) > 0.05                     # the check is not vacuous


def test_few_training_steps_decrease_loss():
    """A few reverse-KL steps on a small batch: everything finite, det J > 0, loss down on a fixed batch."""
    torch.manual_seed(0)
    flow = ex.LoopFlow()
    U_fixed = ex.haar_su3(512, _gen(6))

    def fixed_loss() -> float:
        ev_U, logdet, signs, _ = flow(U_fixed, create_graph=False)
        assert bool((signs > 0).all())
        return float(-(ex.target_log_density(ev_U) + logdet).mean())

    before = fixed_loss()
    losses = ex.train(flow, steps=8, batch=128, lr=1e-2, generator=_gen(7), log_every=0)
    after = fixed_loss()
    assert len(losses) == 8 and all(math.isfinite(x) for x in losses)
    assert all(bool(torch.isfinite(p).all()) for p in flow.parameters())
    assert math.isfinite(before) and math.isfinite(after)
    assert after < before - 0.3, (before, after)                     # optimum: -log Z = -1.236; start about 0

    ev = ex.evaluate(flow, 1024, _gen(8), chunk=512)
    s = ex.summarize(ev, ex.weyl_reference())
    assert all(math.isfinite(s[k]) for k in ("ess", "log_Z", "log_Z_se", "x", "xw", "kl", "max_dv_norm"))
    assert s["n_nonpositive_det"] == 0
    assert s["ess"] > s["ess_prior"]


@pytest.mark.skipif(os.environ.get("LAG_FULL_FLOW") != "1",
                    reason="full section-7 run (about 6 min on a CPU); set LAG_FULL_FLOW=1 to run it")
def test_full_run_acceptance():
    """The full section-7 run: ESS >= 99 % on two sets of 32,768 samples, det J > 0, ||D omega||_2 < 1, log Z."""
    assert ex.main([]) == 0
