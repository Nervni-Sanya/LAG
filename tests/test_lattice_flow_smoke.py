"""Smoke tests of examples/lattice_flow_2d.py (MATH_SPEC_LATTICE.md section 8; build order section 11, step 6).

Fast checks only: the parameter count and the identity at initialisation, the flow's total log-determinant against a
brute-force Jacobian of all links, a few training steps, the acceptance logic, and a tiny run through the command-line
path (train, save, reload, evaluate). The full section-8 runs are not part of the default suite; on 6 CPU threads
section 8 reports 1705 s of training for SU(3) at beta = 3 and 833 s for SU(2) at beta = 2, and the two evaluation
sets add about 21 and 7 minutes. Run them with

    python examples/lattice_flow_2d.py --N 3
    python examples/lattice_flow_2d.py --N 2

or, through pytest, with the environment variable LAG_FULL_LATTICE_FLOW=1 (test_full_run_acceptance, both groups).
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from lag.lattice import C, exact_logZ_2d, haar, masked_layer, wilson_action

_PATH = Path(__file__).resolve().parents[1] / "examples" / "lattice_flow_2d.py"
_spec = importlib.util.spec_from_file_location("lattice_flow_2d", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


@pytest.fixture(autouse=True)
def float64_default():
    """The example's Flow and Coeff (verbatim from the reference) need float64 as the default dtype, as in main()."""
    prev, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)
    torch.set_num_threads(threads)


def _field(B: int, L: int, N: int) -> Tensor:
    return haar(B * 2 * L * L, N).reshape(B, 2, L, L, N, N)


def _randomise(flow: torch.nn.Module, seed: int, scale: float) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for cf in flow.coeffs:
            for p in cf.net[-1].parameters():
                p.copy_(scale * torch.randn(p.shape, generator=g))


@pytest.mark.parametrize("N", [2, 3])
def test_parameter_count_and_identity_at_initialisation(N):
    """17,760 parameters (section 8); zero-initialised output layers make the untrained flow the identity."""
    torch.manual_seed(0)
    flow = ex.Flow(N, 3)
    assert sum(p.numel() for p in flow.parameters()) == 17760 and len(flow.order) == 12
    U0 = _field(3, 4, N)
    U, ld, sg, mx = flow(U0, create_graph=False)
    assert torch.equal(U, U0)
    assert bool((ld == 0).all()) and bool((sg == 1).all()) and bool((mx == 0).all())


@pytest.mark.parametrize("N,L", [(2, 4), (3, 2)])
def test_flow_logdet_matches_bruteforce_jacobian(N, L):
    """Total log|det J| of the 12 sub-layers (Proposition 1, summed over the stack) equals the slogdet of the
    right-trivialised Jacobian of the whole map on all 2 L^2 links, computed by autograd."""
    torch.manual_seed(1)
    flow = ex.Flow(N, 3)
    _randomise(flow, seed=2, scale=0.2)                              # max ||D omega||_2 0.66 (SU(2)), 0.47 (SU(3))
    G = flow.G
    U = _field(1, L, N)

    def fwd(X: Tensor) -> Tensor:                    # the U path of Flow.forward without the log-determinant
        for (mu, p), cf in zip(flow.order, flow.coeffs):
            X = masked_layer(X, mu, p, cf, G, logdet=False)
        return X

    nl = U[..., 0, 0].numel()
    I = torch.eye(N, dtype=C)
    out0 = fwd(U).detach()

    def f(xi: Tensor) -> Tensor:
        return G.vee(fwd((I + G.hat(xi)).reshape(U.shape) @ U) @ out0.mH).reshape(nl, G.d)
    try:
        J = jacobian(f, torch.zeros(nl, G.d), vectorize=True)
    except Exception:
        J = jacobian(f, torch.zeros(nl, G.d))
    sign, ref = torch.linalg.slogdet(J.reshape(nl * G.d, nl * G.d).detach())
    Uo, ld, sg, mx = flow(U, create_graph=False)
    assert float((Uo - out0).abs().max()) == 0.0
    assert abs(float(ld[0]) - float(ref)) < 1e-10
    assert float(sg[0]) == 1.0 and float(sign) == 1.0
    assert 0.05 < float(mx[0]) < 1                                   # non-trivial, and inside Proposition 3


def test_few_training_steps_decrease_loss():
    """A few reverse-KL steps (SU(2), one sweep): finite, det J > 0, loss down on a fixed batch."""
    torch.manual_seed(0)
    flow = ex.Flow(2, 1)
    L, beta = 4, 2.0
    U_fixed = _field(256, L, 2)

    def fixed_loss() -> float:
        Uo, ld, sg, _ = flow(U_fixed, create_graph=False)
        assert bool((sg > 0).all())
        return float(-(-wilson_action(Uo, beta) + ld).mean())

    before = fixed_loss()
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    hist = ex.train(flow, opt, sched, 2, L, beta, 10, 64, exact_logZ_2d(beta, L * L, 2))
    after = fixed_loss()
    assert [h["step"] for h in hist] == [0, 9]
    assert all(math.isfinite(h["KL"]) and h["min sign"] == 1.0 for h in hist)
    assert all(bool(torch.isfinite(p).all()) for p in flow.parameters())
    assert after < before - 1.0, (before, after)                   # KL of the identity flow is 7.3 here


def _record(L: int, n: int = 1000, z: float = 0.5, kl: float = 0.05, var: float = 0.01, sign: float = 1.0,
            dw: float = 0.3, ess: float = 0.9, zpl: float = 0.5) -> tuple[dict, dict]:
    common = {"L": L, "V": L * L, "n": n, "ESS": ess, "Var(log w) per plaquette": var,
              "min sign det J": sign, "max ||D omega||_2": dw}
    set1 = dict(common, **{"log Z z-score": z, "reverse KL (exact log Z + E[S - logdet])": kl})
    set2 = dict(common, **{"log Z z": z, "reverse KL": kl, "plaquette reweighted z": zpl})
    return set1, set2


def test_acceptance_logic():
    """Each section-8 check passes on good records and fails on the one record that violates it."""
    def checks(bad_set: int, **bad):
        s1, s2 = zip(*(_record(L) for L in (4, 8)))
        s1, s2 = list(s1), list(s2)
        if bad:
            b1, b2 = _record(8, **bad)
            if bad_set in (0, 1): s1[1] = b1
            if bad_set in (0, 2): s2[1] = b2
        return dict(ex.acceptance(s1, s2, 4, 0.8))

    assert all(checks(0).values()) and len(checks(0)) == 6
    se = math.sqrt(0.01 * 64 / 1000)
    for bad_set, bad, failing in [(1, {"z": -3.5}, "log Z"), (2, {"zpl": 3.2}, "plaquette"),
                                  (1, {"kl": -4 * se}, "reverse KL"), (2, {"sign": -1.0}, "det J"),
                                  (1, {"dw": 1.05}, "D omega")]:
        res = checks(bad_set, **bad)
        assert [k for k, ok in res.items() if not ok] == [k for k in res if failing in k], (bad, res)
    assert all(checks(1, kl=-2 * se).values())                     # within noise
    s1, s2 = zip(*(_record(L, ess=0.75 if L == 4 else 0.1) for L in (4, 8)))
    res = dict(ex.acceptance(list(s1), list(s2), 4, 0.8))
    assert [k for k, ok in res.items() if not ok] == [k for k in res if "ESS" in k]
    assert len(ex.acceptance(list(s1), list(s2), 4, None)) == 5    # no threshold: no ESS check


def test_tiny_run_save_reload(tmp_path):
    """The command-line path end to end on a tiny SU(2) run; --load reproduces the evaluation of the trained flow
    bit for bit; dtype and thread count are restored afterwards."""
    torch.set_default_dtype(torch.float32)
    threads = torch.get_num_threads()
    out = str(tmp_path / "tiny")
    args = ["--N", "2", "--steps", "3", "--batch", "16", "--eval-L", "4", "--eval-n", "48", "--eval2-n", "40",
            "--threads", "2"]
    res = ex.run(args + ["--out", out])
    assert torch.get_default_dtype() == torch.float32 and torch.get_num_threads() == threads
    assert res["parameters"] == 17760 and [h["step"] for h in res["history"]] == [0, 2]
    assert res["args"]["beta"] == 2.0 and res["args"]["ess_min"] == 0.65
    assert [r["n"] for r in res["eval"]] == [48] and [r["n"] for r in res["eval2"]] == [40]
    assert len(res["acceptance"]) == 6 and all(isinstance(ok, bool) for _, ok in res["acceptance"])
    for r in res["eval"] + res["eval2"]:
        assert r["min sign det J"] == 1.0 and math.isfinite(r["log Z"]) and 0 < r["ESS"] <= 1
    assert Path(out + ".pt").exists() and Path(out + ".json").exists()
    again = ex.run(args + ["--load", out + ".pt"])
    drop = lambda r: {k: v for k, v in r.items() if k != "seconds"}
    assert [drop(r) for r in again["eval"]] == [drop(r) for r in res["eval"]]
    assert [drop(r) for r in again["eval2"]] == [drop(r) for r in res["eval2"]]
    assert "history" not in again


@pytest.mark.skipif(os.environ.get("LAG_FULL_LATTICE_FLOW") != "1",
                    reason="full section-8 runs (about 50 min for SU(3) and 21 min for SU(2) on 6 CPU threads); "
                           "set LAG_FULL_LATTICE_FLOW=1 to run them")
@pytest.mark.parametrize("N", [3, 2])
def test_full_run_acceptance(N):
    """The section-8 run for SU(N): train at L = 4, evaluate two sets at L = 4, 8, 16; every acceptance check passes."""
    assert ex.main(["--N", str(N)]) == 0
