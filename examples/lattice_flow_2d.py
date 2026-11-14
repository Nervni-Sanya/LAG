"""Masked, gauge-equivariant flow for 2D SU(N) lattice gauge theory on a torus: the end-to-end validation of
MATH_SPEC_LATTICE.md section 8 (build order section 11, step 6).

Model (section 8): every sub-layer updates the links of a mask M(mu, p) with the loop head of MATH_SPEC.md section 4.3
on the two plaquettes through the link, omega = proj_su(sum_s c1_s W_s + c2_s W_s^2). The coefficients come from a
4 -> 32 -> 32 -> 8 tanh MLP of (Re, Im) tr W_s / N for both loops, bounded by 0.3 tanh, with a zero-initialised output
layer, so the untrained flow is the identity. Three sweeps make 12 sub-layers and 17,760 parameters. The total
log-determinant is Proposition 1 with batched Dv (lag.lattice.masked_layer). Training minimises the reverse KL
divergence on L = 4 only; evaluation uses fresh Haar samples at L = 4, 8 and 16 with the same parameters (the model is
local and translation-invariant). Exact log Z and plaquette from the torus character expansion (lag.lattice.exact).

Two independent evaluation sets, as in section 8:
  set 1  `evaluate`, seeds 1000 + L, 16384 / 8192 / 2048 samples (lattice_flow_2d.py, at the end of training);
  set 2  `evaluate_snis`, seeds 5000 + L, 32768 / 16384 / 4096 samples, with self-normalised importance-sampling
         error bars for the reweighted plaquette (eval_saved.py).
Coeff, Flow, evaluate and the training loop are moved verbatim from the reference lattice_flow_2d.py, and
evaluate_snis from eval_saved.py; the acceptance checks and this command line are new.

Acceptance checks (section 8, from single-seed runs; confirm on a second seed before pinning thresholds):
  - ESS >= 0.8 (SU(3), beta = 3) or >= 0.65 (SU(2), beta = 2) at the training volume, on both sets;
  - |z| <= 3 for log Z at every evaluated volume (both sets) and for the reweighted plaquette (set 2, SNIS error);
  - reverse KL >= 0 within noise (at least -3 standard errors of mean log w) at every volume, both sets;
  - all det J > 0 and max ||D omega||_2 < 1, both sets.

Run from the repository root (6 CPU threads by default, as in section 8; times there are for that setting):
    python examples/lattice_flow_2d.py --N 3     # SU(3), beta = 3, 300 steps: 1705 s training + about 21 min evaluation
    python examples/lattice_flow_2d.py --N 2     # SU(2), beta = 2, 400 steps:  833 s training + about 7 min evaluation
    python examples/lattice_flow_2d.py --N 2 --steps 20 --eval-L 4,8 --eval-n 512,128 --eval2-n 512,128   # a quick look
    python examples/lattice_flow_2d.py --N 3 --load lattice_flow_su3_b3.pt    # evaluate a saved model (no training)
--out PREFIX writes PREFIX.pt (after training) and PREFIX.json. The exit status is 0 when every acceptance check
passes and 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

try:
    import lag  # noqa: F401
except ImportError:                                    # source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lag.lattice import GaugeGroup, exact_logZ_2d, haar, masked_layer, plaquettes, wilson_action

# defaults of the section-8 runs, per gauge group: (beta, Adam steps)
SECTION8 = {3: (3.0, 300), 2: (2.0, 400)}
# acceptance thresholds (MATH_SPEC_LATTICE.md section 8)
ESS_MIN = {(3, 3.0): 0.8, (2, 2.0): 0.65}             # at the training volume
Z_MAX = 3.0
KL_NOISE = 3.0                                         # reverse KL >= -KL_NOISE standard errors of mean log w
DW_NORM_MAX = 1.0


# ------------------------------------------------------------------ model (verbatim from lattice_flow_2d.py)
class Coeff(torch.nn.Module):
    """Gauge-invariant coefficients (c1_s, c2_s) for the 2 plaquette loops of a link from the traces of both loops."""
    def __init__(s, n_loops: int, N: int, width: int = 32, scale: float = 0.3) -> None:
        super().__init__(); s.N, s.scale, s.n = N, scale, n_loops
        s.net = torch.nn.Sequential(torch.nn.Linear(2 * n_loops, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 4 * n_loops))
        torch.nn.init.zeros_(s.net[-1].weight); torch.nn.init.zeros_(s.net[-1].bias)

    def forward(s, t: Tensor) -> tuple[Tensor, Tensor]:
        """Traces t (..., n_loops) complex -> (c1, c2), each (..., n_loops) complex, bounded by scale * tanh."""
        o = s.scale * torch.tanh(s.net(torch.cat([t.real, t.imag], -1) / s.N)); n = s.n
        return torch.complex(o[..., :n], o[..., n:2 * n]), torch.complex(o[..., 2 * n:3 * n], o[..., 3 * n:])


class Flow(torch.nn.Module):
    """sweeps x 4 masked sub-layers (mu, p) = (0, 0), (0, 1), (1, 0), (1, 1) on a 2D lattice, one Coeff per sub-layer.

    Needs float64 as the default dtype (main() sets it). forward(U) returns (U', total log|det J| (B,), smallest sign
    of the per-link determinants (B,), max ||D omega||_2 (B,)).
    """
    def __init__(s, N: int, sweeps: int, width: int = 32, scale: float = 0.3) -> None:
        super().__init__(); s.G = GaugeGroup(N); s.order = [(mu, p) for _ in range(sweeps) for mu in (0, 1) for p in (0, 1)]
        s.coeffs = torch.nn.ModuleList([Coeff(2, N, width, scale) for _ in s.order])

    def forward(s, U: Tensor, create_graph: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ld = torch.zeros(U.shape[0]); sg = torch.ones(U.shape[0]); mx = torch.zeros(U.shape[0])
        for (mu, p), cf in zip(s.order, s.coeffs):
            U, l, g, m = masked_layer(U, mu, p, cf, s.G, create_graph=create_graph); ld = ld + l; sg = torch.minimum(sg, g); mx = torch.maximum(mx, m)
        return U, ld, sg, mx


# ------------------------------------------------------------------ training and evaluation
def train(flow: Flow, opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LRScheduler, N: int, L: int,
          beta: float, steps: int, batch: int, ex: float) -> list[dict[str, float]]:
    """Reverse-KL training on fresh Haar samples (global random stream); the loop of lattice_flow_2d.py.

    loss = -mean log w, log w = -S(U') + log|det J|, so loss + exact log Z (ex) is the reverse KL. Records and prints
    the KL, batch ESS, smallest det sign and max ||D omega||_2 every 25 steps and at the last step.
    """
    hist = []; t0 = time.time()
    for it in range(steps):
        U = haar(batch * 2 * L * L, N).reshape(batch, 2, L, L, N, N)
        Uo, ld, sg, mx = flow(U); lw = -wilson_action(Uo, beta) + ld; loss = -lw.mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 25 == 0 or it == steps - 1:
            w = torch.exp(lw.detach() - lw.detach().max()); ess = float(w.sum() ** 2 / (len(w) * (w * w).sum()))
            hist.append({"step": it, "KL": ex + float(loss.detach()), "batch ESS": ess, "min sign": float(sg.min()), "max ||Dw||": float(mx.max()), "s": round(time.time() - t0, 1)})
            print(hist[-1], flush=True)
    return hist


def evaluate(flow: Flow, N: int, L: int, beta: float, n: int, chunk: int) -> dict[str, float]:
    """Set 1 (lattice_flow_2d.py): n fresh Haar samples at L (global random stream; the caller seeds it), in chunks.

    log Z = log mean w with its delta-method error, z-score against the exact value, reverse KL, Var(log w) per
    plaquette, plaquette averages (the error is that of the unweighted mean), det signs, max ||D omega||_2, and the
    exact ESS of the Haar prior. L >= 4 (the chunk formula of the callers divides by L^2 // 16).
    """
    lws, pls, sgs, mxs = [], [], [], []
    for i in range(0, n, chunk):
        b = min(chunk, n - i); U = haar(b * 2 * L * L, N).reshape(b, 2, L, L, N, N)
        Uo, ld, sg, mx = flow(U, create_graph=False)
        lws.append((-wilson_action(Uo, beta) + ld).detach()); sgs.append(sg.detach()); mxs.append(mx.detach())
        pls.append((plaquettes(Uo).diagonal(dim1=-2, dim2=-1).sum(-1).real / N).flatten(1).mean(-1).detach())
    lw, pl = torch.cat(lws), torch.cat(pls); V = L * L
    w = torch.exp(lw - lw.max()); wn = w / w.sum(); ess = float(w.sum() ** 2 / (len(w) * (w * w).sum()))
    lz = float(torch.logsumexp(lw, 0) - math.log(len(lw))); se = float(w.std() / w.mean() / len(w) ** 0.5)
    ex = exact_logZ_2d(beta, V, N); h = 1e-4; ex_pl = (exact_logZ_2d(beta + h, V, N) - exact_logZ_2d(beta - h, V, N)) / (2 * h) / V
    pl_rw = float((wn * pl).sum()); pl_se = float(pl.std() / len(pl) ** 0.5)
    return {"L": L, "V": V, "n": len(lw), "ESS": ess, "ESS^(1/V)": ess ** (1 / V), "log Z": lz, "log Z stderr": se, "log Z exact": ex, "log Z z-score": (lz - ex) / se,
            "reverse KL (exact log Z + E[S - logdet])": ex - float(lw.mean()), "reverse KL per plaquette": (ex - float(lw.mean())) / V,
            "Var(log w) per plaquette": float(lw.var()) / V, "plaquette unweighted": float(pl.mean()), "plaquette reweighted": pl_rw, "plaquette stderr": pl_se,
            "plaquette exact": ex_pl, "min sign det J": float(torch.cat(sgs).min()), "max ||D omega||_2": float(torch.cat(mxs).max()),
            "Haar prior ESS (exact)": math.exp(2 * ex - exact_logZ_2d(2 * beta, V, N))}


def evaluate_snis(flow: Flow, N: int, L: int, beta: float, n: int) -> dict[str, float]:
    """Set 2 (eval_saved.py): like `evaluate`, plus self-normalised importance-sampling (SNIS) errors and z-scores for
    the reweighted plaquette, and the ESS predicted from exp(-Var log w). The caller seeds the global random stream.

    The body of the per-L loop of eval_saved.py with a.N -> N and a.beta -> beta. L >= 4.
    """
    t0 = time.time(); chunk = max(16, 4096 // (L * L // 16)); lw, pl, sg, mx = [], [], [], []
    for i in range(0, n, chunk):
        b = min(chunk, n - i); U = haar(b * 2 * L * L, N).reshape(b, 2, L, L, N, N)
        Uo, ld, s, m = flow(U, create_graph=False)
        lw.append((-wilson_action(Uo, beta) + ld).detach()); sg.append(s.detach()); mx.append(m.detach())
        pl.append((plaquettes(Uo).diagonal(dim1=-2, dim2=-1).sum(-1).real / N).flatten(1).mean(-1).detach())
    lw, pl = torch.cat(lw), torch.cat(pl); V = L * L; w = torch.exp(lw - lw.max()); W = w.sum()
    ess = float(W ** 2 / (len(w) * (w * w).sum())); lz = float(torch.logsumexp(lw, 0) - math.log(len(lw))); lz_se = float(w.std() / w.mean() / len(w) ** 0.5)
    ex = exact_logZ_2d(beta, V, N); h = 1e-4; ex_pl = (exact_logZ_2d(beta + h, V, N) - exact_logZ_2d(beta - h, V, N)) / (2 * h) / V
    rw = float((w * pl).sum() / W); rw_se = float(((w ** 2 * (pl - rw) ** 2).sum()).sqrt() / W)
    un = float(pl.mean()); un_se = float(pl.std() / len(pl) ** 0.5)
    r = {"L": L, "V": V, "n": len(lw), "ESS": ess, "ESS predicted exp(-Var log w)": math.exp(-float(lw.var())), "Var(log w) per plaquette": float(lw.var()) / V,
         "log Z": lz, "log Z stderr": lz_se, "log Z exact": ex, "log Z z": (lz - ex) / lz_se, "reverse KL": ex - float(lw.mean()), "reverse KL per plaquette": (ex - float(lw.mean())) / V,
         "plaquette exact": ex_pl, "plaquette unweighted": un, "plaquette unweighted stderr": un_se, "plaquette unweighted z": (un - ex_pl) / un_se,
         "plaquette reweighted": rw, "plaquette reweighted stderr (SNIS)": rw_se, "plaquette reweighted z": (rw - ex_pl) / rw_se,
         "min sign det J": float(torch.cat(sg).min()), "max ||D omega||_2": float(torch.cat(mx).max()), "Haar prior ESS (exact)": math.exp(2 * ex - exact_logZ_2d(2 * beta, V, N)), "seconds": round(time.time() - t0, 1)}
    return r


# ------------------------------------------------------------------ acceptance
def _kl_and_se(r: dict[str, float]) -> tuple[float, float]:
    """Reverse KL of an evaluation record and the standard error of mean log w, sqrt(Var(log w) / n)."""
    kl = r["reverse KL"] if "reverse KL" in r else r["reverse KL (exact log Z + E[S - logdet])"]
    return kl, math.sqrt(r["Var(log w) per plaquette"] * r["V"] / r["n"])


def acceptance(set1: list[dict[str, float]], set2: list[dict[str, float]], L_train: int,
               ess_min: float | None) -> list[tuple[str, bool]]:
    """The acceptance checks of section 8, as (description, passed). The ESS check needs a threshold (ess_min)."""
    both = set1 + set2
    checks = []
    if ess_min is not None:
        at_train = [r for r in both if r["L"] == L_train]
        checks.append((f"ESS >= {ess_min:g} at the training volume L = {L_train} (both sets)",
                       len(at_train) == 2 and all(r["ESS"] >= ess_min for r in at_train)))
    checks.append((f"|z| <= {Z_MAX:g} for log Z at every volume (both sets)",
                   all(abs(r["log Z z"] if "log Z z" in r else r["log Z z-score"]) <= Z_MAX for r in both)))
    checks.append((f"|z| <= {Z_MAX:g} for the reweighted plaquette at every volume (set 2, SNIS error)",
                   all(abs(r["plaquette reweighted z"]) <= Z_MAX for r in set2)))
    checks.append((f"reverse KL >= 0 within noise (>= -{KL_NOISE:g} standard errors) at every volume (both sets)",
                   all(kl >= -KL_NOISE * se for kl, se in map(_kl_and_se, both))))
    checks.append(("all det J > 0 (every sub-layer, link and sample; both sets)", all(r["min sign det J"] > 0 for r in both)))
    checks.append((f"max ||D omega||_2 < {DW_NORM_MAX:g} (both sets)", all(r["max ||D omega||_2"] < DW_NORM_MAX for r in both)))
    return checks


# ------------------------------------------------------------------ command line
def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=3, choices=(2, 3), help="SU(N), N = 2 or 3 (default 3)")
    ap.add_argument("--L", type=int, default=4, help="training lattice size (default 4)")
    ap.add_argument("--beta", type=float, default=None, help="coupling (default: 3 for SU(3), 2 for SU(2), as in section 8)")
    ap.add_argument("--sweeps", type=int, default=3, help="sweeps of 4 sub-layers (default 3)")
    ap.add_argument("--steps", type=int, default=None, help="Adam steps (default: 300 for SU(3), 400 for SU(2))")
    ap.add_argument("--batch", type=int, default=256, help="training batch (default 256)")
    ap.add_argument("--lr", type=float, default=5e-3, help="Adam learning rate, cosine decay (default 0.005)")
    ap.add_argument("--seed", type=int, default=0, help="seed of initialisation and training samples (default 0)")
    ap.add_argument("--eval-L", type=str, default="4,8,16", help="evaluation lattice sizes, even and >= 4 (default 4,8,16)")
    ap.add_argument("--eval-n", type=str, default="16384,8192,2048", help="set-1 samples per size (default 16384,8192,2048)")
    ap.add_argument("--eval2-n", type=str, default="32768,16384,4096", help="set-2 samples per size (default 32768,16384,4096)")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (default 6)")
    ap.add_argument("--ess-min", type=float, default=None, help="ESS threshold at the training volume (default: section 8)")
    ap.add_argument("--load", type=str, default=None, help="evaluate this saved state dict instead of training")
    ap.add_argument("--out", type=str, default=None, help="write OUT.pt after training and OUT.json at the end")
    a = ap.parse_args(argv)
    beta, steps = SECTION8[a.N]
    a.beta = beta if a.beta is None else a.beta
    a.steps = steps if a.steps is None else a.steps
    a.ess_min = ESS_MIN.get((a.N, a.beta)) if a.ess_min is None else a.ess_min
    if len(_ints(a.eval_L)) != len(_ints(a.eval_n)) or len(_ints(a.eval_L)) != len(_ints(a.eval2_n)):
        ap.error("--eval-L, --eval-n and --eval2-n need the same number of entries")
    if any(Le < 4 or Le % 2 for Le in _ints(a.eval_L)) or a.L % 2:
        ap.error("lattice sizes must be even, and evaluation sizes at least 4")
    return a


def run(argv: list[str] | None = None) -> dict:
    """Train (or load), evaluate both sets, check acceptance; returns the record, with "acceptance" as (name, passed).

    Sets float64 as the default dtype and the requested thread count for the run and restores both afterwards.
    """
    a = parse_args(argv)
    prev_dtype, prev_threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    try:
        return _run(a)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_num_threads(prev_threads)


def _run(a: argparse.Namespace) -> dict:
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    flow = Flow(a.N, a.sweeps)
    n_par = sum(p.numel() for p in flow.parameters())
    res = {"args": vars(a).copy(), "parameters": n_par, "torch": torch.__version__, "eval": [], "eval2": []}
    print(f"SU({a.N}) L={a.L} beta={a.beta}: {n_par} parameters, {len(flow.order)} sub-layers; float64, "
          f"{torch.get_num_threads()} CPU threads, torch {torch.__version__}", flush=True)
    if a.load:
        flow.load_state_dict(torch.load(a.load)); flow.eval()
        print(f"loaded {a.load}; no training", flush=True)
    else:
        opt = torch.optim.Adam(flow.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        ex = exact_logZ_2d(a.beta, a.L * a.L, a.N); t0 = time.time()
        print(f"training: {a.steps} Adam steps, batch {a.batch}, lr {a.lr:g} (cosine decay), seed {a.seed}; exact log Z {ex:.5f}", flush=True)
        res["history"] = train(flow, opt, sched, a.N, a.L, a.beta, a.steps, a.batch, ex)
        res["train seconds"] = round(time.time() - t0, 1)
        if a.out:
            torch.save(flow.state_dict(), a.out + ".pt")
    rnd = lambda r: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
    for Le, ne in zip(_ints(a.eval_L), _ints(a.eval_n)):                    # set 1, as lattice_flow_2d.py
        te = time.time(); torch.manual_seed(1000 + Le)
        r = evaluate(flow, a.N, Le, a.beta, ne, chunk=max(16, 4096 // (Le * Le // 16))); r["seconds"] = round(time.time() - te, 1); res["eval"].append(r)
        print("set 1", rnd(r), flush=True)
    for Le, ne in zip(_ints(a.eval_L), _ints(a.eval2_n)):                   # set 2, as eval_saved.py
        torch.manual_seed(5000 + Le)
        r = evaluate_snis(flow, a.N, Le, a.beta, ne); res["eval2"].append(r)
        print("set 2", rnd(r), flush=True)
    res["acceptance"] = acceptance(res["eval"], res["eval2"], a.L, a.ess_min)
    _report(res, a)
    if a.out:
        json.dump(res, open(a.out + ".json", "w"), indent=1)
    return res


def _report(res: dict, a: argparse.Namespace) -> None:
    print(f"\nSU({a.N}), beta = {a.beta}: evaluation (set 1 / set 2)")
    print(f"  {'L':>3} {'samples':>13} {'ESS':>15} {'log Z (set 1)':>20} {'log Z (set 2)':>20} {'exact':>9} {'z':>13}"
          f" {'plaquette rw (SNIS z)':>24} {'exact':>8} {'KL':>7} {'min sign':>8} {'max||Dw||':>9}")
    for u, v in zip(res["eval"], res["eval2"]):
        print(f"  {v['L']:>3} {u['n']:>6}/{v['n']:<6} {100 * u['ESS']:6.1f}%/{100 * v['ESS']:5.1f}%"
              f" {u['log Z']:10.4f}+-{u['log Z stderr']:.4f} {v['log Z']:10.4f}+-{v['log Z stderr']:.4f} {v['log Z exact']:9.4f}"
              f" {u['log Z z-score']:+6.2f}/{v['log Z z']:+5.2f} {v['plaquette reweighted']:9.5f}+-{v['plaquette reweighted stderr (SNIS)']:.5f}"
              f" ({v['plaquette reweighted z']:+4.1f}) {v['plaquette exact']:8.5f} {v['reverse KL']:7.3f}"
              f" {min(u['min sign det J'], v['min sign det J']):+8.0f} {max(u['max ||D omega||_2'], v['max ||D omega||_2']):9.3f}")
    if a.ess_min is None:
        print(f"  (no ESS threshold for SU({a.N}) at beta = {a.beta}; pass --ess-min to check one)")
    print("Acceptance:")
    for name, ok in res["acceptance"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def main(argv: list[str] | None = None) -> int:
    """Run the example; 0 if every acceptance check passes, 1 otherwise."""
    res = run(argv)
    return 0 if all(ok for _, ok in res["acceptance"]) else 1


if __name__ == "__main__":
    sys.exit(main())
