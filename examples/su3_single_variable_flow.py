"""Normalizing flow on a single SU(3) variable: the end-to-end validation of MATH_SPEC.md section 7.

Build order MATH_SPEC.md section 10, step 5. Target density with respect to Haar measure on SU(3):

    p(U) = exp((beta/3) Re tr U) / Z,    beta = 6.

Model: six layers U -> exp(omega(U)) U, each with the loop head of section 4.3 for W = U,

    omega = proj_su(c1(t) W + c2(t) W^2),    t = tr W,

where (Re c1, Im c1, Re c2, Im c2) come from a 2 -> 32 -> 32 -> 4 tanh MLP of (Re t, Im t) / 3 whose output
layer is zero-initialised, so the untrained flow is the identity. That is 1284 parameters per layer and 7704
in total, the count of section 7. The spec does not record the activation, input scaling, batch size or
learning rate; those are choices of this example.

Per layer, log|det J| = log|det(1 + dexpL_v Dv)| (section 3.1) with v = vee(omega), dexpL_v from
SU3().dexpL (the generic block exponential of section 2.3) and Dv from the xi-perturbation of section 3.5.
Only first derivatives of su3_exp are needed: Dv perturbs U before the layer's exponential.
Haar samples are the QR decomposition of a complex Gaussian matrix with the phase fix, divided by a cube root
of the determinant. Training minimises the reverse KL divergence with Adam. Exact reference values come from
the Weyl integration formula on a 600 x 600 torus grid.

Run from the repository root:

    python examples/su3_single_variable_flow.py                          # the section-7 run
    python examples/su3_single_variable_flow.py --steps 40 --batch 256   # a quick look

The exit status is 0 when every acceptance check passes and 1 otherwise.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

try:
    import lag  # noqa: F401
except ImportError:                                    # source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lag.exp import su3_exp
from lag.groups import SU3, MatrixLieGroup
from lag.heads import head_loops
from lag.jacobian import logabsdet_step

F64 = torch.float64
C128 = torch.complex128
BETA = 6.0
_I3 = torch.eye(3, dtype=C128)

# defaults of the section-7 run
N_LAYERS = 6
HIDDEN = 32
STEPS = 350
BATCH = 1024
LR = 1e-2
EVAL_SAMPLES = 32768

# acceptance thresholds (MATH_SPEC.md sections 3.5 and 7)
ESS_MIN = 0.99
DV_NORM_MAX = 1.0
Z_SCORE_MAX = 3.0


# ------------------------------------------------------------------ target, samples, exact values
def re_tr(U: Tensor) -> Tensor:
    """Re tr U. U: (..., 3, 3) complex -> (...) real."""
    return U.diagonal(dim1=-2, dim2=-1).sum(-1).real


def target_log_density(U: Tensor, beta: float = BETA) -> Tensor:
    """Unnormalised log p(U) = (beta/3) Re tr U with respect to Haar measure. U: (..., 3, 3) -> (...)."""
    return beta / 3 * re_tr(U)


def haar_su3(n: int, generator: torch.Generator | None = None) -> Tensor:
    """n Haar-distributed SU(3) matrices, (n, 3, 3) complex128.

    QR of a complex Gaussian matrix, Q -> Q diag(R_ii / |R_ii|) (Haar on U(3)), then division by the principal
    cube root of det Q. The last map commutes with left multiplication by SU(3), so the result is Haar on SU(3).
    """
    Z = torch.randn(n, 3, 3, dtype=C128, generator=generator)
    Q, R = torch.linalg.qr(Z)
    d = R.diagonal(dim1=-2, dim2=-1)
    Q = Q * (d / d.abs()).unsqueeze(-2)
    return Q / torch.linalg.det(Q).pow(1 / 3)[..., None, None]


def weyl_reference(beta: float = BETA, n: int = 600) -> dict[str, float]:
    """Exact log Z, <Re tr U>/3 and Haar-prior ESS from the Weyl integration formula.

    Eigenphases (t1, t2, -t1-t2) on an n x n grid of the torus; Haar measure induces the density
    |Delta|^2 = prod_{j<k} |e^{i t_j} - e^{i t_k}|^2 (normalised on the grid; its mean is |W| = 3! = 6).
    The periodic trapezoid rule converges spectrally here. ESS of the Haar prior = Z(beta)^2 / Z(2 beta).
    """
    th = torch.arange(n, dtype=F64) * (2 * math.pi / n)
    t1, t2 = torch.meshgrid(th, th, indexing="ij")
    t3 = -(t1 + t2)
    vdm = (4 * torch.sin((t1 - t2) / 2) ** 2 * 4 * torch.sin((t1 - t3) / 2) ** 2
           * 4 * torch.sin((t2 - t3) / 2) ** 2)
    weight = vdm / vdm.sum()
    retr = torch.cos(t1) + torch.cos(t2) + torch.cos(t3)
    boltz = weight * torch.exp(beta / 3 * retr)
    Z = boltz.sum()
    Z2 = (weight * torch.exp(2 * beta / 3 * retr)).sum()
    return {
        "log_Z": float(torch.log(Z)),
        "mean_re_tr_3": float((boltz * retr).sum() / Z / 3),
        "ess_prior": float(Z * Z / Z2),
        "weyl_norm": float(vdm.mean()),
    }


# ------------------------------------------------------------------ model
class LoopLayer(nn.Module):
    """One layer U -> exp(omega(U)) U with the loop head of MATH_SPEC.md section 4.3 for W = U."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=F64), nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=F64), nn.Tanh(),
            nn.Linear(hidden, 4, dtype=F64),
        )
        nn.init.zeros_(self.net[-1].weight)               # identity map at initialisation
        nn.init.zeros_(self.net[-1].bias)

    def coeff(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Invariant coefficients (c1, c2) of the loop head. t = tr W: (B,) complex -> two (B,) complex."""
        c = self.net(torch.stack([t.real, t.imag], -1) / 3)
        return torch.complex(c[..., 0], c[..., 1]), torch.complex(c[..., 2], c[..., 3])

    def forward(self, U: Tensor, group: MatrixLieGroup,
                create_graph: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """U: (B, 3, 3) in SU(3) -> (exp(omega) U, sign det J, log|det J|, Dv); Dv: (B, 8, 8).

        Dv[:, a, b] = d v_a(exp(hat xi) U) / d xi_b at xi = 0 (MATH_SPEC.md section 3.5); first order in xi is
        enough, so U is perturbed as (1 + hat xi) U. create_graph=True keeps Dv differentiable for training;
        with create_graph=False the outputs are detached.
        """
        with torch.enable_grad():
            xi = torch.zeros(U.shape[0], group.dim, dtype=F64, requires_grad=True)
            W = (_I3 + group.hat(xi)) @ U                 # equals U at xi = 0
            omega = head_loops(W, self.coeff)
            v = group.vee(omega)
            Dv = torch.stack([torch.autograd.grad(v[:, a].sum(), xi, create_graph=create_graph,
                                                  retain_graph=True)[0] for a in range(group.dim)], 1)
            sign, logabsdet = logabsdet_step(group.dexpL(v), Dv)        # section 3.1
            U_new = su3_exp(omega) @ U                                    # section 2.4
        if not create_graph:
            U_new, sign, logabsdet, Dv = U_new.detach(), sign.detach(), logabsdet.detach(), Dv.detach()
        return U_new, sign, logabsdet, Dv


class LoopFlow(nn.Module):
    """Stack of LoopLayer's on SU(3); the generic group core supplies hat, vee and dexpL (section 2.3)."""

    def __init__(self, n_layers: int = N_LAYERS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.group = SU3()
        self.layers = nn.ModuleList(LoopLayer(hidden) for _ in range(n_layers))

    def forward(self, U: Tensor, create_graph: bool = True,
                with_norms: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """U: (B, 3, 3) -> (U_K, sum of log|det J|, signs (B, K), spectral norms of Dv (B, K) or None)."""
        logdet = torch.zeros(U.shape[0], dtype=F64)
        signs, norms = [], []
        for layer in self.layers:
            U, sign, ld, Dv = layer(U, self.group, create_graph)
            logdet = logdet + ld
            signs.append(sign.detach())
            if with_norms:
                norms.append(torch.linalg.matrix_norm(Dv.detach(), 2))
        return U, logdet, torch.stack(signs, -1), torch.stack(norms, -1) if with_norms else None


def n_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ------------------------------------------------------------------ training and evaluation
def ess_fraction(logw: Tensor) -> float:
    """Normalised effective sample size (sum w)^2 / (N sum w^2) from log-weights."""
    return float(torch.exp(2 * torch.logsumexp(logw, 0) - torch.logsumexp(2 * logw, 0)) / logw.numel())


def train(flow: LoopFlow, steps: int, batch: int, lr: float, generator: torch.Generator,
          log_every: int = 25, log_Z: float | None = None,
          log: Callable[[str], None] = print) -> list[float]:
    """Reverse-KL training with Adam and a cosine learning-rate decay; returns the per-step losses.

    loss = -mean log w with log w = log p~(U_K) - log q(U_K) and log q(U_K) = -sum_k log|det J_k| (Haar base).
    With the exact log Z, loss + log Z is the reverse KL divergence.
    """
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        U, logdet, signs, _ = flow(haar_su3(batch, generator))
        logw = target_log_density(U) + logdet
        loss = -logw.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        value = float(loss.detach())
        losses.append(value)
        if log_every and (step == 1 or step % log_every == 0 or step == steps):
            kl = f"{value + log_Z:10.3e}" if log_Z is not None else " " * 10
            log(f"  {step:5d}  {value:+10.5f}  {kl}  {100 * ess_fraction(logw.detach()):7.2f} %"
                f"  {int((signs <= 0).sum()):6d}  {time.perf_counter() - t0:8.1f} s")
    return losses


def evaluate(flow: LoopFlow, n: int, generator: torch.Generator, chunk: int = 8192) -> dict[str, Tensor]:
    """Push n fresh Haar samples through the flow (chunks of `chunk`); parameters are not differentiated.

    Returns log-weights of the flow and of the Haar prior, Re tr U_K / 3, det signs (n, K) and ||Dv||_2 (n, K).
    """
    flags = [p.requires_grad for p in flow.parameters()]
    flow.requires_grad_(False)
    out: dict[str, list[Tensor]] = {"logw": [], "logw_prior": [], "x": [], "signs": [], "dv_norms": []}
    try:
        for start in range(0, n, chunk):
            U0 = haar_su3(min(chunk, n - start), generator)
            U, logdet, signs, norms = flow(U0, create_graph=False, with_norms=True)
            out["logw"].append(target_log_density(U) + logdet)
            out["logw_prior"].append(target_log_density(U0))
            out["x"].append(re_tr(U) / 3)
            out["signs"].append(signs)
            out["dv_norms"].append(norms)
    finally:
        for p, f in zip(flow.parameters(), flags):
            p.requires_grad_(f)
    return {k: torch.cat(v) for k, v in out.items()}


def summarize(ev: dict[str, Tensor], ref: dict[str, float]) -> dict[str, float]:
    """Estimates with standard errors and z-scores against the exact values.

    log Z = log mean w (delta-method error); <Re tr U>/3 unweighted (flow samples as they are) and
    self-normalised importance-weighted; reverse KL = log Z_exact - mean log w.
    """
    logw, x = ev["logw"], ev["x"]
    N = logw.numel()
    w = torch.exp(logw - logw.max())
    log_Z = float(torch.logsumexp(logw, 0) - math.log(N))
    log_Z_se = float(w.std() / w.mean() / math.sqrt(N))
    x_mean, x_se = float(x.mean()), float(x.std() / math.sqrt(N))
    p = w / w.sum()
    xw_mean = float((p * x).sum())
    xw_se = float(torch.sqrt((p * p * (x - xw_mean) ** 2).sum()))
    return {
        "n": N,
        "ess": ess_fraction(logw),
        "ess_prior": ess_fraction(ev["logw_prior"]),
        "log_Z": log_Z, "log_Z_se": log_Z_se, "log_Z_z": (log_Z - ref["log_Z"]) / log_Z_se,
        "x": x_mean, "x_se": x_se, "x_z": (x_mean - ref["mean_re_tr_3"]) / x_se,
        "xw": xw_mean, "xw_se": xw_se, "xw_z": (xw_mean - ref["mean_re_tr_3"]) / xw_se,
        "kl": ref["log_Z"] - float(logw.mean()),
        "n_nonpositive_det": int((ev["signs"] <= 0).sum()),
        "max_dv_norm": float(ev["dv_norms"].max()),
    }


def acceptance(sets: list[dict[str, float]]) -> list[tuple[str, bool]]:
    """The acceptance checks of the section-7 run, as (description, passed)."""
    return [
        (f"ESS >= {100 * ESS_MIN:.0f} % on every set", all(s["ess"] >= ESS_MIN for s in sets)),
        ("all det J > 0 (every layer, every sample)", all(s["n_nonpositive_det"] == 0 for s in sets)),
        (f"max ||D omega||_2 < {DV_NORM_MAX:g} (section 3.5)", all(s["max_dv_norm"] < DV_NORM_MAX for s in sets)),
        (f"log Z within {Z_SCORE_MAX:g} standard errors of the exact value",
         all(abs(s["log_Z_z"]) < Z_SCORE_MAX for s in sets)),
        (f"<Re tr U>/3 within {Z_SCORE_MAX:g} standard errors of the exact value (unweighted and reweighted)",
         all(abs(s["x_z"]) < Z_SCORE_MAX and abs(s["xw_z"]) < Z_SCORE_MAX for s in sets)),
    ]


# ------------------------------------------------------------------ command line
def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate on two independent sample sets, print the section-7 table; 0 if every check passes."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=STEPS, help=f"Adam steps (default {STEPS})")
    ap.add_argument("--seed", type=int, default=0, help="seed of initialisation and sampling (default 0)")
    ap.add_argument("--batch", type=int, default=BATCH, help=f"training batch size (default {BATCH})")
    ap.add_argument("--lr", type=float, default=LR, help=f"initial Adam learning rate (default {LR:g})")
    ap.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES,
                    help=f"samples per evaluation set (default {EVAL_SAMPLES})")
    ap.add_argument("--log-every", type=int, default=25, help="print every n steps (default 25)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    flow = LoopFlow()
    ref = weyl_reference()
    print("SU(3) single-variable flow (MATH_SPEC.md section 7)")
    print(f"  {n_parameters(flow)} parameters in {len(flow.layers)} layers; batch {args.batch}, "
          f"{args.steps} Adam steps, lr {args.lr:g} (cosine decay), seed {args.seed}; "
          f"float64, {torch.get_num_threads()} CPU threads, torch {torch.__version__}")
    print("Exact values (Weyl integration formula, 600 x 600 torus grid):")
    print(f"  log Z = {ref['log_Z']:.6f}   <Re tr U>/3 = {ref['mean_re_tr_3']:.6f}   "
          f"ESS of the Haar prior = {100 * ref['ess_prior']:.2f} %")

    print("Training (reverse KL):")
    print(f"  {'step':>5}  {'loss':>10}  {'KL':>10}  {'ESS':>9}  {'det<=0':>6}  {'time':>10}")
    t0 = time.perf_counter()
    train(flow, args.steps, args.batch, args.lr, _generator(args.seed), log_every=args.log_every,
          log_Z=ref["log_Z"])
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    sets = [summarize(evaluate(flow, args.eval_samples, _generator(args.seed + 10_000 * k)), ref) for k in (1, 2)]
    eval_time = time.perf_counter() - t1

    print(f"Results: two independent sets of {args.eval_samples} samples "
          f"(training {train_time:.1f} s, evaluation {eval_time:.1f} s)")
    rows = [
        ("ESS of the trained flow", lambda s: f"{100 * s['ess']:.2f} %", ""),
        ("ESS of the Haar prior (estimate)", lambda s: f"{100 * s['ess_prior']:.2f} %", f"{100 * ref['ess_prior']:.2f} %"),
        ("log Z", lambda s: f"{s['log_Z']:.5f} +- {s['log_Z_se']:.5f}", f"{ref['log_Z']:.5f}"),
        ("<Re tr U>/3, unweighted", lambda s: f"{s['x']:.4f} +- {s['x_se']:.4f}", f"{ref['mean_re_tr_3']:.4f}"),
        ("<Re tr U>/3, reweighted", lambda s: f"{s['xw']:.4f} +- {s['xw_se']:.4f}", f"{ref['mean_re_tr_3']:.4f}"),
        ("z-scores: log Z, unweighted, reweighted",
         lambda s: f"{s['log_Z_z']:+.2f}, {s['x_z']:+.2f}, {s['xw_z']:+.2f}", ""),
        ("reverse KL", lambda s: f"{s['kl']:.2e}", "0"),
        ("det J <= 0 (count over layers x samples)", lambda s: f"{s['n_nonpositive_det']}", "0"),
        ("max ||D omega||_2", lambda s: f"{s['max_dv_norm']:.3f}", "< 1"),
    ]
    print(f"  {'quantity':<42}{'set 1':>22}{'set 2':>22}{'exact':>12}")
    for name, fmt, exact in rows:
        print(f"  {name:<42}{fmt(sets[0]):>22}{fmt(sets[1]):>22}{exact:>12}")

    checks = acceptance(sets)
    print("Acceptance:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
