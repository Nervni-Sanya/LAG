"""Acceptance suite for lag.lattice: numerical verification of MATH_SPEC_LATTICE.md.

Ported from verify_lattice.py. Each of the 64 recorded results is one pytest item (named by the check) and maps to
a row of MATH_SPEC_LATTICE.md section 9 or to the volume table of section 4. The seeds are those of the original, and
the tolerances with their direction (upper bound, or lower bound for a negative control) are pinned in EXPECTED_TOL.
The 11 results the original only records are asserted against the qualitative claims of the spec (QUALITATIVE).

Differences from verify_lattice.py; each leaves every recorded value bit for bit unchanged on the same torch version:
  - lag.lattice.mask refuses odd L (MATH_SPEC_LATTICE.md section 11), so the two odd-L negative controls pass the
    parity set explicitly through `active` (parity_set below, the reference mask without the check);
  - sec_inverse calls lag.lattice.invert_masked_layer, the fixed-point iteration of the original moved into the library;
  - check() also records each tolerance and its direction in TOL.

Run:
    python -m pytest tests/test_lattice.py -v        (or: pytest tests/ -v)
    python tests/test_lattice.py                     (CLI: writes verify_lattice_results.json to the current directory,
                                                       exits with status 1 if any check fails)
"""
import pathlib, sys
if __name__ == "__main__":  # run as a script: make the repository root importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, math, os, time, torch
import pytest
from torch.autograd.functional import jacobian
from lag.lattice import (C, GaugeGroup, exact_logZ_2d, gauge_transform, haar, head, invert_masked_layer, logdet_spectral,
                         mask, masked_layer, staples, validate_mask, wilson_action)
R, FAIL, TOL = {}, [], {}
def nanmax(xs):
    """max that propagates NaN: Python's max skips a NaN unless it comes first, which would hide a failure."""
    xs = [float(x) for x in xs]
    return float("nan") if any(math.isnan(x) for x in xs) else max(xs)
def check(name, value, tol, lower=False):
    R[name] = value; TOL[name] = (tol, lower); v = nanmax(value) if isinstance(value, (list, tuple)) else value
    ok = (v >= tol) if lower else (v <= tol)
    if not ok: FAIL.append((name, v, tol))
def rand_coeff(n_loops, scale=0.25, seed=0, only_first=False):
    g = torch.Generator().manual_seed(seed); W1 = torch.randn(4 * n_loops, 2 * n_loops, generator=g) / 2; b1 = torch.randn(4 * n_loops, generator=g)
    def coeff(t):
        n = t.shape[-1]; o = scale * torch.tanh(torch.cat([t.real, t.imag], -1) @ W1[:4 * n, :2 * n].T + b1[:4 * n])
        c1, c2 = torch.complex(o[..., :n], o[..., n:2 * n]), torch.complex(o[..., 2 * n:3 * n], o[..., 3 * n:])
        if only_first:
            keep = torch.zeros(n); keep[0] = 1; c1, c2 = c1 * keep, c2 * keep
        return c1, c2
    return coeff
def field(B, D, L, N): return haar(B * D * L ** D, N).reshape(B, D, *[L] * D, N, N)
def parity_set(L, D, mu, p):
    """The reference mask without the even-L check of lag.lattice.mask: for the odd-L negative controls only, where
    the transverse parity wraps around the torus and the set is not independent."""
    grids = torch.meshgrid(*[torch.arange(L)] * D, indexing="ij")
    return (sum(g for nu, g in enumerate(grids) if nu != mu) % 2) == p

def full_jacobian(fn, U, G):
    """Right-trivialised Jacobian of a map on all links, by autograd: (n_links d, n_links d). U has batch 1."""
    N, d = G.N, G.d; nl = U[..., 0, 0].numel(); out0 = fn(U); I = torch.eye(N, dtype=C)
    def f(xi):
        return G.vee(fn((I + G.hat(xi)).reshape(U.shape) @ U) @ out0.mH).reshape(nl, d)
    try: J = jacobian(f, torch.zeros(nl, d), vectorize=True)
    except Exception: J = jacobian(f, torch.zeros(nl, d))
    return J.reshape(nl * d, nl * d)

# ---------------------------------------------------------------- L1: block structure and total log-determinant
def sec_blocks():
    torch.manual_seed(0)
    for (D, L, N, subs) in ((2, 4, 2, "all"), (2, 4, 3, "all"), (3, 4, 2, [(0, 0), (2, 1)]), (4, 2, 2, [(0, 0), (3, 1)]),
                            (3, 4, 3, [(1, 0), (2, 1)]), (4, 2, 3, [(0, 1), (3, 0)])):
        G = GaugeGroup(N); d = G.d; U = field(1, D, L, N); V = L ** D; tag = f"D={D} L={L} SU({N})"
        subs = [(mu, p) for mu in range(D) for p in (0, 1)] if subs == "all" else subs
        e_ld, e_off, e_fz = [], [], []
        for k, (mu, p) in enumerate(subs):
            cf = rand_coeff(2 * (D - 1), seed=10 + k)
            J = full_jacobian(lambda X: masked_layer(X, mu, p, cf, G, logdet=False), U, G)
            _, ld, sg, _ = masked_layer(U, mu, p, cf, G)
            act = torch.zeros(D * V, dtype=torch.bool); act[mu * V + mask(L, D, mu, p).reshape(-1).nonzero().squeeze(-1)] = True
            ia = act.repeat_interleave(d); JAA = J[ia][:, ia]; nA = int(act.sum())
            blk = JAA.reshape(nA, d, nA, d).permute(0, 2, 1, 3); off = blk.clone(); off[torch.arange(nA), torch.arange(nA)] = 0
            e_off.append(float(off.abs().max())); e_fz.append(float(torch.maximum(J[~ia][:, ia].abs().max(), (J[~ia][:, ~ia] - torch.eye(int((~ia).sum()))).abs().max())))
            e_ld.append(float(abs(torch.linalg.slogdet(J)[1] - ld[0])))
        check(f"{tag}: |logdet brute force - masked formula| over {len(subs)} sub-layers", nanmax(e_ld), 1e-9)
        check(f"{tag}: off-diagonal blocks of J_AA", nanmax(e_off), 1e-12)
        check(f"{tag}: frozen rows of J differ from identity", nanmax(e_fz), 1e-12)
        if D == 2:                                     # negative control: every mu-link active at once
            cf = rand_coeff(2, seed=99); allm = torch.ones(L, L, dtype=torch.bool)
            J = full_jacobian(lambda X: masked_layer(X, 0, 0, cf, G, logdet=False, active=allm), U, G)
            _, ld, _, _ = masked_layer(U, 0, 0, cf, G, active=allm)
            ia = torch.zeros(D * V, dtype=torch.bool); ia[:V] = True; ia = ia.repeat_interleave(d); JAA = J[ia][:, ia]
            blk = JAA.reshape(V, d, V, d).permute(0, 2, 1, 3).clone(); blk[torch.arange(V), torch.arange(V)] = 0
            check(f"{tag} NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero", float(blk.abs().max()), 1e-3, lower=True)
            check(f"{tag} NEGATIVE CONTROL: |logdet brute force - sum of per-link dets| is non-zero", float(abs(torch.linalg.slogdet(J)[1] - ld[0])), 1e-3, lower=True)
            # full sweep = composition of the 2D sub-layers
            cfs = [rand_coeff(2, seed=40 + k) for k in range(4)]; order = [(mu, p) for mu in range(2) for p in (0, 1)]
            def sweep(X, with_ld=False):
                tot = 0.0
                for (mu, p), cf in zip(order, cfs):
                    if with_ld: X, ld, _, _ = masked_layer(X, mu, p, cf, G); tot = tot + ld
                    else: X = masked_layer(X, mu, p, cf, G, logdet=False)
                return (X, tot) if with_ld else X
            J = full_jacobian(sweep, U, G); _, tot = sweep(U, True)
            check(f"{tag}: full sweep (4 sub-layers) |logdet brute force - sum of masked formulas|", float(abs(torch.linalg.slogdet(J)[1] - tot[0])), 1e-8)
            R[f"{tag}: full sweep changes every link"] = bool(((sweep(U) - U).abs().amax((-1, -2)) > 0).all())
    G = GaugeGroup(2); U = field(1, 2, 3, 2); cf = rand_coeff(2, seed=98); odd = parity_set(3, 2, 0, 0)      # odd L: transverse parity wraps around the torus (mask() refuses it)
    J = full_jacobian(lambda X: masked_layer(X, 0, 0, cf, G, logdet=False, active=odd), U, G); _, ld, _, _ = masked_layer(U, 0, 0, cf, G, active=odd)
    check("D=2 L=3 SU(2) NEGATIVE CONTROL (odd L): |logdet brute force - masked formula| is non-zero", float(abs(torch.linalg.slogdet(J)[1] - ld[0])), 1e-3, lower=True)

# ---------------------------------------------------------------- L1b: numerical mask validator (section 3.3)
def sec_validate():
    torch.manual_seed(7)
    G = GaugeGroup(3); U = field(1, 2, 4, 3); cf = rand_coeff(2, seed=21)
    check("validate_mask D=2 L=4 SU(3) M(0,0): change of other active connections (valid mask, must be 0)", validate_mask(U, 0, 0, cf, G), 0.0)
    G2 = GaugeGroup(2); U3 = field(1, 3, 4, 2); cf3 = rand_coeff(4, seed=22)
    check("validate_mask D=3 L=4 SU(2) M(1,1): change of other active connections (valid mask, must be 0)", validate_mask(U3, 1, 1, cf3, G2), 0.0)
    check("validate_mask NEGATIVE CONTROL (every mu-link active): change detected", validate_mask(U, 0, 0, cf, G, active=torch.ones(4, 4, dtype=torch.bool)), 1e-6, lower=True)
    check("validate_mask NEGATIVE CONTROL (odd L = 3): change detected", validate_mask(field(1, 2, 3, 3), 0, 0, cf, G, active=parity_set(3, 2, 0, 0)), 1e-6, lower=True)

# ---------------------------------------------------------------- L2: gauge covariance, invariance of the log-determinant
def sec_gauge():
    torch.manual_seed(1)
    for (D, L, N) in ((2, 4, 3), (3, 4, 2), (4, 2, 3)):
        G = GaugeGroup(N); U = field(2, D, L, N); g = haar(2 * L ** D, N).reshape(2, *[L] * D, N, N); cf = rand_coeff(2 * (D - 1), seed=3)
        U1, ld, _, _ = masked_layer(U, D - 1, 1, cf, G); U2, ld2, _, _ = masked_layer(gauge_transform(U, g), D - 1, 1, cf, G)
        check(f"D={D} L={L} SU({N}): gauge covariance of the layer, gauge invariance of log det", [float((U2 - gauge_transform(U1, g)).abs().max()), float((ld - ld2).abs().max())], 1e-11)
        check(f"D={D} L={L} SU({N}): output stays in SU(N) (unitarity, det)", [float((U1.mH @ U1 - torch.eye(N)).abs().max()), float((torch.linalg.det(U1) - 1).abs().max())], 1e-12)

# ---------------------------------------------------------------- L3: spectral formula for a single unitary loop
def sec_spectral():
    torch.manual_seed(2)
    for N in (2, 3):
        G = GaugeGroup(N); W = haar(300, N); cf = rand_coeff(1, seed=5)
        xi = torch.zeros(300, G.d, requires_grad=True); om = lambda X: G.vee(head(X.unsqueeze(-3), cf))
        w_xi = om((torch.eye(N, dtype=C) + G.hat(xi)) @ W)
        Dw = torch.stack([torch.autograd.grad(w_xi[:, a].sum(), xi, retain_graph=True)[0] for a in range(G.d)], -2)
        ld = torch.linalg.slogdet(torch.eye(G.d) + G.dexpL(om(W)) @ Dw)[1]
        check(f"SU({N}) single loop: |general d x d formula - spectral formula| (300 Haar samples)", float((ld - logdet_spectral(W, cf)).abs().max()), 1e-8)
    # on the lattice: if only one plaquette enters omega, the link Jacobian equals the spectral formula at that plaquette
    for N in (2, 3):
        G = GaugeGroup(N); D, L = 2, 4; U = field(2, D, L, N)
        idx = mask(L, D, 0, 1).reshape(-1).nonzero().squeeze(-1)
        Ua = U[:, 0].reshape(2, L * L, N, N)[:, idx]; St = staples(U, 0).reshape(2, L * L, 2, N, N)[:, idx]
        W0 = Ua @ St[:, :, 0]; ctx = (Ua @ St[:, :, 1]).diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        cfB = rand_coeff(2, seed=9)
        def cf_mixed(t):                                # omega uses loop 1 only, but its coefficients also read loop 2
            c1, c2 = cfB(t); keep = torch.tensor([1.0, 0.0]); return c1 * keep, c2 * keep
        def cf_frozen(t):                               # the naive shortcut: loop-2 trace treated as frozen context
            c1, c2 = cfB(torch.cat([t, ctx], -1)); return c1[..., :1], c2[..., :1]
        _, ld_true, _, _ = masked_layer(U, 0, 1, cf_mixed, G)
        check(f"SU({N}) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong", float((ld_true - logdet_spectral(W0, cf_frozen).sum(-1)).abs().max()), 1e-3, lower=True)
        # class-function case: coefficients depend on tr W0 only
        cfA = rand_coeff(1, seed=7)
        def cf_two(t): c1, c2 = cfA(t[..., :1]); z = torch.zeros_like(c1); return torch.cat([c1, z], -1), torch.cat([c2, z], -1)
        _, ldA, _, _ = masked_layer(U, 0, 1, cf_two, G)
        check(f"SU({N}) 2D lattice, single-plaquette class-function kernel: |masked formula - sum of spectral formulas|", float((ldA - logdet_spectral(W0, cfA).sum(-1)).abs().max()), 1e-8)

# ---------------------------------------------------------------- L4: inverse of a masked layer by fixed-point iteration
def sec_inverse():
    torch.manual_seed(3)
    for N in (2, 3):
        G = GaugeGroup(N); D, L = 2, 4; U = field(2, D, L, N); cf = rand_coeff(2, scale=0.1, seed=8); mu, p = 1, 0
        Uout, ld, sg, nm = masked_layer(U, mu, p, cf, G)
        idx = mask(L, D, mu, p).reshape(-1).nonzero().squeeze(-1); V = L ** D
        Uin, it, res = invert_masked_layer(Uout, mu, p, cf, G)          # staples are frozen: same as in U; tol 1e-14, at most 200 iterations
        X = Uin[:, mu].reshape(2, V, N, N)[:, idx]
        check(f"SU({N}) fixed-point inverse: |recovered - original active links|", float((X - U[:, mu].reshape(2, V, N, N)[:, idx]).abs().max()), 1e-12)
        R[f"SU({N}) fixed-point inverse: iterations, max ||D omega||_2"] = [it, float(nm.max())]

# ---------------------------------------------------------------- L5: exact partition function on the 2D torus
def sec_exact():
    torch.manual_seed(4)
    check("SU(3): log c_0(beta=6) from exact_logZ_2d(V -> large)/V vs single-variable log Z 1.235890", abs(exact_logZ_2d(6.0, 10 ** 6, 3) / 10 ** 6 - 1.2358900916), 1e-8)
    for N in (2, 3):
        check(f"SU({N}): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16", abs(exact_logZ_2d(6.0, 16, N, pmax=8) - exact_logZ_2d(6.0, 16, N, pmax=5)), 1e-10)
        beta, L = 1.5, 2; n = 400000; lw = -wilson_action(field(n, 2, L, N), beta)
        est = torch.logsumexp(lw, 0) - math.log(n); w = torch.exp(lw - lw.max()); se = float(w.std() / w.mean() / n ** 0.5)
        ex = exact_logZ_2d(beta, L * L, N)
        R[f"SU({N}) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact"] = [float(est), se, ex]
        check(f"SU({N}) 2x2 torus beta=1.5: |log Z Monte Carlo - exact character expansion| in stderr units", abs(float(est) - ex) / se, 4.0)
        R[f"SU({N}) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison)"] = exact_logZ_2d(beta, L * L, N, pmax=0)
    for N, beta in ((2, 3.0), (3, 4.0)):                # 1x1 torus: Z = sum_r a_r, every representation enters linearly
        n = 1_000_000; lw = torch.cat([-wilson_action(field(n // 4, 2, 1, N), beta) for _ in range(4)])
        est = float(torch.logsumexp(lw, 0) - math.log(n)); w = torch.exp(lw - lw.max()); se = float(w.std() / w.mean() / n ** 0.5)
        ex = exact_logZ_2d(beta, 1, N, pmax=12); triv = exact_logZ_2d(beta, 1, N, pmax=0)
        R[f"SU({N}) 1x1 torus beta={beta}: log Z Monte Carlo, stderr, exact"] = [est, se, ex]
        check(f"SU({N}) 1x1 torus beta={beta}: |log Z Monte Carlo - exact| in stderr units", abs(est - ex) / se, 4.0)
        check(f"SU({N}) 1x1 torus beta={beta}: trivial-representation-only value excluded, in stderr units", abs(triv - ex) / se, 20.0, lower=True)

# ---------------------------------------------------------------- L6: cost versus volume
def sec_scaling():
    torch.manual_seed(5); torch.set_num_threads(max(1, os.cpu_count() or 1)); G = GaugeGroup(3); rows = []
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 8))
    def coeff(t):
        o = 0.2 * torch.tanh(net(torch.cat([t.real, t.imag], -1) / 3)); n = t.shape[-1]
        return torch.complex(o[..., :n], o[..., n:2 * n]), torch.complex(o[..., 2 * n:3 * n], o[..., 3 * n:])
    for L in (4, 8, 16, 32):
        U = field(1, 2, L, 3); ts = []
        for _ in range(4):
            t0 = time.perf_counter(); _, ld, _, _ = masked_layer(U, 0, 0, coeff, G); t1 = time.perf_counter()
            net.zero_grad(); ld.sum().backward(); ts.append((t1 - t0, time.perf_counter() - t1))
        fw, bw = sorted(t[0] for t in ts)[1], sorted(t[1] for t in ts)[1]; nA = L * L // 2
        rows.append({"L": L, "links": 2 * L * L, "active links": nA, "ms layer + log det": round(1e3 * fw, 1), "ms backward": round(1e3 * bw, 1), "microseconds per active link": round(1e6 * (fw + bw) / nA, 1)})
    R["2D SU(3), one sub-layer, batch 1: cost versus volume (8 reverse passes for D omega at every volume)"] = rows
    check("cost per active link: ratio L=32 / L=8 (linear scaling means <= about 1)", rows[3]["microseconds per active link"] / rows[1]["microseconds per active link"], 1.5)

# ---------------------------------------------------------------- pytest machinery
EXPECTED = {
    'sec_blocks': [
        'D=2 L=4 SU(2): |logdet brute force - masked formula| over 4 sub-layers',
        'D=2 L=4 SU(2): off-diagonal blocks of J_AA',
        'D=2 L=4 SU(2): frozen rows of J differ from identity',
        'D=2 L=4 SU(2) NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero',
        'D=2 L=4 SU(2) NEGATIVE CONTROL: |logdet brute force - sum of per-link dets| is non-zero',
        'D=2 L=4 SU(2): full sweep (4 sub-layers) |logdet brute force - sum of masked formulas|',
        'D=2 L=4 SU(2): full sweep changes every link',
        'D=2 L=4 SU(3): |logdet brute force - masked formula| over 4 sub-layers',
        'D=2 L=4 SU(3): off-diagonal blocks of J_AA',
        'D=2 L=4 SU(3): frozen rows of J differ from identity',
        'D=2 L=4 SU(3) NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero',
        'D=2 L=4 SU(3) NEGATIVE CONTROL: |logdet brute force - sum of per-link dets| is non-zero',
        'D=2 L=4 SU(3): full sweep (4 sub-layers) |logdet brute force - sum of masked formulas|',
        'D=2 L=4 SU(3): full sweep changes every link',
        'D=3 L=4 SU(2): |logdet brute force - masked formula| over 2 sub-layers',
        'D=3 L=4 SU(2): off-diagonal blocks of J_AA',
        'D=3 L=4 SU(2): frozen rows of J differ from identity',
        'D=4 L=2 SU(2): |logdet brute force - masked formula| over 2 sub-layers',
        'D=4 L=2 SU(2): off-diagonal blocks of J_AA',
        'D=4 L=2 SU(2): frozen rows of J differ from identity',
        'D=3 L=4 SU(3): |logdet brute force - masked formula| over 2 sub-layers',
        'D=3 L=4 SU(3): off-diagonal blocks of J_AA',
        'D=3 L=4 SU(3): frozen rows of J differ from identity',
        'D=4 L=2 SU(3): |logdet brute force - masked formula| over 2 sub-layers',
        'D=4 L=2 SU(3): off-diagonal blocks of J_AA',
        'D=4 L=2 SU(3): frozen rows of J differ from identity',
        'D=2 L=3 SU(2) NEGATIVE CONTROL (odd L): |logdet brute force - masked formula| is non-zero',
    ],
    'sec_validate': [
        'validate_mask D=2 L=4 SU(3) M(0,0): change of other active connections (valid mask, must be 0)',
        'validate_mask D=3 L=4 SU(2) M(1,1): change of other active connections (valid mask, must be 0)',
        'validate_mask NEGATIVE CONTROL (every mu-link active): change detected',
        'validate_mask NEGATIVE CONTROL (odd L = 3): change detected',
    ],
    'sec_gauge': [
        'D=2 L=4 SU(3): gauge covariance of the layer, gauge invariance of log det',
        'D=2 L=4 SU(3): output stays in SU(N) (unitarity, det)',
        'D=3 L=4 SU(2): gauge covariance of the layer, gauge invariance of log det',
        'D=3 L=4 SU(2): output stays in SU(N) (unitarity, det)',
        'D=4 L=2 SU(3): gauge covariance of the layer, gauge invariance of log det',
        'D=4 L=2 SU(3): output stays in SU(N) (unitarity, det)',
    ],
    'sec_spectral': [
        'SU(2) single loop: |general d x d formula - spectral formula| (300 Haar samples)',
        'SU(3) single loop: |general d x d formula - spectral formula| (300 Haar samples)',
        "SU(2) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong",
        'SU(2) 2D lattice, single-plaquette class-function kernel: |masked formula - sum of spectral formulas|',
        "SU(3) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong",
        'SU(3) 2D lattice, single-plaquette class-function kernel: |masked formula - sum of spectral formulas|',
    ],
    'sec_inverse': [
        'SU(2) fixed-point inverse: |recovered - original active links|',
        'SU(2) fixed-point inverse: iterations, max ||D omega||_2',
        'SU(3) fixed-point inverse: |recovered - original active links|',
        'SU(3) fixed-point inverse: iterations, max ||D omega||_2',
    ],
    'sec_exact': [
        'SU(3): log c_0(beta=6) from exact_logZ_2d(V -> large)/V vs single-variable log Z 1.235890',
        'SU(2): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16',
        'SU(2) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact',
        'SU(2) 2x2 torus beta=1.5: |log Z Monte Carlo - exact character expansion| in stderr units',
        'SU(2) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison)',
        'SU(3): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16',
        'SU(3) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact',
        'SU(3) 2x2 torus beta=1.5: |log Z Monte Carlo - exact character expansion| in stderr units',
        'SU(3) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison)',
        'SU(2) 1x1 torus beta=3.0: log Z Monte Carlo, stderr, exact',
        'SU(2) 1x1 torus beta=3.0: |log Z Monte Carlo - exact| in stderr units',
        'SU(2) 1x1 torus beta=3.0: trivial-representation-only value excluded, in stderr units',
        'SU(3) 1x1 torus beta=4.0: log Z Monte Carlo, stderr, exact',
        'SU(3) 1x1 torus beta=4.0: |log Z Monte Carlo - exact| in stderr units',
        'SU(3) 1x1 torus beta=4.0: trivial-representation-only value excluded, in stderr units',
    ],
    'sec_scaling': [
        '2D SU(3), one sub-layer, batch 1: cost versus volume (8 reverse passes for D omega at every volume)',
        'cost per active link: ratio L=32 / L=8 (linear scaling means <= about 1)',
    ],
}

SECTIONS = {"sec_blocks": sec_blocks, "sec_validate": sec_validate, "sec_gauge": sec_gauge, "sec_spectral": sec_spectral,
            "sec_inverse": sec_inverse, "sec_exact": sec_exact, "sec_scaling": sec_scaling}

# Frozen tolerances of the 53 results recorded with check(): (tol, lower) as in the original
# verify_lattice.py; lower=True marks a negative control (value must be >= tol), otherwise value <= tol.
# They pin TOL, so a loosened tolerance in a section function cannot silently weaken the suite.
EXPECTED_TOL = {
    'D=2 L=4 SU(2): |logdet brute force - masked formula| over 4 sub-layers': (1e-09, False),
    'D=2 L=4 SU(2): off-diagonal blocks of J_AA': (1e-12, False),
    'D=2 L=4 SU(2): frozen rows of J differ from identity': (1e-12, False),
    'D=2 L=4 SU(2) NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero': (0.001, True),
    'D=2 L=4 SU(2) NEGATIVE CONTROL: |logdet brute force - sum of per-link dets| is non-zero': (0.001, True),
    'D=2 L=4 SU(2): full sweep (4 sub-layers) |logdet brute force - sum of masked formulas|': (1e-08, False),
    'D=2 L=4 SU(3): |logdet brute force - masked formula| over 4 sub-layers': (1e-09, False),
    'D=2 L=4 SU(3): off-diagonal blocks of J_AA': (1e-12, False),
    'D=2 L=4 SU(3): frozen rows of J differ from identity': (1e-12, False),
    'D=2 L=4 SU(3) NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero': (0.001, True),
    'D=2 L=4 SU(3) NEGATIVE CONTROL: |logdet brute force - sum of per-link dets| is non-zero': (0.001, True),
    'D=2 L=4 SU(3): full sweep (4 sub-layers) |logdet brute force - sum of masked formulas|': (1e-08, False),
    'D=3 L=4 SU(2): |logdet brute force - masked formula| over 2 sub-layers': (1e-09, False),
    'D=3 L=4 SU(2): off-diagonal blocks of J_AA': (1e-12, False),
    'D=3 L=4 SU(2): frozen rows of J differ from identity': (1e-12, False),
    'D=4 L=2 SU(2): |logdet brute force - masked formula| over 2 sub-layers': (1e-09, False),
    'D=4 L=2 SU(2): off-diagonal blocks of J_AA': (1e-12, False),
    'D=4 L=2 SU(2): frozen rows of J differ from identity': (1e-12, False),
    'D=3 L=4 SU(3): |logdet brute force - masked formula| over 2 sub-layers': (1e-09, False),
    'D=3 L=4 SU(3): off-diagonal blocks of J_AA': (1e-12, False),
    'D=3 L=4 SU(3): frozen rows of J differ from identity': (1e-12, False),
    'D=4 L=2 SU(3): |logdet brute force - masked formula| over 2 sub-layers': (1e-09, False),
    'D=4 L=2 SU(3): off-diagonal blocks of J_AA': (1e-12, False),
    'D=4 L=2 SU(3): frozen rows of J differ from identity': (1e-12, False),
    'D=2 L=3 SU(2) NEGATIVE CONTROL (odd L): |logdet brute force - masked formula| is non-zero': (0.001, True),
    'validate_mask D=2 L=4 SU(3) M(0,0): change of other active connections (valid mask, must be 0)': (0.0, False),
    'validate_mask D=3 L=4 SU(2) M(1,1): change of other active connections (valid mask, must be 0)': (0.0, False),
    'validate_mask NEGATIVE CONTROL (every mu-link active): change detected': (1e-06, True),
    'validate_mask NEGATIVE CONTROL (odd L = 3): change detected': (1e-06, True),
    'D=2 L=4 SU(3): gauge covariance of the layer, gauge invariance of log det': (1e-11, False),
    'D=2 L=4 SU(3): output stays in SU(N) (unitarity, det)': (1e-12, False),
    'D=3 L=4 SU(2): gauge covariance of the layer, gauge invariance of log det': (1e-11, False),
    'D=3 L=4 SU(2): output stays in SU(N) (unitarity, det)': (1e-12, False),
    'D=4 L=2 SU(3): gauge covariance of the layer, gauge invariance of log det': (1e-11, False),
    'D=4 L=2 SU(3): output stays in SU(N) (unitarity, det)': (1e-12, False),
    'SU(2) single loop: |general d x d formula - spectral formula| (300 Haar samples)': (1e-08, False),
    'SU(3) single loop: |general d x d formula - spectral formula| (300 Haar samples)': (1e-08, False),
    "SU(2) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong": (0.001, True),
    'SU(2) 2D lattice, single-plaquette class-function kernel: |masked formula - sum of spectral formulas|': (1e-08, False),
    "SU(3) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong": (0.001, True),
    'SU(3) 2D lattice, single-plaquette class-function kernel: |masked formula - sum of spectral formulas|': (1e-08, False),
    'SU(2) fixed-point inverse: |recovered - original active links|': (1e-12, False),
    'SU(3) fixed-point inverse: |recovered - original active links|': (1e-12, False),
    'SU(3): log c_0(beta=6) from exact_logZ_2d(V -> large)/V vs single-variable log Z 1.235890': (1e-08, False),
    'SU(2): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16': (1e-10, False),
    'SU(2) 2x2 torus beta=1.5: |log Z Monte Carlo - exact character expansion| in stderr units': (4.0, False),
    'SU(3): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16': (1e-10, False),
    'SU(3) 2x2 torus beta=1.5: |log Z Monte Carlo - exact character expansion| in stderr units': (4.0, False),
    'SU(2) 1x1 torus beta=3.0: |log Z Monte Carlo - exact| in stderr units': (4.0, False),
    'SU(2) 1x1 torus beta=3.0: trivial-representation-only value excluded, in stderr units': (20.0, True),
    'SU(3) 1x1 torus beta=4.0: |log Z Monte Carlo - exact| in stderr units': (4.0, False),
    'SU(3) 1x1 torus beta=4.0: trivial-representation-only value excluded, in stderr units': (20.0, True),
    'cost per active link: ratio L=32 / L=8 (linear scaling means <= about 1)': (1.5, False),
}

def _mc_triple(v):
    """[log Z Monte Carlo, stderr, exact]: finite, with a positive standard error."""
    return len(v) == 3 and all(math.isfinite(float(x)) for x in v) and v[1] > 0

def _below_exact(triple_name):
    """Z = sum_r a_r^V with V = 4: every representation adds a_r^4 >= 0, so the trivial-only log Z is below the exact."""
    return lambda v, res: v < res[triple_name][2]

def _volume_table(rows):
    """Section 4 volume table: L = 4, 8, 16, 32 in 2D, half of the 2 L^2 links active, positive times."""
    return ([r["L"] for r in rows] == [4, 8, 16, 32]
            and all(r["links"] == 2 * r["L"] ** 2 and r["active links"] == r["L"] ** 2 // 2 for r in rows)
            and all(r[k] > 0 for r in rows for k in ("ms layer + log det", "ms backward", "microseconds per active link")))

# Results recorded without a tolerance (MATH_SPEC_LATTICE.md section 9): assert the qualitative claim of the spec
# instead. Each lambda receives the value and the results of its section.
QUALITATIVE = {
    # section 3.2: a sweep of the 2D sub-layers (mu, p) updates every link
    "D=2 L=4 SU(2): full sweep changes every link": lambda v, res: v is True,
    "D=2 L=4 SU(3): full sweep changes every link": lambda v, res: v is True,
    # section 6, Proposition 3: q = max ||D omega||_2 < 1, and the iteration stopped before its cap of 200
    "SU(2) fixed-point inverse: iterations, max ||D omega||_2": lambda v, res: 0 < v[0] < 200 and 0 < v[1] < 1,
    "SU(3) fixed-point inverse: iterations, max ||D omega||_2": lambda v, res: 0 < v[0] < 200 and 0 < v[1] < 1,
    # section 7: Monte Carlo log Z, its standard error and the exact value (compared in the z-score rows)
    "SU(2) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact": lambda v, res: _mc_triple(v),
    "SU(3) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact": lambda v, res: _mc_triple(v),
    "SU(2) 1x1 torus beta=3.0: log Z Monte Carlo, stderr, exact": lambda v, res: _mc_triple(v),
    "SU(3) 1x1 torus beta=4.0: log Z Monte Carlo, stderr, exact": lambda v, res: _mc_triple(v),
    # section 7: the 2x2 torus keeps every representation's contribution positive
    "SU(2) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison)":
        _below_exact("SU(2) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact"),
    "SU(3) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison)":
        _below_exact("SU(3) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact"),
    # section 4: the volume table
    "2D SU(3), one sub-layer, batch 1: cost versus volume (8 reverse passes for D omega at every volume)":
        lambda v, res: _volume_table(v),
}

_CACHE = {}
def _section(name):
    """Run section function `name` at most once per session; return its results or re-raise its exception.

    Runs in float64 (the original sets the default dtype at import) and restores the default dtype and the number of
    threads afterwards (sec_scaling raises it to the CPU count, as the original does).
    """
    if name not in _CACHE:
        prev, threads, before = torch.get_default_dtype(), torch.get_num_threads(), set(R)
        torch.set_default_dtype(torch.float64)
        try:
            SECTIONS[name]()
            produced = [k for k in R if k not in before]
            missing = [k for k in EXPECTED[name] if k not in produced]
            extra = [k for k in produced if k not in EXPECTED[name]]
            if missing or extra:
                _CACHE[name] = AssertionError(f"{name}: recorded names differ from EXPECTED; missing={missing}, unexpected={extra}")
            else:
                _CACHE[name] = {k: R[k] for k in produced}
        except Exception as e:
            _CACHE[name] = e
        finally:
            torch.set_default_dtype(prev)
            torch.set_num_threads(threads)
    res = _CACHE[name]
    if isinstance(res, Exception): raise res
    return res

def _finite(value):
    if isinstance(value, dict): return _finite(list(value.values()))
    vals = value if isinstance(value, (list, tuple)) else [value]
    return all(_finite(x) if isinstance(x, (dict, list, tuple)) else math.isfinite(float(x)) for x in vals)

def test_expected_covers_all_results():
    """64 results in all: 53 with a pinned tolerance and 11 asserted qualitatively, each exactly once."""
    names = [n for ns in EXPECTED.values() for n in ns]
    assert len(names) == len(set(names)) == 64
    assert set(EXPECTED_TOL) | set(QUALITATIVE) == set(names) and not set(EXPECTED_TOL) & set(QUALITATIVE)
    assert len(EXPECTED_TOL) == 53 and len(QUALITATIVE) == 11

@pytest.mark.parametrize("section,name", [pytest.param(s, n, id=n) for s, names in EXPECTED.items() for n in names])
def test_lattice(section, name):
    res = _section(section)
    value = res[name]
    assert _finite(value), f"{name}: non-finite value {value}"
    if name in EXPECTED_TOL:
        assert name in TOL and TOL[name] == EXPECTED_TOL[name], \
            f"{name}: tolerance changed or no longer checked ({TOL.get(name)} != {EXPECTED_TOL[name]})"
        tol, lower = TOL[name]
        v = nanmax(value) if isinstance(value, (list, tuple)) else value
        if lower:
            assert v >= tol, f"{name}: negative control {value} below its lower bound {tol}"
        else:
            assert v <= tol, f"{name}: value {value} exceeds tolerance {tol}"
    else:
        assert name not in TOL, f"{name}: recorded with a tolerance, but EXPECTED_TOL has none"
        assert QUALITATIVE[name](value, res), f"{name}: qualitative claim of MATH_SPEC_LATTICE.md violated, value {value}"

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    t0 = time.time()
    for sec in (sec_blocks, sec_validate, sec_gauge, sec_spectral, sec_inverse, sec_exact, sec_scaling):
        ts = time.time(); sec(); print(f"{sec.__name__}: {time.time() - ts:.1f} s", flush=True)
    n_results = len(R)                                  # the environment record added next is metadata, not a result
    R["environment"] = {"torch": torch.__version__, "seconds": round(time.time() - t0, 1), "failures": len(FAIL), "failed": [f[0] for f in FAIL]}
    json.dump(R, open("verify_lattice_results.json", "w"), indent=1)
    print(f"{n_results} results recorded, {len(FAIL)} failures"); [print("FAIL", f) for f in FAIL]
    sys.exit(1 if FAIL else 0)
