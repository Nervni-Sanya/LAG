"""Acceptance suite for LAG: numerical verification of every formula in MATH_SPEC.md.

Ported from verify_math_spec.py. Each of the 47 recorded results is one pytest item
(named by the check) and maps to a row of MATH_SPEC.md section 8.

Run:
    python -m pytest tests/test_math_spec.py -v      (or: pytest tests/ -v)
    python tests/test_math_spec.py                   (CLI: writes verify_results.json to the current directory)
"""
import pathlib, sys
if __name__ == "__main__":  # run as a script: make the repository root importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, math, torch
import pytest
from torch.autograd.functional import jacobian
from lag.algebra import C, T2, T3, so3_hat, so3_vee, su_hat, su_vee, proj_su, se3_hat, se3_vee_first_order, se3_ad
from lag.exp import so3_exp, su2_exp, se3_exp, su3_exp, dexpL_so3, dexpL_se3, dexpL_generic, exp_vjp_normal
from lag.jacobian import logabsdet_step
from lag.heads import eps_dual_su2, cross_dual_su3, head_fundamental, head_adjoint, head_loops
mexp = torch.linalg.matrix_exp
cross = torch.linalg.cross
R, FAIL, TOL = {}, [], {}
def check(name, value, tol):
    R[name] = value; TOL[name] = tol; v = max(value) if isinstance(value, (list, tuple)) else value
    if not (v <= tol): FAIL.append((name, v, tol))
def field_factory(nin, nout, scale=0.6):
    W = torch.randn(nout, nin) / nin ** 0.5; b = torch.randn(nout)
    return lambda x: scale * torch.tanh(W @ x + b)
def rand_su(N): return mexp(su_hat(torch.randn(N * N - 1), T2 if N == 2 else T3))
def net(nin, nout):
    W = torch.randn(nout, nin) / nin ** 0.5; b = torch.randn(nout)
    return lambda x: torch.tanh(x @ W.T + b)

# ---------------------------------------------------------------- S2: exponentials and dexp
def sec_exp():
    torch.manual_seed(0); w = torch.randn(50, 3); xi = torch.randn(50, 6); X = su_hat(torch.randn(2000, 8), T3)
    check("exp so3 closed vs matrix_exp", float((so3_exp(w) - mexp(so3_hat(w))).abs().max()), 1e-13)
    check("exp su2 closed vs matrix_exp", float((su2_exp(w) - mexp(su_hat(w, T2))).abs().max()), 1e-13)
    check("exp se3 closed vs matrix_exp", float((se3_exp(xi) - mexp(se3_hat(xi))).abs().max()), 1e-13)
    check("dexpL so3 closed vs phi1 block-exp", float((dexpL_so3(w) - dexpL_generic(so3_hat(w))).abs().max()), 1e-13)
    check("dexpL so3 small angle", float((dexpL_so3(1e-3 * w) - dexpL_generic(so3_hat(1e-3 * w))).abs().max()), 1e-13)
    check("dexpL se3 closed (Q = D_rho J) vs phi1 block-exp", float((dexpL_se3(xi) - dexpL_generic(se3_ad(xi))).abs().max()), 1e-12)
    check("dexpL se3 small angle", float((dexpL_se3(1e-2 * xi) - dexpL_generic(se3_ad(1e-2 * xi))).abs().max()), 1e-13)
    check("exp su3 closed: generic", float((su3_exp(X) - mexp(X)).abs().max()), 1e-13)
    check("exp su3 closed: norm x10", float((su3_exp(10 * X) - mexp(10 * X)).abs().max()), 1e-12)
    check("exp su3 closed: norm x1e-7", float((su3_exp(1e-7 * X) - mexp(1e-7 * X)).abs().max()), 1e-14)
    check("exp su3 closed: X = 0", float((su3_exp(0 * X[:2]) - torch.eye(3)).abs().max()), 0.0)
    g = rand_su(3); w8 = torch.zeros(3, 8); w8[0, 7] = 1.7; w8[1, 7] = -2.3; w8[2, 7] = 1.1; w8[2, 2] = 1e-9
    Xd = g @ su_hat(w8, T3) @ g.mH
    check("exp su3 closed: degenerate eigenvalues", float((su3_exp(Xd) - mexp(Xd)).abs().max()), 1e-13)
    E = su3_exp(X); check("exp su3 closed: unitarity, det", [float((E.mH @ E - torch.eye(3)).abs().max()), float((torch.linalg.det(E) - 1).abs().max())], 1e-13)
    Cw = torch.randn(3, 3, dtype=C)
    def grad_of(f, wv):
        wv = wv.clone().requires_grad_(True); (f(su_hat(wv, T3)) * Cw).sum().real.backward(); return wv.grad
    worst = 0.0; bad = 0
    for gap in (1.0, 1e-4, 1e-8, 0.0):
        for _ in range(25):
            wv = torch.zeros(8); wv[7] = 1.7; wv[2] = gap; gg = rand_su(3); wr = su_vee(gg @ su_hat(wv, T3) @ gg.mH, T3)
            ga, gm = grad_of(su3_exp, wr), grad_of(mexp, wr); bad += int(not torch.isfinite(ga).all())
            worst = max(worst, float((ga - gm).abs().max() / gm.abs().max()))
    check("grad through su3 closed form vs matrix_exp (incl. degenerate): rel err", worst, 1e-8); check("grad through su3 closed form: non-finite count", bad, 0)
    check("grad through su3 closed form at X = 0", float((grad_of(su3_exp, torch.zeros(8)) - grad_of(mexp, torch.zeros(8))).abs().max()), 1e-14)
    Xr = Xd[0].clone().requires_grad_(True); Eb = torch.randn(3, 3, dtype=C); (mexp(Xr) * Eb.conj()).sum().real.backward()
    check("Daleckii-Krein exp VJP at degenerate X", float((exp_vjp_normal(Xd[0], Eb) - Xr.grad).abs().max()), 1e-13)

# ---------------------------------------------------------------- S3: Jacobian log-determinants
def sec_jacobian():
    torch.manual_seed(1)
    def rkmk4_v(f, exp, bracket, h):
        def v(U):
            k1 = h * f(U); k2 = h * f(exp(k1 / 2) @ U); k3 = h * f(exp(k2 / 2 - bracket(k1, k2) / 8) @ U); k4 = h * f(exp(k3) @ U)
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6 - bracket(k1, k4) / 12
        return v
    def group_case(name, d, U, hat, exp, vee1, feat, dexpL, bracket):
        f0 = field_factory(feat(U).numel(), d); f = lambda V: f0(feat(V)); n = U.shape[-1]; I = torch.eye(n, dtype=U.dtype)
        Ad = lambda g: torch.stack([vee1(g @ hat(e) @ torch.linalg.inv(g) + I) - vee1(I) for e in torch.eye(d)], 1)
        for step, vfun in (("Lie-Euler", f), ("RKMK4", rkmk4_v(f, exp, bracket, 1.0))):
            Phi = lambda V: exp(vfun(V)) @ V; P0inv = torch.linalg.inv(Phi(U))
            Jbf = jacobian(lambda x: vee1(Phi(exp(x) @ U) @ P0inv), torch.zeros(d))
            Dv = jacobian(lambda x: vfun(exp(x) @ U), torch.zeros(d)); v = vfun(U)
            Jf = Ad(exp(v)) @ (torch.eye(d) + dexpL(v) @ Dv)
            check(f"Jacobian {name} {step}: |J_formula - J_bruteforce|", float((Jf - Jbf).abs().max()), 1e-12)
            check(f"Jacobian {name} {step}: |logdet formula - brute force|", float(abs(logabsdet_step(dexpL(v), Dv)[1] - torch.linalg.slogdet(Jbf)[1])), 1e-12)
    ri = lambda M: torch.cat([M.real.reshape(-1), M.imag.reshape(-1)])
    se3_br = lambda a, b: torch.cat([cross(a[:3], b[:3]), cross(a[:3], b[3:]) + cross(a[3:], b[:3])])
    group_case("SO(3)", 3, so3_exp(torch.randn(3)), so3_hat, so3_exp, so3_vee, lambda V: V.reshape(-1), dexpL_so3, cross)
    group_case("SU(2)", 3, su2_exp(torch.randn(3)), lambda w: su_hat(w, T2), su2_exp, lambda M: su_vee(M, T2), ri, dexpL_so3, cross)
    group_case("SE(3)", 6, se3_exp(torch.randn(6)), se3_hat, se3_exp, se3_vee_first_order, lambda V: V[:3].reshape(-1), dexpL_se3, se3_br)
    z = torch.randn(3); f = field_factory(3, 3); A = torch.stack([so3_hat(e) @ z for e in torch.eye(3)], 1)
    check("Jacobian vector state SO(3) on R^3", float(abs(logabsdet_step(dexpL_so3(f(z)), jacobian(f, z), A)[1] - torch.linalg.slogdet(jacobian(lambda x: so3_exp(f(x)) @ x, z))[1])), 1e-12)
    z = torch.randn(4); f = field_factory(4, 3); c2 = lambda x: torch.complex(x[:2], x[2:]); r4 = lambda u: torch.cat([u.real, u.imag])
    A = torch.stack([r4(T2[a] @ c2(z)) for a in range(3)], 1)
    check("Jacobian vector state SU(2) on C^2 (3x3 det vs 4x4 autograd)", float(abs(logabsdet_step(dexpL_so3(f(z)), jacobian(f, z), A)[1] - torch.linalg.slogdet(jacobian(lambda x: r4(su2_exp(f(x)) @ c2(x)), z))[1])), 1e-12)
    q = torch.randn(200000, 4); q = q / q.norm(dim=1, keepdim=True); a, b, c, d = q.T
    U = torch.stack([torch.stack([a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], -1), torch.stack([2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], -1), torch.stack([2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d], -1)], -2)
    Wg = torch.randn(3, 9) / 3; bg = torch.randn(3)
    for scale in (0.25, 2.5):
        x = torch.zeros(len(U), 3, requires_grad=True); om = scale * torch.tanh(((torch.eye(3) + so3_hat(x)) @ U).reshape(-1, 9) @ Wg.T + bg)
        Dw = torch.stack([torch.autograd.grad(om[:, k].sum(), x, retain_graph=True)[0] for k in range(3)], 1)
        det = torch.linalg.det(torch.eye(3) + dexpL_so3(om.detach()) @ Dw)
        R[f"Haar SO(3), field scale {scale}: mean det J, stderr, mean |det J|, min det J, max ||D omega||"] = [float(det.mean()), float(det.std() / len(det) ** 0.5), float(det.abs().mean()), float(det.min()), float(torch.linalg.matrix_norm(Dw, 2).max())]
        check(f"Haar SO(3), field scale {scale}: |mean det J - 1| in stderr units", float(abs(det.mean() - 1) / (det.std() / len(det) ** 0.5)), 4.0)

# ---------------------------------------------------------------- S4: equivariant heads
def sec_heads():
    torch.manual_seed(5)
    for N in (2, 3):
        K = 4; z = torch.randn(K, N, dtype=C); g = rand_su(N); Tn = T2 if N == 2 else T3
        def aug(z):
            if N == 2: return torch.cat([z, eps_dual_su2(z)], 0)
            return torch.cat([z, torch.stack([cross_dual_su3(z[a], z[b]) for a in range(K) for b in range(a + 1, K)])], 0)
        M = aug(z).shape[0]; mlp = net(2 * M * M, 2 * K * M * M)
        coeff = lambda G: (lambda o: torch.complex(o[:K * M * M], o[K * M * M:]).reshape(K, M, M))(mlp(torch.cat([G.real.reshape(-1), G.imag.reshape(-1)])))
        om = lambda z: head_fundamental(aug(z), coeff); o1, o2 = om(z @ g.T), g @ om(z) @ g.mH
        check(f"SU({N}) fundamental head: rel equivariance error", float((o1 - o2).abs().max() / o2.abs().max()), 1e-12)
        check(f"SU({N}) fundamental head: output in su(N) (trace, anti-Hermitian)", [float(o2.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()), float((o2 + o2.mH).abs().max())], 1e-12)
        expf = (lambda X: su2_exp(su_vee(X, T2))) if N == 2 else su3_exp
        layer = lambda z: torch.einsum("aij,aj->ai", expf(om(z)), z)
        check(f"SU({N}) layer z -> exp(omega) z: equivariance, norm drift", [float((layer(z @ g.T) - layer(z) @ g.T).abs().max()), float((layer(z).norm(dim=1) - z.norm(dim=1)).abs().max())], 1e-12)
        Xs = torch.stack([su_hat(torch.randn(N * N - 1), Tn) for _ in range(K)]); m1, m2, m3 = net(K * K, K * K), net(K * K, K ** 3), net(K * K, K ** 3)
        cf = lambda I2: (m1(I2.reshape(-1)).reshape(K, K), m2(I2.reshape(-1)).reshape(K, K, K), m3(I2.reshape(-1)).reshape(K, K, K))
        oa, ob = head_adjoint(Xs, cf), head_adjoint(g @ Xs @ g.mH, cf)
        check(f"SU({N}) adjoint head: rel equivariance error", float((ob - g @ oa @ g.mH).abs().max() / oa.abs().max()), 1e-12)
        anti = torch.einsum("aij,bjk->abik", Xs, Xs); R[f"SU({N}) d-type covariant i{{X,Y}} traceless: max abs (0 for SU(2))"] = float(proj_su(1j * (anti + anti.transpose(0, 1))).abs().max())
    Lx = 4; U = torch.stack([torch.stack([torch.stack([rand_su(3) for _ in range(Lx)]) for _ in range(Lx)]) for _ in range(2)])
    sh = lambda A, mu, k=1: torch.roll(A, -k, dims=mu)
    def loops(U):
        out = []
        for mu in (0, 1):
            nu = 1 - mu
            out.append(U[mu] @ sh(U[nu], mu) @ sh(U[mu], nu).mH @ U[nu].mH + U[mu] @ sh(sh(U[nu], mu), nu, -1).mH @ sh(U[mu], nu, -1).mH @ sh(U[nu], nu, -1))
        return torch.stack(out)
    cl = net(2, 4); coeffL = lambda t: (lambda o: (torch.complex(o[..., 0], o[..., 1]), torch.complex(o[..., 2], o[..., 3])))(0.3 * cl(torch.stack([t.real, t.imag], -1)))
    layerL = lambda U: su3_exp(head_loops(loops(U), coeffL)) @ U
    gx = torch.stack([torch.stack([rand_su(3) for _ in range(Lx)]) for _ in range(Lx)]); gauge = lambda U: torch.stack([gx @ U[mu] @ sh(gx, mu).mH for mu in (0, 1)])
    check("lattice SU(3) 4x4 stout-type layer: gauge covariance", float((layerL(gauge(U)) - gauge(layerL(U))).abs().max()), 1e-12)

# ---------------------------------------------------------------- S5: RKMK4 reverse sweep
def sec_adjoint():
    torch.manual_seed(6); K = 3; Wf = torch.randn(3 * K, 3 * K) / 3; bf = torch.randn(3 * K)
    field = lambda Z: torch.tanh(Z.reshape(-1) @ Wf.T + bf).reshape(K, 3); act = lambda v, Z: torch.einsum("aij,aj->ai", so3_exp(v), Z)
    def rkmk4(Z, h):
        k1 = h * field(Z); k2 = h * field(act(k1 / 2, Z)); k3 = h * field(act(k2 / 2 - cross(k1, k2) / 8, Z)); k4 = h * field(act(k3, Z))
        return act((k1 + 2 * k2 + 2 * k3 + k4) / 6 - cross(k1, k4) / 12, Z)
    def act_vjp(v, Z, ybar):
        Et = torch.einsum("aji,aj->ai", so3_exp(v), ybar); return torch.einsum("aji,aj->ai", dexpL_so3(v), cross(Z, Et)), Et
    def field_vjp(Y, kbar):
        Y = Y.detach().requires_grad_(True); return torch.autograd.grad((field(Y) * kbar).sum(), Y)[0]
    def rkmk4_vjp(Z, h, zbar1):
        k1 = h * field(Z); y2 = act(k1 / 2, Z); k2 = h * field(y2); u3 = k2 / 2 - cross(k1, k2) / 8; y3 = act(u3, Z); k3 = h * field(y3); y4 = act(k3, Z); k4 = h * field(y4)
        v = (k1 + 2 * k2 + 2 * k3 + k4) / 6 - cross(k1, k4) / 12
        vb, Zb = act_vjp(v, Z, zbar1)
        k1b = vb / 6 - cross(k4, vb) / 12; k4b = vb / 6 + cross(k1, vb) / 12; k2b = vb / 3; k3b = vb / 3
        y4b = h * field_vjp(y4, k4b); t, zb = act_vjp(k3, Z, y4b); k3b = k3b + t; Zb = Zb + zb
        y3b = h * field_vjp(y3, k3b); u3b, zb = act_vjp(u3, Z, y3b); Zb = Zb + zb
        k2b = k2b + u3b / 2 - cross(u3b, k1) / 8; k1b = k1b - cross(k2, u3b) / 8
        y2b = h * field_vjp(y2, k2b); t, zb = act_vjp(k1 / 2, Z, y2b); k1b = k1b + t / 2; Zb = Zb + zb
        return Zb + h * field_vjp(Z, k1b)
    Z0 = torch.randn(K, 3, requires_grad=True); zbar = torch.randn(K, 3); (rkmk4(Z0, 0.7) * zbar).sum().backward()
    check("RKMK4 hand-written reverse sweep vs autograd", float((rkmk4_vjp(Z0.detach(), 0.7, zbar) - Z0.grad).abs().max()), 1e-13)

# ---------------------------------------------------------------- pytest machinery
EXPECTED = {"sec_exp":["exp so3 closed vs matrix_exp","exp su2 closed vs matrix_exp","exp se3 closed vs matrix_exp","dexpL so3 closed vs phi1 block-exp","dexpL so3 small angle","dexpL se3 closed (Q = D_rho J) vs phi1 block-exp","dexpL se3 small angle","exp su3 closed: generic","exp su3 closed: norm x10","exp su3 closed: norm x1e-7","exp su3 closed: X = 0","exp su3 closed: degenerate eigenvalues","exp su3 closed: unitarity, det","grad through su3 closed form vs matrix_exp (incl. degenerate): rel err","grad through su3 closed form: non-finite count","grad through su3 closed form at X = 0","Daleckii-Krein exp VJP at degenerate X"],
"sec_jacobian":["Jacobian SO(3) Lie-Euler: |J_formula - J_bruteforce|","Jacobian SO(3) Lie-Euler: |logdet formula - brute force|","Jacobian SO(3) RKMK4: |J_formula - J_bruteforce|","Jacobian SO(3) RKMK4: |logdet formula - brute force|","Jacobian SU(2) Lie-Euler: |J_formula - J_bruteforce|","Jacobian SU(2) Lie-Euler: |logdet formula - brute force|","Jacobian SU(2) RKMK4: |J_formula - J_bruteforce|","Jacobian SU(2) RKMK4: |logdet formula - brute force|","Jacobian SE(3) Lie-Euler: |J_formula - J_bruteforce|","Jacobian SE(3) Lie-Euler: |logdet formula - brute force|","Jacobian SE(3) RKMK4: |J_formula - J_bruteforce|","Jacobian SE(3) RKMK4: |logdet formula - brute force|","Jacobian vector state SO(3) on R^3","Jacobian vector state SU(2) on C^2 (3x3 det vs 4x4 autograd)","Haar SO(3), field scale 0.25: mean det J, stderr, mean |det J|, min det J, max ||D omega||","Haar SO(3), field scale 0.25: |mean det J - 1| in stderr units","Haar SO(3), field scale 2.5: mean det J, stderr, mean |det J|, min det J, max ||D omega||","Haar SO(3), field scale 2.5: |mean det J - 1| in stderr units"],
"sec_heads":["SU(2) fundamental head: rel equivariance error","SU(2) fundamental head: output in su(N) (trace, anti-Hermitian)","SU(2) layer z -> exp(omega) z: equivariance, norm drift","SU(2) adjoint head: rel equivariance error","SU(2) d-type covariant i{X,Y} traceless: max abs (0 for SU(2))","SU(3) fundamental head: rel equivariance error","SU(3) fundamental head: output in su(N) (trace, anti-Hermitian)","SU(3) layer z -> exp(omega) z: equivariance, norm drift","SU(3) adjoint head: rel equivariance error","SU(3) d-type covariant i{X,Y} traceless: max abs (0 for SU(2))","lattice SU(3) 4x4 stout-type layer: gauge covariance"],
"sec_adjoint":["RKMK4 hand-written reverse sweep vs autograd"]}

SECTIONS = {"sec_exp": sec_exp, "sec_jacobian": sec_jacobian, "sec_heads": sec_heads, "sec_adjoint": sec_adjoint}

# Results recorded without a tolerance: assert the qualitative claim of MATH_SPEC.md instead.
QUALITATIVE = {
    # MATH_SPEC.md 3.5: sufficient condition ||D omega|| < 1 holds, so no folding (det J > 0)
    "Haar SO(3), field scale 0.25: mean det J, stderr, mean |det J|, min det J, max ||D omega||": lambda v: v[4] < 1 and v[3] > 0,
    # MATH_SPEC.md 3.5 table: the large field folds, which shows the Haar test has power
    "Haar SO(3), field scale 2.5: mean det J, stderr, mean |det J|, min det J, max ||D omega||": lambda v: v[4] > 1 and v[3] < 0,
    # MATH_SPEC.md 4.2: the d-type covariant vanishes identically for SU(2)
    "SU(2) d-type covariant i{X,Y} traceless: max abs (0 for SU(2))": lambda v: v <= 1e-12,
    # MATH_SPEC.md 4.2: it does not vanish for SU(3) (measured 1.56)
    "SU(3) d-type covariant i{X,Y} traceless: max abs (0 for SU(2))": lambda v: v > 0.1,
}

# Frozen tolerances of the 43 results recorded with check(): the literal values of the original verify_math_spec.py.
# They pin TOL, so a loosened tolerance in a section function cannot silently weaken the suite.
EXPECTED_TOL = {
    'exp so3 closed vs matrix_exp': 1e-13,
    'exp su2 closed vs matrix_exp': 1e-13,
    'exp se3 closed vs matrix_exp': 1e-13,
    'dexpL so3 closed vs phi1 block-exp': 1e-13,
    'dexpL so3 small angle': 1e-13,
    'dexpL se3 closed (Q = D_rho J) vs phi1 block-exp': 1e-12,
    'dexpL se3 small angle': 1e-13,
    'exp su3 closed: generic': 1e-13,
    'exp su3 closed: norm x10': 1e-12,
    'exp su3 closed: norm x1e-7': 1e-14,
    'exp su3 closed: X = 0': 0.0,
    'exp su3 closed: degenerate eigenvalues': 1e-13,
    'exp su3 closed: unitarity, det': 1e-13,
    'grad through su3 closed form vs matrix_exp (incl. degenerate): rel err': 1e-08,
    'grad through su3 closed form: non-finite count': 0,
    'grad through su3 closed form at X = 0': 1e-14,
    'Daleckii-Krein exp VJP at degenerate X': 1e-13,
    'Jacobian SO(3) Lie-Euler: |J_formula - J_bruteforce|': 1e-12,
    'Jacobian SO(3) Lie-Euler: |logdet formula - brute force|': 1e-12,
    'Jacobian SO(3) RKMK4: |J_formula - J_bruteforce|': 1e-12,
    'Jacobian SO(3) RKMK4: |logdet formula - brute force|': 1e-12,
    'Jacobian SU(2) Lie-Euler: |J_formula - J_bruteforce|': 1e-12,
    'Jacobian SU(2) Lie-Euler: |logdet formula - brute force|': 1e-12,
    'Jacobian SU(2) RKMK4: |J_formula - J_bruteforce|': 1e-12,
    'Jacobian SU(2) RKMK4: |logdet formula - brute force|': 1e-12,
    'Jacobian SE(3) Lie-Euler: |J_formula - J_bruteforce|': 1e-12,
    'Jacobian SE(3) Lie-Euler: |logdet formula - brute force|': 1e-12,
    'Jacobian SE(3) RKMK4: |J_formula - J_bruteforce|': 1e-12,
    'Jacobian SE(3) RKMK4: |logdet formula - brute force|': 1e-12,
    'Jacobian vector state SO(3) on R^3': 1e-12,
    'Jacobian vector state SU(2) on C^2 (3x3 det vs 4x4 autograd)': 1e-12,
    'Haar SO(3), field scale 0.25: |mean det J - 1| in stderr units': 4.0,
    'Haar SO(3), field scale 2.5: |mean det J - 1| in stderr units': 4.0,
    'SU(2) fundamental head: rel equivariance error': 1e-12,
    'SU(2) fundamental head: output in su(N) (trace, anti-Hermitian)': 1e-12,
    'SU(2) layer z -> exp(omega) z: equivariance, norm drift': 1e-12,
    'SU(2) adjoint head: rel equivariance error': 1e-12,
    'SU(3) fundamental head: rel equivariance error': 1e-12,
    'SU(3) fundamental head: output in su(N) (trace, anti-Hermitian)': 1e-12,
    'SU(3) layer z -> exp(omega) z: equivariance, norm drift': 1e-12,
    'SU(3) adjoint head: rel equivariance error': 1e-12,
    'lattice SU(3) 4x4 stout-type layer: gauge covariance': 1e-12,
    'RKMK4 hand-written reverse sweep vs autograd': 1e-13,
}

_CACHE = {}
def _section(name):
    """Run section function `name` at most once per session; return its results or re-raise its exception."""
    if name not in _CACHE:
        prev = torch.get_default_dtype(); before = set(R)
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
    res = _CACHE[name]
    if isinstance(res, Exception): raise res
    return res

def _finite(value):
    vals = value if isinstance(value, (list, tuple)) else [value]
    return all(math.isfinite(float(x)) for x in vals)

@pytest.mark.parametrize("section,name", [pytest.param(s, n, id=n) for s, names in EXPECTED.items() for n in names])
def test_math_spec(section, name):
    value = _section(section)[name]
    assert _finite(value), f"{name}: non-finite value {value}"
    if name in TOL:
        assert name in EXPECTED_TOL and TOL[name] == EXPECTED_TOL[name], \
            f"{name}: tolerance changed ({TOL[name]} != {EXPECTED_TOL.get(name)})"
        v = max(value) if isinstance(value, (list, tuple)) else value
        assert v <= TOL[name], f"{name}: value {value} exceeds tolerance {TOL[name]}"
    else:
        assert QUALITATIVE[name](value), f"{name}: qualitative claim of MATH_SPEC.md violated, value {value}"

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    sec_exp(); sec_jacobian(); sec_heads(); sec_adjoint()
    json.dump(R, open("verify_results.json", "w"), indent=1)
    print(f"{len(R)} results recorded, {len(FAIL)} failures"); [print("FAIL", f) for f in FAIL]
