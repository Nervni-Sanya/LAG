# LAG — MATH_SPEC

Mathematical specification for the first build of `lag`. Every formula below has a reference implementation in
`lagmath.py` and a numerical check in `verify_math_spec.py` (47 recorded results, 0 failures; see §8).
Nothing in this file is stated without a check unless it is marked **[not verified]**.

## 0. Architect's notes

**Scope decisions**

1. *Formulas are verified before they are specified.* A coding agent implements a spec literally; the checks in
   `verify_math_spec.py` are the acceptance tests for the corresponding library functions.
2. *One end-to-end test was run before any repository work* (§7): a normalizing flow on a single SU(3) variable built
   only from §2–§4. Effective sample size went from 11.2 % (prior) to 99.96 %.
3. *Adjusted premises* relative to the request:
   - There is no separate "RKMK4 log-determinant formula". Any step of the form $x \mapsto \exp(v(x))\cdot x$ has the
     same $d\times d$ determinant (§3); for RKMK4, $v$ is the composite increment.
   - If $v$ does not depend on the variable being updated, the log-determinant is exactly zero (§3.4). Such layers are
     valid Monte Carlo proposals but cannot reshape a density.
   - An explicit step is not automatically invertible. §3.5 gives a sufficient condition and a test.
   - `torch.linalg.matrix_exp` is not slow in the forward pass for $3\times3$ matrices; the closed form gains in the
     backward pass (§2.4).
   - Autograd through RKMK4 is already the exact gradient of the discrete map. The reverse sweep in §5 is for custom
     `autograd.Function`s and memory control, not for correctness.

## 1. Conventions

| Algebra | Basis | Bracket | Notes |
|---|---|---|---|
| $\mathfrak{so}(3)$ | $\hat w=[w]_\times$ | $[\hat a,\hat b]=\widehat{a\times b}$ | $\mathrm{ad}_w=[w]_\times$ |
| $\mathfrak{su}(2)$ | $T_a=-\tfrac{i}{2}\sigma_a$ | $[T_a,T_b]=\epsilon_{abc}T_c$ | $\mathrm{ad}_w=[w]_\times$; $\theta=\lVert w\rVert$ is the SO(3) angle |
| $\mathfrak{su}(3)$ | $T_a=-\tfrac{i}{2}\lambda_a$ | $[T_a,T_b]=f_{abc}T_c$ | $\operatorname{tr}(T_aT_b)=-\tfrac12\delta_{ab}$ |
| $\mathfrak{se}(3)$ | $\xi=(\phi,\rho)$, $\hat\xi=\begin{pmatrix}[\phi]_\times&\rho\\0&0\end{pmatrix}$ | | $\mathrm{ad}_\xi=\begin{pmatrix}\Phi&0\\P&\Phi\end{pmatrix}$, $\Phi=[\phi]_\times$, $P=[\rho]_\times$ |

- $\mathrm{vee}$ for $\mathfrak{su}(N)$: $w_a=-2\,\mathrm{Re}\operatorname{tr}(T_aX)$.
- Projection $\mathfrak{gl}(N,\mathbb C)\to\mathfrak{su}(N)$: $P_{\mathfrak{su}}(X)=\tfrac12(X-X^\dagger)-\tfrac1{2N}\operatorname{tr}(X-X^\dagger)\,\mathbb 1$.
- Left-trivialised differential of the exponential:
  $\mathrm d\exp(\hat w)=\exp(\hat w)\,\widehat{\mathrm{dexp}^L_w\,\mathrm dw}$, with
  $\mathrm{dexp}^L_w=\dfrac{1-e^{-\mathrm{ad}_w}}{\mathrm{ad}_w}=\sum_{k\ge0}\dfrac{(-\mathrm{ad}_w)^k}{(k+1)!}$.
- Group-valued states are perturbed on the left: $U\mapsto\exp(\hat\xi)U$ (right-trivialised coordinates $\xi$).
- PyTorch complex gradients follow the PyTorch convention (gradient with respect to $\bar X$).

## 2. Exponentials and their differentials

Let $a(\theta)=\dfrac{1-\cos\theta}{\theta^2}$, $b(\theta)=\dfrac{\theta-\sin\theta}{\theta^3}$,
$\alpha(\theta)=\dfrac{a'(\theta)}{\theta}=\dfrac{\theta\sin\theta-2(1-\cos\theta)}{\theta^4}$,
$\beta(\theta)=\dfrac{b'(\theta)}{\theta}=\dfrac{1-\cos\theta}{\theta^4}-\dfrac{3(\theta-\sin\theta)}{\theta^5}$.
All four are entire in $\theta^2$; below $\theta^2<0.25$ use the power series
$a=\sum_k\frac{(-1)^k\theta^{2k}}{(2k+2)!}$, $b=\sum_k\frac{(-1)^k\theta^{2k}}{(2k+3)!}$,
$\alpha=\sum_{k\ge1}\frac{(-1)^k2k\,\theta^{2k-2}}{(2k+2)!}$, $\beta=\sum_{k\ge1}\frac{(-1)^k2k\,\theta^{2k-2}}{(2k+3)!}$.
Never branch on $\sqrt{\cdot}$ of a quantity that can be zero: gradients must be finite at $w=0$, the zero-initialised start.

### 2.1 SO(3) and SU(2)

$$\exp(\hat w)=\mathbb 1+(1-\theta^2 b)\,\Phi+a\,\Phi^2,\qquad \mathrm{dexp}^L_w=\mathbb 1-a\,\Phi+b\,\Phi^2,\qquad \Phi=[w]_\times,\ \theta=\lVert w\rVert .$$

SU(2): $\exp(w_aT_a)=\cos\tfrac\theta2\,\mathbb 1+\dfrac{\sin(\theta/2)}{\theta/2}\,w_aT_a$. In the basis $T_a$ the matrix
$\mathrm{dexp}^L_w$ is the same $3\times3$ matrix as for SO(3).

### 2.2 SE(3)

$$\exp(\hat\xi)=\begin{pmatrix}\exp\Phi & V\rho\\ 0&1\end{pmatrix},\quad V=\mathbb 1+a\Phi+b\Phi^2,\qquad
\mathrm{dexp}^L_\xi=\begin{pmatrix}J&0\\Q&J\end{pmatrix},\quad J=\mathbb 1-a\Phi+b\Phi^2,$$
$$Q=-a\,P+b\,(\Phi P+P\Phi)+(\phi\!\cdot\!\rho)\,\bigl(-\alpha\,\Phi+\beta\,\Phi^2\bigr).$$

*Derivation.* $\mathrm{ad}_\xi=\Phi+\varepsilon P$ has dual-number block structure, so for any analytic $F$,
$F(\mathrm{ad}_\xi)=\begin{pmatrix}F(\Phi)&0\\ DF(\Phi)[P]&F(\Phi)\end{pmatrix}$. Since $\Phi+sP=[\phi+s\rho]_\times$,
$Q=\frac{\mathrm d}{\mathrm ds}J(\phi+s\rho)\big|_{s=0}$, which gives the expression above using
$\frac{\mathrm d\theta}{\mathrm ds}=\frac{\phi\cdot\rho}{\theta}$.

### 2.3 Any algebra (fallback)

$\mathrm{dexp}^L_w=\varphi_1(-\mathrm{ad}_w)$ is the upper-right block of
$\exp\begin{pmatrix}-\mathrm{ad}_w&\mathbb 1\\0&0\end{pmatrix}$: one matrix exponential of size $2d$.
This is what §7 uses for $\mathfrak{su}(3)$. A closed form for $\mathfrak{su}(3)$ is an open item (§9).

### 2.4 SU(3) closed-form exponential

Let $X\in\mathfrak{su}(3)$, $Q=-iX$ (Hermitian, traceless), $c_1=\tfrac12\operatorname{tr}Q^2\ge0$, $c_0=\det Q=\tfrac13\operatorname{tr}Q^3$.
The characteristic polynomial is $\lambda^3-c_1\lambda-c_0=0$. With $s=\sqrt{c_1/3}$ and
$\varphi=\tfrac13\arccos\!\bigl(\mathrm{clamp}(c_0/2s^3,-1,1)\bigr)\in[0,\tfrac\pi3]$,
$$\lambda_k=2s\cos\!\bigl(\varphi-\tfrac{2\pi k}{3}\bigr),\quad k=0,1,2;\qquad \lambda_0\ge\lambda_1\ge\lambda_2,\quad \lambda_0-\lambda_2\ge3s .$$
Only two eigenvalues are independent ($\lambda_0+\lambda_1+\lambda_2=0$). Newton interpolation of $f(\lambda)=e^{i\lambda}$ on the spectrum gives
$$\exp X=h_0\,\mathbb 1+h_1\,(Q-\lambda_0)+h_2\,(Q-\lambda_0)(Q-\lambda_1),$$
$$h_0=e^{i\lambda_0},\quad h_1=f[\lambda_0,\lambda_1],\quad h_2=\frac{f[\lambda_1,\lambda_2]-f[\lambda_0,\lambda_1]}{\lambda_2-\lambda_0},\qquad
f[p,q]=i\,e^{i(p+q)/2}\,\mathrm{sinc}\!\Bigl(\frac{q-p}{2}\Bigr),\ \ \mathrm{sinc}\,x=\frac{\sin x}{x}.$$

*Why it is stable.* First divided differences use $\mathrm{sinc}$, so coinciding eigenvalues ($\lambda_0=\lambda_1$ or
$\lambda_1=\lambda_2$) cause no division by a small number. The only division is by $\lambda_2-\lambda_0\ge3s$, which
vanishes only at $X=0$. The formula is scale-invariantly accurate down to $\lVert w\rVert=10^{-8}$; a Taylor series
(order 4) is used only for $c_1<10^{-12}$.
The Cayley–Hamilton coefficients of $\exp X=f_0+f_1Q+f_2Q^2$ follow as
$f_2=h_2$, $f_1=h_1-h_2(\lambda_0+\lambda_1)$, $f_0=h_0-h_1\lambda_0+h_2\lambda_0\lambda_1$.

```python
def su3_exp(X, taylor_cut=1e-12, K=4):
    Q = -1j * X; c1 = trace(Q @ Q).real / 2; small = c1 < taylor_cut
    Qs = where(small, diag(1, 0, -1), Q)                 # safe input for the unused branch (no NaN gradients)
    c1s = trace(Qs @ Qs).real / 2; c0s = trace(Qs @ Qs @ Qs).real / 3; s = sqrt(c1s / 3)
    phi = arccos(clamp(c0s / (2 * s**3), -1, 1)) / 3
    lam = [2 * s * cos(phi - 2 * pi * k / 3) for k in range(3)]
    dd = lambda p, q: 1j * exp(0.5j * (p + q)) * sinc((q - p) / 2)        # sinc(x) = sin(x)/x
    h0 = exp(1j * lam[0]); h1 = dd(lam[0], lam[1]); h2 = (dd(lam[1], lam[2]) - h1) / (lam[2] - lam[0])
    closed = h0 * I + h1 * (Qs - lam[0] * I) + h2 * (Qs - lam[0] * I) @ (Qs - lam[1] * I)
    return where(small, taylor(X, K), closed)
```

*Backward pass.* Autograd through this function was finite and correct in all 1500 stress cases, including exactly
degenerate eigenvalues (worst relative error 5.0e-11 in that run, at gap $10^{-10}$; 5.2e-10 in the 100-case check of the verification script). A backward pass that is
exact by construction at degeneracy is the Daleckii–Krein formula, for $X=V\,\mathrm{diag}(i\lambda)\,V^\dagger$:
$$\bar X=V\bigl[(V^\dagger\bar E\,V)\circ\overline{\Gamma}\bigr]V^\dagger,\qquad \Gamma_{jk}=e^{i(\lambda_j+\lambda_k)/2}\,\mathrm{sinc}\!\Bigl(\frac{\lambda_j-\lambda_k}{2}\Bigr).$$
It needs eigenvectors (`eigh`), so use it only inside a custom `autograd.Function` when guaranteed safety matters.

*Speed* (CPU, 12 threads, 16,384 matrices, complex128, milliseconds):

| Method | forward | backward | total |
|---|---|---|---|
| `torch.linalg.matrix_exp` | 13.9 | 58.3 | 72.2 |
| closed form, both branches evaluated | 17.6 | 31.2 | 48.8 |
| closed form, no small-norm branch | 10.8 | 15.5 | 26.3 |

The forward pass alone is not faster than `matrix_exp`; the gain is in the backward pass. Evaluate the Taylor branch only
on the masked subset to approach the last row. GPU timings are **[not measured]**.

## 3. Jacobian log-determinants

### 3.1 Master formula

Let $G$ be a unimodular matrix Lie group of dimension $d$ and let a step be $\Phi(x)=\exp\bigl(\hat v(x)\bigr)\cdot x$.

**Group-valued state** ($x=U\in G$, Haar measure). Define $(Dv)_{ab}=\partial v_a(\exp(\hat\xi)U)/\partial\xi_b\big|_{\xi=0}$. Then
$$J=\mathrm{Ad}_{\exp v}\,\bigl(\mathbb 1_d+\mathrm{dexp}^L_v\,Dv\bigr),\qquad
\boxed{\ \log\lvert\det J\rvert=\log\bigl\lvert\det\bigl(\mathbb 1_d+\mathrm{dexp}^L_v\,Dv\bigr)\bigr\rvert\ }$$

*Derivation.* $\delta\Phi\,\Phi^{-1}=\mathrm{dexp}^R_v(Dv\,\xi)+\mathrm{Ad}_{\exp v}\xi$ with
$\mathrm{dexp}^R_v=\mathrm{Ad}_{\exp v}\,\mathrm{dexp}^L_v$, and $\lvert\det\mathrm{Ad}\rvert=1$ for unimodular groups
(SO(3), SU(N), SE(3) all qualify). Because $J$ is written in invariant frames, its determinant is the Jacobian with
respect to Haar measure.

**Vector-valued state** ($x=z\in\mathbb R^n$, Lebesgue measure; $\mathbb C^N\cong\mathbb R^{2N}$). With $Dv=\partial v/\partial z$
($d\times n$) and $A(z)\in\mathbb R^{n\times d}$, $A(z)e_a=\hat e_a z$,
$$J=e^{\hat v}\bigl(\mathbb 1_n+A(z)\,\mathrm{dexp}^L_v\,Dv\bigr),\qquad
\boxed{\ \log\lvert\det J\rvert=\log\bigl\lvert\det\bigl(\mathbb 1_d+\mathrm{dexp}^L_v\,Dv\,A(z)\bigr)\bigr\rvert\ }$$
by Sylvester's identity. The determinant is $d\times d$ whatever $n$ is. Because the step preserves every orbit, the same
number is the Jacobian on the orbit (sphere) with its induced measure.

### 3.2 Lie–Euler

$v(x)=h\,\omega_\theta(x)$. Use §3.1 with the closed-form $\mathrm{dexp}^L$ of §2. $Dv$ costs $d$ reverse-mode passes
through the connection head (3 for SO(3)/SU(2), 6 for SE(3), 8 for SU(3)), batched over samples.

### 3.3 RKMK4

With $f=\omega_\theta$:
$$k_1=hf(x),\ \ k_2=hf\bigl(e^{k_1/2}x\bigr),\ \ k_3=hf\bigl(e^{k_2/2-[k_1,k_2]/8}x\bigr),\ \ k_4=hf\bigl(e^{k_3}x\bigr),$$
$$v=\tfrac16(k_1+2k_2+2k_3+k_4)-\tfrac1{12}[k_1,k_4],\qquad \Phi(x)=e^{\hat v}x .$$
The log-determinant is §3.1 with this composite $v$; $Dv$ is obtained by differentiating through the four stages. No
further simplification exists. Note that this is the determinant of the *discrete* map, which is what a sampler needs;
it differs from $\int\operatorname{div}$ of the continuous flow by the integrator's discretisation error.

### 3.4 Corollary: coupling layers are volume-preserving

If $v$ depends only on other (frozen) variables, $Dv=0$ and $\log\lvert\det J\rvert=0$ exactly. A stack of such layers
leaves the Haar density unchanged: it is a valid Metropolis proposal but not a density-reshaping flow. To reshape a
density, $v$ must depend on the variable it updates (as the loop head of §4.3 does).

### 3.5 Invertibility

**Proposition.** Let $G$ be compact and connected, with a basis in which $\mathrm{ad}$ is skew (true for the bases of §1).
If $\sup_x\lVert Dv(x)\rVert_2<1$, then $\Phi$ is a diffeomorphism of $G$.

*Sketch.* $\mathrm{dexp}^L_v$ is normal with eigenvalues of modulus $\lvert\mathrm{sinc}(\vartheta/2)\rvert\le1$, so
$\lVert\mathrm{dexp}^L_vDv\rVert<1$ and $J$ is invertible everywhere; $\Phi$ is a local diffeomorphism of a compact
connected manifold, hence a covering map; it is homotopic to the identity through $tv$, hence of degree 1, hence a single
sheet. For SE(3) (non-compact) $\lVert\mathrm{dexp}^L\rVert$ is not bounded by 1; check $\det>0$ directly **[bound not derived]**.

**Test.** $\mathbb E_{\rm Haar}[\det J]=1$ always (degree). $\mathbb E_{\rm Haar}[\lvert\det J\rvert]=1$ only without folding.

| SO(3), 200,000 Haar samples | $\max\lVert Dv\rVert_2$ | mean $\det J$ | mean $\lvert\det J\rvert$ | min $\det J$ |
|---|---|---|---|---|
| small field | 0.36 | 1.00012 ± 0.00033 | 1.00012 | 0.68 |
| large field | 3.55 | 1.0059 ± 0.0027 | 1.240 | -2.49 |

Always take the sign from `slogdet`; a negative determinant anywhere means the density formula is invalid.

```python
def logabsdet_step(dexpL_v, Dv, A=None):            # group state: A=None ; vector state: A = (n, d)
    M = dexpL_v @ Dv if A is None else dexpL_v @ Dv @ A
    return torch.linalg.slogdet(torch.eye(M.shape[-1]) + M)      # (sign, log|det|)

# batched Dv for a group-valued state (rows by reverse mode; samples are independent)
xi = zeros(B, d, requires_grad=True)
v_xi = vee(v_fn((I + hat(xi)) @ U))                 # first order in xi is enough
Dv = stack([grad(v_xi[:, a].sum(), xi, create_graph=True)[0] for a in range(d)], 1)
```

## 4. Equivariant connection heads for SU(2) and SU(3)

Requirement: $\omega(g\cdot x)=g\,\omega(x)\,g^{-1}$. Then $x\mapsto\exp(\omega(x))\cdot x$ is equivariant. The invariant
tensors of SU(N) are the Hermitian form $\delta$ and the antisymmetric $\epsilon$; every formula below is a contraction with these.

### 4.1 Fundamental vectors $z_a\in\mathbb C^N$, $a=1..K$

| | Invariants | Extra covariant vectors (transform like $z$) |
|---|---|---|
| U(N) part | $G_{ab}=z_a^\dagger z_b$ | — |
| SU(2) only | $B_{ab}=\epsilon_{ij}z_a^iz_b^j$ | $\tilde z_a=\epsilon\,\bar z_a=(\bar z_a^2,-\bar z_a^1)$ |
| SU(3) only | $D_{abc}=\det[z_a\,z_b\,z_c]$ | $w_{ab}=\overline{z_a\times z_b}$ |

The $\epsilon$-invariants change under a U(1) phase, so they are what distinguishes SU(N) from U(N). They need no separate
code path: with the augmented set $\{v\}=\{z_a\}\cup\{\tilde z_a\}$ (or $\cup\{w_{ab}\}$), the Hermitian Gram matrix
$G=VV^\dagger$ already contains them ($\tilde z_a^\dagger z_b=-B_{ab}$, $w_{ab}^\dagger z_c=D_{abc}$).

$$\boxed{\ \omega_a=P_{\mathfrak{su}}\Bigl(\sum_{b,c}C^{(a)}_{bc}(G)\;v_b\,v_c^\dagger\Bigr),\qquad C^{(a)}_{bc}\in\mathbb C\ \text{any function of the invariants}\ }$$

*Why.* $v\mapsto gv$ gives $v_bv_c^\dagger\mapsto g\,v_bv_c^\dagger\,g^\dagger$; $P_{\mathfrak{su}}$ commutes with conjugation; $C$ is invariant.

*Pitfall.* With a single vector and no augmentation, $P_{\mathfrak{su}}(c\,zz^\dagger)=i\,\mathrm{Im}(c)\,(zz^\dagger-\tfrac{|z|^2}{N})$,
which acts on $z$ as a pure phase. Non-trivial dynamics need $K\ge2$ or the $\epsilon$-augmented vectors.

*Completeness.* By the first fundamental theorem for SU(N), invariants of vectors and covectors are generated by
$\delta$- and $\epsilon$-contractions, so this family spans the polynomial equivariant maps **[stated from theory, not verified numerically]**.

```python
def head_fundamental(V, coeff):                     # V: (M, N) augmented covariant vectors
    G = V @ V.mH                                    # all invariants
    return proj_su(einsum("abc,bi,cj->aij", coeff(G), V, V.conj()))   # coeff(G): (K, M, M) complex
```

### 4.2 Adjoint features $X_a\in\mathfrak{su}(N)$ ($X\mapsto gXg^\dagger$)

Invariants: $\operatorname{tr}(X_aX_b)$, $\operatorname{tr}(X_aX_bX_c)$. Covariants:
$$X_a,\qquad [X_a,X_b]\ \ (f\text{-type}),\qquad i\Bigl(\{X_a,X_b\}-\tfrac2N\operatorname{tr}(X_aX_b)\mathbb 1\Bigr)\ \ (d\text{-type}).$$
The $d$-type covariant vanishes identically for SU(2) (measured 0) and not for SU(3) (measured magnitude 1.56).
$\omega_a=\sum c^{(1)}_{ab}X_b+\sum c^{(2)}_{abc}[X_b,X_c]+\sum c^{(3)}_{abc}\,d(X_b,X_c)$ with real invariant coefficients.

### 4.3 Group-valued features (lattice links)

Let $W$ be an untraced closed loop based at the start site $x$ of the link, so $W\mapsto g(x)\,W\,g(x)^\dagger$, and $t=\operatorname{tr}W$.
$$\boxed{\ \omega=P_{\mathfrak{su}}\bigl(c_1(t)\,W+c_2(t)\,W^2\bigr),\qquad c_1,c_2\in\mathbb C\ }\qquad U'=\exp(\omega)\,U .$$
For SU(3), Cayley–Hamilton reduces every higher power and $W^\dagger$ to $\{\mathbb 1,W,W^2\}$ with invariant coefficients,
and $\operatorname{tr}W$ fixes the conjugacy class, so no further terms or invariants are needed for a single loop.
$\omega$ commutes with $W$, so it has two real degrees of freedom per loop; the four real coefficient functions are redundant by design.
For a link $U_\mu(x)$ take $W=U_\mu(x)\cdot\sum(\text{staples})$ (plaquettes containing the link, opened at $x$).
Gauge covariance on a $4\times4$ periodic SU(3) lattice: 4.7e-15.

The log-determinant of a full lattice layer needs a masking pattern so that the Jacobian is block-triangular; each
diagonal block is then §3.1 with $Dv$ taken with respect to the active link **[masking pattern not specified here]**.

## 5. RKMK4 backward pass

Autograd through §3.3 is exact for the discrete map. Use the following only to write custom `autograd.Function`s
(closed-form $\mathrm{dexp}$ instead of differentiating through $\exp$, smaller graphs). Bars denote cotangents.

**Primitive 1 — action** $y=e^{\hat v}z$:
$$\bar z\mathrel{+}=E^{\top}\bar y,\qquad \bar v\mathrel{+}=(\mathrm{dexp}^L_v)^{\top}A(z)^{\top}E^{\top}\bar y,\qquad E=e^{\hat v}$$
(for SO(3) on $\mathbb R^3$: $A(z)^\top u=z\times u$; for complex representations use $E^\dagger$ and real parts).

**Primitive 2 — bracket** $c=[a,b]$, $c_c=f_{abc}a_ab_b$:
$$\bar a_a\mathrel{+}=f_{abc}\,b_b\,\bar c_c,\qquad \bar b_b\mathrel{+}=f_{abc}\,a_a\,\bar c_c\qquad(\mathfrak{so}(3):\ \bar a\mathrel{+}=b\times\bar c,\ \ \bar b\mathrel{+}=\bar c\times a).$$

**Primitive 3 — field** $k=hf(y)$: $\bar y\mathrel{+}=h\,(\partial f/\partial y)^\top\bar k$ and parameter gradients, both by autograd on the user's network.

**Reverse sweep** (forward quantities $k_i$, $y_2=e^{k_1/2}z$, $u_3=k_2/2-[k_1,k_2]/8$, $y_3=e^{u_3}z$, $y_4=e^{k_3}z$ are stored):

```
(v̄, z̄)      ← action_vjp(v, z, z̄₁)
k̄₁ = v̄/6 − ⟨bracket: ā of [k₁,k₄]⟩/12 ;  k̄₄ = v̄/6 − ⟨b̄ of [k₁,k₄]⟩/12 ;  k̄₂ = k̄₃ = v̄/3
ȳ₄ = field_vjp(y₄, k̄₄) ;  (t, z̄') ← action_vjp(k₃, z, ȳ₄) ;  k̄₃ += t ;  z̄ += z̄'
ȳ₃ = field_vjp(y₃, k̄₃) ;  (ū₃, z̄') ← action_vjp(u₃, z, ȳ₃) ;  z̄ += z̄'
k̄₂ += ū₃/2 − ⟨b̄ of [k₁,k₂]⟩/8 ;  k̄₁ += −⟨ā of [k₁,k₂]⟩/8
ȳ₂ = field_vjp(y₂, k̄₂) ;  (t, z̄') ← action_vjp(k₁/2, z, ȳ₂) ;  k̄₁ += t/2 ;  z̄ += z̄'
z̄ += field_vjp(z, k̄₁)
```

Here `field_vjp` includes the factor $h$ (Primitive 3), and ⟨ā of [a,b]⟩, ⟨b̄ of [a,b]⟩ are Primitive 2 evaluated with
$\bar c$ equal to the cotangent being distributed ($\bar v$ or $\bar u_3$).

Hand-written sweep against autograd (SO(3), three coupled vectors): 1.1e-16. The exact signs are in `sec_adjoint` of the script.

*Memory.* RKMK4 is not algebraically reversible, so there is no exact constant-memory adjoint; the continuous adjoint
returns the gradient of the ODE, not of the discrete map. Store the four stages, or checkpoint per step.

## 6. Reference implementations

| Spec section | `lagmath.py` |
|---|---|
| §2 | `so3_exp`, `su2_exp`, `se3_exp`, `su3_exp`, `dexpL_so3`, `dexpL_se3`, `dexpL_generic`, `exp_vjp_normal` |
| §3 | `logabsdet_step` |
| §4 | `proj_su`, `eps_dual_su2`, `cross_dual_su3`, `head_fundamental`, `head_adjoint`, `head_loops` |
| §5 | `sec_adjoint` in `verify_math_spec.py` |

## 7. End-to-end validation: flow on a single SU(3) variable

Target $p(U)\propto\exp\bigl(\tfrac\beta3\mathrm{Re}\operatorname{tr}U\bigr)$, $\beta=6$, with respect to Haar measure.
Six layers $U\mapsto\exp(\omega(U))U$ with the loop head of §4.3 ($W=U$), `su3_exp` of §2.4, the log-determinant of §3.1
with $\mathrm{dexp}^L$ from §2.3; 7704 parameters, zero-initialised output layers, reverse-KL training, 350 Adam steps
(471.9 s on CPU). Exact reference values come from the Weyl integration formula on a $600\times600$ torus grid.

| Quantity | Value |
|---|---|
| ESS of the Haar prior against the target (exact) | 11.2 % |
| ESS of the trained flow, two independent sets of 32,768 samples | 99.96 %, 99.96 % |
| $\log Z$: flow estimate / exact | 1.23583 / 1.23589 |
| Reverse KL at the end of training | 2.9e-04 |
| $\langle\mathrm{Re}\operatorname{tr}U\rangle/3$: unweighted flow samples / exact | 0.4224 ± 0.0014 / 0.4225 |
| Sign of $\det J$ over all layers and samples | all positive |
| $\max\lVert D\omega\rVert_2$ | 0.36 (< 1, so §3.5 applies) |

This validates §2.4, §3.1, §3.5 and §4.3 jointly, on one seed. It is a single-variable problem: it says nothing yet about
multi-link lattices, where the masking pattern and cost of $Dv$ dominate.

## 8. Verification results

From `verify_results.json`, produced by `python verify_math_spec.py` in a clean process (float64, CPU). Values are maximum
absolute errors unless the name says otherwise.

| Check | Measured |
|---|---|
| exp so3 closed vs matrix_exp | 1.2e-15 |
| exp su2 closed vs matrix_exp | 7.0e-16 |
| exp se3 closed vs matrix_exp | 1.8e-15 |
| dexpL so3 closed vs phi1 block-exp | 4.4e-16 |
| dexpL so3 small angle | 3.3e-16 |
| dexpL se3 closed (Q = D_rho J) vs phi1 block-exp | 8.9e-16 |
| dexpL se3 small angle | 2.2e-16 |
| exp su3 closed: generic | 2.4e-15 |
| exp su3 closed: norm x10 | 2.6e-14 |
| exp su3 closed: norm x1e-7 | 5.6e-16 |
| exp su3 closed: X = 0 | 0 |
| exp su3 closed: degenerate eigenvalues | 3.3e-16 |
| exp su3 closed: unitarity, det | 3.1e-15, 2.4e-15 |
| grad through su3 closed form vs matrix_exp (incl. degenerate): rel err | 5.2e-10 |
| grad through su3 closed form: non-finite count | 0 |
| grad through su3 closed form at X = 0 | 2.2e-16 |
| Daleckii-Krein exp VJP at degenerate X | 9.2e-16 |
| Jacobian SO(3) Lie-Euler: \|J_formula - J_bruteforce\| | 2.2e-16 |
| Jacobian SO(3) Lie-Euler: \|logdet formula - brute force\| | 3.9e-16 |
| Jacobian SO(3) RKMK4: \|J_formula - J_bruteforce\| | 2.2e-16 |
| Jacobian SO(3) RKMK4: \|logdet formula - brute force\| | 2.2e-16 |
| Jacobian SU(2) Lie-Euler: \|J_formula - J_bruteforce\| | 1.1e-16 |
| Jacobian SU(2) Lie-Euler: \|logdet formula - brute force\| | 1.9e-16 |
| Jacobian SU(2) RKMK4: \|J_formula - J_bruteforce\| | 3.3e-16 |
| Jacobian SU(2) RKMK4: \|logdet formula - brute force\| | 7.4e-16 |
| Jacobian SE(3) Lie-Euler: \|J_formula - J_bruteforce\| | 4.4e-16 |
| Jacobian SE(3) Lie-Euler: \|logdet formula - brute force\| | 7.8e-16 |
| Jacobian SE(3) RKMK4: \|J_formula - J_bruteforce\| | 8.9e-16 |
| Jacobian SE(3) RKMK4: \|logdet formula - brute force\| | 2.2e-16 |
| Jacobian vector state SO(3) on R^3 | 0 |
| Jacobian vector state SU(2) on C^2 (3x3 det vs 4x4 autograd) | 5.7e-16 |
| SU(2) fundamental head: rel equivariance error | 7.9e-16 |
| SU(2) fundamental head: output in su(N) (trace, anti-Hermitian) | 5.4e-16, 8.9e-16 |
| SU(2) layer z -> exp(omega) z: equivariance, norm drift | 6.1e-15, 2.2e-16 |
| SU(2) adjoint head: rel equivariance error | 1.5e-15 |
| SU(2) d-type covariant i{X,Y} traceless: max abs (0 for SU(2)) | 0 |
| SU(3) fundamental head: rel equivariance error | 1.2e-15 |
| SU(3) fundamental head: output in su(N) (trace, anti-Hermitian) | 1.1e-14, 2.5e-15 |
| SU(3) layer z -> exp(omega) z: equivariance, norm drift | 3.0e-14, 6.7e-16 |
| SU(3) adjoint head: rel equivariance error | 1.8e-15 |
| SU(3) d-type covariant i{X,Y} traceless: max abs (0 for SU(2)) | 1.6e+00 |
| lattice SU(3) 4x4 stout-type layer: gauge covariance | 4.7e-15 |
| RKMK4 hand-written reverse sweep vs autograd | 1.1e-16 |

## 9. Open items

1. Closed-form $\mathrm{dexp}^L$ for $\mathfrak{su}(3)$ (currently the $16\times16$ block exponential of §2.3).
2. Masking patterns and total log-determinant for multi-link lattice layers (§4.3).
3. Invertibility bound for SE(3) and other non-compact groups (§3.5).
4. Completeness of the heads in §4 is argued from invariant theory, not tested.
5. An eigenvector-free backward pass for `su3_exp` (the Morningstar–Peardon derivative formulas) was not re-derived.
6. float32 behaviour and GPU timings were not measured.
7. The flow validation used one seed and one target.

## 10. Build order for the repository

1. `lag.algebra` (bases, hat/vee, `proj_su`, structure constants) and `lag.exp` (§2) — tests: `sec_exp`.
2. `lag.jacobian` (§3) — tests: `sec_jacobian`, including the Haar degree test.
3. `lag.heads` (§4) — tests: `sec_heads`, including the lattice gauge-covariance test.
4. `lag.flows` (`lie_euler`, `rkmk4`, optional custom backward of §5) — tests: `sec_adjoint`, convergence order.
5. Example `examples/su3_single_variable_flow.py` — acceptance: ESS ≥ 99 % and all $\det J>0$ (§7).
6. Merge with the generic `MatrixLieGroup` core (`lag_proto.py`): the closed forms here become backend overrides.

## References

Cited from memory; check before publication. H. Munthe-Kaas, *High order Runge–Kutta methods on manifolds* (1999).
A. Iserles et al., *Lie-group methods*, Acta Numerica (2000). C. Morningstar, M. Peardon, *Analytic smearing of SU(3) link
variables in lattice QCD* (2004). D. Boyda et al., *Sampling using SU(N) gauge equivariant flows* (2021).
T. Barfoot, *State Estimation for Robotics* (SE(3) Jacobians). N. Higham, *Functions of Matrices* (block formula for
Fréchet derivatives; Daleckii–Krein).
