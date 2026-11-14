# LAG — MATH_SPEC_LATTICE

Multi-link lattice gauge layers: masking, total log-determinant, inverse and cost. Companion to `MATH_SPEC.md`,
whose conventions (§1), exponentials (§2), master log-determinant formula (§3.1) and loop head (§4.3) are used
throughout. Reference implementation: `laglattice.py`, built on `lag` at commit `3bcd2d18d0c2`. Verification:
`verify_lattice.py` (64 recorded results, 0 failures, 9 negative controls). End-to-end test:
`lattice_flow_2d.py`. Other records: `lattice_cost_profile.json` (§4), `exact_bessel_checks.json` (§7),
`lattice_flow_*.json` and `lattice_eval_*.json` (§8). `fill_spec_lattice.py` generates this file from
`MATH_SPEC_LATTICE_template.md`, the §8 text in `e2e_section.md` and these records. Nothing below is stated without
a check unless it is marked **[not verified]**.

## 0. Results in brief

1. **Masking theorem (§2).** If every updated link's connection depends only on that link and on frozen links,
   the Jacobian of the whole lattice map is block-triangular, and $\log|\det J|$ is the sum of the single-link
   formula of MATH_SPEC §3.1 over the updated links. Brute-force Jacobians of the full lattice map agree to
   1.1e-13 (D = 2, 3, 4; SU(2) and SU(3)). With the condition violated, the same code is wrong by up to 2.9
   and raises no error.
2. **Masks for plaquette kernels (§3).** Update one direction μ at a time, on the links whose transverse
   coordinates have a fixed parity: 2D sub-layers per sweep, L even.
3. **Cost (§4).** The log-determinant of a whole sub-layer needs d reverse passes (3 for SU(2), 8 for SU(3)) at
   any volume. It multiplies the cost of the layer alone by 13.2 for SU(3) and 3.2 for SU(2); differentiating
   the generic su(3) $\mathrm{dexp}^L$ is 42 % of the SU(3) total.
4. **Spectral shortcut (§5), inverse (§6), exact 2D reference (§7)**, each verified.
5. **End-to-end (§8).** A 17,760-parameter masked flow trained on a 4×4 torus reaches ESS 90.9 % for SU(3) at β = 3 and 76.3 % for SU(2) at β = 2, where the Haar prior has 2.6e-05 and 2.7e-05. Applied without retraining at L = 8 and 16, its log Z estimates stay within 1.9 standard errors of the exact values, all det J > 0, and the ESS follows $e^{-V\sigma^2}$ with a constant per-plaquette $\sigma^2$.

## 1. Setting

Lattice $\Lambda=\mathbb Z_L^D$ (periodic), links $(x,\mu)$, $U_\mu(x)\in SU(N)$, reference measure $\prod_{\rm links}$ Haar.
Gauge transformations $U_\mu(x)\mapsto g(x)\,U_\mu(x)\,g(x+\hat\mu)^\dagger$. Plaquette
$P_{\mu\nu}(x)=U_\mu(x)\,U_\nu(x+\hat\mu)\,U_\mu(x+\hat\nu)^\dagger\,U_\nu(x)^\dagger$, Wilson action
$S(U)=-\frac{\beta}{N}\sum_{x,\ \mu<\nu}\mathrm{Re}\operatorname{tr}P_{\mu\nu}(x)$, target $p(U)\propto e^{-S(U)}$.

A **sub-layer** splits the links into an active set $A$ and a frozen set $F$:
$$U'_l=\exp(\omega_l)\,U_l\quad(l\in A),\qquad U'_f=U_f\quad(f\in F).$$
If $\omega_l$, $l=(x,\mu)$, is built from untraced loops based at $x$ (MATH_SPEC §4.3), then
$\omega_l\mapsto g(x)\,\omega_l\,g(x)^\dagger$ and the sub-layer is gauge-covariant for any choice of $A$.
Coordinates: every link is perturbed on the left, $U_l\mapsto\exp(\hat\xi_l)U_l$ (right-trivialised,
$\xi_l\in\mathbb R^d$), as in MATH_SPEC §1; $v_l=\mathrm{vee}(\omega_l)$.

## 2. Masking theorem

**Proposition 1.** Suppose that for every $l\in A$, $\omega_l$ depends only on $U_l$ and on $\{U_f: f\in F\}$.
Ordering the coordinates as $(A,F)$,
$$J=\begin{pmatrix}\operatorname{diag}_{l\in A}J_l & J_{AF}\\ 0 & \mathbb 1\end{pmatrix},\qquad
J_l=\mathrm{Ad}_{\exp v_l}\bigl(\mathbb 1_d+\mathrm{dexp}^L_{v_l}\,Dv_l\bigr),\qquad
(Dv_l)_{ab}=\frac{\partial v_l^a}{\partial\xi_l^b}\Big|_{\xi=0},$$
$$\boxed{\ \log|\det J|=\sum_{l\in A}\log\bigl|\det\bigl(\mathbb 1_d+\mathrm{dexp}^L_{v_l}\,Dv_l\bigr)\bigr|,\qquad
\operatorname{sign}\det J=\prod_{l\in A}\operatorname{sign}\det\bigl(\mathbb 1_d+\mathrm{dexp}^L_{v_l}Dv_l\bigr)\ }$$

*Proof.* Frozen links are unchanged, so their rows of $J$ are rows of the identity. For $l\ne m$ in $A$,
$U'_l=\exp(\omega_l(U_l,U_F))\,U_l$ does not depend on $U_m$, so $J_{lm}=0$. The diagonal block $J_l$ is the
single-variable Jacobian of MATH_SPEC §3.1 with $U_F$ held fixed, and $|\det\mathrm{Ad}|=1$ because SU(N) is
unimodular. A block-triangular determinant is the product of its diagonal blocks. ∎

A stack of sub-layers has the sum of their log-determinants (chain rule; the reference measure is invariant), so
for samples pushed forward from the Haar prior $\log q(U_{\rm out})=-\sum_k\log|\det J_k|$.

**Corollary (batched $Dv$).** Under the same condition $\partial v_m/\partial\xi_l=0$ for $m\ne l$, so the gradient
of $\sum_{m\in A}v_m^a$ with respect to all $\xi$ at once is row $a$ of every $Dv_l$. One reverse pass per algebra
index gives the log-determinant of the whole sub-layer, for every active link and every configuration of the batch:
$d$ passes at any volume.

```python
xi   = zeros(B, n_active, d, requires_grad=True)                # one perturbation per active link
w_xi = vee(omega((I + hat(xi)) @ U_active))                     # staples are frozen: computed once, outside
Dw   = stack([grad(w_xi[..., a].sum(), xi, create_graph=True)[0] for a in range(d)], -2)
sign, ld = slogdet(I_d + dexpL(w) @ Dw)                         # (B, n_active)
logdet = ld.sum(-1)                                             # (B,)  ; create_graph=False for evaluation
```

*Failure mode.* If the condition is violated, the same code silently returns $\sum_m\partial v_m/\partial\xi_l$.
In the negative controls (§9) the log-determinant is then wrong by 1.4–2.9 (every μ-link active) and 0.11
(odd L). A library must validate masks (§3.3) rather than trust them.

## 3. Masks

### 3.1 Plaquette staples and the conflict graph

Staples of $(x,\mu)$, for each $\nu\ne\mu$, oriented so that $U_\mu(x)\cdot\text{staple}$ is a plaquette based at $x$:
$$\text{upper: }U_\nu(x+\hat\mu)\,U_\mu(x+\hat\nu)^\dagger\,U_\nu(x)^\dagger,\qquad
\text{lower: }U_\nu(x+\hat\mu-\hat\nu)^\dagger\,U_\mu(x-\hat\nu)^\dagger\,U_\nu(x-\hat\nu).$$
A kernel reading these $2(D-1)$ loops depends on the $\nu$-links at $x,\ x+\hat\mu,\ x-\hat\nu,\ x+\hat\mu-\hat\nu$
and on the $\mu$-links at $x\pm\hat\nu$. Conflict graph: $l\sim m$ iff $l$ and $m$ lie on a common plaquette (for
plaquette kernels "reads" is symmetric). An active set must be an independent set and a sweep is a proper colouring.
The four links of every plaquette need four different colours, so a sweep has at least 4 sub-layers in any D.

### 3.2 The mask family $M(\mu,p)$

$$M(\mu,p)=\Bigl\{(x,\mu):\ \textstyle\sum_{\nu\ne\mu}x_\nu\equiv p\pmod 2\Bigr\},\qquad \mu=0,\dots,D-1,\ \ p\in\{0,1\}.$$
- No two active links share a plaquette: other directions are frozen, and two μ-links share a plaquette only at
  $x$ and $x\pm\hat\nu$, whose transverse parities differ.
- Well defined only for **L even** in every direction $\nu\ne\mu$; for odd L the parity wraps around the torus.
- A sweep is the $2D$ sub-layers $(\mu,p)$: every link is updated exactly once per sweep, a fraction $1/(2D)$ per
  sub-layer. In D = 2 this meets the lower bound of 4; for D > 2 masks with fewer sub-layers may exist (open item).
- In the $(\mu,\nu)$-planes every plaquette contains exactly one active link (its two μ-links have opposite parity).
- With constant $c_1$ and $c_2=0$ the update has the form of a stout-smearing step (Morningstar–Peardon) applied on a
  mask; trainable invariant coefficients make it a flow layer.

### 3.3 Validating a mask

For any kernel, check independence numerically on a small lattice before use: perturb one active link at a time
and verify that the connections of all other active links are unchanged (`validate_mask` in `laglattice.py`;
positive and negative cases in §9). The brute-force Jacobian comparison of §9 is the stronger test.

## 4. Cost

Per sub-layer with $n_A$ active links: forward (staples, ω, exp) is $O(n_A)$; the log-determinant adds $d$ reverse
passes for $Dv$, $\mathrm{dexp}^L$ (closed form for SU(2); the $2d\times2d$ block exponential of MATH_SPEC §2.3 for
SU(3)) and a $d\times d$ `slogdet`; training adds a double backward through the $Dv$ passes.

The stage breakdown and the overhead table below are recorded in `lattice_cost_profile.json`; the volume-scaling
table after them is `sec_scaling` of `verify_lattice.py`.

Stage breakdown of one sub-layer, 2D, L = 4, batch 256 (2048 active links), CPU, 6 threads, float64:

| SU(3) stage | ms | share |
|---|---|---|
| layer (staples, omega, exp) | 13.1 | 5 % |
| D omega (8 reverse passes) | 55.6 | 19 % |
| dexpL | 44.9 | 15 % |
| slogdet | 1.7 | 1 % |
| backward of all | 175.3 | 60 % |
| total | 290.6 | |

| SU(2) stage | ms | share |
|---|---|---|
| layer (staples, omega, exp) | 10.6 | 15 % |
| D omega (3 reverse passes) | 21.2 | 31 % |
| dexpL | 4.1 | 6 % |
| slogdet | 0.7 | 1 % |
| backward of all | 32.4 | 47 % |
| total | 69.0 | |

Forward + backward of one sub-layer, same setting (ms). The middle column drops the gradient through $\mathrm{dexp}^L$
(not a valid gradient; it measures the share of that term):

| Group | layer only | with log det, $\mathrm{dexp}^L$ detached | with log det | overhead factor |
|---|---|---|---|---|
| SU(3) | 22.1 | 170.2 | 291.0 | 13.2 |
| SU(2) | 20.0 | 53.2 | 63.1 | 3.2 |

Volume scaling, 2D SU(3), one sub-layer, batch 1, forward with log-determinant and backward (8 reverse passes at every volume):

| L | links | active links | ms forward | ms backward | µs per active link |
|---|---|---|---|---|---|
| 4 | 32 | 8 | 26.9 | 19.8 | 5837.3 |
| 8 | 128 | 32 | 29.9 | 22.9 | 1650.0 |
| 16 | 512 | 128 | 39.3 | 36.6 | 593.5 |
| 32 | 2048 | 512 | 66.7 | 67.8 | 262.7 |

The time per active link falls with volume because fixed per-call overhead is amortised; the number of passes does not grow.

## 5. Spectral shortcut for single-loop kernels

**Proposition 2.** Let $W\in SU(N)$ have eigen-angles $\theta$ ($z_j=e^{i\theta_j}$) and let
$\omega(W)=P_{\mathfrak{su}}\bigl(c_1(t)W+c_2(t)W^2\bigr)$, $t=\operatorname{tr}W$ (the coefficients may also depend
on fixed parameters). Then $W'=\exp(\omega(W))\,W$ has the eigenvectors of $W$ and the eigen-angles
$$\theta'_j=\theta_j+a_j-\bar a,\qquad a_j=\mathrm{Im}\bigl(c_1z_j+c_2z_j^2\bigr),\qquad \bar a=\tfrac1N\textstyle\sum_k a_k,$$
and with respect to Haar measure
$$\log|\det J|=\log\Bigl|\det\frac{\partial\theta'}{\partial\theta}\Bigr|+\sum_{j<k}\log\frac{\sin^2\bigl((\theta'_j-\theta'_k)/2\bigr)}{\sin^2\bigl((\theta_j-\theta_k)/2\bigr)} .$$

*Proof sketch.* $c_1W+c_2W^2$ is diagonal in the eigenbasis of the unitary $W$, and $P_{\mathfrak{su}}$ keeps it
diagonal there, with eigenvalues $i(a_j-\bar a)$. Haar measure factorises (Weyl) into a part on the eigenvectors,
which the map leaves alone, and the torus part $\prod_{j<k}|z_j-z_k|^2\,d\theta$, with
$|z_j-z_k|^2=4\sin^2\frac{\theta_j-\theta_k}2$. The $N\times N$ matrix $\partial\theta'/\partial\theta$ (all $N$
angles treated as independent) has the determinant of its restriction to $\sum\theta=\text{const}$: it maps that
hyperplane into itself and sends the normal $\mathbf 1$ to $\mathbf 1$ plus a tangent vector. ∎

*On the lattice.* If $\omega_l$ depends on $U_l$ only through one loop $W=U_lS$ with $S$ frozen, then
$W'=\exp(\omega(W))W$, and right translation by $S$ preserves Haar measure, so $\log|\det J_l|$ is Proposition 2 at
$W$. The coefficients may use loops containing no active link as context. They must not read another loop
through $l$: the negative control in §9 shows the shortcut is then wrong by 0.18–0.38. In 2D with $M(\mu,p)$
every plaquette contains an active link, so such a kernel sees only its own plaquette (plus larger frozen loops).
Cost: eigen-angles of $W$, $N$ angle-derivative passes, no $\mathrm{dexp}^L$. **[Gradient at degenerate spectra not
treated.]**

## 6. Inverse

For an active link, $X=\exp(-v(X,U_F))\,Y$, where $Y$ is the output link; the frozen links are unchanged, so the
staples are known from the output.

**Proposition 3.** On a compact connected group with $\mathrm{ad}$ skew in the chosen basis (MATH_SPEC §3.5), if
$q=\sup\lVert Dv\rVert_2<1$ the map $X\mapsto\exp(-v(X,U_F))\,Y$ is a contraction with constant $q$ in the
bi-invariant metric, and the fixed-point iteration converges from any start, the error decreasing like $q^k$.

*Proof sketch.* In right-trivialised coordinates its derivative is $-\mathrm{dexp}^R_{-v}\,Dv$ up to an isometry,
and $\lVert\mathrm{dexp}^R\rVert\le1$ (eigenvalues of modulus $|\mathrm{sinc}(\vartheta/2)|$), so its norm is at most
$q$. Integrating along minimising geodesics turns this into a Lipschitz bound; the group is complete, so Banach's
theorem applies. ∎

Measured on a 4×4 lattice: $q$ = 0.30 (SU(2)) and 0.34 (SU(3)). The iteration stopped after 26 and 26 iterations (successive iterates within
$10^{-14}$), faster than the bound $q^k$, which uses the largest $\lVert Dv\rVert_2$ over all links; it recovered the
input links to 2.9e-15. Sub-layers are inverted in reverse order.

## 7. Exact reference: 2D lattice gauge theory on a torus

Character expansion $e^{\frac\beta N\mathrm{Re}\operatorname{tr}U}=\sum_r d_r\,a_r\,\chi_r(U)$,
$a_r=\frac1{d_r}\int dU\,\overline{\chi_r(U)}\,e^{\frac\beta N\mathrm{Re}\operatorname{tr}U}$. Integrating out the links
of a closed surface with $V$ plaquettes gives $Z=\sum_r d_r^{\chi}a_r^V$ with $\chi$ the Euler characteristic (Migdal);
for the torus $\chi=0$:
$$\boxed{\ Z_{\rm torus}(\beta,V)=\sum_r a_r(\beta)^V\ }\qquad
\langle\mathrm{Re}\operatorname{tr}P\rangle/N=\frac1V\,\partial_\beta\log Z,\qquad
\mathrm{ESS}_{\rm Haar}=\frac{Z(\beta)^2}{Z(2\beta)} .$$
- SU(2): $a_k=2I_{k+1}(\beta)/\beta$. The numerical Weyl integration of `exact_logZ_2d` agrees to 3.8e-14
  (`exact_bessel_checks.json`).
- SU(3): $a_{(p,q)}$ by Weyl integration on the maximal torus. The single-plaquette integral
  $a_0=\sum_{q\in\mathbb Z}\det\bigl[I_{i-j+q}(\beta/3)\bigr]_{i,j=1}^3$ agrees to 2.2e-15 (`exact_bessel_checks.json`).
- Monte Carlo on 1×1 and 2×2 tori agrees within 1.7 standard errors; on the 1×1 torus, dropping the non-trivial
  representations would be excluded by more than 400 standard errors (§9).

## 8. End-to-end validation: flows for 2D SU(2) and SU(3) on a torus

**Model.** Each sub-layer uses masks $M(\mu,p)$ and the loop head of MATH_SPEC §4.3 on the two plaquettes through the
link, $\omega=P_{\mathfrak{su}}\bigl(\sum_s c_{1,s}W_s+c_{2,s}W_s^2\bigr)$. The coefficients come from a 4 → 32 → 32 → 8 tanh
MLP of $(\mathrm{Re},\mathrm{Im})\operatorname{tr}W_s/N$ for both loops, bounded by $0.3\tanh$, with a zero-initialised output
layer, so the untrained flow is the identity. Three sweeps make 12 sub-layers and 17,760 parameters. The log-determinant is
Proposition 1 with batched $Dv$; $\mathrm{dexp}^L$ is the closed form for SU(2) and the block exponential for SU(3).
Training minimises the reverse KL divergence on L = 4 only. Evaluation uses fresh Haar samples at L = 4, 8 and 16 with
the same parameters: the model is local and translation-invariant, so it applies to any even L without retraining.
Two independent evaluation sets were drawn: set 1 at the end of training, set 2 from the saved model (`eval_saved.py`,
which also gives self-normalised importance-sampling error bars for reweighted observables).

**What this tests.** The weights $w=e^{-S}/q$ give an unbiased estimate of $Z$ only if $\log q$, i.e. the total
log-determinant, is exact. A missing cross term would bias $\log Z$ by an amount that grows with the volume. The
estimates agree with the exact values at every volume, up to 16 times the training volume (256 plaquettes, 512 links):
the largest deviation over both groups, three volumes and two sample sets is 1.94 standard errors. The reverse KL,
$\log Z_{\rm exact}-\mathbb E_q[\log w]$, must be non-negative when the log-determinant is right; it is
positive in all 12 evaluations (smallest 0.046). Reweighted plaquette averages agree with the exact values within 2.1 standard errors, while the
unweighted flow samples are biased by up to 38 standard errors, so the reweighting is doing real work.

**SU(3), β = 3.** Trained at L = 4 for 300 Adam steps (batch 256, learning rate 0.005, cosine decay), 1705 s on 6 CPU threads; final training batch: KL 0.062, ESS 91.2 %.

| L | plaquettes | samples (set 1 / set 2) | Haar-prior ESS (exact) | flow ESS (set 1 / set 2) | predicted $e^{-\mathrm{Var}\log w}$ | Var log w / plaquette | KL / plaquette | log Z set 1 | log Z set 2 | log Z exact | z (set 1 / set 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 16 | 16384 / 32768 | 2.6e-05 | 90.6 % / 90.9 % | 91.1 % | 0.00581 | 0.00286 | 4.6171 ± 0.0025 | 4.6180 ± 0.0018 | 4.6170 | +0.03 / +0.57 |
| 8 | 64 | 8192 / 16384 | 4.9e-19 | 69.2 % / 68.5 % | 68.8 % | 0.00584 | 0.00288 | 18.4734 ± 0.0074 | 18.4717 ± 0.0053 | 18.4681 | +0.72 / +0.68 |
| 16 | 256 | 2048 / 4096 | 5.7e-74 | 26.2 % / 21.4 % | 23.1 % | 0.00573 | 0.00298 | 73.8385 ± 0.0371 | 73.8506 ± 0.0299 | 73.8724 | -0.91 / -0.73 |

| L | ⟨Re tr P⟩/N exact | unweighted flow samples (z) | reweighted, SNIS error (z) | min sign det J (both sets) | max ‖Dω‖₂ |
|---|---|---|---|---|---|
| 4 | 0.20308 | 0.19701 ± 0.00037 (-16.5) | 0.20258 ± 0.00042 (-1.2) | +1 | 0.137 |
| 8 | 0.20308 | 0.19709 ± 0.00026 (-23.3) | 0.20238 ± 0.00033 (-2.1) | +1 | 0.141 |
| 16 | 0.20308 | 0.19759 ± 0.00026 (-21.3) | 0.20310 ± 0.00063 (+0.0) | +1 | 0.142 |

**SU(2), β = 2.** Trained at L = 4 for 400 Adam steps (batch 256, learning rate 0.005, cosine decay), 833 s on 6 CPU threads; final training batch: KL 0.169, ESS 70.9 %.

| L | plaquettes | samples (set 1 / set 2) | Haar-prior ESS (exact) | flow ESS (set 1 / set 2) | predicted $e^{-\mathrm{Var}\log w}$ | Var log w / plaquette | KL / plaquette | log Z set 1 | log Z set 2 | log Z exact | z (set 1 / set 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 16 | 16384 / 32768 | 2.7e-05 | 77.4 % / 76.3 % | 75.5 % | 0.01760 | 0.00828 | 7.4257 ± 0.0042 | 7.4321 ± 0.0031 | 7.4262 | -0.11 / +1.94 |
| 8 | 64 | 8192 / 16384 | 5.5e-19 | 36.3 % / 35.4 % | 33.3 % | 0.01717 | 0.00864 | 29.7128 ± 0.0146 | 29.6923 ± 0.0106 | 29.7046 | +0.56 / -1.17 |
| 16 | 256 | 2048 / 4096 | 9.4e-74 | 1.3 % / 3.0 % | 1.0 % | 0.01781 | 0.00854 | 118.8232 ± 0.1955 | 118.8338 ± 0.0893 | 118.8184 | +0.02 / +0.17 |

| L | ⟨Re tr P⟩/N exact | unweighted flow samples (z) | reweighted, SNIS error (z) | min sign det J (both sets) | max ‖Dω‖₂ |
|---|---|---|---|---|---|
| 4 | 0.43313 | 0.42007 ± 0.00052 (-25.0) | 0.43337 ± 0.00070 (+0.4) | +1 | 0.488 |
| 8 | 0.43313 | 0.41920 ± 0.00037 (-38.0) | 0.43196 ± 0.00070 (-1.7) | +1 | 0.488 |
| 16 | 0.43313 | 0.41942 ± 0.00037 (-36.8) | 0.43131 ± 0.00190 (-1.0) | +1 | 0.488 |


**Volume scaling.** The variance of $\log w$ per plaquette is constant across volumes (mean 0.0058 for SU(3) and
0.0175 for SU(2)), the KL per plaquette is half of it, and the effective sample size follows
$\mathrm{ESS}\approx e^{-\operatorname{Var}\log w}=e^{-V\sigma^2}$ (the "predicted" column), as for near-Gaussian
log-weights; at an ESS of a few percent (SU(2), L = 16) the estimate of the ESS itself is dominated by a few weights
and is correspondingly noisy. Keeping ESS ≥ 1/e at $V$ plaquettes therefore needs $\sigma^2\le1/V$: for a $16^4$ lattice (393,216
plaquettes) that is $2.5\times10^{-6}$. This is the volume-scaling problem of flow sampling in general. It concerns the
model and its training, not the log-determinant, which stays exact at every volume tested.

**Acceptance checks for `examples/lattice_flow_2d.py`** (from these single-seed runs; confirm on a second seed before
pinning thresholds): at the training volume ESS ≥ 0.8 (SU(3), β = 3) and ≥ 0.65 (SU(2), β = 2); at every evaluated
volume $|z|\le3$ for $\log Z$ and for the reweighted plaquette, reverse KL ≥ 0 within noise, all $\det J>0$, and
$\max\lVert D\omega\rVert_2<1$.


## 9. Verification results

From `verify_lattice_results.json`, produced by `python verify_lattice.py` in a clean process (float64, CPU,
torch 2.14.1+cpu). The file holds the 64 results (the rows below and the volume table of §4) and an
`environment` record with the torch version, run time and failures. "NEGATIVE CONTROL" rows must be large; all
other error rows must be small.

A recorded value that is NaN fails its check, and the script exits with status 1 if any check fails. The following
are recorded without a tolerance and do not affect the exit status:
- "full sweep changes every link";
- the inverse iteration counts with max ‖Dω‖₂;
- the Monte Carlo triples of the exact reference, and the trivial-representation-only log Z of the 2×2 tori (the 1×1
  exclusions are checked);
- the volume table of §4.

Coverage. Brute-force Jacobians:
- D = 2 (L = 4) tests all four sub-layers $(\mu,p)$ for both groups, plus the full sweep and the every-μ-link
  negative control.
- D = 3 (L = 4) and D = 4 (L = 2) test two sub-layers per group.
- D = 4 runs only at L = 2, where $x+\hat\nu$ and $x-\hat\nu$ coincide.
- The block checks compare $\log|\det J|$, not the sign of $\det J$.
- The odd-L control is SU(2), D = 2, L = 3. It draws its field from the global random stream after the cases above,
  so its value (0.11 here) shifts when cases are added before it; only its lower bound is asserted.

Gauge covariance runs for SU(3) at D = 2 and 4 and for SU(2) at D = 3. `validate_mask` runs for SU(3) at D = 2 and
for SU(2) at D = 3.

| Check | Measured |
|---|---|
| D=2 L=4 SU(2): \|logdet brute force - masked formula\| over 4 sub-layers | 2.7e-15 |
| D=2 L=4 SU(2): off-diagonal blocks of J_AA | 0 |
| D=2 L=4 SU(2): frozen rows of J differ from identity | 5.6e-16 |
| D=2 L=4 SU(2) NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero | 0.49259 |
| D=2 L=4 SU(2) NEGATIVE CONTROL: \|logdet brute force - sum of per-link dets\| is non-zero | 2.8947 |
| D=2 L=4 SU(2): full sweep (4 sub-layers) \|logdet brute force - sum of masked formulas\| | 1.1e-14 |
| D=2 L=4 SU(2): full sweep changes every link | True |
| D=2 L=4 SU(3): \|logdet brute force - masked formula\| over 4 sub-layers | 1.2e-14 |
| D=2 L=4 SU(3): off-diagonal blocks of J_AA | 0 |
| D=2 L=4 SU(3): frozen rows of J differ from identity | 1.1e-15 |
| D=2 L=4 SU(3) NEGATIVE CONTROL (all mu-links active): off-diagonal blocks are non-zero | 0.44988 |
| D=2 L=4 SU(3) NEGATIVE CONTROL: \|logdet brute force - sum of per-link dets\| is non-zero | 1.3788 |
| D=2 L=4 SU(3): full sweep (4 sub-layers) \|logdet brute force - sum of masked formulas\| | 1.6e-14 |
| D=2 L=4 SU(3): full sweep changes every link | True |
| D=3 L=4 SU(2): \|logdet brute force - masked formula\| over 2 sub-layers | 3.6e-14 |
| D=3 L=4 SU(2): off-diagonal blocks of J_AA | 0 |
| D=3 L=4 SU(2): frozen rows of J differ from identity | 7.8e-16 |
| D=4 L=2 SU(2): \|logdet brute force - masked formula\| over 2 sub-layers | 1.8e-14 |
| D=4 L=2 SU(2): off-diagonal blocks of J_AA | 0 |
| D=4 L=2 SU(2): frozen rows of J differ from identity | 6.7e-16 |
| D=3 L=4 SU(3): \|logdet brute force - masked formula\| over 2 sub-layers | 1.1e-13 |
| D=3 L=4 SU(3): off-diagonal blocks of J_AA | 0 |
| D=3 L=4 SU(3): frozen rows of J differ from identity | 1.1e-15 |
| D=4 L=2 SU(3): \|logdet brute force - masked formula\| over 2 sub-layers | 2.8e-14 |
| D=4 L=2 SU(3): off-diagonal blocks of J_AA | 0 |
| D=4 L=2 SU(3): frozen rows of J differ from identity | 8.9e-16 |
| D=2 L=3 SU(2) NEGATIVE CONTROL (odd L): \|logdet brute force - masked formula\| is non-zero | 0.10945 |
| validate_mask D=2 L=4 SU(3) M(0,0): change of other active connections (valid mask, must be 0) | 0 |
| validate_mask D=3 L=4 SU(2) M(1,1): change of other active connections (valid mask, must be 0) | 0 |
| validate_mask NEGATIVE CONTROL (every mu-link active): change detected | 0.0013237 |
| validate_mask NEGATIVE CONTROL (odd L = 3): change detected | 7.0e-04 |
| D=2 L=4 SU(3): gauge covariance of the layer, gauge invariance of log det | 1.1e-15, 1.5e-14 |
| D=2 L=4 SU(3): output stays in SU(N) (unitarity, det) | 1.1e-15, 1.1e-15 |
| D=3 L=4 SU(2): gauge covariance of the layer, gauge invariance of log det | 1.3e-15, 1.2e-13 |
| D=3 L=4 SU(2): output stays in SU(N) (unitarity, det) | 1.1e-15, 9.0e-16 |
| D=4 L=2 SU(3): gauge covariance of the layer, gauge invariance of log det | 1.8e-15, 2.1e-14 |
| D=4 L=2 SU(3): output stays in SU(N) (unitarity, det) | 1.3e-15, 1.2e-15 |
| SU(2) single loop: \|general d x d formula - spectral formula\| (300 Haar samples) | 4.9e-15 |
| SU(3) single loop: \|general d x d formula - spectral formula\| (300 Haar samples) | 1.4e-14 |
| SU(2) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong | 0.17677 |
| SU(2) 2D lattice, single-plaquette class-function kernel: \|masked formula - sum of spectral formulas\| | 1.8e-15 |
| SU(3) NEGATIVE CONTROL: spectral shortcut with another loop through the link as 'context' is wrong | 0.37518 |
| SU(3) 2D lattice, single-plaquette class-function kernel: \|masked formula - sum of spectral formulas\| | 6.2e-15 |
| SU(2) fixed-point inverse: \|recovered - original active links\| | 9.0e-16 |
| SU(2) fixed-point inverse: iterations, max \|\|D omega\|\|_2 | 26, 0.29789 |
| SU(3) fixed-point inverse: \|recovered - original active links\| | 2.9e-15 |
| SU(3) fixed-point inverse: iterations, max \|\|D omega\|\|_2 | 26, 0.33949 |
| SU(3): log c_0(beta=6) from exact_logZ_2d(V -> large)/V vs single-variable log Z 1.235890 | 3.1e-12 |
| SU(2): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16 | 0 |
| SU(2) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact | 1.0887, 0.0035458, 1.0907 |
| SU(2) 2x2 torus beta=1.5: \|log Z Monte Carlo - exact character expansion\| in stderr units | 0.55764 |
| SU(2) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison) | 1.0767 |
| SU(3): exact log Z converged in representation cutoff (pmax 8 vs 5), beta=6 V=16 | 0 |
| SU(3) 2x2 torus beta=1.5: log Z Monte Carlo, stderr, exact | 0.26948, 0.0014623, 0.27062 |
| SU(3) 2x2 torus beta=1.5: \|log Z Monte Carlo - exact character expansion\| in stderr units | 0.781 |
| SU(3) 2x2 torus beta=1.5: trivial-representation-only log Z (for comparison) | 0.27047 |
| SU(2) 1x1 torus beta=3.0: log Z Monte Carlo, stderr, exact | 1.6213, 0.0010872, 1.623 |
| SU(2) 1x1 torus beta=3.0: \|log Z Monte Carlo - exact\| in stderr units | 1.5448 |
| SU(2) 1x1 torus beta=3.0: trivial-representation-only value excluded, in stderr units | 601.47 |
| SU(3) 1x1 torus beta=4.0: log Z Monte Carlo, stderr, exact | 1.1208, 0.0014804, 1.1233 |
| SU(3) 1x1 torus beta=4.0: \|log Z Monte Carlo - exact\| in stderr units | 1.6719 |
| SU(3) 1x1 torus beta=4.0: trivial-representation-only value excluded, in stderr units | 400.82 |
| cost per active link: ratio L=32 / L=8 (linear scaling means <= about 1) | 0.15921 |

## 10. Open items

1. Closed-form $\mathrm{dexp}^L$ for su(3): 42 % of the SU(3) sub-layer cost (MATH_SPEC open item 1).
2. Masks with fewer than $2D$ sub-layers for D > 2 (lower bound 4), and masks for larger loops.
3. Spectral shortcut: stable eigen-angle gradients at degenerate spectra, and a layer family built on it.
4. Volume scaling of the effective sample size (§8). It is a property of the model and its training, not of the
   log-determinant, which stays exact at every volume.
5. Four-dimensional flows (training, timings, lattices beyond L = 2), fermions, float32 and GPU timings: not
   measured. D = 4 enters only the L = 2 Jacobian and gauge checks of §9.

## 11. Build order for `lag.lattice`

1. `lag/lattice/field.py`: field layout, `haar`, `shift`, `staples`, `plaquettes`, `wilson_action`, `gauge_transform`.
2. `lag/lattice/masks.py`: `mask(L, D, mu, p)` with an even-L assertion; `validate_mask` (§3.3).
3. `lag/lattice/layers.py`: `masked_layer` with batched $Dv$ and a `create_graph` flag; the inverse of §6.
4. `lag/lattice/spectral.py`: `logdet_spectral` (§5).
5. `lag/lattice/exact.py`: `exact_logZ_2d` and the Bessel forms of §7.
6. `examples/lattice_flow_2d.py`: §8, with its acceptance checks.
7. `tests/test_lattice.py`: port of `verify_lattice.py`, seeds and tolerances pinned.

## References

Cited from memory; check before publication. A. A. Migdal, *Recursion equations in gauge field theories* (1975).
C. Morningstar, M. Peardon, *Analytic smearing of SU(3) link variables in lattice QCD* (2004).
M. S. Albergo, G. Kanwar, P. E. Shanahan, *Flow-based generative models for Markov chain Monte Carlo in lattice
field theory* (2019). G. Kanwar et al., *Equivariant flow-based sampling for lattice gauge theory* (2020).
D. Boyda et al., *Sampling using SU(N) gauge equivariant flows* (2021).
