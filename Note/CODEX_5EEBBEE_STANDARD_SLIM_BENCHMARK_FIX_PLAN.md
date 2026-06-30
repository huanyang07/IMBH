# Codex Next-Step Brief: Repair the Standard No-Wind Slim-Disk Thin-Limit Benchmark

Repository reviewed:

```text
https://github.com/huanyang07/IMBH
commit: 5eebbee
```

Start from:

```text
Note/GPT_REVIEW_PROMPT_STANDARD_SLIM_BENCHMARK.md
Note/CODEX_STANDARD_NO_WIND_SLIM_DISK_BENCHMARK_PLAN.md
scripts/run_standard_slim_benchmark_thin_limit.py
outputs/tables/slim_benchmark_thin_limit.md
outputs/tables/slim_benchmark_thin_limit.json
outputs/figures/slim_benchmark_thin_limit_profiles.png
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py
```

## Executive assessment

The standard single-BH no-wind thin-limit benchmark does **not** pass yet.

However, the current failure does **not** yet prove that the transonic
equations are physically wrong. The low-\(\dot M\) profiles at
\(\dot M/\dot M_{\rm Edd}=10^{-4}\) and \(10^{-3}\) are qualitatively thin:

```text
H/R is small.
Qadv/Qvisc is tiny.
Omega/OmegaK is close to one at the percent level or below.
The PW nuSigma error is a few percent in the median.
No quick phase-space radial fold is detected.
```

The current benchmark failure is mainly a **solver/continuation/polish
problem**, with a possible hidden formulation issue still to be tested.

The most important current evidence:

```text
Mdot/Edd = 1e-4:
    residual = 3.14e-4
    dominant = interval_R
    sonic = no
    Rson = 44.9 rg
    lambda0/lK_ISCO = 1.323
    H/R max = 2.2e-3
    Qadv/Qvisc = 1.5e-6
    PW nuSigma median error = 2.0e-2

Mdot/Edd = 1e-3:
    residual = 6.36e-5
    dominant = C2
    sonic = no
    Rson = 5.84 rg
    lambda0/lK_ISCO = 0.999
    H/R max = 3.4e-3
    Qadv/Qvisc = 2.6e-6
    PW nuSigma median error = 3.8e-2

Mdot/Edd = 1e-2:
    residual = 0.995
    dominant = outer_energy
    sonic = yes
    outer_thin = no
    Rson = 4.62 rg
    lambda0/lK_ISCO = 0.660
    Omega error = 0.091
```

Interpretation:

```text
1e-3 is close to the desired thin transonic branch but fails sonic polish.
1e-4 is nearly thin in the outer disk but has a bad sonic eigenvalue.
1e-2 has jumped to a bad branch during the R_out extension.
```

The next sprint should not proceed to higher-Mdot slim sequences or IMRI wind.
It should first make the low-Mdot benchmark pass in a controlled way.

---

# 1. Do not treat this as a physical slim-disk failure yet

The current benchmark was intentionally low resolution:

```text
N = 24
R_out ladder = 300 -> 1000 -> 10000 rg
max_nfev = 700
outer_closure = thin_value
interval_residual_form = differential
stress_factor = 1.0
```

The current run also extends \(R_{\rm out}\) by large jumps while remapping
profiles that have not actually converged. That is enough to explain much of
the current behavior.

The \(10^{-2}\) case in particular should be treated as a continuation failure,
not a physics result. The profile is visibly not a proper thin disk:

```text
outer_energy residual ~ 1
Omega/OmegaK error ~ 9%
lambda0/lK_ISCO ~ 0.66
Rson ~ 4.6 rg
```

That is not a valid low-Mdot slim solution.

---

# 2. Make the primary target \(10^{-3}\), not \(10^{-4}\)

At very low accretion rate, the transonic eigenproblem becomes stiff because
the radial velocity and advective terms are extremely small. The \(10^{-4}\)
case currently finds an unphysical sonic point around:

```text
Rson ~ 45 rg
lambda0/lK_ISCO ~ 1.32
```

That is not useful as the first transonic benchmark target.

Use the following order:

```text
Primary target:
    Mdot/Mdot_Edd = 1e-3

Then:
    3e-4
    1e-4

Then:
    3e-3
    1e-2
```

The \(10^{-3}\) case already has the right sonic radius and angular momentum
eigenvalue. It is the best first root to polish.

---

# 3. First isolate the problem at fixed \(R_{\rm out}\)

Do not start by jumping to \(R_{\rm out}=10^4r_g\).

For each \(\dot M\), solve first at:

```text
R_out = 300 rg
R_out = 1000 rg
```

and only after those pass, extend to:

```text
R_out = 2000, 3000, 5000, 7000, 10000 rg
```

Use a logarithmic ladder:

```text
300, 500, 800, 1200, 2000, 3000, 5000, 7000, 10000
```

Do not use:

```text
300 -> 1000 -> 10000
```

for production validation.

At each \(R_{\rm out}\), require:

```text
physical residual < 1e-6 to 1e-5
sonic residuals controlled
outer residuals controlled
profile smooth
```

before using that solution as the seed for the next \(R_{\rm out}\).

---

# 4. Increase resolution before judging the equations

The current N=24 grid is too coarse for a transonic benchmark extending from
\(R_{\rm son}\sim6r_g\) to \(R_{\rm out}=10^4r_g\).

Use a resolution tied to logarithmic radial span:

```python
N = max(64, ceil(16 * log(R_out/R_son)))
```

or, more simply for the benchmark:

```text
R_out = 300 rg:    N = 64
R_out = 1000 rg:   N = 80
R_out = 3000 rg:   N = 96
R_out = 10000 rg:  N = 128
```

The current oscillatory signs in \(\Omega/\Omega_K-1\) and `nuSigma` are
consistent with under-resolved collocation plus incomplete polish.

---

# 5. Build a true analytic thin-disk seed

The current script seeds the transonic solve from the reduced nearly-Keplerian
solver. That is useful, but for a standard thin-disk benchmark the best seed is
the analytic thin-disk solution itself.

For each radius, set:

```math
\lambda_0 = l_K(R_{\rm ISCO})/(r_gc)
```

initially.

Then define:

```math
W_{\rm req}(R)
=
\frac{\dot M [l_K(R)-l_0]}{2\pi R^2}.
```

Use the thin Keplerian shear:

```math
Q_{\rm visc,thin}
=
-W_{\rm req}\,\Omega_K\,\frac{d\ln\Omega_K}{d\ln R}.
```

Find \(\Sigma,T\) from:

```math
W_{\rm model}(\Sigma,T)=W_{\rm req},
```

```math
Q_{\rm rad}(\Sigma,T)=Q_{\rm visc,thin}.
```

Then compute:

```math
u=\frac{\dot M}{2\pi R\Sigma}.
```

This gives an analytic thin seed satisfying:

```text
angular momentum balance,
thermal balance,
continuity,
outer thin boundary,
```

before the transonic solve starts.

Use this seed first for:

```text
Mdot/Edd = 1e-3
R_out = 300 and 1000 rg
N = 64--80
```

Then compare to the reduced-solver seed.

---

# 6. Use a square sonic polish, not the current overdetermined sonic block

The current collocation residual includes:

```text
D, C1, C2
```

all at once. That overdetermines the sonic point by one row. The current
\(10^{-3}\) result is dominated by `C2`, which is exactly the kind of failure
that can occur when the sonic block is overdetermined and not polished.

For fixed-\(\dot M\) polishing, use:

```text
D = 0
C_pivot = 0
```

as the production sonic system, while auditing:

```text
unused C
K
smin/smax
```

Run both pivots:

```text
pivot = C1
pivot = C2
```

At the true solution they should converge to the same profile and make the
unused compatibility small.

Create:

```text
outputs/tables/slim_benchmark_sonic_pivot_polish.md
```

Columns:

```text
Mdot/Edd
R_out
N
pivot
physical residual
D
C1
C2
K
Rson
lambda0/lK_ISCO
dominant
optimizer_success
nfev
```

Acceptance for \(10^{-3}\):

```text
physical residual <= 1e-6--1e-5
D,C1,C2,K all <= few x 1e-6--1e-5
Rson ~= 5.8--6.1 rg
lambda0/lK_ISCO ~= 1
```

---

# 7. Use integrated defects for solving, differential residuals for auditing

The differential interval residual has \(1/\Delta x\) amplification and can be
badly conditioned over large radial spans.

For the solve, use:

```text
interval_residual_form = integrated
integrated_residual_weighting = none or inverse_sqrt_dx
```

For the physical audit, continue reporting the differential residual.

Recommended sequence:

```text
integrated solve -> differential polish -> physical audit.
```

Do not judge a solution only by the integrated residual.

---

# 8. Use block Jacobian and larger max_nfev

The current benchmark uses:

```text
use_stage_block_jacobian = False
MAX_NFEV_STAGE = 450
MAX_NFEV_FINAL = 700
```

For the next benchmark sprint use:

```text
use_stage_block_jacobian = True
MAX_NFEV_STAGE = 2000
MAX_NFEV_FINAL = 4000
```

For this validation benchmark, computational cost is less important than
determining whether the transonic equations are correct.

Also record:

```text
optimizer_success
optimality
cost
line-search reductions if available
condition estimates
```

A row with residual below tolerance but `max_nfev` should be considered
numerically acceptable only if all physics diagnostics pass.

---

# 9. Add residual localization

Before making more changes, output residual profiles for the current failed
cases.

Create:

```text
outputs/tables/slim_benchmark_residual_profile_mdot1e-3.md
outputs/tables/slim_benchmark_residual_profile_mdot1e-2.md
outputs/figures/slim_benchmark_residual_profiles.png
```

For every interval/node report:

```text
R/rg
interval_R
interval_E
outer residual contribution if endpoint
D,C1,C2,K if sonic node
Omega/OmegaK-1
Qadv/Qvisc
H/R
PW nuSigma error
qbalance
condition(A)
smin/smax(A)
```

This will distinguish:

```text
sonic-localized failure,
outer-boundary failure,
interior radial-momentum failure,
global under-resolution.
```

The current table only gives the dominant block, which is not enough.

---

# 10. Run targeted formulation checks

## 10.1 Radial momentum sign / pressure-gradient sign

In the thin disk, pressure support should give:

```math
\frac{\Omega^2-\Omega_K^2}{\Omega_K^2}
\sim
\left(\frac{H}{R}\right)^2
\frac{d\ln \Pi}{d\ln R}.
```

Since \(H/R\sim10^{-3}\) to \(10^{-2}\), the expected pressure correction is
tiny:

```text
~1e-6 to 1e-4
```

Current \(\Omega/\Omega_K-1\) errors are:

```text
~4e-3 for 1e-4 and 1e-3
~9e-2 for 1e-2
```

So add a diagnostic comparing:

```text
measured Omega/OmegaK - 1
expected pressure-support correction
radial inertia correction
```

If measured deviations exceed expected corrections by orders of magnitude, the
issue is numerical or a sign/normalization problem.

## 10.2 Stress normalization

Run:

```text
stress_factor = 1.0
stress_factor = 1.5
```

Why:

```text
The default transonic class has stress_factor=1.5, but the benchmark script
uses 1.0. The thin-disk normalization and alpha-Pi convention must be made
explicit.
```

Compare:

```text
Rson
lambda0
PW nuSigma error
Q balance
residual
```

Do not proceed until one convention is chosen and documented.

## 10.3 One-face/two-face cooling convention

Make a static analytic thin-disk residual audit.

Using the analytic seed, compute:

```text
Qvisc_twoface
Qrad_twoface
Qvisc_oneface
Qrad_oneface
```

Confirm that the code's actual `Q_visc` and `Q_rad` are in the same convention.

Expected:

```text
Qvisc_twoface = Qrad_twoface
```

to numerical precision for the analytic seed.

## 10.4 Angular-momentum normalization

For any candidate profile compute:

```math
\mathcal R_J
=
\frac{2\pi R^2 W}{\dot M(l-l_0)}-1.
```

This should be zero because angular momentum is algebraic in the current
formulation. If not, the diagnostic construction is inconsistent.

## 10.5 Entropy/advection sign

At low Mdot, \(Q_{\rm adv}\) should be negligible. Its sign is not important
for the thin-limit acceptance, but its magnitude should remain tiny:

```text
|Qadv/Qvisc| < 1e-4 to 1e-3
```

The current values pass this check for 1e-4 and 1e-3.

---

# 11. Rebuild the benchmark order

## Stage A: analytic-seed residual audit only

No optimizer.

For:

```text
Mdot/Edd = 1e-3
R_out = 300, 1000, 3000, 10000
N = 64--128
```

evaluate the analytic thin seed residuals.

Output:

```text
outputs/tables/slim_benchmark_analytic_seed_residual_audit.md
```

If analytic seed already has large residual in the code's radial/energy
equations, the formulation or convention is inconsistent.

## Stage B: fixed-eigenvalue profile solve

Hold:

```text
Rson = 5.9 rg
lambda0 = lK_ISCO
```

Remove or strongly downweight sonic rows.

Solve profile only.

Goal:

```text
interval and outer residuals < 1e-6--1e-5
Omega/OmegaK close to expected pressure correction
Qvisc ~ Qrad
```

This isolates bulk equations from sonic regularity.

## Stage C: free Rson, fixed lambda0

Free \(R_{\rm son}\), keep \(\lambda_0=l_K(R_{\rm ISCO})\).

Goal:

```text
Rson stays near ~6 rg
bulk residual remains small
```

## Stage D: free Rson and lambda0 with square sonic pivot

Use:

```text
D + C_pivot
```

as sonic production rows.

Goal:

```text
D,C1,C2,K all small
lambda0/lK_ISCO near 1
Rson near ~6 rg
```

## Stage E: extend R_out gradually

Only after Stage D passes at \(R_{\rm out}=300\) and \(1000\):

```text
R_out = 1500, 2000, 3000, 5000, 7000, 10000
```

Use enough nodes and do not seed from unpolished solutions.

## Stage F: move to Mdot=1e-4 and 1e-2

After \(10^{-3}\) passes, use it as the anchor.

For \(10^{-4}\), continue downward in:

```text
1e-3 -> 7e-4 -> 5e-4 -> 3e-4 -> 1e-4
```

For \(10^{-2}\), continue upward in:

```text
1e-3 -> 2e-3 -> 3e-3 -> 5e-3 -> 7e-3 -> 1e-2
```

Do not jump by a factor of ten.

---

# 12. Acceptance criteria for the repaired thin-limit benchmark

For \(10^{-3}\):

```text
accepted = yes
physical = yes
equations = yes
sonic = yes
outer_thin = yes
residual <= 1e-6--1e-5
Rson = 5.5--6.5 rg
lambda0/lK_ISCO = 1 +/- 0.01
Omega error <= 1e-3 if using high resolution, <= few e-3 for smoke
|Qadv/Qvisc| <= 1e-4
H/R <= 0.01
PW nuSigma median error <= 1e-2--few e-2
Q balance median <= 1e-2
no blocking R fold
```

For \(10^{-4}\):

```text
allow somewhat weaker sonic tolerance initially
but require Rson not to run to tens of rg
and lambda0/lK_ISCO near 1
```

For \(10^{-2}\):

```text
must not have outer_energy ~ 1
must not have lambda0/lK_ISCO ~ 0.66
must not have Omega error ~ 0.09
```

---

# 13. Interpreting possible outcomes

## Outcome 1: analytic seed fails the residual audit

Then the equations/conventions are inconsistent. Check:

```text
two-face vs one-face cooling,
stress normalization,
radial momentum sign,
entropy sign,
PW derivative,
vertical Pi definition.
```

## Outcome 2: analytic seed passes, fixed-eigen profile solve fails

Then the optimizer/residual scaling is the issue.

Use integrated defects, block Jacobian, and larger max_nfev.

## Outcome 3: fixed-eigen solve passes but free sonic solve fails

Then the sonic regularity implementation is the issue.

Use square sonic pivot and dedicated sonic polish.

## Outcome 4: fixed R_out passes but R_out extension fails

Then outer continuation/remapping is the issue.

Use finer R_out ladder and append outer nodes instead of stretching the whole
grid.

## Outcome 5: all low-Mdot passes but higher Mdot folds

Then the solver is probably correct, and the fold in the IMRI problem becomes
physically meaningful.

---

# 14. Concrete scripts to add

```text
scripts/run_standard_slim_analytic_seed_audit.py
scripts/run_standard_slim_fixed_eigen_profile.py
scripts/run_standard_slim_sonic_pivot_polish.py
scripts/run_standard_slim_rout_continuation.py
scripts/run_standard_slim_mdot_ladder.py
scripts/run_standard_slim_formulation_checks.py
```

Outputs:

```text
outputs/tables/slim_benchmark_analytic_seed_residual_audit.md
outputs/tables/slim_benchmark_fixed_eigen_profile.md
outputs/tables/slim_benchmark_sonic_pivot_polish.md
outputs/tables/slim_benchmark_rout_continuation.md
outputs/tables/slim_benchmark_mdot_ladder.md
outputs/tables/slim_benchmark_formulation_checks.md
outputs/tables/slim_benchmark_residual_profile.md
```

---

# 15. Compact Codex prompt

```text
The first standard single-BH no-wind thin benchmark at commit 5eebbee does not
pass, but the 1e-4 and 1e-3 rows are qualitatively thin. The 1e-3 row is close:
Rson=5.84 rg, lambda0/lK_ISCO=0.999, H/R=0.0034, Qadv/Qvisc=2.6e-6, but it
fails sonic C2 at residual 6.4e-5. The 1e-2 row is a bad continuation branch:
outer_energy~1, Omega error~0.09, lambda0/lK_ISCO=0.66.

Next sprint:
1. Make Mdot/Edd=1e-3 the primary target.
2. Build an analytic thin-disk seed from W_req and Qvisc=Qrad.
3. Audit the analytic seed residual before optimization.
4. Solve at fixed R_out=300 and 1000 first, with N=64--80.
5. Use square sonic polish D+C_pivot, pivots C1 and C2 separately.
6. Use integrated defects for solving and differential residuals for auditing.
7. Increase max_nfev and enable block Jacobian.
8. Add residual localization profiles.
9. Only after 1e-3 passes, continue down to 1e-4 and up to 1e-2 with small
   mdot steps and small R_out steps.
10. Do not move to high Mdot or wind until the thin-limit benchmark passes.
```

---

# 16. Bottom line

The standard no-wind benchmark has not yet shown a fatal physics problem.

It has shown that:

```text
the current full transonic solve is not yet robust enough to recover the
low-Mdot thin-disk limit at production tolerances.
```

Fix the benchmark in this order:

```text
analytic seed -> fixed-eigen bulk solve -> square sonic polish -> gradual
R_out continuation -> gradual Mdot ladder.
```
