# Causal inner continuum-lift and metric-conditioning audit — WP10c9d6c3

Date: 2026-07-29
Analyzed base commit: `da2d7612cc9a2fff7093bee705f3f5fbe2d2101d`
Analyzed parent: `c28dd65708ee817fd23d0c619dbb0afd5f991178`

## Binding classification

```text
smooth_continuum_four_level_export_direction_certified
```

WP10c9d6c3 changes no physical or numerical operator. It preserves the
historical WP10c9d6c2 classification

```text
four_level_uniform_asymptotic_direction_rejected
```

but shows that the rejection does not reproduce when the background and
perturbations are defined as one smooth continuum object and projected as
proper-measure finite-volume cell averages on all four grids.

The only authorized next step is prospective held-out **uniform** validation
under the new continuum-lift contract. Direct operator redesign, embedded
discrimination, nonlinear evolution, production promotion, fixed-Q
averaging, and reduced slow-time evolution remain blocked.

## Question

WP10c9d6c2 used two PCHIP continuations of grid-defined common perturbations.
They agreed on the old pass/fail result, but their maximum propagated
difference was

\[
4.48\times10^{-5},
\]

which was slightly larger than the primary N256/N512 instantaneous maximum
difference

\[
3.91\times10^{-5}.
\]

The old continuation gate used only an absolute `0.005` bound and a history
cosine. It recorded, but did not gate, the uncertainty relative to the
fine-grid discretization difference.

WP10c9d6c3 asks:

1. does a genuinely smooth common finite-volume background remain
   admissible and method-certified on N64/N128/N256/N512;
2. do grid-independent analytic perturbations have projection uncertainty
   below ten percent of the fine physical-export difference; and
3. do those profiles pass the unchanged four-level physical-export
   contraction and error-direction gates?

## Continuum construction

The discrete primitive state is treated according to its declared semantics:
a proper-measure finite-volume cell average.

The primary background is a 24-coefficient quintic B-spline in log radius:

- continuity: \(C^4\);
- least-squares fit to the N128 primitive cell averages;
- exact existing inner and outer physical boundary anchors;
- projection by Kerr-Schild proper-measure quadrature.

An independently fitted degree-seven \(C^6\) profile supplies a background
representation check. The construction results are:

| Quantity | Result |
|---|---:|
| quintic scaled N128 cell-fit defect | \(8.76\times10^{-6}\) |
| septic scaled N128 cell-fit defect | \(6.61\times10^{-6}\) |
| maximum boundary-anchor defect | \(8.88\times10^{-16}\) |
| maximum quintic/septic fine-grid difference | \(1.13\times10^{-5}\) |
| N128 reference projection relative defect | \(7.79\times10^{-7}\) |
| maximum reconstruction factor change | \(0\) |

The primary and independent background representations are therefore smooth,
well-conditioned, mutually consistent, and inactive on the admissibility
limiter.

## Analytic perturbations

Two physical perturbations are declared directly as smooth functions of
radius:

1. `calibration_mixed`, containing an inner component and a broader
   \(R\simeq3.05\,r_g\) component;
2. `heldout_near_excision`, centered at \(2.20\,r_g\) with a distinct
   five-field mixture.

Neither is interpolated from a discrete grid. Each is projected independently
onto N64/N128/N256/N512 using order-24 Kerr-Schild cell quadrature. Order-12
projection is propagated as an independent lift check.

The maximum primary/secondary projection differences are:

| Quantity | Calibration | Held-out |
|---|---:|---:|
| initial-state relative difference | \(2.27\times10^{-16}\) | \(5.37\times10^{-16}\) |
| initial-rate relative difference | \(2.02\times10^{-14}\) | \(8.56\times10^{-15}\) |
| history lift/fine-spatial ratio | \(1.36\times10^{-11}\) | \(1.32\times10^{-12}\) |
| cumulative lift/fine-spatial ratio | \(3.86\times10^{-12}\) | \(8.53\times10^{-13}\) |

All are far below the predeclared `0.10` relative uncertainty gate.

## Method gates

Fresh self-consistent monolithic frozen tangents were built on every grid.
All inherited method gates pass.

| Quantity | Maximum/minimum over four grids |
|---|---:|
| maximum stationary directional defect | \(1.80\times10^{-8}\) |
| maximum storage-rate directional defect | \(7.55\times10^{-12}\) |
| maximum export directional defect | \(3.86\times10^{-10}\) |
| maximum restart defect | \(1.86\times10^{-14}\) |
| minimum absolute characteristic speed | \(0.43185\,c\) |
| minimum characteristic spectral gap | \(5.03\times10^{-3}\,c\) |
| incoming excision characteristics | \(0\) |

The reconstruction, descriptor-component, base-rate balance, generator
factorization, center-broken path, characteristic-conditioning, and
production-neutrality gates also pass on all four levels.

## Binding four-level results

The unchanged historical gates are

\[
p_{\rm RMS}\ge0.75,\qquad
p_{\max}\ge0.75,\qquad
\min_a p_a\ge0.75,
\]

\[
d_{\rm fine}\le0.05,\qquad
\cos_{\rm history}\ge0.90,\qquad
\cos_{\rm error}\ge0.90.
\]

### Calibration profile

| Metric | Instantaneous | Cumulative |
|---|---:|---:|
| RMS observed order | \(2.2184\) | \(2.0648\) |
| maximum observed order | \(2.2108\) | \(2.1153\) |
| minimum component order | \(2.0412\) | \(1.8821\) |
| fine maximum physical difference | \(2.51\times10^{-6}\) | \(8.83\times10^{-7}\) |
| history cosine | \(0.999999996\) | \(0.9999999997\) |
| refinement-error cosine | \(0.97834\) | \(0.99034\) |

### Held-out near-excision profile

| Metric | Instantaneous | Cumulative |
|---|---:|---:|
| RMS observed order | \(2.3852\) | \(2.4027\) |
| maximum observed order | \(2.3671\) | \(2.4217\) |
| minimum component order | \(1.9101\) | \(1.9750\) |
| fine maximum physical difference | \(1.93\times10^{-5}\) | \(1.99\times10^{-6}\) |
| history cosine | \(0.999998958\) | \(0.999999973\) |
| refinement-error cosine | \(0.97122\) | \(0.97291\) |

Both analytic profiles pass every unchanged instantaneous and cumulative
gate.

## Conditioned diagnostics

Time-weighted diagnostics agree with the historical pass:

- calibration weighted order/error cosine:
  \(2.2184/0.97837\);
- held-out weighted order/error cosine:
  \(2.3833/0.97112\);
- minimum component \(L^2_t\) orders:
  \(2.0418\) and \(1.9101\);
- minimum component \(L^\infty_t\) orders:
  \(2.0131\) and \(1.9149\).

Peak locations migrate in both profiles, but the maximum errors still
contract at approximately second order. The error-history SVDs are dominated
by one direction:

\[
f_1=0.99838
\]

for the calibration profile and

\[
f_1=0.98340
\]

for the held-out profile. The declared second-mode threshold is not reached.

Fixed-power

\[
F_h=F_*+a h^2,\qquad
F_h=F_*+a h^2+b h^3,\qquad
F_h=F_*+a h^2+b h^4
\]

fits are stable. The maximum \(h^2+h^3\) versus \(h^2+h^4\) continuum
differences are \(3.52\times10^{-7}\) and \(1.02\times10^{-5}\), respectively.

## Interpretation

The prior four-level rejection remains valid for the exact profiles and
continuations used by WP10c9d6c2. It must not be relabeled.

The new result establishes a narrower and more useful fact:

> The unchanged monolithic uniform operator has clean, approximately
> second-order physical-export convergence and a stable error direction for
> two independently declared smooth, proper-measure continuum
> perturbations, including a held-out near-excision profile.

Consequently, the earlier low error cosine is not sufficient evidence for an
operator redesign. It is sensitive to how the background and perturbation
fiber are continued between grids.

This audit changed both the continuum background representation and the
perturbation definition. It therefore does not uniquely attribute the old
failure to the PCHIP background, the discrete common-mode continuation, or
their interaction. It also does not establish convergence for arbitrary
profiles.

## Authorized next package

### WP10c9d6c4 — prospective held-out uniform validation

Freeze the c3 background, projection rules, physical scales, and historical
gates before adding further profiles.

The next package should include:

1. a smooth proper-measure fit to the historical common perturbation as a
   calibration-only profile;
2. at least three additional analytic held-out profiles with distinct
   supports, widths, and five-field mixtures;
3. one broad profile extending toward the outer part of the uniform inner
   domain;
4. one first-cell-dominated but smooth outgoing profile;
5. projection-order uncertainty below ten percent of the fine export
   difference for every profile.

Every significant instantaneous and cumulative component must retain:

\[
p\ge0.75,\qquad
d_{\rm fine}\le0.05,\qquad
\cos_{\rm history}\ge0.90,\qquad
\cos_{\rm error}\ge0.90.
\]

Only if all prospective held-outs pass may embedded export discrimination be
reconsidered. A held-out failure should instead select a smooth-profile local
truncation audit; it must not trigger an immediate fitted operator change.

## Hard stops

Do not:

- amend or relabel WP10c9d6c2;
- claim that the old continuation failure has one uniquely identified cause;
- redesign the near-excision operator from the old failure alone;
- run N1024;
- change production defaults;
- start embedded or nonlinear evolution;
- begin fixed-Q averaging or reduced slow-time evolution;
- add tide, wind, hot-state, S-curve, or QPE-cycle physics.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_continuum_lift_wp10c9d6c3/
```

The package contains:

- the continuum spline knots and coefficients;
- all four projected backgrounds and physical perturbations;
- primary and independent-projection histories;
- cumulative exports and final states;
- error-history matrices and continuum extrapolates;
- configuration, provenance, source hashes, and SHA-256 checksums.

Generation command:

```text
PYTHONPATH=src:scripts python3 \
  scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py
```

Focused verification:

```text
4 passed
```

Full repository verification:

```text
883 passed
4 subtests passed
1 repository-hygiene policy failure
```

The sole failure is the existing tracked-tree ceiling:

```text
978 < 850  -> false
```

No scientific, numerical, canonical-evidence, or WP10c9d6c3 test fails.
