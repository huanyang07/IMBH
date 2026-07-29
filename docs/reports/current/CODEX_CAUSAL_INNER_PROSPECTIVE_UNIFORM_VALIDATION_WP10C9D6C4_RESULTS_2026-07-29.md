# Causal inner prospective held-out uniform validation — WP10c9d6c4

Date: 2026-07-29
Analyzed base commit: `3d973aa28c68d242bd33c88efe2226e2e48eb281`
Analyzed parent: `da2d7612cc9a2fff7093bee705f3f5fbe2d2101d`

## Binding classification

```text
prospective_heldout_uniform_validation_failed
```

WP10c9d6c4 changes no physical or numerical operator. The smooth continuum
background, proper-measure projection, monolithic tangent, fixed physical
observable scales, and all physical-export gates are inherited unchanged from
WP10c9d6c3.

The method and continuum-lift contracts pass. Three of four newly declared
held-out profiles pass the strict N128/N256/N512 instantaneous and cumulative
export-direction gates. The smooth first-cell-dominated outgoing profile fails
the instantaneous refinement-error cosine:

\[
\cos_{\rm error}=0.85964<0.90.
\]

The smooth calibration fit to the historical common perturbation also fails,
with instantaneous and cumulative fine-triplet error cosines

\[
0.45857,\qquad 0.89226.
\]

The only authorized next step is a smooth-profile, radius- and block-resolved
local-truncation audit. Embedded discrimination, direct operator redesign,
nonlinear evolution, production promotion, fixed-\(Q\) averaging, and reduced
slow-time evolution remain blocked.

## Purpose

WP10c9d6c3 showed that the unchanged uniform monolithic operator converges
cleanly for two smooth continuum perturbations. That was enough to stop an
immediate redesign, but not enough to establish that the result was general.

WP10c9d6c4 therefore freezes the c3 contract before testing:

1. a smooth proper-measure representation of the historical common
   perturbation, used only as calibration; and
2. four prospective analytic held-outs with different supports, widths, and
   five-field mixtures.

No profile, coefficient, amplitude, path, tolerance, or gate was adjusted
after inspecting the propagated export results.

## Frozen profile suite

The profile-definition SHA-256 is

```text
ea204dd405ddd5331feb269d93def2239f2643671919379abfc83fa9c4bb178a
```

The suite is:

| Profile | Role | Definition |
|---|---|---|
| `historical_common_smooth_fit` | calibration only | C4 quintic proper-measure fit to the committed historical N128 common perturbation |
| `heldout_mid_inner` | held-out | narrow mixed five-field profile centered at \(4.60\,r_g\) |
| `heldout_broad_outer_inner` | held-out | two-component broad profile extending toward the outer inner domain |
| `heldout_first_cell_outgoing` | held-out | smooth inward-acoustic profile centered at \(1.84\,r_g\), leaving through excision |
| `heldout_two_lobe_mixed` | held-out | oppositely signed mixed lobes centered at \(2.75\) and \(3.75\,r_g\) |

The first-cell profile uses the physical inward-acoustic eigenvector of the
frozen smooth background at \(1.84\,r_g\). Its coordinate speed is

\[
-0.700544\,c,
\]

so it is outgoing from the computational domain and supplies no incoming
excision data. Its peak remains inside the physical band occupied by the
coarse-grid first cell on every resolution.

## Historical calibration construction

The historical target is the committed N128 common physical perturbation. A
24-coefficient quintic C4 spline in log radius is fitted to its
proper-measure cell averages, with:

- inner boundary value fixed to the first target cell average;
- zero outer perturbation;
- an independent degree-seven C6 fit as a representation check.

The calibration fit has:

| Quantity | Result |
|---|---:|
| scaled relative L2 fit defect | \(5.00\times10^{-2}\) |
| scaled fit cosine | \(0.998747\) |
| quintic/septic maximum representation difference | \(9.29\times10^{-3}\) |

This is not claimed to reconstruct every grid-scale feature of the historical
fiber. It is a smooth calibration profile retaining its dominant physical
direction.

## Construction and method gates

The c3 background and scales reproduce bitwise:

\[
\max d_{\rm frozen}=0.
\]

All nominal, half-amplitude, and sign-reversed perturbed states remain on the
inactive reconstruction/admissibility branch. The four monolithic tangents
pass all inherited:

- reconstruction and descriptor-component identities;
- stationary, storage-rate, and export directional checks;
- base-rate balance and generator factorization;
- characteristic conditioning and excision causality;
- center-broken path checks.

The sign/amplitude propagation defects are both exactly zero for the declared
N128 and N512 checks.

### Restart normalization

For the N512 first-cell outgoing profile, the final state is nearly zero after
the packet leaves through excision. The historical final-state-relative
restart diagnostic is therefore ill-conditioned and reaches

\[
1.97\times10^{-8}.
\]

WP10c9d6c4 retains that number as a diagnostic but gates a conservative upper
bound normalized by the maximum initial/final state norm. The binding maximum
is

\[
3.55\times10^{-15}.
\]

This changes no state, trajectory, or physical threshold. It prevents a
roundoff-sized absolute split/restart difference from being divided by a
vanishing final packet.

## Continuum-lift uncertainty

Every profile is projected independently as a Kerr–Schild proper-measure
finite-volume cell average using quadrature orders 24 and 12.

| Profile | History lift / fine spatial | Cumulative lift / fine spatial |
|---|---:|---:|
| historical calibration | \(7.06\times10^{-13}\) | \(1.69\times10^{-12}\) |
| mid-inner | \(1.02\times10^{-12}\) | \(1.24\times10^{-13}\) |
| broad outer-inner | \(1.71\times10^{-11}\) | \(1.62\times10^{-12}\) |
| first-cell outgoing | \(2.36\times10^{-13}\) | \(1.88\times10^{-13}\) |
| two-lobe mixed | \(5.50\times10^{-13}\) | \(4.60\times10^{-13}\) |

All are far below the unchanged `0.10` gate. The failures cannot be attributed
to projection-order uncertainty.

## Binding N128/N256/N512 results

The unchanged gates are

\[
p_{\rm RMS},p_{\max},\min_a p_a\ge0.75,
\]

\[
d_{\rm fine}\le0.05,\qquad
\cos_{\rm history}\ge0.90,\qquad
\cos_{\rm error}\ge0.90.
\]

### Prospective held-outs

| Profile | History | \(p_{\rm RMS}\) | \(p_{\max}\) | \(\min p_a\) | \(d_{\rm fine,max}\) | \(\cos_{\rm error}\) | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| mid-inner | instantaneous | 2.191 | 2.299 | 2.182 | \(2.92\times10^{-6}\) | 0.9684 | yes |
| mid-inner | cumulative | 1.837 | 2.029 | 1.598 | \(1.08\times10^{-6}\) | 0.9845 | yes |
| broad outer-inner | instantaneous | 2.079 | 2.113 | 1.280 | \(2.99\times10^{-7}\) | 0.9989 | yes |
| broad outer-inner | cumulative | 2.070 | 2.074 | 1.255 | \(2.64\times10^{-7}\) | 0.9997 | yes |
| first-cell outgoing | instantaneous | 1.980 | 1.424 | 1.720 | \(3.61\times10^{-5}\) | **0.8596** | **no** |
| first-cell outgoing | cumulative | 2.173 | 2.309 | 1.630 | \(8.60\times10^{-7}\) | 0.9263 | yes |
| two-lobe mixed | instantaneous | 2.408 | 2.414 | 1.878 | \(4.34\times10^{-5}\) | 0.9567 | yes |
| two-lobe mixed | cumulative | 2.370 | 2.412 | 1.889 | \(5.09\times10^{-6}\) | 0.9607 | yes |

Three held-outs pass both histories. The first-cell profile contracts strongly
in every reported norm and component, but its instantaneous error vector has
not reached the required common direction.

For that profile, the six inner/net M/J/E component error cosines lie between
approximately `0.8586` and `0.8616`. The cumulative equivalents improve to
approximately `0.9172`–`0.9272`. This points to a transient near-excision
effect rather than a failure of cumulative conservative exchange.

### Historical calibration

| History | \(p_{\rm RMS}\) | \(p_{\max}\) | \(\min p_a\) | \(d_{\rm fine,max}\) | \(\cos_{\rm error}\) | Pass |
|---|---:|---:|---:|---:|---:|---|
| instantaneous | 2.651 | 3.451 | 2.109 | \(3.62\times10^{-5}\) | **0.4586** | **no** |
| cumulative | 2.246 | 2.398 | 2.133 | \(7.42\times10^{-7}\) | **0.8923** | **no** |

The calibration failure is entirely a direction failure: all fine-triplet
orders and maximum-difference gates pass by wide margins. The instantaneous
inner/net M/J/E component cosines range from approximately `0.416` to `0.483`.

The error-history SVD has first/second mode fractions

\[
0.8891,\qquad0.1072,
\]

for the calibration profile. This is materially more multi-directional than
the passing held-outs. It supports retaining the historical profile as a
calibration diagnostic rather than using it as the sole definition of
accuracy.

## Interpretation

WP10c9d6c4 narrows the conclusion in both directions.

Positive evidence:

1. The unchanged monolithic uniform operator passes three diverse prospective
   held-outs.
2. Broad, mid-inner, and two-lobe perturbations all have stable fine-grid
   error directions.
3. Lift uncertainty, tangent construction, causality, admissibility,
   amplitude, sign, and restart diagnostics do not explain the remaining
   failures.

Negative evidence:

1. Uniform convergence is not yet certified for every prospective held-out.
2. A smooth packet concentrated in the first physical cell band misses the
   instantaneous direction gate.
3. A smooth fit to the historical common perturbation retains its
   direction-sensitive behavior even though its norms contract strongly.

Therefore:

> A global or multi-block redesign remains premature, but an
> evidence-preserving near-excision local-truncation audit is now justified.

The next audit must distinguish an actual first-half-cell truncation defect
from a narrow-packet pre-asymptotic crossover. It must not fit a new operator
to either failed profile.

## Authorized next package

### WP10c9d6c5 — smooth-profile local-truncation audit

Freeze the exact c4 background, profiles, scales, and tangents.

Use:

- `heldout_first_cell_outgoing` as the prospective failed target;
- `historical_common_smooth_fit` as calibration-only corroboration;
- the three passing held-outs as negative controls.

At the initial time and over a short pre-exit interval:

1. evaluate the linearized monolithic residual/generator action on
   N64/N128/N256/N512;
2. construct an independent high-order continuum or manufactured action for
   the same smooth profiles;
3. resolve the truncation defect by radius and by:
   - inner shared face;
   - first-cell conservative transport;
   - mapped storage;
   - responsive-height storage;
   - shear-principal path;
   - height-principal path;
   - local stress relaxation;
   - geometry;
   - cooling;
   - stream source;
   - lower responsive-height work;
4. use fixed physical bands, not a resolution-dependent number of cells;
5. require the same controlling term or subspace on N128/N256 and
   N256/N512;
6. compare failed profiles with all passing controls;
7. preserve exact M/J/E and block ledgers.

### Binding decisions

| c5 result | Authorized response |
|---|---|
| Stable first half-cell/inner-face defect in both failed profiles | one outgoing boundary-fitted half-cell candidate |
| Stable space/storage defect over a fixed band | one compatible fixed-band space–storage candidate |
| Stable principal-path or lower-source defect | one targeted path/source consistency candidate |
| Narrow-packet crossover with contracting continuum error and no stable local defect | no redesign; add prospective width/support validation |
| No stable localized mechanism | no fitted local repair; reconsider the strict direction certificate or broader architecture |

No operator intervention is authorized until c5 selects one mechanism.

## Hard stops

Do not:

- relabel WP10c9d6c2 or WP10c9d6c3;
- tune the failed profile, width, amplitude, gate, or characteristic family;
- treat strong norm contraction as a retroactive pass;
- infer a boundary defect from packet location alone;
- begin embedded discrimination;
- implement a half-cell or space–storage correction before c5;
- run N1024;
- change production defaults;
- begin nonlinear, fixed-\(Q\), or reduced slow-time evolution.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_prospective_uniform_validation_wp10c9d6c4/
```

The package contains:

- all four projected backgrounds and five projected profiles;
- primary and independent-projection histories;
- cumulative exports and final states;
- historical calibration spline coefficients;
- fixed outgoing characteristic vector and speed;
- conditioned metrics, error-history matrices, and continuum extrapolates;
- configuration, provenance, source hashes, and SHA-256 checksums.

Generation command:

```text
PYTHONPATH=src:scripts python3 \
  scripts/run_causal_inner_prospective_uniform_validation_wp10c9d6c4.py
```

Focused c/c1/c2/c3/c4 and canonical verification:

```text
24 passed
```

Full repository verification:

```text
887 passed
4 subtests passed
1 repository-hygiene policy failure
```

The sole failure is the existing tracked-tree ceiling:

```text
986 < 850  -> false
```

No scientific, numerical, method, or canonical-evidence test fails.
