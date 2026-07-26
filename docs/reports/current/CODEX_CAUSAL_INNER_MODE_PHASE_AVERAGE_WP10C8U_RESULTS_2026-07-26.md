# WP10c8u dense inner-mode phase and activity audit

Date: 2026-07-26

Base commit:
`3ccdb9532359acbaa197e066a800a9119dfe60ef`

Truth meshes: N64 and N128

Production physics changed: no

Production spatial operator changed: no

Production BDF formula changed: no

New truth evolution run: no

Reduced evolution run: no

Formal fast-time average certified: no

Architecture change authorized: no

## Executive result

WP10c8u recomputes fresh primitive and 34-coordinate rates at every saved
WP10c8t state. The four N64 trajectories contribute `101/201` coarse/fine
states per side and the four N128 trajectories contribute the same, for
`1208` dense state/rate evaluations in total. Every previously committed
sparse rate, coordinate, primitive, and scaled primitive-rate sample is
reproduced bitwise with zero relative defect.

The binding classification is:

> `localized_inner_phase_spatially_unresolved`

The dense histories show that the late endpoint disagreement reported by
WP10c8t begins almost immediately:

```text
first amplitude-gate failure                    0.002500 s
first combined same-time-gate failure           0.002500 s
first signed direction-cosine failure           0.003750 s
first significant shell-0 stress-sign mismatch  0.004375 s
```

The N64/N128 signed rate cosine starts at `0.99981770`, falls to `0.38650`
at `0.00375 s`, reaches `-0.88057` at `0.004375 s`, and is `-0.76092` at
`0.125 s`. Its minimum over the dense history is `-0.98303`. The maximum
amplitude-ratio defect is `3.97077`.

This is not merely an endpoint sign convention. The pair orientation is
fixed by the excellent positive initial alignment, and all binding
comparisons use the signed cosine.

## Small net slip does not establish averaging

The final signed coordinate-slip maxima are small:

```text
N64   1.96696e-7
N128  1.05678e-7
```

The corresponding absolute impulses are:

```text
N64   2.47829e-6
N128  1.28888e-6
```

They are about `12.60` and `12.20` times the signed maxima. This confirms
substantial cancellation. It does not show that the unresolved activity has
disappeared or that its nonlinear averaged effect is negligible.

The sliding-window results are decisive:

| Window | N64 mean-norm range | N128 mean-norm range | N64 RMS range | N128 RMS range |
|---:|---:|---:|---:|---:|
| `0.010 s` | `0.938-79.072` | `0.530-31.566` | `1.370-116.035` | `0.763-92.690` |
| `0.025 s` | `1.110-26.482` | `0.805-9.817` | `2.666-79.232` | `1.121-59.986` |
| `0.050 s` | `0.742-7.905` | `0.680-4.671` | `4.917-56.726` | `1.431-42.505` |
| `0.100 s` | `1.210-5.524` | `0.181-2.830` | `8.029-40.399` | `2.354-30.080` |

No tested window length passes the same-time N64/N128 direction-and-amplitude
gate for every starting phase. Even the longest-window maximum mean norms,
`5.524` and `2.830`, remain far above the `0.10` averaging-plausibility
reserve.

The `0-0.10 s` averages happen to align reasonably well, with signed cosine
`0.97465` and amplitude-ratio defect `0.37481`. Moving the same `0.10 s`
window to `0.025-0.125 s` changes the signed cosine to `-0.66200`. This
window-placement sensitivity forbids interpreting one favorable running
average as the formal fast-time average.

The repository therefore certifies only:

> the freely evolving equal-coordinate pair leaves a small signed
> slow-coordinate displacement over `0.125 s`, while retaining large,
> phase-sensitive, spatially unresolved fast activity.

It does not certify

\[
\left\langle F(Q,Z)\right\rangle_{\rm fast}
\]

at fixed `Q`, a periodic orbit, a unique conditional invariant measure, or
an initial-slip closure.

## Phase history

The dense fine-grid shell-0 stress-rate zero crossings are:

```text
N64   0.008253, 0.034185, 0.069333, 0.113450 s
N128  0.004100, 0.015980, 0.032454, 0.050611, 0.061257 s
```

The different first crossing and subsequent spacing show that the N128
history is not a simple time-shifted copy of N64. A fitted lag may be useful
diagnostically, but no time shift is used for a binding gate.

The fine-grid temporal uncertainty is small at the endpoint:

```text
N64   0.0295168 gate units
N128  0.0352984 gate units
```

The maximum coarse/fine rate uncertainty over the complete transient is
larger (`2.9493` at N64 and `8.4212` at N128), so individual early
instantaneous rates are not promoted beyond the declared matched-history
evidence. The spatial phase split remains much larger and occurs in both
coarse and fine histories.

## Localization and physical attribution

After conservative N128-to-N64 restriction, the primitive-rate difference
remains predominantly in shell 0:

```text
fine N64 shell-0 rate L1 fraction at t=0 / 0.125 s
0.93881 / 0.69813

restricted fine N128 shell-0 rate L1 fraction at t=0 / 0.125 s
0.96012 / 0.87202
```

The cross-mesh normalized primitive-rate profile cosine starts at `0.99110`
but is only `0.13135` at `0.125 s`; its minimum is `-0.87525`.

Exact term decompositions were performed at nine event times on both meshes.
All decompositions pass. The maximum reconstructed-rate defects are
`3.70e-11` at N64 and `7.04e-11` at N128; storage-action reconstruction
defects remain below `2.65e-16`.

At the first amplitude failure (`0.0025 s`), the largest primitive-rate
half-difference terms are the inner boundary characteristic transport and
the neighboring perfect-fluid transport:

```text
N64 boundary characteristic  2.2005e-3
N64 perfect-fluid transport   8.3585e-4

N128 boundary characteristic  2.6118e-3
N128 perfect-fluid transport  1.3715e-3
```

They are concentrated in the first few cells, approximately
`1.84-2.21 rg`, and primarily act through `log Sigma`. Perfect-fluid
geometry, stress relaxation, and Rusanov transport are smaller. Stream and
radiative-cooling terms are not controlling. By `0.00375-0.004375 s`, N64
is still boundary-transport dominated while N128 is perfect-fluid-transport
dominated. The first cross-mesh departure is therefore an inner
transport/geometry phase problem, not a stream-source, cooling, or external
macro-interface ambiguity.

This attribution does not prove that the inner boundary condition is wrong.
It shows that the current N64/N128 semidiscretizations do not yet deliver a
common fast phase/frequency for the boundary-adjacent shell-0 response.

## Diagnostic subspace result

Weighted shell-0 POD is retained only as a dimensional diagnostic.

The first six N64/N128 principal cosines are:

```text
state  0.98368, 0.84607, 0.76033, 0.49390, 0.32654, 0.02468
rate   0.99698, 0.94763, 0.66532, 0.50242, 0.24517, 0.16631
```

The first one or two directions are similar, but the remaining subspaces
are not sufficiently aligned. These POD results do not establish a physical
eigenmode, a converged oscillation frequency, or a two-coordinate reduced
state. No DMD/eigenmode claim is made from this one short non-normal
transient.

## Missing even-response evidence

WP10c8u measures the odd plus/minus response and its signed/absolute
activity. It does not measure

\[
\frac{F(x_+)+F(x_-)}{2}-F(x_0),
\]

because no matching unperturbed center trajectory through `0.125 s` exists
under the same fixed-step contract. Thus nonlinear quadratic drift cannot
be bounded from the present caches. Small signed plus/minus slip must not be
used as evidence that the averaged even response is small.

## Decision

WP10c8u does not authorize:

- a formal fast-time average;
- an initial-slip map;
- a scalar relaxation variable;
- a two-coordinate oscillatory mode;
- an embedded inner patch as the selected production architecture;
- a reduced macrostep;
- tide, wind, hot-state, stability, or cycle claims.

The active blocker is now:

> the boundary-adjacent shell-0 fast phase/frequency is not spatially
> converged between N64 and N128.

## Locked next plan: WP10c8v

### Phase 1 — Inner local spatial-phase preflight

Construct a conservative local inner-domain audit covering at least
`1.8-6.5 rg`, using the exact production inner boundary, geometry, flux,
stress, and responsive-height descriptor. Freeze or constrain the exterior
slow-shell coordinates and provide matched physical trace data at the local
outer edge.

Compare the N64- and N128-equivalent local grids and exactly one additional
local factor-two refinement. Do not launch a global N256 trajectory.

Measure:

- signed shell-0 rate history;
- zero crossings, phase velocity, and damping envelope;
- radial centroid and width;
- boundary, perfect-fluid, stress, Rusanov, geometry, cooling, and relaxation
  contributions;
- conservative coupling to the frozen exterior;
- temporal refinement separately from spatial refinement.

### Phase 2 — Multiple fixed-`Q` lifts

Only if the local phase/frequency converges, run several equal-`Q` inner
lifts:

- the original plus/minus pair;
- at least two amplitudes;
- one held-out fiber direction;
- the unperturbed center state.

Hold the slow coordinates fixed to the declared tolerance or restore them
with an explicit constraint. Determine whether the conditional fast system
approaches a fixed point, periodic orbit, low-dimensional invariant set, or
no common attractor.

### Phase 3 — Odd and even averaged forcing

For every converged lift, calculate:

\[
r_{\rm odd}=\frac{F_+-F_-}{2},
\qquad
r_{\rm even}=\frac{F_++F_-}{2}-F_0,
\]

together with signed slip, absolute impulse, RMS amplitude, and
window/phase dependence.

No formal average may be certified without:

- a mesh-convergent fast invariant object or measure;
- lift-independent averaged forcing;
- bounded even/quadratic correction;
- stable window-length and starting-phase results;
- a declared slow-coordinate drift tolerance.

### Phase 4 — Architecture decision

- Converged fast object with negligible odd/even average: prototype a
  startup microburst or initial-slip correction.
- One converged oscillatory pair with non-negligible mean: retain two real
  inner coordinates.
- Spatially unresolved local phase after local refinement: use an embedded
  fine inner patch; do not fit mesh-dependent phase into abstract moments.
- Several converged localized modes: retain a small inner-state vector or
  patch.
- Distributed response: move to the conservative staggered radial
  finite-volume/PDE architecture.

Every candidate state must undergo a new worst-case exact equal-coordinate
slow-rate fiber audit before reduced evolution.

## Machine evidence and reproducibility

Primary evidence:

```text
outputs/tables/causal_inner_mode_phase_average_audit_wp10c8u.json
outputs/tables/causal_inner_mode_phase_average_audit_wp10c8u_arrays.npz
```

The arrays SHA256 is:

```text
863dcaf818451beef17cc55f043d590cb437dd2f0993bf5d01df6e2082d0e26b
```

The JSON SHA256 is:

```text
ac59572600c0d0b03413135b63ab56ae632a193e968f24c0526e83af96bdf1fb
```

The runner SHA256 recorded in the JSON and reproduced after the run is:

```text
088b48cf6630fe01cb4d495efab87952c51b2f2a604da9c941a22502f5391a00
```

All eight dense caches store their own arrays hashes and sparse-reproduction
records. The dense-cache producer hash is preserved as provenance, while
analysis-only runner changes do not invalidate scientifically unchanged
rate caches.
