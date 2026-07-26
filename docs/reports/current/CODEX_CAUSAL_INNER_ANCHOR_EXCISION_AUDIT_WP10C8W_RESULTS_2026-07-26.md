# WP10c8w independent anchor and inner-excision audit

Date: 2026-07-26

Base commit:
`6764fc117ce453b4deb5c6b1c275a19c7352b4be`

Parent evidence: WP10c8u and WP10c8v

Local equivalent meshes: N64, N128, and N256

Production physics changed: no

Production inner boundary changed: no

Production outer Roche boundary changed: no

New nonlinear truth evolution run: no

Formal fast-time average certified: no

Reduced architecture selected: no

## Executive result

WP10c8w removes the main provenance limitation in WP10c8v: the binding
N256 local anchor is no longer a direct N128 prolongation. It is corrected
onto the same declared local moment fiber, its physical rate is evaluated
independently through the full descriptor balance, and its plus/minus pair
is restored to exact equal coordinates.

That independent anchor passes. The spatial-phase conclusion does not
improve enough to pass.

The binding classification is:

> `independent_anchor_passed_excision_or_phase_unresolved`

The least-bad audit trace is a one-sided linear primitive extrapolation
limited so that the inner face has no incoming physical characteristic. It
slightly improves the N64/N128 common-exterior history, but the independent
N128/N256 comparison still gives:

```text
state observed order       0.49438
rate observed order       -0.03308
minimum/final rate cosine  0.44970
frequency relative defect  0.41103
```

against the locked gates:

```text
state/rate order       >= 0.75
signed cosine          >= 0.90
frequency defect       <= 0.10
zero-crossing defect   <= 0.10
damping defect         <= 0.25
```

Only the damping defect passes. A cell-centered trace is rejected before
propagation because it makes the local descriptor/generator catastrophically
ill conditioned and unstable.

Moving the excision edge by one or two N128 cells has a substantial effect
on the N128 transient but a much smaller effect at N256. This is encouraging
evidence that pure edge-location sensitivity contracts, but it does not
repair the N128/N256 phase law or meet the predeclared multi-mesh placement
gate.

WP10c8w therefore does not authorize:

- N512 or an embedded patch;
- replacement of the production inner trace;
- nonlinear fixed-`Q` odd/even averaging;
- an initial-slip map;
- a new reduced coordinate;
- reduced macrostepping.

## Independent N256 anchor

The N128 local moment values define the target. N128 prolongation supplies
only the N256 initial guess. The N256 primitive state is then projected in
the weighted normal space of the exact nonlinear coordinate map while the
outer buffer is frozen with a `1e12` weight multiplier.

The corrected anchor gives:

| Quantity | Result | Gate |
|---|---:|---:|
| Coordinate rank | `12/12` | full |
| Constraint condition | `2.3566e3` | `<= 1e10` |
| Maximum coordinate defect | `1.5921e-14` | `<= 1e-10` |
| Maximum active primitive-scale correction | `6.9056e-3` | `<= 1e-2` |
| Active weighted correction norm | `1.0929e-3` | diagnostic |
| Maximum buffer primitive-scale correction | `2.6149e-13` | `<= 1e-8` |
| Buffer weighted correction norm | `7.7552e-14` | diagnostic |

The correction is larger than the deliberately tiny mode-0 input amplitude
in a few cells. That amplitude is not an appropriate anchor-consistency
scale. The binding quantities are instead the primitive-column-scaled
correction and the exact coordinate defect.

The independently corrected primitive array has SHA-256:

```text
3a3929b2db9c2a683ee3f43fcf4669d8a4dbadcc666b06a70af3e8a6b803f6bf
```

## Exact equal-coordinate local pairs

The N64, N128, and N256 plus/minus pairs are corrected independently on
their local fibers.

| Mesh | Maximum pair coordinate defect | Pair gate | Operator gate |
|---:|---:|---:|---:|
| N64 | `3.72e-16` | pass | pass |
| N128 | `1.78e-15` | pass | pass |
| N256 | `1.78e-15` | pass | pass |

All three base and pair states pass the local physical contract:

- `H/R < 0.10`;
- scattering optical depth above `18.87`;
- zero incoming inner characteristics;
- zero light-cone excess;
- exact algebraic maps;
- descriptor-balanced physical rates.

The local outer face uses the audit-only frozen-exterior Rusanov trace. The
production Roche outer characteristic count is therefore intentionally not
applied to this truncated local domain.

The inherited generators are safely damped:

| Mesh | Spectral abscissa | Growth exponent over `0.125 s` |
|---:|---:|---:|
| N64 | `-3.15497 s^-1` | `-0.39437` |
| N128 | `-5.37989 s^-1` | `-0.67249` |
| N256 | `-9.92292 s^-1` | `-1.24036` |

## Inner-trace screen

Three audit traces were compared while leaving production defaults
unchanged.

### Inherited production reconstruction

The inherited trace is stable, but the N64/N128 common-exterior state/rate
history differences remain `1.2732/1.6069`; their minimum signed cosines
are `-0.3330/-0.7275`.

### Cell-centered trace

The cell-centered trace is rejected at N64 before propagation:

```text
generator factorization defect   3.584e3
storage-action defect            1.0
spectral abscissa                1.0029e23 s^-1
growth exponent over 0.125 s     1.2536e22
```

This is not a viable excision treatment for the present Gauss-storage
descriptor. The fail-fast propagation-safety gate prevents the matrix
exponential from hanging on this invalid operator.

### One-sided outgoing linear trace

The selected audit trace:

1. forms a spacing-aware one-sided linear extrapolation in `log R`;
2. applies the coupled primitive admissibility limiter;
3. further limits the trace if necessary until all physical
   characteristics point out through the excision face;
4. evaluates one common physical inner flux.

It is stable and keeps zero incoming inner characteristics. At N64/N128 it
is only modestly better than the inherited trace:

| Metric | Inherited | Outgoing linear |
|---|---:|---:|
| Maximum exterior state difference | `1.27324` | `1.27078` |
| Maximum exterior rate difference | `1.60690` | `1.52661` |
| Minimum exterior rate cosine | `-0.72746` | `-0.68195` |

This small improvement is insufficient to justify a production change.

## Independent N128/N256 phase result

Using the outgoing-linear audit trace:

| Quantity | N64/N128 | N128/N256 |
|---|---:|---:|
| Maximum state L2 difference | `1.27078` | `0.90208` |
| Minimum state cosine | `-0.33196` | `0.68325` |
| Maximum rate L2 difference | `1.52661` | `1.56202` |
| Minimum rate cosine | `-0.68195` | `0.44970` |

The state difference contracts only weakly, while the rate difference grows
slightly. The corresponding observed orders are:

```text
state  0.49438
rate  -0.03308
```

The shell-0 stress-rate signal gives:

| Mesh | Diagnostic frequency | Envelope log slope |
|---:|---:|---:|
| N64 | unresolved | `-13.4785 s^-1` |
| N128 | `31.9539 Hz` | `-9.5362 s^-1` |
| N256 | `54.2534 Hz` | `-12.0976 s^-1` |

For N128/N256:

```text
maximum zero-crossing relative defect  0.27646
frequency relative defect              0.41103
damping relative defect                0.21173
```

Thus the independently constructed fine anchor does not rescue spatial
phase convergence. The negative WP10c8v result was not primarily an
artifact of direct prolongation.

## Excision-position sensitivity

The inner edge was shifted outward on the same nested lattice by:

- zero cells;
- one N128 cell / two N256 cells;
- two N128 cells / four N256 cells.

All tested edges remain inside the horizon. Histories are compared only in
the common exterior region `R >= 2.2 rg`.

### One-N128-cell physical shift

| Mesh | Maximum state defect | Maximum rate defect | Minimum rate cosine |
|---:|---:|---:|---:|
| N128 | `0.12485` | `0.24009` | `0.97328` |
| N256 | `0.001034` | `0.004561` | `0.999990` |

### Two-N128-cell physical shift

| Mesh | Maximum state defect | Maximum rate defect | Minimum rate cosine |
|---:|---:|---:|---:|
| N128 | `0.27888` | `0.50833` | `0.86438` |
| N256 | `0.006181` | `0.020334` | `0.999794` |

The N256 histories are far less sensitive to edge placement than N128. The
strict multi-mesh `0.10` history gate nevertheless fails at N128, and the
selected N128/N256 phase comparison remains far outside its spatial gates.

This result narrows the diagnosis:

> the exact excision position is not the sole blocker; the
> boundary-adjacent storage/transport discretization still lacks a
> mesh-convergent phase law.

## Interpretation

WP10c8w establishes:

1. an independently corrected N256 local anchor and exact pair can be built
   under the declared moment and physical contracts;
2. the direct-prolongation limitation in WP10c8v does not explain the phase
   failure;
3. a cell-centered trace is incompatible with the present local descriptor;
4. the outgoing-linear trace is stable and slightly better at N64/N128 but
   fails the independent N128/N256 contraction and phase gates;
5. edge-position sensitivity decreases sharply at N256, but this does not
   make the full phase response spatially convergent.

It does not establish:

- continuum nonconvergence;
- a valid replacement production boundary;
- a converged fast frequency or damping rate;
- a nonlinear N256 truth trajectory;
- a fixed-`Q` invariant fast measure;
- negligible odd or even averaged forcing;
- a required embedded-patch resolution.

## Decision

The binding decisions are:

```text
independent anchor consistency              PASS
selected trace spatial refinement           FAIL
strict excision-placement insensitivity     FAIL
N512 or embedded patch authorization        NO
fixed-Q averaging authorization             NO
production boundary replacement             NO
```

The next package must isolate the boundary operator itself before another
factor-two history refinement.

## Locked next plan: WP10c8x

### Phase 1 — Separate inner flux and storage traces

The present face reconstruction feeds both the inner physical flux and the
Gauss/path storage construction. Add audit-only, independently selectable
inner traces for:

- physical flux;
- mapped storage;
- responsive-height storage.

Keep all production defaults unchanged. Reproduce the inherited operator
exactly when every override is `inherit`.

Use the existing N64/N128/N256 anchors to determine whether the controlling
boundary row is caused by:

- physical perfect-fluid/stress transport;
- mapped storage;
- responsive-height storage;
- or their cancellation.

### Phase 2 — Static boundary consistency tests

Before another history propagation, use smooth admissible outgoing
manufactured profiles on nested N64/N128/N256/N512 local grids.

Measure separately:

- inner trace error;
- central perfect-fluid flux error;
- stress-flux error;
- Rusanov contribution;
- first-cell flux divergence;
- mapped and height-storage action;
- complete boundary-row residual.

Require:

```text
descriptor factorization defect <= 1e-8
storage-action defect           <= 5e-5
no incoming inner characteristic
boundary-row order              >= 1.5 on smooth profiles
```

Reject any candidate that produces branch-sensitive finite-difference
Jacobians, singular storage, or an explosive generator.

### Phase 3 — One derived characteristic-space candidate

If the static audit identifies flux reconstruction as the blocker, derive
one one-sided characteristic-space extrapolation rather than adding another
ad hoc primitive limiter. It must:

- prescribe no incoming physical data;
- reconstruct only from the interior;
- return one conservative physical face flux;
- preserve positivity and causal-stress admissibility;
- have a deterministic differentiable branch near the anchor.

If storage quadrature is controlling, repair that boundary quadrature
instead and leave the physical flux untouched.

### Phase 4 — Bounded N128/N256 history gate

Only candidates passing the static contract receive a `0.125 s` frozen
linear history test on the independent N128/N256 anchors.

Retain the WP10c8w gates:

```text
state/rate order          >= 0.75
same-time signed cosine   >= 0.90
zero-crossing defect      <= 0.10
frequency defect          <= 0.10
damping defect            <= 0.25
placement history defect  <= 0.10
```

### Phase 5 — Conditional escalation

- If one boundary operator passes: authorize one independent N512 local
  refinement, followed by nonlinear local truth.
- If only permanent fine resolution converges: authorize an embedded inner
  patch design with conservative two-way coupling.
- If no formally consistent boundary operator converges: revisit the inner
  finite-volume/storage formulation before any reduction.

Do not run fixed-`Q` averaging, select reduced coordinates, or alter the
production boundary in WP10c8x.

## Machine evidence and reproducibility

Primary machine evidence:

```text
outputs/tables/causal_inner_anchor_excision_audit_wp10c8w.json
outputs/tables/causal_inner_anchor_excision_audit_wp10c8w_arrays.npz
outputs/checkpoints/causal_five_field_wp10c8w/
```

Primary arrays SHA-256:

```text
91cf5202a05bb8a638f7bfb1b975d98a34de7fc6307533420efbae5eeecbc5aa
```

Runner:

```text
scripts/run_causal_inner_anchor_excision_audit_wp10c8w.py
```

Runner SHA-256:

```text
ea5bf64a19b5ebed830b2ecbf9174a0f71a12263a5cf8ceece155b600c1d1245
```

The runtime outputs remain ignored. Their hashes are recorded here so the
binding evidence can be verified without committing large generated
artifacts.
