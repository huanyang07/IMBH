# WP10c8x Inner Boundary Consistency Results

- Date: 2026-07-26
- Base commit: `6764fc117ce453b4deb5c6b1c275a19c7352b4be`
- Classification:
  `static_pass_but_common_initial_mode_unresolved`
- Production boundary changed: no
- New nonlinear truth evolution: no
- N512 history, fixed-`Q` averaging, and reduced architecture: not
  authorized

## Executive verdict

WP10c8x separates the audit-only inner physical-flux reconstruction from
the mapped-storage reconstruction while leaving the production default
unchanged. Responsive-height temporal storage remains a cell-local
primitive-space one-form and is therefore independent of both face-trace
overrides.

The smooth manufactured first-cell audit gives a clear result:

1. the production trace is formally consistent and has the smallest
   boundary-row error;
2. the one-sided outgoing-linear flux trace is also second order;
3. cell-centered flux and storage controls are only first order and are
   rejected;
4. separating storage from flux does not repair the previously observed
   phase failure.

The bounded history comparison adds an important qualification. The
inherited N128/N256 mode-0 pairs are exact equal-coordinate pairs on their
own meshes, but they are not a sufficiently common spatial initial
perturbation. At `t=0`, after conservative restriction to the common
exterior region, the state and rate comparisons are

```text
                         state       rate
signed cosine            0.93831     0.93321
N256/N128 amplitude      0.80411     0.65381
relative L2 difference   0.37093     0.45517
```

These fail the new common-initial-mode precondition

```text
signed cosine >= 0.99
amplitude defect <= 0.05
relative L2 defect <= 0.10
```

Consequently:

> WP10c8x rejects boundary flux/storage conflation as the explanation for
> the phase problem, but the present histories cannot by themselves certify
> a continuum boundary-phase failure because their initial perturbations
> are not spatially matched tightly enough.

The next binding task is a simultaneous, continuum-matched equal-coordinate
fiber lift across N64/N128/N256 before another boundary or patch decision.

## Implemented trace separation

`CausalFiveFieldDAEContext` now has independently selectable audit-only
controls:

```text
inner_flux_trace_override
inner_storage_trace_override
```

The legacy combined

```text
inner_boundary_trace_override
```

remains as a compatibility fallback. Setting both new controls to
`linear_outgoing` reproduces the legacy combined setting exactly.

The reconstruction API now declares its purpose:

```text
purpose="flux"
purpose="storage"
```

The physical inner and interior fluxes use the flux reconstruction. The
Gauss mapped state and Gauss source path use the storage reconstruction.
The responsive-height temporal-storage path does not consume either face
trace. Unit tests verify these independence and compatibility identities.

All production defaults remain `inherit`; no physical boundary condition
was changed.

## Static smooth-profile audit

The manufactured primitive chart is a degree-five Chebyshev profile in
`ln R`, fitted to the independently corrected WP10c8w N256 anchor. It is
sampled on nested local grids with global-equivalent sizes

```text
N64, N128, N256, N512
```

and a 32-point Gauss reference is used for the exact first-cell storage and
source terms.

The audit measures:

- inner flux and storage trace error;
- central perfect-fluid and causal-stress flux error;
- Rusanov contribution;
- first-cell mapped storage and source error;
- complete boundary transport and first-cell residual error;
- inner characteristic direction.

### Passing candidates

| Candidate | Flux trace | Storage trace | N256/N512 complete-row order | N512 complete-row error | Result |
|---|---|---|---:|---:|---|
| production | inherit | inherit | `2.9324` | `2.27e-8` | pass |
| flux-linear | linear | inherit | `2.0401` | `1.80e-6` | pass |
| storage-linear | inherit | linear | `2.9324` | `2.27e-8` | pass |
| both-linear | linear | linear | `2.0401` | `1.80e-6` | pass |

At N512, the production trace gives:

```text
inner trace error                 5.41e-9
total inner flux error            2.76e-9
perfect-fluid inner flux error    2.76e-9
stress inner flux error           1.99e-12
first-cell mapped-storage error   1.33e-6
first-cell source error           3.34e-8
boundary transport error          2.85e-8
complete boundary-row error       2.27e-8
maximum inner speed / c          -0.65657
```

The outgoing-linear flux trace remains causal and second order, but its
N512 complete-row coefficient is about eighty times larger than the
production trace. The static evidence therefore does not support replacing
the production boundary.

### Rejected controls

| Candidate | N256/N512 complete-row order | N512 complete-row error | Reason |
|---|---:|---:|---|
| flux cell-centered | `0.9901` | `8.17e-4` | first-order boundary transport |
| storage cell-centered | `0.9936` | `3.79e-4` | first-order storage/source row |

All smooth equal-state face-1 Rusanov contributions vanish exactly. The
static defect is therefore not caused by a hidden dissipative jump.

## Bounded frozen-history audit

Only the four static-pass candidates receive N64/N128/N256 descriptor and
`0.125 s` frozen-linear histories. Every final operator passes:

```text
maximum factorization defect       3.55e-15
maximum storage-action defect      2.92e-9
inner incoming characteristics     0
propagation growth exponent        < 0
```

No candidate passes the history contract.

| Candidate family | State order | Rate order | Min fine signed cosine | Zero-crossing defect | Frequency defect | Damping defect |
|---|---:|---:|---:|---:|---:|---:|
| production / storage-linear | `0.4972` | `0.0408` | `0.4497` | `0.2773` | `0.4325` | `0.5086` |
| flux-linear / both-linear | `0.4944` | `-0.0331` | `0.4497` | `0.2765` | `0.4110` | `0.2117` |

The linear flux trace improves the damping comparison, but leaves the
state/rate order, same-time direction, zero crossings, and frequency far
outside the gates.

### Exact operator equivalences

Across N64, N128, and N256:

```text
production == storage-linear
flux-linear == both-linear
```

bitwise for:

- descriptor matrix;
- stationary Jacobian;
- storage-rate derivative;
- evolving generator;
- base physical rate.

This proves that the outgoing-linear storage override changes only an
unused boundary face chart at these anchors; the active Gauss storage slope
is already the same. The only dynamically consequential alternative in
this package is the flux trace, and it does not cure the phase failure.

## Interpretation

### What is established

- The production inner boundary row is high-order consistent on smooth
  outgoing profiles.
- The physical-flux and mapped-storage reconstructions are now independently
  selectable and tested.
- Responsive-height temporal storage is independent of those face controls.
- A cell-centered trace is not an acceptable smooth boundary treatment.
- A one-sided outgoing-linear flux is formally consistent but has a larger
  error coefficient than production.
- The prior bounded phase failure is not caused by one trace being reused
  simultaneously for physical flux and mapped/height storage.
- No tested candidate authorizes N512 history, fixed-`Q` averaging, a
  production boundary change, or a reduced architecture.

### What is not established

- The current histories do not prove continuum nonconvergence of the
  production boundary.
- The N128/N256 initial perturbations fail a common-profile gate before any
  evolution.
- No physical fast invariant object, phase law, relaxation law, or averaged
  slow forcing is certified.
- No embedded patch or extra reduced coordinate is authorized.

## Main problems and solutions

### Problem 1: the exact fibers are not a common spatial perturbation

Exact equality of the 34 retained coordinates on each mesh does not imply
that the unresolved radial profile is the same continuum object. The
present N128/N256 state and rate differences already fail at `t=0`.

#### Solution

Construct the pair jointly across meshes. Use one smooth physical
shell-0 perturbation profile and solve a constrained minimum-correction
problem on every mesh that enforces:

- exact equal retained coordinates on the plus/minus pair;
- exact physical and DAE gates;
- a common normalized restricted state profile;
- a fixed plus/minus orientation;
- minimal correction outside the active inner region.

The common-initial-mode gate must pass before any phase history is binding.

### Problem 2: the boundary alternatives do not improve dynamic phase

Both formally consistent flux choices retain low/negative contraction
orders and a fine signed cosine near `0.45`.

#### Solution

Do not add another ad hoc primitive trace. First repeat the comparison from
a common initial profile. If the common-profile history still fails, perform
a mode-resolved near-horizon dispersion audit and then change the inner
finite-volume formulation or retain permanent local fine resolution.

### Problem 3: the tested response may be a grid-sensitive lifted mode

The mode begins near the excision region, but its starting amplitude and
shape differ materially between N128 and N256. A phase comparison cannot
separate lift error from operator dispersion under those conditions.

#### Solution

Track both:

```text
same-time solution error
same-mode phase/amplitude error
```

after matching the initial profile. Report the projected discrete spectrum,
excited-mode weights, radial wavelength, group motion, and cells per
wavelength. POD/DMD remains diagnostic; no physical eigenmode is selected
without cross-mesh convergence and held-out-fiber support.

### Problem 4: failed operator caches must not become binding evidence

During development, a failed intermediate operator payload was reproduced
successfully under a forced rebuild. Such a negative cache must not be
treated as final evidence.

#### Solution

WP10c8x now reuses only operator caches whose complete descriptor,
storage-action, state, characteristic, and factorization contract has
already passed. Final equivalence pairs were independently rebuilt and are
bitwise equal on all three meshes.

## Locked next plan: WP10c8y

### Phase 1 — Freeze WP10c8x evidence

Freeze and hash:

- all smooth-profile rows;
- final N64/N128/N256 operators;
- production/storage and flux/both equivalence checks;
- current history and initial-pair metrics;
- exact coordinate and normalization definitions.

Do not change production physics, BDF2, the five-shell layout, or `q_34`.

### Phase 2 — Build one continuum-matched inner perturbation

Define one smooth, compact shell-0 physical perturbation in a declared
continuum chart. Start with the dominant causal-stress/radial-transport
profile, but do not use a mesh eigenvector as the definition.

Simultaneously lift it at N64/N128/N256 by minimizing the common-profile
defect subject to:

```text
exact plus/minus q_34 equality
maximum coordinate defect <= 2e-10
all physical and DAE gates
fixed pair orientation
common continuum normalization
no buffer contamination
```

### Phase 3 — Binding initial-profile gate

Before propagation require, for both state and fresh rate:

```text
N128/N256 signed cosine       >= 0.99
amplitude defect              <= 0.05
relative L2 profile defect    <= 0.10
```

Apply the same checks to N64/N128 diagnostically. If the state gate cannot
be met while maintaining exact equal coordinates, classify the unresolved
fiber itself as nonconvergent and stop.

### Phase 4 — Repeat only two history families

If Phase 3 passes, run only:

```text
production flux + production storage
linear flux + production storage
```

The storage-linear branches are proven redundant. Retain the WP10c8x
operator, safety, order, signed-cosine, zero-crossing, frequency, and
damping gates.

### Phase 5 — Mode-resolved diagnosis

If both histories still fail:

1. project the common initial perturbation onto the local discrete modes;
2. compare the significant N128/N256 mode subspaces;
3. report frequency, damping, radial wavelength, centroid motion, and
   cells per wavelength;
4. distinguish boundary closure error from bulk near-horizon dispersion.

No time shift may satisfy a binding same-time gate.

### Phase 6 — Architecture decision

- Common initial profile and convergent history: authorize one independent
  N512 local confirmation.
- Common profile but boundary-dependent nonconvergence: redesign the inner
  characteristic finite-volume closure.
- Common profile but boundary-insensitive underresolution: authorize a
  conservative embedded inner patch with permanent fine resolution.
- No cross-mesh common equal-coordinate fiber: reject this mode family as a
  continuum reduced-state diagnostic and construct a new physical
  perturbation basis.
- Distributed significant modes: move to the conservative staggered coarse
  finite-volume/PDE architecture.

No fixed-`Q` averaging, reduced coordinate selection, macrostep, tide,
wind, hot-state, or cycle calculation is authorized in WP10c8y.

## Machine evidence

```text
outputs/tables/causal_inner_boundary_consistency_audit_wp10c8x.json
outputs/tables/causal_inner_boundary_consistency_audit_wp10c8x_arrays.npz
outputs/checkpoints/causal_five_field_wp10c8x/
```

Runner:

```text
scripts/run_causal_inner_boundary_consistency_audit_wp10c8x.py
```

The final JSON records hashes for the runner, core DAE, core spatial audit,
primary arrays, and every cached operator.
