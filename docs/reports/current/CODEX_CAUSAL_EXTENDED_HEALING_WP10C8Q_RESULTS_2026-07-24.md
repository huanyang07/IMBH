# WP10c8q extended coordinate-fiber healing and slow-rate audit

Date: 2026-07-24
Base commit: `1e42839c094d3e7c2dc89e963a681e0004afa556`
Production physics changed: no
Production exact-max Rusanov flux changed: no
Production descriptor or BDF integrator changed: no
Moment ladder changed: no

> **Superseded rank interpretation:** WP10c8r subsequently established that
> the independent slow-rate cases used in the rank-two interface-4 SVD have
> absolute interface-4 half-spreads of only `2.65e-11` to `1.18e-8` gate
> units. Unit normalization removed that scientific amplitude information.
> The persistent original interface-4 direction remains certified, but the
> two-component interface-4 authorization below is withdrawn. See
> `CODEX_CAUSAL_INTERFACE_STATE_SUFFICIENCY_WP10C8R_RESULTS_2026-07-24.md`.

## Decision

WP10c8q closes three questions left by WP10c8p:

1. the decisive hidden mode is not a flux-gauge artifact;
2. it does not heal through the complete certified `0.125 s` window;
3. its complete interface-4 `M/J/E_K` response is not one-dimensional when
   an independent equal-coordinate direction is tested.

The binding classification is:

> `wp10c8q_persistent_localized_multimode_interface_state`

The existing 34-coordinate instantaneous leading slow-time equation remains
unclosed. A single prescribed interface flux or one scalar relaxation
amplitude is rejected for the tested state. The next admissible architecture
is a conservative interface-state vector localized initially at interface 4.
The six tested responses occupy a well-resolved two-dimensional transport
plane, so a two-coordinate prototype is justified; it is not yet a certified
production closure.

## Existing-evidence divergence audit

No new evolution is needed to reject the proposed flux-gauge interpretation.
The five-shell/four-internal-face incidence matrix has rank `4/4`. The initial
gate-normalized incidence ratios are:

| Mesh | Mass | Angular momentum | Killing energy |
|---|---:|---:|---:|
| N64 | `1.49965` | `1.47523` | `1.49953` |
| N128 | `1.51701` | `1.48827` | `1.51687` |

These are order unity, not near zero. Interfaces 3 and 4 have opposite signed
half-differences, so their contributions to the intervening shell divergence
add rather than cancel.

On loading-time slow time, the original equal-coordinate pair has maximum
34-coordinate rate half-differences `3.09033` at N64 and `2.92695` at N128.
The shell `M/J/E_K` redistribution also remains order unity per unit slow time.
For example, the N64 shell-3 values are

```text
(-0.592845, -0.659025, -0.593084)
```

and the matched N128 values are

```text
(-0.497954, -0.550090, -0.498136).
```

The complete shell ledgers reconcile these rates with internal boundary
transport. Source and responsive-height differences are negligible on the
frozen shell scales. The decisive hidden mode therefore causes real
conservative redistribution between retained shells.

## Exact perfect-fluid trace attribution

The finite interface-4 central-perfect-fluid difference is decomposed with a
path-integrated symmetric primitive-trace Jacobian. The endpoint difference,
component sum, and production flux agree with maximum relative defects
`4.80e-14` at N64 and `3.87e-14` at N128; quadrature defects are below
`3.14e-16`.

Both meshes identify the same controlling trace component:

> left reconstructed radial velocity divided by `c`.

When left and right copies are grouped by primitive, radial velocity accounts
for essentially the complete mass and Killing-energy difference and slightly
more than the angular-momentum difference, with the azimuthal contribution
providing the small compensating correction. Temperature and causal stress
are negligible in this attribution. This sharpens the physical diagnosis to
a perfect-fluid face-trace/throughflow ambiguity, not causal-stress memory or
Rusanov switching.

## Exact-history N64 extension

The exact WP10c8p coarse/fine plus/minus trajectories are replayed to
`0.025 s`, with their increment-primary BDF histories persisted. The four
states are then continued without another BDF1 startup to exact outputs at
`0.05`, `0.10`, and `0.125 s`.

The coarse continuations use 80 BDF2 steps; the fine continuations use 160
BDF2 steps. All four pass:

| Quantity | Worst value |
|---|---:|
| Scaled nonlinear residual | `9.9982e-12` |
| Scaled algebraic residual | `4.87e-13` |
| Discrete BDF ledger defect | `2.95e-11` |
| Cumulative physical-ledger defect, coarse | `1.10e-5` |
| Cumulative physical-ledger defect, fine | `2.72e-6` |
| Physical shell `M/J/E_K` ledger defect, coarse | `6.21e-4` |
| Physical shell `M/J/E_K` ledger defect, fine | `3.12e-4` |
| Flux reconstruction defect | `3.01e-15` |
| Fresh-rate directional stability defect | `1.18e-7` |
| Fresh-vector-field step defect | `1.71e-5` |

Every state and fresh-rate gate passes. Split continuation from `0.10` to
`0.125 s` is bitwise identical on both sides, with maximum state difference
exactly zero.

### No healing through `0.125 s`

The fine gate-normalized interface-4 transport history is:

| Time (s) | Mass | Angular momentum | Killing energy |
|---:|---:|---:|---:|
| `0` | `0.21680834` | `0.32452995` | `0.21723405` |
| `0.025` | `0.21680579` | `0.32452655` | `0.21723150` |
| `0.050` | `0.21680324` | `0.32452313` | `0.21722895` |
| `0.100` | `0.21679811` | `0.32451626` | `0.21722381` |
| `0.125` | `0.21679553` | `0.32451281` | `0.21722122` |

The controlling transport accumulates only `5.2813e-5` e-folds, against the
two-e-fold minimum required before fitting a relaxation law. Maximum temporal
uncertainty is `2.6164e-7` gate units. Final coordinate drift is
`4.5406e-7`, the interface-specific impulse fraction is `9.6116e-8`, and no
late regrowth appears.

The ambiguity remains localized: interface 4 controls, while the largest
secondary-interface response is only `0.08899` of the interface-4 maximum.
The original trajectory history remains rank one with
`sigma_2/sigma_1 = 1.13e-6`, but that fact applies to one fiber direction
only.

## Direct slow-rate fiber audit

WP10c8q also searches the exact 34-coordinate fiber for the direction
maximizing the complete gate-weighted rate ambiguity on

```text
T = t / t_load.
```

The N64 nullity is 286. All nonlinear lifts close the 34 coordinates below
`3.56e-15`, pass the physical/DAE/storage/fresh-rate gates, and retain a
significant slow-rate ambiguity.

### Amplitude ladder

The primary N64 direction gives maximum slow-rate half-spreads:

| Seed multiplier | Maximum spread per unit slow time |
|---:|---:|
| `5e-4` | `215.52076` |
| `1e-3` | `431.04155` |
| `2e-3` | `862.08307` |

The amplitude-linearity defect is `4.83e-8`.

### Independent direction, anchor, and mesh

The held-out direction is constructed in the weighted constraint-null space
orthogonal to the primary direction. Its weighted orthogonality defect is
`3.62e-18`, and it produces a maximum slow-rate half-spread `178.60984`.
The held-out `t=0.10 s` anchor gives `873.76870`; the unoptimized N128
prolongation gives `362.66519`. Every case is numerically admissible.

### Rank result

The same-anchor transport matrix contains the three primary amplitudes and
the independent held-out direction:

| Audit | `sigma_2/sigma_1` | `sigma_3/sigma_1` | Rank-one gate |
|---|---:|---:|---:|
| N64 same anchor | `0.57838` | `7.17e-5` | fail |
| N64/N128 primary pair | `0.63078` | `0` | fail |
| N64 primary/held-out anchor | `0.25490` | `0` | fail |
| All six cases | `0.80067` | `8.79e-5` | fail |

The second singular direction is order unity, while the third remains below
`1e-4` of the first. Thus one scalar transport amplitude is not sufficient,
but a common two-dimensional `M/J/E_K` transport plane is strongly supported
for the tested cases.

## Interpretation

The binding result is narrower than a continuum no-go theorem:

- the 34 shell moments do not determine the leading instantaneous slow-time
  rate on the certified N64/N128 truth discretizations;
- the missing response does not heal within `0.125 s`;
- it is localized near interface 4 and dominated by perfect-fluid face
  traces;
- at least two independent transport amplitudes are required at the primary
  anchor;
- the data do not supply a relaxation law, a long-time memory kernel, or a
  production reduced evolution.

The result does not yet require interface states at every shell face. It
authorizes one localized two-component prototype at interface 4, followed by
a new worst-case augmented-fiber search.

## Locked next plan: WP10c8r

### 1. Identify two physical interface coordinates

Apply the exact path-integrated trace attribution to both the primary and
held-out N64 pairs and their N128 confirmations. Determine which two
combinations of reconstructed density, radial velocity, azimuthal velocity,
enthalpy/temperature, and stress span the measured transport plane.

Prefer physical face variables over independently fitted `M/J/E_K` fluxes.
The first candidate should encode normal perfect-fluid throughflow; the
second must be selected from the measured held-out trace combination.

### 2. Build an audit-only conservative face state

Introduce

```text
Z_4 = (z_4,1, z_4,2)
```

only at interface 4. Construct the complete `M/J/E_K` flux from one common
physical face state. The identical flux must enter the two neighboring shell
equations with equal magnitude and opposite sign. Do not absorb responsive-
height storage into the face flux.

### 3. Derive dynamics; do not assume relaxation

No exponential decay was measured. Derive the two face-state rates from the
full local equations or a conservative fine-buffer problem. A fitted
Maxwell-Cattaneo or one-pole relaxation law is not authorized.

### 4. Repeat the exact nonlinear fiber audit

Augment the 34 coordinates with `Z_4` and repeat the worst-case complete
slow-rate search at:

- the primary N64 anchor;
- an independent held-out direction;
- the `0.10 s` N64 anchor;
- the prolonged N128 case.

The old primary and held-out pairs must no longer be equal in the augmented
state. Search for a new worst-case equal-`(q_34,Z_4)` counterexample rather
than checking only the old directions.

### 5. Architecture gate

- If the augmented fiber passes and the two-state dynamics are mesh
  consistent, authorize a short conservative interface-state evolution
  prototype through `0.125 s`.
- If another localized mode appears, enlarge the interface state only by the
  measured rank.
- If independent failures appear at several interfaces or in distributed
  radial structure, stop expanding the moment ODE and build a staggered
  conservative coarse finite-volume/PDE model.

No loading-time macrostep, tide, wind, hot-state, stability, or cycle search
is authorized by WP10c8q.

## Verification and evidence

- Focused tests: `56 passed`.
- Full repository: `705 passed, 4 subtests passed`.
- `git diff --check`: passed.

Primary artifacts:

- `outputs/tables/causal_extended_healing_wp10c8q.json`
- `outputs/tables/causal_extended_healing_wp10c8q_arrays.npz`
- `outputs/checkpoints/causal_five_field_wp10c8q/`

Final hashes:

- JSON:
  `3a6241cf95bd558f15277f8426f1b5b753d54b39e2c02babd67fb32f6bfc8c2d`
- arrays:
  `56b9adc7a747c7f597df0a52e07cdf02ae19c410171b4d9123c8e49052668c4c`
- slow-rate fiber JSON:
  `cb328f7d8cd1488753900f0ec41bebb2f6956dcbadfbed66cacf941f2c668764`
- slow-rate fiber arrays:
  `17f61c3f24d034ccc4b87b708c1367313bae58eaf5963300a8ee979830cc2365`
- minus replay:
  `3d4d07df8a07be7ba1b83fad798382f53660900760cae154f19e15a52070c11f`
- plus replay:
  `38abe6e1858e07da37e32be5ef5d8d26bef0652d4d350689368d4d8ae7457317`
