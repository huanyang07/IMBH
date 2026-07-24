# WP10c8s complete-rate nonlinear localization and fail-fast healing

Date: 2026-07-24

Base commit: `4a209ef0cdec7e835e5e61e0a518eb348989a65c`

Production physics changed: no

Production exact-max Rusanov flux changed: no

Production descriptor or BDF integrator changed: no

Moment ladder changed: no

New truth evolution run: yes, one matched N64 `h/h/2` natural-healing
experiment

## Decision

WP10c8s constructs exact finite-amplitude equal-`q_34` pairs for the
significant complete-rate modes identified by WP10c8r, localizes their
nonlinear slow-rate ambiguities, and applies the locked natural-healing gate.

The binding result is:

> `wp10c8s_single_interface4_route_rejected_by_independent_nonhealing_complete_rate_mode`

An independent inner-shell mode with negligible interface-4 transport
remains decisively above the healing gate at `0.025 s`. Its
uncertainty-inclusive final lower bound is

```text
16.5633 gate units
```

against the declared healing threshold

```text
0.10 gate units.
```

Therefore the hypothesis

> all significant complete-rate modes other than the original interface-4
> family heal, so `q_34` plus one interface-4 coordinate can close the reduced
> state

is rejected.

This is a fail-fast architecture result. It does **not** yet prove that a
staggered coarse PDE is required, establish permanent memory, or determine
the final number of localized states.

## Exact nonlinear equal-coordinate cases

Six independent physical families pass the exact nonlinear lift, physical
state, DAE, descriptor/storage, and fresh-rate contracts. Modes 0 and 4 reuse
the exact WP10c8q pairs; modes 1, 2, 3, and 7 are newly corrected onto the
finite-amplitude `q_34` fiber.

| Case | Controlling coordinate | Maximum nonlinear slow-rate half-spread |
|---|---|---:|
| mode 0, inner stress | shell-0 stress storage | `178.610` |
| mode 1, independent inner stress | shell-0 stress storage | `142.440` |
| mode 2, inner thermal | shell-0 mean `log(T)` | `71.2308` |
| mode 3, inner radial momentum | shell-0 radial momentum | `26.9498` |
| mode 4, middle stress | shell-1 stress storage | `431.042` |
| mode 7, source-shell stress | shell-2 stress storage | `98.8505` |

Every case is far above the predeclared `0.25` nonlinear significance gate.
The result confirms that the WP10c8r tangent spectrum corresponds to
finite-amplitude nonclosure rather than only to an infinitesimal
linearization.

## Cross-mesh tangent support

The matched N64/N128 tangent responses agree in direction and scale:

| Case | Absolute response cosine | N128/N64 maximum ratio |
|---|---:|---:|
| mode 0 | `0.999819` | `1.0497` |
| mode 1 | `0.999571` | `1.0509` |
| mode 2 | `0.999427` | `1.0090` |
| mode 3 | `0.999988` | `1.0602` |
| mode 4 | `0.999810` | `1.0579` |
| mode 7 | `0.983974` | `0.6495` |

All pass the locked cosine and amplitude-ratio gates. Mode 7 is the least
mesh-stable case and must not control an architecture decision without a
nonlinear N128 confirmation.

This is cross-mesh support for the selected tangent families. It is not an
N128 natural-healing result.

## Nonlinear localization

The cellwise state, primitive-rate, mapped-storage, responsive-height,
flux-divergence, and inferred-source differences are audited separately.

| Case | Dominant state shell fraction | Dominant rate shell fraction | Controlling dynamic term |
|---|---:|---:|---|
| mode 0 | shell 0: `0.969` | shell 0: `0.994` | flux divergence |
| mode 1 | shell 0: `0.964` | shell 0: `0.993` | mapped storage |
| mode 2 | shell 0: `0.923` | shell 0: `0.991` | mapped storage |
| mode 3 | shell 0: `0.939` | shell 0: `0.982` | flux divergence |
| mode 4 | shell 1: `0.539` | shell 1: `0.539` | flux divergence |
| mode 7 | shell 1: `0.336` | shell 0: `0.801` | mapped storage |

Modes 0-3 are strongly localized in the innermost retained shell. Mode 4 is
broader within the next shell. Mode 7 is distributed across state, rate, and
source/storage terms.

The main architectural correction from WP10c8r is now nonlinear:

> the missing complete-rate information is not confined to significant
> macro-interface-4 transport.

At least one independent inner-shell storage/transport state must be
resolved or eliminated by a demonstrated nonlinear healing mechanism.

## Frozen-tangent healing preflight

The frozen N64 tangent predicts final-to-initial rate ratios at `0.125 s` of

```text
mode 0: 0.00806
mode 1: 0.00832
mode 2: 0.00869
mode 3: 0.03076
mode 4: 0.96939
mode 7: 1.00717
```

This divides the selected directions into four apparently fast inner modes
and two apparently persistent modes. The preflight is explicitly
nonbinding.

The completed nonlinear mode-0 experiment contradicts the architectural
use of this prediction: the mode decays, but remains more than two orders of
magnitude above the healing gate at `0.025 s`. A frozen tangent cannot decide
whether a finite-amplitude mode is removable.

## Binding natural-healing experiment

The fail-fast case is `mode_0_inner_stress_existing`, an independent
complete-rate family with negligible interface-4 response.

The synchronized trajectories use:

```text
N64
duration       = 0.025 s
coarse         = 10 steps, dt = 0.0025 s
fine           = 20 steps, dt = 0.00125 s
plus/minus     = identical fixed schedules
startup        = one fresh BDF1 step per trajectory, then BDF2
```

All four trajectories pass their nonlinear, algebraic, state, discrete
ledger, and physical-ledger contracts.

| Diagnostic | Result |
|---|---:|
| Initial maximum slow-rate upper spread | `178.609843` |
| Fine final maximum spread | `17.91094` |
| Final upper spread | `19.25861` |
| Final lower spread | `16.56327` |
| Healing gate | `0.10` |
| Apparent controlling e-folds | `2.22725` |
| Maximum retained-coordinate drift upper bound | `2.23e-7` |
| Maximum coarse physical-ledger defect | `1.883e-3` |
| Maximum fine physical-ledger defect | `4.694e-4` |
| Maximum discrete-ledger defect | `4.30e-12` |

The coarse/fine difference does not satisfy the stricter curve-resolution
contract:

```text
maximum temporal uncertainty          = 8.00712
maximum relative temporal uncertainty = 0.29078
```

Consequently, WP10c8s does not certify a precise decay law or relaxation
time. It does certify the binary nonhealing decision because even the
conservative fine-minus-uncertainty lower bound,

```text
16.56327,
```

is separated from the `0.10` gate by a factor of about `166`.

The result is therefore classified as:

> `natural_healing_rejected_with_resolved_lower_bound_through_0.025s`

## Why the remaining healing matrix stopped

The mode-0 trajectory alone falsifies the single-interface-4 architecture
hypothesis. No result from modes 1, 2, 3, 4, or 7 can restore that route.

The completed four trajectories required:

| Trajectory | Function evaluations | Jacobians | Wall time |
|---|---:|---:|---:|
| coarse minus | `2407` | `51` | `1583 s` |
| coarse plus | `2407` | `51` | `1577 s` |
| fine minus | `4814` | `102` | `2981 s` |
| fine plus | `4767` | `101` | `2955 s` |

Running the full six-case matrix before revising the architecture would
consume substantial work without changing the locked binary decision. The
five unrun cases remain authorized diagnostic cases, not failed or healed
cases. No interrupted or incomplete trajectory is included in the evidence.

## What WP10c8s does and does not establish

Established:

- the six significant tangent families have exact finite-amplitude
  equal-`q_34` nonlinear counterparts;
- the families have matched N64/N128 tangent support;
- modes 0-3 are strongly localized in the innermost shell;
- mode 0 is independent of the significant interface-4 family;
- mode 0 does not heal to the declared gate by `0.025 s`;
- `q_34` plus only one interface-4 coordinate is insufficient.

Not established:

- a continuum nonclosure theorem;
- permanent memory of mode 0;
- a precise mode-0 relaxation time;
- natural-healing behavior of the five pending cases;
- the minimum final reduced-state dimension;
- whether localized extra states or a staggered coarse finite-volume/PDE
  model is preferable;
- any loading-time macrostep, tide, wind, hot state, stability, or cycle
  result.

## Locked next plan: WP10c8t

### 1. Preserve exact continuation history

The current trajectory caches preserve states but do not yet expose the
complete terminal `CausalFiveFieldBDFHistory` as a reusable continuation
artifact.

Replay the mode-0 fine plus/minus trajectories to `0.025 s`, persist:

- the exact terminal BDF history;
- the previous increment-primary increment;
- the previous timestep;
- state and history hashes.

Require bitwise equality with the committed endpoint. Do not reconstruct the
previous increment by subtracting large stored states.

### 2. Extend the binding inner mode

Continue the exact mode-0 pair without another BDF1 startup to:

```text
0.05 s
0.10 s
0.125 s
```

Use a matched `h/h/2` contract sized from the measured `0.025 s` uncertainty.
The purpose is to bound the nonlinear decay curve and accumulated
slow-coordinate impulse, not merely to repeat the binary `0.025 s` decision.

### 3. Confirm the architecture-controlling endpoint at N128

Run N128 only at the endpoint that changes the architecture decision.
Require matched localization, rate direction, accumulated ledgers, and a
mesh-supported persistence or healing classification.

### 4. Decide between localized and distributed state

- If mode 0 heals with a small bounded impulse, construct and validate an
  initial-slip/healed closure.
- If it remains one localized low-rank mode, introduce one measured inner
  state in addition to any independently required interface state.
- If several localized inner modes persist, retain only their measured local
  rank and rerun the exact augmented-fiber search.
- If persistent modes occupy several shells, sources, or interfaces, stop
  extending the moment ODE and implement a conservative staggered coarse
  finite-volume/PDE prototype.

Every augmented coordinate set must undergo a new worst-case finite-amplitude
equal-coordinate slow-rate audit.

### 5. Keep downstream physics blocked

Do not run a loading-time macrostep, tide, wind, hot-state, stability, or
cycle search in WP10c8t.

## Verification and evidence

- Focused WP10c8s tests: `7 passed`.
- Causal reduction regression matrix: `77 passed`.
- Full repository: `716 passed, 4 subtests passed`.
- Production code changed: no.
- Exact nonlinear pairs: six.
- New natural-healing cases completed: one N64 `h/h/2` case.
- N128 natural-healing confirmation: not run.

Primary artifacts:

- `outputs/tables/causal_complete_rate_healing_wp10c8s.json`
- `outputs/tables/causal_complete_rate_healing_wp10c8s_arrays.npz`

Final hashes:

- JSON:
  `d1b59d540d9da8a130235b2522488034dc0221b15a6cbf48e830a9e4c5c6eec0`
- arrays:
  `ef4e36b3c59f4f3e1219a14530bdc79d42a6f56eb3438232cc21a7e647c39715`
