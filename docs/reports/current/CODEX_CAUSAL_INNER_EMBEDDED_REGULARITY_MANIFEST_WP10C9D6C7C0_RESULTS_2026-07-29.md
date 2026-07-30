# Causal Inner Embedded Endpoint-Regularity Manifest

## WP10c9d6c7c0 — 2026-07-29

Analyzed base:

```text
fad76852220d7c304fff9016ff99ada64d404eff
```

Manifest:

```text
b230ce7a3c7e7546d0d706ee8f9bcfa3102c6c69be5f67a29aa451e1b5d9706b
```

## Binding classification

```text
endpoint_interface_regularity_manifest_frozen_
uniform_control_preflight_authorized
```

This definitions-only package freezes the next controls selected by the
negative WP10c9d6c7b result. It changes no operator and propagates no state.

The new controls may proceed only to the unchanged uniform state and
13-export preflight. Embedded propagation remains blocked until every new
profile passes that uniform contract.

Nonlinear physical evolution, production promotion, fixed-Q experiments,
and reduced slow-time evolution remain blocked.

## Preserved c7b result

WP10c9d6c7b remains formally rejected:

| Historical control | Instantaneous error cosine | Result |
|---|---:|---|
| `p3__inward_shear` | `0.78701` | fail |
| `p3__outward_shear` | `0.78979` | fail |
| `p5__inward_shear` | `0.98960` | pass |
| `p5__outward_shear` | `0.98992` | pass |

The new manifest cannot relabel those outcomes. It is designed to distinguish
the two live explanations for the `p3` failure:

1. insufficient zero-extension regularity at an active coupling stencil;
2. coupling-stencil activation itself, independent of the C2 profile class.

## Frozen factorized controls

Four new base profiles are frozen:

| Profile | Window | Zero-extension regularity | Support endpoint | Role |
|---|---|---|---:|---|
| `p4__inward_shear` | `sin^4(pi x)` | C3 | coupling face 48 | active endpoint |
| `p4__outward_shear` | `sin^4(pi x)` | C3 | coupling face 48 | active endpoint |
| `p3_buffer45__inward_shear` | `sin^3(pi x)` | C2 | parent face 45 | exact-zero buffer |
| `p3_buffer45__outward_shear` | `sin^3(pi x)` | C2 | parent face 45 | exact-zero buffer |

Both signs and amplitude factors `0.5` and `1.0` produce 16 binding
variants. Their formulas, projections, roles, thresholds, and decision table
are now immutable.

The controls are factorized:

- `p3 -> p4 -> p5` varies endpoint regularity at the same coupling surface;
- `p3 -> p3_buffer45` keeps C2 regularity but moves the support endpoint
  three parent cells inside the coupling face.

No taper, fitted coefficient, shifted coupling radius, or post-result
activity threshold is allowed.

## Uniform eligibility

All four new profiles pass the pre-existing N128 spectral, alias, endpoint,
purity, and projection gates:

| Profile class | `theta_99` | Alias fraction | Endpoint fraction | Global purity | Minimum active-cell purity | Projection defect |
|---|---:|---:|---:|---:|---:|---:|
| C3 `p4` shear | `0.24544` | `8.013e-4` | `3.760e-6` | `0.999999992` | `0.999997801` | `7.813e-13` |
| C2 buffered `p3` shear | `0.23317` | `7.182e-4` | `8.680e-5` | `0.999999992` | `0.999993732` | `7.111e-13` |

The inherited gates remain:

```text
theta_99                         <= 0.30
Nyquist alias fraction           <= 0.001
support endpoint fraction        <= 0.005
global selected-family purity    >= 0.995
active-cell family purity        >= 0.98
projection replay defect         <= 2e-12
```

The physical field scales replay exactly.

## Embedded definition and trace checks

All new projections restrict consistently across the three frozen embedded
layouts:

```text
maximum restriction defect          8.1603e-13
maximum exterior norm               0
maximum reconstruction-weight defect 5.7997e-16
```

The C3 endpoint controls remain demonstrably active at the coupling face:

| Layout | Maximum C3 coupling-trace fraction |
|---|---:|
| N128-equivalent inner | `1.1245e-5` |
| N256-equivalent inner | `1.1829e-6` |
| N512-equivalent inner | `1.0936e-7` |

All exceed the prospectively frozen `1e-10` trace-activity threshold.

The buffered C2 controls have:

```text
zero-buffer parent cells:  3
fine-grid buffer cells:     3 / 6 / 12
buffer norm:                0
left coupling trace:        0
right coupling trace:       0
```

They therefore isolate the effect of keeping the same C2 endpoint outside the
coupling reconstruction stencil.

## Frozen execution order

### WP10c9d6c7c1 phase 1 — uniform preflight

Propagate only the 16 new variants on the unchanged uniform
N128/N256/N512 grids. Require the existing:

```text
state/reference gates
13-export instantaneous and cumulative gates
exact semigroup-integral gate
sign/amplitude scaling
restart replay
```

If any new profile fails uniformly, stop. Do not run it on the embedded
layouts and do not change its definition.

### WP10c9d6c7c1 phase 2 — embedded discrimination

Only uniform passes may run on the same three c7a embedded layouts. Retain:

```text
RMS order                         >= 0.75
maximum order                     >= 0.75
significant-component order       >= 0.75
fine normalized difference        <= 0.05
history cosine                    >= 0.90
refinement-error cosine           >= 0.90
reference uncertainty/fine error  <= 0.10
```

Use the same active-domain 13 exports, common faces, shared coupling flux,
prefix ledgers, interface states, and characteristic-energy diagnostics.
The characteristic-energy relative-activity threshold remains `1e-8`; no
absolute reflection threshold is added.

## Binding decision table

| Prospective result | Authorized interpretation |
|---|---|
| C3 `p4` and buffered C2 `p3` both pass | Endpoint/interface regularity crossover; certify only the declared smoother or buffered class; no operator redesign |
| C3 `p4` fails but buffered C2 `p3` passes | Active endpoint/coupling-stencil hypothesis selected; local truncation audit only |
| C3 `p4` passes but buffered C2 `p3` fails | Short-support or global `p3` pre-asymptotic hypothesis; no interface redesign |
| Both prospective classes fail | No regularized embedded class selected; stop before operator change |

Association still will not prove causality. An operator intervention may be
considered only after a resolved, uniformly passing control exhibits a stable
local coupling defect.

## Stop gates

Do not:

- amend c7b or its `0.90` error-cosine gate;
- change any new window, support endpoint, activity threshold, or coupling
  radius after this manifest;
- skip the uniform preflight;
- propagate an ineligible subset while dropping a failed sibling;
- tune the interface operator to the historical `p3` result;
- begin nonlinear, production, fixed-Q, or reduced slow-time work;
- run N1024 as a rescue.

## Verification

```text
7 passed
```

Canonical evidence is stored in:

```text
results/canonical/
causal_inner_embedded_regularity_manifest_wp10c9d6c7c0/
```
