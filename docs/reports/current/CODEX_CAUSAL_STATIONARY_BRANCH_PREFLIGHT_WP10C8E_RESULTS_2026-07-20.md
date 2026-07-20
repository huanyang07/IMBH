# WP10c8e Stationary-Branch Preflight

Date: 2026-07-20

Base commit under test:
`4247696c2c65039fc4c08d6aaca7cbace8be6636`

## Decision

The tested source-compatible seeds do not authorize a stationary root solve
or pseudo-arclength continuation:

```text
decision                         wp10c8e_stationary_anchor_solve_not_authorized
preflight mesh                   N16
source amplitudes                0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1
valid source-compatible seeds    0.1, 0.2, 0.5, 1
full-rank stationary Jacobians   4/4
physical damped Newton trials    0
N32 confirmation                 not authorized
full root/continuation           not run
```

This is a stop decision for the tested anchors. It is not a global
nonexistence theorem for every stationary solution of the causal equations.

## Low-Source Validity Boundary

The current source-compatible initializer scales the inner surface density
to match stream throughput. At zero source the requested surface density
vanishes. At source amplitudes `0.01`, `0.03`, and `0.05`, at least one
column leaves the optically thick domain required by diffusion cooling.

The present equations therefore do not supply the zero- or very-weak-source
anchor assumed by a conventional continuation from zero stream supply.
Changing the cooling closure to manufacture such an anchor is outside this
package.

## Integrated Ledger Preflight

At amplitudes `0.1-1`, the source-compatible seeds match the global mass
throughput at roundoff:

```text
mass-ledger relative defect      2.24e-16 to 1.79e-15
```

They are not stationary angular-momentum or energy states:

| Source amplitude | Angular-momentum defect | Killing-energy defect |
|---:|---:|---:|
| 0.1 | 0.8956 | 0.3621 |
| 0.2 | 0.8968 | 0.3374 |
| 0.5 | 0.8976 | 0.3225 |
| 1.0 | 0.8978 | 0.3175 |

The initializer was designed for causal source-compatible startup, not for
global steady torque and cooling balance. These large defects explain why a
local stationary Newton correction is severe despite exact mass matching.

## Stationary Rank And Conditioning

The exactly reduced primitive stationary Jacobian is full rank at every valid
amplitude:

| Source amplitude | Rank | Condition estimate | Maximum scaled correction |
|---:|---:|---:|---:|
| 0.1 | `80/80` | `5.25e10` | 180.3 |
| 0.2 | `80/80` | `4.39e10` | 69.2 |
| 0.5 | `80/80` | `3.25e10` | 122.8 |
| 1.0 | `80/80` | `2.34e10` | 202.2 |

The earlier stationary rank defect of the unreduced full DAE is therefore
not the active issue here. The reduced root map is locally invertible, but
its correction is very large and badly conditioned.

## Damped Predictor Result

Each amplitude uses exactly one reduced Newton direction and the fixed
damping ladder

```text
1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128
```

Large trials violate subluminal primitive constraints. Smaller trials can be
mapped back into a finite DAE state, but none passes all physical state gates.
At amplitudes `0.1` and `0.2`, the smallest trial changes the maximum scaled
stationary residual only to about `0.992` of its initial value; at larger
amplitudes even that trial increases the residual strongly.

There is therefore no admissible local descent direction from these
source-compatible seeds under the predeclared bounded predictor. N32 and a
full nonlinear stationary solve are correctly skipped.

## Interpretation

The result separates three statements:

1. the current causal stationary primitive map is full rank;
2. the startup initializer is far from angular-momentum and energy balance;
3. a local Newton continuation from that initializer is not viable.

It does not establish that no disconnected cool/warm stationary root exists.
Finding such a root would require a materially different global homotopy or
an independently supplied stationary anchor. The old hot-branch search and
the present result give no justification for another open-ended scan.

## Next Authorization

The stationary-branch route is closed from the tested anchors. The next
package may choose one bounded alternative:

1. an observable-specific, stability-preserving transfer realization of the
   certified full descriptor; or
2. an equation-free microburst pilot that projects by a small factor and is
   rejected immediately if off-manifold or ledger errors grow.

Any future stationary attempt requires an independent, physically valid
anchor and a predeclared homotopy. It may not restart from these failed
source-compatible predictors or relax optical-depth, velocity, or ledger
gates.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_stationary_branch_preflight_wp10c8e.json
SHA256 849bd92c2d80d1ceea0491af388dc6aa01513d3449c0e437b62b47814f829ae1
```

The full repository suite passes `573` tests plus `4` subtests.
