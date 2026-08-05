# Coarse third nonlinear duration-rung completion WP10c9d6c7c3b5c3d

## Scope

This package executes the definitions frozen by WP10c9d6c7c3b5c3c.  It
continues the committed coarse canonical-time base and generic-five-field
perturbed BDF2 histories from `2e-3 s` to `5e-3 s`.  It changes no physical
or numerical operator and no production default.

The main controller uses the one frozen integer-`100 us` target source and
`dt <= 4e-4 s`.  Independent serialized replay covers `4.4-5.0 ms`, and the
strict reference uses `dt <= 1e-4 s` over `4.8-5.0 ms`.

## Classification

`coarse_third_rung_completion_certified_remaining_third_rung_breadth_manifest_authorized`

Base stage passes: `True`.

Perturbed stage passes: `True`.

Strict response passes: `True`.

Authorized next: `WP10c9d6c7c3b5c3e_third_duration_rung_breadth_manifest`.

The third-rung breadth and spatial gates, fourth duration rung, fixed-Q experiments and reduced evolution remain blocked.

## Binding results

| Gate | Base | Perturbed |
|---|---:|---:|
| Main accepted comparisons | 8 | 8 |
| Replay accepted comparisons | 2 | 2 |
| Strict accepted comparisons | 2 | 2 |
| Rejected attempts | 0 | 0 |
| Maximum main local error | `9.5941e-8` | `9.6816e-8` |
| Maximum strict local error | `7.5655e-9` | `7.6335e-9` |
| Maximum scaled residual | `9.4980e-11` | `8.1457e-11` |
| Maximum mapped endpoint/path defect | `4.2810e-12` | `5.4601e-12` |
| Maximum export-ledger defect | `4.1806e-12` | `1.1143e-12` |
| Maximum discrete-ledger defect | `0` | `0` |
| Incoming excision characteristics | `0` | `0` |

Both serialized replay branches are bitwise identical to their direct
branches in target labels, primitive states, all 13 Tier-I exports, complete
primitive/mapped/responsive-height BDF histories, previous timestep, and
final restart payload.

The main-versus-strict perturbation-response comparison passes with

```text
maximum scaled state difference      1.8599699558308203e-10
maximum scaled Tier-I difference     4.6208075913218675e-11
state-history cosine                 1.0
Tier-I-history cosine                0.9999999999467827
```

The final states remain admissible.  The maximum `H/R` is `0.09850` for the
base and `0.09844` for the perturbed trajectory; minimum scattering optical
depths are `19.0101` and `19.0993`; reconstruction factors remain exactly
one.

## Interpretation

The corrected canonical-target coarse generic trajectory is certified
through `5e-3 s`, approximately one N128 cell-crossing clock.  This is the
first completed nonlinear duration rung at a physically meaningful fast
transport time.  It does not yet establish profile breadth or spatial
convergence at that horizon.

The historical WP10c9d6c7c3b5c2d classification remains formally failed and
is not amended.  Its one-ULP target-grid cause was isolated separately; this
package uses the prospectively frozen canonical target source throughout.

## Decision

Only a definitions-only third-rung breadth manifest is authorized next.  It
must freeze the remaining coarse held-out profiles and the middle/fine
generic spatial confirmation before any propagation.  The `2e-2 s` fourth
rung, nonlinear fixed-Q experiments, and reduced slow evolution remain
blocked.

## Reproducibility

Canonical evidence is stored under
`results/canonical/causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d/`.
The complete execution took `22224.52 s`.  The summary, configuration,
provenance, decisive arrays, and SHA-256 manifest are committed together with
the runner and focused evidence test.
