# Coarse held-out third nonlinear duration-rung screen WP10c9d6c7c3b5c3f

## Scope

This package executes the coarse held-out stage frozen by
WP10c9d6c7c3b5c3e. It changes no physical or numerical operator and no
production default. The original full-amplitude inward acoustic, outward
acoustic, buffered material, and inward shear-acoustic profiles are each
continued from their committed `37.5/40 us` BDF2 histories through `5e-3 s`.

Every profile uses the one canonical target source, a single tangent within
one process, the certified c3d base response by hash, a serialized
`4.4-5.0 ms` replay, and an independent `4.8-5.0 ms` strict shadow with
`dt <= 1e-4 s`. Durable per-profile caches are accepted only after the full
main/replay/strict stage passes.

## Classification

`coarse_heldout_third_rung_duration_breadth_certified_generic_spatial_confirmation_manifest_authorized`

Completed profiles: `p4__inward_acoustic, p4__outward_acoustic, p3_buffer45__material, p4__inward_shear_acoustic_mix`.

Failed profile: `None`.

Authorized next: `WP10c9d6c7c3b5c3g_third_duration_rung_spatial_confirmation_manifest`.

Middle/fine propagation, the fourth duration rung, fixed-Q experiments and reduced evolution remain blocked.

## Binding results

All four held-out profiles pass. Each main trajectory accepts 22 BDF
comparisons with zero retries; every replay and strict shadow accepts two
comparisons with zero retries.

| Profile | Maximum main local error | Maximum scaled residual | Maximum mapped closure | Strict state difference | Strict Tier-I difference |
|---|---:|---:|---:|---:|---:|
| Inward acoustic | `1.0112e-7` | `8.6192e-11` | `8.8572e-11` | `1.2883e-10` | `2.7224e-11` |
| Outward acoustic | `1.0083e-7` | `8.5036e-11` | `9.0328e-11` | `1.7688e-10` | `1.5493e-11` |
| Buffered material | `1.0127e-7` | `9.2707e-11` | `8.7080e-11` | `2.2940e-10` | `5.6383e-11` |
| Inward shear-acoustic mix | `1.0066e-7` | `8.5824e-11` | `7.8865e-11` | `2.0511e-10` | `1.8644e-11` |

Across the package:

- the worst main local-error estimate is `1.0127e-7`, below the frozen
  controller tolerance;
- the worst scaled nonlinear residual is `9.2707e-11`, below `1e-10`;
- the discrete ledger defect is exactly zero;
- the worst export-ledger defect is `5.8075e-11`;
- the worst mapped endpoint/path closure defect is `9.0328e-11`;
- the reconstruction factor remains exactly one;
- the maximum final `H/R` is `0.0985901`;
- the minimum final scattering optical depth is `19.0973`;
- no incoming excision characteristic appears.

Every serialized replay is bitwise identical to its direct branch in target
labels, primitive states, all 13 Tier-I exports, primitive/mapped/
responsive-height histories, previous timestep, and final restart payload.

The strict response comparisons remain far inside the frozen `5e-3`
difference envelope. State-history cosines are one to roundoff. The minimum
Tier-I-history cosine is `0.9999999994734261`.

## Interpretation

The corrected canonical-target coarse nonlinear trajectory is now certified
through approximately one N128 cell-crossing clock for acoustic, material,
mixed shear-acoustic, and the previously certified generic five-field
responses. This removes profile breadth as the remaining coarse-grid blocker
at the third duration rung.

The result is not a spatial certificate at `5e-3 s`. Only the coarse layout
has run for these held-outs, and the generic five-field response still needs
prospectively frozen middle/fine confirmation on active coupling faces
`96/192` with conservative common-parent state restriction and correct-face
Tier-I exports.

The historical WP10c9d6c7c3b5c2d classification remains formally failed and
is not amended. Its one-ULP target-grid cause was isolated separately; this
package uses the corrected canonical target source throughout.

## Decision

Only the definitions-only WP10c9d6c7c3b5c3g spatial-confirmation manifest is
authorized next. It may freeze middle/fine generic propagation, replay,
strict temporal shadows, spatial gates, and uncertainty routing, but may not
propagate a state.

The `2e-2 s` fourth duration rung is not yet authorized. Fixed-Q experiments,
reduced slow evolution, tide, wind, production promotion, and N1024 remain
blocked.

## Reproducibility

Canonical evidence is stored under
`results/canonical/causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_wp10c9d6c7c3b5c3f/`.
The complete four-profile execution took `84862.29 s`. The summary,
configuration, provenance, decisive arrays, SHA-256 manifest, runner, and
focused evidence tests are committed together. The focused suite reports
`3 passed`.
