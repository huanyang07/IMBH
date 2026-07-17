# Causal Bounded Duration WP10c5n Results

Date: 2026-07-18

## Verdict

The source-compatible causal evolution reaches the bounded target separately
at N16 and N32, but it does not pass the declared equal-time mesh gate:

```text
N16 endpoint at 1e-9 t_load                         PASSED
N32 endpoint at the exact N16 physical time         PASSED
per-step numerical and physical gates               PASSED
endpoint characteristic, map, Roche, and rank gates PASSED
N16/N32 response mesh gate                          FAILED
further duration extension                          NOT AUTHORIZED
N64/N96, tide, wind, stability, hot/cycle searches  NOT AUTHORIZED
```

The controlling mismatch is the baseline-subtracted thickness response:

```text
measured max |Delta ln(H/R)_N16 - Delta ln(H/R)_N32|  1.25571e-2
declared gate                                          5.00000e-3
```

No tolerance was relaxed and no third resolution was launched.

## Scope

Both trajectories restart from the certified WP10c5m checkpoints and retain:

```text
coordinate/equations       one-domain ingoing-Kerr-Schild five-field DAE
stream                     exact compact-C2 circularized regression stream
stream supply              5 Mdot_Edd
outer Roche channel        energetically closed
tide                       off
wind                       off
nonlinear residual gate    1e-10
algebraic residual gate    1e-11
conservation gates         1e-10
```

N16 is evolved first to exactly `1e-9` of its initial loading time. N32 is
attempted only after N16 passes and lands at exactly the same physical time.

## Bounded Audit Corrections

Two audit semantics were corrected before the accepted run.

### Rectangular descriptor rank

The `5N x (15N+5)` descriptor has structurally zero algebraic columns.
LAPACK `dgeequ` therefore cannot equilibrate that rectangular matrix and
reported a zero-column code. The descriptor gate is its declared full row
rank. The square consistency matrix retains the scaled and equilibrated full
rank gate.

This changes no equation, state, rank threshold, or evolution tolerance.

### Algebraic map scaling

The evolved face maps contain cancellations among dimensional terms as large
as about `1e32`. Their raw absolute residual can therefore be about `1e11`
while the scaled algebraic residual remains near roundoff. WP10c5n uses the
existing production algebraic gate:

```text
maximum scaled algebraic residual <= 1e-11.
```

The raw dimensional value remains a non-gating diagnostic. At the WP10c5m
N16 checkpoint the two values are:

```text
raw absolute map residual     3.86547e10
scaled algebraic map residual 3.42462e-15
```

## Per-Resolution Results

| Quantity | N16 | N32 |
|---|---:|---:|
| Exact elapsed time (s) | `6.781724319e-4` | `6.781724319e-4` |
| Loading-time fraction | `1.00000e-9` | `1.08173e-9` |
| Extension accepted steps | 87 | 50 |
| Rejected attempts | 0 | 0 |
| Aggregate mass defect | `7.91e-13` | `4.44e-12` |
| Max five-field ledger defect | `1.61e-12` | `2.31e-12` |
| Final `Mdot_inner/Mdot_stream` | `-1.007867` | `-1.007530` |
| Final `Mdot_outer/Mdot_stream` | `0` | `0` |
| Extension storage/injected mass | `-4.29864e-3` | `-4.14643e-3` |
| Final maximum `H/R` | `0.0987564` | `0.0992248` |
| Final minimum scattering depth | `18.6773` | `18.6707` |
| Inner incoming modes | 0 | 0 |
| Outer incoming responses | 2 | 2 |
| Roche channel open | no | no |
| Descriptor rank | `80/80` | `160/160` |
| Consistency rank | `245/245` | `485/485` |
| Equilibrated consistency condition | `8.42e6` | `7.49e6` |

The N16 first continuation step replays bitwise. Both final restart files also
round-trip bitwise.

## Mesh Comparison

The conserved and boundary responses agree well:

```text
extension storage/injected-mass difference  1.52207e-4
inner flux/supply difference                3.37132e-4
outer flux/supply difference                0
final maximum H/R relative difference       4.72047e-3
minimum optical-depth relative difference   3.54489e-4
exact elapsed-time defect                    0
```

The shape response does not:

```text
maximum Delta ln(H/R) response difference  1.25571e-2
RMS Delta ln(H/R) response difference      8.33686e-3
required maximum                           <= 5.0e-3
```

The largest discrepancy is broad rather than a one-cell edge artifact. It is
centered around `12-16 rg`; at `15.04 rg`,

```text
N16 Delta ln(H/R)  -2.01926e-2
N32 Delta ln(H/R)  -7.63551e-3
difference         -1.25571e-2
```

The checkpoint-to-endpoint response gives the same diagnosis, with maximum
difference `1.15556e-2`.

## Interpretation

WP10c5n does not show a characteristic failure, Roche opening, loss of
optical thickness, rank loss, nonlinear breakdown, or conservation problem.
Both meshes evolve smoothly and remain close in global mass flux, thickness
maximum, and optical depth.

It also does not certify a continuum trajectory. The profile response differs
too much under the declared N16-to-N32 refinement.

One confounder must be removed before interpreting this as spatial
nonconvergence. WP10c5m constructs each mesh independently by imposing
`H/R=0.1` at that mesh's first cell center. Consequently,

```text
N16 inner temperature  4.13515e6 K
N32 inner temperature  4.30749e6 K
```

and `make_causal_five_field_seed()` interpolates between the moving first and
last cell centers. The two meshes therefore do not sample one common
continuum primitive profile. Baseline subtraction removes the initial
profile difference from the plotted response, but it does not make the two
evolution operators act on identical physical initial data.

This is a plausible explanation for part of the broad inner-disk mismatch.
It is not yet demonstrated to be the complete explanation.

## Locked Next Work

The next package must address only the common-data question:

1. Define one analytic primitive profile using fixed physical radial anchors,
   not each mesh's first and last cell centers.
2. Use one common inner thermodynamic state. Do not retune temperature
   independently by resolution.
3. Make the inner face treatment and the order-unity stream-throughput target
   consistent with that common profile.
4. Require N16/N32 common-radius primitive agreement at initialization,
   alongside zero inner incoming modes, optical thickness, a closed Roche
   channel, exact scaled maps, and full consistency rank.
5. Repeat only the short WP10c5m equal-time startup first.
6. Repeat WP10c5n only if that short common-data gate passes.
7. Keep N64/N96, longer duration, tide, wind, stability, hot-state, and
   limit-cycle searches blocked.

Do not reinterpret the failed `5e-3` gate, add a mesh-dependent fitted
profile, or launch a third evolved resolution.

## Verification

```text
focused causal evolution/DAE tests  26 passed
full repository suite              485 passed, 4 subtests passed
repository hygiene                 passed for 625 tracked files
Python compilation                 passed
git diff --check                   passed
```

## Reproduction

WP10c5m must be run first because ignored restart artifacts are not stored in
Git:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-source-compatible-startup-audit

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-source-compatible-duration-audit
```

Machine-readable output:

```text
outputs/tables/causal_five_field_source_compatible_duration_wp10c5n.json
```
