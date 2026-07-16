# Global Source-Free Refinement WP6a-R Results

## Scope

This work package repeats the bounded source-free relaxation on `N=96` at
the exact shared physical times used by `N=64`, then applies one conservative
`N=96 -> N=128` remap to the last exact state. The remap is a classification
snapshot only: it performs no time advance and no nonlinear solve.

The stream source is zero throughout. The inner boundary is the causally
outgoing plunge at `4.5 rg`; the outer Hill/Roche channel uses the certified
closed/choked contract.

## N96 Exact-Time Results

| Reference `t/t_load` | Inner mass/supply | Inner J/source | Inner E/source | max `H/R` | `L_v/Delta R` | `L_v/H` |
|---:|---:|---:|---:|---:|---:|---:|
| `2e-7` | -0.175716 | -0.040980 | -5.01049 | 0.1411645 | 2.190 | 2.947 |
| `5e-7` | -0.179124 | -0.041779 | -5.10459 | 0.1411648 | 2.006 | 2.587 |
| `1e-6` | -0.189012 | -0.044089 | -5.38379 | 0.1411658 | 1.698 | 2.021 |

All three milestones are immutable and checksummed. The Hill/Roche edge stays
closed, with normalized available energy near `-0.08641`, and the maximum
accepted storage-scaled ledger defect is `4.52e-16`.

The relaxation gate fails. From `2e-7 -> 5e-7`, the maximum fixed-radius Mach
change is `1.308` and `|Delta ln T|=0.0403`. From `5e-7 -> 1e-6`, the changes
worsen to:

```text
inner mass flux / supply       0.00989
relative angular flux          0.0553
relative total-energy flux     0.0547
fixed-radius Mach              8.367
fixed-radius |Delta ln Sigma|  0.0375
fixed-radius |Delta ln T|      0.1718
relative max H/R               7.50e-6
```

The nearly unchanged global thickness therefore does not indicate local
relaxation. The innermost plunge state continues to reorganize and the
controller remains in the causally disconnected first cell at `90.3%` of its
relative-thickness limit.

## N64/N96 Comparison

At exact `1e-6 t_load`, refinement improves the local resolution measure from

```text
N64: L_v/Delta R = 0.900, two cells inside the sonic radius
N96: L_v/Delta R = 1.698, three cells inside the sonic radius
```

but it does not give mesh-independent primitives. At that same physical time:

```text
inner mass-flux difference / supply = 0.00647
relative angular-flux difference    = 0.0331
relative energy-flux difference     = 0.0299
relative max-H/R difference         = 4.14e-4
maximum fixed-radius Mach difference= 4.078
maximum fixed-radius |Delta lnSigma|= 0.0707
maximum fixed-radius |Delta lnT|    = 0.0969
```

Thus the global ledgers and thickness are stable while the narrow plunge
profile is not.

## One-Shot N128 Classification

A new log-radius overlap remapper preserves every cell-integrated conserved
total. Its `N=96 -> N=128` defects are at most `1.17e-16`; the remapper is
covered by conservation tests.

The remap does not certify a finer evolved solution. It exposes the unresolved
profile dependence:

```text
inner mass-flux difference / supply = 0.00478
maximum fixed-radius Mach difference= 10.34
maximum fixed-radius |Delta lnSigma|= 0.0412
maximum fixed-radius |Delta lnT|    = 0.255
N128 remap L_v/Delta R              = 2.807
```

The Roche edge remains closed and the remapped totals are exact, but the
primitive reconstruction is strongly representation-sensitive in the plunge.
This is a failed mesh-independence gate, not an `N=128` physical trajectory.

## Physical Interpretation

The source-off experiment is a counterfactual diagnostic, not a candidate
global equilibrium. It retains nonzero inward throughput while imposing no
mass source and zero Roche overflow. Its disk mass must therefore drain:

```text
dM_disk/dt = -Mdot_inner != 0.
```

No amount of source-free time relaxation can turn that state into a nontrivial
global steady reference. The useful result of WP5/WP6 is narrower: the early
inner evolution is independent of the distant stream and is dominated by the
mapping/discrete-operator adjustment.

## Decision

1. Do not freeze any source-free milestone as a production reference.
2. Do not start the stream ramp from the draining high-throughput source-free
   control.
3. Close further source-free continuation; the optional `2e-6` extension was
   stopped after the exact `1e-6` milestone because the relaxation metrics
   were worsening and each `0.0095 s` step required roughly two minutes.
4. Preserve all exact milestones and the N128 remap snapshot as negative
   evidence.

## Superseding WP6b Result

The planned global source-balanced steady projection was audited and rejected
as physically incompatible before implementation. With a closed Roche edge,
zero outer torque, no tide, and no wind, the domain has no outlet for the
roughly `84%` of supplied mass that does not initially accrete, nor for its
angular momentum.

WP6b instead projected only the causally outgoing supersonic plunge. The local
N64/N96 roots pass residual and rank gates, but the source-on hold passes only
at N64; N96 exceeds the fixed-radius Mach-drift gate. The result is documented
in:

```text
CODEX_GLOBAL_INNER_PLUNGE_PROJECTION_WP6B_RESULTS_2026-07-15.md
```

The next initializer is therefore one solver-generated low-throughput remnant
disk with `|Mdot_inner|/Mdot_stream<=0.01`, followed by matched N64/N96
source-off/source-on holds. Do not return to source-free relaxation or another
projection-width variant. Wind and physical tide remain blocked by that
initialization gate.

## Verification

```text
targeted global evolution tests: 56 passed
full repository suite:             369 passed, 4 subtests
N96 milestone ledgers:            <= 4.52e-16
N96 exact milestones:             3 checksummed files
N128 remap conservation:          <= 1.17e-16
```

Machine-readable diagnostics:

```text
outputs/tables/global_source_free_relaxation_N96.json
outputs/tables/global_source_free_refinement_snapshot_N128.json
```
