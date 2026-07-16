# Transonic Sonic-Gradient WP4 Results

**Date:** 2026-07-14
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Starting commit:** `35dbd2f`
**Scope:** bounded sonic-offset, ODE-tolerance, regular-root, accepted-resolution,
and early-evolution audit. No physical equation or acceptance gate was changed.

## Verdict

WP4 passes for its intended mapping scope.

The finite first-interval sonic-gradient mismatch decreases from

```text
accepted 96/64 coupled root:    0.376287878
accepted 144/96 coupled root:   0.188309395
ratio, fine/coarse:             0.50044
```

while the physical stationary plunge and matched early global evolution are
already insensitive to the accepted coupled-root resolution. The mismatch is
therefore retained as a converging derivative diagnostic, not evidence for a
different plunge trajectory.

WP5, the source-on/source-off tendency and matched-trajectory control, is now
the next work package.

## Correction Found During The Audit

The conservative mapper continued the plunge with
`context.base.inner_params`. In an open-overflow eigenvalue solve, that object
can retain the seed accretion rate, while the accepted rate lives in
`evaluation.trial_context.inner_params`.

This distinction is large when the accepted `96/64` open state is evaluated
directly from the wall-based context. It is small but nonzero for the chained
`144/96` context:

```text
inherited 96/64 rate:   1.3828602442e22 g/s
accepted 144/96 rate:   1.3820690226e22 g/s
relative difference:    5.72e-4
```

The mapper now uses the accepted trial parameters and explicitly verifies
that the mapped plunge carries `evaluation.mdot_inner`.

The old scalar sonic-root scan also used a fixed interval around `a=0`. The
nullspace coordinate origin can shift by thousands even when the physical
gradient barely changes. The scan is now centered on the resolved outer
gradient. This changes only the root search coordinate; the L'Hopital
regularity condition and production equations are unchanged.

## Stationary Audit

Both accepted coupled roots have two regular sonic derivative branches. The
selected inward branch remains close to the first resolved outer interval;
the alternate branch remains more than `23.5` gradient units away.

| Source root | Selected gradient | Mismatch | Alternate distance |
|---|---:|---:|---:|
| `96/64` | `(-10.27626, 2.90704)` | `0.37629` | `23.5263` |
| `144/96` | `(-10.36213, 2.93465)` | `0.18831` | `23.9404` |

At `4.5 rg`, coarse minus fine gives:

```text
radial Mach difference:        -0.02630
Delta ln Sigma:                -0.003142
Delta ln T:                    -0.000547
```

At `5.0 rg`, the corresponding values are `-0.01706`, `-0.008386`, and
`-0.002186`. Both inner faces have zero incoming radial characteristics.

Changing the sonic offset from `1e-6` to `1e-5` or `1e-7`, and tightening the
IVP tolerances from `(1e-9,1e-11)` to `(1e-11,1e-13)`, changes the stationary
`u`, `T`, and `Sigma` arrays by at most `5.2e-8` in logarithmic norm.

## Matched Early Evolution

Both accepted roots were conservatively mapped to the same `N=64` global
mesh and evolved to the exact shared time

```text
0.0015197296624436523 s
```

which is `1e-9` of the fine-root `N=64` loading time. Each run required one
accepted step, no retries, and fewer than nine nonlinear evaluations.

| Quantity | `96/64` source | `144/96` source | Difference |
|---|---:|---:|---:|
| Inner mass flux / supply | `-0.1534083` | `-0.1531550` | `-2.53e-4` |
| Inner angular flux | `-2.06073e42` | `-2.05754e42` | `0.155%` relative |
| Inner total-energy flux | `6.50060e41` | `6.48419e41` | `0.253%` relative |
| Maximum `H/R` | `0.1412868` | `0.1411050` | `0.129%` relative |
| Disk mass change | `8.666e-10` | `8.468e-10` | `1.98e-11` absolute |

At fixed `4.65`, `4.75`, and `5.0 rg`, the Mach differences are `-0.0253`,
`-0.0230`, and `-0.0192`. The emergent sonic radii differ by about
`0.0090 rg`.

Both runs are controlled by the relative-thickness change in cell zero. That
cell is supersonic, has no incoming radial characteristic, and is causally
disconnected from the outer disk. The controller uses only about four percent
of its unchanged two-percent limit at this audit time.

The maximum storage-scaled ledger defects are `4.49e-16` and `1.92e-16`.

## Decision

The declared WP4 outcomes were:

1. decreasing mismatch: certify the mapping;
2. finite mismatch with invariant flux/evolution: keep it as a derivative
   diagnostic;
3. material trajectory change: redesign the initialization.

The result satisfies the first two. The corrected stationary and global
states do not show a material trajectory change. No further sonic-offset,
tolerance, branch, or mapping scan is authorized.

The existing WP3 exact-common-time files predate the converged-rate mapper
correction. Their scientific mesh conclusion is supported by this audit, but
they should not be used as bitwise restart references after the correction.
All future runs must regenerate their initial state with the corrected mapper.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/run_transonic_sonic_gradient_audit.py \
  --output outputs/tables/transonic_sonic_gradient_audit.json \
  --maximum-nfev 600
```

The runner writes content-addressed stationary seeds and a SHA-256 manifest
under `outputs/checkpoints/sonic_gradient_audit/`. Adaptive early-evolution
milestones are immutable and checksummed by the shared WP1 checkpoint
contract.

## Next Step

Proceed to WP5 at `N=64`:

1. evaluate instantaneous source-on, source-off, and source-only tendencies
   from one corrected immutable initial state;
2. run matched source-on/source-off trajectories to the same physical time;
3. compare inner `M/J/E`, fixed-radius plunge state, sonic radius, `H/R`, disk
   mass, thermal energy, and controller location;
4. classify early evolution as mapping relaxation, source driven, or mixed.

WP6 long extension, tide, and wind remain blocked until WP5 is interpreted.
