# Causal Matched Source and Compatible Startup WP10c5l-m Results

Date: 2026-07-18

## Verdict

The causal five-field model now passes two gates that were required before
extending no-tide evolution:

```text
WP10c5l matched source-on/source-off N16 control       PASSED
WP10c5l equal-time N16/N32 source-response mesh gate   PASSED
WP10c5m source-compatible N16/N32 initial data         PASSED
WP10c5m short adaptive N16/N32 startup                 PASSED
WP10c5m equal-time response mesh gate                  PASSED
bounded duration extension                             AUTHORIZED
long evolution, stability, hot state, or limit cycle  NOT CERTIFIED
tide and wind                                          NOT AUTHORIZED
```

The result removes the arbitrary `9.2e4`-times-supply inner throughput that
dominated WP10c5k. It does not establish a relaxed disk or a physical
ballistic stream-impact state.

## Physical Context

Both packages retain the same controlled problem:

```text
domain                         1.8-335 rg
stream center                  240 rg
stream logarithmic half-width 0.08
stream supply                  5 Mdot_Edd
source moments                 exact compact-C2 cell moments
outer Hill/Roche channel       energetically closed
tide                           off
wind                           off
```

The stream is the exact circularized regression fixture. It is not yet a
ballistic Layer-1 calibration.

## WP10c5l: Matched Source Control

### Contract

Source-on and source-off trajectories:

- begin from a bitwise-identical algebraic state;
- use the same sparse backend;
- accept exactly the same timestep history;
- retain separate endogenous thermal, geometric, stress, and boundary terms;
- compare the four prescribed Killing-equivalent stream moments only after
  those response-dependent terms are included.

For one step, the physical balance is

```text
storage + vertical storage + boundary transport
    - endogenous source - prescribed stream = 0.
```

The relaxing-stress field has no prescribed stream source. It remains an
explicit residual audit, but is not incorrectly classified as a fifth stream
moment.

### Results

| Quantity | N16 | N32 |
|---|---:|---:|
| Accepted paired steps | 8 | 5 |
| Equal physical time (s) | `1.72581e-7` | `1.72581e-7` |
| Rejected paired attempts | 0 | 0 |
| Maximum four-moment source-relative defect | `3.25291e-6` | `1.07613e-6` |
| Maximum four-moment balanced defect | `1.62646e-6` | `5.37954e-7` |
| Maximum isolated `Delta ln(H/R)` | `9.88672e-10` | `1.80491e-10` |
| Stored mass response / injected mass | `0.9999999984` | `0.9999999970` |

The N16/N32 differences are:

```text
stored-mass response / injected mass  1.38092e-9
inner mass-flux response / supply     2.68797e-11
outer mass-flux response / supply     0
maximum Delta ln(H/R) profile         1.02854e-9
```

All are far below their declared gates.

### Bounded Numerical Corrections

The first control run revealed two implementation issues:

1. The reconstruction helper consumes `ln R`, but the new matched audit passed
   `R`. That produced a spurious large thickness response even though the
   physical endpoint states were nearly identical. The caller now passes
   `ln R`.
2. With the ordinary repeated-step residual target, the radial source moment
   missed the unchanged `1e-4` recovery gate at `1.469e-4`. A trial at
   `1e-12` stagnated near the finite-difference noise floor and was stopped.
   A bounded `1e-10` paired target passed without changing the physical
   moment gate.

No tolerance was relaxed to accept the result.

## WP10c5m: Source-Compatible Initial Data

### Construction

The old preflight datum used

```text
Sigma_inner = 1e7 g cm^-2
T_inner     = 3e7 K
```

and drove an inner mass flux about `9.2e4` times the stream supply. Simply
rescaling `Sigma` while retaining `3e7 K` would produce `H/R` above 200 and
would leave the one-zone validity range.

The new datum therefore solves two independent scalar conditions:

```text
|Mdot_inner| / Mdot_stream = 1
H_inner / R_inner          = 0.1
```

The rest-mass face flux is linear in the inner surface density, so the first
condition is inverted directly. The second is solved by log-temperature
bisection without a floor or state clipping.

| Quantity | N16 | N32 |
|---|---:|---:|
| `Sigma_inner` (`g cm^-2`) | `108.8499774` | `108.8499774` |
| `T_inner` (K) | `4.13515e6` | `4.30749e6` |
| `Mdot_inner/Mdot_stream` | `-1.0000000` | `-1.0000000` |
| Maximum `H/R` | `0.1000000` | `0.1000000` |
| Minimum scattering depth | `18.5045` | `18.5045` |
| Inner incoming modes | 0 | 0 |
| Outer channel | closed | closed |
| Outer incoming responses | 2 | 2 |
| Initial algebraic-map residual | 0 | 0 |
| Consistency-system rank | `245/245` | `485/485` |

The reported rank is the scaled/equilibrated numerical rank. Raw unscaled
singular-value thresholds remain a conditioning diagnostic, not the
production rank decision.

### Sparse Initial Polish

The first N16 repeated run solved all eight states but missed the aggregate
mass gate:

```text
measured mass defect  1.68843e-10
required              <= 1e-10
```

The miss came from the dense reference first step, whose scaled residual was
about `3.0e-10`. The same physical step was polished with the certified sparse
backend at a `1e-10` target, and all subsequent source-compatible steps used
the same target. The fixed mass gate then passed; it was not relaxed.

### Repeated Startup Results

| Quantity | N16 | N32 |
|---|---:|---:|
| Accepted steps | 8 | 7 |
| Equal physical time (s) | `5.54201e-5` | `5.54201e-5` |
| Rejected attempts | 0 | 0 |
| Mass-ledger relative defect | `6.59048e-13` | `1.56858e-11` |
| Net stored mass / injected mass | `-3.62575e-4` | `-3.59752e-4` |
| Final `Mdot_inner/Mdot_stream` | `-1.0006436` | `-1.0006162` |
| Final maximum `H/R` | `0.0998980` | `0.0999363` |
| Minimum scattering depth | `>18.5` | `>18.5` |

The N16/N32 mesh differences are:

```text
net stored-mass response / injected mass  2.82274e-6
inner mass flux / supply                  2.74513e-5
outer mass flux / supply                  0
maximum Delta ln(H/R) response            1.00148e-3
RMS Delta ln(H/R) response                6.72757e-4
```

The response-profile gate is `5e-3`. The raw cell-center maximum thickness
differs by only `3.83e-4` relatively. A one-cell edge extrapolation gives a
larger non-gating reconstructed-maximum difference of `6.21%`; this remains
reported and must not be confused with the passing baseline-subtracted
response profile.

## Scientific Interpretation

The previous short trajectory was almost entirely arbitrary seed drainage.
The new datum instead begins with:

- inner throughput equal to the supplied stream rate;
- a closed outer mass channel;
- finite optical thickness;
- no incoming inner characteristic;
- moderate thickness;
- exact primitive and face maps;
- full scaled/equilibrated consistency rank.

Over the short certified interval, source and inner drainage nearly cancel:
the disk storage changes by only about `-3.6e-4` of the injected mass. This is
the first practical initial state from which a no-tide loading trajectory can
be extended without an immediate five-order-of-magnitude mass-flux mismatch.

This does not show whether the state relaxes, accumulates, cools, heats,
overflows, or becomes unstable at longer times.

## Locked Next Work

1. Restart from the certified WP10c5m N16/N32 states.
2. Extend N16 first to a geometric loading-time target near `1e-9 t_load`.
3. Preserve the exact sparse equations, `1e-10` source-compatible step target,
   mass/J/energy ledgers, closed-channel active set, optical-depth gate, and
   no-clipping rule.
4. Stop at the first failed physical or numerical gate. Do not add a new
   damping or tolerance scan.
5. If N16 passes, evolve N32 to the exact same physical time and compare the
   baseline-subtracted state response.
6. Increase duration geometrically only after the equal-time mesh gate passes.
7. Keep N64/N96, physical tide, wind, stability spectra, hot-state claims, and
   limit-cycle searches blocked until a practical no-tide duration ladder is
   established.

## Verification

```text
focused causal evolution/DAE tests  26 passed
full repository suite              485 passed, 4 subtests passed
repository hygiene                 passed for 624 tracked files
Python compilation                 passed
git diff --check                   passed
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-matched-source-control-audit

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-source-compatible-startup-audit
```

Machine-readable outputs and restart files are written under ignored
`outputs/` in accordance with the artifact policy.
