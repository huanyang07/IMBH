# Unified Source-Band Certification Results

Date: 2026-07-11

> **Superseded numerical gate:** This report records the source-only and
> ordinary-Jacobian result. The subsequent block-Jacobian/bordered work in
> `CODEX_UNIFIED_BLOCK_JACOBIAN_CONTINUATION_RESULTS_2026-07-11.md` certifies
> exact-source anchors through `eta_E=9.6703`. The source and quadrature audits
> below remain valid.

## Scope

This work tests the low-launch-energy `Mdot_inner/Edd=5`, `Rout=335 rg`,
`Rinj=240 rg`, `f_s=0.30`, `epsilon_w=0.20` unified conservative branch.
It deliberately keeps the existing wind physics fixed.

The certification gate is **rejected**. Exact source transport is implemented
and verified, but the `eta_E=10,8,7` states do not reach the exploratory
`3e-5` residual threshold.

## Implemented Numerical Infrastructure

1. The compact stream mass is integrated analytically in every cell from its
   cumulative primitive.
2. Stream angular-momentum and energy moments use the same exact mass increment
   with the current constant `l_s` and `B_s` capture closure.
3. One shared interval transport operator now serves production and tests.
4. Stream, wind, radiation, external torque, and external power integrals are
   retained as separate diagnostic fields.
5. A Gauss-Legendre transport audit supports arbitrary fixed quadrature order.
6. A source-only grid replaces only the compact support while preserving all
   inherited inner and outer nodes.
7. A multidomain grid preserves the first 12 sonic nodes, allocates an explicit
   source block, and refines the broad inner/outer disk.
8. The certification runner performs a frozen-source local correction, a
   separate inner/sonic correction, and a final global polish.

The production residual no longer differentiates the known compact source and
then numerically reintegrates it.

## Exact Source and Quadrature Audits

For every tested grid,

```text
sum exact stream increments / imposed stream supply = 1.000000000000
```

At 64 source nodes, the old Simpson source normalization differs from the
exact value by only `9.62e-9 Mdot_inner`. Thus source normalization is no
longer capable of explaining an `O(1e-4)` residual.

On the best `eta_E=10`, 64-source-node state, normalized transport differences
are:

| contribution | Simpson vs 8-point max | 8-point vs 16-point max |
|---|---:|---:|
| mass | `3.24e-7` | `3.38e-9` |
| angular momentum | `3.44e-7` | `3.60e-9` |
| energy | `2.05e-7` | `2.09e-9` |

The quadrature differences peak near `R=7.7-8.0 rg`, not in the source band.
Transport quadrature is therefore not the remaining source-interface floor.

## Source Resolution Results

The source-only sweep at `eta_E=10` gives:

| requested source nodes | total nodes | best global max | dominant behavior |
|---:|---:|---:|---|
| 24 | 386 | `1.18e-4` | source/right-interface angular row |
| 32 | 394 | `1.05e-4` | source/right-interface angular row |
| 48 | 410 | `9.70e-5` | left-interface angular row |
| 64 | 426 | `9.13e-5` | broad left-interface angular row |

At 64 source nodes the source-band maximum is `6.02e-5`; the first radial
interval is `7.22e-5`. Increasing the source halo from two to eight inherited
nodes worsens the global maximum to `1.38e-4` and moves the peak to the new
right halo edge. This is defect export, not convergence.

A broad `N=512` multidomain check with 64 source nodes gives:

```text
global maximum       = 9.96e-5
source-band maximum  = 6.27e-5
first radial interval= 9.96e-5
```

Uniformly adding broad-domain nodes therefore does not satisfy the gate.

## Low-Eta Results

The 64-source-node checks give:

| eta_E | global max | source max | first interval | dominant family |
|---:|---:|---:|---:|---|
| 10 | `9.13e-5` | `6.02e-5` | `7.22e-5` | broad angular momentum |
| 8 | `1.33e-4` | `9.48e-5` | `1.33e-4` | inner radial momentum |
| 7 | `2.52e-4` | `1.19e-4` | `2.52e-4` | inner radial momentum |

The `eta_E=8` and `eta_E=7` failures are not solely source-band failures.

## Residual Localization

For the best `eta_E=10` state:

- mass residual is broad and peaks near `R=16-17 rg` at `5.14e-5`;
- angular residual spans approximately `R=205-216 rg` and peaks at
  `9.13e-5`;
- radial residual peaks in the first interval at `7.22e-5`;
- energy residual remains below `7.3e-6`;
- energy compatibility remains below `2.93e-5`.

The remaining error is consequently a broad state-collocation/correction
problem. It is not an unresolved compact-source integral.

## Newton Audit

Field-specific finite-difference scales reduce the best `eta_E=10` maximum to
`7.52e-5`, mainly by improving radial rows, but then terminate on `xtol` with a
large reported gradient. Smaller finite-difference steps do not remove the
floor.

The square production Jacobian is structurally full rank but numerically
singular at this checkpoint. Direct sparse factorization reports exact
singularity. A row/column-scaled least-norm Newton step requires a line-search
factor of only `2.44e-4` and does not materially reduce the mass/angular floors.

## Scientific Interpretation

The source-transport repair is successful, but the low-`eta_E` branch is not
certified. The previous wording "source-band quadrature bottleneck" is now too
narrow. The active obstruction is a coupled broad conservative
state-collocation/Jacobian problem, with an additional inner radial defect that
grows as `eta_E` decreases.

No wind-power or terminal-Bernoulli physics change was introduced because the
precondition for that step was not met. The result still supports the earlier
physical conclusion: lowering `eta_E` changes wind mass per available power
without producing a distinctly hotter disk state.

## Recommended Next Numerical Step

1. Derive block-local analytic Jacobian entries for the conservative mass and
   angular-momentum rows, including local wind derivatives.
2. Add row/column conditioning and a bordered continuation corrector so the
   numerical near-null direction is represented explicitly.
3. Re-test the exact-source `eta_E=10` state at N426 before any new wind
   closure is introduced.
4. If the analytic bordered system still has no strict root, classify the
   current fixed-inner positive-flux steady formulation as discretely
   inconsistent at low launch energy and move to the absolute-supply,
   signed-flux boundary problem.

## Verification

```text
203 passed, 4 subtests passed
```

Reproduction entry point:

```text
scripts/run_unified_conservative_source_band_certification.py
```

Generated checkpoints and tables remain under ignored `outputs/` paths.
