# Causal Spatial Balance WP10c7i Results

Date: 2026-07-19

## Verdict

WP10c7i passes its method-level contract.

The remaining WP10c7h spatial error was not fixed by a constant
reference-state residual correction. It required two state-dependent
consistency upgrades:

1. admissibility-preserving quadratic primitive traces at every causal
   Rusanov face, including one-sided physical-boundary traces;
2. four-point cell source quadrature whose shear and height rates are
   evaluated from the same local reconstructed path as the thermodynamic
   state.

Together with measure-weighted cell storage, the selected operator reduces
the N32/N64 full-domain `d log(H/R)/dt` discrepancy from
`2.37228699 s^-1` to `0.0993839131 s^-1`.

```text
full-domain reduction                           23.8699x
required full-domain reduction                  20x
15-60 rg reduction                              13.4970x
required 15-60 rg reduction                     10x
full and diagnosed observed order               2.36087
required order                                  1.8
projected bounded-horizon difference             0.00152799
maximum pre-trajectory budget                    0.0025
```

The general high-order operator is therefore sufficient. No
baseline-specific flux or source correction is retained.

WP10c7i did not run a disk trajectory. It authorizes exactly one bounded
N32/N64 fixed-BDF2 trajectory with fresh operator-compatible histories.

## Locked Scope

The legacy defaults remain unchanged:

```text
spatial reconstruction                          piecewise_constant
physical boundary trace                         cell_centered
cell rate scheme                                arithmetic_face
cell source quadrature                          midpoint
cell storage quadrature                         midpoint
```

The selected opt-in WP10c7i configuration is:

```text
spatial reconstruction                          quadratic_admissible
physical boundary trace                         plm_one_sided
cell rate scheme                                arithmetic_face
cell source quadrature                          gauss_legendre_4_local_rates
cell storage quadrature                         gauss_legendre_4
```

No Riemann solver, characteristic boundary map, prescribed stream moment,
physical source, thermodynamic closure, DAE count, or temporal method is
changed.

## Ablation Result

All rows use the same source-compatible continuum state, conservative
N64-to-N32 restriction, and exact DAE-consistent tangent.

| Variant | N32/N64 full tangent | Projected endpoint | Full order | Reduction |
|---|---:|---:|---:|---:|
| Current smooth PLM | `2.37229 s^-1` | `0.0364730` | `1.34935` | `1.000x` |
| One-sided boundary trace | `1.34138 s^-1` | `0.0206232` | `2.17191` | `1.769x` |
| Higher-order cell rates | `2.64036 s^-1` | `0.0405945` | `1.19489` | `0.898x` |
| Four-point state/source quadrature | `2.25326 s^-1` | `0.0346430` | `1.29173` | `1.053x` |
| Measure-weighted storage | `2.94730 s^-1` | `0.0453135` | `0.30375` | `0.805x` |
| All prior PLM consistency upgrades | `0.424969 s^-1` | `0.00653372` | `2.59511` | `5.582x` |
| Quadratic faces, cell-centered rates | `0.272512 s^-1` | `0.00418976` | `0.90553` | `8.705x` |
| Smooth PLM, local reconstructed rates | `0.424681 s^-1` | `0.00652931` | `2.59650` | `5.586x` |
| Quadratic faces and local rates | `0.0993839 s^-1` | `0.00152799` | `2.36087` | `23.870x` |

Only the final row passes the amplitude, order, and reduction gates.

## Error Classification

The ablation identifies three distinct effects.

First, one-sided reconstruction removes the old first-cell boundary maximum,
but it does not change the interior thermodynamic discrepancy.

Second, measure-weighted storage and four-point endogenous-source integration
must be used together. Either change alone moves the controlling error without
closing the transport/source cancellation.

Third, with quadratic faces but cell-centered source rates, the N32/N64
horizon mismatch is `0.272512 s^-1`; its vertical-work component is
`0.266261 s^-1`. The prior Gauss rule varied the thermodynamic state inside
the cell while holding `d log(H)/d tau` fixed at the cell center. Evaluating
the shear and height rates from the same reconstructed path removes that
inconsistency and moves the final maximum outside the horizon, near
`16.3242 rg`.

This is a state-dependent discretization repair. A constant baseline
correction was tested during development, improved only the zeroth-order
residual, and left the controlling tangent essentially unchanged. It was
removed before certification.

## Reconstruction Contract

The selected face operator reconstructs the complete primitive chart:

```text
(ln Sigma, beta_R, beta_phi, ln T, specific causal stress)
```

Each cell uses a spacing-aware quadratic in `ln(R)` for its two face traces.
The physical endpoints use a smooth four-point trace when four cells are
available. One coupled factor scales the complete candidate chart toward its
cell state until both traces satisfy the causal admissibility and physical
outer-boundary contracts.

For a primitive profile exactly quadratic in `ln(R)`, all left and right face
traces agree with the analytic face values to the test tolerance. A separate
test deliberately activates the coupled admissibility factor and verifies
that every reconstructed velocity remains subluminal.

## Source And Storage Contract

The four-point rule is normalized to the exact Kerr-Schild cell measure.
Positive thermodynamic charts are reconstructed at each quadrature node.
For the selected local-rate mode, covariant shear and `d log(H)/d tau` are
calculated from a centered `2e-5` log-radius directional derivative along
that same reconstructed path.

The prescribed compact-C2 stream remains a cell-integrated external source
and is not requadratured. Its four moments are bitwise unchanged between the
legacy and selected operators.

The primitive map stores a measure-weighted cell average when the selected
storage quadrature is active. The increment-primary conserved variable
therefore remains the primary physical cell storage; no subtraction of large
unmapped states is reintroduced.

## DAE And Jacobian Gates

The selected consistency systems remain full rank:

```text
N16                                             245/245
N32                                             485/485
maximum scaled consistency defect               1.42109e-14
maximum tangent reconstruction defect, N32       9.58093e-10
```

The widened sparsity includes quadratic face stencils, physical-boundary
admissibility coupling, source reconstruction, and measure-weighted primitive
maps.

At N4:

```text
Jacobian dimensions                              65 x 65
declared nonzeros                                1305
color count                                      23
maximum omitted dense derivative                 0
maximum colored/dense relative defect            0
```

The N16-N128 residual/JVP audit uses a `2e-4` chart perturbation, selected
from a recorded step ladder to avoid cancellation noise. Its maximum
term-reconstruction relative defect is `7.55788e-8`, below `2e-7`.

Every algebraic map closes exactly, and exact-stream recovery has zero
relative defect in the machine evidence.

## Verification

```text
focused causal evolution/DAE/spatial tests       59 passed
complete repository suite                        549 passed
complete repository subtests                     4 passed
Python byte compilation                          passed
git diff whitespace check                        passed
```

## Evidence

Runtime evidence remains ignored by repository policy:

```text
outputs/tables/causal_spatial_balance_wp10c7i.json
SHA256 4117d618aa7a955bc03828de7b9fc4201ba533648d484c9226332de3614dfff2

outputs/tables/causal_spatial_balance_wp10c7i_arrays.npz
SHA256 34d8a9540beb6e5c62e1a2ab6a609af4e7c73fa4e64d928fb60cbc650077d781
```

The canonical JSON reports:

```text
general high-order repair sufficient             true
reference-state fluctuation operator required    false
invariants passed                                true
overall WP10c7i method result                    true
```

## Authorization

The next bounded package may:

1. generate fresh N32 and N64 source-compatible states with the selected
   operator;
2. build fresh BDF1/BDF2 histories rather than reuse PLM checkpoints;
3. run the exact WP10c7h physical interval at fixed S32/S64;
4. keep raw temporal `Delta log(H/R)` uncertainty below `2.5e-4`;
5. require the exact N32/N64 response difference to satisfy `0.005`;
6. preserve all state, rank, source, ledger, and bitwise-restart gates;
7. retain smooth PLM as the direct A/B control.

No N128 trajectory, longer duration, tide, wind, stability, hot-state, or
cycle calculation is authorized before that bounded trajectory passes.
