# WP10c9c0c — Common-mode family transfer and local shear work
## Verdict

WP10c9c0c completes the two locked audits requested by WP10c9c0b:

1. an exact five-family decomposition of the unchanged WP10c8y common
   perturbation;
2. a cell-resolved physical-block work ledger for the unchanged WP10c9a pure
   inward-shear packet.

Every algebraic and instantaneous ledger contract passes. The scientific
localization gate does not:

```text
common_mode_failure_remains_multifamily_or_nonlocal
```

The common-mode fine-pair state and rate errors are largest in the propagated
outward-shear component at `0.125 s`. The coarse-pair rate error is instead
largest in inward shear. Meanwhile, the pure inward-shear refinement defect is
distributed across central perfect-fluid transport, mapped descriptor
dependence, geometry, and numerical dissipation. No physical block supplies
the predeclared `50%` absolute-activity dominance.

The radial profiles also disagree:

```text
common outward-shear error centroid       5.046 rg
pure inward controlling-work centroid     3.658 rg
profile cosine                            0.1739
required cosine                           0.90
```

The result rejects the idea that the WP10c8y common-mode failure can already
be explained by one localized version of the pure inward-shear defect.

Production remains unchanged. WP10c9c1, a path candidate, a new truth
trajectory, fixed-`Q` averaging, and reduced slow evolution remain
unauthorized.

## Frozen scope

The package reads and hashes:

- the WP10c8y common N64/N128/N256 states, rates, generators, amplitudes, and
  exact equal-coordinate lift;
- the WP10c9a ratio-1/2/4 pure inward-shear histories;
- the WP10c9c0b exact physical generator blocks.

It does not:

- alter the DAE, Rusanov flux, source split, descriptor, boundary, or BDF
  method;
- run a new nonlinear or frozen-linear truth trajectory;
- construct a new operator;
- change a scientific gate.

Matrix exponentials are used only to propagate the exact algebraic
decomposition of the already-committed frozen common-mode initial state.

## Five-family decomposition contracts

At every cell, the initial dimensionless primitive perturbation is decomposed
with the exact oblique principal projectors

```text
P_f = r_f l_f,
sum_f P_f = I.
```

Each component is propagated separately under the same unchanged full
generator. Linearity requires

```text
exp(Gt) x_0 = sum_f exp(Gt) P_f x_0.
```

The measured maximum defects are:

| Contract | Maximum | Gate | Pass |
|---|---:|---:|---|
| Projector identity | `8.88e-16` | `<=1e-10` | yes |
| Projector idempotence | `3.55e-15` | `<=1e-10` | yes |
| Cross-projector annihilation | `2.66e-15` | `<=1e-10` | yes |
| Initial decomposition | `8.98e-16` | `<=1e-12` | yes |
| State-history reconstruction | `5.23e-15` | `<=1e-12` | yes |
| Rate-history reconstruction | `1.83e-13` | `<=1e-12` | yes |
| Pairwise state Gram ledger | `1.95e-15` | `<=1e-12` | yes |
| Family cross-work ledger | `5.64e-15` | `<=1e-12` | yes |

The maximum local characteristic-basis condition number is `17.36`; the
maximum local-rest eigenpair defect is `2.19e-16`.

## Common-mode spatial result

The exact family decomposition reproduces the previously reported
underresolution:

| Quantity | N64/N128 defect | N128/N256 defect | Order |
|---|---:|---:|---:|
| State history | `0.17690` | `0.12681` | `0.4802` |
| Rate history | `0.32135` | `0.49792` | `-0.6317` |

At the controlling endpoint:

| Pair | Quantity | Largest error component | Signed fraction of total error norm squared |
|---|---|---|---:|
| N64/N128 | state | outward shear | `3.425` |
| N64/N128 | rate | inward shear | `2.861` |
| N128/N256 | state | outward shear | `1.556` |
| N128/N256 | rate | outward shear | `1.453` |

Fractions can exceed one because the five oblique-family error components
cancel. This is itself important: a large diagonal family error does not imply
that the total error is a one-family phenomenon.

The controlling family changes between the coarse and fine rate pairs. The
failure is therefore not a mesh-stable single-family defect under the declared
common physical chart.

## Initial family content and cross-work

The fine initial lifts have closely matched five-family energy compositions:

| Family | N128 | N256 |
|---|---:|---:|
| Inward acoustic | `0.16198` | `0.16183` |
| Inward shear | `0.31816` | `0.31814` |
| Material | `0.00779` | `0.00780` |
| Outward shear | `0.37432` | `0.37460` |
| Outward acoustic | `0.13775` | `0.13763` |

The common perturbation is therefore not a pure inward packet. It contains
comparable inward and outward shear content plus substantial acoustic
content.

For both N128 and N256 the largest final cumulative interaction is the
inward/outward-shear cross-work:

```text
N128  +11.98673   37.08% of absolute pairwise work
N256  +11.97581   37.04% of absolute pairwise work
```

This large term is balanced mainly by inward and outward shear self-work:

```text
N128 outward self-work  -7.21274
N128 inward self-work   -6.20040

N256 outward self-work  -7.20225
N256 inward self-work   -6.19146
```

The largest N128/N256 cumulative-cross-work difference is also the
inward/outward shear pair, reaching `0.03566` at `0.005 s`, but its final
difference is only `-0.01092` compared with a cumulative magnitude near
`11.98`.

Thus the shear-pair interaction is physically large, but it is not by itself
a demonstrated nonconvergent operator block.

## Pure inward-shear local work ledger

For each ratio and every output time, the selected inward-shear energy density
and block work close cell by cell:

```text
dE_i/dt = sum_k x_i^T H_i (G_k x)_i.
```

The maximum instantaneous block closure defect is `9.92e-16`. The
201-sample cumulative trapezoid defect is `2.60e-5`; this is not a method
failure. WP10c9c0b already demonstrated second-order audit-quadrature
convergence to `2.65e-7` with 801 samples. The binding localization here uses
the exact instantaneous decomposition and the much larger refinement work
differences.

The ratio-2/ratio-4 cumulative-work difference is largest at `0.039375 s`.
Its absolute block activity is:

| Exact block | Absolute activity | Fraction |
|---|---:|---:|
| Central perfect-fluid transport | `2.117e-3` | `0.3826` |
| Mapped descriptor dependence | `1.768e-3` | `0.3195` |
| Perfect-fluid geometry | `7.918e-4` | `0.1431` |
| Rusanov transport | `3.576e-4` | `0.0646` |
| Responsive-height source | `2.829e-4` | `0.0511` |
| Stress relaxation/principal source | `1.573e-4` | `0.0284` |

The controlling central-transport profile peaks at `3.531 rg`, with 80% of
its absolute activity between `2.547` and `4.699 rg`.

No block reaches the predeclared `0.50` dominance gate. The exact result is a
transport/descriptor/geometry cancellation, consistent with the negative
one-at-a-time ablation result of WP10c9c0b.

## Cross-audit comparison

The common-mode fine-pair rate error is controlled at `0.125 s` by propagated
outward shear. The pure inward work defect is controlled at `0.039375 s` by
central perfect-fluid transport.

After putting both profiles on the common physical radial interval:

```text
radial profile cosine = 0.17390 < 0.90.
```

The predeclared joint-localization gate requires:

1. at least `25%` signed family-pair contribution;
2. at least `50%` absolute block dominance;
3. radial-profile cosine at least `0.90`.

Only the first condition passes. The common-mode and pure-packet defects must
not be identified as the same localized mechanism.

## Scientific interpretation

WP10c9c0c supports four conclusions:

1. The common-mode failure is not a hidden algebraic decomposition error.
2. The common perturbation is intrinsically multifamily, with large
   inward/outward-shear cross-work.
3. The pure inward-shear selected-energy defect is generated by several
   compensating physical blocks, not one face or source term.
4. Correlating the pure inward packet with the common mode is insufficient to
   select an operator change.

This result does not prove that all operator changes are unnecessary. It
shows that the next audit must evaluate the exact physical generator blocks
on the **common trajectory itself**, retaining receiver/source family labels.

## Locked next package: WP10c9c0d

WP10c9c0d should remain cache-based and production-neutral.

1. Build or load the exact physical generator-block decomposition for the
   WP10c8y N64/N128/N256 common-mode grids.
2. Evaluate the four-index work ledger

   ```text
   W[f_receiver, f_source, block, time]
       = <x_f, G_block x_g>.
   ```

3. Require exact closure to the existing family cross-work and full
   common-mode energy-rate ledgers.
4. Restrict every block/family-pair work density conservatively between
   N64/N128/N256.
5. Identify the first time and radius at which the fine-pair rate error loses
   contraction, rather than using only the final maximum.
6. Apply absolute significance before normalizing a block or pair.
7. Test whether one block mediates the large inward/outward-shear interaction
   and whether the same block controls the outward-shear fine-pair error.
8. Keep the exact transport pieces separate:
   - inner boundary;
   - central perfect-fluid transport;
   - central stress transport;
   - Rusanov transport;
   - outer boundary.
9. Keep mapped and responsive-height descriptor dependence separate from
   cell-local sources.
10. Do not implement a path flux or another production candidate unless one
    mesh-stable, absolutely significant, radially localized common-mode
    block/family interaction passes a predeclared dominance gate.

If the common-trajectory ledger remains multi-block, close WP10c9c as a
negative low-dimensional operator-localization result. The next architecture
should then retain a converged localized inner micro-solver rather than tune a
single flux/source term against one packet.

## Reproduction

```text
PYTHONPATH=src:scripts \
python scripts/run_causal_inner_family_transfer_audit_wp10c9c0c.py

PYTHONPATH=src:scripts \
python -m pytest -q \
  tests/test_causal_inner_family_transfer.py \
  tests/test_causal_inner_family_transfer_audit_wp10c9c0c.py
```

Machine evidence:

- `outputs/tables/causal_inner_family_transfer_audit_wp10c9c0c.json`
- `outputs/tables/causal_inner_family_transfer_audit_wp10c9c0c_arrays.npz`
