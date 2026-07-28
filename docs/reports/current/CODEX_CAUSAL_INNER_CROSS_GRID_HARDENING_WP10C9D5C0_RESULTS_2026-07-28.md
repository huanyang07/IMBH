# WP10c9d5c0 Cross-Grid Frozen Derivative and Metric Hardening

Date: 2026-07-28

Analyzed base: `9c2a4ac6fa464a43fbaed3318cf5e1233a70fe55`

## Binding classification

WP10c9d5c0 selects:

```text
cross_grid_derivative_or_physical_sensitivity_failed_extended_localization_blocked
```

The cross-grid directional-derivative gate fails. The fail-fast contract
therefore blocks:

- derivative-choice physical-export propagation;
- direct face-flux parity and stride audits;
- WP10c9d5c1 extended/grouped localization;
- WP10c9d5c2 self-consistent space-storage tangent;
- frozen recertification, nonlinear work, fixed-Q averaging, and reduced
  slow evolution.

The rejected WP10c9d5 candidate and the WP10c9d5b Branch-D decision remain
unchanged. This package does not reinterpret either result.

## Scope

The audit uses the exact committed WP10c9d5 replay states and tests the
candidate stationary residual on all three embedded grids:

- fixed N128 exterior with N128-equivalent inner resolution;
- fixed N128 exterior with N256-equivalent inner resolution;
- fixed N128 exterior with N512-equivalent inner resolution.

Rows are certified through `5`, `8`, and `12 rg`, both with and without the
complete three-cell reconstruction halo.

The matched continuum directions are:

- the exact common mode;
- two smooth global-inner fields;
- two near-excision fields;
- all five first-cell primitive coordinate directions.

The smooth fields use a cell-volume-weighted scaled norm. The original
max-component normalization is not used for certification.

For every direction the audit compares:

```text
stored 4e-5 matrix action
centered directional actions at 5e-6, 1e-5, 2e-5, 4e-5, and 8e-5
Richardson estimates from the 1e-5/2e-5 and 2e-5/4e-5 pairs
```

The predeclared gates are:

| Gate | Limit |
|---|---:|
| Stored-matrix action defect | `5e-5` |
| Centered `2e-5 -> 4e-5` change | `2e-5` |
| Fine/coarse Richardson difference | `2e-5` |

No threshold or direction normalization was changed after inspecting the
result.

## Metric correction

`causal_radial_history_convergence` now distinguishes:

- `history_cosine`, comparing medium and fine physical histories;
- `error_cosine`, comparing the coarse-medium and medium-fine refinement
  errors.

It also accepts fixed physical component scales. A true relative activity
threshold is then formed from those scales instead of multiplying the
requested threshold by `float.tiny`.

The historical cosine field names remain as compatibility aliases so that
the committed WP10c9d5b evidence is not changed.

## Cross-grid derivative result

The table reports the worst value over all six certified regions for each
non-coordinate direction:

| Grid | Direction | Result | Matrix defect | `2e-5 -> 4e-5` | Richardson difference |
|---|---|---:|---:|---:|---:|
| N128 inner | common mode | pass | `3.1953e-5` | `6.9733e-6` | `1.8529e-5` |
| N128 inner | global inner 0 | **fail** | `5.9737e-5` | `3.4652e-6` | `1.0281e-5` |
| N128 inner | global inner 1 | pass | `2.2476e-5` | `2.2552e-6` | `7.2899e-6` |
| N128 inner | near excision 0 | pass | `2.6642e-6` | `1.2476e-7` | `3.9636e-7` |
| N128 inner | near excision 1 | pass | `1.8689e-5` | `8.2525e-7` | `9.4853e-7` |
| N256 inner | common mode | **fail** | `1.9292e-5` | `9.5711e-6` | `2.0727e-5` |
| N256 inner | global inner 0 | **fail** | `1.0642e-4` | `1.4065e-5` | `4.0882e-5` |
| N256 inner | global inner 1 | **fail** | `5.7026e-5` | `9.5855e-6` | `2.6831e-5` |
| N256 inner | near excision 0 | pass | `4.8182e-6` | `5.7345e-7` | `1.6498e-6` |
| N256 inner | near excision 1 | pass | `3.3141e-5` | `1.0106e-6` | `2.9157e-6` |
| N512 inner | common mode | **fail** | `1.6976e-5` | `1.1876e-5` | `3.4696e-5` |
| N512 inner | global inner 0 | **fail** | `1.8266e-4` | `5.2543e-5` | `1.4937e-4` |
| N512 inner | global inner 1 | **fail** | `6.4226e-5` | `3.3192e-5` | `1.1547e-4` |
| N512 inner | near excision 0 | pass | `9.3590e-6` | `2.2181e-6` | `6.2557e-6` |
| N512 inner | near excision 1 | **fail** | `6.1569e-5` | `2.9942e-6` | `1.0566e-5` |

All five first-cell coordinate directions pass on every grid. Their
stored-matrix defects are between about `5e-14` and `2e-11`, and their
centered/Richardson differences remain well below the declared limits.

This is an important positive control: the failure is not a generic
first-cell coordinate derivative defect. It appears in matched smooth
multi-cell directions, grows with inner resolution, and affects the exact
common mode on the medium and fine grids.

## Spatial localization of the derivative uncertainty

Postprocessing the committed Richardson arrays shows that the discrepancy is
not the outermost-domain effect found by WP10c9d5a:

- N128/global-inner-0 has `66.0%` of its squared discrepancy through `5 rg`
  and `96.95%` through `12 rg`;
- the failing N256 common/global directions have `58.1-87.9%` through
  `5 rg` and effectively all of it through `12 rg`;
- the failing N512 common/global directions have `64.3-81.5%` through
  `5 rg` and effectively all of it through `12 rg`;
- the failing N512 near-excision direction has effectively all of its
  discrepancy through `5 rg`.

The largest cells are often between roughly `1.8` and `2.7 rg`, while the
global directions also contain smaller contributions extending toward
`10 rg`.

Thus the refined audit finds a genuine inner-domain, resolution-dependent
derivative uncertainty. The earlier scoped N128-inner certificate cannot be
extrapolated to the N256- and N512-equivalent grids.

## Fail-fast decision

Because the directional derivative stage fails:

```text
physical_sensitivity.executed = false
wp10c9d5c1_extended_localization_authorized = false
```

No alternative generators were propagated, no recovery radius was inferred,
and no grouped block attribution was performed. Running those stages would
attach physical meaning to a refinement order whose medium and fine
generators do not yet have independently certified derivatives.

## Required next work

The next package must repair and independently certify the frozen derivative
construction before localization resumes. It should:

1. localize the failed smooth-direction JVP differences by face, cell,
   residual field, and physical residual block;
2. distinguish nested finite-difference principal-basis noise from colored
   simultaneous-perturbation error;
3. construct a second derivative using analytic/AD local maps,
   frozen-projector or ordered real-QZ subspaces, or a rigorously certified
   adaptive high-order sparse difference;
4. require independent derivatives to agree on the common and held-out
   smooth directions on all three embedded grids;
5. only then propagate the common physical exports and reconsider
   WP10c9d5c1.

This is not authorization to tune the finite-difference step, stationary
speed tolerance, path, trace, or one physical block.

## Reproducibility

Canonical evidence is stored under:

```text
results/canonical/causal_inner_cross_grid_hardening_wp10c9d5c0/
```

The package contains configuration, all decisive directional arrays,
provenance, the complete summary, and SHA-256 checksums. The binding run took
`1832.57 s` on Python `3.12.13`, NumPy `2.3.5`, and SciPy `1.18.0` with
Apple Accelerate BLAS/LAPACK.
