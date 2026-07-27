# WP10c9c0d — Direct common-mode block-by-family attribution

## Verdict

WP10c9c0d applies the exact four-index physical transfer ledger directly to
the unchanged WP10c8y N64/N128/N256 common-mode histories:

```text
W[block, receiver family, source family, time]
    = <x_receiver, G_block x_source>.
```

All generator-decomposition, receiver-action, and cross-work contracts pass.
The scientific localization gate does not:

```text
common_mode_defect_remains_multiblock_after_direct_ledger
```

The large inward/outward-shear interaction is physically mediated mainly by
the inner-boundary transport block on every mesh:

```text
N64   59.29%
N128  56.96%
N256  56.74%
```

That fact does not localize the refinement failure. The N128/N256 defect in
the same interaction is divided almost equally between:

```text
central perfect-fluid transport   45.67%
inner-boundary transport          45.57%
```

The outward-shear-source rate error is instead led by central perfect-fluid
transport into outward shear, but that individual term supplies only
`26.82%` of the absolute block/receiver activity. The full common rate error
is still more distributed: its largest individual term supplies only
`6.60%`.

Thus the block that mediates the physical shear exchange is not the unique
block that controls either its cross-mesh defect or the outward-shear rate
error. No single face, source, descriptor term, or path-flux change is
selected.

WP10c9c1 remains unauthorized. Production, truth trajectories, fixed-`Q`
averaging, and reduced slow evolution remain unchanged and closed.

## Frozen scope

The package uses only:

- the frozen WP10c8x production generators and exact base physical rates;
- the WP10c8y common N64/N128/N256 amplitudes and histories;
- the WP10c9c0c exact five-family decomposition;
- the physical residual blocks introduced and certified in WP10c9c0b.

It does not:

- alter the DAE, numerical flux, source split, descriptor, boundary, or BDF
  method;
- run a new nonlinear or frozen-linear trajectory;
- implement a path-conservative candidate;
- change any scientific gate.

## Native-coordinate decomposition requirement

The WP10c8x generator was constructed in its own frozen primitive scaling.
The mapped-storage rate derivative is a nested finite difference. Therefore
it is not numerically valid to change the primitive scaling first and then
recompute the nested derivative with the same dimensionless step: that
changes the physical perturbation size.

WP10c9c0d consequently uses the following exact order:

1. decompose each cached WP10c8x generator in its original primitive
   coordinates;
2. use the exact cached WP10c8x base physical rate in the storage-rate
   derivative;
3. close every native physical block to the native generator;
4. similarity-transform each completed block into the WP10c8y common
   coordinates.

This is recorded as:

```text
native_wp10c8x_decomposition_then_exact_similarity_rescale
```

Applying the operations in the opposite order produces a large artificial
storage-derivative remainder and is rejected as an audit construction.

## Generator decomposition contracts

The exact physical block schema is:

- inner-boundary transport;
- central perfect-fluid transport;
- central stress transport;
- Rusanov transport;
- outer-boundary transport;
- perfect-fluid geometry;
- radiative cooling;
- stream source;
- stress geometry;
- stress relaxation;
- responsive-height/vertical work;
- mapped descriptor rate dependence;
- responsive-height descriptor rate dependence;
- an explicitly measured numerical remainder.

The maximum contract measurements over N64/N128/N256 are:

| Contract | Maximum | Gate | Pass |
|---|---:|---:|---|
| Base residual reconstruction | `2.92e-15` | `<=1e-10` | yes |
| Stationary Jacobian reconstruction | `1.03e-11` | `<=2e-10` | yes |
| Final transformed generator reconstruction | `2.12e-16` | `<=1e-12` | yes |
| Sparse mass-solve defect | `9.49e-16` | `<=1e-10` | yes |
| Transformed unattributed generator fraction | `1.63e-9` | `<=1e-7` | yes |

The transformed unattributed fractions are:

| Mesh | Unattributed fraction |
|---|---:|
| N64 | `1.17e-9` |
| N128 | `1.30e-9` |
| N256 | `1.63e-9` |

They are far too small to control the result.

## Four-index ledger contracts

For every physical block, source family, receiver family, mesh, and output
time, WP10c9c0d evaluates the receiver-projected vector action

```text
r[b, f, g] = P_f G_b x_g
```

as well as the scalar cross-work

```text
W[b, f, g] = <x_f, G_b x_g>.
```

The maximum exact-closure measurements are:

| Contract | Maximum | Gate | Pass |
|---|---:|---:|---|
| Receiver-action closure | `6.58e-14` | `<=1e-10` | yes |
| Block/family cross-work closure | `7.89e-15` | `<=1e-10` | yes |
| Reproduction of WP10c9c0c parent cross-work | `1.37e-15` | `<=1e-10` | yes |
| Five-projector identity | `8.88e-16` | `<=1e-10` | yes |

The reconstructed N64/N128 and N128/N256 rate-error vectors close below
`1.95e-13` and `3.21e-13`, respectively. The conclusion is therefore not a
missing family or physical block.

## Full common-mode rate-error attribution

The unchanged total rate-history errors are:

| Pair | Maximum relative rate error | Controlling time |
|---|---:|---:|
| N64/N128 | `0.32135` | `0.125 s` |
| N128/N256 | `0.49792` | `0.125 s` |

For both refinement pairs, the largest single
block/receiver/source contribution is:

```text
block     = central perfect-fluid transport
receiver  = material
source    = inward acoustic
```

Its absolute-activity fractions are only:

| Pair | Fraction | Absolute significance |
|---|---:|---:|
| N64/N128 | `0.06558` | `2.987` |
| N128/N256 | `0.06600` | `1.816` |

The term is absolutely significant but nowhere near the predeclared `0.50`
dominance gate. Its maximum fine-pair activity fraction over the full history
is only `0.06712`.

The first time the fine error ceases to contract relative to the coarse error
is:

```text
t = 0.04125 s.
```

At that onset, the same component is largest, but its activity fraction is
only `0.05901`. The component remains multi-block by construction; its
declared `50%` persistence fraction is zero.

The controlling radial centroids agree closely:

```text
N64/N128 profile centroid   4.1942 rg
N128/N256 profile centroid  4.1804 rg
relative centroid defect    0.00329
```

However, the full radial activity-profile cosine is only:

```text
0.89187 < 0.90.
```

Even without that marginal profile failure, the dominance and persistence
gates fail decisively.

## Inward/outward-shear mediation

For the two directed interactions

```text
inward receiver  <- outward source
outward receiver <- inward source,
```

the final cumulative absolute activity by physical block gives:

| Mesh | Leading block | Activity fraction | Inward<-outward | Outward<-inward |
|---|---|---:|---:|---:|
| N64 | Inner boundary | `0.59286` | `+6.5569` | `+6.3048` |
| N128 | Inner boundary | `0.56956` | `+8.1408` | `+8.0046` |
| N256 | Inner boundary | `0.56738` | `+8.3462` | `+8.2789` |

This is a real and mesh-stable physical statement: the excision-side
transport mediates most of the large shear-pair exchange.

It is not an error-localization statement. The N128/N256 cumulative
interaction difference is:

| Block | Absolute defect fraction |
|---|---:|
| Central perfect-fluid transport | `0.45668` |
| Inner-boundary transport | `0.45568` |
| Rusanov transport | `0.04653` |
| Stress relaxation | `0.02454` |
| Mapped descriptor dependence | `0.01456` |

No block reaches `0.50`, and the two leading blocks nearly cancel. Modifying
the inner boundary alone would therefore tune a physical mediator without
capturing the equally large neighboring transport defect.

## Outward-shear-source error

The source-conditioned outward-shear histories provide the direct test
locked by WP10c9c0c:

| Pair | Relative rate error | Leading block/receiver | Activity fraction |
|---|---:|---|---:|
| N64/N128 | `0.64742` | Central perfect -> outward shear | `0.16857` |
| N128/N256 | `0.61329` | Central perfect -> outward shear | `0.26816` |

Central perfect transport is stable as the leading term, but it remains one
part of a strongly cancelling block/receiver decomposition. It does not pass
the `0.50` dominance gate.

Most importantly:

```text
physical shear-pair mediator      = inner-boundary transport
fine shear-pair defect leader     = central perfect transport (45.67%)
outward-shear rate-error leader   = central perfect transport (26.82%)
```

The same block does not control all three objects.

## Scientific decision

WP10c9c0d closes the WP10c9c localization sequence as a negative result:

1. the common trajectory is intrinsically multifamily;
2. its large inward/outward-shear exchange has a stable physical mediator;
3. the cross-mesh error is nevertheless a coupled inner-boundary/central-
   transport/descriptor/source cancellation;
4. no single common-mode interaction passes all significance, dominance,
   persistence, and radial-convergence gates.

This result does **not** say that the inner boundary is irrelevant. It says
that the evidence does not authorize changing that boundary, the central
perfect flux, the Rusanov term, a descriptor term, or a path term in
isolation.

The binding decisions are:

```text
WP10c9c1 path candidate             not authorized
single production operator change  not authorized
new truth trajectory               not authorized
fixed-Q averaging                  not authorized
reduced slow evolution             not authorized
```

## Recommended next architecture gate

The next target should no longer be a one-term repair or a fitted scalar
inner mode. The viable reduction candidate is a conservative localized inner
micro-solver that exports only mesh-converged slow observables to the outer
model.

Before any constrained fast experiment, run a cache-first observable
preflight:

1. Freeze the micro/macro interface and the shared conservative
   mass/angular-momentum/Killing-energy flux definitions.
2. Using the existing common and embedded-patch histories, compare:
   - time-integrated shared interface fluxes;
   - horizon fluxes;
   - integrated inner storage;
   - responsive-height work;
   - cooling and complete physical ledgers.
3. Separate same-time phase-sensitive values from window-integrated
   conservative outputs.
4. Require spatial and temporal convergence of every exported macro
   observable; do not require a branch-projected internal phase to converge
   if it provably does not affect the exported ledger.
5. If the conservative outputs converge, authorize a bounded,
   constraint-consistent micro-solver feasibility test with multiple exact
   equal-macrostate lifts.
6. If those outputs do not converge, stop reduction work and redesign the
   complete coupled near-horizon spatial operator rather than tuning one
   block.

Only after that preflight should the project enter WP10c9d and ask whether
the constrained inner fast problem approaches a steady state, periodic
orbit, invariant measure, multiple attractors, or irreducible memory.

No tide, wind, hot-state, loading-time, S-curve, or QPE-cycle work is
authorized.

## Verification

```text
Focused c9c0b/c/d tests   10 passed
Full repository suite     787 passed, 4 subtests passed
Repository hygiene        773 tracked files passed
Diff whitespace check     passed
```

## Reproduction

```text
PYTHONPATH=src:scripts \
python scripts/run_causal_inner_common_block_family_audit_wp10c9c0d.py

PYTHONPATH=src:scripts \
python -m pytest -q \
  tests/test_causal_inner_shear_energy_ledger.py \
  tests/test_causal_inner_family_transfer.py \
  tests/test_causal_inner_common_block_family_audit_wp10c9c0d.py
```

Machine evidence:

- `outputs/tables/causal_inner_common_block_family_audit_wp10c9c0d.json`
- `outputs/tables/causal_inner_common_block_family_audit_wp10c9c0d_arrays.npz`
- `outputs/checkpoints/causal_inner_common_block_family_wp10c9c0d/`
