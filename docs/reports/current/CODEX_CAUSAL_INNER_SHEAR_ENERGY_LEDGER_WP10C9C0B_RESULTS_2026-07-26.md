# WP10c9c0b — Full shear-energy ledger and operator attribution

## Verdict

WP10c9c0b completes the exact full-generator energy ledger requested by
WP10c9c0. Every method and reconstruction contract passes, but the scientific
selected-family gate does not:

```text
selected_shear_energy_defect_is_transport_window_or_family_transfer_sensitive
```

The principal result is:

```text
inward total shear energy order             1.48775   pass
inward orthogonal selected-family order     0.48778   fail
inward orthogonal complement order         -0.11715   fail
outward total shear energy order             2.34417   pass
outward orthogonal selected-family order     2.35534   pass
```

The new selected/complement split is orthogonal in the positive physical shear
energy. Its energy partition defect is `3.23e-15`; it removes the large
normalization-dependent branch cross term found in WP10c9c0. The remaining
inward result is therefore not explained solely by arbitrary eigenvector
normalization.

The exact block ledger also rejects a single-block explanation:

- the selected preserving-rate history converges at order `1.078`;
- its cumulative work reaches only order `0.561`;
- the selected/complement transfer-rate and cumulative-work orders are
  `-0.795` and `-0.601`;
- three different one-at-a-time removals pass, so no unique operator block is
  identified;
- removing the boundary block worsens the selected order to `0.147`;
- removing scalar Rusanov dissipation worsens it to `-0.928`;
- the final `residual_unattributed` block is only `1.34e-9` of the generator
  Frobenius norm and has no material effect.

The evidence supports a coupled, non-normal transport/descriptor transfer
mechanism rather than one faulty path sign, boundary flux, or viscosity.

Production remains unchanged. WP10c9c1, a nonlinear path candidate, nonlinear
truth, fixed-`Q` averaging, and reduced slow evolution remain unauthorized.

## Frozen scope

The package reuses and hashes:

- the exact WP10c9a N128/N256/N512 inner-patch packet configurations;
- the unchanged scalar-Rusanov production generators;
- both pure shear packet histories through `0.125 s`;
- the WP10c9c0 root-cause result and its sign/projector contracts.

It adds only audit infrastructure:

1. an energy-orthogonal selected-family/complement projector;
2. an exact physical decomposition of each frozen evolving generator;
3. full-domain and fixed-window quadratic-energy ledgers;
4. preserving-versus-transfer ledgers;
5. absolutely significant block-rate comparisons;
6. bounded one-at-a-time and cumulative generator ablations.

No production residual, numerical flux, boundary, physics, or initial packet
is changed.

## Orthogonal shear-energy partition

For each cell, the complete two-shear invariant subspace is equipped with the
positive local-rest shear energy pulled into the primitive chart. For selected
family direction `v`, the projector is the energy-orthogonal projector

```text
P_v = v (v^T H v)^(-1) v^T H.
```

The complementary shear projector is

```text
P_perp = P_shear - P_v.
```

Across all three meshes:

| Contract | Maximum/minimum | Gate | Pass |
|---|---:|---:|---|
| Shear/family/projector defect | `6.37e-12` | `<=2e-6` | yes |
| Energy self-adjoint defect | `5.33e-16` | `<=2e-6` | yes |
| Energy partition defect | `3.23e-15` | `<=1e-10` | yes |
| Minimum positive energy eigenvalue | `2.60e-3` | `>0` | yes |

The selected and complement energies therefore add to total shear energy
without an uncontrolled cross term.

## Exact generator decomposition

At each refinement ratio the frozen generator is reconstructed from:

- central perfect-fluid and stress transport;
- the Rusanov penalty;
- inner and outer boundary transport;
- geometry, cooling, and stream sources;
- the combined resolved-shear-principal/local-relaxation source;
- responsive-height source and descriptor terms;
- mapped-storage rate dependence.

The same descriptor matrix is used to convert every residual block into a
primitive-rate generator block. The small difference from the independently
cached full generator is retained explicitly as `residual_unattributed`.

| Ratio | State dimension | Stationary-Jacobian defect | Pre-remainder generator defect | Final defect | Unattributed fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | `320` | `7.47e-12` | `3.54e-10` | `0` | `1.09e-9` |
| 2 | `560` | `8.94e-12` | `3.87e-10` | `0` | `1.19e-9` |
| 4 | `1040` | `8.68e-12` | `4.78e-10` | `0` | `1.34e-9` |

The maximum base-residual reconstruction defect is `8.49e-16`, and the
maximum descriptor solve defect is `9.92e-16`.

One semantic limitation is explicit: the implemented
`source_stress_relaxation` contains both the resolved shear-gradient principal
contribution and the local Maxwell-Cattaneo decay. WP10c9c0b keeps that exact
combined block. It does not claim to have separated those terms.

## Energy-ledger contracts

For a frozen state history `x(t)` and energy matrix `H`, the instantaneous
ledger uses

```text
E       = 1/2 x^T H x
dE/dt   = x^T H G x
        = sum_k x^T H G_k x.
```

All full, selected, and complement ledgers close:

| Contract | Maximum | Gate | Pass |
|---|---:|---:|---|
| Instantaneous energy partition | `1.03e-15` | `<=1e-10` | yes |
| Instantaneous block ledger | `5.63e-15` | `<=1e-10` | yes |
| Instantaneous source partition | `2.44e-15` | `<=1e-10` | yes |
| 801-sample integrated defect | `2.65e-7` | `<=1e-6` | yes |
| Integrated temporal order | `1.99961` | `>=1.8` | yes |

The 201-sample defect is `4.24e-6`; refining only the audit quadrature to 801
samples reduces it at second order. No physical trajectory is changed.

## Spatial energy result

The full and fixed-window histories give:

| Family/measure | Coarse defect | Fine defect | Order |
|---|---:|---:|---:|
| Inward full total | `1.700e-3` | `6.061e-4` | `1.4878` |
| Inward full selected | `6.290e-3` | `4.485e-3` | `0.4878` |
| Inward full complement | `3.572e-3` | `3.875e-3` | `-0.1171` |
| Inward fixed-window total | `4.088e-2` | `2.133e-2` | `0.9389` |
| Inward fixed-window selected | `3.376e-2` | `1.574e-2` | `1.1012` |
| Inward comoving total | `7.241e-3` | `3.548e-3` | `1.0290` |
| Outward full total | `1.097e-2` | `2.161e-3` | `2.3442` |
| Outward full selected | `1.501e-2` | `2.933e-3` | `2.3553` |

At least `98.69%` of inward total shear energy remains inside the declared
comoving window on every mesh. The result is therefore not produced by losing
most of the packet from the audit domain. However, the opposite behavior of
the inward full/fixed selected histories and the outward full/fixed histories
shows that a fixed packet window is not a universal damping observable.

The inward fine selected-energy defect grows smoothly:

```text
t = 0.004375 s    defect = 1.12e-4
t = 0.036250 s    defect = 1.00e-3
t = 0.064375 s    defect = 2.01e-3
t = 0.125000 s    defect = 4.49e-3
```

It is not a single late endpoint jump.

## Preserving and transfer ledgers

The generator is partitioned algebraically into:

```text
G_preserving =
    P_selected G P_selected
  + P_complement G P_complement
  + P_non-shear G P_non-shear

G_transfer = G - G_preserving.
```

This is a cellwise energy-orthogonal partition. It is not yet a
parallel-transported family connection.

For the inward selected energy:

| Quantity | Coarse defect | Fine defect | Order |
|---|---:|---:|---:|
| Preserving rate | `6.576e-2 /s` | `3.115e-2 /s` | `1.0781` |
| Preserving cumulative work | `1.563e-3` | `1.059e-3` | `0.5608` |
| Transfer rate | `3.384e-3 /s` | `5.870e-3 /s` | `-0.7950` |
| Transfer cumulative work | `1.358e-4` | `2.061e-4` | `-0.6012` |

The final selected transfer work is:

```text
ratio 1   -0.0043314
ratio 2   -0.0044672
ratio 4   -0.0042612
```

The transfer contribution is small in net magnitude but has a mesh-sensitive
signed history. The preserving contribution is much larger and convergent
instantaneously, yet its accumulated cancellation reaches only order `0.56`.
Both effects matter; the transfer ledger alone is not a complete explanation.

## Absolutely significant block attribution

Each block rate is normalized by one common full selected-rate scale. This
prevents a scientifically negligible block from appearing important merely
because it is normalized by its own tiny magnitude.

For the inward packet:

| Block | Maximum amplitude / full scale | Fine defect / full scale | Rate order |
|---|---:|---:|---:|
| Boundaries | `0.2005` | `3.356e-3` | `1.4200` |
| Conservative transport | `2.0349` | `2.394e-3` | `2.1210` |
| Geometry and cooling | `2.8126` | `1.453e-3` | `1.3085` |
| Mapped descriptor | `1.3077` | `1.333e-3` | `2.5348` |
| Numerical dissipation | `4.201e-3` | `5.463e-4` | `2.7084` |
| Stress principal plus relaxation | `0.4715` | `3.885e-4` | `0.9581` |
| Responsive height | `3.576e-3` | `1.048e-4` | `0.4813` |
| Unattributed | `1.73e-9` | `7.17e-10` | `2.1047` |

The boundary block has the largest fine absolute rate defect, peaking near
`0.070625 s`, but its own rate converges at order `1.42`. The conservative,
geometry, mapped-descriptor, dissipation, and combined stress blocks also have
convergent rate histories. Responsive height has a low rate order but only
`1.05e-4` of the full fine defect scale, and removing it does not restore the
selected-energy gate.

Thus no isolated block has the same signature as the final selected-energy
history.

## Bounded ablations

The one-at-a-time inward selected-energy orders are:

| Variant | Selected-energy order | Interpretation |
|---|---:|---|
| Full generator | `0.4878` | binding failure |
| Without geometry/cooling | `2.5307` | passes, altered dynamics |
| Without conservative transport | `2.3477` | passes, altered dynamics |
| Without mapped descriptor | `1.0786` | passes, altered dynamics |
| Without stress principal/relaxation | `0.7044` | still fails |
| Without responsive height | `0.6885` | still fails |
| Without boundaries | `0.1465` | worse |
| Without numerical dissipation | `-0.9282` | substantially worse |
| Without unattributed remainder | `0.4878` | unchanged |

The cumulative ladder passes through geometry/cooling at order `1.0786`; adding
the mapped-descriptor block returns the exact full result, `0.4878`. But the
mapped-descriptor rate block itself converges at order `2.53`, and removing
either conservative transport or geometry/cooling also changes the result to
a pass. Therefore the mapped block is an interaction trigger, not a uniquely
identified faulty discretization.

Ablations change the equations and are sensitivity controls, not candidate
physical models. Three independently significant removals pass, so the audit
does not authorize modifying any one of them.

## Updated scientific interpretation

WP10c9c0b establishes:

1. total physical shear-subspace energy is spatially convergent;
2. the orthogonal inward selected/complement partition remains pre-asymptotic;
3. the old failure is not solely eigenvector normalization;
4. scalar numerical viscosity is not the cause—removing it is worse;
5. the boundary is not the cause—removing it is worse;
6. the path split remains unproved as a cause;
7. multiple individually convergent blocks create a low-order accumulated
   result through non-normal interaction and cancellation.

This is not sufficient to promote the current operator to nonlinear reduction
work. The earlier WP10c8y common perturbation still has basis-invariant
state/rate nonconvergence. Conversely, it would also be wrong to redesign one
operator block using only the selected-family diagnostic.

## Recommended next package: WP10c9c0c

### Goal

Identify where the first mesh-dependent family transfer is generated and
which pair of characteristic families creates the already observed
common-mode failure, using existing frozen generators and histories only.

### Phase 1 — Freeze the evidence

Freeze and hash:

- WP10c8y common-mode generators and histories;
- WP10c9a pure-family packet histories;
- WP10c9c0 sign/projector evidence;
- WP10c9c0b projectors, generator blocks, ledgers, and ablations.

Keep production unchanged. Run no new truth trajectory.

### Phase 2 — Face- and radius-resolved local shear-energy balance

Derive a discrete local balance of the form

```text
dE_shear,i/dt
  + Phi_shear,i+1/2 - Phi_shear,i-1/2
  = W_preserving,i + W_transfer,i.
```

Resolve separately:

- central perfect-fluid and stress face transport;
- Rusanov face dissipation;
- inner and outer boundary transport;
- the derivative-dependent stress source;
- local Maxwell-Cattaneo relaxation;
- mapped and responsive-height descriptor work;
- projector variation/connection work;
- transfer to the orthogonal shear and non-shear subspaces.

The present combined stress source may be split only by an exact residual
identity and small-jump check. Do not infer its two pieces from a fitted
history.

Require:

```text
instantaneous local-to-global ledger defect    <= 1e-10
801-sample integrated defect                   <= 1e-6
integrated temporal order                      >= 1.8
shared conservative face-flux defect           <= 1e-12
```

Track the first significant N256/N512 departure by face, cell, time, and
operator term. Apply the unchanged `1e-4` absolute-significance filter.

### Phase 3 — Exact common-mode family-pair decomposition

On the existing WP10c8y frozen generators:

1. project the common continuum initial perturbation into all five local
   characteristic families in one common physical metric;
2. propagate every family component separately;
3. verify that their sum reproduces the unchanged common trajectory to
   `1e-12`;
4. calculate all pairwise state, rate, physical-energy, and cross-work terms;
5. conservatively restrict them to common N64/N128/N256 representations;
6. identify which absolutely significant family pair creates the negative
   common-mode rate order.

POD/DMD fitting is not part of this package.

### Phase 4 — Coupled localization

Correlate the pairwise common-mode departure with the pure inward-shear local
energy ledger. Determine whether the controlling effect is:

- excision loss;
- conservative face transport;
- shear/non-shear family conversion;
- descriptor-induced non-normal coupling;
- the combined shear-principal/relaxation source;
- or a distributed cancellation with no localized controller.

Do not use a time shift for a binding pass.

### Phase 5 — Decision

- **One localized, absolutely significant face/block coupling:** authorize an
  audit-only correction of that coupling with independent Fourier,
  manufactured-wave, and held-out packet tests.
- **Only the selected partition fails while the common physical state/rate and
  total energy pass:** retire selected-family damping as a binding production
  observable.
- **A convergent family pair has non-negligible physical transfer:** retain the
  pair in the future inner micro-solver; do not suppress it numerically.
- **Several distributed pairs remain nonconvergent:** stop fitting local
  coordinates and redesign the complete near-horizon finite-volume/descriptor
  coupling.

Only after the common physical state/rate history converges may the project
run nonlinear patch truth or constrained fast experiments.

## Hard stops

Do not:

- authorize WP10c9c1 from this result;
- change production defaults;
- tune a path, viscosity, or descriptor coefficient against these packets;
- use one-at-a-time ablations as physical models;
- launch N1024 brute-force refinement;
- run fixed-`Q` averaging or a reduced macrostep;
- add tide, wind, hot-state, or cycle physics.

## Verification

The implementation was checked at three levels:

```text
focused WP10c9c0/c0b tests          12 passed
causal/characteristic regression   111 passed
full repository suite              781 passed + 4 subtests
repository hygiene                 773 tracked files passed
```

The optimized evidence-runner ledger was also compared directly with the
public reference ledger on the regression DAE; all energy, rate, block,
source-partition, preserving, and transfer arrays agreed to machine precision.

The final JSON/NPZ, runner, core module, and all three decomposition-cache
hashes were reverified after the run.

## Reproduction

```bash
PYTHONPATH=src:scripts python \
  scripts/run_causal_inner_shear_energy_ledger_wp10c9c0b.py

PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_causal_inner_shear_energy_ledger.py \
  tests/test_causal_inner_shear_energy_ledger_audit_wp10c9c0b.py
```

Machine evidence:

- `outputs/tables/causal_inner_shear_energy_ledger_wp10c9c0b.json`
- `outputs/tables/causal_inner_shear_energy_ledger_wp10c9c0b_arrays.npz`
- `outputs/checkpoints/causal_inner_shear_energy_ledger_wp10c9c0b/`
