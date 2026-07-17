# Causal Five-Field Reduced Null Audit WP10c5c Results

Date: 2026-07-17

## Verdict

The `N=16` stationary near-null direction is not an exact missing constraint.
After exact elimination of the conserved-state and face-flux identity blocks,
the primitive-only stationary response is full rank:

```text
full flux-primary response       244 / 245
algebraic identity block         165 / 165
reduced primitive response        80 / 80
outer (lnT, chi) response           2 / 2
```

The old `244/245` gate failure is conditioning introduced by the redundant
flux-primary embedding and its declared scaling at a nonroot seed. It is not
evidence for a missing Roche condition, an exact stationary nullspace, or a
physical marginal equilibrium.

No stationary root, timestep, tide, wind, or physical branch search was run in
this work package.

## Exact Reduction

Write the stationary unknowns as

```text
q = (cell conserved states, all face fluxes),
p = cell primitives.
```

The primitive-map and face-map rows form the square algebraic block

```text
A_qq delta q + A_qp delta p = 0.
```

They give

```text
delta q = -A_qq^(-1) A_qp delta p.
```

Substitution into the conservation rows gives the scaled primitive response

```text
J_red = A_pp - A_pq A_qq^(-1) A_qp,
```

with dimensions `5N x 5N`.

An independent implementation remaps the conserved states and every physical
face flux from a perturbed primitive vector, then finite-differences only the
conservation rows. This direct reduced Jacobian is compared with the Schur
matrix rather than assumed equivalent.

At the published closed seed and finite-difference step `2e-6`:

```text
algebraic-block condition estimate          1.0000000001
relative Frobenius direct/Schur defect       2.99e-11
operator-scaled directional defect           5.04e-12
algebraic tangent reconstruction defect      1.10e-16
alignment with full weakest direction        0.999999999993
```

The relative error obtained by dividing by the weak response itself can be a
few percent. That ratio is not used as an equivalence gate because both the
numerator and denominator are near the finite-difference floor. The
operator-scaled and matrix-norm comparisons pass.

## Published Closed Seed

The published seed is flux consistent but is not a stationary root:

```text
maximum scaled conservation residual = 1.73719
primitive-map residual                = 0
face-map residual                     = 0
```

The reduced response is stable across the same three finite-difference steps:

| FD step | Reduced rank | Smallest singular value |
|---:|---:|---:|
| `1e-6` | `80/80` | `4.35861e-9` |
| `2e-6` | `80/80` | `4.35715e-9` |
| `5e-6` | `80/80` | `4.35760e-9` |

The relative spread is `3.36e-4`. The weakest primitive direction remains
localized at the outermost cell, with field norms dominated by `lnT` and
specific stress. It is weak but nonzero under the declared reduced scaling.

The final two-variable Schur response after eliminating the other 78
primitives is also full rank:

```text
outer thermal/stress rank       2/2
condition estimate              5.82e6
smallest singular value         4.42e-6
```

This excludes an exact missing outer thermal or stress condition at this seed.

## Closed/Open Roche Comparison

The same gas+radiation Hill/Roche provider was evaluated on a controlled
thermodynamic pair with `outer Sigma=1e4`:

| Gate | Outer T | Reduced rank | Reduced sigma_min | Outer rank | Outer condition |
|---|---:|---:|---:|---:|---:|
| closed | `8e5 K` | `80/80` | `3.57e-9` | `2/2` | `1.96e6` |
| choked/open | `1e6 K` | `80/80` | `3.90e-7` | `2/2` | `3.22e5` |

Opening the physical channel strengthens the primitive and outer responses; it
does not supply a missing rank direction. Because the smooth seed temperature
profile changes with its outer endpoint, this is an active-branch diagnostic,
not an identical-state boundary derivative comparison.

## Interpretation

The bounded classification is:

```text
full primitive response with flux-primary embedding conditioning
at a nonstationary seed
```

The result does not demonstrate:

- a stationary equilibrium;
- an exact nullspace;
- thermal marginality;
- a slow physical mode;
- stability or instability;
- a limit-cycle precursor.

The full flux-primary variables remain useful for conservative implicit
assembly. Their gate-defined `244/245` rank should not be mistaken for the
rank of the physical primitive stationary operator.

## Gate Decision

```text
reduced operator equivalence       PASS
algebraic elimination              PASS
reduced primitive rank             PASS
outer thermal/stress rank          PASS
missing Roche condition            NOT FOUND
exact stationary nullspace         NOT FOUND
N64/N96 stationary roots           NOT AUTHORIZED
tide                               BLOCKED
wind                               BLOCKED
```

The next authorized action is the separately scoped index-one
consistent-initial-data gate. It must balance the nonzero conservation
residual with a storage tangent while remaining tangent to all 165 algebraic
constraints. It must not force the conservation residual to zero.

## Verification

```text
PYTHONPATH=src python3 -m pytest -q \
  tests/test_causal_inner_dae_system.py

PYTHONPATH=src python3 \
  scripts/run_causal_five_field_reduced_null_audit_wp10c5c.py
```

Machine-readable output:

```text
outputs/tables/causal_five_field_reduced_null_audit_wp10c5c.json
```
