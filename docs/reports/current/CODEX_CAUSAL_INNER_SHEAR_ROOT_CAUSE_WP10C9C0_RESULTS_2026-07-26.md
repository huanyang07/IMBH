# WP10c9c0 — Causal-shear root-cause proof

## Verdict

WP10c9c0 completes the predeclared root-cause stop test and does **not**
authorize WP10c9c1:

```text
path_inconsistency_not_proved_selected_shear_damping_persists
```

The implemented principal sign is now explicit and certified:

```text
A = U_p + A_height
B = F_p - C_pr
```

where `C_pr` is the coefficient of the derivative-dependent source on the
right-hand side. Accordingly, the straight-path jump audited here is

```text
Delta F - integral C_pr(Psi) Psi_s ds.
```

The sign, path reversal, derivative plateau, two-family projector, and positive
shear symmetrizer all pass. Both the current split and a monolithic
complete-principal reference also converge in the constant-coefficient Fourier
and variable-coefficient manufactured-wave tests.

The binding full-operator experiment then preserves the entire production
frozen generator and adds only

```text
G_monolithic_principal - G_current_split_principal.
```

That correction does not repair the inward-shear selected-family damping
order:

```text
-0.07156  production scalar Rusanov
-0.02646  WP10c9b characteristic matrix
-0.08947  WP10c9c0 monolithic-principal correction
```

Therefore the conservative-flux/nonconservative-source split is not
demonstrated to be the controlling defect. Implementing a nonlinear path flux
now would violate the binding stop gate.

An important diagnostic refinement also emerged. The physical
symmetrizer-based **total shear-subspace energy** converges for the inward
packet at order `1.49`, `1.45`, and `1.55` for the three operators above, even
though selected-branch amplitude, branch self-energy, and the scaled descriptor
norm do not. About `21-24%` of total shear energy leaves the original
measurement window. A naive split into independently normalized branch
self-energies is strongly non-orthogonal: its cross term reaches `4.27` times
the small net total. The remaining problem is therefore more specific:

```text
mesh-sensitive shear-family partition / non-normal transfer,
not nonconvergence of the total symmetrizer-based shear-subspace energy.
```

Production is unchanged. No nonlinear path candidate, common-mode rerun,
nonlinear patch, fixed-`Q` average, or reduced model is authorized.

## Frozen scope

The package reuses and hashes:

- the WP10c9a N128/N256/N512 scalar-Rusanov packet evidence;
- the WP10c9b five-family characteristic-matrix evidence;
- the exact matched anchors, amplitudes, layouts, and packet definitions;
- the unchanged `0.125 s`, 201-output packet contract.

The audit adds:

1. a sign-explicit decomposition of mapped storage, responsive-height
   storage, physical flux, shear principal source, and vertical principal
   source;
2. a declared straight-path small-jump identity;
3. the complete-coordinate two-shear invariant projector;
4. a positive local-rest shear symmetrizer and its coordinate-subspace pullback;
5. principal-only and physical-relaxation Fourier symbols;
6. an independent variable-coefficient manufactured wave;
7. one matched monolithic-principal correction to the unchanged full
   generator;
8. characteristic, descriptor, and symmetrizer-based shear-energy damping
   measures.

It does not implement a finite-amplitude path-conservative operator.

## Sign, derivative, projector, and energy contracts

The exact identities are:

```text
A = mapped_storage + responsive_height_storage
C_pr = C_shear + C_height
B = physical_flux - C_pr
```

Across `1.90, 2.20, 3.00, 5.00, 6.40 rg`:

| Contract | Maximum/minimum result | Gate | Pass |
|---|---:|---:|---|
| Straight-path small-jump defect | `1.255e-9` | `<=2e-8` | yes |
| Path-reversal defect | `3.24e-30` | `<=1e-12` | yes |
| Five-point derivative-step plateau | `3.839e-6` | `<=2e-5` | yes |
| Shear-speed step defect | `1.892e-8` | `<=2e-5` | yes |
| Shear-projector step defect | `6.790e-6` | `<=2e-5` | yes |
| Projector idempotence/complement defect | `1.705e-13` | `<=2e-10` | yes |
| Analytic local-rest projector defect | `5.80e-17` | `<=2e-10` | yes |
| Analytic local-rest eigenpair defect | `5.55e-17` | `<=2e-10` | yes |
| Local-rest symmetrizer defect | `0` | `<=1e-12` | yes |
| Minimum energy eigenvalue | `9.882e-3` | `>0` | yes |
| Maximum coordinate energy condition | `19.92` | finite | yes |

The derivative sweep is:

```text
5e-5, 1e-4, 2e-4, 4e-4
```

in the primitive column scales. It covers every component matrix, rather than
only the final eigensystem.

The coordinate shear speeds remain separated throughout the sampled inner
domain:

| Radius | Inward-rest shear speed | Outward-rest shear speed | Gap |
|---:|---:|---:|---:|
| `1.90 rg` | `-0.68257 c` | `-0.66324 c` | `0.01934 c` |
| `2.20 rg` | `-0.66231 c` | `-0.64344 c` | `0.01887 c` |
| `3.00 rg` | `-0.62028 c` | `-0.60259 c` | `0.01769 c` |
| `5.00 rg` | `-0.55747 c` | `-0.54212 c` | `0.01535 c` |
| `6.40 rg` | `-0.53140 c` | `-0.51715 c` | `0.01425 c` |

Both branches are coordinate-inward in this plunging region. There is no
zero-speed crossing or ill-conditioned shear cluster in the audited interval.

## Constant-coefficient Fourier audit

The continuum principal generator is compared with:

- the current quadratic-flux/arithmetic-source split;
- a monolithic complete-`B` characteristic discretization;
- a centered monolithic reference;
- physical-flux-only and principal-source-only attribution controls.

Principal-only and principal-plus-stress-relaxation symbols are evaluated
separately. This prevents physical Maxwell-Cattaneo relaxation from being
counted as numerical damping.

The binding asymptotic range is

```text
theta = kh <= 0.40
```

on the coarse grid. `theta=0.80` is retained as a high-wavenumber diagnostic,
not used to claim an asymptotic order.

Minimum observed orders over all five radii and binding wavenumbers are:

| Operator | Phase | Numerical damping | Relaxing eigenvalue |
|---|---:|---:|---:|
| Current split | `2.00036` | `2.97105` | `2.00060` |
| Monolithic complete principal | `2.00037` | `2.97110` | `2.00113` |

Both operators pass the `1.8` local-order gate. The current split does not fail
the constant-coefficient symbol test.

## Variable-coefficient manufactured wave

An independent smooth primitive-chart wave is differentiated analytically over
`2.15-5.40 rg`. Its exact target is

```text
B(R) p_R(R),
```

not `d[B(R)p]/dR`; no spurious `B_R p` term is introduced.

| Operator | N128→N256 order | N256→N512 order |
|---|---:|---:|
| Current split | `2.09217` | `2.02419` |
| Monolithic | `2.09453` | `2.02516` |

Fine relative errors are:

```text
current split = 4.715e-4
monolithic    = 4.618e-4
```

Both pass. The smooth variable-coefficient test therefore also fails to
implicate the principal split.

## Binding corrected full-generator packets

The final root-cause experiment does not replace the production generator with
the simplified principal reference. It uses

```text
G_corrected =
    G_full_production
    + G_monolithic_principal
    - G_current_split_principal.
```

All lower-order, geometry, boundary, coupling, and production scalar-Rusanov
blocks remain those of the frozen full generator. The relative Frobenius size
of the inserted correction is:

| Inner refinement ratio | Relative correction |
|---:|---:|
| 1 | `0.04331` |
| 2 | `0.04789` |
| 4 | `0.05126` |

The correction is large enough to be meaningful; its failure is not a
roundoff-null experiment.

| Family | State order | Rate order | Phase order | Characteristic damping | Total shear-energy damping | Selected-branch self-energy damping | Descriptor damping | Fine cosine | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Inward shear | `1.6003` | `1.0236` | `1.6252` | `-0.0895` | `1.5543` | `-0.3665` | `0.5731` | `0.99867` | **no** |
| Outward shear | `1.6176` | `1.0186` | `1.8815` | `2.2866` | `2.3091` | `2.4637` | `3.1231` | `0.99872` | yes |

Thus:

- same-time state and rate directions converge;
- the phase converges;
- total inward shear energy converges;
- selected-branch characteristic amplitude does not converge;
- the non-orthogonal selected-branch self-energy does not converge;
- the secondary scaled descriptor norm remains below the damping gate.

The monolithic correction does not change the architecture-controlling
classification.

## Why the energy result matters

The local-rest shear block is symmetrized by a positive matrix equivalent to

```text
H_sh = diag(h, 1 / (h c_nu^2)).
```

The coordinate audit projects each perturbation into the complete implemented
two-shear invariant subspace and evaluates the pullback of this energy. It is
invariant under a rescaling of the two coordinate eigenvectors, unlike a raw
Euclidean characteristic coefficient.

For the inward packet:

| Operator | Characteristic damping order | Total shear-energy order | Selected-branch self-energy order | Descriptor order |
|---|---:|---:|---:|---:|
| Production scalar Rusanov | `-0.0716` | `1.4878` | `-0.3566` | `0.4675` |
| WP10c9b matrix penalty | `-0.0265` | `1.4482` | `-0.3867` | `0.4836` |
| Monolithic correction | `-0.0895` | `1.5543` | `-0.3665` | `0.5731` |

Across all three operators:

- the opposite-branch self-energy is about `0.343` of the sum of the two
  branch self-energies;
- the branch cross term can reach `4.27` times the small net total, proving
  that this coordinate branch split is not an orthogonal physical-energy
  decomposition;
- the minimum energy fraction remaining in the original packet window is
  about `0.756-0.789`;
- the total symmetrizer-based shear-subspace energy ladder converges;
- the selected-branch self-energy and descriptor partitions do not.

The previous phrase “nonconvergent shear damping” was therefore too broad.
The nonconvergence is in a family-resolved diagnostic of a converging
symmetrizer-based total, not in total shear-energy loss itself. The exact
coordinate congruence and connection terms must be derived before either
branch self-energy is called a physical branch energy.

This does not automatically pass the operator. A reduced model may care about
the directional branch, and the unresolved descriptor norm may represent real
non-normal transfer. It does mean that a new dissipative or path flux should
not be designed to “repair” total damping that already converges.

## Root-cause decision

The predeclared decision logic gives:

```text
current split locally failed       = false
monolithic locally passed          = true
corrected full packet passed       = false
path inconsistency proved          = false
WP10c9c1 authorized                = false
```

The specific nonlinear path

```text
Delta F - integral C_pr(Psi) Psi_s ds
```

remains mathematically well defined and its small-jump sign is certified. This
package does not show that it is the remedy for the observed full-generator
packet diagnostic.

## Main remaining problems

### 1. Selected-family energy transfer is not reconciled

The two independently normalized branch self-energies have a large cross term
and therefore are not an orthogonal decomposition of the total. Spatially
varying projectors can also exchange branch content even when the total
two-shear energy follows a convergent balance.

The present packet gate treats the selected characteristic coefficient as if
it were an independently damped scalar. That assumption is not established in
the plunging, variable-coefficient, non-normal system.

### 2. The complete semidiscrete shear-energy ledger is not yet assembled

The audit measures total, selected, opposite, descriptor, and window energy,
but does not yet split their rates into:

- physical stress relaxation;
- energy crossing the packet window;
- inner excision loss;
- shear-projector rotation/connection work;
- coupling to acoustic/contact families;
- numerical dissipation;
- lower-order geometry and responsive-height work.

Without that ledger, selected-family decay cannot be assigned to a defective
operator block.

### 3. The full generator contains blocks absent from both passing local tests

The Fourier and manufactured references contain the complete principal matrix
and physical stress relaxation, but not every full-generator lower-order,
boundary, and non-normal coupling contribution.

The failure appears only after those full blocks are restored. The next
diagnosis must therefore ablate those blocks on the unchanged generator rather
than construct a new interface flux.

### 4. The packet measurement window is not closed

At least `21%` of total shear energy leaves the original support window during
the run. A pointwise or fixed-window amplitude can mix true damping with
advection, spreading, and boundary loss.

## Locked follow-up before WP10c9c1

WP10c9c1 remains closed. The next bounded package should:

1. derive the exact semidiscrete two-shear energy balance from the full
   generator;
2. record physical relaxation, numerical dissipation, window/excision flux,
   projector-rotation work, and cross-family transfer separately;
3. use an `H_sh`-orthogonal transported shear projector rather than comparing
   independently normalized eigenvectors at each cell;
4. decompose
   `G_full` into complete-principal, scalar dissipation, stress relaxation,
   responsive-height, geometry/cooling, boundary, and residual coupling
   blocks;
5. run one-at-a-time and cumulative ablations on the unchanged
   N128/N256/N512 inward packet;
6. compare fixed-window, comoving-window, total-domain, selected-branch,
   total-shear, and descriptor measures;
7. retain the outward packet as a passing control.

Binding decisions:

| Result | Decision |
|---|---|
| Total symmetrizer energy and its exact ledger converge; selected split is fully explained by projector rotation/window flux | Replace the old scalar damping gate with a declared physical energy/transfer gate and re-audit the unchanged operator |
| One lower-order or boundary block uniquely destroys the energy/branch balance | Repair only that block, then rerun both shear ladders |
| A path correction becomes uniquely controlling after the exact ledger | Authorize one sign-explicit WP10c9c1 candidate |
| Total energy itself loses convergence under the exact ledger | Redesign the full near-horizon spatial operator |
| No rapidly convergent physical fast object emerges | Retain the localized inner solver; do not force a Markovian slow closure |

No candidate coefficient, path, tolerance, or packet profile may be tuned
against the binding inward packet.

## Reduced slow-time implication

Reduced slow evolution remains possible, but WP10c9d is not authorized.

The converging symmetrizer-based shear-subspace energy is encouraging: it
suggests that the inner fast subsystem may possess a mesh-convergent physical
energy balance once the exact coordinate congruence and connection ledger are
derived, even though its instantaneous characteristic partition is
non-normal. That favors a future conservative micro-macro or quasi-steady
closure over a fitted scalar shear amplitude.

The required order remains:

```text
exact full shear-energy ledger
-> identify/fix or retire the selected-family damping gate
-> nonlinear spatial convergence
-> constraint-consistent fixed-Q fast experiments
-> choose quasi-steady, cycle-averaged, HMM, hysteretic,
   or retained-inner-solver closure
```

## Verification

Machine evidence:

- `outputs/tables/causal_inner_shear_root_cause_audit_wp10c9c0.json`
- `outputs/tables/causal_inner_shear_root_cause_audit_wp10c9c0_arrays.npz`

The package includes focused tests for:

- sign-explicit principal decomposition;
- path reversal and small-jump closure;
- positive shear energy and projector identities;
- derivative-step plateau;
- Fourier-symbol convergence;
- exact quadratic reconstruction;
- independent manufactured-wave convergence;
- split/monolithic frozen-generator construction;
- machine-evidence stop classification.

The expensive parent N128/N256/N512 histories are reused rather than
regenerated. The only new packet histories are the two shear families under
the isolated monolithic-principal correction.

Verification completed on 2026-07-26:

```text
focused WP10c9c0 tests: 8 passed
full repository suite: 777 passed, 4 subtests passed
repository hygiene:    passed for 773 tracked files
git diff check:        passed
```
