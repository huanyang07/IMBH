# WP10c9d0 — Conservative inner-micro-solver export preflight

## Verdict

WP10c9d0 asks whether the already cached inner common-mode histories export
mesh-convergent slow observables even though their internal characteristic
phase does not converge.

The answer is negative for the conservative embedded-patch ladder:

```text
conservative_micro_exports_fail_spatial_gate
```

The uniform N64/N128/N256 ladder passes both its instantaneous and cumulative
physical-export gates. The N128-exterior embedded ladder does not. Refining
its inner patch from N128-equivalent to N256-equivalent and then
N512-equivalent makes the differences in inner M/J/E flux, conservative net
drive, cooling, and responsive-height work larger rather than smaller.

The binding cumulative exported-vector orders are:

```text
RMS-history order       -1.4342
maximum-error order     -1.0224
```

The fine-pair direction remains close (`minimum cosine = 0.99835`) and the
fine normalized difference remains below the loose `0.10` magnitude gate
(`0.08931`). The failure is specifically lack of contraction under inner
refinement, not loss of algebraic conservation or a random direction.

Therefore:

```text
fixed-Q constrained micro-solver       not authorized
fast-time averaging                    not authorized
reduced slow evolution                 not authorized
single-block/path repair               not authorized
complete coupled inner-operator audit  required
```

## Frozen scope

The package is cache first and production neutral. It uses:

- the WP10c8y uniform N64/N128/N256 common-mode state/rate histories;
- the WP10c8z conservative N128-exterior patch histories with
  N128/N256/N512-equivalent inner resolution;
- the unchanged production numerical flux, integrated source, cooling, and
  responsive-height implementations;
- the cached complete descriptor, descriptor-rate derivative, and stationary
  Jacobian for the uniform ladder.

It does not:

- change the production DAE, flux, reconstruction, source split, descriptor,
  excision trace, patch coupling, or BDF method;
- run a new nonlinear or frozen-linear trajectory;
- construct a constrained fixed-`Q` problem;
- average a freely evolving transient into a closure;
- promote an embedded patch to production.

## Physical observable contract

At 101 exact cached times between `0` and `0.125 s`, each tangent history is
mapped back through the implemented nonlinear physical maps using centered
directional evaluations. The declared export vector is:

1. inner mass, angular-momentum, and Killing-energy flux;
2. interface mass, angular-momentum, and Killing-energy flux;
3. active-domain conservative M/J/E net drive;
4. integrated cooling angular-momentum/Killing-energy source;
5. integrated responsive-height angular-momentum/Killing-energy work.

The conservative drive uses the implemented sign convention:

```text
net drive = inner flux - interface flux + integrated sources.
```

The embedded patch uses its actual shared interior coupling face. Sources are
integrated only over the refined inner microdomain.

Instantaneous histories and cumulative trapezoidal integrals are gated
separately. Each physical component is normalized by its own maximum response
over the complete three-level ladder; components below `1e-8` of their
corresponding baseline scale are excluded from relative order calculations.

## Method contracts

All audit-method contracts pass:

| Contract | Maximum | Gate |
|---|---:|---:|
| Directional-step sweep | `9.30e-4` | `<=1e-3` |
| 51/101-point cumulative sampling | `7.76e-4` | `<=5e-3` |
| Direct physical net drive versus stationary matrix | `2.37e-5` | `<=2e-3` |
| Complete instantaneous descriptor ledger | `5.22e-15` | `<=1e-8` |
| Embedded shared-face state flux | `0` | exact |
| Embedded telescoping defect | `0` | exact |

The directional-map allowance reserves one percent of the `0.10` physical
observable gate. The independent stationary-matrix comparison is much
tighter and verifies the physical flux/source extraction directly.

### Frozen-tangent cumulative storage qualification

The cached evolving tangent contains

```text
M delta-p-dot + DM[p-dot-base] delta-p + J delta-p = 0
```

at one frozen background. Its instantaneous descriptor ledger closes to
roundoff. It does not evolve the background descriptor `M(p)` in time, so
`M(base) delta-p(t)` is not a finite-time state primitive whose derivative
contains the frozen `DM[p-dot-base] delta-p` term.

For that reason, comparing the cumulative frozen net drive with
`M(base)[delta-p(t)-delta-p(0)]` is retained only as a non-integrability
diagnostic. It is not a physical or method gate. A nonlinear constrained
microtrajectory would require the complete responsive-height physical ledger,
but that trajectory is not authorized by this preflight.

## Uniform ladder

The uniform N64/N128/N256 local histories pass:

| Export group | Form | RMS order | Maximum order | Fine maximum | Fine minimum cosine |
|---|---|---:|---:|---:|---:|
| Complete export | Instantaneous | `4.7050` | `4.7011` | `0.03844` | `0.99978` |
| Complete export | Cumulative | `4.4085` | `4.3989` | `0.04740` | `0.99958` |
| Net drive | Instantaneous | `4.0418` | `3.1808` | `0.00546` | `0.99986` |
| Net drive | Cumulative | `6.7135` | `6.2724` | `0.00160` | `0.99999` |
| Cooling/height | Instantaneous | `2.2441` | `2.0063` | `0.03844` | `0.99962` |
| Cooling/height | Cumulative | `1.7688` | `1.7486` | `0.04740` | `0.99951` |

The unusually high complete-export order comes from a large N64/N128
outer-interface discrepancy followed by a very small N128/N256 discrepancy.
The binding fine pair nevertheless passes the independent magnitude and
direction gates, while the cooling/height subvector shows an ordinary
approximately second-order trend.

This establishes that the physical observable map itself is usable and that
the common-mode failure is not universal across every grid ladder.

## Conservative embedded-patch ladder

The embedded N128-exterior ladder fails:

| Export group | Form | RMS order | Maximum order | Fine maximum | Fine minimum cosine |
|---|---|---:|---:|---:|---:|
| Complete export | Instantaneous | `-0.0914` | `0.4560` | `0.03937` | `0.99954` |
| Complete export | Cumulative | `-1.4342` | `-1.0224` | `0.08931` | `0.99835` |
| Inner boundary flux | Instantaneous | `-1.2321` | `-0.8100` | `0.01747` | `0.99749` |
| Inner boundary flux | Cumulative | `-1.5965` | `-1.4351` | `0.02660` | `0.99532` |
| Net drive | Instantaneous | `-1.2301` | `-0.8101` | `0.01746` | `0.99750` |
| Net drive | Cumulative | `-1.5967` | `-1.4351` | `0.02660` | `0.99534` |
| Cooling/height | Instantaneous | `0.0104` | `0.4560` | `0.03937` | `0.99879` |
| Cooling/height | Cumulative | `-1.3624` | `-1.0224` | `0.08931` | `0.99869` |

The shared coupling-face response is below the declared absolute-significance
threshold in this perturbation. The significant boundary response is the
inner/excision flux. Thus the failed export convergence is generated inside
the refined near-horizon domain, not by the coarse/fine coupling face.

This agrees with WP10c8z, where moving the coupling radius had negligible
effect and the coupling signal was tiny, but it is stronger: the failure now
appears in the conservative quantities that an inner micro-solver would have
to return to a slow outer model.

## Scientific interpretation

WP10c9c0d left open one viable possibility: internal phase might fail while
the exported conservative means still converge. WP10c9d0 closes that
possibility for the present embedded operator.

The result is not:

- a conservation failure;
- a coarse/fine coupling failure;
- a directional finite-difference failure;
- a statement that reduced slow-time evolution is impossible in principle.

It is:

```text
the present refined inner bulk operator does not supply a mesh-convergent
conservative response, even after time integration.
```

Consequently, retaining the current inner solver as a microclosure would
average a spatial-discretization defect into the slow dynamics.

## Next architecture gate

The next package must treat the near-horizon spatial residual as one coupled
object. It should not tune the inner trace, central perfect flux, Rusanov
penalty, shear source, or descriptor in isolation.

The recommended next step is a production-neutral coupled-operator design
audit:

1. Define the complete implemented quasilinear principal operator and all
   lower-order geometric/relaxation terms with their exact residual signs.
2. Construct one well-balanced face-fluctuation reference for the full
   five-field principal system, preserving one shared conservative M/J/E
   flux and separately ledgering nonconservative stress/height work.
3. Require exact constant-background preservation and second-order
   manufactured variable-background convergence before any packet history.
4. Evaluate the unchanged WP10c8y common mode and the WP10c9d0 physical export
   vector on N128/N256/N512-equivalent grids.
5. Keep production defaults unchanged. Promote nothing unless both internal
   histories and conservative exports contract.

Fixed-`Q`, nonlinear truth, tide, wind, hot-state, loading-time, S-curve, and
QPE-cycle work remain closed.

## Verification

The focused tests and full repository verification are recorded in the
canonical project status after the package result is finalized.

## Reproduction

```text
PYTHONPATH=src:scripts \
python scripts/run_causal_inner_micro_export_preflight_wp10c9d0.py

PYTHONPATH=src:scripts \
python -m pytest -q \
  tests/test_causal_inner_micro_export_preflight_wp10c9d0.py
```

Machine evidence:

- `outputs/tables/causal_inner_micro_export_preflight_wp10c9d0.json`
- `outputs/tables/causal_inner_micro_export_preflight_wp10c9d0_arrays.npz`
- `outputs/checkpoints/causal_inner_micro_export_preflight_wp10c9d0/`
