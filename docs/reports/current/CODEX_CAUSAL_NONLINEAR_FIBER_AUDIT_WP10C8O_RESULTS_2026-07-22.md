# WP10c8o exact nonlinear coordinate-fiber audit

Date: 2026-07-22
Base commit: `4dc5cea0342d35135e31078669e7e71ba7d16cf9`
Production physics changed: no
Production exact-max Rusanov flux changed: no
Production descriptor or BDF integrator changed: no
Moment ladder changed: no

## Decision

WP10c8o finds and cross-mesh confirms an exact finite-amplitude
equal-coordinate counterexample:

\[
\boxed{
\text{the predeclared 34-coordinate output-identifiable/conservative
instantaneous Markov closure is rejected for the certified N64/N128 truth
discretizations.}
}
\]

The decisive N64 pair has the same exact richest-level coordinates to a
maximum normalized defect of `1.17e-15`, passes every state and fresh-rate
gate, but differs in macro-interface-4 angular-momentum flux by

\[
E_O=0.32452995>0.25.
\]

Each N64 cell's physical perturbation was copied into its two N128 children
and corrected to the exact N128 coordinate fiber. No N128 output
optimization was performed. The N128 pair has a maximum normalized
coordinate defect of `1.78e-15` and reproduces the same controlling output at

\[
E_O=0.26608550>0.25.
\]

This is a binding one-sided no-go result for raw instantaneous closure. It is
not a proof that no healed closure, memory model, or conservative coarse PDE
can work, and it is not a continuum-limit no-go theorem.

## What was tested

The frozen retained state was the richest WP10c8i five-shell level:

- five-shell instantaneous rest mass, angular momentum, and Killing energy;
- shell-mean log temperature;
- shell radial momentum;
- shell causal-stress storage;
- four targeted log-temperature/log-surface-density shape moments.

There are 34 coordinates in total. Responsive-height storage remains the
existing vector one-form and cumulative height work remains a path ledger;
neither was converted into an instantaneous coordinate.

For each signed seed, the scaled primitive increment was written

\[
\delta p_\pm=\pm\alpha d+Qz_\pm,
\]

where `d` is projected into the frozen weighted constraint-null space and the
columns of `Q` form a weighted-orthonormal constraint-normal basis. A
34-dimensional nonlinear corrector solved

\[
C_{\rm nonlinear}(p_0+\delta p_\pm)=C_{\rm nonlinear}(p_0)
\]

independently on both sides. Moving only in `Q` preserves the signed seed
projection and prevents the corrector from collapsing both states back to the
anchor.

Every trial state was rebuilt with the production primitive-to-conserved and
exact-max face-flux maps. Coordinate values were evaluated directly from the
finite state; no reduced tangent was used as an equality surrogate.

## Locked N64 matrix

The fail-fast N64 matrix used the `t=0.025 s` construction anchor.

### Strongest saved richest-level direction

| Multiplier | Maximum exact nonlinear half-spread | Controller |
|---:|---:|---|
| `2.5e-4` | `0.08113249` | interface-4 angular momentum |
| `5.0e-4` | `0.16226498` | interface-4 angular momentum |
| `1.0e-3` | `0.32452995` | interface-4 angular momentum |
| `2.0e-3` | `0.64905988` | interface-4 angular momentum |

The response is linear to displayed precision over this amplitude range. The
smallest counterexample is the predeclared `1.0e-3` multiplier.

### Independent face-58 switch witness

| Weighted radius | Opposite pair controllers at face 58 | Maximum half-spread |
|---:|---:|---:|
| `0.0055323802` | no | `0.27183474` |
| `0.0058177345` | no | `0.28585571` |
| `0.0058293816` | yes | `0.28642800` |
| `0.0061147360` | yes | `0.30044897` |

All four pairs independently exceed `0.25`. The half-spread varies smoothly
through the exact Rusanov controller switch. Therefore the closure failure is
not created solely by nonsmooth candidate selection: unresolved
angular-momentum transport exists on both sides of the switching surface.

## Exact-lift and physical gates

Across all eight N64 pairs and the one N128 confirmation pair:

| Quantity | Measured maximum/minimum | Gate |
|---|---:|---:|
| Per-side normalized coordinate defect | `1.78e-15` | `<=1e-10` |
| Pairwise normalized coordinate defect | `3.55e-15` | `<=2e-10` |
| Weighted radius | `0.006115` | `<=1` |
| Pointwise declared-amplitude ratio | `0.015504` | `<=1` |
| Normal-space correction fraction | `0.001004` | `<=0.25` |
| Weighted direction cosine | `0.99999950` | `>=0.99` |
| Reconstruction admissibility factor | exactly `1` | inactive unity branch |
| State gates | all passed | all required |

The constraint-normal condition estimates are `3.28e7` at N64 and `3.95e7`
at N128, below the `1e10` gate.

## Fresh nonlinear rates

Fresh coarse-coordinate rates were evaluated for the decisive N64 pair and
the N128 confirmation pair. Each primitive rate came from the branch-frozen
Track-A nonlinear vector field with fixed anchor primitive and conservation
scales. The coordinate rate was then obtained independently by centered
directional differentiation of the exact nonlinear coordinate-value map:

\[
O_{\dot C}=0.025\,\frac{dC/dt}{C_{\rm scale}}.
\]

The three directional scaled steps were `5e-5`, `1e-4`, and `2e-4`.

- maximum coordinate-rate step-ladder defect: `4.16e-9 < 0.005`;
- maximum branch-frozen vector-field step defect: `2.66e-7 < 0.005`;
- storage-component reconstruction defect: exactly zero at displayed
  precision;
- all fresh-rate gates pass on both sides and both meshes.

The full output stacks contain 255 N64 rows and 319 N128 rows. The controlling
row remains interface-4 angular-momentum flux after the fresh rate rows are
added.

## Decisive-pair descriptor and storage certification

The two sides of the decisive N64 pair and both sides of its N128
confirmation were independently re-audited against the full-DAE/Schur
descriptor. The Schur matrices were converted into the frozen WP10c8i
primitive/conservation scaling before comparison. All four sides pass:

| Quantity | Worst measured value | Gate |
|---|---:|---:|
| Descriptor rank | `320/320` N64; `640/640` N128 | full rank |
| Descriptor condition estimate | `4.54054e9` | `<=1e12` |
| Track-A versus full-Schur descriptor defect | `8.72e-11` | `<=1e-8` |
| Track-A versus full-Schur primitive-rate defect | `1.79e-8` | `<=5e-3` |
| Scaled descriptor-balance defect | `3.46e-16` | `<=1e-10` |
| Independent path/matrix storage-action defect | `1.66e-7` | `<=5e-5` |
| Storage-action step defect | `4.96e-7` | `<=5e-3` |
| Forbidden responsive-height mass/stress action | exactly `0` | `<=1e-12` |

The responsive-height action is independently nonzero in radial momentum,
angular momentum, and Killing energy on every audited side. Full-Schur
algebraic solve/reconstruction defects remain below `5.18e-17`, and the
descriptor algebraic row is zero at displayed precision. The rebuilt anchor
interface-flux scales agree with the frozen parent scales within `6.29e-16`;
the frozen parent scales, not recomputed pair scales, normalize every reported
interface flux.

## Cross-mesh confirmation

For the decisive pair, the fixed anchor-scaled interface-4 angular-momentum
flux values are

```text
N64  minus/plus = -0.163057084 / -0.162408024
N128 minus/plus = -0.162999725 / -0.162467554
```

The N128 spread is about `18.0%` below the N64 spread, but both independently
cross the locked `0.25` rejection gate and select the same physical output.
Their absolute gate-normalized spread disagreement is `0.05844445`, below the
predeclared `0.10` cross-mesh compatibility gate. This is sufficient for the
one-sided two-mesh counterexample. It is not a cross-mesh convergence
certification of a future reduced model or a continuum extrapolation.

## Interpretation

The 34 retained moments do not identify the instantaneous outward
angular-momentum transport through the outer internal shell interface near
the source region. Equal shell storage, shell thermodynamics, radial momentum,
stress storage, and selected shape moments still permit materially different
torque/flux states.

Consequently:

- do not build a deterministic algebraic 34-coordinate slow-time ODE;
- do not repeat WP10c8i or add generic moments blindly;
- do not treat exact Rusanov switching as the primary obstruction;
- retain the full DAE as the truth model;
- classify whether the measured transport direction heals, persists as
  memory, or requires a conservative coarse transport field.

## Updated next plan: WP10c8p

1. Freeze the exact N64/N128 decisive pair and its provenance. Do not optimize
   another direction.
2. Start both lifted states with BDF1; the truth-checkpoint BDF2 histories are
   invalid after lifting. Run matched, unconstrained production-DAE
   microbursts to exact `0.01` and `0.025 s` offsets.
3. Record the same full output half-spread, the 34-coordinate separation, all
   physical ledgers, and deterministic replay. This is natural healing, not
   constrained healing.
4. Classify the result:
   - decay below `0.10`, by at least a factor of two, with no late regrowth:
     a healed/equation-free closure remains possible, but raw algebraic
     closure stays rejected;
   - persistent spread above `0.10`: retain memory or a transport coordinate;
   - one approximately exponential relaxation: test one dynamic auxiliary;
   - several distributed/non-monotone responses: prefer a conservative coarse
     effective PDE.
5. If drift of the retained coordinates prevents a fair fiber comparison,
   implement a separate audit-only augmented DAE with 34 Lagrange multipliers.
   Ledger the artificial constraint impulse/work separately. Do not describe
   that system as the unmodified physical DAE and do not use post-step
   projection.
6. Only after the healing classification may one add exactly one measured
   transport coordinate—most plausibly the interface-4 angular-momentum
   flux/torque—or one dynamic memory variable, then repeat the same paired
   N64/N128 test.
7. No factor-two/factor-four macrostep, loading-time evolution, tide, wind,
   hot-state, or cycle search is authorized.

## Evidence

Primary machine-readable result:

- `outputs/tables/causal_nonlinear_fiber_audit_wp10c8o.json`
- `outputs/tables/causal_nonlinear_fiber_audit_wp10c8o_arrays.npz`

The NPZ retains the exact signed states, coordinates, increments, coordinate
and output-name schemas, static and fresh-rate outputs, native/common-radius
thickness profiles, interface fluxes, descriptor singular values, independent
path/matrix storage actions, and controller codes.

Artifact SHA-256 values:

```text
JSON d882e17c0de929f8b06e1d993121ef4f207221daad3195a531abae5f85f71af0
NPZ  3642fa56c70f1928f84d9ff5467b1f6f70de5ebb30fb9fad315cb16e48ac54cd
```

## Verification

- focused nonlinear-fiber and moment-audit tests: `14 passed`;
- repository suite: `690 passed, 4 subtests passed`;
- repository hygiene: passed for `712` tracked files;
- Python compilation and `git diff --check`: passed;
- NPZ artifact hash, every listed source hash, and parent evidence hashes:
  verified.
