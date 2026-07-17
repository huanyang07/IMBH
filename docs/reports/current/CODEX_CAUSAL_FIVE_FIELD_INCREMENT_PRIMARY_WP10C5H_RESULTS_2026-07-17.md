# Causal Five-Field Increment-Primary WP10c5h Results

Date: 2026-07-17

## Verdict

The complete five-field causal DAE passes its first source-free
increment-primary startup gate at both N16 and N32.

```text
N16 count/rank                         245/245
N32 count/rank                         485/485
N16 maximum scaled residual            8.80e-9
N32 maximum scaled residual            3.64e-9
N16 full/two-half relative error        2.76e-6
N32 full/two-half relative error        1.01e-6
decision                                STARTUP GATE PASSED
```

This unlocks short, adaptive, no-tide causal evolution. It does not certify a
physical relaxed state, stability, a hot advective state, or a limit cycle.
The audited context has no stream source, and tide and wind remain blocked.

## Increment-Primary DAE

The physical equations and the exact `15N+5` count are unchanged. Newton now
solves for

```text
(Delta U_cell, Delta p_cell, Delta F_face).
```

The new absolute state is

\[
U^{n+1}=U^n+\Delta U,\qquad
p^{n+1}=p^n+\Delta p,\qquad
F^{n+1}=F^n+\Delta F.
\]

The conservation rows use the declared conserved increment directly:

\[
\frac{{\cal V}_i\Delta U_i}{c\Delta t}
+F_{i+1/2}^{n+1}-F_{i-1/2}^{n+1}
-S_i^{n+1}
+W_{H,i}^{n+1}=0.
\]

Primitive recovery and numerical face fluxes remain algebraic rows evaluated
at the new state:

\[
U^{n+1}-U(p^{n+1})=0,
\qquad
F^{n+1}-F_{\rm num}(p^{n+1})=0.
\]

The responsive-height temporal one-form retains the path-integrated
representation certified by WP10c5g. Conserved storage no longer uses
endpoint subtraction or a finite-difference path integral.

## Focused Contracts

Four focused tests establish:

1. zero increment reproduces the stationary residual and zero storage;
2. a declared conserved increment enters storage exactly as
   \({\cal V}\Delta U/(c\Delta t)\);
3. a resolved increment agrees with the old endpoint form;
4. the small-grid increment-primary backward-Euler Jacobian is full rank.

No equation, source, boundary condition, tolerance, or physical closure was
changed.

## Count And Rank

The tiny timesteps make the un-equilibrated storage block scale as
\(1/\Delta t\). With the existing state-aware physical scaling, a raw
relative SVD rank test therefore reports `229/245` at N16 and `453/485` at
N32 even though the square Newton systems solve cleanly.

The declared rank gate uses standard LAPACK row/column equilibration before
the numerical-rank decision. This operation changes units, not equations or
rank.

| Mesh | Count | Equilibrated rank | Equilibrated condition |
|---:|---:|---:|---:|
| N16 | `245 x 245` | `245/245` | `8.50e6` |
| N32 | `485 x 485` | `485/485` | `7.44e6` |

The same full ranks are retained at the last evaluated Newton matrices and at
both half steps.

## Tiny-Step Results

Both meshes use the first declared target, a maximum scaled primitive change
of `1e-4`.

| Quantity | N16 | N32 |
|---|---:|---:|
| Timestep (s) | `1.56892e-8` | `2.27952e-8` |
| Maximum scaled change | `9.99679e-5` | `9.99687e-5` |
| Maximum scaled residual | `8.79746e-9` | `3.64410e-9` |
| Maximum scaled algebraic residual | `4.50e-15` | `2.53e-15` |
| Ledger defect | `1.99e-17` | `3.45e-17` |
| Minimum scattering depth | `1.70e4` | `1.70e4` |
| Newton iterations | `2` | `2` |

The Roche boundary remains on the same closed active set. No clipping,
projection, tolerance relaxation, or timestep retry is used.

## Equal-Physical-Time Check

One full backward-Euler step is compared with two independently solved half
steps. The second half is started from the accepted midpoint, uses a
state-aware scale at that midpoint, and preserves all rank, residual,
algebraic, optical-depth, active-set, and ledger gates.

| Quantity | N16 | N32 |
|---|---:|---:|
| Maximum scaled full-step change | `9.99679e-5` | `9.99687e-5` |
| Full versus two-half error | `2.75585e-10` | `1.00915e-10` |
| Relative temporal error | `2.75674e-6` | `1.00946e-6` |
| Declared relative gate | `5.0e-2` | `5.0e-2` |

The N16 error is controlled by the face-flux block. The N32 error is
controlled by the primitive block. The decrease under refinement argues
against a hidden coarse-grid cancellation.

## Classification

WP10c5h certifies for its numerical scope:

- direct primary conserved storage in the complete flux-primary DAE;
- square and equilibrated-full-rank N16 and N32 Newton systems;
- accepted bounded tiny steps at both meshes;
- equal-physical-time full-step versus two-half-step consistency;
- unchanged physical active set and roundoff global ledgers.

WP10c5h does not certify:

- source-on evolution;
- repeated or long-time evolution;
- a stationary or relaxed causal state;
- stability, a hot branch, or a limit cycle;
- a physical tide or wind.

## Locked Next Work

The next package is WP10c5i, not a hot-state search.

1. Freeze the dense N16/N32 result as the reference residual and Jacobian.
2. Add the exact compact stream moments to the causal context and repeat one
   bounded source-on N16/N32 startup and temporal comparison.
3. Implement one sparse/local Jacobian backend for the same increment-primary
   residual. Require identical accept/reject decisions and a state difference
   below `0.1` times the measured temporal error.
4. Run a short adaptive no-tide source-on sequence at N16, including rejected
   step recovery and a bitwise restart.
5. Reach the same physical time at N32 and compare conserved fluxes, primitive
   profiles, Roche state, optical depth, and model-validity diagnostics.
6. Only after that repeated-step mesh gate may N64/N96 or a longer no-tide
   relaxation be attempted.

Distributed tide remains later. Wind remains after the no-wind tidal and
stability behavior is understood.

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_five_field_consistent_initial_step_wp10c5d.py \
  --increment-primary-audit \
  --output \
  outputs/tables/causal_five_field_increment_primary_wp10c5h.json
```

The machine-readable output is generated under ignored `outputs/` in
accordance with the artifact policy.

Repository verification:

```text
479 tests passed
4 subtests passed
repository hygiene passed for 621 tracked files
Python compile checks passed
git diff --check passed
```
