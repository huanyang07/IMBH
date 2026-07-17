# Responsive-height thermal ledger WP10c3b results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** quasi-hydrostatic dynamic column height, physical column acoustic
tangent, optically thick cooling, vertical pressure work, comoving
four-force transformation, and non-double-counted stress work. No stream,
tide, wind, stationary disk root, or timestep was run.

## Verdict

WP10c3b passes its bounded local thermal/source gate:

```text
gas/radiation states audited                    9
maximum H/R in the audit                        0.217853
maximum primitive-recovery defect               5.3427e-13
maximum dynamic acoustic eigenvalue defect      8.3267e-17
maximum shear eigenvalue defect                 3.4695e-18
maximum light-cone excess                       0
inside-horizon incoming modes                   0
maximum comoving-source identity defect         1.1524e-15
maximum vertical-work identity defect           5.2320e-16
maximum local cooling/compression defect        8.0554e-26
minimum finite-volume source order              2.000016
N128 maximum source-integral error              1.1594e-5
temporal work antisymmetry defect               0
torque-work product-rule defect                 0
explicit total-energy viscous source            0
```

The responsive-height thermodynamics, cooling, vertical work, and causal
stress now share one local energy convention. This is not yet a production
disk solution.

## Responsive vertical closure

The selected local closure is

```text
Sigma = 2 rho H,
P     = R_g rho T + a T^4/3,
Pi    = 2 H P,
Omega_perp^2 H^2 = Pi/Sigma.
```

The positive quadratic root defines `H(Sigma,T,Omega_perp)`. Analytic
derivatives with respect to all three inputs agree with independent centered
finite differences. Including the frequency derivative is important: a
global flow moving through nonuniform gravity must use

```text
dlnH/dtau
    = H_Sigma dlnSigma/dtau
    + H_T dlnT/dtau
    + H_Omega dlnOmega_perp/dtau.
```

WP10c3b deliberately accepts `Omega_perp` from a provider. The audit runner
uses the Newtonian orbital frequency only to construct bounded test states.
That choice is not promoted as the unique proper vertical frequency for a
noncircular relativistic plunge.

## Physical acoustic tangent

The responsive height changes the principal thermodynamics. The correct
column adiabat is

```text
de - (P/rho^2) d rho = 0,
```

with `rho=Sigma/(2H)` constrained by the hydrostatic surface. The sound speed
is therefore

```text
a_col^2
    = c^2 (dPi/dSigma)_s
      / (c^2 + e + Pi/Sigma).
```

It is not the fixed-height three-dimensional gas+radiation derivative used by
WP10c1. A direct local-rest generalized-eigenvalue matrix, including the
temporal vertical-work term, gives

```text
(-a_col/c, 0, +a_col/c)
```

with a maximum defect of `8.33e-17`.

The existing standard conservative-flux Jacobian by itself is not an
independent test of this closure because the dynamic vertical-work term
changes the time-derivative mass matrix. The declared local DAE matrix is the
appropriate audit. The full nonlinear coupled system must repeat the
characteristic proof after all migrated sources and boundary variables are
present.

## Cooling and vertical work

The two-face diffusion cooling rate is

```text
Q_rad = 16 sigma_SB T^4/(3 kappa Sigma),
tau_sc = kappa Sigma/2.
```

The signed comoving rates are

```text
q_rad = -Q_rad,
q_H   = -Pi dlnH/dtau.
```

Compression therefore heats the column. Each exchange is transformed with

```text
G^mu = q u^mu/c^3.
```

The stationary Killing chart receives

```text
(0, alpha G_R, alpha G_phi, -alpha G_t).
```

Across all nine states, contraction with the four-velocity recovers the
declared comoving rate and the orthogonal comoving momentum vanishes below
`1.16e-15`. Choosing `dlnH/dtau=-Q_rad/Pi` cancels cooling locally in all four
Killing-source components to `8.06e-26` relative to the cooling source.

This cancellation is a source-identity test, not a claim that physical disk
compression will exactly balance cooling.

## Radial and temporal work identity

The enthalpy-compatible column identity is

```text
Pi/Sigma^2 dSigma - P/rho^2 d rho
    = (P/rho) dlnH.
```

It closes below `5.24e-16`. The temporal finite-volume correction

```text
(Pi_old+Pi_new) ln(H_new/H_old)/2
```

is exactly antisymmetric when the states are exchanged. It is a declared
second-order path approximation, not a new state function.

## Stress-work partition

WP10c3a already derives torque and Killing power from one covariant stress
tensor. WP10c3b keeps that ledger:

```text
Delta(Omega G)
    = Omegabar Delta G + Gbar Delta Omega.
```

The midpoint product rule closes exactly. The shear-conversion term may appear
in an internal-energy or entropy partition, but the explicit source in the
conservative total Killing-energy equation remains zero. Adding an independent
`Q_visc` there would double count stress work already transported by the
tensor flux.

## Finite-volume source gate

A smooth moving-column cooling/vertical-work source over `10-30 rg` was
integrated with exact Kerr-Schild column measures on `N=16,32,64,128`.
Radial momentum, angular momentum, and Killing-energy source integrals all
converge at second order. The minimum observed order is `2.000016`; the N128
maximum component error is `1.1594e-5`.

## Classification

```text
numerical status:
    supported but not fully certified for the local thermal/source contract

physical status:
    diagnostic only

production status:
    blocked
```

WP10c3b does not include:

1. a unique relativistic `Omega_perp` closure for the plunging region;
2. stream mass, angular-momentum, and energy moments;
3. the physical Hill/Roche boundary in the Kerr-Schild chart;
4. the final nonlinear coupled characteristic proof;
5. a stationary root or implicit evolution;
6. tide or wind.

## Locked next step

Proceed to WP10c4 only:

1. migrate exact stream mass/angular/energy moments into the one-domain
   Kerr-Schild finite-volume chart;
2. migrate the closed-to-choked Hill/Roche boundary with the same Killing and
   Jacobi ledgers;
3. make the `Omega_perp` provider explicit and auditable;
4. preserve the WP10c3a stress tensor and WP10c3b cooling/height-work source;
5. repeat local source, boundary, and characteristic/rank audits;
6. run no stationary root, long evolution, distributed tide, or wind.

Do not revive the old Paczynski-Wiita plunge or add another inner/outer splice.

## Verification

```text
focused thermal tests          13 passed
focused thermal/stress tests   33 passed
complete repository suite      440 passed, 4 subtests passed
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_thermal_wp10c3b.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_causal_inner_thermal_wp10c3b.py
```
