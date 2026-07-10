# Mdot=5 phase critical classification results

Compact endpoint, convergence, and source-shape evidence is retained under
`results/canonical/`. Raw trajectories remain available at tag
`pre-cleanup-p0-2026-07-11` and in the verified legacy archive.

Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.

## Low-u continuation

The critical branch was reparameterized by decreasing `logu`, with `p_logu=-1`. This removes `p_R` from the denominator and follows the positive radial sheet without clipping.

| dt | complete | final logu | final R (rg) | final p_R | fitted R limit (rg) | p_R decay rate | max H |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | True | 4.0000 | 225.521245 | 1.130e-10 | 225.521245 | 2.098 | 5.284e-09 |
| 0.0100 | True | 4.0000 | 225.521257 | 1.089e-10 | 225.521257 | 1.893 | 1.006e-07 |

### Physical asymptotics

| logu | R (rg) | p_R | Sigma | rho | H/R | tau | Mach | Qadv/Qvisc | Qwind/Qvisc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9.473 | 225.520147 | 9.907e-06 | 3.095e+06 | 2.260e-04 | 2.056e-02 | 5.261e+05 | 3.140e-04 | 9.997e-01 | 1.452e-27 |
| 8.563 | 225.521104 | 1.512e-06 | 7.688e+06 | 8.863e-04 | 1.302e-02 | 1.307e+06 | 1.995e-04 | 1.000e+00 | 1.227e-58 |
| 7.653 | 225.521236 | 1.854e-07 | 1.910e+07 | 3.472e-03 | 8.259e-03 | 3.247e+06 | 1.266e-04 | 1.000e+00 | 6.554e-90 |
| 6.733 | 225.521253 | 2.678e-08 | 4.793e+07 | 1.380e-02 | 5.213e-03 | 8.148e+06 | 7.995e-05 | 1.000e+00 | 4.593e-94 |
| 5.823 | 225.521257 | 4.413e-09 | 1.191e+08 | 5.405e-02 | 3.307e-03 | 2.024e+07 | 5.073e-05 | 1.000e+00 | 0.000e+00 |
| 4.913 | 225.521257 | 6.872e-10 | 2.958e+08 | 2.117e-01 | 2.098e-03 | 5.029e+07 | 3.219e-05 | 9.999e-01 | 1.414e-04 |
| 4.000 | 225.521257 | 1.089e-10 | 7.369e+08 | 8.323e-01 | 1.329e-03 | 1.253e+08 | 2.039e-05 | 9.997e-01 | 3.477e-04 |

Tail power laws `quantity proportional to u^a`:

- `Sigma`: `a=-1.0000`.
- `rho`: `a=-1.5000`.
- `H_over_R`: `a=0.5000`.
- `tau`: `a=-1.0000`.
- `Mach_eff`: `a=0.5000`.
- `p_R`: `a=1.9346`.

### Critical eigenstructure

| logu | R (rg) | p_R | sigma_min(A) | cond(A) | C=u_min^T c | null alignment | |dz/dlnR| | u_min | v_min |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 9.473 | 225.520147 | 9.907e-06 | 1.886e-06 | 1.973e+06 | 1.908e-01 | 1.000000 | 1.010e+05 | (1.000, 0.001) | (0.999, -0.036) |
| 8.383 | 225.521155 | 1.022e-06 | 1.894e-07 | 1.470e+04 | 2.000e-01 | 1.000000 | 1.017e+06 | (0.993, 0.120) | (0.962, 0.273) |
| 7.283 | 225.521247 | 8.226e-08 | 1.155e-08 | 1.459e+05 | 1.968e-01 | 1.000000 | 1.686e+07 | (0.999, 0.040) | (0.721, 0.693) |
| 6.193 | 225.521256 | 9.994e-09 | 1.252e-09 | 1.331e+06 | 1.956e-01 | 1.000000 | 1.415e+08 | (1.000, 0.013) | (0.707, 0.707) |
| 5.093 | 225.521257 | 9.838e-10 | 1.387e-10 | 1.201e+07 | 1.954e-01 | 1.000000 | 1.437e+09 | (1.000, 0.004) | (0.707, 0.707) |
| 4.000 | 225.521257 | 1.089e-10 | 1.559e-11 | 1.068e+08 | 1.953e-01 | 1.000000 | 1.299e+10 | (1.000, 0.002) | (0.707, 0.707) |

`sigma_min(A)` and `p_R` vanish while compatibility remains near `0.19` and the physical derivative diverges. This excludes a regular critical point. Because `u` tends to zero and density variables diverge, it is an asymptotic singular boundary rather than a finite-state fold.

## Bordered intrinsic corrector

| arc target | accepted steps | crossed p_R=0 | final R (rg) | final p_R |
|---:|---:|---|---:|---:|
| 0.0050 | 30 | False | 225.520435 | 7.297e-06 |
| 0.0025 | 30 | False | 225.520302 | 8.500e-06 |

## Re-solved source-shape branches

| branch | anchor R (rg) | source edge (rg) | complete | final p_R | fitted R limit (rg) | source cumulative |
|---|---:|---:|---|---:|---:|---:|
| compact_c2 | 194.578563 | 221.547923 | True | 6.166e-09 | 225.521265 | 0.004038 |
| compact_c4 | 194.578563 | 221.547923 | True | 6.101e-09 | 225.520418 | 0.000527 |
| compact_cinf | 194.578563 | 221.547923 | True | 6.247e-09 | 225.522669 | 0.010345 |
| compact_c2_wide | 194.578563 | 217.160980 | True | 6.806e-09 | 225.522515 | 0.027373 |

The full source-shape spread in fitted limiting radius is `0.002251 rg`, well below the predeclared `0.05 rg` sensitivity threshold.

## Angular closure audit

| assumed stream l_s | max FV defect | RMS |
|---|---:|---:|
| disk_local | 4.324e-08 | 9.478e-09 |
| keplerian_local | 4.987e-08 | 1.093e-08 |
| keplerian_injection | 5.192e-08 | 1.138e-08 |

This angular audit covers the low-u endpoint tail only. It supports closure consistency there, but it does not promote a stream angular-momentum law to the global production equations.

## Decision

- Classification: **finite-radius low-u stagnation/singular boundary**.
- Finite-state fold: `False`.
- Regular critical point: `False`.
- Source-shape audit resolved: `True`.
- Source-shape sensitive: `False`.
- Global steady branch certified: `False`.

The positive-p_R branch remains regular when logu is used as the continuation coordinate, but approaches u=0 at finite radius while surface density and optical depth diverge. No step-converged finite-state p_R sign change is found, and no admissible outer radial branch is available.

Eta continuation remains paused.

## Files

- table: `outputs/tables/m5_eta_phase_critical_classification_98p125_N164.json`
- profiles: `outputs/tables/m5_eta_phase_critical_classification_98p125_N164_profiles.json`
- figure: `outputs/figures/m5_eta_phase_critical_classification_98p125_N164.png`
