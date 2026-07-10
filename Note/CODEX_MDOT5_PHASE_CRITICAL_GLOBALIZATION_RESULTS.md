# Mdot=5 phase critical-point classification and globalization

Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.

## Exact homogeneous DAE audit

Production now uses the direct homogeneous residual `H(z,p)`; the divided radial residual is audit-only.

| checkpoint | Rcrit (rg) | p_R | sigma_min(A) | cond(A) | u_min^T c | null alignment | dH/dp rank | phase J smin | phase J cond | H/direct max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| refine2 | 223.395553 | 2.751e-02 | 1.206e-03 | 3.784e+03 | 6.666e-02 | 0.999776 | 3 | 6.033e-05 | 2.798e+06 | 8.305e-13 |
| quarter | 223.980659 | 1.726e-02 | 9.899e-04 | 4.680e+03 | 7.715e-02 | 0.999912 | 3 | 1.270e-04 | 1.330e+06 | 5.613e-13 |
| half | 224.566890 | 8.925e-03 | 7.260e-04 | 6.503e+03 | 9.775e-02 | 0.999976 | 3 | 1.607e-04 | 1.053e+06 | 9.900e-14 |
| threequarter | 225.155325 | 3.276e-03 | 3.993e-04 | 1.220e+04 | 1.291e-01 | 0.999997 | 3 | 1.294e-04 | 1.310e+06 | 8.100e-14 |
| f8125 | 225.302805 | 1.924e-03 | 2.658e-04 | 1.869e+04 | 1.413e-01 | 0.999999 | 3 | 1.208e-04 | 1.401e+06 | 2.846e-12 |
| f84375 | 225.376624 | 1.290e-03 | 1.881e-04 | 2.676e+04 | 1.486e-01 | 1.000000 | 3 | 1.162e-04 | 1.455e+06 | 2.312e-12 |
| f859375 | 225.413558 | 9.639e-04 | 1.452e-04 | 3.492e+04 | 1.531e-01 | 1.000000 | 3 | 1.229e-04 | 1.375e+06 | 2.525e-12 |
| f875 | 225.450457 | 6.572e-04 | 1.029e-04 | 4.960e+04 | 1.589e-01 | 1.000000 | 3 | 1.329e-04 | 1.272e+06 | 1.880e-12 |
| f8828125 | 225.468943 | 4.979e-04 | 7.984e-05 | 6.409e+04 | 1.624e-01 | 1.000000 | 3 | 1.342e-04 | 1.262e+06 | 1.546e-13 |

## Moving cut-cell interface

| fraction | Rint (rg) | cut radial | cut energy | global FV mass | outside radial | outside energy | interface FV energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500000 | 224.566890 | 1.481e-01 | 3.941e+00 | 4.115e-03 | 1.481e-01 | 3.941e+00 | 2.748e-01 |
| 0.750000 | 225.155325 | 3.222e-01 | 9.488e+00 | 4.212e-03 | 3.222e-01 | 9.488e+00 | 2.103e-01 |
| 0.843750 | 225.376624 | 4.906e-01 | 1.547e+01 | 4.227e-03 | 4.906e-01 | 1.547e+01 | 2.025e-01 |
| 0.882812 | 225.468943 | 6.013e-01 | 2.282e+01 | 4.245e-03 | 6.013e-01 | 2.282e+01 | 1.690e-01 |

### Coupled source-tail cut-cell corrector

| fraction | initial max | final max | cut radial | cut energy | cut mass | interface energy | source max | right drift | accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.500000 | 3.941e+00 | 5.640e-01 | 1.287e-01 | 7.170e-04 | 1.777e-02 | 3.568e-04 | 5.640e-01 | 5.460e-03 | False |
| 0.750000 | 9.488e+00 | 9.220e-01 | 1.797e-01 | 2.136e-03 | 4.130e-03 | 1.842e-03 | 9.220e-01 | 1.009e-02 | False |
| 0.843750 | 1.547e+01 | 4.376e-01 | 4.385e-02 | 3.642e-02 | 2.369e-02 | 7.225e-04 | 4.376e-01 | 4.044e-03 | False |
| 0.882812 | 2.282e+01 | 1.007e+00 | 2.634e-01 | 1.353e-02 | 3.537e-02 | 1.574e-05 | 1.007e+00 | 1.032e-02 | False |

## Signed arclength continuation

| step | ds | R (rg) | p_R | radial | energy | F-prime | cond(A) | accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2.000e-02 | 225.471136 | 4.754e-04 | 3.385e-17 | 2.045e-12 | 2.855e-14 | 6.672e+04 | yes |
| 1 | 2.300e-02 | 225.473540 | 4.517e-04 | 4.331e-17 | 2.231e-12 | 3.091e-14 | 6.988e+04 | yes |
| 2 | 2.645e-02 | 225.476156 | 4.261e-04 | 3.721e-17 | 2.125e-12 | 2.958e-14 | 7.369e+04 | yes |
| 3 | 3.042e-02 | 225.478983 | 3.984e-04 | 8.560e-17 | 1.615e-12 | 2.228e-14 | 7.832e+04 | yes |
| 4 | 3.498e-02 | 225.482007 | 3.690e-04 | 1.684e-17 | 5.085e-15 | 3.768e-18 | 8.398e+04 | yes |
| 5 | 4.023e-02 | 225.485211 | 3.379e-04 | 8.757e-18 | 1.182e-15 | 5.497e-17 | 9.099e+04 | yes |
| 6 | 4.626e-02 | 225.488564 | 3.055e-04 | 1.486e-18 | 1.375e-15 | 8.958e-17 | 9.974e+04 | yes |
| 7 | 5.000e-02 | 225.491828 | 2.741e-04 | 1.369e-17 | 1.056e-15 | 2.336e-17 | 1.101e+05 | yes |
| 8 | 5.000e-02 | 225.494758 | 2.461e-04 | 4.506e-18 | 4.232e-15 | 2.675e-17 | 1.215e+05 | yes |
| 9 | 5.000e-02 | 225.497389 | 2.211e-04 | 1.139e-17 | 3.889e-15 | 6.595e-17 | 1.340e+05 | yes |
| 10 | 5.000e-02 | 225.499753 | 1.987e-04 | 6.947e-18 | 3.999e-15 | 7.909e-17 | 1.478e+05 | yes |
| 11 | 5.000e-02 | 225.501878 | 1.786e-04 | 1.143e-17 | 4.624e-15 | 3.060e-17 | 1.628e+05 | yes |
| 12 | 5.000e-02 | 225.503788 | 1.606e-04 | 1.050e-17 | 6.659e-16 | 6.920e-17 | 1.794e+05 | yes |
| 13 | 5.000e-02 | 225.506102 | -2.671e-09 | 2.525e-05 | 2.083e-07 | 1.715e-06 | 1.975e+05 | yes |
| 14 | 5.000e-02 | 225.506103 | -2.037e-05 | 1.880e-05 | 8.151e-06 | 3.243e-06 | 2.174e+05 | yes |
| 15 | 5.000e-02 | 225.505881 | -2.416e-05 | 6.917e-06 | 1.604e-05 | 7.988e-07 | 2.393e+05 | yes |
| 16 | 5.000e-02 | 225.505662 | -2.273e-05 | 8.941e-07 | 2.055e-05 | 1.711e-08 | 2.632e+05 | yes |
| 17 | 5.000e-02 | 225.505495 | -1.687e-05 | 1.198e-06 | 1.928e-05 | 4.644e-08 | 2.894e+05 | yes |
| 18 | 5.000e-02 | 225.505368 | -1.688e-05 | 1.287e-06 | 1.886e-05 | 4.924e-08 | 3.180e+05 | yes |
| 19 | 5.000e-02 | 225.505241 | -1.610e-05 | 1.350e-06 | 1.831e-05 | 4.984e-08 | 3.492e+05 | yes |
| 20 | 5.000e-02 | 225.505120 | -1.543e-05 | 1.425e-06 | 1.783e-05 | 5.166e-08 | 3.834e+05 | yes |
| 21 | 5.000e-02 | 225.505006 | -1.472e-05 | 1.500e-06 | 1.738e-05 | 5.347e-08 | 4.206e+05 | yes |
| 22 | 5.000e-02 | 225.504899 | -1.400e-05 | 1.574e-06 | 1.696e-05 | 5.509e-08 | 4.612e+05 | yes |

### Arclength step-size gate

| ds | accepted steps | total s | crossed | last positive R (rg) | last positive p_R | last positive logu |
|---:|---:|---:|---|---:|---:|---:|
| 0.0500 | 23 | 1.021 | True | 225.503788 | 1.606e-04 | 10.821851 |
| 0.0100 | 132 | 1.320 | True | 225.517084 | 3.760e-05 | 10.126430 |
| 0.0050 | 323 | 1.615 | True | 225.519182 | 1.859e-05 | 9.782404 |
| 0.0025 | 750 | 1.875 | False | 225.520147 | 9.901e-06 | 9.472817 |

Finite-state fold certified: `False`.
Interpretation: step-sensitive sheet switch; the resolved positive branch approaches a finite-radius, low-u singular limit.
Tail fit: `p_R ~ exp(-2.032 s)`, with estimated limiting radius `225.521245 rg`.

## Source and angular audits

- Critical radius: `225.468943 rg`.
- Distance from compact-source inner edge: `3.921020 rg` (`Delta lnR=1.754350e-02`).
- Source value/first/second log-radius derivatives: `5.328180e+22`, `7.989311e+24`, `6.707141e+26`.
- Angular FV audit maximum: `5.383457e-04`.
- Angular assumption: diagnostic uses l_w=l_disk and l_s=l_disk, with the configured cumulative stream torque derivative as tau_s/Mdot; this is not a production closure.

### Frozen-state source-shape diagnostic

| variant | inner edge (rg) | H radial | H energy | H F-prime | sigma_min(A) | null p_R |
|---|---:|---:|---:|---:|---:|---:|
| compact_c2 | 221.547923 | 1.396e-07 | -8.157e-08 | -1.077e-08 | 7.984e-05 | -1.408e-03 |
| compact_c4 | 221.547923 | 1.396e-07 | -8.157e-08 | -2.631e-04 | 7.984e-05 | -1.393e-03 |
| compact_cinf | 221.547923 | 1.396e-07 | -8.157e-08 | 5.419e-04 | 7.984e-05 | -1.441e-03 |
| compact_c2_wide | 217.160980 | 1.396e-07 | -8.157e-08 | 6.642e-04 | 7.984e-05 | -1.449e-03 |

## Classification

- Coarse-step signed p_R zero crossing recovered: `True`.
- Accepted arclength steps: `23`.
- Closest accepted critical radius: `225.506102 rg`.
- Second positive-p_R branch found: `False`.
- Step-size-certified finite-state fold: `False`.
- Branch classification: step-sensitive sheet switch; the resolved positive branch approaches a finite-radius, low-u singular limit.

The growing compatibility scalar rules out a regular critical crossing. The homogeneous tangent Jacobian remains full rank, so there is no detected DAE-index change. The apparent signed crossing is not stable under arclength refinement and is therefore retained only as a rejected diagnostic sheet switch.

The coupled moving-interface correctors also fail by four or more orders of magnitude relative to the exploratory gate. N164 global certification, a higher-N check, and eta continuation are therefore deferred.

## Reproducibility

- Primary table: `outputs/tables/m5_eta_phase_critical_globalization_98p125_N164.json`.
- Profile table: `outputs/tables/m5_eta_phase_critical_globalization_98p125_N164_profiles.json`.
- Diagnostic figure: `outputs/figures/m5_eta_phase_critical_globalization_98p125_N164.png`.
- Fine arclength checkpoints: `outputs/checkpoints/m5_eta_phase_critical_arc_ds001_98p125_N164/`, `outputs/checkpoints/m5_eta_phase_critical_arc_ds0005_98p125_N164/`, and `outputs/checkpoints/m5_eta_phase_critical_arc_ds00025_98p125_N164/`.
- Regression status: `166 passed, 4 subtests passed`.

Eta continuation remains paused until the phase/cut-cell composite satisfies the global conservation and exterior residual gates.
