# GPT Review Prompt: Standard No-Wind Slim-Disk Benchmark Failure

Please review the latest repository state:

```text
https://github.com/huanyang07/IMBH
```

Start with these files:

```text
Note/CODEX_STANDARD_NO_WIND_SLIM_DISK_BENCHMARK_PLAN.md
scripts/run_standard_slim_benchmark_thin_limit.py
outputs/tables/slim_benchmark_thin_limit.md
outputs/tables/slim_benchmark_thin_limit.json
outputs/figures/slim_benchmark_thin_limit_profiles.png
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
src/imri_qpe/layer3_minidisk_1d/transonic_local.py
src/imri_qpe/layer3_minidisk_1d/isolated_slim_solver.py
```

Context:

We previously found that the IMRI/minidisk no-wind branch is blocked by a
radial projection fold near the inner region. You suggested a control
benchmark: before adding wind, test whether the same code can reproduce a
standard single-BH no-wind slim disk.

I implemented the first low-Mdot thin-limit benchmark:

```text
scripts/run_standard_slim_benchmark_thin_limit.py
```

Configuration:

```text
Paczynski-Wiita single-BH potential
no wind, no stream, no tide
constant Mdot
alpha = 0.01
stress_factor = 1.0
R_out ladder = 300 -> 1000 -> 10000 rg
N = 24 first-pass low-resolution control
Mdot/Mdot_Edd = 1e-4, 1e-3, 1e-2
Mdot_Edd = L_Edd/(0.1 c^2)
```

Result summary:

The benchmark does **not** pass.

Sanity checks pass:

```text
PW dlnOmegaK/dlnR check: pass
PW lK minimum near 6 rg: pass
vertical state positivity: pass
residual scales finite: pass
```

Thin-limit rows:

```text
Mdot/Edd = 1e-4:
    accepted = no
    residual = 3.137e-4
    dominant = interval_R
    sonic = no
    outer_thin = yes
    Omega error = 4.40e-3
    Qadv/Qvisc = 1.50e-6
    H/R max = 2.21e-3
    PW nuSigma median error = 2.04e-2
    quick fold audit = no_fold_within_smax

Mdot/Edd = 1e-3:
    accepted = no
    residual = 6.363e-5
    dominant = C2
    sonic = no
    outer_thin = yes
    Omega error = 4.34e-3
    Qadv/Qvisc = 2.56e-6
    H/R max = 3.44e-3
    PW nuSigma median error = 3.81e-2
    quick fold audit = no_fold_within_smax

Mdot/Edd = 1e-2:
    accepted = no
    residual = 9.948e-1
    dominant = outer_energy
    sonic = yes
    outer_thin = no
    Omega error = 9.14e-2
    Qadv/Qvisc = 2.95e-4
    H/R max = 6.64e-3
    PW nuSigma median error = 4.14e-2
    quick fold audit = no_fold_within_smax
```

Interpretation so far:

The low-Mdot solutions look qualitatively thin for 1e-4 and 1e-3:

```text
Omega/OmegaK is close to 1
Qadv/Qvisc is tiny
H/R is small
PW nuSigma agreement is a few percent
no quick phase-space fold is found
```

But the full transonic benchmark still fails because sonic regularity and/or
global residual polish are not good enough. At 1e-2, the final outer extension
to R_out=1e4 becomes poor and is dominated by the far outer energy residual.

Important implementation detail:

The direct jump to R_out=1e4 was too slow, so the script uses an outer-radius
ladder:

```text
R_out = 300 -> 1000 -> 10000 rg
```

The first stage uses `solve_low_mdot_transonic_homotopy`; later stages use
`solve_transonic_outer_branch` from remapped profiles.

Questions for review:

1. Does this failure look mainly like a numerical continuation/polish problem,
   or does it indicate a formulation problem in the transonic equations?
2. For the 1e-4 and 1e-3 rows, should we first fix sonic regularity polishing
   at fixed outer radius, or improve outer-radius continuation/remapping?
3. For the 1e-2 row, is the outer-energy failure likely caused by the far
   thermal boundary, the low N=24 resolution, the direct extension strategy, or
   the thin boundary condition itself?
4. Which checks should be run next to distinguish:
   - radial momentum sign or pressure-gradient sign error;
   - stress normalization / stress_factor convention;
   - one-face vs two-face cooling convention;
   - angular-momentum integral normalization;
   - sonic compatibility scaling or pivot choice;
   - insufficient staged continuation or mesh resolution?
5. What is the most efficient next sprint before trying the higher-Mdot slim
   sequence or returning to the IMRI wind model?

Please propose a prioritized fix/test plan. The goal is to make the standard
single-BH no-wind thin-limit benchmark pass before interpreting the IMRI
minidisk fold physically.
