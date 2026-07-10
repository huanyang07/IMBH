# Mdot=5 Global Conservative Flux Variable Results

Date: 2026-07-08

## Target

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source, local-Mdot wind
- `eta_E = 98.125`
- `N = 164`
- start checkpoint:
  `outputs/checkpoints/m5_eta_global_fv_mass_sourceband_correct_98p125_N164/stage_00_etaE_98p125_N164.npz`

## Implementation

Added an opt-in conservative mass-flux local block to
`scripts/run_mdot5_local_mdot_eta_continuation.py`.

New controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_VARIABLE=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_DIAGNOSTIC_ONLY=1`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_MIN_RG`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_MAX_RG`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_MAX_NFEV`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_SOURCE_GUARD_WEIGHT`
- `IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_MASS_FLUX_SKIP_SOURCE_BAND_DYNAMICS`

The local unknown vector is now

```text
X = { logu_i, logT_i, F_i },  F_i = Mdot_i / Mdot_inner
```

inside the selected radius window. The code maps `F_i` back to `logMdot_i`
only when evaluating the existing disk equations.

The active flux mass row is

```text
F_{i+1} - F_i - int_i(Mdot_wind_prime - Mdot_stream_prime)dlnR / Mdot_inner = 0
```

The diagnostic closure

```text
F_i - 2*pi*R_i*Sigma_i*u_i/Mdot_inner
```

is identically small in this code because `Sigma` is still algebraically
computed from the tabulated local `Mdot`.

I also added fixed source-band replacement guard rows. These reuse the
source-band HS/FV auxiliary state from the checkpoint and its sparse pattern,
so the optimizer is discouraged from fixing broad FV mass by damaging the
source-band representation.

## Verification

```text
python3 -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
pytest -q
```

Result:

```text
160 passed, 2 subtests passed
```

## Key Diagnostic

The flux-variable derivative audit confirms GPT's diagnosis:

```text
dFV/dF_left       ~ 0.999999
dFV/dF_right      ~ 1.000000
dFV/dlogT_left    ~ 1.36e-5
dFV/dlogu_left    ~ 2.13e-6
```

So the broad FV defect is dominated by the mass-flux representation, not by
thermodynamic response.

## Runs

| run | nfev | alpha | local FV | global FV | outside old | compat | local R | local E | peak R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic 40-120 | 0 | 0 | 3.808e-4 | 3.736e-4 | 1.263e-5 | 1.263e-5 | 1.166e-5 | 4.113e-6 | 69.754 |
| stage1 40-120 | 4 | 1.287e-3 | 3.803e-4 | 3.731e-4 | 2.088e-5 | 2.088e-5 | 1.171e-5 | 4.120e-6 | 69.754 |
| stage2 20-150 | 5 | 1.287e-3 | 3.803e-4 | 3.731e-4 | 2.197e-5 | 2.197e-5 | 1.174e-5 | 4.514e-6 | 69.754 |
| stage3 5-330 | 18 | 1.000e-4 | 3.808e-4 | 3.736e-4 | 2.329e-5 | 2.329e-5 | 5.405e-2 | 1.113 | 69.754 |
| stage2 guard1 | 3 | 9.103e-2 | 3.659e-4 | 3.590e-4 | 1.559e-5 | 1.559e-5 | 1.147e-5 | 4.985e-6 | 69.754 |
| stage2 guard5 | 8 | 2.133e-1 | 3.465e-4 | 3.399e-4 | 1.726e-5 | 1.726e-5 | 1.107e-5 | 6.373e-6 | 69.754 |
| stage2 guard20 | 8 | 2.133e-1 | 3.465e-4 | 3.399e-4 | 1.677e-5 | 1.677e-5 | 1.106e-5 | 6.446e-6 | 69.754 |
| guard20 pass2 | 8 | 2.133e-1 | 3.196e-4 | 3.134e-4 | 2.785e-5 | 2.785e-5 | 9.877e-6 | 9.853e-6 | 69.754 |
| guard20 pass3 | 8 | 5.000e-1 | 2.701e-4 | 2.648e-4 | 4.767e-5 | 4.767e-5 | 1.312e-5 | 1.686e-5 | 69.754 |
| guard20 pass4 long | 24 | 1.000 | 2.227e-4 | 2.167e-4 | 6.735e-5 | 6.735e-5 | 1.785e-5 | 2.269e-5 | 150.176 |

## Interpretation

The flux variable is the right direction: it directly exposes the conserved
mass-flux degree of freedom and reduces the global FV mass defect from
`3.736e-4` to `2.167e-4`.

However, this is not yet a certified solution. The final peak moves to the
edge of the active window near `R ~ 150 rg`, and the source/outside guard grows
to `~6.7e-5`. This means the current local block is still exporting part of the
conservative mismatch to the block boundary.

The full-window `5-330 rg` attempt without source-band dynamics handling is
not meaningful: it reactivates old source-band radial/energy rows and sees
`R~5.4e-2`, `E~1.113`.

I added `GLOBAL_MASS_FLUX_SKIP_SOURCE_BAND_DYNAMICS=1` to allow future
wide-window runs where old source-band dynamics are skipped and the HS/FV
source-band guard rows carry that region. A first `20-300 rg` run with this
mode was stopped after about nine minutes with no output, so the wide mode now
needs a more efficient Jacobian or staged-window strategy before it is practical.

## Current Bottleneck

The original `logMdot` representation problem is confirmed, but replacing it
locally with `F` is not sufficient unless the conservative flux formulation is
made global or smoothly staged across the wind-active region. The next obstacle
is block-boundary/source-guard coupling, not wind physics.

## Suggested Next Step

Implement the flux variable as a true production/global formulation rather than
a local correction:

1. Replace all production mass rows by `F` FV rows in the square system.
2. Keep `F` as the primary mass variable in the unknown vector, not a local
   temporary remapping to `logMdot`.
3. Use source-band HS/FV rows inside the source band and flux FV rows outside.
4. Add sparse analytic/local derivatives for the FV mass rows so the wide
   `20-300 rg` and full-disk solves are affordable.
5. Only then retry eta continuation below `eta_E=98.125`.
