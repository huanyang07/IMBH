# Mdot=3 Buffer Sensitivity and Mdot=5 Compact Stream Results

Date: 2026-07-04

## Purpose

Before pushing the compact stream-fed no-wind branch from `Mdot_inner/Edd=3`
to `Mdot_inner/Edd=5`, I first checked whether the accepted
`Mdot_inner/Edd=3`, `f_s=0.50`, compact source solution is sensitive to the
outer-buffer placement. The goal was to separate a real high-rate obstruction
from the known outer-tail/buffer numerical closure issue.

## Shared Setup

- No wind, no stream heating.
- Compact C2 stream source with `source_shape_blend=1`.
- `Rout=335 rg`, `Rinj=240 rg`, torque injection radius `240 rg`.
- `torque_delta_l_fraction=+0.005`.
- Outer closure: `pressure_supported_thin_energy`.
- Outer-buffer weights `(R,E,B)=(0.001,0.001,0.0001)`.
- Energy merit off; physical interval-E gate on at `1e-5`.
- Residual-aware remeshing enabled every accepted step and on rejects.

## Mdot=3, f_s=0.50 Outer-Buffer Sensitivity

Reference checkpoint:

- `outputs/checkpoints/high_mdot_stream_m3_compact_cert_N896_fs050_no_energy_merit/m3n896fs050_mass_0p5_torque_0p005_mdot_3_N896.npz`

| R_buffer/rg | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 295 | 1.893e-09 | 1.893e-09 | 1.109e-09 | 0.500 | 0.3304 | 0.2671 | 1.081 | 0.2676 | 4.502 |
| 300 | 7.341e-11 | 5.269e-11 | 5.880e-11 | 0.500 | 0.3304 | 0.2663 | 1.081 | 0.2676 | 4.502 |
| 305 | 1.743e-09 | 1.743e-09 | 1.264e-09 | 0.500 | 0.3304 | 0.2664 | 1.081 | 0.2676 | 4.502 |

Output tables and figures:

- `outputs/tables/high_mdot_stream_m3_compact_cert_N896_fs050_no_energy_merit.md`
- `outputs/tables/high_mdot_stream_m3_fs050_buffer295_N896.md`
- `outputs/tables/high_mdot_stream_m3_fs050_buffer305_N896.md`
- `outputs/figures/high_mdot_stream_m3_fs050_buffer295_N896.png`
- `outputs/figures/high_mdot_stream_m3_fs050_buffer305_N896.png`

Interpretation: the solution passes this small buffer sensitivity check. The
full residual is smallest at the original `R_buffer=300 rg`, but the physical
diagnostics are effectively unchanged across `295,300,305 rg`. This supports
moving to `Mdot_inner/Edd=5`; the `Mdot=3`, `f_s=0.50` solution is not a
fragile artifact of one precise buffer radius.

## Mdot=5 No-Stream Parent

The finite-`Rout=300 rg` no-stream parent was continued from the existing
`Mdot=3` parent to `Mdot=5`.

Output:

- `outputs/tables/high_mdot_finite_Rout300_nowind_m3_to_m5_adaptive.md`
- `outputs/figures/high_mdot_finite_Rout300_nowind_m3_to_m5_adaptive.png`
- `outputs/checkpoints/high_mdot_finite_Rout300_nowind_m3_to_m5_adaptive/up_mdot_5.npz`

Final `Mdot_inner/Edd=5` no-stream parent:

- final full residual: `1.755e-10`
- dominant residual: `interval_E`
- `Rson=4.360 rg`
- `H/R=0.1016`
- accepted as a strong anchor

## Mdot=5 Compact Stream Scout to f_s=0.30

Starting from the no-stream `Mdot=5` parent, I ran a compact stream source
scout at `N=640` to `f_s=0.30`. The first nonzero stream step from `f_s=0`
had a poor tangent seed, so the adaptive controller shrank to `df_s=0.0125`.
Once on the branch, tangent prediction became effective and the ladder reached
`f_s=0.30`.

Output:

- `outputs/tables/high_mdot_stream_m5_compact_scout_N640_000_to030_no_energy_merit.md`
- `outputs/figures/high_mdot_stream_m5_compact_scout_N640_000_to030_no_energy_merit.png`
- `outputs/checkpoints/high_mdot_stream_m5_compact_scout_N640_000_to030_no_energy_merit/m5n640fast_mass_0p3_torque_0p005_mdot_5_N640.npz`

N640 endpoint at `f_s=0.30`:

- final full residual: `1.167e-07`
- physical `interval_E`: `3.829e-09`
- `Mdot_outer/Mdot_inner=0.700`
- `source_integral/Mdot_inner=0.3001`
- `f_adv_global=0.4881`
- `f_adv_inner=0.4711`
- `Lrad/LEdd=1.342`
- `max H/R=0.3158`
- `Rson=4.361 rg`

## N768 and N896 Certification at Mdot=5, f_s=0.30

The `f_s=0.30` endpoint was remapped and repolished at `N=768` and `N=896`.

| N | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 | 1.167e-07 | 3.829e-09 | 5.509e-08 | 0.700 | 0.4881 | 0.4711 | 1.342 | 0.3158 | 4.361 |
| 768 | 9.732e-08 | 2.423e-09 | 7.053e-08 | 0.700 | 0.4881 | 0.4715 | 1.342 | 0.3158 | 4.361 |
| 896 | 7.491e-08 | 1.852e-09 | 1.430e-08 | 0.700 | 0.4881 | 0.4706 | 1.342 | 0.3158 | 4.361 |

Output:

- `outputs/tables/high_mdot_stream_m5_compact_cert_N768_fs030_no_energy_merit.md`
- `outputs/tables/high_mdot_stream_m5_compact_cert_N896_fs030_no_energy_merit.md`
- `outputs/figures/high_mdot_stream_m5_compact_cert_N768_fs030_no_energy_merit.png`
- `outputs/figures/high_mdot_stream_m5_compact_cert_N896_fs030_no_energy_merit.png`
- `outputs/checkpoints/high_mdot_stream_m5_compact_cert_N768_fs030_no_energy_merit/m5n768fs030_mass_0p3_torque_0p005_mdot_5_N768.npz`
- `outputs/checkpoints/high_mdot_stream_m5_compact_cert_N896_fs030_no_energy_merit/m5n896fs030_mass_0p3_torque_0p005_mdot_5_N896.npz`

Interpretation: the `Mdot_inner/Edd=5`, compact no-wind, `f_s=0.30` branch is
mesh-supported over `N=640,768,896` with stable physics diagnostics. This is a
stronger advective/hot solution than the `Mdot=3`, `f_s=0.50` case:
`f_adv_global` rises from about `0.330` to `0.488`, `f_adv_inner` from about
`0.266` to `0.471`, and `H/R` from about `0.268` to `0.316`.

## Mdot=5 Continuation from f_s=0.30 to f_s=0.50

Starting from the certified `N=896`, `f_s=0.30` endpoint, I continued the same
compact no-wind branch to `f_s=0.50`. The tangent predictor and
residual-remeshing loop behaved well: after the first `df_s=0.025` step, the
cost-aware controller grew to `df_s=0.0375--0.05`, and every step accepted as a
strict anchor.

Output:

- `outputs/tables/high_mdot_stream_m5_compact_N896_030_to050_no_energy_merit.md`
- `outputs/figures/high_mdot_stream_m5_compact_N896_030_to050_no_energy_merit.png`
- `outputs/checkpoints/high_mdot_stream_m5_compact_N896_030_to050_no_energy_merit/m5n896fast_mass_0p5_torque_0p005_mdot_5_N896.npz`

| f_s | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3250 | 6.216e-08 | 1.482e-09 | 2.086e-09 | 0.6750 | 0.4887 | 0.4714 | 1.340 | 0.3158 | 4.361 |
| 0.3625 | 4.678e-08 | 3.893e-11 | 1.498e-11 | 0.6375 | 0.4897 | 0.4708 | 1.337 | 0.3157 | 4.361 |
| 0.4125 | 5.446e-08 | 4.136e-09 | 3.453e-09 | 0.5875 | 0.4909 | 0.4711 | 1.332 | 0.3156 | 4.361 |
| 0.4625 | 4.806e-08 | 8.750e-10 | 3.751e-09 | 0.5375 | 0.4921 | 0.4712 | 1.328 | 0.3155 | 4.361 |
| 0.5000 | 2.639e-08 | 8.546e-10 | 3.296e-09 | 0.5000 | 0.4929 | 0.4717 | 1.324 | 0.3155 | 4.361 |

Interpretation: the `Mdot_inner/Edd=5` compact no-wind branch continues
smoothly to `f_s=0.50`. The flow remains strongly advective but stable:
`f_adv_global` rises mildly from `0.488` to `0.493`, `f_adv_inner` stays near
`0.471`, and the sonic radius and thickness are nearly unchanged. The luminosity
decreases from `Lrad/LEdd=1.342` at `f_s=0.30` to `1.324` at `f_s=0.50`, as
less mass is supplied through the far outer boundary.

## Mdot=5, f_s=0.50 Outer-Buffer Sensitivity

I repeated the small buffer-radius sensitivity check at the new high-rate
endpoint using `R_buffer=295,300,305 rg`.

| R_buffer/rg | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 295 | 2.749e-08 | 2.945e-09 | 1.367e-09 | 0.500 | 0.4929 | 0.4714 | 1.325 | 0.3155 | 4.361 |
| 300 | 2.639e-08 | 8.546e-10 | 3.296e-09 | 0.500 | 0.4929 | 0.4717 | 1.324 | 0.3155 | 4.361 |
| 305 | 2.749e-08 | 2.942e-09 | 1.368e-09 | 0.500 | 0.4929 | 0.4715 | 1.325 | 0.3155 | 4.361 |

Output:

- `outputs/tables/high_mdot_stream_m5_fs050_buffer295_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs050_buffer305_N896.md`
- `outputs/figures/high_mdot_stream_m5_fs050_buffer295_N896.png`
- `outputs/figures/high_mdot_stream_m5_fs050_buffer305_N896.png`

Interpretation: the `Mdot_inner/Edd=5`, `f_s=0.50` endpoint passes the same
small outer-buffer sensitivity check. The residual-remeshed repolishes required
more total work than the continuation steps, but the final physical diagnostics
are insensitive to shifting the buffer by `+-5 rg`.

## Mdot=5 Continuation from f_s=0.50 to f_s=0.80

I then continued the same `N=896` compact no-wind branch from `f_s=0.50` to
`f_s=0.80`. The raw tangent seed becomes progressively worse at high source
fraction, but residual-remeshing keeps the corrected seed polishable. Every
step accepted as an anchor.

Output:

- `outputs/tables/high_mdot_stream_m5_compact_N896_050_to080_no_energy_merit.md`
- `outputs/figures/high_mdot_stream_m5_compact_N896_050_to080_no_energy_merit.png`
- `outputs/checkpoints/high_mdot_stream_m5_compact_N896_050_to080_no_energy_merit/m5n896fast2_mass_0p8_torque_0p005_mdot_5_N896.npz`

| f_s | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 2.067e-08 | 7.338e-09 | 1.091e-08 | 0.45 | 0.4941 | 0.4715 | 1.320 | 0.3154 | 4.361 |
| 0.60 | 1.338e-08 | 2.598e-10 | 2.963e-10 | 0.40 | 0.4952 | 0.4713 | 1.316 | 0.3154 | 4.361 |
| 0.65 | 5.556e-09 | 1.892e-09 | 1.929e-09 | 0.35 | 0.4962 | 0.4717 | 1.312 | 0.3153 | 4.361 |
| 0.70 | 2.340e-09 | 2.340e-09 | 1.009e-08 | 0.30 | 0.4972 | 0.4713 | 1.307 | 0.3153 | 4.361 |
| 0.75 | 7.838e-10 | 7.838e-10 | 2.498e-09 | 0.25 | 0.4981 | 0.4716 | 1.303 | 0.3152 | 4.361 |
| 0.80 | 2.783e-10 | 2.783e-10 | 2.166e-10 | 0.20 | 0.4990 | 0.4716 | 1.300 | 0.3152 | 4.361 |

Interpretation: the no-wind compact stream-fed branch remains smooth to
`f_s=0.80`. The global advective fraction approaches `0.5`, while the inner
advective fraction remains near `0.472`; the sonic radius and disk thickness are
nearly unchanged. This is the strongest high-rate no-wind benchmark recovered so
far in the finite-minidisk stream setup.

## Mdot=5, f_s=0.80 Outer-Buffer Sensitivity

I repeated the small buffer-radius sensitivity check at `f_s=0.80`.

| R_buffer/rg | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 295 | 1.692e-10 | 1.692e-10 | 4.836e-11 | 0.200 | 0.4990 | 0.4715 | 1.300 | 0.3152 | 4.361 |
| 300 | 2.783e-10 | 2.783e-10 | 2.166e-10 | 0.200 | 0.4990 | 0.4716 | 1.300 | 0.3152 | 4.361 |
| 305 | 1.724e-10 | 1.724e-10 | 4.981e-11 | 0.200 | 0.4990 | 0.4715 | 1.300 | 0.3152 | 4.361 |

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_buffer295_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_buffer305_N896.md`
- `outputs/figures/high_mdot_stream_m5_fs080_buffer295_N896.png`
- `outputs/figures/high_mdot_stream_m5_fs080_buffer305_N896.png`

Interpretation: the `f_s=0.80` endpoint passes the same small
outer-buffer-placement check. This is not yet a full source-shape or
outer-closure homotopy, but it indicates that the newly recovered high-source
endpoint is not pinned to one precise buffer radius.

## Mdot=5, f_s=0.80 Source-Geometry Scan

I then tested whether the high-source endpoint survives moving the compact
source center. Since `Rout=335 rg`, the target geometry centers are:

- `Rinj/Rout=0.70`: `Rinj=234.5 rg`
- `Rinj/Rout=0.75`: `Rinj=251.25 rg`
- `Rinj/Rout=0.80`: `Rinj=268.0 rg`

The original high-source branch used `Rinj=240 rg`, i.e. `Rinj/Rout=0.716`.

Direct one-shot changes in `Rinj` are numerically stiff. The `0.70` and `0.75`
cases converged directly, but the one-shot `0.80` jump was interrupted after it
entered a very expensive finite-difference Jacobian/polish path. The
`0.80` endpoint was recovered by staged source-center continuation:
`251.25 -> 256 -> 260 -> 264 -> 268 rg`.

Primary geometry endpoints:

| Rinj/Rout | Rinj/rg | final full | phys E | buffer E | Mdot_outer/Mdot_inner | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 234.5 | 6.135e-10 | 6.135e-10 | 3.441e-11 | 0.200 | 0.5000 | 0.4714 | 1.295 | 0.3151 | 4.361 |
| 0.75 | 251.25 | 2.977e-09 | 2.977e-09 | 8.892e-09 | 0.200 | 0.4971 | 0.4710 | 1.308 | 0.3154 | 4.361 |
| 0.80 | 268.0 | 5.233e-10 | 5.233e-10 | 2.288e-09 | 0.200 | 0.4945 | 0.4708 | 1.320 | 0.3157 | 4.361 |

Staged `0.80` path:

| Rinj/rg | final full | phys E | f_adv_global | Lrad/LEdd | max H/R |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.575e-09 | 2.575e-09 | 0.4963 | 1.312 | 0.3155 |
| 260 | 5.240e-10 | 5.240e-10 | 0.4957 | 1.314 | 0.3156 |
| 264 | 1.877e-09 | 1.877e-09 | 0.4951 | 1.317 | 0.3157 |
| 268 | 5.233e-10 | 5.233e-10 | 0.4945 | 1.320 | 0.3157 |

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_geometry070_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_geometry075_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_geometry_rinj256_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_geometry_rinj260_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_geometry_rinj264_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_geometry_rinj268_N896.md`
- corresponding figures in `outputs/figures/`

Interpretation: the `Mdot_inner/Edd=5`, `f_s=0.80` high-source branch survives
a meaningful compact-source geometry scan. The physics varies smoothly:
moving the source outward from `Rinj/Rout=0.70` to `0.80` lowers
`f_adv_global` from about `0.500` to `0.495` and raises `Lrad/LEdd` from about
`1.295` to `1.320`, while `f_adv_inner`, `H/R`, and `Rson` remain nearly
fixed. The caveat is numerical rather than physical: source-center changes need
staged continuation because direct jumps produce large residuals.

## Mdot=5, f_s=0.80 Compatible Robin Closure Check

Finally, I tested the existing `pressure_supported_robin_energy` closure at the
same `Mdot_inner/Edd=5`, `f_s=0.80`, `Rinj=240 rg` endpoint.

Important detail: the Robin target must be computed with the full stream/source
fields included. A first diagnostic that omitted the source terms gave the wrong
target (`-0.322632`), reproducing the old failure mode: accepted only loosely
with a raw outer angular residual that Newton could not reduce. Using the
runner's full source-aware slope-refresh convention gives the compatible target

```text
outer_robin_slope_target = 4.120072070565861e-4
```

With that target, the homotopy `chi=0.25 -> 0.5 -> 1.0` passes:

| chi | final full | phys E | buffer E | outer omega | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 3.922e-07 | 4.761e-10 | 1.314e-09 | 3.922e-03 | 0.4990 | 0.4715 | 1.300 | 0.3152 | 4.361 | Newton converged |
| 0.50 | 7.836e-07 | 4.761e-10 | 1.314e-09 | 7.836e-03 | 0.4990 | 0.4715 | 1.300 | 0.3152 | 4.361 | seed accepted; Newton did not reduce square residual |
| 1.00 | 1.566e-06 | 4.761e-10 | 1.314e-09 | 1.566e-02 | 0.4990 | 0.4715 | 1.300 | 0.3152 | 4.361 | seed accepted; Newton did not reduce square residual |

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_robin_localtarget_chi025_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_robin_localtarget_chi050_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_robin_localtarget_chi100_N896.md`
- corresponding figures in `outputs/figures/`

Interpretation: the branch survives a compatible Robin closure homotopy without
changing the physical diagnostics. This is a useful consistency check, but not
yet a broad reservoir-closure validation: the accepted `chi=0.5` and `chi=1.0`
rows rely on the seed already satisfying the compatible Robin condition closely,
and the current Newton/Jacobian machinery does not actively reduce the Robin
angular residual. A deliberately different Robin target should therefore wait
until the Robin row has better target refresh/Jacobian treatment.

## Mdot=5, f_s=0.80 Stream-Heating Ladder

After the no-wind compact stream branch passed the buffer, source-geometry, and
compatible-Robin checks, I added a heating-efficiency continuation mode to
`scripts/run_standard_slim_stream_mass_annulus_scan.py`. This keeps the same
high-Mdot infrastructure as the no-wind branch: compact C2 source, residual
remeshing, outer buffer, physical energy gate, and high-rate checkpoints.

The heating term is the existing positive source-proportional
`stream_heating_rate`, so the energy residual is

```text
Q_visc + Q_stream - Q_rad - Q_adv = 0
```

with `Q_stream` tied to positive `stream_source_prime`.

### Calibration

At the no-heating `Mdot_inner/Edd=5`, `f_s=0.80`, `N=896` checkpoint, a
no-solve calibration showed the approximate scale:

| eta_heat | seed residual | max Qstream/Qvisc | integrated Qstream/Qvisc | peak R/rg |
|---:|---:|---:|---:|---:|
| 1e-4 | 1.281e-03 | 1.632e-03 | 3.269e-06 | 239.2 |
| 1e-3 | 1.281e-02 | 1.632e-02 | 3.269e-05 | 239.2 |
| 1e-2 | 1.271e-01 | 1.632e-01 | 3.269e-04 | 239.2 |
| 3e-2 | 3.587e-01 | 4.897e-01 | 9.807e-04 | 239.2 |
| 1e-1 | 7.883e-01 | 1.632e+00 | 3.269e-03 | 239.2 |

### Newton ladder

The staged heating ladder accepted cleanly through `eta_heat=0.1`.

| eta_heat | final full | phys E | max Qstream/Qvisc | integrated Qstream/Qvisc | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.542e-10 | 1.542e-10 | 0 | 0 | 0.4990 | 0.4715 | 1.300 | 0.3152 | 4.361 |
| 1e-4 | 1.822e-10 | 1.822e-10 | 1.634e-03 | 3.269e-06 | 0.4990 | 0.4721 | 1.300 | 0.3152 | 4.361 |
| 3e-4 | 4.468e-09 | 4.468e-09 | 4.910e-03 | 9.806e-06 | 0.4990 | 0.4713 | 1.300 | 0.3152 | 4.361 |
| 1e-3 | 2.036e-10 | 2.036e-10 | 1.648e-02 | 3.269e-05 | 0.4989 | 0.4717 | 1.300 | 0.3152 | 4.361 |
| 3e-3 | 2.142e-10 | 2.142e-10 | 5.036e-02 | 9.807e-05 | 0.4989 | 0.4720 | 1.300 | 0.3152 | 4.361 |
| 1e-2 | 1.269e-10 | 1.269e-10 | 1.791e-01 | 3.269e-04 | 0.4987 | 0.4713 | 1.301 | 0.3153 | 4.361 |
| 3e-2 | 1.094e-10 | 1.094e-10 | 6.674e-01 | 9.808e-04 | n/a | n/a | n/a | n/a | n/a |
| 1e-1 | 1.021e-10 | 1.021e-10 | 1.586e+01 | 3.270e-03 | 0.4962 | 0.4713 | 1.315 | 0.3159 | 4.361 |

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_heating_scout_eta0_to1e3_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta1e3_to1e2_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta1e2_to3e2_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta3e2_to1e1_N896.md`
- corresponding figures/checkpoints in `outputs/figures/` and
  `outputs/checkpoints/`

Interpretation: the heated branch is numerically smooth through `eta_heat=0.1`.
Even though the local peak `Qstream/Qvisc` becomes large at `eta=0.1`, the
integrated heating budget is still small, about `3.27e-3` of the integrated
viscous heating. The global disk diagnostics change only mildly:
`f_adv_global` shifts from about `0.499` to `0.496`, `Lrad/LEdd` rises from
about `1.300` to `1.315`, and `H/R` and `Rson` remain essentially fixed.

### Eta=0.1 robustness checks

The aggressive `eta_heat=0.1` checkpoint also passed the same small outer-buffer
check:

| R_buffer/rg | final full | phys E | max Qstream/Qvisc | integrated Qstream/Qvisc |
|---:|---:|---:|---:|---:|
| 295 | 1.861e-10 | 1.861e-10 | 1.593e+01 | 3.269e-03 |
| 300 | 1.021e-10 | 1.021e-10 | 1.586e+01 | 3.270e-03 |
| 305 | 2.274e-10 | 2.274e-10 | 1.593e+01 | 3.269e-03 |

N spot checks also accepted:

| N | final full | phys E | max Qstream/Qvisc | integrated Qstream/Qvisc | f_adv_global | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 768 | 1.228e-10 | 1.228e-10 | 9.696e+01 | 3.271e-03 | 0.4962 | 1.315 | 0.3159 | 4.361 |
| 896 | 1.021e-10 | 1.021e-10 | 1.586e+01 | 3.270e-03 | 0.4962 | 1.315 | 0.3159 | 4.361 |
| 1024 | 4.141e-10 | 4.141e-10 | 1.735e+01 | 3.270e-03 | 0.4962 | 1.315 | 0.3159 | 4.361 |

The integrated heating ratio and physical diagnostics are stable across
`N=768,896,1024`. The pointwise maximum `Qstream/Qvisc` is not robust at N768
because the denominator can become very small locally; the integrated ratio is
the more meaningful heating-budget diagnostic.

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta01_buffer295_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta01_buffer305_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta01_N768_spot.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta01_N1024_spot.md`

### Stronger heating: eta_heat=0.3 and 1.0

I then restarted from the accepted `eta_heat=0.1` checkpoint and pushed the
same compact-source, no-wind branch to stronger stream heating:

| eta_heat | initial full | final full | phys E | integrated Qstream/Qvisc | f_adv_global | f_adv_inner | Lrad/LEdd | max H/R | Rson/rg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 6.386e-01 | 9.964e-11 | 9.964e-11 | 9.798e-03 | 0.4889 | 0.4697 | 1.346 | 0.3176 | 4.359 |
| 1.0 | 6.955e-01 | 9.688e-12 | 9.680e-12 | 3.290e-02 | 0.4581 | 0.4608 | 1.455 | 0.3262 | 4.349 |

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta01_to1_N896.md`
- `outputs/figures/high_mdot_stream_m5_fs080_heating_eta01_to1_N896.png`
- `outputs/checkpoints/high_mdot_stream_m5_fs080_heating_eta01_to1_N896/`

The `eta_heat=1.0` point also passed the quick robustness checks:

| check | final full | phys E | integrated Qstream/Qvisc | f_adv_global | Lrad/LEdd | max H/R | Rson/rg |
|---|---:|---:|---:|---:|---:|---:|---:|
| N=768 | 1.406e-10 | 1.406e-10 | 3.291e-02 | 0.4578 | 1.455 | 0.3262 | 4.349 |
| N=896 | 9.688e-12 | 9.680e-12 | 3.290e-02 | 0.4581 | 1.455 | 0.3262 | 4.349 |
| N=1024 | 1.141e-09 | 1.141e-09 | 3.289e-02 | 0.4579 | 1.455 | 0.3262 | 4.349 |
| R_buffer=295 rg | 9.513e-12 | 9.513e-12 | 3.292e-02 | 0.4582 | 1.455 | 0.3262 | 4.349 |
| R_buffer=305 rg | 9.379e-12 | 9.379e-12 | 3.292e-02 | 0.4582 | 1.455 | 0.3262 | 4.349 |

Output:

- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta1_N768_spot.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta1_N1024_spot.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta1_buffer295_N896.md`
- `outputs/tables/high_mdot_stream_m5_fs080_heating_eta1_buffer305_N896.md`

Interpretation: stream heating through `eta_heat=1` is numerically robust on
this branch, but it does not recover a stronger advective/hot solution. The
global advective fraction actually decreases from about `0.499` at no heating
to about `0.458` at `eta_heat=1`, while `Lrad/LEdd` rises from about `1.300` to
about `1.455`. The disk thickens only mildly (`max H/R` from about `0.315` to
`0.326`) and the sonic radius is essentially unchanged. In this formulation,
stream heating mostly adds radiated luminosity rather than creating a new
advective branch.

## Caveats

- This is still no-wind; stream heating is now included only as a conservative
  energy source, with no wind mass/energy removal.
- The outer-buffer residual is weighted and remains the dominant full residual
  channel in the stream-fed runs, even though the physical `interval_E` is well
  below the `1e-5` gate.
- The result should not yet be called a final wind/hot-branch model; it is now
  a robust no-wind compact-stream high-rate benchmark with stream heating
  through `eta_heat=1.0`.

## Recommended Next Step

Next, add the first controlled wind term. Heating alone is stable but does not
move the branch toward a hotter, more advective state. Start with a small,
localized sink fraction at the same compact-source geometry and track mass,
energy, luminosity, and advective budgets separately. Keep the `eta_heat=0`,
`0.1`, and `1.0` no-wind solutions as regression anchors.

## Verification

`PYTHONPATH=src /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest`

Result after the stream-heating implementation and stronger-heating checks:
`149 passed in 2.87s`.
