# Codex Sprint A Results: Standard No-Wind High-Mdot Scout

Date: 2026-07-01

This sprint followed the updated GPT recommendation in:

- `Note/CODEX_UPDATED_PLAN_HIGH_MDOT_NOWIND_LIMIT_CYCLE.md`

The purpose was to answer the first necessary question before adding wind:

```text
Can the standard no-wind slim solver recover a more advective high-Mdot branch
when Mdot_inner is actually raised above Eddington?
```

## Implementation

New scripts:

- `scripts/run_standard_slim_high_mdot_no_wind_ladder.py`
- `scripts/run_standard_slim_high_mdot_newton_scout.py`
- `scripts/run_standard_slim_high_mdot_advection_diagnostics.py`
- `scripts/run_standard_slim_high_mdot_checkpoint_certification.py`

The first script is a configuration wrapper around the existing adaptive
continuation ladder.  In practice, the full adaptive N640/N512 machinery was
too expensive for quick exploration because repeated outer-slope refresh and
least-squares fallback dominated runtime.

I therefore added the lean Newton-only scout:

- fixed target list,
- tangent/secant/copy seed selection,
- square Newton polish only,
- no least-squares fallback,
- one optional pressure-supported outer-slope Picard refresh,
- diagnostics written at every target.

This is intended as a branch-topology scout, not final certification.  I then
added the checkpoint certification script to rerun selected scout states at
higher N with explicit pivot/pass logging and pressure-supported outer-slope
Picard refresh.

## Outputs

Scout tables and figures:

- `outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout_N512_to1p1.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout_N512_1p1_to2_smallstep.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout_N512_1p2_to2_relaxed.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout_N512_1p45_to2_loose.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout_N512_2_to3_loose.md`
- `outputs/figures/slim_benchmark_high_mdot_no_wind_scout_to3_advection_diagnostics.png`

Combined diagnostic table:

- `outputs/tables/slim_benchmark_high_mdot_no_wind_scout_to3_advection_diagnostics.md`

Combined diagnostic figure:

- `outputs/figures/slim_benchmark_high_mdot_no_wind_scout_to3_advection_diagnostics.png`

Certification and N640 extension outputs:

- `outputs/tables/slim_benchmark_high_mdot_no_wind_certification_m2_N640_refine.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_certification_m3_N640_refine.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_newton_scout_N640_3_to5.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_N640_to5_advection_diagnostics.md`
- `outputs/figures/slim_benchmark_high_mdot_no_wind_N640_to5_advection_diagnostics.png`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_N640_to5_residual_profile.md`
- `outputs/figures/slim_benchmark_high_mdot_no_wind_N640_to5_residual_profile.png`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_m4_adaptive_outer_mesh.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_m5_adaptive_outer_mesh.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_m5_adaptive_outer_mesh_strong.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_m5_adaptive_outer_mesh_N768_spot.md`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_certified_to5_advection_diagnostics.md`
- `outputs/figures/slim_benchmark_high_mdot_no_wind_certified_to5_advection_diagnostics.png`
- `outputs/tables/slim_benchmark_high_mdot_no_wind_certified_to5_residual_profile.md`
- `outputs/figures/slim_benchmark_high_mdot_no_wind_certified_to5_residual_profile.png`

## Result

The initial N512 scout reaches `Mdot_inner/Edd = 3` without sonic or
outer-boundary failure.  The dominant residual is `interval_E` throughout.

Selected diagnostics:

| `Mdot/Edd` | residual | strict? | `f_adv_global` | `f_adv_inner` (`R<20rg`) | `f_adv_pos` | `Lrad/LEdd` | max H/R | Rson/rg |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| `1.0` | `1.754e-7` | yes | `0.0319` | `-0.121` | `0.114` | `0.593` | `0.157` | `5.071` |
| `1.2` | `1.038e-5` | no | `0.0658` | `-0.0756` | `0.145` | `0.684` | `0.175` | `4.950` |
| `1.6` | `2.357e-5` | no | `0.1289` | `0.00969` | `0.201` | `0.838` | `0.204` | `4.774` |
| `2.0` | `3.416e-5` | no | `0.1865` | `0.0945` | `0.250` | `0.966` | `0.227` | `4.662` |
| `2.6` | `5.607e-5` | no | `0.2599` | `0.193` | `0.313` | `1.124` | `0.253` | `4.554` |
| `3.0` | `7.754e-5` | no | `0.301` | `0.256` | `0.349` | `1.213` | `0.268` | `4.503` |

## Interpretation

The scout supports GPT's suggested picture:

- Raising `Mdot_inner` does drive the standard no-wind solution toward a more
  advective branch.
- `f_adv_inner` crosses from negative to positive near `Mdot/Edd ~ 1.5--1.6`
  using the current sign convention and `R<20rg` diagnostic.
- `f_adv_pos` and `f_adv_global` grow monotonically.
- `Lrad` grows sublinearly: from `0.593 L_Edd` at `Mdot/Edd=1` to
  `1.213 L_Edd` at `Mdot/Edd=3`, while the supplied `Mdot` triples.
- The disk thickens but remains within a height-integrated regime in this
  scout: max `H/R` rises from `0.157` to `0.268`.
- The sonic point moves inward smoothly from `5.071 rg` to `4.503 rg`.

This means the previous fixed-`Mdot_inner=1` stream-heating experiment did not
contradict the high-state limit-cycle picture.  It simply was not driving the
inner disk to high enough accretion rate.

## Certification update

The N512 `Mdot/Edd=2` scout checkpoint was not quite strict:

- seed residual: `3.416e-5`
- one C2 Newton pass: `1.067e-5`
- two C2 outer-slope Picard passes: `1.035e-5`
- C1 pivot: also `1.067e-5`

This showed that the near-miss was not a sonic problem or a bad compatibility
pivot.  Remapping the same state to N640 immediately dropped the residual below
the strict `1e-5` acceptance line:

| `Mdot/Edd` | N | residual | strict accepted? | `f_adv_inner` | dominant |
|---:|---:|---:|:---:|---:|---|
| `2.0` | `640` | `6.450e-6` | yes | `0.0943` | `interval_E` |
| `3.0` | `640` | `7.262e-6` | yes | `0.2562` | `interval_E` |

Both N640 certifications remain above the tighter `3e-6` anchor threshold, but
they are stable under repeated outer-slope Picard refresh.  The remaining
residual is still the energy interval residual, so the next accuracy bottleneck
is mesh/discretization quality rather than physical branch failure.

## N640 scout to 5

Starting from the refined N640 `Mdot/Edd=3` checkpoint, I continued the no-wind
standard slim branch to `Mdot/Edd=5` using the lean Newton scout with relaxed
acceptance `1.5e-4`.

| `Mdot/Edd` | residual | scout accepted? | `f_adv_global` | `f_adv_inner` (`R<20rg`) | `Lrad/LEdd` | max H/R | Rson/rg |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| `3.0` | `7.262e-6` | yes, strict | `0.3018` | `0.2562` | `1.213` | `0.2677` | `4.502` |
| `3.25` | `5.360e-5` | yes, relaxed | `0.3250` | `0.2872` | `1.263` | `0.2755` | `4.476` |
| `3.5` | `5.852e-5` | yes, relaxed | `0.3464` | `0.3155` | `1.311` | `0.2827` | `4.454` |
| `4.0` | `1.176e-4` | yes, relaxed | `0.3846` | `0.3701` | `1.396` | `0.2956` | `4.416` |
| `4.5` | `1.297e-4` | yes, relaxed | `0.4177` | `0.4123` | `1.472` | `0.3066` | `4.385` |
| `5.0` | `1.349e-4` | yes, relaxed | `0.4465` | `0.4485` | `1.541` | `0.3164` | `4.360` |

This is a branch-topology result, not strict certification above `Mdot/Edd=3`.
The smooth monotonic trends are encouraging: the inner disk becomes increasingly
advective, the luminosity grows sublinearly with supplied accretion rate, the
disk remains optically thick, and max `H/R` remains about `0.32` at
`Mdot/Edd=5`.

## Residual localization

The high-Mdot residual-profile audit localizes the dominant error to the outer
edge, not to the sonic/inner advective region.

| case | `Mdot/Edd` | full | peak `interval_E` R/rg | peak `interval_E` | median abs `interval_E` | p90 abs `interval_E` |
|---|---:|---:|---:|---:|---:|---:|
| anchor | `1` | `5.175e-6` | `9956` | `5.175e-6` | `4.571e-13` | `4.874e-12` |
| m2 | `2` | `6.450e-6` | `9964` | `6.450e-6` | `4.197e-13` | `2.431e-12` |
| m3 | `3` | `7.262e-6` | `9964` | `7.262e-6` | `4.270e-13` | `4.164e-12` |
| m4 | `4` | `1.176e-4` | `9964` | `1.176e-4` | `1.476e-12` | `7.428e-11` |
| m5 | `5` | `1.349e-4` | `9964` | `1.349e-4` | `1.765e-12` | `8.298e-11` |

So the large residual above `Mdot/Edd=3` is highly localized in the final outer
intervals near `R_out=10000 rg`; the median and 90th-percentile interval-energy
errors remain tiny.  This strongly favors an outer-boundary/adaptive-outer-grid
fix over uniform N growth or inner sonic surgery.

## Adaptive outer mesh certification to 5

I then applied the existing residual-based adaptive outer mesh to the relaxed
N640 `Mdot/Edd=4` and `5` scout checkpoints.  This directly targeted the final
outer intervals where the residual-profile audit found the `interval_E` peak.

For `Mdot/Edd=4`, N640 was sufficient:

| strength | N | residual | anchor? | outer 1% nodes | outer 5% nodes | peak R/rg |
|---:|---:|---:|:---:|---:|---:|---:|
| `4` | `640` | `2.507e-6` | yes | `15` | `45` | `9979` |
| `8` | `640` | `3.207e-6` | no | `20` | `52` | `9983` |
| `16` | `640` | `3.279e-6` | no | `25` | `65` | `9985` |

The moderate `s4` mesh is best for `Mdot/Edd=4`; stronger concentration
slightly overdoes the final-edge correction.

For `Mdot/Edd=5`, N640 becomes strict but not anchor, while one N768 spot check
does reach anchor quality:

| strength | N | residual | anchor? | outer 1% nodes | outer 5% nodes | peak R/rg |
|---:|---:|---:|:---:|---:|---:|---:|
| `4` | `640` | `4.988e-6` | no | `15` | `45` | `9979` |
| `8` | `640` | `4.884e-6` | no | `20` | `51` | `9983` |
| `16` | `640` | `4.338e-6` | no | `25` | `64` | `9985` |
| `24` | `640` | `4.033e-6` | no | `27` | `77` | `9986` |
| `32` | `640` | `3.846e-6` | no | `28` | `89` | `9986` |
| `32` | `768` | `2.293e-6` | yes | `33` | `106` | `9965` |

Thus the previous high-rate obstruction was an outer-grid resolution problem.
It is not a sonic regularity problem, not a failure of the advective branch, and
not a sign that the standard no-wind slim branch is absent.

The updated certified diagnostic sequence is:

| `Mdot/Edd` | N | residual | anchor? | `f_adv_global` | `f_adv_inner` (`R<20rg`) | `Lrad/LEdd` | max H/R | Rson/rg |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| `1` | `512` | `5.175e-6` | no | `0.0347` | `-0.1151` | `0.5934` | `0.1572` | `5.084` |
| `2` | `640` | `6.450e-6` | no | `0.1865` | `0.0943` | `0.9660` | `0.2269` | `4.661` |
| `3` | `640` | `7.262e-6` | no | `0.3018` | `0.2562` | `1.213` | `0.2677` | `4.502` |
| `4` | `640` | `2.507e-6` | yes | `0.3894` | `0.3830` | `1.396` | `0.2956` | `4.415` |
| `5` | `768` | `2.293e-6` | yes | `0.4534` | `0.4666` | `1.541` | `0.3164` | `4.360` |

The physical trend is essentially unchanged by remeshing: advection rises
smoothly, radiative luminosity grows sublinearly, the sonic point moves inward
smoothly, and the disk remains optically thick with max `H/R ~ 0.32` at
`Mdot/Edd=5`.

## Caveat

The strict status is now:

- `Mdot/Edd=1`: strict accepted, pre-existing standard benchmark.
- `Mdot/Edd=2`: strict accepted at N640, but not a `3e-6` anchor.
- `Mdot/Edd=3`: strict accepted at N640, but not a `3e-6` anchor.
- `Mdot/Edd=4`: anchor at N640 after adaptive outer remeshing.
- `Mdot/Edd=5`: anchor with an N768 adaptive-outer-grid spot check; N640 is
  strict accepted but remains above the `3e-6` anchor line.

Therefore:

```text
The no-wind standard slim branch is now certified to Mdot/Edd=5 at strict
residual, with anchor-quality points at 4 and 5 after targeted adaptive outer
remeshing.
```

Remaining caveats:

- `Mdot/Edd=2` and `3` are strict but not yet anchor-quality.
- The `Mdot/Edd=5` anchor currently relies on N768; this is acceptable as a
  spot check but should be made part of a staged adaptive-N ladder before
  claiming a production sequence.
- This is still the standard no-wind slim benchmark, not the stream-fed/wind
  IMRI model.

## Updated next step

The numerical bottleneck identified in this sprint is now solved for the
standard no-wind branch through `Mdot/Edd=5`.  The next useful choices are:

1. Extend the standard no-wind adaptive branch to `Mdot/Edd=7` and `10` using
   the same residual-localization/remeshing procedure.
2. Fix the finite-minidisk stream source/sink sign in the source-loading
   scripts and rerun the stream-fed no-wind annulus tests.
3. Only after the corrected stream-fed no-wind model behaves cleanly should wind
   be added back.
