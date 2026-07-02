# Codex High-Mdot Stream Bridge Results

Date: 2026-07-02

This note reports the actual runs requested after GPT's high-`Mdot`
stream-bridge plan.  The scope was points 2--5 from the working plan:

```text
2. finite-Rout high-Mdot no-stream bridge
3. high-Mdot conservative stream-source bridge
4. stream-source interval_E residual localization
5. small stream angular-momentum source/torque test
```

## Code/diagnostic changes

I added the bridge diagnostics requested by GPT to the finite-boundary and
stream-source runners:

- `f_adv_global`
- `f_adv_inner(R<20rg)`
- `f_adv_pos`
- `Lrad/LEdd`
- peak interval-energy residual radius
- median and p90 interval-energy residual

I also patched `scripts/run_standard_slim_mdot_residual_profile.py` so it
preserves stream source/sink/torque/heating fields from checkpoints.  This was
necessary for the `f_s=0.49` interval-localization diagnostic to analyze the
actual stream-source problem rather than silently reverting to a no-source
model.

## Point 2: finite-Rout high-Mdot no-stream bridge

Outputs:

- `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m2.md`
- `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m3_staged.md`
- `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m3_5000_to3000_fine.md`
- `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m5_staged.md`
- `outputs/tables/high_mdot_finite_Rout_nowind_bridge_m5_500_to300_refresh.md`

### `Mdot/Edd = 2`

The finite-radius no-stream branch reaches `Rout = 300 rg`.

| `Rout/rg` | residual | accepted | dominant | `f_adv_global` | `f_adv_inner` | `f_adv_pos` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| `3000` | `5.410e-6` | yes | `interval_E` | `0.1876` | `0.0935` | `0.2520` | `0.9593` | `0.2269` | `4.661` |
| `1000` | `1.057e-6` | yes | `outer_omega` | `0.1906` | `0.0927` | `0.2559` | `0.9412` | `0.2269` | `4.660` |
| `500` | `2.151e-6` | yes | `outer_omega` | `0.1948` | `0.0939` | `0.2614` | `0.9158` | `0.2269` | `4.660` |
| `300` | `4.441e-6` | yes | `outer_omega` | `0.1995` | `0.0931` | `0.2681` | `0.8844` | `0.2269` | `4.660` |

Interpretation: the high-`Mdot=2` advective branch survives finite truncation
to `Rout=300 rg`; the inner advective fraction remains positive and close to
the large-radius parent.

### `Mdot/Edd = 3`

The branch reaches `Rout = 4000 rg` with fine staging, but not `3500--3000 rg`
under the current pressure-supported closure/mesh.

| run | last accepted `Rout/rg` | residual | next failed `Rout/rg` | failed residual | dominant |
|---|---:|---:|---:|---:|---|
| staged | `5000` | `1.663e-6` | `3000` | `1.837e-5` | `interval_E` |
| fine | `4000` | `4.871e-6` | `3500` | `1.133e-5` | `interval_E` |

At the accepted `Rout=4000` point:

```text
f_adv_global = 0.3031
f_adv_inner  = 0.2576
f_adv_pos    = 0.3498
Lrad/LEdd    = 1.209
max H/R      = 0.2677
Rson         = 4.502 rg
```

Interpretation: the hot branch remains physically smooth where accepted, but
finite-radius continuation stalls on `interval_E` before the minidisk-sized
radius.  This looks numerical/closure related, not a sonic failure.

### `Mdot/Edd = 5`

The branch reaches `Rout = 500 rg`, but not `300 rg`.

| `Rout/rg` | residual | accepted | dominant | `f_adv_global` | `f_adv_inner` | `f_adv_pos` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| `7000` | `2.880e-7` | yes | `outer_omega` | `0.4536` | `0.4668` | `0.4773` | `1.538` | `0.3164` | `4.360` |
| `5000` | `2.430e-6` | yes | `interval_E` | `0.4543` | `0.4663` | `0.4780` | `1.534` | `0.3164` | `4.360` |
| `3000` | `9.011e-7` | yes | `interval_E` | `0.4560` | `0.4681` | `0.4795` | `1.524` | `0.3164` | `4.360` |
| `1000` | `2.642e-6` | yes | `outer_omega` | `0.4635` | `0.4681` | `0.4871` | `1.480` | `0.3164` | `4.360` |
| `500` | `9.987e-6` | yes | `outer_omega` | `0.4727` | `0.4691` | `0.4967` | `1.422` | `0.3164` | `4.360` |
| `300` | `2.277e-5` | no | `outer_omega` | `0.4798` | `0.4694` | `0.5044` | `1.362` | `0.3167` | `4.360` |

A refreshed `500 -> 300` retry still failed at `2.200e-5`, dominated by
`outer_omega`.

Interpretation: the `Mdot=5` hot branch survives finite truncation at least to
`Rout=500 rg`, with strong advection preserved.  The `Rout=300 rg` obstruction
is an outer angular/finite-boundary closure issue, not loss of the hot branch.

## Point 3: conservative stream-source bridge

The only fully successful `Rout=300 rg` high-`Mdot` no-stream parent is
currently `Mdot/Edd=2`, so the source bridge was run there.

### Broad source annulus, width `0.30`

Outputs:

- `outputs/tables/high_mdot_stream_source_bridge_m2_broad.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_broad_smallstep.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_broad_relaxed_probe.md`

Broad source failed strict acceptance even at `f_s=0.01`, dominated by
`outer_omega`:

```text
f_s = 0.01
Mdot_outer/Mdot_inner = 0.991843
final residual = 1.451e-5  (strict run)
dominant = outer_omega
interval_R ~ 3e-13
interval_E ~ 8e-12
```

A relaxed diagnostic pass accepted the same point at `2.226e-5`, with interval
residuals still tiny.  This says the broad-source obstruction is almost purely
outer angular closure, not source-annulus energy stiffness.

### Narrow source annulus, width `0.08`

Outputs:

- `outputs/tables/high_mdot_stream_source_bridge_m2_narrow_smoke.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_narrow_ladder.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_narrow_0p1_0p2_smallstep.md`
- `outputs/tables/high_mdot_stream_source_bridge_m2_narrow_0p2_0p3_smallstep.md`

Narrow source succeeds to `f_s=0.30` with corrected mass budget and positive
inner advection preserved.

Key accepted rows:

| `f_s` | `Mdot_outer/Mdot_inner` | residual | accepted | dominant | `f_adv_global` | `f_adv_inner` | `f_adv_pos` | `Lrad/LEdd` | max `H/R` | `Rson/rg` |
|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| `0.01` | `0.99004` | `1.503e-6` | yes | `outer_omega` | `0.1996` | `0.0931` | `0.2682` | `0.8842` | `0.2269` | `4.660` |
| `0.10` | `0.90038` | `8.122e-6` | yes | `outer_omega` | `0.2002` | `0.0931` | `0.2689` | `0.8821` | `0.2269` | `4.660` |
| `0.20` | `0.80075` | `4.072e-6` | yes | `outer_omega` | `0.2007` | `0.0931` | `0.2696` | `0.8799` | `0.2269` | `4.660` |
| `0.30` | `0.70113` | `3.733e-6` | yes | `outer_omega` | `0.2013` | `0.0931` | `0.2702` | `0.8777` | `0.2269` | `4.660` |

Interpretation: at `Mdot_inner/Edd=2`, the finite minidisk remains on a mildly
advective high-`Mdot` branch under narrow conservative stream feeding up to
`f_s=0.30`.  The inner advective fraction stays essentially unchanged from the
finite-`Rout` parent, so conservative mass loading does not destroy the hot
branch at this rate.

## Point 4: `f_s=0.49` interval_E localization

Output:

- `outputs/tables/source_fraction_interval_E_diagnostics_fs049.md`
- `outputs/figures/source_fraction_interval_E_diagnostics_fs049.png`

Result:

```text
case = Mdot_inner/Edd=1, Rout=300 rg, f_s=0.49, width=0.08
full residual = 2.438e-6
dominant = interval_E
peak interval_E radius = 270 rg
peak interval_R radius = 269.1 rg
source annulus center = 240 rg
```

The top interval peaks all lie around `R ~ 264--274 rg`, where
`stream_source_prime/Mdot_inner ~ 0.4--1.0`.

Interpretation: the high-source `interval_E` bottleneck is localized in/near
the stream-source annulus and outer source tail, not at the sonic point.  The
right numerical fix is a source-annulus residual/mesh monitor and/or a
source-fraction tangent predictor.

## Point 5: small stream angular-momentum source / torque

Outputs:

- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquem005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquep005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquem0005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquep0005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquem001.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquem002.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_broad_fs001_torquem0025.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_narrow_fs010_torquem0005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_narrow_fs010_torquep0005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_narrow_fs030_torquem0005.md`
- `outputs/tables/high_mdot_stream_torque_bridge_m2_narrow_fs030_torquep0005.md`

Broad-source torque probe:

- `delta_l/lK = +/-0.05` fails already at zero source.
- `delta_l/lK = -0.005, -0.01, -0.02, -0.025` progressively improves the
  `f_s=0.01` broad-source residual but does not cross strict acceptance.
- Best broad-source torque probe:

```text
f_s = 0.01
delta_l/lK = -0.025
residual = 1.058e-5
dominant = outer_omega
```

Narrow-source torque probe:

| `f_s` | `delta_l/lK` | residual | accepted | anchor | `f_adv_global` | `f_adv_inner` | `Lrad/LEdd` |
|---:|---:|---:|:---:|:---:|---:|---:|---:|
| `0.10` | `-0.005` | `4.913e-6` | yes | no | `0.2002` | `0.0931` | `0.8822` |
| `0.10` | `+0.005` | `1.000e-6` | yes | yes | `0.2002` | `0.0931` | `0.8821` |
| `0.30` | `-0.005` | `2.120e-6` | yes | yes | `0.2013` | `0.0931` | `0.8777` |
| `0.30` | `+0.005` | `3.619e-7` | yes | yes | `0.2013` | `0.0931` | `0.8776` |

Interpretation: the narrow high-`Mdot=2` stream-fed bridge tolerates small
distributed angular-momentum source terms, and positive small torque actually
improves the outer angular residual.  The current implementation should not
jump directly to `|delta_l/lK| ~ 0.05--0.2`; that is too large a deformation
for this finite-boundary BVP.

## Scientific status after these runs

The answer improved:

```text
Yes, we now have a credible first bridge from the recovered no-wind advective
branch into a finite, corrected stream-fed minidisk, but only for
Mdot_inner/Edd = 2 so far.
```

At `Mdot_inner/Edd=2`, `Rout=300 rg`, narrow conservative stream source up to
`f_s=0.30` preserves:

```text
f_adv_inner ~ 0.093
f_adv_global ~ 0.20
f_adv_pos ~ 0.27
Lrad/LEdd ~ 0.88
max H/R ~ 0.227
Rson ~ 4.66 rg
```

This is not as hot as the `Mdot=5` standard branch, but it is a genuine
positive-advection finite stream-fed high-`Mdot` branch.

The next bottlenecks are:

1. `Mdot=3` finite-`Rout` stalls at `Rout~3500--3000 rg`, dominated by
   `interval_E`.
2. `Mdot=5` finite-`Rout` reaches `Rout=500 rg` but fails at `300 rg`,
   dominated by `outer_omega`.
3. Broad source annuli at `Mdot=2, Rout=300` are blocked by outer angular
   closure even for `f_s=0.01`.
4. High-source `f_s=0.49` energy residual is localized in the source annulus,
   so source-annulus adaptive mesh is needed before stronger source claims.

## Recommended next step

Do not add wind yet.

Next best move:

```text
Implement a finite-boundary/outer-closure continuation improvement:
  - allow outer angular condition to relax through a distributed torque/source
    constraint or a pressure-supported outer residual with slope Picard;
  - add adaptive residual mesh monitors for finite-boundary interval_E and
    source-annulus interval_E;
  - retry Mdot=3 to Rout=300 and Mdot=5 to Rout=300.
```

In parallel, preserve the successful `Mdot=2, Rout=300, f_s=0.30` narrow-source
branch as the first finite stream-fed high-`Mdot` bridge regression anchor.

