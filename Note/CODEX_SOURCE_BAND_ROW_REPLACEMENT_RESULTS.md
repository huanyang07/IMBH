# Source-Band Row-Replacement Results

Date: 2026-07-07

## Goal

Implement GPT's recommendation after the two-layer source-band test:

- stop stacking old midpoint rows and new source-band rows in the same
  source/buffer intervals;
- add explicit row modes;
- demote the hard slope-interface condition to an audit;
- certify mass replacement at `eta_E = 100` before lowering `eta_E`;
- only then try implicit radial/energy source-band rows.

Target:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
compact-C2 stream source
local-Mdot mass-loaded wind
eta_E = 100
N = 164
```

Primary starting checkpoint:

```text
outputs/checkpoints/m5_source_band_replacement_chi050_eta100_N164/
    stage_00_etaE_100_N164.npz
```

## Code Changes

Primary file:

- `scripts/run_mdot5_local_mdot_eta_continuation.py`

New controls:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_ROW_MODE`
  - `append`: old diagnostic behavior;
  - `blend`: one active blended mass row, no separately active old buffer rows;
  - `replace`: direct new replacement row;
  - `audit`: compute audits with old rows active.
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_SLOPE_INTERFACE_ACTIVE`
  - default false; slope mismatch is now an audit unless explicitly re-enabled.
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_X_SCALE`
  - supports `jac` for local sparse finite-difference column scaling.
- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_DIFF_STEP`
  - optional finite-difference step for the local replacement least-squares solve.

New diagnostics:

- active row mode;
- raw FV mass residual separate from the active blended row;
- slope-interface audit separate from active interface rows;
- sparse-FD solver scaling metadata.
- evaluate-only source-band replacement mode:
  `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_EVALUATE_ONLY`;
- implicit slope seed selector:
  `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_IMPLICIT_SEED`
  with `profile`, `ode`, `blend`, and `implicit_lstsq`;
- optional blend coefficient:
  `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_REPLACEMENT_IMPLICIT_SEED_BLEND`.

## Verification

Compile:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/imbh_pycache \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
```

Result: passed.

Regression suite:

```bash
PYTHONPATH=src \
  /Users/huanyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m pytest
```

Result: `160 passed in 2.88s`.

Latest rerun after the implicit-seed diagnostics:

```text
160 passed in 3.28s
```

## Mass-Replacement Runs

All runs used `chi_impl = 0`, slope-interface inactive, and old source rows as
audits rather than hard vetoes in `blend/replace` modes.

| run | mode | halo | active | outside old | active mass row | raw FV mass | interface | old source audit | alpha | nfev |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m5_source_band_rowblend_halo8_chi055_eta100_N164` | blend | 8 | 5.780e-06 | 5.780e-06 | 2.984e-06 | 1.179e-02 | 8.676e-09 | 2.963e-02 | 1.0 | 5 |
| `m5_source_band_rowblend_halo8_chi060_eta100_N164` | blend | 8 | 6.499e-04 | 6.737e-06 | 6.499e-04 | 1.177e-02 | 7.174e-07 | 3.175e-02 | 0.5 | 13 |
| `m5_source_band_rowreplace_halo8_mass_eta100_N164` | replace | 8 | 2.983e-04 | 5.780e-06 | 2.983e-04 | 2.983e-04 | 2.286e-06 | 5.885e-01 | 1.0 | 4 |
| `m5_source_band_rowreplace_halo16_mass_eta100_N164` | replace | 16 | 1.302e-04 | 5.780e-06 | 1.302e-04 | 1.302e-04 | 2.912e-06 | 5.867e-01 | 1.0 | 4 |
| `m5_source_band_rowreplace_halo24_mass_eta100_N164` | replace | 24 | 5.301e-05 | 5.780e-06 | 5.301e-05 | 5.301e-05 | 1.186e-06 | 5.860e-01 | 1.0 | 4 |
| `m5_source_band_rowreplace_halo32_mass_eta100_N164` | replace | 32 | 7.456e-06 | 5.780e-06 | 7.456e-06 | 7.456e-06 | 1.666e-07 | 5.856e-01 | 1.0 | 3 |

## Interpretation

The `blend` mode is not sufficient for certification by itself. It can make the
active blended mass row small while the raw FV mass defect remains large:

```text
chi_mass=0.55 blend:
    active = 5.78e-6
    raw FV mass = 1.18e-2
```

So blend is useful only as a homotopy/scout; certification must use raw FV mass.

Direct `replace` mode is the important result. Increasing the released overlap
halo makes the raw FV mass defect fall monotonically:

```text
halo 8  -> raw FV mass = 2.98e-4
halo 16 -> raw FV mass = 1.30e-4
halo 24 -> raw FV mass = 5.30e-5
halo 32 -> raw FV mass = 7.46e-6
```

At `halo=32`, mass replacement is strict under the replacement active residual:

```text
active residual = 7.46e-6
outside old     = 5.78e-6
raw FV mass     = 7.46e-6
interface C0    = 1.67e-7
```

This means GPT's row-replacement diagnosis was correct. The previous `O(5e-5)`
mass floor was mainly a too-narrow/over-constrained source-buffer interface
problem, not physical branch loss.

The old source midpoint audit becomes large, `~0.586`, in direct replacement.
This is expected and should not be used as a production veto inside the
replacement band, because the identity audit already showed old midpoint source
rows are inconsistent with endpoint-compatible ODE/FV views.

## Implicit-Row Scouts

After the strict halo32 mass-replacement checkpoint, I tried turning on tiny
implicit radial/energy rows.

Small halo8 scout from the halo32 mass checkpoint:

```text
run: m5_source_band_rowreplace_halo8_from_halo32_impl0001_eta100_N164
chi_impl = 0.001
alpha = 0
active = 1.502e-2
outside old = 1.502e-2
raw FV mass = 7.451e-6
implicit ODE = 4.563e-3
Simpson = 1.095e-4
condition(A) max = 1.80e5
```

This failed because using halo8 reactivated old rows in the halo32 region that
had moved during the mass-replacement solve. So implicit continuation must use
the same wide overlap as the mass-certified state.

Wide halo32 implicit scout:

```text
chi_impl = 0.001
halo = 32
x_scale = jac
diff_step = 1e-5
max_nfev = 20
```

This was stopped as cost-limited before writing an artifact. Even with sparse
finite-difference column scaling, each implicit evaluation is too expensive for
routine continuation at this block size.

### Evaluate-Only Implicit Diagnostics

I added an evaluate-only path so the source-band replacement residual can be
assembled without entering the expensive local least-squares solve.

Profile-slope seed, same halo32 mass-certified checkpoint:

| chi_impl | active | raw FV mass | implicit ODE | Simpson | outside old | cond(A) max | smin(A) min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-4` | 4.563e-04 | 7.456e-06 | 4.563e-04 | 1.095e-05 | 5.780e-06 | 1.800e+05 | 2.380e-05 |
| `3e-4` | 1.369e-03 | 7.456e-06 | 1.369e-03 | 3.284e-05 | 5.780e-06 | 1.800e+05 | 2.380e-05 |
| `1e-3` | 4.563e-03 | 7.456e-06 | 4.563e-03 | 1.095e-04 | 5.780e-06 | 1.800e+05 | 2.380e-05 |

This shows the active residual scales almost exactly with `chi_impl`; the
unweighted ODE mismatch is `~4.56`, while mass and outside rows stay strict.

I then tested whether the mismatch was just a bad auxiliary slope seed.

At `chi_impl=1e-4`:

| seed | active | raw FV mass | implicit ODE | Simpson | outside old | cond(A) max | smin(A) min | max g_node | max g_mid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `profile` | 4.563e-04 | 7.456e-06 | 4.563e-04 | 1.095e-05 | 5.780e-06 | 1.800e+05 | 2.380e-05 | 2.394e+01 | 3.465e+01 |
| `implicit_lstsq` | 3.558e-04 | 7.456e-06 | 3.558e-04 | 8.511e-07 | 5.780e-06 | 6.844e+05 | 6.202e-06 | 3.144e+02 | 1.002e+02 |
| `blend`, 0.001 ODE | 1.144e-02 | 7.456e-06 | 1.144e-02 | 7.599e-04 | 5.780e-06 | 1.800e+05 | 2.380e-05 | 2.368e+01 | 5.638e+02 |
| `ode` | 2.005e+00 | 7.456e-06 | 2.005e+00 | 6.806e-03 | 5.780e-06 | 5.455e+06 | 7.756e-07 | 1.383e+03 | 5.000e+03 |

Pure ODE seeding is not usable here. It drives midpoint slopes into the
configured slope bound and worsens the implicit rows by orders of magnitude.
Even a `0.001` ODE blend is worse than the profile seed.

The `implicit_lstsq` seed is informative but not a solution. It solves a fixed
state linearized slope system and makes Simpson compatibility very small
(`8.5e-7`), so the state jump can be represented by auxiliary slopes. But the
ODE rows still sit at `3.6e-4` for `chi_impl=1e-4`, and the required slopes are
much larger. This points to a source-band ODE/state compatibility problem in
the ill-conditioned compact source annulus, not merely poor initial slopes.

I also retried the actual halo32 `chi_impl=1e-4` local solve with profile
slopes, `x_scale=jac`, `diff_step=1e-5`, and `max_nfev=40`. It remained inside
finite-difference least-squares for several minutes and was stopped before
writing an artifact. This is a numerical-efficiency bottleneck, not evidence
for physical branch loss.

## Current Status

Completed:

1. Added row modes: `append`, `blend`, `replace`, `audit`.
2. Stopped hard-stacking active old buffer rows in `blend/replace` modes.
3. Demoted slope-interface matching to an audit by default.
4. Added raw FV mass diagnostics separate from blended active rows.
5. Added local sparse-FD solver controls (`x_scale`, `diff_step`).
6. Certified mass-only replacement at `eta_E=100`, `halo=32`, `chi_mass=1`.
7. Added evaluate-only source-band replacement diagnostics.
8. Added profile/ODE/blend/linear-LS implicit slope seed modes.
9. Verified that explicit ODE slope seeding is harmful in the source band, and
   that linear-LS slopes fix Simpson compatibility but not the ODE residual
   floor.

Not yet completed:

1. Certified implicit radial/energy replacement.
2. Certified `eta_E=90`.
3. Removed the hidden ODE/FV contradiction for radial/energy rows.
4. Made the halo32 implicit local solve cheap enough for continuation.

## Next Recommendation

Do not lower `eta_E` yet.

The next numerical bottleneck is no longer FV mass replacement. It is implicit
radial/energy source-band rows plus the cost of their local Jacobian. The path
forward should be:

1. Freeze `m5_source_band_rowreplace_halo32_mass_eta100_N164` as the new
   mass-certified eta_E=100 source-band anchor.
2. Add a true analytic/block Jacobian or a reduced slope-only Newton block for
   implicit source-band rows:
   - local derivatives of `A g + c` with respect to `(logu, logT, logMdot, g)`;
   - exact Simpson compatibility derivatives;
   - exact C0 interface derivatives;
   - block row/column scaling.
3. Do not use explicit ODE-inversion slope seeding in this band. Prefer the
   profile seed or the linear-LS slope seed as a diagnostic.
4. Retry halo32 implicit continuation with:
   - `chi_impl = 1e-4, 3e-4, 1e-3`;
   - slope-interface inactive;
   - same halo32 overlap;
   - strict raw FV mass and outside-old audits.
5. Only after implicit rows are at least exploratory strict should `eta_E=95`
   or `eta_E=90` be retried.
