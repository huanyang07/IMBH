# Mdot=5 Lobatto Interface Defect Results

Date: 2026-07-09

Target:

- `Mdot_inner/Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact source
- local-Mdot wind
- `eta_E = 98.125`
- `N = 164`

Starting checkpoint:

```text
outputs/checkpoints/m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164/stage_00_etaE_98p125_N164.npz
```

## Implementation

Added Lobatto interface diagnostics:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_INTERFACE_DIAGNOSTIC=1
```

The diagnostic reports, at the peak true Lobatto radial row:

- peak interval, point, radius, component;
- source Lobatto state and polynomial slope;
- neighbor finite-difference slope;
- direct local ODE slope `g_direct = -A^{-1}c`;
- radial decomposition:
  - `A_Ru g_u`
  - `A_RT g_T`
  - `c_R`
  - total radial row;
- source-vs-direct and source-vs-neighbor slope distances.

Added targeted one-element interface relaxation:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_INTERFACE_RELAXATION=left
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_INTERFACE_RELAXATION_INTERVALS=1
```

This extends the Lobatto corrector window by exactly one left-neighbor interval. The checkpoint loader now supports partial Lobatto aux reuse, so a new window such as `132..158` preserves the corrected checkpoint aux on `133..158` and initializes only the new interface interval from the current state.

Added optional weak derivative compatibility rows:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_INTERFACE_COMPATIBILITY_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_INTERFACE_COMPATIBILITY_SIDE
```

These rows compare the source-edge Lobatto derivative against the neighboring finite-difference derivative. They do not replace the physical ODE rows.

## Interface Diagnostic

Best-seed evaluate-only run:

```text
m5_eta_lobatto_interface_diag_best_98p125_N164
```

Peak:

```text
interval = 133
point    = left
R        = 203.104 rg
component = radial
radial row = 0.197252
neighbor interval = 132
```

Radial decomposition at the peak:

```text
A_Ru*g_u  =  0.010738
A_RT*g_T  = -0.045130
c_R       =  0.231644
total     =  0.197252
```

Slope comparison:

```text
source Lobatto g      = ( 1.2266, -0.6323)
neighbor FD g         = (-1.6091, -2.1892)
direct -A^{-1}c g     = (-200.55, 21.35)
source-neighbor norm  = 3.235
source-direct norm    = 202.97
```

Using the neighbor finite-difference slope with the source coefficients would reduce the radial row:

```text
source slope residual   = 0.197252
neighbor slope residual = 0.061311
```

Interpretation:

- The radial row is derivative/interface driven.
- The direct ODE slope exists algebraically but is enormous, indicating a near-singular local radial row.
- The neighboring slope is much more physical and already reduces the residual by a factor of about 3.

## Relaxation Tests

| run | final_full | true ODE | radial | energy | FV mass | peak | source-neighbor |
|---|---:|---:|---:|---:|---:|---|---:|
| `m5_eta_lobatto_interface_diag_best_98p125_N164` | 5.980e-01 | 1.973e-01 | 1.973e-01 | 6.221e-03 | 1.538e-04 | 133/left/radial @ 203.1 rg | 3.235e+00 |
| `m5_eta_lobatto_interface_relax_left1_mass100_98p125_N164` | 8.803e-01 | 1.858e-01 | 1.858e-01 | 4.475e-03 | 6.107e-05 | 132/left/radial @ 200.9 rg | 4.043e+00 |
| `m5_eta_lobatto_interface_compat_w0005_98p125_N164` | 9.899e-01 | 1.750e-01 | 1.750e-01 | 4.179e-03 | 5.182e-05 | 145/left/radial @ 232.9 rg | 7.116e+01 |
| `m5_eta_lobatto_interface_compat_w002_98p125_N164` | 9.655e-01 | 1.733e-01 | 1.733e-01 | 4.168e-03 | 5.695e-05 | 145/left/radial @ 232.9 rg | 6.954e+01 |

## Findings

One-element left relaxation:

- reduces radial `0.197 -> 0.186`;
- improves energy `6.22e-3 -> 4.48e-3`;
- improves FV mass `1.54e-4 -> 6.11e-5`;
- but worsens the global full residual and moves the radial peak to the new left edge, interval `132`.

Weak compatibility rows:

- reduce radial to `0.173..0.175`;
- improve energy and FV mass;
- but move the peak to interval `145` and create a very large source-neighbor derivative mismatch, `~70`.

So the compatibility row is useful as a diagnostic but is not production-safe yet.

## Interpretation

The remaining defect is not fixed by simply adding one interval or weakly matching the boundary derivative. The peak behaves like a moving left-interface/front condition in a nearly singular radial row:

- one more interface element moves the peak to the new left edge;
- weak compatibility moves the defect into the source interior;
- the physical energy and mass rows stay much better than before, so the hard remaining problem is radial derivative regularization/transition.

This supports GPT's interpretation that the source element is internally much better than before, but it also shows that the source-to-global radial interface needs a formulation-level transition rather than a single weak derivative row.

## Recommended Next Step

Do not continue eta yet.

Next implementation should be a dedicated radial transition/buffer formulation:

1. Add a small two- or three-element Lobatto transition block on the left side.
2. Keep true radial ODE active in the transition block.
3. Keep energy as a guard or lower-weight row there.
4. Use finite-volume mass rows throughout.
5. Add smoothness/compatibility on derivative change, not direct equality to a single neighbor slope:

```text
(z'_i - z'_{i-1}) / dx = bounded or weakly penalized
```

6. Accept only if:
   - radial decreases monotonically without moving to the new left edge;
   - energy remains `< few e-3`, ideally `<1e-3`;
   - FV mass remains `<1e-4`;
   - global full residual does not regress strongly.

Best current seed remains:

```text
m5_eta_lobatto_state_corrector_mass100_node1e3_98p125_N164
```

The one-element and weak-compatibility checkpoints are diagnostic, not production anchors.
