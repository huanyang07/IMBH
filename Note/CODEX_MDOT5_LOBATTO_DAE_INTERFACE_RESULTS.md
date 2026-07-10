# Mdot=5 Lobatto DAE Interface Results

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

Added an opt-in local DAE interface mode:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE=1
```

The DAE mode appends independent interface tangent variables to the existing Lobatto aux vector:

```text
g_U, g_Theta, g_F
```

For the first implementation the target is one point only:

```text
interval 133, left point, R = 203.1 rg
```

The state representation remains the true Lobatto polynomial. Only the target point's ODE evaluation uses the independent DAE tangent.

Added controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_SIDE
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_SEED_MODE
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_RADIAL_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_ENERGY_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_FPRIME_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_TANGENT_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_G_TRUST
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_FPRIME_TRUST
IMBH_MDOT5_LOCAL_MDOT_ETA_GLOBAL_FLUX_LOBATTO_DAE_INTERFACE_DIRECT_CLIP
```

Rows added at the DAE point:

- radial: `A_R(z) g + c_R(z)`;
- energy: `A_E(z) g + c_E(z)`;
- Fprime compatibility: `g_F - (Mwind_prime - Mstream_prime)/Mdot_inner`;
- soft tangent selection: `g - g_seed`.

The usual Lobatto FV mass row stays active. For the DAE target point it uses `g_F` in the Simpson mass integral.

Checkpoint output now stores:

```text
source_lobatto_element_aux_dae_targets
source_lobatto_element_aux_dae_g
```

## Results

| run | full | ODE | radial | energy | FV mass | DAE radial | DAE energy | DAE Fprime | DAE tangent | peak | nfev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `dae_eval_lobatto_98p125_N164` | 5.980e-1 | 1.973e-1 | 1.973e-1 | 6.221e-3 | 1.538e-4 | 1.973e-1 | 8.020e-4 | 0.000e0 | 0.000e0 | 133/left @ 203.1 | 1 |
| `dae_lobatto_w1e2_98p125_N164` | 1.097e0 | 1.864e-1 | 1.864e-1 | 3.022e-3 | 7.274e-5 | 1.154e-3 | 3.022e-3 | 3.827e-4 | 1.171e0 | 145/left @ 232.9 | 42 |
| `dae_lobatto_w1e2_anchor100_98p125_N164` | 5.953e-1 | 1.813e-1 | 1.813e-1 | 3.158e-2 | 6.139e-5 | 1.705e-1 | 3.158e-2 | 1.085e-5 | 1.602e0 | 134/left @ 205.3 | 6 |
| `dae_lobatto_rad10_e003_t1e3_anchor100_98p125_N164` | 5.824e-1 | 1.821e-1 | 1.821e-1 | 1.116e-1 | 6.219e-5 | 8.556e-6 | 1.116e-1 | 1.046e-3 | 1.830e0 | 134/left @ 205.3 | 28 |
| `dae_neighbor_rad10_e003_t1e3_anchor100_98p125_N164` | 5.824e-1 | 1.821e-1 | 1.821e-1 | 1.098e-1 | 6.219e-5 | 9.060e-6 | 1.098e-1 | 1.049e-3 | 8.855e-1 | 134/left @ 205.3 | 25 |

Final DAE tangents for the radial-focused anchored runs:

```text
lobatto seed:
    target = [interval_pos=0, interval=133, point=left]
    g = ( 1.8571, -2.4620, 0.15720 )

neighbor seed:
    target = [interval_pos=0, interval=133, point=left]
    g = ( -0.7236, -2.1451, 0.15486 )
```

## Interpretation

The DAE degree of freedom is real:

- the free-source run reduces the target DAE radial residual from `1.97e-1` to `1.15e-3`;
- the radial-focused anchored runs reduce the target DAE radial residual to `~9e-6`;
- FV mass remains good in the anchored radial-focused runs, `~6.2e-5`.

However, single-point DAE is not yet a physical interface solution:

- when the source state is free, the target DAE point improves but the residual migrates into the source interior;
- when the state is strongly anchored and DAE radial is forced down, the DAE energy row rises to `~0.11`;
- the global true Lobatto radial maximum remains `~0.18`, now at interval `134/left`, immediately next to the DAE point;
- neighbor seeding reduces tangent excursion but does not remove the energy tradeoff.

The stopped balanced scout with high radial and full energy weight was too slow with the current dense finite-difference Jacobian. That is a numerical-efficiency warning for future DAE work.

## Conclusion

The local DAE radial-interface concept is promising but incomplete. A single independent tangent at one Lobatto point can remove the target radial residual, but it converts the problem into either:

1. residual migration to a neighboring source point, or
2. a local energy residual at the same DAE point.

This supports moving from a single DAE point to a one-interval DAE element.

## Recommended Next Step

Implement a one-interval DAE Lobatto element over interval `133`:

- independent tangents at left/mid/right:

```text
g_U, g_Theta, g_F at L/M/R
```

- production rows:
  - radial ODE at L/M/R;
  - energy ODE at L/M/R;
  - FV mass using `g_F`;
  - Simpson compatibility between state polynomial and DAE tangents;
  - soft tangent regularization to Lobatto or neighbor seed.

Then test:

```text
seed = lobatto, neighbor
tangent_weight = 1e-3, 1e-2
radial_weight = 10
energy_weight = 1, 3
state/midpoint anchors = 100 initially
```

Acceptance remains:

- radial `<1e-2` first, then `<1e-3`;
- energy `<1e-3`;
- FV mass `<1e-4`;
- no migration of the peak to interval `134`.

Do not resume `eta_E` continuation yet.
