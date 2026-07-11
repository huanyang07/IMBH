# Signed-Flux Absolute-Stream Results

Date: 2026-07-11

## Motivation

The unified transonic solver cannot impose a physical tidal wall or open
decretion boundary. It requires positive mass flux and reconstructs

```text
Sigma = Mdot / (2 pi R u).
```

Setting its outer flux to zero would either violate state bounds or evacuate
the edge. It cannot represent finite-density stagnation. The new module
`signed_flux_disk.py` therefore evolves annular mass independently of radial
velocity and permits either sign of mass flux.

## Formulation

The first bridge model is wind-free, nearly Keplerian, and uses
Paczynski-Wiita gravity. The viscous torque is

```text
G = -2 pi R^3 nu Sigma dOmega_K/dR,
```

and the inward-positive face mass flux is obtained without division by mass
flux or velocity:

```text
Mdot = dG/dl_K.
```

Annular masses obey the exact finite-volume ledger

```text
dM_cell/dt = Mdot_outer - Mdot_inner + S_M.
```

The angular flux is

```text
J = Mdot l_K - G.
```

The code reports the difference between the angular change implied by the
fixed-Keplerian mass update and the boundary/source angular ledger. This is the
explicit mixing or external torque required by a source whose injected
specific angular momentum differs from the local disk value.

No surface-density clipping is used. Explicit steps that would create
non-positive annular mass are rejected. A sparse backward-Euler operator is
provided for viscous evolution.

## Ring-Spreading Validation

A finite-density ring was evolved for `2e-4` viscous times with zero inner
torque and a tidal outer wall. All resolutions use 20 implicit steps.

| N | mass budget error | angular budget error | variance initial | variance final | signed flow |
|---:|---:|---:|---:|---:|---|
| 64 | `2.64e-16` | `3.49e-16` | `0.133745` | `0.134229` | inflow + decretion |
| 128 | `3.96e-16` | `3.49e-16` | `0.133911` | `0.134395` | inflow + decretion |
| 256 | `1.32e-16` | `1.74e-16` | `0.133953` | `0.134436` | inflow + decretion |

The original explicit pilot required more than 20,000 positivity-limited steps
at N256. Backward Euler removes this diffusion-CFL bottleneck while retaining
machine-level integrated conservation.

## Absolute Stream Supply

The steady pilot prescribes

```text
Mdot_stream = 5 Mdot_Edd
R_source    = 240 rg
R_circ      = 248.96693 rg
alpha       = 0.01
H/R         = 0.1  (prescribed-viscosity control)
R_out       = 335 rg.
```

The stream source is normalized exactly in cell-integrated mass and carries
one explicit specific angular momentum from the circularization orbit.

### Tidal wall

| N | Mdot_inner/Mdot_stream | Mdot_outer/Mdot_stream | outer torque/stream J | required mixing torque/stream J |
|---:|---:|---:|---:|---:|
| 128 | `1.000000` | `0` | `0.75189424` | `-0.01709235` |
| 256 | `1.000000` | `0` | `0.75189403` | `-0.01709255` |
| 512 | `1.000000` | `0` | `0.75189398` | `-0.01709260` |

All supplied mass accretes. The outer surface density remains finite and the
required tidal torque is an output rather than a fitted angular offset.

### Open zero-torque boundary

| N | Mdot_inner/Mdot_stream | outward overflow/Mdot_stream | stagnation radius |
|---:|---:|---:|---:|
| 128 | `0.18851166` | `0.81148834` | `223.633 rg` |
| 256 | `0.18851188` | `0.81148812` | `223.608 rg` |
| 512 | `0.18851194` | `0.81148806` | `223.614 rg` |

The open solution contains a regular finite-density zero-flux crossing. The
inner accretion rate emerges as about `0.943 Mdot_Edd`, while about
`4.057 Mdot_Edd` leaves through the outer boundary.

The `-1.709%` mixing-torque requirement is stable with resolution. It comes
from depositing material with the angular momentum of `R_circ=248.97 rg` in a
finite band centered at `240 rg`; it must be included explicitly in the next
angular closure rather than hidden in a torque fraction.

## Scientific Status

The mass/angular transport problem now has a mathematically regular answer:

- a tidal wall processes the full supply and exports angular momentum to the
  companion;
- an open edge produces an accretion/decretion split and finite stagnation;
- failure of steady processing can later be represented by accumulation;
- no equation divides by `Mdot` or `u`.

This is not yet a hot-branch calculation. Viscosity and `H/R` are prescribed,
and the module does not yet evolve internal energy or radiation. The next
implementation must add thermal energy, ballistic stream energy, and interface
flux coupling to the existing inner no-wind slim solver.

## Reproduction

```text
scripts/run_signed_flux_ring_validation.py
scripts/run_absolute_stream_signed_flux_pilot.py
```

Verification:

```text
216 passed, 4 subtests passed
```
