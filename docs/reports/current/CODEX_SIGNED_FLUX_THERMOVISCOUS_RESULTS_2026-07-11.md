# Signed-Flux Thermoviscous Results

Date: 2026-07-11

> Superseded for physical interpretation by
> `CODEX_SIGNED_FLUX_ANGULAR_CLOSURE_RESULTS_2026-07-11.md`. The `f_adv`
> terminology below refers only to internal-energy export.

## Scope

This work adds thermal energy to the independent-surface-density, signed-flux
disk. The absolute source remains

```text
Mdot_stream = 5 Mdot_Edd
Rinj        = 240 rg
Rcirc       = 248.96693 rg
Rout        = 335 rg
alpha       = 0.01.
```

Both tidal-wall and open zero-torque outer boundaries are tested. No wind is
included.

## Thermal Ledger

The evolved annular thermal energy is

```text
E_th,i = Sigma_i e_i A_i,
```

using the existing Paczynski-Wiita gas+radiation vertical closure. Upwind
internal-energy transport uses the signed face mass flux:

```text
F_e = Mdot e_donor.
```

The fixed-grid thermal equation is

```text
dE_th,i/dt = F_e,out - F_e,in
             + Qvisc A_i
             + Mdot_stream,i (B_stream-Eorb,i)
             - Qrad A_i.
```

The stream impact term is derived from one injected circularization state;
mass, angular momentum, and energy are not independently tuned.

Radiative cooling is treated implicitly and temperature is recovered from the
monotonic gas+radiation internal-energy relation without clipping. The steady
thermal solve uses the same cell energy ledger.

## Thermoviscous Coupling

The initial fixed-viscosity pilot was followed by a damped fixed point:

```text
nu = alpha H(Sigma,T)^2 Omega_K.
```

Each outer iteration solves steady signed mass transport, then steady thermal
balance, then refreshes viscosity. Both boundary families converge in 33
iterations with maximum log-viscosity mismatch below `0.002`.

## Tidal-Wall Results

| N | Mdot_inner/Mdot_stream | f_adv | max H/R | Lrad/LEdd | tau_min | thermal residual |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | `1.000000` | `0.54504` | `0.34464` | `1.33153` | `16.20` | `8.0e-13` |
| 128 | `1.000000` | `0.54703` | `0.34271` | `1.32501` | `6.33` | `4.5e-12` |
| 256 | `1.000000` | `0.54801` | `0.34176` | `1.32196` | `2.70` | `1.6e-12` |
| 512 | `1.000000` | `0.54850` | `0.34129` | `1.32049` | `1.23` | `1.2e-11` |

The tidal wall therefore supports a mesh-converging, optically thick,
moderately thick, strongly advective thermal state in the outer/reservoir
model. The wall torque allows the full stream supply to accrete.

## Open-Overflow Results

| N | Mdot_inner/Mdot_stream | overflow fraction | f_adv | max H/R | Lrad/LEdd | tau_min |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | `0.188511` | `0.811489` | `0.05517` | `0.15780` | `0.51902` | `30.31` |
| 128 | `0.188512` | `0.811488` | `0.05436` | `0.15752` | `0.51916` | `12.06` |
| 256 | `0.188512` | `0.811488` | `0.05394` | `0.15737` | `0.51932` | `5.19` |
| 512 | `0.188512` | `0.811488` | `0.05372` | `0.15730` | `0.51942` | `2.38` |

The open boundary remains a cooler accretion/decretion solution. Most supplied
mass carries angular momentum out through the edge instead of heating and
draining through the inner disk.

## Stream Heating

Because `Rinj` and `Rcirc` are close, direct stream impact contributes only

```text
0.14% of viscous power  (tidal wall)
0.75% of viscous power  (open overflow).
```

The thermal split is therefore controlled mainly by global angular transport
and throughput, not an arbitrarily strong impact-heating term.

## Conservation and Validity

- Fixed-Sigma energy roots close below `2e-11` normalized.
- Global energy-ledger defects are at roundoff.
- Thermoviscous feedback preserves the mass split from the signed-flux solve.
- All N512 cells remain optically thick, although the tidal inner-interface
  cell is close to the diffusion limit (`tau~1.23`).
- No negative mass or thermal-energy clipping is used.

## Scientific Interpretation

This is the first robust hot/advective solution found under a physical absolute
stream supply and explicit tidal boundary. It is evidence that the desired hot
topology can emerge when angular momentum has a physical sink and the inner
rate is allowed to respond to the feeding problem.

It is not yet the final advective slim-disk branch. The current thermal flux
advects internal energy, not the complete enthalpy/mechanical energy flux; the
pressure-work identity has not yet been promoted into this new core. The inner
boundary is nearly Keplerian and must still be matched to the certified
transonic slim solution.

## Next Steps

1. Promote the thermal equation to a total-energy/enthalpy conservative ledger
   with explicit pressure work and verify equivalence to the slim entropy form.
2. Match mass, angular momentum, and energy fluxes to the existing inner
   no-wind transonic solver at an optically thick interface.
3. Add coupled IMEX time evolution of annular mass and thermal energy.
4. Test whether the tidal hot state is steady, accumulating, or cyclic when
   the full source rate and thermal instability are evolved.
5. Add wind only after the no-wind total-energy interface closes.

## Reproduction

```text
scripts/run_signed_flux_thermal_pilot.py
scripts/run_signed_flux_thermoviscous_pilot.py
```

Verification:

```text
220 passed, 4 subtests passed
```
