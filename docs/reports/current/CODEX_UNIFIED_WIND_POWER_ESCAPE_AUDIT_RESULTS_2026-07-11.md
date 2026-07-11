# Unified Wind-Power and Escape Audit

Date: 2026-07-11

## Scope

This audit uses the strict `eta_E=8` unified conservative checkpoints at
`N=426,512,640`. It changes the numerical representation of the wind-energy
sink without changing the wind mass law, angular-momentum lever arm, launch
multiplier, source, boundary conditions, or disk equations.

## Power-Primary Ledger

The previous carried-energy expression was

```text
Mdot_wind' B_wind.
```

The new production representation separates it into

```text
Mdot_wind' [B_disk + Omega (l_w-l)] + P_wind',
P_wind' = 2 pi R^2 Q_wind,
Mdot_wind' = P_wind' / E_launch.
```

Because

```text
B_wind = B_disk + Omega (l_w-l) + E_launch,
```

the two residuals are algebraically identical. Both modes remain available as
an explicit regression switch.

The eta=8 checkpoints confirm this identity:

| N | maximum residual | max mode difference | wind-energy split relative error |
|---:|---:|---:|---:|
| 426 | `2.1314e-5` | `8.47e-20` | `2.20e-16` |
| 512 | `2.1247e-5` | `4.24e-20` | `2.17e-16` |
| 640 | `2.1220e-5` | `8.22e-20` | `2.22e-16` |

No repolish was performed because the residual change is many orders of
magnitude below roundoff-relevant solver tolerances.

## Terminal-Bernoulli Audit

For a marginally unbound target `B_infinity=0`, the required launch energy is

```text
E_required = -B_disk - Omega (l_w-l).
```

The diagnostic compares this with the prescribed `E_launch` and evaluates

```text
B_wind = B_disk + Omega (l_w-l) + E_launch.
```

Results are mesh stable:

| N | wind/Mdot_inner | escaping wind-mass fraction | mass-weighted Bwind/c2 | mass-weighted v_inf/c |
|---:|---:|---:|---:|---:|
| 426 | `0.017113` | `>0.999999999999` | `0.10252` | `0.40746` |
| 512 | `0.017089` | `>0.999999999999` | `0.10211` | `0.40678` |
| 640 | `0.017084` | `>0.999999999999` | `0.10202` | `0.40665` |

Across active wind cells, the prescribed launch energy exceeds the positive
marginal-escape requirement by factors of approximately `10.36-38.3`. Thus
the eta=8 wind is already safely unbound under the present Bernoulli ledger.

The innermost cells have a formal Newtonian terminal-speed equivalent slightly
above `c` (`1.03-1.05 c` near `R=4.5-4.7 rg`). They contain only about
`1e-4-4e-4` of the wind mass. This is not a relativistic velocity prediction;
it is a validity warning that the Newtonian energy-limited launch multiplier
cannot be interpreted literally in the deepest potential.

## Scientific Interpretation

The previous low-eta wall was numerical and is now crossed, but eta=8 is not
close to a marginally escaping wind. Lowering eta further would primarily
increase mass per unit wind power. It is neither required to unbind the wind
nor supported as the main route to a distinct hot/advective topology.

The present branch is therefore:

- numerically mesh supported through eta=8;
- energetically unbound under its own prescribed Bernoulli ledger;
- weakly mass loaded at about `1.71%` of `Mdot_inner`;
- not a newly recovered hot branch;
- not yet a calibrated physical wind because eta remains an artificial launch
  multiplier and the inner terminal-speed diagnostic leaves the model's
  nonrelativistic domain.

## Next Step

1. Replace eta as the physical control with a target terminal Bernoulli or
   terminal speed, while retaining the power-primary energy variable.
2. Add a bounded mass-loading constraint when the marginal escape energy tends
   to zero.
3. Change the feeding problem to prescribed absolute stream supply with the
   inner accretion rate as an outcome.
4. Compare tidal-wall and open-overflow boundaries, allowing signed flux where
   required.
5. Introduce ballistic stream energy and conservative impact heating before
   claiming a search for a new hot topology.

## Reproduction

```text
scripts/run_unified_conservative_wind_power_escape_audit.py
outputs/tables/unified_conservative_wind_power_escape_audit.json
```

Verification:

```text
208 passed, 4 subtests passed
```
