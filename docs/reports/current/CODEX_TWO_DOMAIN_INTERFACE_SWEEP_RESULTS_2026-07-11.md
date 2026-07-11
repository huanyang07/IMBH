# Two-Domain Interface Sweep Results

## Scope

The certified no-wind `Mdot/Mdot_Edd=5` transonic profile supplies

```text
(Mdot, J, F_E)
```

at interfaces near `30`, `40`, `50`, and `60 r_g`. Each flux triple drives a
new angularly closed, total-energy, tidal-wall reservoir extending to
`335 r_g` with an absolute `5 Mdot_Edd` stream. The sweep uses reservoir meshes
`N=128` and `N=256`.

This is a one-way no-wind coupling test. It checks whether the two existing
descriptions can form an interface-independent conservative composite before a
fully coupled eigenvalue solve is attempted.

## Numerical Result

All eight reservoir roots converge. At `N=256`:

- maximum relative `(Mdot, J, F_E)` mismatch: `2.05e-16`;
- composite luminosity: approximately `1.456-1.459 L_Edd`;
- composite-luminosity interface-position spread: `0.203%`;
- outer tidal torque: `0.7599099` of the stream angular flux;
- maximum `H/R`: `0.300-0.320`.

The `N=128` composite-luminosity spread is `0.236%`. Between `N=128` and
`N=256`, the composite luminosity changes by at most `0.050%`, while maximum
`H/R` changes by at most `0.53%`.

The outer luminosity alone changes strongly with interface radius, as it must:
more radiation is assigned to the inner domain when the interface moves
outward. Adding the inner and outer luminosities removes that bookkeeping
dependence.

## Primitive-State Gate

The smooth-match gate fails on both meshes.

| Interface target | N256 maximum primitive mismatch | Dominant term |
|---:|---:|---|
| `30 r_g` | `0.3320` | integrated pressure |
| `40 r_g` | `0.3269` | integrated pressure |
| `50 r_g` | `0.3295` | integrated pressure |
| `60 r_g` | `0.3339` | integrated pressure |

The reservoir surface density is lower by approximately `12-22%` in log units
across the sweep, while its Keplerian angular frequency differs from the
pressure-supported transonic value by roughly `4-5%`. These mismatches are
mesh-stable and do not shrink as the interface moves.

## Verdict

The prescribed-flux machinery works, and the total-energy conventions are
consistent across the two domains. The result is therefore a valid
conservative numerical composite.

It is not a smooth physical disk solution. The persistent pressure jump means
that the fixed-Keplerian outer closure and pressure-supported inner closure do
not share the same primitive state even when they carry identical conserved
fluxes.

The next implementation should promote the reservoir rotation to radial force
balance,

```text
Omega^2 = Omega_K^2 + (1 / (R Sigma)) dPi/dR,
```

with the corresponding non-Keplerian angular momentum and shear used in the
viscous torque. An explicit resolved transition layer is the fallback if that
closure does not remove the mismatch.

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_two_domain_interface_sweep.py
PYTHONPATH=src python3 scripts/build_two_domain_interface_sweep_canonical.py
```
