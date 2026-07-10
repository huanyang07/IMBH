# Mdot=5 angular-momentum ledger results

Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.

Sign convention: `Mdot` is inward-positive, `dMdot/dlnR = Mwind' - Mstream'`, and the net inward angular flux is `J=Mdot*l-G`. The conservative ledger is

```text
dJ/dlnR = Mwind' * l_w - Mstream' * l_s + tau_ext.
```

## Closure comparison

| closure | phase point max | phase FV max | global point max | required torque correction | peak R (rg) |
|---|---:|---:|---:|---:|---:|
| `representation` | 6.770e-17 | 9.754e-06 | 7.033e-16 | 0.000e+00 | 225.351 |
| `local_disk_prescribed` | 4.366e-01 | 5.383e-04 | 7.915e+00 | 0.000e+00 | 225.469 |
| `local_disk_required` | 4.517e-17 | 9.754e-06 | 5.169e-16 | 4.366e-01 | 225.048 |
| `keplerian_injection_prescribed` | 5.119e-01 | 6.069e-04 | 8.038e+00 | 0.000e+00 | 225.469 |
| `keplerian_injection_required` | 4.517e-17 | 9.754e-06 | 5.169e-16 | 5.119e-01 | 225.048 |
| `keplerian_local_prescribed` | 4.916e-01 | 5.806e-04 | 7.991e+00 | 0.000e+00 | 225.469 |

## Finding

The exact `representation` closure assigns source and wind material the specific angular momentum carried by the algebraic net flux, `J/Mdot=l-G/Mdot`, and treats `Mdot*d(stream_l)/dlnR` as a separate torque. Its pointwise ledger closes algebraically.

The previous provisional audit instead assigned both source and wind the full local disk `l` while retaining the same explicit torque. When `Mdot` varies, this omits the viscous-loading correction `-Mdot' G/Mdot`.

For the accepted phase branch, that provisional point defect reaches `4.366e-01` and its FV defect reaches `5.383e-04`. The exact representation FV floor is `9.754e-06`.

Allowing the external torque to absorb the missing local-disk loading term restores pointwise conservation; the required correction reaches `4.366e-01` in units of `Mdot_inner*lK` per `dlnR`.

## Production decision

- Keep `representation` as the exact audit of the current algebraic model.
- Do not call it a physical stream closure: its carried `l_s=l_w=J/Mdot` is a representation identity.
- For a physical model, specify `l_s(R)`, `l_w(R)`, and `tau_ext(R)` independently and promote the angular flux equation to production.
- The independent outer-manifold search can classify the current mathematical closure, but physical flux matching must report the explicit closure used.

## Files

- summary: `outputs/tables/m5_eta_angular_momentum_ledger_98p125_N164.json`
- profiles: `outputs/tables/m5_eta_angular_momentum_ledger_98p125_N164_profiles.json`
- figure: `outputs/figures/m5_eta_angular_momentum_ledger_98p125_N164.png`
