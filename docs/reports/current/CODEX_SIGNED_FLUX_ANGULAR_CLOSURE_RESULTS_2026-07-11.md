# Signed-Flux Angular-Closure Results

Date: 2026-07-11

## Scope

This work implements WP0-WP1 from the review of commit `53566fa`. It freezes
the previous wall/open states, fixes the thermoviscous final-state acceptance
check, introduces one immutable stream source, and includes the stream angular
moment dynamically in the steady finite-volume solution.

No total-energy redesign, physical distributed tidal law, inner transonic
match, time-dependent source evolution, or wind was added.

## Conservative Source and Angular Ledger

The source carries one set of cell-integrated moments `(S_M,S_J,S_E)`. The
inward-positive face fluxes obey

```text
Mdot[i+1] - Mdot[i] + S_M[i] = 0
J[i+1] - J[i] + S_J[i] + T_ext[i] = 0
J = Mdot l_K - G.
```

The open control imposes zero viscous torque at both edges. The ideal tidal
wall imposes zero outer mass flux and returns the required outer torque. The
old mass-only solve is retained only for reproduction. Source-bearing time
steps reject a nonzero angular defect until coupled angular evolution exists.

## Analytic Controls

For `Rin=6.1 rg`, `Rout=335 rg`, and
`l_s=l_K(248.96693 rg)`, N128, N256, and N512 all give

```text
Mdot_in/Mdot_stream                 = 0.17006459595780934
G_out/(Mdot_stream l_s), wall      = 0.768986583605257
maximum unnamed angular defect     < 9e-16 relative.
```

The open stagnation radius converges near `222.35 rg`. These results agree
with the discrete boundary identities and continuum targets `0.1700646` and
`0.7689866`.

## Thermoviscous Decision Gate

| Closure and boundary | inner/stream | outer/stream | internal-energy export | max H/R | Lrad/LEdd | outer torque/stream J | unnamed angular defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| `53566fa` wall | `1.000000` | `0` | `0.548496` | `0.341291` | `1.320491` | `0.751894` | `-0.017093` |
| WP1 wall | `1.000000` | `0` | `0.548047` | `0.341310` | `1.322932` | `0.768987` | `<9e-16` |
| `53566fa` open | `0.188512` | `-0.811488` | `0.053718` | `0.157296` | `0.519418` | `0` | `-0.017093` |
| WP1 open | `0.170065` | `-0.829935` | `0.036550` | `0.146436` | `0.479212` | `~0` | `<4e-16` |

The hot wall candidate survives the physical stream-angular closure with
negligible change in thickness and a `0.18%` increase in luminosity. The open
state becomes cooler because less supplied mass reaches the inner edge.

## Corrected Diagnostics

The previous `f_adv`-like quantity is now
`internal_energy_export_fraction`. It is not the slim-disk entropy-advection
fraction. The telescoping check is `internal_energy_ledger_defect`; it does not
certify total energy.

After final polishing, the N512 log-viscosity mismatches are `1.04e-6` for the
wall and `1.90e-4` for the open state, below the `2e-3` production tolerance.

## Validity Audit

At N512, the minimum scattering optical depths are `1.229` for the wall and
`3.137` for the open state. At fixed `10 rg`, the radial-pressure force
fractions are `0.1160` and `0.0222`, respectively. The wall exceeds the
provisional `0.10` fixed-Keplerian gate near `10 rg`, while the full-grid
maximum diverges near the `6.1 rg` boundary where `d ln l_K/d ln R` approaches
zero. The reservoir must not be used to the ISCO in production.

Only scattering optical depth exists in the current closure. Effective
optical depth is reported as unavailable rather than inferred without an
absorption opacity.

## Scientific Status

The steady angularly closed solutions are numerically supported. The wall is
still a diagnostic hot-reservoir candidate because the thermal ledger is not
total energy, the wall lacks a binary-calibrated torque and power law, the
inner Keplerian approximation fails, and stability is untested.

## Canonical Evidence

```text
results/canonical/signed_flux_legacy_53566fa_N512/
results/canonical/signed_flux_angular_closed_wp1_N512/
```

Both cases contain prescribed-viscosity and thermoviscous wall/open states,
configuration, summary, provenance, and checksums.

## Next Gate

The next implementation is the total-energy column ledger with face flux

```text
F_E = Mdot B_col - Omega G,
```

including source energy, radiation, vertical work, and named tidal power
without double-counting viscous work. Wind remains deferred.

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_absolute_stream_signed_flux_pilot.py
PYTHONPATH=src python3 scripts/run_signed_flux_thermoviscous_pilot.py
PYTHONPATH=src python3 scripts/build_signed_flux_wp1_canonical.py
```

Verification:

```text
226 passed, 4 subtests passed
repository hygiene passed
```
