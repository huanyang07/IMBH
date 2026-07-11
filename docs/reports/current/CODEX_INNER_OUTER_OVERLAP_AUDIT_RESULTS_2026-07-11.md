# Inner/Outer Overlap Audit Results

## Scope

This work package compares the certified no-wind `Mdot/Mdot_Edd=5` transonic
benchmark with the corrected `R_in=10 r_g`, `N=512` signed total-energy wall
and open controls. The search window is `12-60 r_g`.

The common gates are:

| Metric | Primary gate |
|---|---:|
| radial pressure-force fraction | `<= 0.05` |
| `d ln(l_K)/d ln(R)` | `>= 0.2` |
| `H/R` | `<= 0.35` |
| radial Mach number | `<= 0.1` |
| scattering depth | `>= 10` |
| effective optical depth | `>= 1` |
| shortest `Sigma/T/H` gradient length divided by `H` | `>= 3` |
| normalized stream source per cell | `<= 1e-8` |

Effective depth is evaluated as

```text
tau_eff = sqrt(tau_abs * (tau_abs + tau_es)).
```

Because the production model has no absorption-opacity closure, this audit
uses a broad diagnostic Kramers bracket. The lower coefficient is used for the
acceptance gate. It does not modify cooling or certify the opacity model.

## Primary Result

No common band passes every primary gate.

| Domain | Primary passing band |
|---|---|
| inner transonic benchmark | none |
| tidal-wall reservoir | none |
| open reservoir | `14.733-59.693 r_g` |
| common transonic + wall | none |
| common transonic + open | none |

The wall and transonic profiles each fail for a combination of radial pressure
support and radial gradient length. The low-opacity effective-depth estimate
also removes their innermost candidate cells. Their scattering depths remain
large, so scattering depth alone would have hidden this limitation.

## Pressure Sensitivity

Repeating only the radial-pressure gate at `epsilon_P <= 0.10` gives:

| Common domains | Candidate band |
|---|---|
| transonic + wall | `29.453-59.693 r_g` |
| transonic + open | `24.197-59.693 r_g` |

These are candidate experiment bands, not certified overlap bands. The result
depends directly on accepting pressure support as large as ten percent in the
nominally Keplerian reservoir.

## Scientific Interpretation

The audit does not justify a direct production splice at a fixed radius. It
does justify a controlled prescribed-flux interface sweep at approximately
`30`, `40`, `50`, and `60 r_g`. That experiment must require:

- closure of `(Mdot, J, F_E)` with the shared sign and energy conventions;
- convergence of primitive-state mismatches rather than fluxes alone;
- weak dependence on interface position;
- and reduction, not concealment, of the pressure-force mismatch.

If those conditions fail, the outer model needs radial momentum and
non-Keplerian rotation. Loosening the pressure gate further is not an acceptable
substitute.

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_inner_outer_overlap_audit.py
PYTHONPATH=src python3 scripts/build_overlap_audit_canonical.py
```
