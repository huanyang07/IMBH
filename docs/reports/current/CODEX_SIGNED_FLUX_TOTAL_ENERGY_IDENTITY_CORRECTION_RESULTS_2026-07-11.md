# Signed-Flux Total-Energy Identity Correction Results

Date: 2026-07-11

## Scope

This work corrects the mixed energy representation in commit `248e43c` before
inner transonic coupling. The prototype transported column enthalpy but used
the pressure-work correction associated with internal-energy transport.

## Corrected Identity

The inward energy flux remains

```text
F_E = Mdot (q + e + Pi/Sigma) - Omega G.
```

Its compatible one-zone work is

```text
W_H = Mdot (Pi dSigma/Sigma^2 - P drho/rho^2)
    = Mdot (P/rho) dlnH.
```

The superseded term differs by `Mdot d(Pi/Sigma)`. It remains valid only when
paired with an internal-energy flux. In source-bearing regions that alternative
also requires the corresponding local mass-source enthalpy transformation.

Both the signed reservoir and ordinary unified conservative residual now use
one shared `enthalpy_vertical_work` operator. The explicitly named legacy
internal-energy audit retains its matching old work term.

## Identity Gates

- Direct analytic differentiation of the actual transonic enthalpy flux closes
  against the entropy equation at `6.316e-15` normalized.
- The uncorrected pointwise mismatch is `0.13353`, so this is a discriminating
  test rather than a near-zero-work case.
- Manufactured cell-work errors at `N=64,128,256,512` are
  `9.410e-9`, `1.178e-9`, `1.474e-10`, and `1.842e-11`.
- Each resolution doubling reduces the maximum cell-integral error by about a
  factor of eight.
- A nonconstant-Mdot manufactured stream case closes mass, angular momentum,
  and energy together and shows the same convergence.

The bookkeeping-only roundoff check is now named
`total_energy_telescoping_defect`. It is distinct from the nonlinear equation
residual, direct identity mismatch, and manufactured-solution error.

## Corrected Near-ISCO Witness

| N | boundary | converged | viscosity mismatch | max H/R | Lrad/LEdd |
|---:|---|---:|---:|---:|---:|
| 256 | wall | no | `0.5298` | `0.32682` | `1.27694` |
| 256 | open | yes | `2.39e-4` | `0.14324` | `0.47741` |
| 512 | wall | no | `3.722` | `0.32673` | `1.27976` |
| 512 | open | no | `1.704` | `0.14352` | `0.47843` |

The energy equation closes while the alpha-viscosity fixed point fails in the
near-ISCO cells. The prior rejection of a fixed-Keplerian reservoir down to
`6.1 rg` therefore survives the identity correction.

## Corrected Rin=10 Controls

| boundary | inner/stream | outer/stream | max H/R | Lrad/LEdd | tau min | viscosity mismatch | energy residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall | `1.000000` | `0` | `0.31141` | `1.02495` | `24.72` | `8.10e-5` | `2.60e-11` |
| open | `0.173337` | `-0.826663` | `0.12780` | `0.35516` | `24.93` | `2.22e-4` | `6.20e-13` |

Relative to the mixed-pairing payload, the wall changes by about `+9.5%` in
maximum `H/R` and `+8.1%` in luminosity. The open control changes by about
`+3.6%` and `+2.3%`, respectively. The previous canonical states are therefore
superseded.

The physical status does not improve: the N512 pressure-force maxima remain
boundary dominated (`7.29` wall and `1.25` open), and the wall pressure-force
fraction is `0.278` at `12 rg` and `0.086` at `15 rg`. Effective optical depth
is not yet available. These are numerical interface controls, not physical
inner matches.

## Canonical Evidence

The regenerated payloads carry
`energy_identity_revision=enthalpy_vertical_work_v2`:

```text
results/canonical/signed_flux_total_energy_near_isco_failure/
results/canonical/signed_flux_total_energy_rin10_N512/
```

Commit `248e43c` remains the immutable historical prototype.

## Next Gate

Add a prescribed inner boundary for `(Mdot,J,F_E)` and a common flux extractor,
then certify a contiguous overlap band across approximately `12-60 rg` before
attempting a two-domain solve. The overlap audit must include effective optical
depth in addition to radial pressure, thickness, Mach number, angular gradient,
and radial scale separation.

## Verification

```text
238 passed, 4 subtests passed
```
