# Pressure-Supported Interface Pilot Results

## Scope

This pilot promotes reservoir rotation from fixed Keplerian motion to the
radial-force target

```text
Omega^2 = Omega_K^2 + (1 / (R Sigma)) dPi/dR,
```

while using the resulting `Omega`, `l`, shear, torque work, orbital energy,
and alpha viscosity consistently. Pressure support is continued through
fractions `0.10`, `0.25`, `0.50`, `0.75`, and `1.0` from the converged
Keplerian state at an interface near `40 r_g`.

Raw trial profiles can contain positive shear or decreasing angular momentum.
The pilot projects log-rotation slopes onto

```text
0.2 <= d ln(l) / d ln(R)
d ln(Omega) / d ln(R) < 0
```

and reports the resulting force-balance mismatch. This projection is a
stability guard, not a claim that exact radial force balance was solved.

## Results

Eight damping/smoothing/resolution configurations were tested.

- All three `N=64` smoothing cases converge through full pressure support.
- None of the five `N=128` cases passes the `2e-3` viscosity/rotation gate.
- The `N=128`, damping `0.10` cases reach pressure fraction `0.50` before the
  fixed-point viscosity mismatch stalls near `0.004`.
- The `N=128`, damping `0.05` case reaches fraction `0.75` but stalls near
  `0.009`.

For the converged coarse roots:

| Metric | Value near `40 r_g` |
|---|---:|
| relative conserved-flux mismatch | `<2.1e-16` |
| rotation mismatch | `0.00366` |
| integrated-pressure log mismatch | `-0.3560` |
| surface-density log mismatch | `-0.3107` |
| maximum radial-force mismatch | `0.01384` |
| minimum `d ln(l)/d ln(R)` | `0.3865` |

The rotation match improves substantially relative to the Keplerian composite,
but the dominant pressure and density discontinuities do not improve. The
result is also not mesh supported.

## Verdict

The pressure-supported bookkeeping is implemented consistently and useful as
a diagnostic. The staggered fixed-point architecture is rejected as the
production route.

The next solver should include `Sigma`, `T`, and `Omega` in one nonlinear
residual containing mass, angular momentum, radial momentum, and total energy.
It must expose radial-force, Rayleigh-stability, primitive-continuity, and
conserved-flux gates independently.

## Reproduction

```text
PYTHONPATH=src python3 scripts/run_pressure_supported_interface_pilot.py
PYTHONPATH=src python3 scripts/build_pressure_supported_interface_pilot_canonical.py
```
