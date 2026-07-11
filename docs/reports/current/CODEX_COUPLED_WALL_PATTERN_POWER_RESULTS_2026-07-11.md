# Coupled Wall Pattern-Power Results

## Scope

This work package tests the first paired angular-momentum and energy contract
for the fully coupled no-wind minidisk. It changes no stream, stress,
thermodynamic, wind, or inner transonic closure.

During the audit, the common-stress builder was found to inherit the no-wind
transonic benchmark's `Rout=10000 rg`, despite the two-domain reports describing
a `335 rg` finite minidisk. The common-stress and coupled chains were therefore
rebuilt on the intended `Rout=335 rg` domain before any tidal interpretation.
The old long-buffer roots remain useful numerical controls but are not Hill
truncation evidence.

## Corrected Coupled Baseline

The corrected `40.0415 rg` coupled root passes the chained mesh sequence

```text
Ninner/Nouter = 96/64 -> 144/96 -> 192/128.
```

At the finest mesh:

```text
maximum residual                    1.45e-9
Jacobian rank                       772 / 772
composite luminosity                1.34823 L_Edd
maximum outer H/R                   0.31040
N144/96 to N192/128 luminosity shift 1.23e-5
N144/96 to N192/128 H/R shift        1.78e-4
```

The corrected interface continuation at `34.97-50.05 rg` also passes, with
composite-luminosity spread `5.22e-5` and fixed `R>=60 rg` thickness spread
`0.284%`.

## Tidal Torque and Power Contract

The ideal wall requires an outward viscous torque `G_out`. The torque applied
to the disk is `-G_out`. The conservative energy flux initially carries
disk-rate work `-Omega_out G_out`. For binary pattern speed `Omega_p`, the
physical external work is `-Omega_p G_out`, and the differential work is
deposited as heat:

```text
P_heat = f (Omega_out - Omega_p) G_out.
```

The heat is distributed over an exact, normalized Hill-band kernel that begins
at `0.35 R_H`. The fiducial geometry gives

```text
R_H                         746.9008 secondary rg
configured 0.5 R_H          373.4504 secondary rg
finite model edge           335 secondary rg
Omega_out/Omega_p           5.687 at f=0
G_out/(Mdot_stream l_stream) 0.759923.
```

The discrete identity

```text
-Omega_out G_out + P_heat
= -[(1-f)Omega_out + f Omega_p] G_out
```

closes to at most `1.7e-16` relative in the accepted stages.

## Result

| Pattern-power fraction | Numerical status | Lrad/LEdd | max tidal-band H/R | max outer H/R |
|---:|---|---:|---:|---:|
| 0.00 | accepted | 1.3482 | 0.1695 | 0.3104 |
| 0.25 | accepted | 1.3835 | 0.4316 | 0.4316 |
| 0.50 | accepted | 1.4086 | 0.5604 | 0.5604 |
| 0.75 | accepted | 1.4221 | 0.6090 | 0.6090 |
| 1.00 | rejected | 1.4320 diagnostic | 0.6213 | 0.6213 |

The physical validity failure occurs at the first nonzero stage, not at the
failed full-power Newton solve. Once `H/R>0.3` in the tidal band, the one-zone
closed wall cannot be treated as a calibrated truncation model.

This result is consistent with the qualitative regime separation reported by
[Martin and Lubow (2011)](https://arxiv.org/abs/1012.4102) and the inefficient
truncation found for thick circumplanetary disks by
[Martin et al. (2023)](https://arxiv.org/abs/2306.17532). Those papers motivate
the validity decision; they do not calibrate this minidisk's torque amplitude.

## Verdict

The paired torque/power implementation is numerically consistent. Perfect
tidal confinement is physically rejected for the heated solution under the
current one-zone closure.

The next steady model must:

1. keep the absolute stream supply fixed;
2. promote `Mdot_inner` to a signed eigenvalue;
3. permit outward mass and energy flux at `335 rg`;
4. continue a distributed tidal torque from the open limit;
5. stop when Hill-band thickness, optical-depth, or torque-capacity gates fail.

No smaller pattern-power steps or relaxed nonlinear tolerances should be used
to force the closed-wall branch. Wind remains deferred until the open tidal
problem and its stability are understood.
