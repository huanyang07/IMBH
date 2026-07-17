# Causal relativistic alpha-shear WP10c3a results

**Date:** 2026-07-17
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Scope:** local relativistic `R-phi` stress, paired angular/Killing-energy
fluxes, finite-speed shear relaxation, and characteristic audit. No cooling,
dynamic vertical work, stream, tide, wind, stationary disk root, or timestep
was run.

## Verdict

WP10c3a passes its bounded local stress gate:

```text
gas/radiation states audited                    9
maximum tensor/work identity defect             8.6654e-16
maximum shear-principal eigenvalue defect       2.7756e-17
maximum light-cone excess                       0
inside-horizon incoming modes                   0
maximum torque/Omega-work defect                4.4409e-16
weak-field common-torque defect at 10000 rg     4.9996e-5
rejected-control minimum |Im lambda|             6.6566e-5
```

The selected stress tensor is covariant, its angular and energy fluxes come
from one tensor, and its transverse modes propagate at finite real speed.
This is not yet a production disk solution.

## Why instantaneous alpha stress was not retained

The first prototype made the rest-frame stress an independent field, inserted
it into the covariant stress-energy tensor, advected `D chi`, and relaxed
`chi` locally toward

```text
chi_alpha = alpha Pi/(Sigma c^2).
```

Its cold weak-field flux Jacobian contains a complex-conjugate pair. Across
finite-difference steps from `1e-3` to `5e-5`, the maximum imaginary part
remains

```text
6.65662e-5 <= max |Im lambda| <= 6.65663e-5.
```

This is a closure defect, not differentiation noise. A local pressure target
changes lower-order relaxation terms but supplies no shear-gradient principal
coupling. The pressure-amplitude-only architecture is therefore rejected for
causal evolution.

## Selected shear law

The accepted prototype follows the finite-relaxation structure used in
causal relativistic accretion models. In the local fluid rest frame,

```text
t^munu = S (e_R^mu e_phi^nu + e_phi^mu e_R^nu),
S      = Sigma chi.
```

The common alpha stress calibrates the equilibrium amplitude at one reference
positive shear rate:

```text
chi_alpha = alpha Pi/(Sigma c^2)
nu_s      = chi_alpha/q_ref.
```

The evolved stress obeys a Maxwell-Cattaneo law,

```text
tau_r u^mu nabla_mu chi + chi = nu_s q,
```

with

```text
c_nu^2/c^2 = nu_s/(tau_r h).
```

The bounded gate selects

```text
c_nu = sqrt(alpha) a
```

and derives `tau_r` from the identity above. This signal-speed calibration is
provisional. It is causal and recovers the project common stress at
`q=q_ref`, but it has not been calibrated against MRI turbulence or a
multidimensional simulation.

The causal-relaxation structure is consistent with the approach of
[Gammie & Popham (1997)](https://arxiv.org/abs/astro-ph/9705117) and the
finite propagation analysis of
[Kley & Papaloizou (1997)](https://arxiv.org/abs/astro-ph/9701072).

## Covariant torque and work

The stress contribution is transformed directly into the stationary Killing
chart:

```text
delta S_i  = alpha t^0_i
delta E_K  = -alpha t^0_t
delta F_i  = alpha t^R_i
delta F_EK = -alpha t^R_t.
```

The audit explicitly caught and removed an incorrect second subtraction of
the radial shift from the stress momentum flux. With the direct tensor flux,
a stationary circular flow satisfies

```text
P_stress = Omega G_stress
```

to `4.45e-16`. This is one identity, not two independently tunable viscous
source terms.

The covariant stress is trace-free and orthogonal to the four-velocity.
Across all nine states, the maximum normalized trace, orthogonality, or radial
work defect is `8.67e-16`.

## Characteristic gate

The local transverse momentum/stress principal matrix is

```text
[ 0              1/h       ]
[ h c_nu^2/c^2   0         ],
```

so its rest-frame modes are exactly `+/-c_nu`. The declared
frozen-coefficient radial principal model has five modes:

```text
two acoustic + one material/contact + two shear.
```

The shear modes use the covariant Valencia characteristic cone with `a`
replaced by `c_nu`. Nine states span:

```text
radii          20, 4.5, 1.8 rg
thermodynamics gas, transition, radiation dominated
```

Every spectrum is real and inside the local light cone. All three
inside-horizon thermodynamic states have zero incoming inner modes.

This audit does not yet prove the characteristic structure of a final
nonlinear Israel-Stewart system after dynamic height, radiation variables, or
additional viscous tensor components are introduced. Those additions must
repeat the full principal audit rather than inherit this result by assumption.

## Weak-field recovery

For stationary circular states, the relativistic torque approaches the
repository common-stress torque:

```text
G -> 2 pi R^2 alpha Pi.
```

The relative defects are:

| Radius | Relative defect |
|---:|---:|
| `100 rg` | `4.96896e-3` |
| `1000 rg` | `4.99631e-4` |
| `10000 rg` | `4.99963e-5` |

This is first-order weak-field recovery in `rg/R`, as expected.

## Classification

```text
numerical status:
    supported but not fully certified for the local stress/flux contract

physical status:
    diagnostic only

production status:
    blocked
```

WP10c3a does not include:

1. radiative cooling or radiation transport;
2. dynamic column height or relativistic vertical work;
3. stream mass/angular/energy moments;
4. the Hill/Roche boundary contract;
5. a final nonlinear coupled characteristic proof;
6. stationary roots or implicit evolution.

## Locked next step

Proceed to WP10c3b only:

1. define one dynamic column-height/vertical-work contract;
2. add radiative cooling to the Killing-energy ledger;
3. prove that stress work is not added again as local viscous heating;
4. test flat, stationary, and source-free energy identities;
5. retain the same causal stress and characteristic gate;
6. keep stream, tide, wind, full-domain mapping, and long evolution disabled.

## Verification

```text
focused causal-stress tests    8 passed
complete repository suite      427 passed, 4 subtests passed
```

Machine-readable evidence:

```text
outputs/tables/causal_inner_stress_wp10c3.json
```

Reproduction:

```bash
PYTHONPATH=src python3 scripts/run_causal_inner_stress_wp10c3.py
```
