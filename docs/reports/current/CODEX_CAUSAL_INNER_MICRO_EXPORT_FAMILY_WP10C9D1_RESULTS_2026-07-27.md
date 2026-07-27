# WP10c9d1 — Characteristic-family attribution of failed micro exports

## Verdict

WP10c9d1 decomposes the failed WP10c9d0 embedded-patch physical export into
the exact five characteristic families:

```text
inward acoustic
inward shear
material
outward shear
outward acoustic
```

The family projectors and nonlinear physical export reconstruction pass, but
no one family passes the complete significance, dominance, persistence, and
cross-pair stability gates:

```text
conservative_export_defect_is_multifamily_full_coupled_operator_required
```

The material family is important. It supplies about `49-55%` of the complete
fine-pair export-error activity and `56%` of the fine cumulative
cooling/height error. It does not dominate every binding object:

- fine instantaneous complete-export fraction: `0.4925 < 0.50`;
- fine instantaneous persistence: `0.238 < 0.50`;
- complete instantaneous cross-pair activity cosine: `0.6817 < 0.90`;
- coarse cumulative complete-export persistence: `0.35 < 0.50`;
- boundary/net-drive activity is shared substantially with both acoustic
  families.

Thus the previous inward-shear diagnosis cannot be replaced by a scalar
material/contact repair. The conservative failure is a coupled
transport/thermal/height response.

No one-family operator, one-block operator, production change, truth
trajectory, constrained average, or reduced evolution is authorized.

## Frozen scope

The package uses only:

- the WP10c8z N128-exterior N128/N256/N512-equivalent patch histories;
- the WP10c9d0 physical M/J/E, cooling, and responsive-height export map;
- the exact five-family primitive projector implementation already certified
  in WP10c9c0c.

It evaluates 21 exact common times from `0` to `0.125 s`. It does not:

- propagate a new state or rate history;
- change the production flux, source, descriptor, boundary, reconstruction,
  patch layout, or timestep method;
- fit a coefficient or rotate the family basis to improve dominance;
- time-shift one mesh relative to another.

## Method contracts

The projector and reconstruction contracts pass on every patch level:

| Contract | Maximum |
|---|---:|
| Projector identity defect | `1.78e-15` |
| Projector idempotence defect | `3.55e-15` |
| Cross-projector defect | `3.55e-15` |
| Eigenpair defect | `2.65e-16` |
| State reconstruction defect | `1.09e-15` |
| Basis condition number | `22.032` |

Each family component is mapped independently through the same nonlinear
directional observable map as WP10c9d0. Their sum reproduces the directly
evaluated total response:

| Patch level | Instantaneous closure | Cumulative closure |
|---|---:|---:|
| N128-equivalent | `1.52e-5` | `2.20e-6` |
| N256-equivalent | `7.98e-6` | `2.74e-6` |
| N512-equivalent | `7.93e-6` | `2.07e-6` |

All are far below the declared `2e-3` gate.

Exactly inactive coupling-face components are removed by the same
absolute-significance filter used in WP10c9d0 before relative attribution.
This prevents individually large but mutually cancelling family values from
being divided by a zero total response.

## Complete physical export attribution

### Instantaneous export

| Pair | Leading family | Activity fraction | Persistence |
|---|---|---:|---:|
| N128/N256-patch | Material | `0.52382` | `0.71429` |
| N256/N512-patch | Material | `0.49248` | `0.23810` |

The leading-family activity profiles have cosine:

```text
0.68172 < 0.90.
```

The controlling observable also changes:

```text
coarse pair  vertical-work Killing energy
fine pair    cooling angular momentum
```

The fine controlling cooling-angular-momentum error is divided as:

| Family | Absolute activity fraction |
|---|---:|
| Material | `0.59614` |
| Inward acoustic | `0.14386` |
| Outward shear | `0.10994` |
| Inward shear | `0.10638` |
| Outward acoustic | `0.04367` |

One component can be material-led while the complete history still fails the
material-family dominance and persistence gates.

### Cumulative export

| Pair | Leading family | Activity fraction | Persistence |
|---|---|---:|---:|
| N128/N256-patch | Material | `0.51654` | `0.35` |
| N256/N512-patch | Material | `0.54969` | `0.80` |

The activity-profile cosine is `0.94283`, but the coarse-pair persistence is
below the fixed `0.50` gate. The controlling observable is cooling angular
momentum on both pairs, with the signed refinement error changing from
`-0.04406` to `+0.08952`.

At the fine endpoint, the controlling component is split:

| Family | Signed normalized contribution | Absolute fraction |
|---|---:|---:|
| Material | `+0.04369` | `0.48804` |
| Outward shear | `+0.01868` | `0.20868` |
| Inward shear | `+0.01443` | `0.16123` |
| Outward acoustic | `+0.00671` | `0.07496` |
| Inward acoustic | `+0.00600` | `0.06708` |

The material family is the largest contribution but does not reach half of
the controlling component's absolute family activity.

## Conservative boundary and net-drive attribution

The boundary-flux and net-drive errors are nearly identical because the
coupling-face perturbation is insignificant on this time interval.

For cumulative net drive:

| Pair | Material fraction | Inward acoustic | Outward acoustic | Profile cosine |
|---|---:|---:|---:|---:|
| N128/N256-patch | `0.48672` | `0.28803` | `0.15044` | |
| N256/N512-patch | `0.54465` | `0.16244` | `0.17285` | `0.93815` |

The coarse pair has no dominant family and only `0.30` material persistence.
For instantaneous net drive, the cross-pair activity cosine is only
`0.61185`; the material fractions are `0.44092/0.49387`.

Therefore the conservative response that couples horizon drainage, stored
mass/angular momentum/energy, and local sources is explicitly multifamily.

## Cooling and responsive-height qualification

The cumulative cooling/height subvector alone is material-dominated:

```text
material activity fraction   0.61731 / 0.56064
persistence                  1.0 / 1.0
profile cosine               0.95204
```

That does not select a material-only production repair. The instantaneous
cooling/height fine-pair material fraction falls to `0.49105`, its
persistence falls to `0.3333`, and the profile cosine is `0.70167`.
Moreover, any production candidate must also repair the multifamily inner
flux and net drive while preserving the responsive-height descriptor and
shared conservative ledger.

## Scientific decision

The WP10c9 sequence now supplies three complementary facts:

1. the original pure-family packet exposed inward-shear selected-branch
   damping sensitivity;
2. the complete common-mode block ledger showed no controlling single
   residual block;
3. the physical conservative-export ledger shows no controlling single
   characteristic family.

The correct conclusion is not that characteristic analysis was irrelevant.
It revealed the coupled transfers that make a one-family repair unsafe.

The binding decision is:

```text
redesign target = complete five-field near-horizon spatial residual
```

The redesign must preserve:

- one shared conservative mass/angular-momentum/Killing-energy face flux;
- exact opposite-sign neighboring-cell coupling;
- the implemented derivative-source sign
  `B = F_p - C_pr`;
- responsive-height temporal storage and work;
- lower-order geometry, cooling, and stress relaxation;
- zero incoming excision characteristics;
- exact stationary and evolving ledgers.

## Next package

The next package should be a production-neutral design and local method gate,
not another physical trajectory.

It should:

1. specify a complete well-balanced five-field fluctuation residual in one
   sign convention;
2. distinguish conservative shared fluxes from nonconservative path
   fluctuations without double counting;
3. include within-cell as well as interface fluctuations for reconstructed
   states;
4. preserve the declared background family and responsive-height descriptor;
5. pass constant-state, equilibrium-path, small-jump, Fourier-symbol, and
   variable-coefficient manufactured tests;
6. expose every conservative and nonconservative contribution in a local
   ledger;
7. leave production defaults unchanged.

Only after those local gates pass may the candidate be differentiated into a
frozen generator and tested against the WP10c9d0/d1 physical export ladder.

Fixed-`Q`, nonlinear truth, tide, wind, hot-state, loading-time, S-curve, and
QPE-cycle work remain closed.

## Verification

```text
Focused WP10c9d0/d1 tests   6 passed
Diff whitespace check       passed
```

The full repository suite is deferred until the next local design package is
either completed or reaches its declared stop.

## Reproduction

```text
PYTHONPATH=src:scripts \
python scripts/run_causal_inner_micro_export_family_audit_wp10c9d1.py

PYTHONPATH=src:scripts \
python -m pytest -q \
  tests/test_causal_inner_micro_export_preflight_wp10c9d0.py \
  tests/test_causal_inner_micro_export_family_audit_wp10c9d1.py
```

Machine evidence:

- `outputs/tables/causal_inner_micro_export_family_audit_wp10c9d1.json`
- `outputs/tables/causal_inner_micro_export_family_audit_wp10c9d1_arrays.npz`
- `outputs/checkpoints/causal_inner_micro_export_family_wp10c9d1/`
