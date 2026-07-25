# WP10c8t exact-history inner-mode healing extension

Date: 2026-07-24

Base commit:
`4a54eb547b5c9f1663ce2480367110d195c0b4bd`

Truth meshes: N64 and N128

Production physics changed: no

Production spatial operator changed: no

Production BDF formula changed: no

Moment ladder changed: no

Reduced evolution run: no

Relaxation law fitted: no

N128 nonlinear confirmation run: yes

## Executive result

WP10c8t extends the binding WP10c8s mode-0 equal-`q_34` pair from
`0.025 s` to `0.125 s` while preserving the exact increment-primary BDF2
history. It uses matched N64 timesteps

```text
coarse h = 1.25e-3 s
fine   h = 6.25e-4 s
```

and exact common outputs at

```text
0
0.0025
0.005
0.010
0.025
0.050
0.075
0.100
0.125 s.
```

The N64 classification is:

> `n64_persistent_localized_inner_mode_through_0p125s`

All trajectory, state, fresh-rate, restart, flux-reconstruction, discrete-
ledger, and physical-ledger contracts pass. The controlling complete
slow-rate spread decays substantially but remains decisively above the
healing gate:

```text
initial uncertainty-inclusive maximum = 178.60984326
final uncertainty-inclusive maximum   =   3.11581464
final uncertainty-exclusive lower     =   3.05678101
healing gate                           =   0.10
```

Thus even the conservative final lower bound is about `30.6` times the
healing gate.

The exact nonlinear N128 confirmation also remains persistent and localized
through `0.125 s`, with every trajectory and diagnostic contract passing.
However, its late complete-rate vector does not satisfy the predeclared
N64/N128 direction-and-amplitude gate. The binding combined classification
is therefore:

> `n128_architecture_confirmation_inconclusive`

This is not a numerical failure and it is not evidence for healing. At N128,
the final uncertainty-exclusive rate lower bound is `0.68557 > 0.25`, the
endpoint temporal uncertainty is `0.03529 < 0.10`, and the final state/rate
shell-0 fractions are `0.90808/0.90020`. But the final N64/N128 rate-vector
absolute cosine is only `0.76088 < 0.90`, and the N128/N64 maximum-amplitude
ratio is `0.23357`, giving a ratio defect `0.76643 > 0.50`.

The result therefore does **not** establish permanent memory, healing, or a
mesh-supported reduced state. It establishes that the tested mode remains
large and localized on both meshes while its late rate phase/amplitude is not
spatially certified.

## Exact continuation history

The committed WP10c8s coarse plus/minus trajectories were replayed to
`0.025 s`. Their states reproduce the committed endpoints bitwise. The
following objects were persisted and checksum-validated:

- exact terminal state;
- previous increment-primary physical increment;
- previous timestep;
- previous responsive-height vertical Killing-storage increment;
- complete `CausalFiveFieldBDFHistory`.

The coarse restart artifacts at `0.025 s` are:

| Side | Restart SHA256 | State SHA256 |
|---|---|---|
| minus | `a71f85a56f8a74e11f452e3722b47842a2d05cf225832ca7af792264dcca66c3` | `f9a32e48d42b994189dfe9fe4ce64b66be860ecc2780a574e79255ec6194d33d` |
| plus | `b9565be204dbec7c8ee3c0802be4e44d4149e69d974d166930eed8fae04b32fb` | `b23e82cf0483754f3226ba1083307d05b582fc8d78cf1ec9fcd256641c33a85f` |

Every segment boundary at `0.05`, `0.075`, `0.10`, and `0.125 s` is
serialized and reloaded before continuation. No additional BDF1 startup is
introduced at a segment boundary. The newly refined `h/2` paths use one BDF1
step only at their original `t=0` startup and BDF2 thereafter.

This closes the continuation-history defect in the earlier cache format:
future extensions can restart from exact multistep history rather than
reconstructing an increment by subtracting two large stored states.

## Numerical contracts

All four N64 trajectories pass.

| Trajectory | Steps | BDF1/BDF2 | Function evaluations | Jacobians | Maximum cumulative physical-ledger defect | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| coarse minus | `100` | `1/99` | `23694` | `502` | `5.09723e-5` | `13786 s` |
| coarse plus | `100` | `1/99` | `23835` | `505` | `5.10069e-5` | `13779 s` |
| fine minus | `200` | `1/199` | `45368` | `961` | `1.27381e-5` | `26164 s` |
| fine plus | `200` | `1/199` | `45368` | `961` | `1.27475e-5` | `26019 s` |

Across the trajectory diagnostics:

```text
all output state gates pass                         yes
all fresh coordinate-rate audits pass              yes
maximum individual physical M/J/E shell defect     6.24059e-4
maximum flux-reconstruction defect                  4.19016e-15
maximum rate-integral/slip reconciliation defect    2.18010e-7
maximum discrete BDF ledger defect                  2.92982e-11
```

The physical shell-ledger requirement is `1e-3`, so the controlling
individual defect remains within the declared contract. The matched
coarse/fine physical defects decrease under timestep refinement.

## Repository verification

The completed package passes:

```text
focused WP10c8t/WP10c8s/BDF regression matrix   41 passed
full repository test suite                      726 passed
full-suite subtests                              4 passed
repository hygiene                              738 tracked files passed
Python compilation of both WP10c8t runners       passed
git whitespace validation                        passed
```

The persisted N64 JSON records the SHA256 of the exact runner used to create
the machine evidence. That digest still matches the current runner:

```text
aafa3b978459dd2ee12587f567533b4bfc42bd13a2e9752ba1f66d075b6e4ee1
```

## Complete slow-rate decay

The controlling uncertainty-inclusive output at every saved time is:

| Time (s) | Controlling output | Fine spread | Coarse spread | Temporal uncertainty | Inclusive upper | Exclusive lower |
|---:|---|---:|---:|---:|---:|---:|
| `0` | shell-0 stress storage | `178.60984` | `178.60984` | `0` | `178.60984` | `178.60984` |
| `0.0025` | shell-0 stress storage | `98.67560` | `101.62488` | `2.94928` | `101.62488` | `95.72633` |
| `0.005` | shell-0 stress storage | `43.44284` | `45.70312` | `2.26028` | `45.70312` | `41.18256` |
| `0.010` | shell-0 radial momentum | `16.35451` | `15.88591` | `0.46860` | `16.82311` | `15.88591` |
| `0.025` | shell-0 stress storage | `17.59704` | `17.91094` | `0.31391` | `17.91094` | `17.28313` |
| `0.050` | shell-0 stress storage | `9.87924` | `9.92872` | `0.04948` | `9.92872` | `9.82977` |
| `0.075` | shell-0 stress storage | `3.13745` | `3.05805` | `0.07940` | `3.21685` | `3.05805` |
| `0.100` | shell-0 stress storage | `4.73718` | `4.82610` | `0.08892` | `4.82610` | `4.64826` |
| `0.125` | shell-0 stress storage | `3.08630` | `3.05678` | `0.02952` | `3.11581` | `3.05678` |

Six outputs are significant at the initial point. Their fine-grid histories
show that the ambiguity remains entirely within the retained shell-0 rate
family:

| Slow-rate output | Initial | `0.050 s` | `0.075 s` | `0.100 s` | `0.125 s` |
|---|---:|---:|---:|---:|---:|
| rest mass | `90.3690` | `4.10988` | `1.44391` | `0.72926` | `0.33373` |
| angular momentum | `51.0896` | `3.89673` | `0.58331` | `0.62611` | `0.60062` |
| Killing energy | `105.5039` | `4.73672` | `1.73792` | `0.83425` | `0.34262` |
| mean log temperature | `16.8911` | `3.69366` | `1.00833` | `0.47346` | `0.70857` |
| radial momentum | `131.4897` | `3.71603` | `1.96383` | `0.70367` | `0.58224` |
| stress storage | `178.6098` | `9.87924` | `3.13745` | `4.73718` | `3.08630` |

The envelope falls by about `4.05` apparent e-folds. However:

- maximum absolute `h/h/2` uncertainty is `2.94928`, above the `0.025`
  strict curve gate;
- maximum relative uncertainty above the significance floor is `0.08049`,
  within the `0.10` relative gate;
- the controlling stress-storage response regrows between `0.075` and
  `0.100 s`;
- the final endpoint uncertainty is `0.02952`, slightly above the strict
  `0.025` curve gate.

Therefore:

```text
binary persistence through 0.125 s             certified at N64
precise decay curve or relaxation time          not certified
single exponential relaxation law               not authorized
```

## Accumulated initial slip

The maximum uncertainty-inclusive accumulated coordinate slip at `0.125 s`
is

```text
1.90088137e-7
```

against the declared `0.10` reserve. It is controlled by shell-0 stress
storage. Direct state change and the slow-rate time integral reconcile to

```text
2.18010290e-7.
```

This is scientifically important: the mode produces a large instantaneous
ambiguity in the loading-time-normalized slow vector field, but only a tiny
coordinate displacement over the tested `0.125 s` fast-time interval.

It does **not** permit the mode to be discarded yet. The rate ambiguity
remains more than thirty times above the healing gate and shows late
regrowth. Extrapolating the small short-window impulse to a loading time
would be unjustified.

## Localization

At `0.125 s`, both timestep resolutions identify shell 0:

| Quantity | Coarse shell-0 L1 fraction | Fine shell-0 L1 fraction | Controlling radius | Controlling field |
|---|---:|---:|---:|---|
| state half-difference | `0.88234` | `0.88185` | `1.875 rg` | causal stress |
| primitive-rate half-difference | `0.94512` | `0.94523` | `2.035 rg` | causal stress |

Every saved state and rate output remains localized in one shell. The
evidence therefore supports a localized inner-disk state or embedded inner
patch as the next architecture candidate. It does not support adding another
interface-4 transport coordinate.

## Exact nonlinear N128 confirmation

The N128 mode-0 pair was constructed directly from the matched N128
complete-rate tangent direction and then corrected to exact finite-amplitude
equal-`q_34` form. It was not obtained by interpolating the N64 pair. The
binding pair checks give:

```text
maximum pairwise q_34 defect                 6.66134e-16
maximum loading-time slow-rate half-spread   187.50124
initial N64/N128 signed rate cosine           0.999818
initial N128/N64 maximum-amplitude ratio      1.049781
all state, fresh-rate, and DAE-storage gates  passed
```

Only this architecture-controlling N128 case was run.

### N128 trajectory contracts

All four N128 trajectories pass.

| Trajectory | Steps | BDF1/BDF2 | Function evaluations | Jacobians | Maximum cumulative physical-ledger defect | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| coarse minus | `100` | `1/99` | `22614` | `479` | `5.10759e-5` | `25922 s` |
| coarse plus | `100` | `1/99` | `22472` | `476` | `5.11384e-5` | `25774 s` |
| fine minus | `200` | `1/199` | `39492` | `836` | `1.27629e-5` | `45669 s` |
| fine plus | `200` | `1/199` | `39680` | `840` | `1.27811e-5` | `46084 s` |

Across the N128 endpoint diagnostics:

```text
all output state gates pass                         yes
all fresh coordinate-rate audits pass              yes
maximum individual physical M/J/E shell defect     6.28736e-4
maximum flux-reconstruction defect                  1.65420e-15
maximum discrete BDF ledger defect                  8.35655e-11
```

The matched temporal endpoint is well resolved:

```text
initial uncertainty-inclusive maximum    187.50123608
final uncertainty-inclusive maximum        0.75615022
final uncertainty-exclusive lower          0.68556706
final temporal uncertainty                  0.03529158
temporal uncertainty gate                   0.10
persistence lower-bound gate                0.25
```

N128 therefore independently rejects healing through `0.125 s`. Its final
state and primitive-rate differences remain in shell 0 with L1 fractions
`0.90808` and `0.90020`. The controlling late primitive-rate support is near
`5.53 rg` and is still a causal-stress response.

### Binding cross-mesh stop

The initial N64/N128 slow-rate directions agree:

```text
signed cosine                  0.99981832
N128/N64 maximum ratio         1.04978109
amplitude-ratio defect         0.04978109
```

The final directions do not:

```text
signed cosine                 -0.76088395
absolute cosine                0.76088395   < 0.90
N64 maximum rate spread        3.08629783
N128 maximum rate spread       0.72085864
N128/N64 maximum ratio         0.23356743
amplitude-ratio defect         0.76643257   > 0.50
```

The discrepancy is dominated by shell-0 stress storage:

| Slow-rate component at `0.125 s` | N64 | N128 |
|---|---:|---:|
| shell-0 stress storage | `-3.08630` | `+0.72086` |
| shell-0 mean log temperature | `-0.70857` | `-0.04406` |
| shell-0 radial momentum | `-0.58224` | `-0.16539` |
| shell-0 angular momentum | `+0.60062` | `+0.10471` |

The signed accumulated 34-coordinate state-slip vectors remain much more
closely aligned (`cosine = 0.97510`), but the N128/N64 maximum-slip ratio is
only `0.47682`. Together with the late stress-rate sign reversal, this
supports a mesh-dependent late phase/amplitude interpretation. It does not
permit the final rate gate to be waived.

## Interpretation

WP10c8t resolves the immediate ambiguity left by WP10c8s:

1. mode 0 is not merely a `0.025 s` startup artifact;
2. it decays strongly but remains above the reduced-model rate tolerance
   through `0.125 s`;
3. its late response is not monotone and cannot be represented by a fitted
   scalar exponential;
4. its integrated short-window slip is very small;
5. its state and rate support remain confined to the innermost retained
   shell;
6. N128 independently remains persistent and localized;
7. the late N64/N128 rate phase and amplitude do not agree.

The correct bounded conclusion is:

> `q_34` plus only an interface-4 state remains insufficient, and the
> architecture-controlling missing information remains localized in the
> inner shell. But its late dynamics are not spatially certified, so no new
> coordinate or embedded inner patch may yet be selected.

## Locked next plan

### 1. Existing-cache N64/N128 phase-and-onset audit

Do not run another truth trajectory first. The exact N64/N128 coarse/fine
trajectory caches contain every fixed step through `0.125 s`.

At exact common times, including at least

```text
0, 0.025, 0.050, 0.075, 0.100, 0.125 s,
```

evaluate the complete signed 34-rate half-difference. Add denser samples only
around the first cross-mesh sign or direction change. Report:

- N64/N128 signed and absolute rate-vector cosine;
- maximum-amplitude ratio and its controlling coordinate;
- rate-vector norm and shell-0 stress-rate zero crossings;
- signed accumulated coordinate-slip direction and amplitude;
- matched coarse/fine temporal uncertainty at every architecture-controlling
  output.

No time shift may be used to make the binding same-time gate pass. A fitted
phase lag is diagnostic only.

### 2. Local inner-shell physical attribution

Conservatively restrict the N128 state/rate half-differences to the N64 inner
shell and decompose the first cross-mesh departure into:

- causal-stress profile shape;
- radial momentum or radial-velocity trace;
- temperature/entropy shape;
- mapped storage;
- responsive-height storage;
- perfect-fluid and stress flux divergence;
- geometry, cooling, relaxation, and source terms.

Determine whether the failed endpoint is:

- one common localized mode with mesh-dependent frequency/phase;
- one common mode with a nonconvergent damping/amplitude;
- or a different mixture of localized modes.

### 3. Architecture decision

- same localized mode with a controlled mesh-convergent frequency and
  amplitude: proceed to the balanced, absolute-significance-filtered
  modes-0-to-3 inner-state audit;
- unresolved inner phase at N128: prefer an embedded fine inner patch and
  define its own spatial-convergence gate before reduced evolution;
- several localized modes: retain a small inner-state vector or embedded
  patch rather than fitting one relaxation coordinate;
- modes spreading into several shells: move to the conservative staggered
  coarse finite-volume/PDE architecture.

Every proposed augmented state must undergo a fresh worst-case exact
finite-amplitude equal-coordinate slow-rate search.

### 4. Hard stops

No loading-time macrostep, reduced evolution, fitted relaxation law, tide,
wind, hot-state search, stability claim, or cycle search is authorized by
WP10c8t.

## Verification and evidence

- Focused WP10c8t/WP10c8s/BDF matrix: `41 passed`.
- Full repository suite: `726 passed, 4 subtests passed`.
- Repository hygiene: `738 tracked files passed`.
- Exact committed coarse endpoint replay: bitwise.
- Segment restart round trips: passed.
- New N64 trajectories: four.
- New N128 trajectories: four.
- N64 accepted trajectory steps: `600`.
- N128 accepted trajectory steps: `600`.
- Exact N128 pair and all N128 trajectory/diagnostic contracts: passed.
- Binding N64/N128 architecture confirmation: inconclusive.

Primary artifacts:

- `outputs/tables/causal_inner_mode_healing_wp10c8t.json`
- `outputs/tables/causal_inner_mode_healing_wp10c8t_arrays.npz`
- `outputs/tables/causal_inner_mode_n128_confirmation_wp10c8t.json`
- `outputs/tables/causal_inner_mode_n128_confirmation_wp10c8t_arrays.npz`

Artifact hashes:

- N64 JSON:
  `e24b9a3f56b1d238bc6bad491a5a207e16c5078b7e16df56b6fadcdf75f1a117`
- N64 arrays:
  `44d537b397215fac824b0d47f73f44bb0ac48175fdebbf65ee35a70215f6ac65`
- N128 JSON:
  `d2db7bf078d5ffff3545f6fb47f7e0fd27e9d65969173442fb4da76931027af2`
- N128 arrays:
  `2df744ddb66b33d70770c4053a51e6a9bd4ec9e35c37a4a1cbbaa62b01f63d2d`
- exact N128 pair JSON:
  `22674b3c72e19086aaabe9cdea13f2254fb75c1bdb1c38f74c8a3db8c6879788`
- exact N128 pair arrays:
  `6e8ddd634c933b38053d043cd22c2680e082d054579cfc3a9afaad7b0330e149`
- N128 confirmation runner:
  `364e80e42de3d48eaa4849e150cc536ad2857cd0920f051ca61b7bc4afd2173a`
