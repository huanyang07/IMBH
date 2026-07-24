# WP10c8r interface-state dimension and sufficiency audit

Date: 2026-07-24

Base commit: `e180139b9ba32e2849506bb09ff924e6d762b54e`

Production physics changed: no

Production exact-max Rusanov flux changed: no

Production descriptor or BDF integrator changed: no

Moment ladder changed: no

New truth evolution run: no

## Decision

WP10c8r applies the missing absolute-significance gate to the WP10c8q
interface-4 rank calculation and audits the complete 34-coordinate slow-rate
operator on the same certified coordinate fiber.

The binding result is:

> `wp10c8r_two_coordinate_interface4_state_not_authorized`

The original WP10c8p/WP10c8q healing direction remains a significant,
persistent, approximately rank-one interface-4 ambiguity. However, none of
the six independent WP10c8q slow-rate fiber cases has a scientifically
significant interface-4 transport response. Their unit-normalized
interface-4 vectors must not be used to infer a second physical face
coordinate.

The complete slow-rate fiber does contain several large independent
directions, but they are not localized as significant macro-interface-4
transport. Therefore the planned two-coordinate augmented fiber is blocked
before coordinates or dynamics are added.

## Correction to the WP10c8q rank interpretation

WP10c8q constructed each interface-4 transport vector and divided it by its
own Euclidean norm before the SVD. That reproduces

```text
sigma_2 / sigma_1 = 0.800670
sigma_3 / sigma_1 = 8.78502e-5
```

with the near-null plane normal

```text
(-0.706703, -0.001342, 0.707509).
```

The small third singular value primarily expresses the close covariance of
the normalized mass and Killing-energy components. More importantly, the
rank calculation did not first require an absolute transport response above
a scientific gate.

The six actual interface-4 maximum component half-spreads are:

| Case | Maximum interface-4 half-spread, gate units |
|---|---:|
| N64 primary, `5e-4` | `1.20402e-9` |
| N64 primary, `1e-3` | `1.34656e-9` |
| N64 primary, `2e-3` | `2.50308e-9` |
| N64 held-out direction | `2.64788e-11` |
| N64 `t=0.10 s` anchor | `2.71808e-9` |
| N128 prolonged primary | `1.17685e-8` |

The predeclared promotion threshold is `0.10` gate units. All six cases are
at least seven orders of magnitude below it. Their largest response across
all four macro-interfaces is only `9.64769e-5` gate units.

The earlier WP10c8q decision combined:

1. significant interface-4 localization of the original WP10c8o healing
   direction; and
2. a rank calculation from different slow-rate directions whose
   interface-4 output was negligible.

Those two facts cannot be combined into a rank-two interface-4 state claim.

## Significance-filtered interface result

The original N64/N128 healing histories supply 49 significant saved
interface-4 vectors, but all belong to one physical fiber family. After
filtering on the `0.10` absolute gate, their normalized singular-value ratios
are

```text
(1.0, 1.27458e-5, 7.86230e-10).
```

Thus the currently supported interface-4 statement is:

> one significant, persistent, approximately rank-one perfect-fluid
> transport ambiguity has been demonstrated.

No independent significant second interface-4 family has been demonstrated.
This does not prove that one interface coordinate is globally sufficient; it
only removes the evidence used to authorize two.

## Complete slow-rate fiber spectrum

WP10c8r next evaluates the complete loading-time-normalized operator

```text
A_q = t_load D(C_34 f) N_34
```

at the N64 `0.025 s`, N64 `0.10 s`, and N128 `0.025 s` anchors. Here `N_34`
is the weighted null basis of the exact 34-coordinate tangent constraints.

### Singular-value ratios

| Anchor | `>=0.5` | `>=0.1` | `>=0.01` | `>=0.001` |
|---|---:|---:|---:|---:|
| N64 `0.025 s` | 2 | 4 | 8 | 14 |
| N64 `0.10 s` | 2 | 5 | 9 | 13 |
| N128 `0.025 s` | 2 | 4 | 7 | 14 |

The first N64 ratios are approximately

```text
1.0000
0.5816
0.3245
0.1531
0.0963
0.0894
0.0287
0.0128
```

The matched N128 ratios are

```text
1.0000
0.5852
0.3148
0.1529
0.0977
0.0893
0.0295
0.0097
```

This is strong mesh agreement for a multi-directional complete-rate
response. It is not a two-dimensional full-state result.

### Amplitude-box screen

Each of the first eight tangent directions is scaled to `1e-3` of its frozen
pointwise admissible amplitude box, matching the decisive WP10c8q nonlinear
seed scale.

At N64 `0.025 s`, the maximum predicted slow-rate responses range from
`8.38` to `430.42` gate units. Their controlling coordinates include:

- shell-0 stress storage;
- shell-0 mean log temperature;
- shell-0 radial momentum;
- shell-0 angular momentum and Killing energy;
- shell-1 stress storage;
- shell-2 stress storage.

The corresponding largest macro-interface responses are all below `0.10`
gate units. They are controlled by interfaces 1, 2, or 3, not interface 4.
The same classification is obtained at N64 `0.10 s` and N128 `0.025 s`.

The tangent spectrum alone is not a nonlinear closure proof. It does,
however, reject the premise that the complete 34-rate ambiguity is already
represented by a significant two-dimensional interface-4 transport plane.

## What remains valid from WP10c8q

WP10c8r does not change the following certified results:

- the five-shell incidence audit rejects a divergence-null flux gauge for the
  original decisive pair;
- the original pair produces real conservative shell redistribution;
- its perfect-fluid trace attribution closes below `4.80e-14`;
- the left reconstructed radial-velocity trace is the largest primitive
  contribution for that pair;
- exact-history N64 evolution passes through `0.125 s`;
- the original interface-4 response accumulates only `5.28e-5` e-folds;
- raw 34-coordinate instantaneous slow-rate closure remains rejected.

The correction concerns only the inference from the later independent
slow-rate cases to a second localized interface-4 coordinate.

## Why the augmented 36-coordinate search was not run

The locked WP10c8r plan required two independent, scientifically significant
interface-4 response families before choosing two physical coordinates.
That gate fails:

```text
significant independent slow-rate interface-4 families = 0
required                                            >= 2
```

Adding two trace coordinates and performing a nonlinear augmented-fiber
search would therefore fit directions selected after unit-normalizing
negligible outputs. No physical second coordinate can be justified from that
evidence.

Stopping before the augmented search is the binding success condition of
this gated package, not an incomplete run.

## Locked next plan: WP10c8s

### 1. Select complete-rate directions

Use the amplitude-admissible complete slow-rate operator to select:

- the controlling per-output N64 direction;
- an independent held-out direction;
- the leading stress, thermal, and radial-momentum singular directions;
- matched N128 confirmations.

Do not select directions from unit-normalized negligible interface outputs.

### 2. Construct exact nonlinear equal-`q_34` pairs

For each selected direction:

- perform the exact nonlinear coordinate correction;
- retain the current amplitude, state, DAE, descriptor, and storage gates;
- require a complete slow-rate half-spread above `0.25`;
- record every macro-interface flux and the full cellwise primitive-rate
  difference.

### 3. Localize the rate ambiguity

Decompose each significant rate response by:

- shell and retained-coordinate family;
- fine-grid cell and physical field;
- fine-face flux divergence;
- physical source;
- responsive-height storage derivative;
- primitive-map/moving-coordinate term.

The audit must distinguish a macro-interface transport state from unresolved
sub-shell storage or gradient structure.

### 4. Measure natural healing

For each independent significant nonlinear pair, run synchronized
history-clean N64 `h/h/2` trajectories to

```text
0.025, 0.05, 0.10, 0.125 s
```

with N128 confirmation only for architecture-controlling cases.

### 5. Architecture decision

- If the additional complete-rate modes heal while the original
  interface-4 direction persists, authorize one physical interface-4
  coordinate candidate followed by a new worst-case augmented-fiber search.
- If another persistent mode is localized, add only its measured physical
  coordinate.
- If several persistent modes occupy several shells or interfaces, stop the
  moment-ODE route and implement a conservative staggered coarse
  finite-volume/PDE model.
- Do not fit a relaxation law without two mesh-supported e-foldings.

No macrostep, tide, wind, hot-state, stability, or cycle search is
authorized.

## Verification and evidence

- Focused tests: `18 passed`.
- Causal reduction test matrix: `63 passed`.
- Full repository: `709 passed, 4 subtests passed`.
- New truth evolution: none.
- Production code changed: none.

Primary artifacts:

- `outputs/tables/causal_interface_state_sufficiency_wp10c8r.json`
- `outputs/tables/causal_interface_state_sufficiency_wp10c8r_arrays.npz`

Final hashes:

- JSON:
  `517948e5dc35e3b53dac42a0d8fe038c58b6e52208e5fb310fd110115c060246`
- arrays:
  `b6ab43e5825327f7e063c30db7196f655b481d664523571aa43b622c23fce944`
