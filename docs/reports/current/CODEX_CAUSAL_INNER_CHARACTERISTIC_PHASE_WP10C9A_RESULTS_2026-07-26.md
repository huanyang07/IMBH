# WP10c9a — Characteristic-family phase and operator preflight

## Verdict

WP10c9a is a binding negative result for the current inner bulk operator and
a positive localization result:

```text
characteristic_rate_phase_unresolved_operator_redesign_required
```

The certified live coarse/fine coupling, production inner excision map,
responsive-height storage, and BDF2 history contracts remain intact. Five
continuum-matched acoustic/contact/shear packets were propagated through the
unchanged N128/N256/N512-equivalent hybrid generators. Four families pass the
packet phase and damping-order gates. The inward causal-shear packet fails the
damping-order gate:

```text
p_damping = -0.07156 < 0.75.
```

Its N256/N512 same-time rate direction is nevertheless well aligned:

```text
minimum signed state/rate cosine = 0.99861.
```

The failure is therefore a fine damping/rate-amplitude error, not the gross
phase reversal seen in the original mixed common mode.

The forcing-side term audit identifies scalar max-speed Rusanov dissipation as
the largest N256/N512 defect for every pure family. For the controlling inward
shear packet the response-side mapped-storage action is the largest balance
block, while the largest forcing blocks are:

| Block | Scaled N256/N512 L2 difference |
|---|---:|
| mapped storage action (balance) | `1.16024e-3` |
| Rusanov transport (forcing) | `8.31732e-4` |
| central perfect-fluid transport | `2.99206e-4` |
| stress relaxation source | `5.72550e-5` |
| evolving-storage remainder | `1.39214e-5` |
| perfect-fluid geometry | `1.56020e-6` |
| responsive-height storage | `1.02892e-6` |
| central stress transport | `1.20972e-7` |

The mapped-storage term is not interpreted as the cause: it is the descriptor
response that balances the nonconvergent rate. The largest forcing-side defect
is the Rusanov penalty.

No candidate is promoted. Rapidity reconstruction and characteristic
perturbation reconstruction both pass the isolated smooth method-order tests,
but their fine characteristic-rate error coefficients are indistinguishable
from production. No candidate has a live-coupled generator or passes the full
packet phase contract.

Consequently:

- the WP10c8z common nonlinear mode is not rerun;
- bounded nonlinear patch truth is not authorized;
- another brute-force refinement is not authorized;
- fixed-`Q` averaging and reduced coordinates remain blocked;
- the next target is a family-resolved characteristic dissipation operator,
  not another reconstruction chart.

## Benchmark correction

The first implementation attempt projected each localized packet onto the
unrelated `q_34` slow-coordinate null fiber. That rotated roughly half of the
intended characteristic direction into other families and was rejected before
binding evidence was published.

The final benchmark uses the exact Schur-reduced primitive descriptor
manifold. This is the correct DAE compatibility condition for a linear wave:
every primitive perturbation has its conserved and face-flux perturbations
reconstructed by the certified algebraic response. The slow-coordinate fiber
is reserved for the nonlinear closure counterexample searches.

Each final packet therefore has:

- continuum-profile cosine exactly one before evolution;
- local principal eigenpair defects below the declared gate;
- a smooth compact support from `2.15` to `5.40 rg`;
- identical continuum family construction on all three hybrid grids;
- no imposed `q_34` moment-null correction.

## Packet ladder

The packet histories run to `0.125 s` with 201 exact matrix-exponential output
times. The table gives the observed order from the N128/N256 and N256/N512
pair differences, plus the minimum same-time fine-pair state/rate cosine.

| Family | State-history order | Rate-history order | Centroid phase order | Damping order | Fine min cosine |
|---|---:|---:|---:|---:|---:|
| inward acoustic | `1.8014` | `1.2082` | `3.2297` | `2.5165` | `0.99852` |
| inward shear | `1.5810` | `1.0110` | `1.5227` | `-0.07156` | `0.99861` |
| material/contact | `1.8310` | `1.2781` | `1.4988` | `1.3731` | `0.99891` |
| outward shear | `1.6272` | `1.0268` | `1.7204` | `2.3174` | `0.99879` |
| outward acoustic | `1.7179` | `1.1151` | `2.8666` | `1.9812` | `0.99850` |

This is materially better than the mixed WP10c8z common mode. It establishes
that the bulk generator transports clean, pure characteristic packets with a
consistent same-time direction. It also isolates the remaining failure:
inward-shear damping does not contract under the last refinement.

The rate-history orders near one are not treated as the smooth
method-of-lines reconstruction gate. The latter is measured independently on
the exact compact packet and its characteristic flux divergence.

## Phase-defect decomposition

At `t=0`, each pure packet was differentiated through the exact production
face and source maps. The tangent balance was split into:

- inner and outer boundary transport;
- central perfect-fluid transport;
- central causal-stress transport;
- scalar Rusanov transport;
- perfect-fluid and stress geometry;
- cooling, vertical work, stress relaxation, and stream source;
- mapped and responsive-height temporal storage;
- an inferred evolving-storage remainder.

All five families identify Rusanov transport as their largest forcing-side
N256/N512 difference. The inward-shear packet is the binding family because
it has the lowest rate-history order and the only failed damping order.

For this compact packet, both inner-excision and outer-boundary tangent
differences are exactly zero. The support is separated from both boundaries.
Together with the WP10c8x/y boundary evidence and the WP10c8z coupling-location
test, this removes the tested excision and coarse/fine coupling maps from the
current leading explanation.

The causal-stress central transport itself is tiny. The problematic shear
family is being damped mainly by the scalar maximum-speed penalty selected
from all five characteristic families, rather than by its own physical shear
transport scale.

## Candidate screen

Three audit-only reconstructions were compared:

1. production primitive-chart quadratic reconstruction;
2. a horizon-regular local rapidity chart for the Eulerian velocity pair;
3. quadratic reconstruction of the selected characteristic perturbation.

All three pass the isolated smooth state/rate order gate for every family:

```text
minimum face-state order       = 2.50099 / 2.50103 / 2.50463 class
minimum characteristic-rate order = 1.97302 / 1.97303 / 1.97881
```

The finest characteristic-rate error ratios relative to production are:

| Candidate | Maximum N512 error ratio to production | Meaningful improvement |
|---|---:|---|
| production primitive quadratic | `1.00000` | no |
| horizon rapidity quadratic | `1.00000` | no |
| characteristic perturbation quadratic | `1.00075` | no |

Thus the reconstruction chart is not selected as the next intervention.
Static and smooth packet reconstruction already behaves at approximately
second order or better. The nonconvergent full-generator damping points
instead to the dissipation matrix used by the numerical flux.

The characteristic-travel-time grid was also considered but not promoted.
The five families have distinct, spatially varying coordinate speeds. A
family-specific equal-travel grid cannot define one conservative production
mesh without first selecting a common monitor and rebuilding the complete
descriptor. WP10c9a supplies no evidence that this more intrusive change is
preferable to repairing the scalar dissipation.

## Method contracts

The inherited and repeated contracts pass:

```text
maximum shared-flux defect          = 0
maximum storage-action defect       = 2.15277e-9
maximum split/restart defect        = 4.32e-15
incoming excision characteristics   = 0
dense/colored parity                = inherited passed WP10c8z
BDF2 bitwise replay                 = inherited passed WP10c8z
```

The result is not caused by loss of conservation, temporal storage,
algebraic reduction, coupling telescoping, or matrix-exponential sampling.

## Main problems and solutions

### Problem 1 — Scalar max-speed Rusanov damping does not respect the slow shear cone

The production penalty uses one largest absolute speed from all five
families. Near the horizon this scalar is usually acoustic/advective, while
the causal-shear packet has its own smaller physical signal speed. The
pure-family audit shows that Rusanov is the largest forcing-side refinement
defect and that inward-shear damping fails to contract.

#### Solution

Build an audit-only family-resolved matrix dissipation:

```text
D(delta U) = R |Lambda| L delta U
```

using the complete five-field coordinate principal system. Keep the central
physical flux, one shared face flux, and exact finite-volume telescoping.
Do not fit a scalar viscosity coefficient to the current transient.

### Problem 2 — The available local-rest eigenvectors are not yet a certified coordinate-face eigensystem

WP10c9a uses local-rest eigenvectors only to construct clean diagnostic
packets. A production matrix dissipation needs the full coordinate principal
operator, including the Valencia boost, responsive-height storage, and
nonconservative shear principal contribution.

#### Solution

Derive and certify left/right coordinate eigenvectors at every face. Require:

- the five analytic acoustic/contact/shear speeds;
- real complete eigenvectors;
- `L R = I`;
- bounded condition number;
- direct principal residual closure;
- continuity through speed ordering and near-ties;
- zero incoming physical mode at excision.

### Problem 3 — Characteristic matrix dissipation must remain conservative and differentiable

A naive eigenvector implementation can lose a common face flux, become noisy
when eigenvectors change sign/order, or break the colored finite-difference
Jacobian.

#### Solution

Use deterministic family ordering/sign conventions and a smooth absolute
value/entropy regularization only where required. Verify dense/colored
Jacobian parity on small grids and retain one identical flux vector in the two
neighboring residual rows.

### Problem 4 — A method-level pass is not a physical phase certification

The rapidity and characteristic reconstruction kernels pass smooth tests but
were not allowed to replace production. Their error coefficients do not
improve the present method and no live-coupled generator exists for them.

#### Solution

Build a complete candidate generator only after the face principal and
dissipation kernels pass. Then repeat all five pure packets at
N128/N256/N512-equivalent resolution. Only a candidate that passes the full
packet ladder may rerun the WP10c8z common mode.

## Locked next plan: WP10c9b

### Phase 1 — Freeze WP10c9a

Freeze and hash:

- all fifteen pure packet initial conditions and histories;
- family bases, speeds, phase, damping, leakage, and cosine histories;
- the complete forcing/storage tangent decomposition;
- all reconstruction-candidate method errors;
- WP10c8z generator and coupling provenance.

### Phase 2 — Certify the full coordinate principal basis

At N128/N256/N512-equivalent faces in the active inner core:

1. assemble the complete five-field coordinate principal pencil, including
   the nonconservative shear principal term;
2. order the five modes as inward acoustic, inward shear, material, outward
   shear, outward acoustic;
3. construct deterministic right and left eigenvectors;
4. compare eigenvalues with the analytic Valencia cones;
5. report eigenpair, biorthogonality, conditioning, and cross-face continuity;
6. verify zero incoming excision modes.

Stop before flux implementation if the coordinate basis is incomplete,
ill-conditioned, or discontinuous.

### Phase 3 — Implement audit-only family-resolved dissipation

Retain the production central physical flux and replace only the audit
dissipation:

```text
F* = 0.5 (F_L + F_R) - 0.5 R |Lambda| L (U_R - U_L).
```

Use a path or symmetric face state consistent with the nonlinear primitive
map. The exact implementation must:

- return one shared mass/angular-momentum/Killing-energy/stress flux;
- telescope exactly;
- reduce to scalar Rusanov when all family speeds are equal;
- remain dissipative in a declared characteristic/symmetrizer norm;
- be deterministic under eigenvector sign and near-speed ties;
- preserve positivity and causality under the existing admissibility gate.

Compare at least:

- scalar production Rusanov;
- full five-family matrix dissipation;
- a bounded shear/contact-preserving variant only if the full matrix exposes
  a specific conditioning failure.

### Phase 4 — Method and Jacobian certification

Require:

```text
constant-state jump/flux defect      = 0
smooth face-state/rate order         >= 1.8
shared-flux/telescoping defect       <= 1e-12
storage-action defect                <= 2e-5
principal eigenpair/biorthogonality  <= declared 1e-10-scale gates
no incoming excision characteristic = true
dense/colored Jacobian parity        <= existing gate
BDF2 split/replay                    = bitwise
```

The matrix dissipation may not be promoted if colored finite differences
switch eigenvector branches or if a smooth packet develops anti-diffusive
growth.

### Phase 5 — Rebuild the bounded live-coupled packet ladder

Only after Phase 4 passes, build N128/N256/N512-equivalent live-coupled
generators with the candidate dissipation and rerun all five packets.

Require:

```text
packet centroid phase order   >= 0.75
packet damping order          >= 0.75
same-time signed cosine       >= 0.90
smooth state/rate order       >= 1.8
response at live coupling     below the existing WP10c8z gate
matched coupling-location test unchanged
```

The inward-shear damping result is binding. Passing only the acoustic packets
is insufficient.

### Phase 6 — Conditional common-mode rerun

Only a candidate that passes every pure-family gate may repeat the exact
WP10c8z common nonlinear mode at N256/N512-equivalent patch resolution.

- Positive contraction: authorize one bounded nonlinear embedded-patch truth
  experiment.
- Pure packets pass but the common mode fails: diagnose nonlinear family
  coupling before any additional refinement.
- Shear packet still fails: derive a path-conservative shear Riemann coupling
  or revisit the causal-stress discretization.
- Several packets fail: redesign the full near-horizon path-conservative
  finite-volume operator.

Do not authorize N1024-equivalent refinement without positive measured
contraction.

### Hard stops

WP10c9b must not:

- change production by default;
- run fixed-`Q` averaging;
- select an initial-slip map or reduced coordinate;
- launch a production embedded patch;
- run loading-time macrosteps;
- add tide, wind, hot-state, or cycle physics.

Every later reduced architecture must repeat the exact worst-case nonlinear
fiber audit after the inner truth operator is spatially certified.

## Machine evidence

```text
outputs/tables/causal_inner_characteristic_phase_audit_wp10c9a.json
outputs/tables/causal_inner_characteristic_phase_audit_wp10c9a_arrays.npz
```

Runner:

```text
scripts/run_causal_inner_characteristic_phase_audit_wp10c9a.py
```

Core packet diagnostics:

```text
src/imri_qpe/layer3_minidisk_1d/causal_inner_characteristic_phase.py
```

Focused tests:

```text
tests/test_causal_inner_characteristic_phase.py
tests/test_causal_inner_characteristic_phase_audit_wp10c9a.py
```
