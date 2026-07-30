# WP10c9d6c7c2b3 — Positive semidiscrete energy-transfer contract

- Classification:
  `positive_fixed_band_arrival_energy_contract_frozen_uniform_validation_authorized`
- Operator changed: `False`
- Propagation executed: `False`
- The c2b1 rejection and c2b2 interpretation are preserved.
- Embedded, nonlinear, fixed-Q, and reduced evolution remain blocked.

## Binding result

A positive, normalization-invariant replacement for the rejected local
face-transmission ratio is frozen.

For a perturbation state \(\delta p\), physical energy metric \(H\), and a
fixed cell band \(B\), define

\[
E_B(t)
=
\frac12\sum_{i\in B}
\delta p_i(t)^T H_i\delta p_i(t)\,\Delta\ln R_i.
\]

The future primary one-way arrival observable is

\[
\boxed{
\mathcal A_{\rm total}
=
\frac{
\int_{\mathcal W}E_{\rm receiving}(t)\,dt
}{
|\mathcal W|E_{\rm initial,source}
}.
}
\]

It is a dimensionless, nonnegative time-averaged occupancy of the receiving
band. It is not assumed to lie below one: the variable background,
descriptor, stress relaxation, geometry, responsive-height work, and lower
sources may amplify or attenuate the stored perturbation energy.

The companion history

\[
E_{\rm receiving}(t)/E_{\rm initial,source}
\]

and the peak value of that history remain convergence diagnostics.

## Frozen physical bands

On the N98 reference geometry:

| Role | Faces | Initial energy |
|---|---:|---:|
| receiving band | `[6,49]` | exactly zero |
| source band | `[52,95]` | positive |
| upstream null-diagnostic band | `[95,98]` | exactly zero |

The same physical bands are mapped by exact nested integer factors:

```text
N98:  receiving [6,49],   source [52,95],   upstream [95,98]
N196: receiving [12,98],  source [104,190], upstream [190,196]
N392: receiving [24,196], source [208,380], upstream [380,392]
```

The receiving band is separated from the packet support by the existing
three-cell interface clearance. The uncertainty-only receiving bands are
the predeclared combinations of lower faces `5/6/7` and upper faces
`48/49`; observed histories may not select or move them.

## Positive-reference preflight

All four nonzero frozen packets have source-band energy

```text
0.2400405126867389
```

to roundoff. The zero null has exactly zero energy. The minimum eigenvalue of
the physical energy metric is

```text
0.21758209501719575,
```

and the maximum five-family energy-partition defect is

```text
1.1562871327412506e-16.
```

The receiving and upstream diagnostic bands are exactly empty initially for
all packets. The replacement therefore has a positive denominator for every
binding packet and a clean zero baseline in the arrival bands.

## Family arrival and leakage

At every cell the state is split by the normalization-invariant,
energy-orthogonal projectors certified in c2a2. The target-family indices are:

```text
acoustic                  [0]
shear                     [1]
mixed shear-acoustic      [0,1]
```

Family arrival uses the same positive band-energy formula after projection.
The binding partition is

\[
\mathcal A_{\rm target}
+
\mathcal A_{\rm leakage}
=
\mathcal A_{\rm total}.
\]

This avoids eigenvector-normalization dependence. It also avoids assigning a
nonlocal descriptor-dual energy action to two geometric faces.

## Arrival windows

The primary windows are inherited from the c2a3 characteristic travel-time
calculation and were fixed before any propagation:

| Packet family | Arrival window |
|---|---:|
| acoustic | `[0,11.141908685984637] s` |
| shear | `[0,11.82686804912109] s` |
| mixed shear-acoustic | `[0,11.82686804912109] s` |

The three existing padding factors `0.5/1.0/1.5`, time-sample counts
`257/513/1025`, and stride checks `1/2/4` are retained as uncertainty
controls. Observed peaks may not reposition the binding windows.

## Exact energy ledger

The c2b2 identity remains the sole binding semidiscrete energy ledger:

```text
stored-energy change
  = integrated descriptor-dual action of all ten physical/storage blocks
  = integrated sum of all actual shared-face and source actions.
```

The future run must report the complete block work, stored-energy change,
shared-face closure, and time-integration residual. It must not reintroduce:

```text
T = power_at_face_6 / power_at_face_49.
```

The c2b1/c2b2 local face ratios remain rejected historical diagnostics.

## Frozen uncertainty and gates

Uncertainty is combined by a conservative deterministic sum or a directly
measured nuisance envelope. Root-sum-square combination is forbidden unless
independence is demonstrated. The components are:

- continuum reference;
- analytic projection;
- invariant-subspace choice;
- receiving-band placement;
- arrival-window padding;
- time sampling;
- restart;
- roundoff.

An error-direction cosine is binding only when both refinement-error norms
exceed the corresponding frozen uncertainty envelope by the factor five.
No slow-impact threshold is introduced because the slow variables and macro
horizon have not been defined.

The future uniform gates remain:

```text
RMS order                              >= 0.75
maximum order                          >= 0.75
significant-component order            >= 0.75
fine normalized difference             <= 0.05
history cosine                         >= 0.90
observable refinement-error cosine     >= 0.90
reference uncertainty / fine difference <= 0.10
energy-ledger defect                   <= 1e-10
```

State and flux quantities must scale linearly with amplitude. Energy
quantities must scale quadratically and be invariant under sign reversal.
The material-family and exact-zero packets are retained as null controls.

## Scientific decision

This package establishes only the following:

1. the c2b1 uniform transmission rejection remains binding;
2. the c2b2 exact semidiscrete identity remains certified;
3. a strictly positive, fixed-band arrival observable is well defined;
4. its spatial bands, time windows, projectors, uncertainty rules, and gates
   are now frozen before propagation;
5. no operator, interface, embedded method, or nonlinear method is selected
   or certified.

The positive contract is not a post-hoc pass for c2b1. It is a new
prospective observable whose numerical behavior has not yet been measured.

## Verification

- Definitions-only canonical tests: `3 passed`.
- Focused c2a–c2b3 chain: `35 passed`.
- Full repository suite: `1061 passed`, `4 subtests passed`, `2 failed`.
- The two failures are the pre-existing canonical-status vocabulary value
  `PROSPECTIVE MANIFEST ONLY` and tracked-file count `1218 >= 850`.
- No scientific or numerical test failed.
- No propagation was executed.

## Authorized next package

Only the following uniform package is authorized:

```text
WP10c9d6c7c2b4_one_way_uniform_arrival_energy_validation
```

It must reproject the frozen analytic packets on N98/N196/N392, propagate
the unchanged tangents, evaluate total/target/leakage arrival energies and
the exact c2b2 ledger, and apply every frozen uncertainty and convergence
gate.

Decision:

- all uniform arrival contracts pass:
  authorize `WP10c9d6c7c2c2` embedded arrival-energy discrimination;
- one binding uniform contract fails:
  freeze that profile and authorize only a local
  truncation/observability audit;
- uncertainty, positivity, or partition fails:
  repair the observable definition before interpretation.

Embedded c2c1/c2c2, nonlinear evolution, fixed-Q experiments, reduced
slow-time evolution, N1024 refinement, and operator/interface redesign
remain blocked.
