# Revised Uniform Arrival/Transfer Contract Manifest

## WP10c9d6c7c2b6a results — 2026-07-30

## Classification

`revised_uniform_arrival_transfer_contract_frozen_recertification_authorized`

Authorized next:

`WP10c9d6c7c2b6b_revised_uniform_arrival_transfer_recertification`

This package is definitions-only. It changed no physical or numerical
operator and propagated no state.

The following historical classifications remain binding and are not
amended:

- `one_way_uniform_arrival_energy_validation_failed_embedded_discrimination_blocked`
  from WP10c9d6c7c2b4;
- `arrival_history_conditioning_and_horizon_audit_complete_shear_family_transfer_audit_required`
  from WP10c9d6c7c2b5a;
- `raw_local_family_leakage_projector_rotation_sensitive_revised_transfer_observable_manifest_authorized`
  from WP10c9d6c7c2b5b.

Embedded, interface-redesign, nonlinear, production, fixed-`Q`, and reduced
slow-time work remain blocked.

## Purpose

WP10c9d6c7c2b4 formally rejected its frozen arrival-history contract. The
subsequent audits separated two causes:

1. the old history test applied an absolute `0.05` difference in units of
   initial packet energy even when physical receiving-band gain reached
   thousands;
2. raw local opposite-family energy mixes physical family transfer with the
   spatial rotation of the local invariant subspaces.

WP10c9d6c7c2b5b found no noncontracting physical or storage block. Its exact
family-transfer identity closes below `4.80e-13`, every active continuum
block truncation order is at least `2.676`, and the two equivalent local
projector algorithms agree below `3.44e-12`. It therefore authorized a new
definitions-only manifest, not an operator change.

This package freezes that revised prospective contract.

## Binding observables

### Tier I — direct physical contract

The complete Tier-I contract is inherited unchanged:

- five-field state history;
- inner and coupling M/J/E exports;
- net M/J/E drive;
- cooling;
- responsive-height work;
- exact conservative ledgers.

The original order, normalized-difference, history-cosine, and observable
error-cosine thresholds are retained.

### Tier II — arrival and covariant transfer

The binding Tier-II quantities are now:

1. positive total receiving-band energy;
2. projector-qualified target-family energy;
3. total receiver work from the exact covariant transfer ledger;
4. target-family receiver work from that ledger;
5. exact family-partition, power, block, stored-energy, and physical-work
   closure.

For a frozen linear generator `G` and family quadratic metric `Q_f`, the
family power is

```text
dE_f/dt = 0.5 u^T (G^T Q_f + Q_f G) u.
```

The certified block/source/receiver tensor is retained:

```text
T_brs = <P_r u, H G_b P_s u>.
```

The raw local opposite-family stored energy, individual block dominance,
the frozen receiving-band midpoint projection, and pointwise interface
stress remain reported diagnostics. None can reject the method alone.

This is not an exception based on the magnitude of the c2b4 failure. It is
a forward definition derived from the exact c2b5b transfer identity and the
demonstrated spatial-projector-rotation sensitivity.

## Revised history accuracy

The physical gain remains

```text
G_h(t) = E_receiving,h(t) / E_initial-source,h.
```

It may legitimately exceed one because the variable background, descriptor,
relaxation, geometry, and lower sources do work on the perturbation.

The binding discretization error is now response-relative:

```text
||G_h - G_reference||
---------------------------------------
max(max_t |G_reference(t)|, 1).
```

Amplitude, unit-shape history, and peak time are reported separately.
The old initial-energy absolute `0.05` gate is not reused and c2b4 is not
reclassified.

The primary and secondary references must be independent 769- and 513-node
continuum/collocation trajectories. Their difference must be at most `0.10`
of the N196/N392 spatial difference. If an independent history reference
cannot be constructed or fails this gate, b6b must stop.

## Projector qualification

The physical family definition uses local descriptor-compatible,
energy-orthogonal invariant subspaces on the frozen background.

Two equivalent algorithms are binding:

- overlap-tracked local generalized-eigenvector projectors;
- local polynomial spectral projectors.

The frozen gates are:

```text
projector algebra defect                  <= 2e-9
equivalent local-projector difference     <= 2e-8
```

An unresolved spectral cluster is a hard stop.

A common high-resolution local field remains a cross-grid sensitivity
diagnostic. A single frozen midpoint projector is intentionally a
spatial-rotation diagnostic; it is not an alternative physical projector
and is not counted as equivalent-projector uncertainty.

## Uncertainty and observability

Uncertainties are combined by a conservative deterministic sum or a directly
measured nuisance envelope. Root-sum-square combination is forbidden unless
independence or a measured covariance is demonstrated.

The complete prospective envelope contains:

- independent continuum-history reference;
- analytic finite-volume projection;
- equivalent local-projector algorithm;
- receiving-band placement;
- arrival-window placement;
- time sampling and quadrature;
- restart;
- roundoff.

An error-direction cosine is binding only when both refinement-error vectors
exceed their complete envelopes by the frozen factor `5`. Otherwise the
direction is classified as non-certifying, neither pass nor fail.

No slow-impact threshold is introduced because the slow state, closure, and
macro horizon are not yet defined.

## Frozen profiles

The next uniform run contains five frozen base cases. The two held-outs are
new interference mixtures within the already certified acoustic/shear
subspace; they are prospective tests of the quadratic arrival and transfer
observables, not new linearly independent characteristic families.

Calibration bases:

- acoustic;
- shear;
- mixed shear–acoustic.

Prospective held-outs, frozen before recertification:

- `(acoustic - shear)/sqrt(2)`;
- `0.5 acoustic + sqrt(3)/2 shear`.

Each base has signs `-1,+1` and amplitudes `0.5,1.0`, giving 20 binding
variants. Sign and amplitude variants are exact symmetry/scaling controls,
not 20 independent profiles.

All five bases have initial source energy approximately
`0.2400405126867389`. Their minimum target-family energy fraction is `1.0`,
and their maximum five-family partition defect is
`2.3125742654825017e-16`.

The exact packet arrays and hashes are committed in canonical evidence.

## Frozen gates

For every binding Tier-I observable and every binding Tier-II total/target
observable:

```text
RMS order                                  >= 0.75
maximum order                              >= 0.75
minimum significant-component order        >= 0.75
fine response-relative difference          <= 0.05
history cosine                             >= 0.90
observable refinement-error cosine         >= 0.90
continuum-history uncertainty / fine diff  <= 0.10
continuum-action difference                <= 2e-5
continuum-action truncation order          >= 0.75
scaling/sign-symmetry defect               <= 1e-10
transfer closure defect                    <= 2e-9
restart replay                             exact
conservative ledgers                       exact
```

The original source and receiving bands, N98/N196/N392 levels, precomputed
travel-time windows, terminal-tail checks, and stride `1/2/4` diagnostics
remain unchanged.

## Binding next decision

WP10c9d6c7c2b6b must choose exactly one branch:

1. **All Tier I, binding Tier II, continuum-reference, and held-out gates
   pass.** Certify the revised uniform arrival/transfer class and authorize
   only a definitions-only embedded manifest.
2. **Raw local leakage fails but total/target energy and the exact binding
   transfer balance pass.** Report the raw leakage as a projector-rotation
   diagnostic; it does not reject the revised class.
3. **Projector algebra or transfer identities fail.** Repair the observable
   definition; no embedded work.
4. **The independent continuum-history reference is unavailable or fails.**
   Stop uniform recertification.
5. **A calibration or held-out binding profile fails.** Freeze the exact
   failure and authorize only a local DAE/observable audit. Do not tune the
   contract.
6. **A stable block truncation defect persists on both refinement pairs.**
   Authorize only that block’s local audit.

## Reproducibility

Canonical evidence:

- `results/canonical/causal_inner_revised_arrival_contract_manifest_wp10c9d6c7c2b6a/config.json`
- `contract_manifest.json`
- `decisive_arrays.npz`
- `summary.json`
- `provenance.json`
- `SHA256SUMS.txt`

The manifest freezes five base packet arrays, 20 variants, target-family
definitions, uncertainty rules, gates, decision branches, predecessor input
hashes, implementation source hashes, and exact Git identity.

## Final statement

WP10c9d6c7c2b6a successfully freezes a revised, prospective uniform
arrival/transfer contract. It does not pass c2b4 retroactively and does not
certify embedded scattering. It authorizes only the N98/N196/N392 b6b
uniform recertification under the exact committed definitions.
