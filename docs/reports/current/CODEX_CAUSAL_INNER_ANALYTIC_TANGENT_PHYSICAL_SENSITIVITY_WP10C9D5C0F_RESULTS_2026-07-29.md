# Causal inner analytic-tangent physical sensitivity — WP10c9d5c0f

Date: 2026-07-29

Analyzed base:

```text
e5fd93352aea3dc920e528bb566b60fa7a3c8b0c
```

## Binding classification

```text
analytic_tangent_physical_sensitivity_passed_
extended_non_tautological_localization_authorized
```

WP10c9d5c0f passes. The physical-export histories are insensitive to
replacing the rejected historical finite-difference candidate generator by
the cross-grid-certified analytic frozen-subspace generator.

This result makes the earlier physical rejection more robust. It does not
reverse that rejection and does not certify the candidate operator.

Authorized next work:

- WP10c9d5c1 extended non-tautological localization only.

Still blocked:

- self-consistent candidate tangent;
- frozen candidate recertification;
- production promotion;
- nonlinear evolution;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

## Question tested

WP10c9d5c0e established that an explicitly linear analytic frozen-subspace
tangent passes all declared method gates on the N128-, N256-, and
N512-equivalent embedded grids. WP10c9d5c0f asks whether using that tangent
instead of the historical finite-difference generator materially changes the
physical conclusion.

The comparison is intentionally controlled:

1. The historical and analytic generators evolve the same initial
   perturbation.
2. Both histories use the same sixth-order moving-projector block/output map.
3. Both histories use the same sixth-order inner-face flux map.
4. Fixed equilibrium physical scales normalize all 13 exported observables.
5. The exact common mode and one held-out near-excision perturbation are both
   binding.

Thus the work package changes the generator representation without
simultaneously redefining the physical observable.

## Results

### Common mode

The largest derivative-choice export difference over all three grids is

\[
d_{\rm derivative}^{\rm common}
=2.29163\times10^{-9}.
\]

The binding analytic medium–fine spatial difference is

\[
d_{\rm spatial}^{\rm common}
=7.34220\times10^{-4},
\]

so

\[
\frac{d_{\rm derivative}^{\rm common}}
     {d_{\rm spatial}^{\rm common}}
=3.12118\times10^{-6}.
\]

The maximum finite-time state-action difference is
\(2.83705\times10^{-8}\), and the maximum first-cell state difference is
\(2.56979\times10^{-8}\).

### Held-out near-excision perturbation

The largest derivative-choice export difference is

\[
d_{\rm derivative}^{\rm heldout}
=2.47198\times10^{-7}.
\]

The binding analytic medium–fine spatial difference is

\[
d_{\rm spatial}^{\rm heldout}
=3.24967\times10^{-2},
\]

so

\[
\frac{d_{\rm derivative}^{\rm heldout}}
     {d_{\rm spatial}^{\rm heldout}}
=7.60687\times10^{-6}.
\]

The maximum finite-time state-action difference is
\(2.00413\times10^{-7}\), and the maximum first-cell state difference is
\(1.72502\times10^{-6}\).

### Propagation replay

The largest split/restart defect across both methods, both perturbations, and
all three grids is

\[
9.44666\times10^{-15}.
\]

## Gates

| Gate | Threshold | Common mode | Held-out near excision | Result |
|---|---:|---:|---:|---|
| Maximum normalized export difference | \(\le 5\times10^{-3}\) | \(2.29\times10^{-9}\) | \(2.47\times10^{-7}\) | Pass |
| Derivative/spatial difference ratio | \(\le 0.10\) | \(3.12\times10^{-6}\) | \(7.61\times10^{-6}\) | Pass |
| Split/restart defect | \(\le 10^{-10}\) | \(2.96\times10^{-15}\) | \(9.45\times10^{-15}\) | Pass |

All gates pass by wide margins.

## Interpretation

The historical and analytic generators differ in matrix norm by roughly
\(0.7\)--\(1.1\times10^{-7}\), but the finite-time physical exports differ
by many orders of magnitude less than the binding medium–fine spatial
differences.

Therefore:

1. The failed WP10c9d5/WP10c9d5b physical-export behavior is not an artifact
   of the historical finite-difference generator.
2. The analytic frozen-subspace tangent is suitable for the next cache-first
   localization audit.
3. The rejected hybrid candidate remains rejected.
4. No evidence here identifies a recovery radius or a causal ledger sector.

## Next authorized package

WP10c9d5c1 must:

1. search through the actual last common face whose complete reconstruction
   halo lies below the coupling interface;
2. require two consecutive surfaces passing instantaneous and cumulative
   M/J/E order, fine-difference, history-cosine, and refinement-error-cosine
   gates;
3. use directly evaluated outer-face JVPs as targets and prefix-reconstructed
   faces only as conservation checks;
4. exclude the target outer face from every explanatory group;
5. use fixed physical M/J/E scales, signed group sums, complete Gram cross
   terms, and no fitted coefficients;
6. treat a stable group as a hypothesis selector rather than causal proof.

Only one evidence-selected intervention may follow:

- recovery before coupling \(\rightarrow\) conservative extraction-surface
  audit;
- stable boundary/first-cell contribution \(\rightarrow\) source-balanced
  outgoing half-cell candidate;
- stable mapped-storage/anchor contribution \(\rightarrow\) self-consistent
  candidate tangent;
- stable principal or lower-source contribution \(\rightarrow\) one targeted
  consistency audit;
- no recovery and no stable mechanism \(\rightarrow\) monolithic conservative
  space–storage DAE replacement.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f/
```

The package records:

- both methods' signal, cumulative, face-flux, and scaled-state histories;
- first-cell histories;
- componentwise physical differences;
- fixed observable scales;
- held-out initial vectors;
- source and parent-canonical hashes;
- runtime environment and BLAS/LAPACK metadata.

The run consumes only committed WP10c9d5c0e replay and decisive arrays.
