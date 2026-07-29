# Causal inner monolithic descriptor-path DAE preflight — WP10c9d6a

Date: 2026-07-29

Analyzed base:

```text
e836df2e2c0e1d180f3a8c56383498578434762e
```

## Binding classification

```text
monolithic_descriptor_path_assembly_certified_
manufactured_preflight_authorized
```

WP10c9d6a certifies one production-neutral primitive-only residual on the
declared five-cell regression context. It authorizes the next
manufactured-equilibrium and manufactured-wave method package only.

Still blocked:

- physical-export discrimination;
- a nonlinear physical trajectory;
- a production operator change;
- fixed-\(Q\) micro-solving;
- reduced slow-time evolution.

## Why the architecture changed slightly

WP10c9d5c1 selected a monolithic conservative space-storage replacement
after finding no recovery surface or stable local repair for the rejected
hybrid. The strict notation

\[
\frac{d}{dt}\mathcal U(p)+R(p)=0
\]

would require the complete temporal descriptor to be the Jacobian of one
state function \(\mathcal U(p)\).

That is true for the mapped conserved storage. It is not true for the
responsive-height contribution. The latter is a temporal one-form of the
schematic form

\[
A_H(p)\,dp,
\]

and its exterior derivative is resolved and nonzero.

The certified architecture is therefore written honestly as

\[
\frac{\Delta U_{\rm mapped}}{c\,\Delta t}
+
\frac{1}{c\,\Delta t}
\int_{\Psi_t} A_H(p)\,dp
+
R_{\rm complete}(p^{n+1})
=0.
\]

The mapped part is an exact endpoint increment. The responsive-height part is
a declared path-dependent temporal nonconservative product. Both use the
same primitive path and the same reconstruction branch. No endpoint-only
responsive-height potential is claimed.

## Temporal integrability audit

At the innermost regression cell, the scaled exterior-derivative diagnostics
are

| Quantity | Measured value |
|---|---:|
| Responsive-height exterior derivative | \(5.3877449\times10^{-1}\) |
| Complete descriptor exterior derivative | \(8.4003378\times10^{-6}\) |

The complete relative value is smaller because the exact mapped-conserved
Jacobian dominates the descriptor norm; it remains converged and nonzero.

A separate loop test traverses the same small
\((\beta_\phi,\log T)\) rectangle in opposite orders. At loop amplitude
\(10^{-4}\), the two responsive-height increments differ by

\[
4.1661806\times10^{-4}
\]

relative to the path increment. This is a direct finite-loop witness that
the responsive-height one-form is not an endpoint state potential.

This negative integrability result does not reject the DAE. It determines
the correct discrete form: a declared temporal path product, or eventually
an augmented state if an endpoint-only formulation becomes necessary.

## Unified residual

The new audit residual has these contracts:

1. exact mapped endpoint storage;
2. path-integrated responsive-height temporal work;
3. one reconstruction for temporal and spatial paths;
4. one shared conservative M/J/E face flux;
5. separate shear-principal and responsive-height-principal fluctuations;
6. local stress relaxation, geometry, cooling, stream, and lower height work
   exactly once;
7. center-broken within-cell paths, including the excision half-cell;
8. no production generator;
9. no production-anchor storage derivative.

The small declared temporal perturbation gives:

| Gate | Result | Threshold |
|---|---:|---:|
| Mapped endpoint/path closure | \(1.0504\times10^{-8}\) | \(2\times10^{-8}\) |
| Temporal reversal | \(0\) | \(2\times10^{-10}\) |
| Collinear path subdivision | \(2.2879\times10^{-15}\) | \(2\times10^{-10}\) |
| Reconstruction-factor change | \(0\) | \(10^{-14}\) |
| Complete block ledger | \(0\) | \(10^{-12}\) |
| Shared conservative-flux telescope | \(0\) | \(10^{-12}\) |
| Center-broken path adjustment | \(5.1108\times10^{-11}\) | \(2\times10^{-8}\) |
| Incoming excision characteristics | \(0\) | \(0\) |
| Source double count | \(0\) | \(0\) |

The center-broken result makes the first-cell half paths explicit without
introducing incoming boundary data. It is a method contract, not yet an
outgoing manufactured-wave certification.

## Directional differentiability

The complete backward-Euler residual was evaluated with shared fourth- and
sixth-order centered stencils at step \(10^{-4}\). Two independent smooth
directions, their sum, and a \(-0.371\) multiple were tested.

| Gate | Maximum defect | Threshold |
|---|---:|---:|
| Fourth/sixth action difference | \(1.1020\times10^{-11}\) | \(10^{-8}\) |
| Additivity | \(3.8808\times10^{-12}\) | \(10^{-8}\) |
| Homogeneity | \(8.2895\times10^{-11}\) | \(10^{-8}\) |

Thus the declared residual is a stable Jacobian target on this regression
context. No Newton solve or physical timestep was performed.

## Interpretation

WP10c9d6a resolves the first architecture question after Branch D:

- a unified residual can be assembled without importing the rejected
  production-anchor tangent;
- its conservative face and complete physical ledgers close exactly;
- its outgoing excision count is preserved;
- its local directional derivative is stable;
- responsive-height temporal work cannot honestly be collapsed into an
  endpoint-only \(\mathcal U(p)\).

The result does **not** show that the spatial/temporal discretization
converges on a radial wave, that equilibrium is preserved, or that M/J/E
exports now converge.

## Authorized next package

WP10c9d6b must remain production-neutral and perform the next four method
gates, in order:

1. exact zero-increment and declared equilibrium balance;
2. a manufactured outgoing near-horizon wave;
3. a variable-coefficient manufactured wave;
4. spatial and temporal refinement of the complete residual.

It must compare against independently integrated cell residuals, exercise
nonzero reconstructed interface jumps, and retain the same temporal path
contract. It must distinguish spatial order from temporal order.

Only if those gates pass may a uniform-grid physical-export ladder be
constructed. Embedded coupling, nonlinear common-mode evolution, fixed-\(Q\)
experiments, and slow reduction remain later conditional packages.

## Reproducibility

Canonical evidence:

```text
results/canonical/
causal_inner_monolithic_dae_preflight_wp10c9d6a/
```

Generation command:

```text
PYTHONPATH=src python3 \
  scripts/run_causal_inner_monolithic_dae_preflight_wp10c9d6a.py
```

Focused verification:

```text
9 passed
```

The full repository suite was not rerun for this bounded method-preflight
package.
