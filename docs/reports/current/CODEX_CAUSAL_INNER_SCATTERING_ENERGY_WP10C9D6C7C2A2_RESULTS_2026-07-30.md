# Manufactured Scattering-Energy Preflight

## WP10c9d6c7c2a2 — 2026-07-30

Analyzed base:

```text
de29e71f05be20c979c52354584b7b694fb26c6e
```

## Binding classification

```text
manufactured_interface_patch_rejected_
unidirectional_characteristic_core
```

The C4 manufactured state and the invariant energy machinery pass their
method gates. The selected bidirectional experiment does not.

At the exact physical interface core, all five complete coordinate
characteristics travel toward smaller radius:

```text
-0.46475173
-0.45343377
-0.44839857
-0.44332349
-0.43185508
```

There are therefore:

```text
0 positive-speed families
5 negative-speed families
```

A packet on the coarse outer side can cross into the fine inner side. No
packet can cross the same exact core from the fine side to the coarse side
without changing the background state and its characteristic signs.

WP10c9d6c7c2a1 required both incidence directions and exact physical-core
parity simultaneously. Those requirements are incompatible for this core.
Uniform c2b propagation is not authorized.

This is a rejection of the selected **bidirectional manufactured experiment**,
not a failure of the coupling operator and not a failure of the energy
definition.

## C4 manufactured primitive state

The 98-cell patch uses:

- the exact 12 parent-cell primitive core corresponding to parent cells
  42–53 and patch cells 43–54;
- a quintic C4 spline through those exact core cell values;
- fourth-order endpoint Taylor jets;
- the frozen degree-nine C4 smootherstep over 12 parent cells;
- the unchanged parent cell-0 and cell-63 primitive states as the two constant
  far anchors.

No physical matrix is interpolated. Every temporal, principal, lower-source,
characteristic, projector and energy quantity is recomputed from the extended
primitive chart.

The construction passes:

```text
core replay defect                 3.55e-15
scaled C4 join defect              0
scaled C4 far-state defect         0
admissible cells                   98 / 98
minimum characteristic speed gap  2.47e-3
maximum eigenvector condition      3.31e3
maximum imaginary part             0
```

## Exact physical-core parity

The patch reproduces the parent core without changing the local physical
operator:

```text
primitive-state defect             5.52e-16
cell-center defect                 2.26e-15
cell-measure defect                9.32e-14
face-measure defect                3.61e-15
temporal-storage-matrix defect     4.83e-15
spatial-principal-matrix defect    6.18e-15
```

The maximum defect is `9.32e-14`, below the frozen `1e-12` gate.

## Normalization-invariant energy

For the complete frozen coordinate pencil,

\[
A p_{,ct}+B p_{,R}=\sum_k C_kp+A f,
\]

the audit forms

\[
K=A^{-1}B
\]

and its scale-invariant spectral projectors \(P_a\). With the frozen physical
primitive scales represented by \(G_0\), the energy metric is

\[
H=\sum_a P_a^T G_0 P_a.
\]

This construction:

- is invariant under characteristic eigenvector sign and normalization;
- is positive definite;
- makes separated family subspaces mutually energy orthogonal;
- symmetrizes the complete coordinate evolution matrix;
- includes responsive-height storage and both principal-source matrices
  through \(A\) and \(B\).

It is a certified mathematical symmetrizer in the declared physical primitive
units. No thermodynamic-entropy claim is made.

The maximum defects over the patch are:

```text
projector identity                 7.57e-14
projector idempotence              3.41e-13
cross-projector product            3.20e-13
energy orthogonality               6.27e-16
symmetrizer                        7.00e-15
eigenpair                          1.52e-13
rescaling invariance               6.82e-13
minimum energy eigenvalue          2.18e-1
```

All projector gates pass.

## Complete variable-coefficient energy ledger

In log radius \(x=\ln R\), define

\[
K_x=\frac{A^{-1}B}{R},
\qquad
Q=H K_x.
\]

For a manufactured perturbation \(u\), the checked identity is

\[
\partial_{ct}\!\left(\frac12u^THu\right)
+
\partial_x\!\left(\frac12u^TQu\right)
=
\sum_k u^T H L_k u
+u^THf
+\frac12u^TQ_{,x}u.
\]

The lower blocks are:

```text
perfect-fluid geometry
stress geometry
radiative cooling
responsive-height lower work
stress relaxation
```

Responsive-height temporal storage is already in \(H\), and
responsive-height and shear principal terms are already in \(Q\). The
last term above records variable-background work. Every term appears once.

The maximum algebraic ledger defect over 98/196/392 reference levels is:

```text
3.49e-16
```

An independent conservative-versus-expanded fourth-order product-rule audit
gives:

```text
N98   5.19860e-3
N196  5.38252e-4
N392  4.01535e-5
```

The reference-uncertainty/fine-error ratio is:

```text
0.07460 <= 0.10
```

The constant-state residual is zero. Pure-family packet energies are
normalization invariant, sign invariant, quadratic in amplitude and close
their characteristic flux identities below `1.6e-14`.

## Why propagation stops

The method gates are not the only c2a2 gates. The parent manifest also froze:

```text
both incidence directions required
```

The attempted fine-to-coarse acoustic and shear packets have negative
coordinate speeds throughout their left support:

```text
outward acoustic: -0.6357 to -0.2557
outward shear:    -0.6525 to -0.4099
```

Their local-rest family labels do not override the coordinate causal cone.
Both packets still move inward and cannot cross the interface toward larger
radius. No travel windows are frozen for an impossible direction.

Changing the interface state until a positive characteristic appears would
violate exact physical-core parity. Retaining the exact core while declaring
fine-to-coarse incidence would mislabel the propagation direction. Neither is
allowed.

## Correct next decision

The next work package must be definitions-only:

```text
WP10c9d6c7c2a3
scattering-scope revision
```

It must choose, without combining the choices:

1. **Physical-core route.** Preserve the exact core and certify only the
   physically available coarse-to-fine interface scattering. This is the
   relevant causal direction for the present inflow.
2. **Generic bidirectional method route.** Define a separate, explicitly
   nonphysical manufactured interface state with separated positive and
   negative families. This can test a generic two-way coupling method but
   must relinquish the exact physical-core claim.

The first route is the preferred next physical audit. The second is optional
method stress testing.

## Hard stops

Do not:

- start uniform c2b propagation;
- claim bidirectional physical scattering;
- alter characteristic signs or fit the background;
- redesign the coupling interface;
- relabel c7c1b;
- begin embedded nonlinear, fixed-Q or reduced slow-time work.

## Canonical evidence

```text
results/canonical/
causal_inner_scattering_energy_wp10c9d6c7c2a2/
```

The package contains the manufactured state, complete matrices, invariant
projectors, energy metrics, packet trials, reference errors, configuration,
manifest, provenance and hashes.

## Verification

```text
31 focused and adjacent tests passed
1040 repository tests passed
4 subtests passed
2 pre-existing policy tests failed
```

The policy failures are unchanged: two older canonical packages still use the
legacy `PROSPECTIVE MANIFEST ONLY` provenance status, and the tracked-file
count is `1179 >= 850`. No scientific or numerical test failed.
