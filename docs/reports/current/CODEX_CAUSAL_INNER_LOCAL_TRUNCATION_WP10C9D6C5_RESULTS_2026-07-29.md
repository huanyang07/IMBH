# Causal inner local truncation and packet-width audit — WP10c9d6c5

Date: 2026-07-29
Analyzed base commit: `c082f62f62f9c5c9f28e61c7f25f4d353a5f7a09`
Analyzed parent: `3d973aa28c68d242bd33c88efe2226e2e48eb281`

## Binding classification

```text
narrow_profile_preasymptotic_width_crossover_no_redesign
```

WP10c9d6c5 changes no physical or numerical operator. It preserves the
WP10c9d6c4 classification

```text
prospective_heldout_uniform_validation_failed
```

without amendment.

The new audit passes its method, independent-continuum, ledger, replay, and
restart contracts. It does not find a noncontracting first-half-cell,
space-storage, principal-path, relaxation, or lower-source defect.

The strict instantaneous direction failure follows packet width rather than
packet center:

- the original narrow profile centered at \(1.84\,r_g\) fails;
- the same narrow profile shifted to \(1.98\,r_g\) also fails;
- doubling the log-radius width makes both centers pass;
- all four cumulative histories pass.

The complete local DAE truncation of the original profile contracts in every
fixed boundary band, with minimum fine-pair order \(1.266\). A fitted time
shift explains only \(22.8\%\) and \(32.2\%\) of the two refinement
differences. Therefore neither a boundary defect nor a phase-only explanation
is selected.

The authorized next step is a prospective, operator-neutral transport-packet
validation with an explicit resolution-per-width contract. Boundary,
space-storage, path, or source redesign is not authorized. Embedded,
nonlinear, production, fixed-\(Q\), and reduced slow-time work remain blocked.

## Purpose

WP10c9d6c4 found one failed prospective profile: an outgoing inward-acoustic
packet centered at \(1.84\,r_g\) with log-radius width \(0.065\). Its
instantaneous export norms contracted strongly, but its refinement-error
cosine was only

\[
0.85964 < 0.90.
\]

The profile has \(94.44\%\) of its peak amplitude at the \(1.8\,r_g\)
excision surface. Its failure could therefore have indicated:

1. a genuine inner-face or first-band truncation inconsistency;
2. a coupled temporal-storage defect;
3. a principal/source defect;
4. an exit-time error; or
5. an under-resolved narrow-packet crossover.

WP10c9d6c5 separates those possibilities without changing the operator.

## Frozen construction

The exact c4 background, field scales, grids, tangents, time samples,
observable scales, profile, and physical gates are replayed. The original
profile lift reproduces c4 with relative defect

\[
1.77\times10^{-114},
\]

and every propagated export history reproduces its frozen c4 counterpart to
at most

\[
1.26\times10^{-15}.
\]

The four diagnostic outgoing profiles are:

| Profile | Center | Log width | Role |
|---|---:|---:|---|
| original | \(1.84\,r_g\) | 0.065 | frozen c4 binding profile |
| wider | \(1.84\,r_g\) | 0.130 | width control |
| shifted | \(1.98\,r_g\) | 0.065 | support control |
| shifted and wider | \(1.98\,r_g\) | 0.130 | width/support control |

They use the same physical inward-acoustic vector and amplitude. No
coefficient, path, tolerance, or operator term is fitted to their results.

## Independent continuum action

The audit constructs the continuum linearized action on two uniform
log-radius collocation grids with 769 and 513 nodes. It uses the same
certified analytic local physical maps but a distinct high-order quintic
collocation derivative and cell integration.

For a smooth base state \(p_*(R)\) and perturbation \(q(R)\), the reference
includes the complete nonstationary linearized DAE:

\[
\frac{\mathcal A}{c}T(p_*)q_t
+
\frac{\mathcal A}{c}T_{,p}(p_*)[q]\,p_{*,t}
+
\delta R_{\rm spatial}[q]
=0.
\]

The second temporal term is essential. It contains the mapped and
responsive-height storage-rate derivatives acting on the continuum base
rate. Omitting it would audit a different DAE.

Across all seven truncation profiles:

| Reference check | Maximum |
|---|---:|
| 769/513 continuum-rate relative difference | \(6.58\times10^{-7}\) |
| 769/513 fixed-physical export difference | \(1.22\times10^{-11}\) |
| pointwise continuum DAE ledger defect | \(6.71\times10^{-16}\) |

The continuum reference is therefore much more accurate than the discrete
N256/N512 differences under study.

## Discrete and continuum ledgers

The discrete action is separated into:

1. mapped temporal action;
2. responsive-height temporal action;
3. mapped storage-rate derivative;
4. responsive-height storage-rate derivative;
5. inner shared face;
6. remaining conservative transport;
7. shear-principal path;
8. height-principal path;
9. local stress relaxation;
10. geometry;
11. cooling;
12. stream source; and
13. lower responsive-height work.

The conservative transport block is split without double counting: the inner
shared-face contribution is removed from the transport remainder, and their
sum equals the original conservative block.

| Ledger check | Maximum | Gate |
|---|---:|---:|
| discrete block sum | \(2.58\times10^{-11}\) | \(2\times10^{-10}\) |
| integrated continuum block sum | \(2.65\times10^{-11}\) | \(2\times10^{-9}\) |
| truncation block sum | \(3.07\times10^{-13}\) | \(2\times10^{-10}\) |
| split/restart envelope defect | \(7.90\times10^{-16}\) | inherited |

All four monolithic tangents retain zero incoming excision characteristics.

## Initial export-map error

The directly evaluated continuum export supplies an independent \(t=0\)
target. For the original narrow packet, the fixed-physical export errors are:

| Grid | Error norm |
|---|---:|
| N128 | \(2.54\times10^{-4}\) |
| N256 | \(7.64\times10^{-5}\) |
| N512 | \(9.68\times10^{-6}\) |

The two fine-pair orders are

\[
1.733,\qquad 2.980,
\]

and the N256/N512 continuum-error direction cosine is

\[
0.99999985.
\]

Thus the nonzero \(t=0\) contribution is an ordinary convergent export-map
error. It is not a noncontracting boundary inconsistency.

## Fixed physical boundary-band truncation

The nested bands use exact common N64 faces:

\[
[1.8,1.95316]\,r_g,\qquad
[1.8,2.11935]\,r_g,\qquad
[1.8,2.29968]\,r_g.
\]

For the original failed profile:

| Band outer radius | Minimum N128/N256 and N256/N512 order | N256/N512 direction cosine |
|---:|---:|---:|
| \(1.95316\,r_g\) | 1.266 | 0.9999994 |
| \(2.11935\,r_g\) | 1.887 | 0.9999989 |
| \(2.29968\,r_g\) | 1.784 | 0.99999998 |

Every band therefore passes the predeclared \(p\ge0.75\) contraction gate.

The signed boundary group is the largest balancing contribution. At the
finest grid its target-aligned fractions are approximately:

\[
0.768,\qquad0.940,\qquad0.959
\]

through the three bands, with fixed-coefficient residual ratios

\[
0.232,\qquad0.0596,\qquad0.0407.
\]

This is useful localization, but it is not causal evidence for a broken
boundary operator:

1. the total truncation contracts strongly;
2. the boundary contribution itself is part of that convergent truncation;
3. no intervention has been performed; and
4. the width controls below distinguish resolution from location.

No mapped-storage, responsive-height-storage, principal, relaxation, or
lower-source group passes the same stable target-alignment criteria.

## Width and support controls

The unchanged c4 instantaneous/cumulative gates are applied to all four
profiles:

| Profile | History | \(p_{\rm RMS}\) | \(p_{\max}\) | \(\min p_a\) | \(d_{\rm fine,max}\) | \(\cos_{\rm error}\) | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| original narrow | instantaneous | 1.980 | 1.424 | 1.720 | \(3.61\times10^{-5}\) | 0.8596 | no |
| original narrow | cumulative | 2.173 | 2.309 | 1.630 | \(8.60\times10^{-7}\) | 0.9263 | yes |
| shifted narrow | instantaneous | 2.227 | 2.323 | 1.754 | \(4.09\times10^{-5}\) | 0.6984 | no |
| shifted narrow | cumulative | 2.365 | 2.142 | 1.802 | \(1.82\times10^{-6}\) | 0.9048 | yes |
| original wider | instantaneous | 2.368 | 2.763 | 1.847 | \(4.68\times10^{-6}\) | 0.9680 | yes |
| original wider | cumulative | 2.172 | 2.303 | 1.841 | \(3.02\times10^{-7}\) | 0.9757 | yes |
| shifted wider | instantaneous | 2.381 | 2.629 | 1.875 | \(4.34\times10^{-6}\) | 0.9359 | yes |
| shifted wider | cumulative | 2.339 | 2.392 | 1.895 | \(4.47\times10^{-7}\) | 0.9752 | yes |

Shifting the narrow profile away from the excision face does not restore the
error direction. Doubling its width does, at both centers.

This result has a direct resolution interpretation. On the N128/N256/N512
active grids, a width \(0.065\) spans only approximately

\[
1.6,\qquad3.2,\qquad6.4
\]

cells per Gaussian log-radius width. The doubled profile spans approximately

\[
3.2,\qquad6.4,\qquad12.7.
\]

The failed fine triplet is therefore comparing a packet that is
under-resolved on its coarsest member with increasingly resolved
representations. A rotating higher-order error contribution is expected in
that crossover and is consistent with the observed strong norm contraction.

The historical c4 failure remains binding and is not retroactively passed.
The c5 conclusion is only that it does not select a boundary redesign.

## Phase-amplitude audit

For the original profile, best-fit time shifts are

\[
\delta t_{128,256}=1.75\times10^{-4}\ {\rm s},
\qquad
\delta t_{256,512}=5.08\times10^{-5}\ {\rm s},
\]

with observed shift order \(1.79\). However, the time-shift components explain
only

\[
22.8\%,\qquad32.2\%
\]

of the two error energies, below the predeclared \(80\%\) gate.

Removing \(t=0\) changes the weighted error cosine from

\[
0.8538
\quad\hbox{to}\quad
0.8496,
\]

so the initial sample is not the cause. Removing the fitted phase component
also does not restore a common error direction.

The phase audit therefore rejects a phase-only explanation.

## Historical calibration representation

The historical calibration is propagated with:

1. the c4 quintic C4 fit;
2. the independent septic C6 fit; and
3. a quintic fit whose inner trace is inferred from a one-sided polynomial
   matching the first five proper-measure cell averages.

Relative to the binding N256/N512 historical export difference:

| Alternative | Representation/fine-spatial ratio |
|---|---:|
| septic C6 | 2.45 |
| one-sided inner trace | 16.61 |

Both exceed the unchanged `0.10` eligibility threshold. The historical
profile therefore remains calibration only and is removed from binding causal
attribution. Its c4 rejection is preserved.

## Interpretation

WP10c9d6c5 separates association from cause:

- the boundary block is where most of the narrow-packet truncation resides;
- that truncation nevertheless contracts faster than the required order;
- its direction is extremely stable on the two fine grids;
- the same strict history failure persists after shifting the narrow packet;
- widening the packet restores the strict history gate;
- a timing shift is not large enough to explain the failure.

Therefore:

\[
\boxed{
\text{resolved packet width, not a demonstrated boundary defect,
controls the c4 direction failure.}
}
\]

No half-cell, space-storage, principal-path, or source intervention is
selected.

## Authorized next package

### Prospective uniform transport-packet validation

The next package must remain operator-neutral and prospective.

1. Freeze the c5 operator, background, scales, and strict export gates.
2. Declare a resolved-profile contract before propagation, including a
   minimum number of cells per log-radius width on the coarsest binding grid.
3. Use several centers, both characteristic signs where physically outgoing,
   multiple five-field mixtures, and widths both above and below the
   resolution threshold.
4. Treat below-threshold narrow packets as stress diagnostics, not as the
   primary continuum certification set.
5. Retain the original and shifted narrow c5 profiles unchanged as historical
   failed controls.
6. Require every resolved prospective profile to pass instantaneous and
   cumulative order, maximum-difference, history-cosine, and
   refinement-error-cosine gates.
7. Keep time-aligned diagnostics explanatory only.
8. If all resolved profiles pass, authorize uniform-to-embedded
   discrimination. If a resolved profile fails, return to local truncation
   with that prospectively frozen profile.

No N1024 refinement rescue is authorized.

## Stop gates

WP10c9d6c5 does not authorize:

- a boundary half-cell redesign;
- a fixed-band space-storage redesign;
- a path/source repair;
- embedded coupling work;
- nonlinear evolution;
- production promotion;
- fixed-\(Q\) averaging;
- reduced slow-time evolution;
- tide, wind, hot-state, S-curve, or QPE-cycle physics.

## Reproducibility

Canonical evidence is stored in:

```text
results/canonical/causal_inner_local_truncation_wp10c9d6c5/
```

It contains:

- the frozen c5 configuration;
- both continuum-reference contracts;
- cellwise radius/block-resolved truncation arrays;
- direct discrete and continuum face-flux JVPs;
- fixed-band vectors and complete Gram matrices;
- all width/support histories;
- phase/amplitude diagnostics;
- historical representation controls;
- source hashes, provenance, and SHA-256 checksums.

The focused c5 suite passes. The known repository tracked-file ceiling is a
separate hygiene issue and must not be mixed into this scientific result.

Verification on the completed package:

- the c3/c4/c5 and canonical-artifact focused suite passes
  `21` tests;
- the repository-wide run completed with `892` tests and `4` subtests
  passing;
- that first broad run exposed two c5 packaging defects (the canonical
  status vocabulary and central manifest refresh), both of which were fixed
  and covered by the passing focused rerun; and
- its only remaining failure is the pre-existing tracked-file ceiling
  policy, not a scientific or numerical test.
