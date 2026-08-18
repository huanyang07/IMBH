# Hidden rank-capacity audit WP10c9d6c7c3b5c4f25ac

## Classification

`two_anchor_R130_pointwise_transfer_capacity_not_ruled_out_direct_structure_preserving_basis_manifest_authorized`

The exact pointwise Eckart-Young transfer tails were evaluated at both saved anchors without constructing or promoting a reduced dynamical model.

Minimum pointwise orders (primary: original 16, safety 37, heldout: original 16, safety 37). Hidden order 130 safety-margin pass: `True`.

## Decisive result

At hidden order 130, the largest pointwise normalized dynamic lower bound is
`1.34e-7` and the largest RMS lower bound is `9.55e-8`. The worst ratio to
the already tightened, tenfold-safety gate is only `1.24e-5`. The face-flux
block has at most 85 output rows, so its rank-130 Eckart-Young tail is exactly
zero. Frequency solves close below `2.44e-14`.

The original transfer limits are first met at pointwise rank 16 at both
anchors. The tenfold-safety limits are first met at rank 37. Thus the R320
budget, which provides 130 hidden coordinates after retaining 162
conservative and 28 exact nonstable coordinates, has substantial pointwise
rank capacity.

## Interpretation

This rules out insufficient pointwise matrix rank as the explanation for the
WP10c9d6c7c3b5c4f25aa failure. The remaining problem is coherent realization:
one fixed stable subspace and one reduced operator must approximate all
frequencies simultaneously. A direct construction should therefore weight
primal resolvent states by relative output error, resolve frequency intervals
rather than only endpoints, and retain the square-root Galerkin test so every
candidate remains Lyapunov stable.

A pass means only that R320 is not ruled out by pointwise transfer rank. Coherent realization, stability-preserving basis selection, online integration, and predictive evolution remain uncertified.

Authorized next artifact: `definitions_only_direct_relative_resolvent_basis_manifest`.
