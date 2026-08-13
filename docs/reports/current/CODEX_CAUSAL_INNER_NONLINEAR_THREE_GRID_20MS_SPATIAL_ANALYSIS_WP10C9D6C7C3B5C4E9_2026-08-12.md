# Three-grid 20 ms spatial analysis WP10c9d6c7c3b5c4e9

## Classification

`three_grid_20ms_state_certified_extraction_tangent_uncertainty_requires_fine_generic_anchor`

This analysis-only package compares the committed coarse and middle nonlinear generic responses with the completed fine block-tangent response. It executes no trajectory.

## State

The common-parent state passes with RMS/max/minimum-component orders `1.975860` / `1.978627` / `1.965967`. Its refinement-error cosine is `0.995040945`, fine difference is `1.137550e-06`, temporal ratio is `7.366344e-02`, and surrogate ratio is `6.246112e-02`.

## Certified extraction partition

Aggregate instantaneous/cumulative/window-mean RMS orders are `1.941925` / `1.932339` / `1.954713`; maximum orders are `1.973389` / `1.953076` / `1.980893`. Refinement-error cosines are `0.995946622`, `0.995541470`, and `0.996701399`.

The conservative fine nonlinear-surrogate ratios are `0.615786` instantaneous, `0.421041` cumulative, and `0.553321` for window means. They exceed the frozen `0.10` gate, so the tangent-only fine response cannot issue the extraction certificate.

The low-order extraction components are confined to cooling and vertical-work channels. Their current nominal orders are diagnostic because the binding fine nonlinear-anchor uncertainty has not yet been removed.

## Decision

The state spatial contract is certified: `True`. The complete 20 ms spatial certificate is issued: `False`. A fine nonlinear generic anchor is required: `True`.

Authorized next: `WP10c9d6c7c3b5c4e10_fine_generic_anchor_manifest_only`.

This is not a physical failure. Fifty-millisecond propagation, fixed-Q experiments, and reduced slow evolution remain blocked.
