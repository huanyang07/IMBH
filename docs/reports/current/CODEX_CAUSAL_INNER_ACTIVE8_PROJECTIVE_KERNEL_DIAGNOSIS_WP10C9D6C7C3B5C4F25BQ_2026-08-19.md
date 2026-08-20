# Active-8 projective-kernel diagnosis WP10c9d6c7c3b5c4f25bq

## Classification

`active8_projective_even_kernel_cubic_odd_architecture_selected_for_new_independent_validation`

The parent tensor rejection is preserved. All 192 exact truth responses remain accepted; only worst-case closure errors failed.

A frozen diagnostic architecture replaces the unregularized even tensor by an inverse-square norm-weighted projective kernel `(d_i.d_j)^2 + (d_i.d_j)^4/320` with Tikhonov regularization `1/64`. The odd cubic tensor and rank-4 algebraic curvature decoder remain unchanged.

Revealed tuning nonlinear median/max: `3.952177e-02` / `2.456362e-01`.

Revealed holdout nonlinear median/max: `3.219734e-02` / `2.321978e-01`.

Because those sets informed architecture selection, this is diagnostic evidence only. A newly generated untouched holdout is binding.

Authorized next artifact: `definitions_only_active8_projective_kernel_independent_validation_manifest`. No trajectory or reduced slow evolution is authorized.
