# Active-8 tensor architecture diagnosis WP10c9d6c7c3b5c4f25bm

## Classification

`active8_kernel_failure_diagnosed_full_tensor_rate_and_rank4_slaved_curvature_architecture_selected`

The 40-center quadratic/cubic plus 90-output cubic/quartic kernel model remains rejected. All prior tuning and holdout data are now revealed and are used only for post-result diagnosis.

A train-only rank-4 odd curvature subspace captures `0.99295292` of hidden snapshot energy. Its revealed validation capacity has maximum full-state error `5.853829e-04` and C_phys residual `1.221461e-07`.

Naively evolving these curvature modes is rejected: the augmented memory projection has `3` unstable eigenvalues and spectral abscissa `5.157958e+03 s^-1`. They must remain an algebraic/slaved decoder.

The replacement closure uses complete homogeneous quadratic/cubic tensors of dimensions 36/120. Retaining 56 revealed directions and adding 64 new training directions gives cubic rank `120` and condition `19.636142`. Fresh validation remains separated by `0.275886`.

The next definitions-only extension contains `192` new signed exact-geometry/rate candidates. No predictive trajectory, cycle, or reduced slow evolution is authorized.
