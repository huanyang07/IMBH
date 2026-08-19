# Active-8 tensor database extension manifest WP10c9d6c7c3b5c4f25bn

## Classification

`active8_full_cubic_rank4_curvature_database_extension_manifest_frozen_geometry_authorized`

This prospective package freezes a complete 36-feature quadratic and 120-feature cubic closure for all 28 active departure rates, plus a four-coordinate cubic algebraic curvature decoder. The online dimension remains 470 and both truth calls and Newton retractions remain zero.

The frozen design reuses 56 revealed directions as training, adds 64 training directions, and reserves 8 new tuning plus 16 untouched holdout directions. It contains 192 new signed exact states.

The complete cubic feature matrix has rank 120 and condition number 19.636142. New validation separation from training is 0.275886.

Leakage is fail-closed: training truth is evaluated first, all coefficient maps are hashed, and only then may tuning and holdout responses be read. The holdout cannot alter coefficients, architecture, or gates.

Only the exact-geometry extension is authorized next. No rate validation, reduced trajectory, predictive cycle, or reduced slow evolution is authorized by this manifest.
