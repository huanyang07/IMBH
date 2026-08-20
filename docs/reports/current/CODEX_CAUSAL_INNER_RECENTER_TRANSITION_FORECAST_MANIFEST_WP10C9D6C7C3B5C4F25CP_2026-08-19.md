# Recenter-transition forecast manifest WP10c9d6c7c3b5c4f25cp

## Classification

`direct_field_recenter_transition_forecast_frozen`

The direct field retrospectively predicts warm-3→warm-4 with full/q/z/a relative errors `3.138777e-03`, `4.082702e-02`, `2.265311e-04`, and `4.361286e-03`.

From accepted warm-4 it predicts old-chart loads `8.546615e-03` and `1.326858e-02` after one and two `1e-7 s` intervals. Thus the frozen forecast places the first recenter trigger in the second interval, below the `0.015` hard limit.

Exactly two fail-fast authentic BDF2 roots are authorized. A predicted state may not become a chart center; only the first accepted authentic state satisfying the frozen trigger may do so.

No physical microburst, predictive cycle, or reduced slow evolution is authorized.
