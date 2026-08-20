# Departure-28 dual-polynomial diagnosis WP10c9d6c7c3b5c4f25bu

## Classification

`departure28_dual_polynomial_architecture_selected_for_independent_validation`

The selected closure uses the full existing 28-dimensional departure state with an even dot-squared kernel and an odd dot-cubed kernel. It adds no dynamic variable.

Revealed 144-to-16 diagnostic validation nonlinear median/max: `9.026677e-03` / `5.562276e-02`; full median/max: `2.024875e-03` / `4.124463e-02`.

Leave-one-direction-out nonlinear median/p95/max: `9.896355e-03` / `5.721568e-02` / `1.745223e-01`; full median/p95/max: `2.144367e-03` / `1.712914e-02` / `4.517848e-02`.

This is diagnostic-only because the architecture was selected after the parent holdout was revealed.

Authorized next artifact: `definitions_only_departure28_dual_polynomial_independent_validation_manifest`. No trajectory, predictive cycle, or reduced slow evolution is authorized.
