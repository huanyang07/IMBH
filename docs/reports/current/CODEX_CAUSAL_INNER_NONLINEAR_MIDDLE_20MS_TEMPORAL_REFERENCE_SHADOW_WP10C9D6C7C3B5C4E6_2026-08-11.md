# Middle 20 ms temporal-reference shadow WP10c9d6c7c3b5c4e6

## Classification

`middle_20ms_response_temporal_reference_hardened_cost_bounded_fine_manifest_authorized`

This package runs only the frozen response-specific `16.0 -> 16.4 ms` interior shadows. It changes no operator and executes no fine propagation.

## Temporal-to-spatial ratios

- `state`: `5.081947e-04` (gate `<= 0.10`).
- `instantaneous_extraction`: `1.226353e-04` (gate `<= 0.10`).
- `cumulative_extraction`: `6.453466e-04` (gate `<= 0.10`).
- `window_mean_extraction`: `1.933259e-02` (gate `<= 0.10`).

Temporal reference hardened: `True`.

Authorized next: `cost_bounded_fine_20ms_spatial_certificate_manifest_only`.

A pass authorizes only a cost-bounded fine definitions manifest. Fine propagation, 50 ms evolution, fixed-Q experiments, and reduced slow evolution remain blocked.
