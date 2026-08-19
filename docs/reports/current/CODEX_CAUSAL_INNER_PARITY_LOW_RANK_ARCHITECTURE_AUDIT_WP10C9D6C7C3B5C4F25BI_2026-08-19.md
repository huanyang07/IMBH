# Parity low-rank architecture audit WP10c9d6c7c3b5c4f25bi

## Classification

`quadratic_cubic_low_rank_departure_architecture_diagnosed_mixed_direction_database_manifest_authorized`

## Diagnosis

This is a post-result architecture diagnosis, not independent validation. It performs no new truth evaluation, retraction, root, or propagation.

The median even/quadratic relative signal grows from `9.166033e-02` at 0.005 to `1.774380e-01` at 0.01. The median odd/cubic signal at 0.01 is `1.670223e-02`.

Rank-3 captures `9.788638e-01` of balanced quadratic output energy; rank-4 captures `9.636836e-01` of balanced cubic output energy.

The selected candidate keeps the exact 162 physical and 280 stable-memory updates, and models only the 28D departure nonlinearity from an active 8D input with low-rank quadratic and cubic outputs. The compressed full-polynomial upper bound is 588 coefficients before any cubic input-tensor compression.

Authorized next artifact: `definitions_only_active8_mixed_direction_parity_database_manifest`. Mixed-direction coefficients, an online integrator, and a predictive cycle remain unvalidated.
