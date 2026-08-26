# Entropy-complete structure-preserving macro-integrator manifest

Classification: `entropy_complete_exact_affine_macro_integrator_bounded_pilot_manifest_frozen`.

The certified 80D macro field is affine inside one local chart. The selected integrator therefore uses a precomputed augmented matrix exponential, with exact time-integrated face fluxes and sources for the M/J/E ledgers.

The binding pilot is four 1 ms steps (4 ms total), an arbitrary-step checkpoint and two-step bitwise suffix replay, one same-horizon semigroup comparison, and one full physical truth audit at the endpoint. The pilot stops at chart coordinate 0.12, below the certified 0.15 atlas boundary.

This package does not claim that one patch covers a cycle; a pass authorizes only pathwise offline patch expansion.

Authorized next: `WP10c9d6c7c3b5c4f25fizfk_entropy_complete_structure_preserving_macro_integrator_implementation` only.
