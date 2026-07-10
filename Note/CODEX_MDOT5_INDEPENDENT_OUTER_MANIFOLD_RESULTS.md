# Mdot=5 independent outer-manifold search

The outer atlas is initialized only from the pre-phase global checkpoint. No accepted inner phase state or tangent is used to construct an outer seed.

Physical matching surface: `R=223.236427 rg`.

## Seed atlas

- Local seed states surveyed: `150`.
- Accepted tangent roots: `146`.
- States with multiple distinct tangent roots: `0`.

## Continued trajectories

| label | start R | steps | minimum R | final R | final p_R | status | scale-validity crossed | match class |
|---|---:|---:|---:|---:|---:|---|---|---|
| `R330` | 328.491 | 180 | 274.537 | 274.537 | -4.127e-01 | `max_steps` | True | `-` |
| `R300` | 297.509 | 8 | 297.504 | 297.504 | -2.932e-11 | `solver_failure` | True | `-` |
| `R270` | 269.005 | 11 | 268.998 | 268.998 | -4.214e-12 | `radial_stagnation_before_match` | True | `-` |
| `R250` | 247.869 | 77 | 247.667 | 247.755 | 1.063e-02 | `radial_turn_before_match` | True | `-` |
| `R235` | 232.936 | 66 | 223.236 | 223.236 | -3.003e-01 | `reached_validity_surface` | True | `distinct_sheet_at_validity_surface` |
| `R230` | 228.126 | 24 | 223.236 | 223.236 | -2.282e-01 | `reached_validity_surface` | True | `distinct_sheet_at_validity_surface` |
| `R230_u_minus` | 228.126 | 21 | 223.236 | 223.236 | -7.835e-01 | `reached_validity_surface` | False | `distinct_sheet_at_validity_surface` |
| `R230_u_minus_0525` | 228.126 | 24 | 223.236 | 223.236 | -4.771e-01 | `reached_validity_surface` | True | `exploratory_near_match` |
| `R230_T_plus` | 228.126 | 178 | 227.850 | 227.850 | -3.361e-06 | `max_steps` | True | `-` |
| `R230_u044_F0042` | 228.126 | 23 | 223.236 | 223.236 | -4.737e-01 | `reached_validity_surface` | True | `exploratory_near_match` |
| `R230_u044_F0045` | 228.126 | 24 | 223.236 | 223.236 | -4.658e-01 | `reached_validity_surface` | True | `exploratory_near_match` |
| `R230_u044_F0048` | 228.126 | 24 | 223.236 | 223.236 | -4.582e-01 | `reached_validity_surface` | True | `exploratory_near_match` |
| `R230_u045_F0045` | 228.126 | 24 | 223.236 | 223.236 | -4.537e-01 | `reached_validity_surface` | True | `exploratory_near_match` |
| `R230_u044_Tm0005_F0045` | 228.126 | 24 | 223.236 | 223.236 | -4.160e-01 | `reached_validity_surface` | True | `exploratory_near_match` |
| `R230_u04418_F004501` | 228.126 | 24 | 223.236 | 223.236 | -4.636e-01 | `reached_validity_surface` | True | `exploratory_near_match` |

## Best conservative match

- trajectory: `R230_u04418_F004501`
- state delta `(logu, logT, F)`: `[1.77057803e-03 1.77045448e-03 1.04193793e-05]`
- maximum state mismatch: `1.770578e-03`
- flux delta: `{'F': 1.0419379309567489e-05, 'angular_flux_scaled': 2.5653421792948272e-06, 'advected_internal_energy_scaled': 3.006551546595699e-06}`
- maximum flux mismatch: `1.041938e-05`

## Local shooting-map audit

- singular values: `[2.85640867e+01 9.96448700e-01 3.26797162e-05]`
- condition number: `8.740617e+05`
- velocity/temperature direction cosine: `0.999999993`

The near-collinearity of the velocity and temperature shooting directions is retained as a geometric diagnostic; a small flux residual alone is not promoted to a strict state connection.

## Interpretation

An independently seeded trajectory reaches an exploratory near-match at the validity surface, but strict state/flux matching is not achieved.

This is a topology search under the exact algebraic representation closure. It is not a physical stream/wind angular-momentum certification.

## Files

- summary: `outputs/tables/m5_eta_independent_outer_manifold_98p125_N164.json`
- profiles: `outputs/tables/m5_eta_independent_outer_manifold_98p125_N164_profiles.json`
- figure: `outputs/figures/m5_eta_independent_outer_manifold_98p125_N164.png`
