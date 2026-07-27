# Canonical Results

This directory contains the compact evidence required for scientific review
and regression testing. It replaces thousands of raw continuation artifacts in
the default branch.

| Case | Status | Purpose |
|---|---|---|
| `no_wind_mdot5` | **CERTIFIED** | Standard no-wind N768 Mdot/Edd=5 anchor |
| `stream_no_wind_mdot2_fs080` | **SUPPORTED BUT NOT FULLY CERTIFIED** | N896 compact stream-fed no-wind anchor |
| `phase_dae_entry_N164` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Global phase-DAE entry/interface state |
| `phase_endpoint_positive_N164` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Compact mathematical endpoint tail and validity audit |
| `phase_endpoint_step_convergence` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Step and bordered-continuation comparison |
| `source_shape_comparison` | **SUPPORTED BUT NOT FULLY CERTIFIED** | C2/C4/C-infinity/wider-source comparison |
| `global_composite_failure` | **REJECTED** | Decisive failed global-tail witness |
| `p0_validity_ledger_outer_manifold` | **DIAGNOSTIC ONLY** | Validity, angular ledger, and outer-manifold review |
| `signed_flux_legacy_53566fa_N512` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Frozen mass-only angular wall/open controls |
| `signed_flux_angular_closed_wp1_N512` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Angularly closed wall/open decision-gate states |
| `signed_flux_total_energy_near_isco_failure` | **REJECTED** | Corrected enthalpy-work N256/N512 failure at the invalid near-ISCO boundary |
| `signed_flux_total_energy_rin10_N512` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Corrected enthalpy-work wall/open interface controls |
| `inner_outer_overlap_audit` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Primary and pressure-sensitivity overlap bands for the transonic and reservoir controls |
| `two_domain_interface_sweep` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Mesh and interface-position sweep of the conservative transonic-to-wall composite |
| `pressure_supported_interface_pilot` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Coarse-grid-only projected pressure-support continuation and N128 failure evidence |
| `common_stress_simultaneous_reservoir` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Shared-stress fixed-Keplerian gate and simultaneous non-Keplerian `30-60 rg` decision evidence |
| `coupled_inner_outer_rank_prototype` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | First square full-rank coupled root at `40.0415 rg` |
| `coupled_mesh_interface_certification` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Chained mesh convergence and full-rank `35-50 rg` numerical-interface invariance |
| `coupled_wall_pattern_power` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **REJECTED** physically | Finite-minidisk paired torque/power continuation; tidal-band thickness invalidates perfect confinement |
| `coupled_open_overflow_eigenvalue` | **SUPPORTED BUT NOT FULLY CERTIFIED** numerically; **DIAGNOSTIC ONLY** physically | Full-rank open roots at `96/64` and `144/96`, plus retained `168/112` endpoint-refinement failure |
| `causal_inner_interface_fluctuation_wp10c9d4a` | **CERTIFIED** for its production-neutral fixed-geometry method scope; **DIAGNOSTIC ONLY** physically | Interface-inclusive quadratic-reconstruction, independent manufactured-wave, and all-family Fourier-symbol gate before radial well balance |
| `causal_inner_radial_fluctuation_wp10c9d4b` | **CERTIFIED** for its production-neutral radial method scope; **DIAGNOSTIC ONLY** physically | Nonuniform-measure radial five-field candidate, complete source ledger, independent manufactured balance, outgoing excision, and candidate FD/assembled stationary-Jacobian gate before frozen-linear export discrimination |
| `causal_inner_frozen_discrimination_wp10c9d5` | **REJECTED** | Same-descriptor production/candidate A/B generators and parity-certified physical-export ladders; aggregate embedded exports improve, but inner and net M/J/E remain nonconvergent |

Each case contains `provenance.json` and `SHA256SUMS.txt`. The global file list
and hashes are recorded in `manifests/canonical_artifacts.csv`.

These artifacts establish only the claims stated in their provenance. In
particular, they do not certify a physical mass-loaded-wind branch or global
nonexistence of a steady far-side solution.
