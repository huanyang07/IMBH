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

Each case contains `provenance.json` and `SHA256SUMS.txt`. The global file list
and hashes are recorded in `manifests/canonical_artifacts.csv`.

These artifacts establish only the claims stated in their provenance. In
particular, they do not certify a physical mass-loaded-wind branch or global
nonexistence of a steady far-side solution.
