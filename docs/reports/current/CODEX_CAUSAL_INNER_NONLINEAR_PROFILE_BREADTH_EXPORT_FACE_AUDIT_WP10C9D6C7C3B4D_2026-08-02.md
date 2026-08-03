# Nonlinear held-out export-face audit WP10c9d6c7c3b4d

## Classification

`heldout_spatial_export_failure_caused_by_active_face_alias_corrected_physical_face_contract_passes`

No trajectory was propagated and no evolution operator or gate was changed. The failed b4b3 and diagnostic b4c classifications remain historical facts.

## Root cause

The shared export-history helper passed parent face 48 to every active layout. That is the physical coupling face only on the coarse grid. The intended active-grid faces are 48, 96 and 192.

- `N128_exterior_N128_inner_c48`: correct face `48`, correct radius `1.886776591e+10`, legacy face-48 radius `1.886776591e+10`
- `N128_exterior_N256_inner_c48`: correct face `96`, correct radius `1.886776591e+10`, legacy face-48 radius `7.081710563e+09`
- `N128_exterior_N512_inner_c48`: correct face `192`, correct radius `1.886776591e+10`, legacy face-48 radius `4.338574101e+09`

## Corrected frozen-contract result

- every corrected instantaneous/cumulative profile passed: `True`
- minimum alias/error alignment: `0.999999998875`
- maximum corrected/legacy error ratio: `5.644478e-05`
- maximum corrected ledger defect: `1.116e-10`

### `p4__inward_acoustic`

- instantaneous RMS/component order, cosine: `3.825977` / `0.863573` / `0.997493436`
- cumulative RMS/component order, cosine: `3.824521` / `0.871797` / `0.997515072`

### `p4__outward_acoustic`

- instantaneous RMS/component order, cosine: `3.848456` / `1.552560` / `0.997478624`
- cumulative RMS/component order, cosine: `3.847152` / `1.552009` / `0.997496988`

### `p3_buffer45__material`

- instantaneous RMS/component order, cosine: `5.026099` / `2.001378` / `0.996923758`
- cumulative RMS/component order, cosine: `5.014138` / `2.002630` / `0.997731873`

### `p4__inward_shear_acoustic_mix`

- instantaneous RMS/component order, cosine: `3.825509` / `2.002939` / `0.997946438`
- cumulative RMS/component order, cosine: `3.824044` / `2.002940` / `0.997966334`

### `p3_buffer45__generic_five_field`

- instantaneous RMS/component order, cosine: `4.946160` / `2.001473` / `0.964475434`
- cumulative RMS/component order, cosine: `4.935595` / `2.001457` / `0.965921032`

## Decision

`WP10c9d6c7c3b5a_variable_step_duration_controller_manifest`

The held-out nonlinear spatial state and Tier-I physical-export breadth contract is now certified under its originally intended physical-face definition. A definitions-only variable-step duration controller is authorized next. Fixed-Q and reduction remain blocked.
