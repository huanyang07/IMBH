# Nonlinear held-out spatial-export localization WP10c9d6c7c3b4c

## Classification

`spatial_failure_localized_to_layout_native_export_map_common_parent_map_passes`

No trajectory was propagated and no operator was changed. The committed coarse, middle and fine full states were conservatively restricted to the same 64-cell parent grid and evaluated through one common coarse-layout nonlinear Tier-I export map.

## Discriminating result

- common-parent export contract passed: `True`
- localized to layout-native export map: `True`
- minimum map-error alignment with native refinement error: `1.000000000`
- maximum common-state fraction of native refinement error: `0.000001`
- maximum decomposition closure defect: `0.000e+00`

## Profiles

### `p4__inward_acoustic`

- native RMS/component order and error cosine: `0.281273` / `0.186662` / `-0.999516241`
- common-map RMS/component order and error cosine: `3.986690` / `1.955564` / `0.998423051`
- fine-pair map alignment / common-state fraction: `1.000000000` / `0.000000`

### `p4__outward_acoustic`

- native RMS/component order and error cosine: `0.279431` / `0.180376` / `-0.999548273`
- common-map RMS/component order and error cosine: `4.024056` / `1.972490` / `0.998300558`
- fine-pair map alignment / common-state fraction: `1.000000000` / `0.000000`

### `p3_buffer45__material`

- native RMS/component order and error cosine: `0.504223` / `0.322938` / `-0.998231866`
- common-map RMS/component order and error cosine: `4.003310` / `1.950377` / `0.999985038`
- fine-pair map alignment / common-state fraction: `1.000000000` / `0.000000`

### `p4__inward_shear_acoustic_mix`

- native RMS/component order and error cosine: `0.286047` / `0.190953` / `-0.999573356`
- common-map RMS/component order and error cosine: `3.986890` / `1.933082` / `0.999448128`
- fine-pair map alignment / common-state fraction: `1.000000000` / `0.000000`

### `p3_buffer45__generic_five_field`

- native RMS/component order and error cosine: `0.504697` / `0.037905` / `-0.998236515`
- common-map RMS/component order and error cosine: `4.003446` / `1.858716` / `0.999980943`
- fine-pair map alignment / common-state fraction: `1.000000000` / `0.000000`

## Interpretation and next package

`WP10c9d6c7c3b4d_layout_native_export_map_audit`

The failed b4b3 classification is preserved. Duration extension, fixed-Q experiments and reduced slow evolution remain blocked. Only the evidence-selected export-map audit is authorized next.
