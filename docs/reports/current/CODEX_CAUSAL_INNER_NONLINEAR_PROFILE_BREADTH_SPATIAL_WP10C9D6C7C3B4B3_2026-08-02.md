# Nonlinear held-out profile spatial confirmation WP10c9d6c7c3b4b3

## Classification

`heldout_profile_spatial_confirmation_failed_duration_extension_blocked`

The five prospectively frozen held-outs were evolved independently on the middle and fine embedded layouts at `dt=1e-5 s` through the common `4e-5 s` horizon. The committed coarse and unperturbed histories were reused by hash.

## Binding results

### `p4__inward_acoustic`

- state RMS/max/component order: `1.994342` / `2.019125` / `1.959896`
- state fine difference / history / error cosine: `3.477e-09` / `1.000000000` / `0.994459220`
- instantaneous export RMS/max/component order: `0.281273` / `0.302616` / `0.186662`
- cumulative export RMS/max/component order: `0.281242` / `0.302813` / `0.186641`
- result: `fail`

### `p4__outward_acoustic`

- state RMS/max/component order: `2.001092` / `2.015264` / `1.992160`
- state fine difference / history / error cosine: `3.027e-09` / `1.000000000` / `0.995920365`
- instantaneous export RMS/max/component order: `0.279431` / `0.298476` / `0.180376`
- cumulative export RMS/max/component order: `0.279404` / `0.298649` / `0.180358`
- result: `fail`

### `p3_buffer45__material`

- state RMS/max/component order: `1.990767` / `2.003287` / `1.978818`
- state fine difference / history / error cosine: `2.774e-09` / `1.000000000` / `0.993496186`
- instantaneous export RMS/max/component order: `0.504223` / `0.544823` / `0.322938`
- cumulative export RMS/max/component order: `0.504180` / `0.545107` / `0.322910`
- result: `fail`

### `p4__inward_shear_acoustic_mix`

- state RMS/max/component order: `2.001727` / `2.016849` / `1.953814`
- state fine difference / history / error cosine: `2.110e-09` / `1.000000000` / `0.991832835`
- instantaneous export RMS/max/component order: `0.286047` / `0.304758` / `0.190953`
- cumulative export RMS/max/component order: `0.286016` / `0.304957` / `0.190932`
- result: `fail`

### `p3_buffer45__generic_five_field`

- state RMS/max/component order: `1.999279` / `2.006661` / `1.983599`
- state fine difference / history / error cosine: `2.254e-09` / `1.000000000` / `0.992110871`
- instantaneous export RMS/max/component order: `0.504697` / `0.545433` / `0.037905`
- cumulative export RMS/max/component order: `0.504654` / `0.545718` / `0.037892`
- result: `fail`

## Method

- maximum scaled residual: `1.709e-11`
- maximum discrete ledger defect: `0.000e+00`
- all checkpoint roundtrips bitwise: `True`
- all split/restart replays bitwise: `True`

## Authorized next

`WP10c9d6c7c3b4b3_spatial_failure_localization`

This completes the frozen short-horizon breadth comparison as a negative spatial-export result. Meaningful-duration evolution, fixed-Q experiments and reduced slow evolution remain blocked. The next package must localize the export-only error rotation from the existing histories before any duration controller can be frozen.
