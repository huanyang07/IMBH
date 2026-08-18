# Saved-R32 memory-selection audit WP10c9d6c7c3b5c4f25m

## Classification

`single_anchor_R32_global_balanced_order_96_selected_cross_anchor_preflight_authorized`

The hash-locked R180/stable-380 saved system was used. No truth root, propagation, new generator assembly, or new truth anchor was executed.

- `global_balanced_r24`: pass `False`; training max/RMS dynamic `3.719955e-01/1.685440e-01`; held-out max/RMS dynamic `3.719929e-01/1.622348e-01`.
- `coherent_channels_s1_r24`: pass `False`; training max/RMS dynamic `5.659006e-01/3.621440e-01`; held-out max/RMS dynamic `5.658008e-01/3.538193e-01`.
- `coherent_channels_s2_r24`: pass `False`; training max/RMS dynamic `4.722354e-01/2.486828e-01`; held-out max/RMS dynamic `4.722343e-01/2.400997e-01`.
- `coherent_channels_s3_r24`: pass `False`; training max/RMS dynamic `3.618601e-01/1.824933e-01`; held-out max/RMS dynamic `3.618584e-01/1.753543e-01`.
- `global_balanced_r48`: pass `False`; training max/RMS dynamic `3.780373e-01/1.699220e-01`; held-out max/RMS dynamic `3.780345e-01/1.634236e-01`.
- `coherent_channels_s1_r48`: pass `False`; training max/RMS dynamic `5.647613e-01/3.615848e-01`; held-out max/RMS dynamic `5.640460e-01/3.533045e-01`.
- `coherent_channels_s2_r48`: pass `False`; training max/RMS dynamic `4.717053e-01/2.485565e-01`; held-out max/RMS dynamic `4.717044e-01/2.399844e-01`.
- `coherent_channels_s3_r48`: pass `False`; training max/RMS dynamic `3.583958e-01/1.811803e-01`; held-out max/RMS dynamic `3.583942e-01/1.740980e-01`.
- `global_balanced_r96`: pass `True`; training max/RMS dynamic `3.851151e-02/1.596679e-02`; held-out max/RMS dynamic `3.850932e-02/1.524674e-02`.
- `coherent_channels_s1_r96`: pass `False`; training max/RMS dynamic `5.630828e-01/3.608717e-01`; held-out max/RMS dynamic `5.623890e-01/3.526149e-01`.
- `coherent_channels_s2_r96`: pass `False`; training max/RMS dynamic `4.543197e-01/2.418939e-01`; held-out max/RMS dynamic `4.543195e-01/2.335995e-01`.
- `coherent_channels_s3_r96`: pass `False`; training max/RMS dynamic `3.278085e-01/1.684709e-01`; held-out max/RMS dynamic `3.278070e-01/1.618160e-01`.

Selected model: `global_balanced_r96` with online continuous dimension `276`. Cumulative Hankel-value orders at 90/95/99/99.9 percent are `15/21/38/87`.

The coherent rank-1/2/3 hypotheses are evaluated rather than inferred from per-frequency matrix rank. The selected single-anchor coefficients remain diagnostic only.

Authorized next artifact: `definitions_only_common_resolved_subspace_cross_anchor_preflight_manifest`. Production coefficients, an online solver, a predictive cycle, and reduced slow evolution remain blocked.
