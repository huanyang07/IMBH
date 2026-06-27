# Smooth-a Flow-Map Grid Scan

Generated while testing the offset sonic flow-map BVP with the branch coordinate `a` promoted to an unknown.

All rows use branch 0, `epsilon_buf=0.02`, `R_match=6500 rg`, and the smooth-a L'Hopital residual.

| checkpoint | N regular | inner grid power | remap | physical | dominant | regular R | regular E | flow | L | a |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| smooth_eps0.02_flowmap_branch0_N16_0p90277664.npz | 16 | 1 | pchip | 0.155404 | regular_R | 0.155404 | 0.016767 | 0.036306 | -0.002609 | -296.523 |
| smooth_eps0.02_flowmap_branch0_N16_ip1p25_0p90277664.npz | 16 | 1.25 | pchip | 0.102626 | regular_R | 0.102626 | 0.019926 | 0.031844 | -0.004617 | -328.440 |
| smooth_eps0.02_flowmap_branch0_N16_ip1p5_0p90277664.npz | 16 | 1.5 | pchip | 0.123249 | regular_R | 0.123249 | 0.023165 | 0.048626 | -0.001813 | -331.590 |
| smooth_eps0.02_flowmap_branch0_N16_ip2_0p90277664.npz | 16 | 2 | pchip | 0.162495 | regular_R | 0.162495 | 0.027027 | 0.119353 | -0.006844 | -324.138 |
| smooth_eps0.02_flowmap_branch0_N32_ip1p25_0p90277664.npz | 32 | 1.25 | pchip | 0.139605 | regular_R | 0.139605 | 0.022816 | 0.054255 | -0.009309 | -328.633 |
| smooth_eps0.02_flowmap_branch0_N32_ip1p5_0p90277664.npz | 32 | 1.5 | pchip | 0.167355 | regular_R | 0.167355 | 0.029764 | 0.095064 | -0.004514 | -332.000 |
| smooth_eps0.02_flowmap_branch0_N32_ip1p5_hermite_0p90277664.npz | 32 | 1.5 | hermite | 0.167355 | regular_R | 0.167355 | 0.029764 | 0.095064 | -0.004514 | -332.000 |

Conclusion: moderate inner stretching improves the N16 result, with the best tested point at `inner_grid_power=1.25`. Plain N32 refinement is not monotone and remains controlled by the near-sonic radial residual, so the next numerical step should be a dedicated sonic micro-domain or multi-block inner grid rather than brute-force uniform refinement.
