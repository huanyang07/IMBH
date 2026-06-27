# Smooth-a Sonic Micro-Domain Scan

Generated while testing a dedicated inner micro-domain in `scripts/run_transonic_two_domain_sonic_flowmap.py`.

All solve rows use branch 0, `epsilon_buf=0.02`, `R_match=6500 rg`, the smooth-a L'Hopital residual, and the mixed global seed where only the buffer point is forced to the sonic flow-map value.

| checkpoint | N regular | R micro/rg | N micro | physical | dominant | micro R | micro E | post-micro R | post-micro E | flow | L | a |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| smooth_eps0.02_flowmap_branch0_N16_mr20_nm8_0p90277664.npz | 16 | 20 | 8 | 0.132321 | micro_R | 0.132321 | 0.013795 | 0.032657 | 0.025780 | 0.048341 | -0.002215 | -321.298 |
| smooth_eps0.02_flowmap_branch0_N16_mr30_nm8_0p90277664.npz | 16 | 30 | 8 | 0.120041 | micro_R | 0.120041 | 0.014861 | 0.003835 | 0.007683 | 0.041512 | -0.002389 | -322.753 |
| smooth_eps0.02_flowmap_branch0_N16_mr30_nm10_0p90277664.npz | 16 | 30 | 10 | 0.130652 | micro_R | 0.130652 | 0.015500 | 0.001458 | 0.008136 | 0.051668 | -0.001222 | -271.922 |
| smooth_eps0.02_flowmap_branch0_N32_mr30_nm16_0p90277664.npz | 32 | 30 | 16 | 0.149252 | micro_R | 0.149252 | 0.014494 | 0.041775 | 0.059074 | 0.063614 | -0.002035 | -323.098 |

For comparison, the best single-block stretched grid from `transonic_two_domain_sonic_flowmap_grid_scan.md` was `N=16`, `inner_grid_power=1.25`, with physical residual `0.102626`.

Residual localization for the best micro-domain row shows the remaining floor is the first micro interval:

| row | interval | R range/rg | radial residual | energy residual |
|---|---:|---|---:|---:|
| N16, Rmicro=30, Nmicro=8 | 0 | 5.5671-6.8717 | -0.120041 | 0.003829 |
| N16, Rmicro=30, Nmicro=8 | 3 | 10.470-12.923 | 0.077464 | 0.014861 |
| N16, Rmicro=30, Nmicro=8 | 2 | 8.4821-10.470 | 0.068415 | 0.014131 |
| N32, Rmicro=30, Nmicro=16 | 0 | 5.5409-6.1578 | -0.149252 | 0.004877 |
| N32, Rmicro=30, Nmicro=16 | 6 | 10.439-11.601 | 0.073444 | 0.014494 |

I also tested a diagnostic seed mode, `IMBH_FLOWMAP_MICRO_SEED_FLOWMAP=1`, which fills every micro-domain node from the regular ODE flow-map instead of only the first buffer node. This produced very large post-micro mismatch even for `R_micro=10 rg`, with seed physical residual about `21.47`, and for `R_micro=30 rg`, with seed physical residual about `44.17`. That indicates the exact local sonic-flow-map branch diverges rapidly from the previous global basin once integrated beyond the buffer.

Conclusion: the multi-block grid successfully isolates the near-sonic problem, but it does not remove the residual floor. The current bottleneck is not just inadequate near-sonic point placement. It is the compatibility between the offset sonic flow-map branch and the outer/global branch, concentrated in the first post-buffer radial equation.
