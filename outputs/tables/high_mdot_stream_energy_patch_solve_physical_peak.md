# High-Source Energy Patch Solve

Checkpoint `outputs/checkpoints/high_mdot_stream_outer_buffer_energy_merit_next_diag4_089825_to08985/energy_merit_next_diag4_mass_0p8985_torque_0p005_mdot_2_N896.npz`.

Windows `260:3`, top-K `0`, node pad `0`, energy weight `5`, prior weight `0.0001`.

| metric | before | after |
| --- | ---: | ---: |
| `full` | 1.202053e-04 | 8.342648e-06 |
| `physical_E` | 1.202053e-04 | 8.342648e-06 |
| `physical_E_l2` | 5.971099e-06 | 3.960045e-07 |
| `buffer_E` | 4.234194e-03 | 4.234194e-03 |
| `terminal_omega` | 2.732940e-06 | 2.732940e-06 |
| `peak_E_rg` | 3.336831e+02 | 3.336831e+02 |
| `peak_E_value` | 4.234194e-03 | 4.234194e-03 |
| `f_adv_global` | 2.043413e-01 | 2.043413e-01 |
| `f_adv_inner` | 9.443351e-02 | 9.443351e-02 |
| `Lrad_LEdd` | 8.665545e-01 | 8.665545e-01 |

Local patch max residual: `6.010263e-04` -> `1.272621e-09` in `6` function evaluations.

Patched checkpoint `outputs/checkpoints/high_mdot_stream_energy_patch_solve_physical_peak/energy_patch_mass_0p8985_N896.npz`.
