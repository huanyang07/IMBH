# Q3 absolute-baseline and observable-memory screen

## Result

Classification: `absolute_extraction_baseline_direction_gate_failed_observable_memory_propagation_not_executed`.

The fail-fast absolute audit stopped the package before any middle or fine memory propagation. No nonlinear trajectory or fixed-Q evolution ran.

The absolute state and exact mapped Q3 storage converge at about second order. The extraction-surface flux, cooling, and responsive-height components also have consistent refinement directions. The coupling-face M/J/E flux and the derived net-drive components contract in magnitude but reverse direction between the coarse-middle and middle-fine pairs, so the predeclared absolute-baseline cosine gate fails.

## Instantaneous component localization

| Component | Order | Cosine | Pass |
|---|---:|---:|---:|
| cooling_angular_momentum | 1.998716 | 0.999996 | True |
| cooling_killing_energy | 1.996829 | 1.000000 | True |
| inner_flux_angular_momentum | 2.024691 | 0.999483 | True |
| inner_flux_killing_energy | 2.033038 | 0.999443 | True |
| inner_flux_mass | 2.072142 | 0.997490 | True |
| interface_flux_angular_momentum | 1.687893 | -0.826458 | False |
| interface_flux_killing_energy | 1.599575 | -0.822788 | False |
| interface_flux_mass | 1.574101 | -0.823332 | False |
| net_drive_angular_momentum | 1.791372 | -0.825783 | False |
| net_drive_killing_energy | 1.803102 | -0.824212 | False |
| net_drive_mass | 1.631733 | -0.826303 | False |
| vertical_work_angular_momentum | 1.998648 | 0.999997 | True |
| vertical_work_killing_energy | 1.996857 | 1.000000 | True |

## Decision

The certified 20 ms response result remains valid, but an absolute slow closure is not authorized. The next package must freeze either a fine-anchored absolute-baseline decomposition or a focused coupling-face baseline localization; it may not relax this result after inspection.
