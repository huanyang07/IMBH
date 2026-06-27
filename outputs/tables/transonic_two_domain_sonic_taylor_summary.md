# Two-Domain Sonic Taylor Patch Summary

Generated during the sonic Taylor regularity sprint.

## Root Scan

The L'Hopital derivative scan at the original two-domain N65 source found two stable roots across `eps = 3e-4 ... 3e-6`. Repeating the scan with three derivative forms,

```text
scaled:         differentiate the fully scaled residual
frozen_scaled:  differentiate the raw residual, then divide by sonic-point scales
raw:            differentiate the unscaled residual
```

gave the same two roots to useful accuracy:

| branch | representative g_u | representative g_T | distance to source first slope | distance to dynamic-patch slope | interpretation |
|---|---:|---:|---:|---:|---|
| negative-a root | -183.5 | 73.6 | 209 | 227 | Far from the existing branch. |
| positive-a root | 161.3 | -96.1 | 183 | 157 | Closest scanned root, but still far from the existing branch. |

The source first-interval slope is approximately `(18.3, 17.8)`. The dynamic-patch N64 slope is approximately `(19.7, -27.9)`. Thus the finite-difference L'Hopital condition selects derivative roots far from the accepted local branch. Holding the residual scaling fixed or using raw equations does not move the roots back toward the physical branch.

## Taylor Patch Tests

| case | seed | N regular | physical | dominant | selected max | Rson/rg | lambda0 | int adv | g_u | g_T | h_u | h_T | status |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Taylor1 | buffer slope | 64 | 1.720e-3 | local_F | 1.720e-3 | 5.956 | 3.677 | 0.07899 | 20.36 | -27.35 | 0 | 0 | Fails; local sonic equation and compatibility diagnostics remain large. |
| Taylor2 | buffer slope | 64 | 1.590e-3 | L | 1.590e-3 | 5.850 | 3.676 | 0.07704 | 12.43 | -25.78 | 1280 | -713 | Fails; L'Hopital residual remains dominant. |
| Taylor2 | L'Hopital-root slope | 64 | 2.015e-4 | C2 | 1.708e-4 | 5.760 | 3.676 | 0.07449 | 220.1 | -125.6 | -1.86e4 | 1.02e4 | Best Taylor N64 case, but still worse than the dynamic patch. |
| Taylor2 | frozen-scale L'Hopital root | 64 | 2.014e-4 | C2 | ~1.7e-4 | ~5.76 | ~3.676 | ~0.0745 | 219.6 | -125.4 | large | large | Confirms differentiating the scaling is not the cause. |
| Taylor2 | soft `C1,C2,K`, weight 0.2 | 64 | ~2.96e-4 active / 6.77e-3 L audit | midpoint/C2/local_F | ~2.96e-4 | 5.764 | 3.676 | 0.07469 | 20.69 | -28.88 | -69.9 | 19.1 | Avoids the far L'Hopital branch but stalls at a few 1e-4. |
| Taylor2 | soft `C1,C2,K`, weight 0.05 | 64 | 2.88e-4 | midpoint | ~2.88e-4 | ~5.764 | ~3.676 | ~0.0747 | ~20.7 | ~-28.9 | ~-69.8 | ~19 | Similar floor; lowering compatibility weight does not help. |
| Taylor2 | soft `C1,C2,K`, weight 0 | 64 | 2.87e-4 | midpoint | 2.87e-4 | 5.764 | 3.676 | 0.07468 | 20.69 | -28.88 | -69.8 | 19.0 | Limiting floor remains even with compatibility rows disabled. |
| Taylor2 refinement | resume from root-seeded N64 | 80 | 6.259e-3 | regular_R | 6.259e-3 | 5.324 | 3.691 | 0.05052 | 234.1 | -124.5 | -1.42e4 | 7652 | Refinement fails badly; branch is not mesh-stable. |

For comparison, the dynamic first-order patch reached `1.35e-5` at Nreg64 and stayed below `6e-4` through Nreg128, although it remained patch-limited.

## Conclusion

The new helper functions and Taylor experiment confirm that branch selection is the next blocker. The naive finite-difference L'Hopital condition is mathematically plausible, but in scaled, frozen-scale, and raw forms it selects a sonic derivative branch far from the previously accepted solution. Root-seeded Taylor2 reduces the residual compared with buffer-seeded Taylor1/Taylor2, but it still fails the science gate and collapses under Nreg refinement.

The fallback local two-point regularity solve without the L'Hopital row also fails to reach the dynamic-patch quality. Soft compatibility weights `0`, `0.05`, and `0.2` all stall near `2.9e-4`; therefore the remaining floor is not just over-weighted compatibility diagnostics. It appears to be a limitation of the current second-order Taylor/midpoint patch formulation itself.

Delta-s convergence is therefore gated: running a Delta-s ladder on this Taylor branch would not be scientifically meaningful until the L'Hopital/compatibility branch issue is fixed.
