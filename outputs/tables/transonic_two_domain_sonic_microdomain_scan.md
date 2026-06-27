# Two-Domain Sonic Micro-Domain Scan

Generated during the sonic micro-domain implementation sprint.

All cases start from the dynamic-patch N64 checkpoint at Mdot/Edd = 0.90277664 and solve only the Nreg64 micro-domain system unless noted. The reported physical residual is the audit residual, i.e. it includes physical differential residuals and the unused sonic compatibility diagnostics.

| case | solve form | Delta_s | N micro | seed physical | polish physical | dominant after polish | note |
|---|---|---:|---:|---:|---:|---|---|
| direct wide buffer | differential | 0.05 | 4 | 4.323 | 1.685e-3 | micro_R | Directly widening the sonic buffer gives a poor extrapolated seed and does not recover. |
| baseline micro | differential | 0.02 | 4 | 2.559e-2 | 4.273e-4 | micro_R | Square `D,K` sonic rows help only slightly; the micro radial residual sets the floor. |
| fewer micro intervals | differential | 0.02 | 2 | 6.999e-4 | 3.861e-4 | micro_R | Better seed, but essentially the same residual floor. |
| one micro interval | differential | 0.02 | 1 | 2.409e-2 | 2.626e-4 | C2 | Best N64 scan, but the unused adjugate compatibility becomes the limiting residual. |
| integrated micro residual | integrated_dx | 0.02 | 2 | 6.999e-4 | 3.862e-4 | micro_R | Same result as differential form, so the bottleneck is not just midpoint residual algebra. |

Conclusion: a naive midpoint micro-domain is diagnostically useful but is not yet a successful replacement for the dynamic sonic patch. The next fix should use an analytic sonic Taylor/regularity expansion, or introduce a dedicated sonic derivative unknown solved from the singular local system before collocating outward.
