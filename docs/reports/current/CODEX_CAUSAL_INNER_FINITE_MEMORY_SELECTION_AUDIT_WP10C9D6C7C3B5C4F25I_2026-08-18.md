# Finite-memory selection audit WP10c9d6c7c3b5c4f25i

## Classification

`compact_finite_memory_failed_larger_conservative_coarse_PDE_fallback_required`

The hash-locked single-anchor R106/stable-454 system was used. No truth root, propagation, or generator assembly was performed.

- r=0: pass=False, max/RMS/DC dynamic error=1.000000e+00/1.000000e+00/1.000000e+00, max total error=9.999388e-01.
- r=2: pass=False, max/RMS/DC dynamic error=1.013067e+00/9.929453e-01/9.833942e-01, max total error=1.013005e+00.
- r=4: pass=False, max/RMS/DC dynamic error=1.071767e+00/9.803591e-01/9.503079e-01, max total error=1.071643e+00.
- r=6: pass=False, max/RMS/DC dynamic error=1.030053e+00/9.840689e-01/9.783735e-01, max total error=9.922150e-01.

Selected memory order: `None`; selected online continuous dimension: `None`. Full-order Gramian numerical pass: `True`.

Authorized next artifact: `definitions_only_larger_conservative_coarse_PDE_manifest`. These coefficients remain single-anchor diagnostics; production coefficients, the online solver, a predictive cycle, and reduced slow evolution are not authorized.
