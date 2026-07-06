# GPT Prompt: Rectangular Source-Band Collocation Next Steps

Please review the GitHub repo:

```text
https://github.com/huanyang07/IMBH
```

Focus on these latest handoff/result files:

```text
Note/CODEX_MDOT5_LOCAL_MDOT_ETA100_COMPACT_CERTIFICATION.md
Note/CODEX_MDOT5_LOCAL_MDOT_ETA95_ETA90_RESULTS.md
Note/CODEX_MDOT5_SPLIT_SOURCE_BAND_COLLOCATION_RESULTS.md
Note/CODEX_MDOT5_RECTANGULAR_SOURCE_BAND_RESULTS.md
```

Relevant implementation files:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
src/imri_qpe/layer3_minidisk_1d/transonic_collocation.py
tests/test_transonic_collocation.py
```

Current status:

```text
1. The Mdot_inner/Edd=5, eta_E=100/95/90 local-Mdot branch is strict under
   the original midpoint differential residual at N168.

2. The eta_E=90 checkpoint is not representation-robust.  Source-band split
   diagnostics expose a compact-source-annulus subcell defect around
   R~235-255 rg.

3. New opt-in interval residual forms were added:
      split_differential
      split_rms_differential

4. The old midpoint-strict eta_E=90 checkpoint has:
      base_final_full ~ 6.53e-6
      split_differential residual ~ 1.48e-1
      split_rms residual ~ 1.05e-1

5. Split-RMS global polishing improves but stalls:
      N168 best ~ 2.63e-2
      N200 polish ~ 2.11e-2

6. A rectangular overdetermined source-band residual was implemented in the
   eta-continuation driver:
      IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_ROWS=1
      fixed quarter-point rows in R=220-300 rg
      base rows: 3N+2
      extra rows: 4(N-1)
      N168 residual shape: (1174,)
      N168 Jacobian shape: (1174,506)

7. Rectangular sparse finite-difference Jacobian works technically:
      build time ~5.56 s
      nnz ~4573
      tests pass: 86 passed

8. Rectangular source-band polish improves the hidden defect:
      augmented residual 0.1477 -> 0.01895
      source-band extra max ~0.01895
      source extra energy dominates
      mass error initially grows

9. Mass-weighted resume reduces mass error:
      source-band extra remains ~0.0192
      mass residual improves to ~2.83e-4
      source-band energy floor is unchanged

10. N200 nested remap and residual remesh are not usable:
      N200 from mass-weighted checkpoint: augmented ~0.0608
      N200 from original midpoint checkpoint: augmented ~0.222
      same-N residual-aware remesh: augmented/mass ~9.27
```

Main question:

```text
What is the best numerical formulation to remove the remaining source-band
energy residual floor near R~245-250 rg?
```

Codex's current interpretation:

```text
The issue is no longer hidden.  Extra residual rows expose it and reduce it,
but the current piecewise-linear state representation plus ordinary global
remapping cannot represent the compact source annulus well enough.

The likely next step is a true source-annulus micro-domain or source-band
subnodes as real state unknowns, with interface continuity at source-band
edges and regular collocation inside the band.  The rectangular extra-row mode
should remain as an audit/diagnostic.
```

Please advise on:

```text
1. Whether to build a source-annulus micro-domain, Hermite/collocation upgrade,
   or another formulation first.

2. How to define the source-band unknowns and residuals so the problem remains
   well-conditioned.

3. Whether the compact C2 source should be represented with analytic finite-
   volume/source-integral constraints rather than pointwise differential rows.

4. How to remap/prolong from the existing midpoint-strict eta_E=90 checkpoint
   without creating large defects.

5. Acceptance criteria for calling eta_E=90 representation-robust before
   continuing eta_E lower or returning to wind/hot-branch claims.
```
