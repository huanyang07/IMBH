# Global Roche Adaptive-Restart Results

**Date:** 2026-07-13, Asia/Shanghai
**Starting commit:** `61630ea`

## Implementation

The global solver now has a deterministic reject/halve/grow controller around
the fully coupled backward-Euler step. Nonlinear residual and ledger gates
remain unchanged. An otherwise converged root is also rejected when the
configured change in `Sigma`, temperature, or `H/R` is too large.

The restart schema stores and verifies:

```text
conservative state and SHA-256
inner reference state and SHA-256
exact grid edges
mechanical offset, hashes, and provenance
elapsed time and next timestep
accepted-step and rejected-attempt counters
run provenance and schema version
```

The loader uses `allow_pickle=False` and rejects mesh or checksum mismatch.

## Physical N64 preflight

The no-tide, no-wind physical Roche run starts at `dt=1e-7 t_load`, reloads
its checkpoint after every accepted step, and targets `5e-7 t_load`.

```text
target reached                    yes
accepted steps                    8
rejected attempts                 2
restart reloads                   8
relative disk-mass increase       4.1064e-7
Roche channel                     closed throughout
```

The controller handled two distinct failures correctly:

1. A nonlinear root passed its equation gate but changed `H/R` by `2.40%`,
   above the declared `2%` step gate. Halving the step reduced the change to
   `0.94%` and passed.
2. A later trial reached 300 function evaluations with residual `6.65e-5`.
   Halving produced an accepted root with residual `4.49e-13`.

All accepted attempts kept individual `Sigma`, temperature, and thickness
changes below their 2% gates. The Jacobi deficit remains negative, moving only
from about `-8.573014e16` to `-8.573009e16 erg/g`; overflow does not open.

## Status

This certifies adaptive rejection and exact restart continuation for a bounded
N64 physical run. It does not certify long evolution: `5e-7 t_load` remains
far shorter than a loading, thermal, or viscous time. The next campaign should
extend the accepted N64 restart geometrically, checkpoint every accepted step,
and introduce N96/N128 comparisons at shared physical times. Distributed tide
and wind remain blocked.
