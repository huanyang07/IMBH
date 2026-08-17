# Fixed-Q Held-Out Continuation WP10c9d6c7c3b5c4f24e14t

## Classification

`heldout_bounded_continuation_certified`

The prospectively frozen three-root continuation passes at the committed 16 ms
held-out state. All attempted roots are accepted, the accepted trajectory
advances `3e-7 s`, and the final warm root replays bitwise from the preceding
arbitrary-BDF2 checkpoint.

## Accepted held-out trajectory

```text
root      residual                 evaluations   exact assemblies   refresh
cold_1    5.949773994387308e-13       17               2            line failure
warm_1    6.138838980914190e-13        9               1            iteration reserve
warm_2    3.584635939980710e-13        8               1            iteration reserve
```

The cold root starts without a carried matrix. Both warm roots start from the
transported Broyden matrix and require the prospectively frozen iteration-6
reserve refresh. Every root stays within its exact-assembly budget.

## Scientific gates

Every root passes the complete centralized numerical, fixed-Q, storage,
reconstruction, reaction, conditioning, physical, primitive-change, and
excision gates. In particular:

- all residuals are below the unchanged `1e-10` production gate;
- Q3 relative defects are at most `2.36e-16`;
- path reconstruction factors remain exactly one;
- raw Schur systems remain rank three with condition numbers below `3.54e4`;
- all arbitrary-BDF2 checkpoints round-trip bitwise;
- no rejected candidate enters history.

The accepted-trajectory cumulative absolute ledger defect is
`3.794725600704226e-16`, far below the frozen `3e-12` gate.

## Restart and replay

`warm_2` is replayed from the committed `warm_1` checkpoint. The replay is
accepted and reproduces both the root result and continuation state bitwise,
including the residual trace, iteration-reserve refresh, and final residual
`3.584635939980710e-13`.

## Interpretation

The primary 20 ms and held-out 16 ms states now both support bounded,
accepted-history-only, multi-step fixed-Q BDF2 continuation under the certified
iteration-reserve policy. This is evidence of state-robust local continuation;
it is not yet an operational-timestep, long-time stability, microburst, or slow
closure certificate.

## Verification and authorization

All canonical artifact checksums close. The focused manifest/execution suite
passes `8/8`.

This certificate authorizes only a definitions-only operational-timestep
manifest. It does not authorize operational-timestep execution, a fixed-Q
microburst, fast averaging, or reduced slow evolution.
