# Fixed-Q Primary-Evidence Aggregation WP10c9d6c7c3b5c4f24e14r

## Classification

`primary_bounded_continuation_evidence_certified`

The deterministic aggregation passes every frozen evidence gate. It reruns no
physical root and adds no trajectory time. The historical
WP10c9d6c7c3b5c4f24e14l `bounded_continuation_failed` classification remains
preserved; this package issues a separate certificate under the subsequently
frozen and independently certified same-history comparison policy.

## Accepted primary trajectory evidence

All four BDF2 roots are accepted and cover `4e-7 s`:

```text
root      residual                 evaluations   exact assemblies
cold_1    4.737492683089679e-13       19               2
warm_1    5.533443390874517e-13        8               1
warm_2    5.048217216618925e-13        8               1
warm_3    8.748886309915929e-13        8               1
```

Every root retains all centralized numerical, physical, storage, reaction,
conditioning, and ledger gates. The accepted-trajectory cumulative absolute
ledger defect is `3.996887103599641e-16`.

## Replay and temporal control

The `warm_2`/`warm_3` suffix replay is bitwise. The matched-endpoint two-half-
step audit passes with:

```text
scaled state difference relative to full-step change   3.363160546067348e-6
physical reaction-action relative difference           1.765640394809987e-5
```

## Cost evidence

The certified warm solve remains cheaper than the same-history cold control:

```text
warm/cold residual-evaluation ratio   0.6153846153846154
warm/cold wall-time ratio             0.6719283804454570
binding maximum for each              0.75
```

## Same-history endpoint equivalence

The certified nonpropagating comparison closes at:

```text
polished scaled state difference       5.064393349130114e-12
polished reaction-action defect        1.875569179324639e-9
binding maximum for each               1e-8
```

The polished control cannot enter history or trajectory. The production root
gate remains `1e-10`.

## Verification

All twelve aggregation gates pass, all canonical checksums close, and the
complete focused manifest/aggregation suite passes `8/8`.

## Authorization boundary

This certificate authorizes only a definitions-only held-out continuation
manifest. It does not authorize held-out execution, an operational timestep
study, a fixed-`Q` microburst, fast averaging, or reduced slow evolution.

The held-out manifest should repeat a shorter state-robustness check at the
committed 16 ms state: one cold BDF2 continuation, two warm BDF2 continuations,
and restart/replay of the final warm step, with the certified iteration-reserve
policy and same-history comparison policy retained unchanged.
