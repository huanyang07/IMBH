# Second-rung perturbed completion WP10c9d6c7c3b5c2d

## Classification

`second_rung_perturbed_completion_failed_later_duration_blocked`

The missing perturbed trajectory reaches `1e-3 s`, but the frozen package
fails its same-tangent bitwise replay requirement. The historical c2
failure therefore remains unchanged and no later-duration propagation is
authorized.

## Numerical result

- main BDF2 comparisons accepted: `9`;
- serialized replay comparisons accepted: `4`;
- strict-shadow comparisons accepted: `4`;
- rejected attempts: `0` in both main and strict branches;
- maximum main/strict local error estimates:
  `1.6934137245584741e-9` / `4.716646528616119e-10`;
- maximum scaled nonlinear residual: `9.173003759707504e-11`;
- maximum discrete-ledger defect: `0`;
- maximum export-ledger defect: `9.060142018136859e-13`;
- maximum mapped endpoint/path closure defect:
  `1.1684408028129287e-11`;
- minimum reconstruction factor: `1`;
- incoming excision characteristics: `0`;
- minimum scattering optical depth: `19.034351598519283`;
- maximum `H/R`: `0.09863110934907435`.

The strict-shadow perturbed-minus-base response passes cleanly. Its maximum
scaled state, instantaneous-export, and cumulative-export differences are
`2.926903164279793e-11`, `6.972629768785949e-12`, and
`3.265385205014159e-12`; all three history cosines are one to the reported
precision.

## Binding replay failure

Canonical target labels agree bitwise, but one accumulated replay label is
one spacing unit away from the corresponding main label. Main and replay
then cease to be bitwise identical in primitive state, Tier-I exports, and
primitive/mapped/responsive-height histories. This makes the complete
perturbed method gate fail even though every integration and physics gate
passes.

A post-run read-only comparison of the durable main/replay cache localizes
the first mismatch to the same `9e-4 s` label:

- maximum primitive-state difference: `1.7763568394002505e-15`
  (`1.16e-16` of the maximum state magnitude);
- maximum direct-export difference: `1.74456832e8`
  (`8.84e-16` of the maximum export magnitude);
- primitive-history norm difference: `4.47e-13` relative;
- mapped-history norm difference: `7.68e-14` relative;
- responsive-height-history norm difference: `2.76e-11` relative.

These diagnostics strongly associate the formal failure with accumulated
time-label sensitivity at roundoff scale. They do not satisfy or relax the
prospectively frozen bitwise requirement, and they do not authorize the
third duration rung.

Authorized next: `none`.

The appropriate follow-up is a separate definitions-first canonical-time
execution/replay audit. That audit must preserve this failure, compare
accumulated-time and canonical-target execution directly under one tangent,
and may authorize the third-rung manifest only if a prospectively frozen
contract passes. Fixed-Q experiments and reduced evolution remain blocked.
