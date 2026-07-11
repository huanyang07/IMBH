# ADR 0007: Treat the inner/reservoir overlap as threshold-sensitive

## Status

Accepted on 2026-07-11.

## Context

The corrected signed-flux reservoir is nearly Keplerian, while the inner slim
disk retains radial force balance. Coupling them requires a radial interval in
which both descriptions are valid and the reservoir's neglected pressure
support is small. Scattering depth alone is insufficient to certify the
optically thick diffusion closure.

## Decision

The production overlap audit uses common gates for radial pressure support,
`d ln(l_K)/d ln(R)`, thickness, radial Mach number, scattering depth,
effective optical depth, radial gradient length, and source exclusion.

The primary pressure gate is `epsilon_P <= 0.05`. A separate
`epsilon_P <= 0.10` run is retained as a sensitivity test, not silently folded
into the primary acceptance result. Effective depth uses a broad Kramers
absorption bracket and the lower-opacity value for acceptance. This opacity is
diagnostic and does not alter the cooling closure.

## Consequences

No common wall/transonic or open/transonic band passes the primary gate in
`12-60 r_g`. The `10%` sensitivity run exposes candidate bands, including a
wall/transonic interval near `29.45-59.69 r_g`. Interface experiments may use
that interval to test whether a coupled solution reduces the pressure
mismatch, but the result cannot yet be called a certified physical match.
