# ADR 0018: Valencia gas+radiation primitive recovery

## Status

Accepted for the local WP10c1 primitive map. This does not select the
production vertical-equilibrium or geometric-source closure.

## Context

ADR 0017 selects a horizon-penetrating Valencia column system with primitive
variables

```text
P = (Sigma, v_hat_R/c, v_hat_phi/c, T)
```

and conserved variables

```text
U = (D, S_R, S_phi, tau).
```

The existing one-zone vertical closure cannot be inserted directly into the
local conserved-to-primitive inversion because it obtains the column height
from the Paczynski-Wiita orbital frequency. That would mix the EOS inversion
with the old gravity model before the Kerr-Schild geometric source contract
has been derived.

## Decision

Certify the thermodynamic primitive map at a fixed proper half-height `H`.
This gives the gravity-independent column relations

```text
rho   = Sigma/(2H)
P     = rho R_g T + a T^4/3
Pi    = 2H P
e     = R_g T/(gamma_g-1) + a T^4/rho
h     = 1 + (e + Pi/Sigma)/c^2.
```

At fixed `H`, changing `Sigma` is exactly equivalent to changing `rho`, so
the volume gas+radiation acoustic derivative used in WP10a remains the
consistent local column derivative.

The fixed height is a thermodynamic chart parameter. WP10c2 must replace it
with a declared geometric/vertical contract in the finite-volume equations.
No near-horizon hydrostatic claim follows from this decision.

## Conserved-to-primitive recovery

Write pressure in mass units as

```text
p = Pi/c^2
```

and define

```text
S^2 = gamma^RR S_R^2 + S_phi^2/gamma_phiphi
Q   = tau + D + p.
```

For each pressure trial,

```text
W     = Q/sqrt(Q^2-S^2)
Sigma = D/W.
```

The internal energy is evaluated in the cancellation-resistant form

```text
e/c^2 = [
    tau - D(W-1) - p(W^2-1)
] / (D W).
```

The EOS monotonically inverts `(Sigma,e)` for `T` and supplies `Pi_EOS`.
The scalar root is

```text
f(p) = p - Pi_EOS(Sigma,T)/c^2 = 0.
```

The pressure bracket is scanned on a logarithmic scale and the final root is
solved in `log p`. Invalid timelike states, non-positive internal energy, and
unbracketed conserved states are rejected. Recovery does not clip velocity,
temperature, pressure, or internal energy.

The Eulerian velocities then follow from

```text
v_hat_R/c   = S_R/[Q sqrt(gamma_RR)]
v_hat_phi/c = S_phi/(Q R).
```

The forward Valencia energy is also evaluated without subtracting two large
rest-mass terms:

```text
tau = D[(W-1) + (e/c^2 + p/Sigma)W] - p.
```

## Independent characteristic audit

The gas+radiation flux Jacobian is differentiated in the local chart

```text
(ln Sigma, v_hat_R/c, v_hat_phi/c, ln T)
```

with a five-point stencil. Its eigenvalues are compared with the analytic
Valencia characteristic speeds using the WP10a causal gas+radiation sound
speed. This derivative is an audit only; it does not alter the production
flux.

## Acceptance result

The bounded matrix contains nine states:

```text
radii             20, 4.5, 1.8 rg
thermodynamics     gas, gas+radiation transition, radiation dominated
rotation included  yes
```

It gives

```text
maximum primitive round-trip defect     7.42e-11
maximum conserved round-trip defect     6.46e-15
maximum characteristic defect           1.94e-8
maximum sound speed                      0.577192 c
inside-horizon incoming modes            0
invalid conserved state rejected         yes
```

WP10c1 therefore passes its local mathematical gate.

## Consequences

1. The selected Valencia architecture now has a tested gas+radiation
   `P -> U -> P` map.
2. No old Paczynski-Wiita plunge checkpoint is mapped into the new variables.
3. This result does not provide geometric source terms, a stationary disk,
   stress, cooling, stream injection, or a Hill/Roche boundary.
4. WP10c2 is next: derive and discretize the source-free Kerr-Schild
   geometric finite-volume system and certify its independent ledgers.
