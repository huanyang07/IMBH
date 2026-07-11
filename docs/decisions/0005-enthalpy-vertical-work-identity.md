# Decision 0005: Enthalpy-Compatible Vertical Work

Status: accepted, 2026-07-11.

The signed and ordinary conservative solvers transport column enthalpy through

```text
F_E = Mdot (q + e + Pi/Sigma) - Omega G.
```

The compatible one-zone work term is therefore

```text
W_H = Mdot (Pi dSigma/Sigma^2 - P drho/rho^2)
    = Mdot (P/rho) dlnH.
```

The previous term

```text
Mdot (dPi/Sigma - P drho/rho^2)
```

is retained only for the explicitly named legacy internal-energy identity.
Pairing it with enthalpy adds `Mdot d(Pi/Sigma)` a second time.

The identity is gated by an analytic derivative of the actual transonic
enthalpy flux, a four-level finite-volume convergence test, and a
source-bearing manufactured mass/angular/energy ledger. Commit `248e43c`
remains the historical mixed-pairing prototype; its canonical payloads are
superseded by regenerated corrected states.
