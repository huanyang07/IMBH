# Causal Inner Band-Limited Balance Search Manifest

## WP10c9d6c6e2a — 2026-07-29

Analyzed base:

```text
5c644d2d5912ceca6c661bec6f5db55c798095a9
```

## Binding classification

```text
bandlimited_conditioning_search_frozen_feasibility_authorized
```

Manifest:

```text
7dba2fd9db6cc093eff8a3307dfe851036fee3fbb243a5100b1f0819d4b44c02
```

This definitions-only package freezes a deterministic search for a
continuum-balanced shear profile that remains inside the existing N128
spectral class. It evaluates no candidate, builds no tangent, propagates no
state, and changes no operator or threshold.

## Frozen candidate library

The six candidates are:

```text
p2_cos1
p2_cos2
p3_cos1
p3_cos2
p4_cos1
p4_cos2
```

For normalized log radius \(x\in[0,1]\), each envelope is

\[
w_{p,m}(x)
=
\sin^p(\pi x)
\left[
1+\alpha_{p,m}\cos(m\pi x)
\right].
\]

The coefficient is determined independently for each shear family from

\[
\alpha_{p,m}
=-
\frac{L[\sin^p(\pi x)r_{\rm sh}(R)]}
{L[\sin^p(\pi x)\cos(m\pi x)r_{\rm sh}(R)]},
\]

where \(L\) is the frozen 769-node continuum initial lower-height-work
angular functional. A 513-node construction tests coefficient stability and
cancellation.

## Frozen feasibility gates

Both inward and outward shear versions of a candidate must pass:

- `theta_99 <=0.30`;
- alias fraction `<=1e-3`;
- endpoint-cell fraction `<=5e-3`;
- global family purity `>=0.995`;
- active-cell family purity `>=0.98`;
- projection defect `<=2e-12`;
- absolute balance coefficient `<=50`;
- 769/513 coefficient difference `<=1e-6`;
- secondary-reference cancellation ratio `<=1e-6`;
- inward/outward coefficient difference `<=1e-10`.

Eligible candidates are sorted by the frozen ascending key:

```text
maximum theta_99 across both families
maximum alias fraction across both families
base power
modulation harmonic
```

No propagated-history quantity may enter selection.

## Authorized next task

```text
WP10c9d6c6e2b_bandlimited_balance_feasibility
```

The next package may evaluate all six frozen candidates and either:

- hash one selected inward/outward profile pair for a later propagation
  manifest; or
- conclude that no eligible cancellation stress profile exists in the
  certified class.

It may not propagate a selected profile in the feasibility commit.

All c6c, c6d, c6e0, and c6e1 classifications remain unchanged.
