# Curvature-adaptive arclength continuation manifest

Classification: `autonomous_curvature_adaptive_endpoint_collocation_continuation_manifest_frozen`.

The selected truth system remains the autonomous original reaction-free field `dy/dt=f_free(y)`. The endpoint is proposed by variable-step AB2, exactly retracted and physically audited, and then supplied with an exact endpoint field evaluation. A cubic Hermite segment is accepted only after its endpoint integral closure passes; every fourth tentative segment also receives an exact blind midpoint field check.

Offline variable-step AB2 replay prevalidates endpoint proposals through 2 ms: the worst steady 2 ms endpoint defect is `1.159013e-02`. Four milliseconds remains an exact-validation-controlled attempt, not a promised operational span; its worst diagnostic predictor defect is `4.989162e-02`.

The span starts at `1.000e-03` s, may grow by at most `2.0x` to `4.000e-03` s, and halves on any candidate failure without propagating the rejected state. The full budget is `288` exact calls and `30.0` wall hours. Using the measured parent cost with reserve projects `26.067` wall hours.

A detected return authorizes only a matched-path refinement/global cycle-map manifest. Equilibrium stability, slow closure, cycle averaging, and reduced slow evolution remain separately gated.
