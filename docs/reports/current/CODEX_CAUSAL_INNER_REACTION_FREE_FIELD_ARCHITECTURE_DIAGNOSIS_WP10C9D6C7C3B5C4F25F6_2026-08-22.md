# Reaction/free-field mathematical architecture diagnosis

Classification: `conservative_free_field_hidden_amplitude_rom_selected_fixed_Q_arclength_retained_sampling_only`.

The fixed-Q equations and their numerical certificates remain valid, but their constrained trajectory is not accepted as the physical fast-time clock. Across four committed cold anchors, the physical free coordinate rate is at most `1.724676e-04` of the constrained rate. The imposed reaction projects onto the fixed-Q tangent by at least `9.999980e-01`.

The free rate lies within `2.524403e-03` of the accepted full-model rank-two secant subspace, while the fixed-Q rate remains at least `9.998864e-01` outside it. The worst nearest full-model secant defect is `3.594297e-02`.

Select the exact conservative split `y=Lq+Z(h0+Va)` driven by the original free field. Fixed-Q arclength remains useful only to propose offline sample states; it supplies neither physical duration nor reduced drift. No new truth rate, root, or BDF microstep was executed.
