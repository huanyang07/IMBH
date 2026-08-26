# Entropy-complete path-conservative spatial manifest

Classification: `entropy_complete_path_conservative_spatial_manifest_frozen`.

The certified seven-field PDE is mixed: four exact physical conservation laws, two exact material-current balances, and one projected nonconservative Israel--Stewart shear row. The selected DLM operator integrates the complete radial principal along a fixed straight primitive path and preserves exact flux-difference parity on every exact-flux row.

Interface dissipation uses the complete midpoint eigenbasis and the absolute characteristic speeds. Negative and positive fluctuations must close to the DLM jump and define one shared flux from both sides on the exact-flux rows. Scalar max-speed Rusanov dissipation is not selected.

Authorized next: `WP10c9d6c7c3b5c4f25fizeg_entropy_complete_path_conservative_interface_audit` only. It may implement and audit interfaces but may not assemble a cell trajectory or take a time step.
