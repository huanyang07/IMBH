# Retained guard-buffer micro-macro manifest

Classification: `retained_guard_buffer_overlap_architecture_frozen_analysis_only_preflight_authorized`.

The physical inner/macro exchange is the shared M/J/E flux at parent face 36. The macro exterior owns cells 36:64. The inner micro-solver may continue through cells 36:48 only as a duplicate numerical guard; those duplicate storage and source terms are not counted in the physical inventory.

Conservative restriction/prolongation and every synchronization reaction must be ledgered, including responsive-height BDF history. Face 36 is not relabelled face 48 or a horizon flux.

Only an existing-state overlap consistency preflight is authorized.
