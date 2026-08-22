# Truth-free conservative hot-mode engine

Classification: `truth_free_conservative_hot_mode_engine_verified`.

One 0.25 ms Heun macro step has endpoint-increment defect `3.821084e-04`, macro-increment defect `1.199859e-16`, and embedded correction `1.572541e-02`.

The 100,000-step update-plus-full-decode benchmark took `7.440373` s (`13440.187` steps/s). Restart and replay are bitwise; a 2x oversized step rejects fail-closed.

Every online truth, fixed-Q reaction, retraction, nonlinear-root, and BDF counter is zero. A pass authorizes only the final adaptive complete-cycle acquisition manifest, not complete-cycle execution itself.
