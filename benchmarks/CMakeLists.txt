# Benchmarks (measurement drivers)
# Built only when MPITERATIVE_BUILD_BENCHMARKS is ON -- these are for
# measurement runs, not CI gating. Organized by iterative-method category
# (stationary, later krylov, multigrid).
add_subdirectory(stationary)
