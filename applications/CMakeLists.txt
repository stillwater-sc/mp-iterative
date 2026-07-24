# Applications (demonstration programs)
# Organized by iterative-method category (stationary, krylov, multigrid);
# each leaf subdirectory contains its own CMakeLists.txt.

# Jacobi iteration convergence across precisions on a 1D Poisson system.
add_subdirectory(stationary/jacobi_precision)

# Gauss-Seidel convergence across precisions x accumulator strategies.
add_subdirectory(stationary/gauss_seidel_precision)

# SOR convergence across precisions x accumulator strategies x omega sweep.
add_subdirectory(stationary/sor_precision)

# CG convergence across precisions x accumulator strategies (naive/fma/quire).
# Reproduces the pure variants of the Universal CG dynamics study.
add_subdirectory(krylov/cg_precision)
