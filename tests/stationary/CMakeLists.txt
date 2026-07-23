# Stationary methods: Jacobi, Gauss-Seidel, SOR.

# Smoke test: Jacobi iteration solves through the MTL5 + Universal composition.
mpiterative_add_test(test_jacobi_smoke)

# Jacobi smoother accumulator strategies: naive vs FMA vs quire, measured
# with posits (issue #3).
mpiterative_add_test(test_jacobi_quire)
