# Stationary methods: Jacobi, Gauss-Seidel, SOR.

# Smoke test: Jacobi iteration solves through the MTL5 + Universal composition.
mpiterative_add_test(test_jacobi_smoke)

# Jacobi smoother accumulator strategies: naive vs FMA vs quire, measured
# with posits (issue #3).
mpiterative_add_test(test_jacobi_quire)

# Gauss-Seidel smoother accumulator strategies + GS-vs-Jacobi convergence
# rate (intra-sweep error propagation), measured with posits (issue #4).
mpiterative_add_test(test_gauss_seidel_quire)

# SOR smoother accumulator strategies + omega relaxation (SOR@omega_opt beats
# GS), measured with posits (issue #5).
mpiterative_add_test(test_sor_quire)
