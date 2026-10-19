# Krylov subspace methods: CG, BiCGSTAB, GMRES, ...

# CG with quire (exact dot product) accumulation of rho/pAp vs naive posit32.
# Migrated from mtl5 (MTL5 must not depend on Universal).
mpiterative_add_test(test_cg_quire)
# TFQMR with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_tfqmr_quire)
# MINRES with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_minres_quire)
# CGS with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_cgs_quire)

# BiCGSTAB with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_bicgstab_quire)
# BiCGSTAB(ell) with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_bicgstab_ell_quire)
# BiCG with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_bicg_quire)

# GMRES with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_gmres_quire)

# QMR with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_qmr_quire)
# IDR(s) with quire (exact dot product) accumulation vs naive posit32.
mpiterative_add_test(test_idr_s_quire)

# The configuration-3 (quire) Krylov rows of the accumulation validation matrix,
# stillwater-sc/mtl5#586 (issue #47): CG and GMRES under plain / FMA /
# compensated / quire at float and posit<32,2>. Configuration 1 is "the product
# rounds", so the compiler must not contract acc += a*b into an FMA.
mpiterative_add_test(test_accumulation_matrix_krylov)
if(NOT MSVC)
	target_compile_options(test_accumulation_matrix_krylov PRIVATE -ffp-contract=off)
endif()
