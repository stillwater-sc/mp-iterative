# Krylov subspace methods: CG, BiCGSTAB, GMRES, ...

# CG with quire (exact dot product) accumulation of rho/pAp vs naive posit32.
# Migrated from mtl5 (MTL5 must not depend on Universal).
mpiterative_add_test(test_cg_quire)
