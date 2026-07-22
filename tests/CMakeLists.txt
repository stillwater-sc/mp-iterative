include(CTest)

# Helper: each test is a self-checking executable linked against sw::mp_iterative.
function(mpiterative_add_test test_name)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::mp_iterative)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Tests are organized by iterative-method category:
#   1. stationary  -- Jacobi, Gauss-Seidel, SOR
#   2. krylov      -- CG, BiCGSTAB, GMRES, ...
#   3. multigrid   -- geometric/algebraic multigrid
add_subdirectory(stationary)
add_subdirectory(krylov)
add_subdirectory(multigrid)
