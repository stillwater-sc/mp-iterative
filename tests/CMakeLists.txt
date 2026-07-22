include(CTest)

# Helper: each test is a self-checking executable linked against sw::mp_iterative.
function(mpiterative_add_test test_name)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::mp_iterative)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Smoke test: Jacobi iteration solves through the MTL5 + Universal composition.
mpiterative_add_test(test_jacobi_smoke)
