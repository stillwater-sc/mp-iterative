# Stationary-method measurement drivers: accumulator-strategy sweeps.
# Each is a plain executable that writes a summary CSV (stdout) and a
# human-readable table (stderr); no external benchmark framework.
function(mpiterative_add_benchmark bench_name)
	add_executable(${bench_name} ${bench_name}.cpp)
	target_link_libraries(${bench_name} PRIVATE sw::mp_iterative)
endfunction()

mpiterative_add_benchmark(benchmark_jacobi)
mpiterative_add_benchmark(benchmark_gauss_seidel)
mpiterative_add_benchmark(benchmark_sor)
