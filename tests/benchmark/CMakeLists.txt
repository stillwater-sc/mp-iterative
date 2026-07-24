# Benchmarking harness: unit tests for the shared measurement utilities.
# The drivers under benchmarks/ are OFF in CI, so these tests are what keep the
# reusable core (problems, reference solve, metrics, csv) under CI coverage.
mpiterative_add_test(test_metrics)

# Matrix Market loader (issue #27): exercised against a tiny committed SPD
# fixture -- real SuiteSparse matrices are never committed. The fixture dir is
# passed in so the test can locate spd_tridiag4.mtx regardless of build layout.
mpiterative_add_test(test_matrix_source)
target_compile_definitions(test_matrix_source PRIVATE
	MPITERATIVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
