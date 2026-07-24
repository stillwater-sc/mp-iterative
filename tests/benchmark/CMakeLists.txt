# Benchmarking harness: unit tests for the shared measurement utilities.
# The drivers under benchmarks/ are OFF in CI, so these tests are what keep the
# reusable core (problems, reference solve, metrics, csv) under CI coverage.
mpiterative_add_test(test_metrics)
