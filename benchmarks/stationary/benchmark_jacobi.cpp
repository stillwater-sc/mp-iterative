// benchmark_jacobi: Jacobi accumulator-strategy sweep on the standard 1D
// Poisson problem, measuring forward error, backward residual, and convergence
// rate vs the theoretical Jacobi spectral radius rho_J = cos(pi/(n+1)).
//
//   ./benchmark_jacobi [n] [sweeps]              summary CSV on stdout
//   ./benchmark_jacobi --history [n] [sweeps]    per-iteration residual history
#include <mtl/itl/smoother/jacobi.hpp>
#include "sw/mp_iterative/benchmark/runner.hpp"

using namespace sw::mp_iterative::benchmark;

int main(int argc, char** argv) {
    return run_stationary_benchmark<mtl::itl::smoother::jacobi>(
        "jacobi", argc, argv, poisson_1d_jacobi_rate);
}
