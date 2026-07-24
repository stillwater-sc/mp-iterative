// benchmark_jacobi: Jacobi accumulator-strategy sweep on the standard Poisson
// problem (1D/2D/3D), measuring forward error, backward residual, and
// convergence rate vs the theoretical Jacobi spectral radius rho_J =
// cos(pi/(m+1)) (m = grid points per dimension).
//
//   ./benchmark_jacobi [1d|2d|3d] [m] [sweeps]              summary CSV on stdout
//   ./benchmark_jacobi --history [1d|2d|3d] [m] [sweeps]    per-iteration history
#include <mtl/itl/smoother/jacobi.hpp>
#include "sw/mp_iterative/benchmark/runner.hpp"

using namespace sw::mp_iterative::benchmark;

int main(int argc, char** argv) {
    return run_stationary_benchmark<mtl::itl::smoother::jacobi>(
        "jacobi", argc, argv, poisson_jacobi_rate);
}
