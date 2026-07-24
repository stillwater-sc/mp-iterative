// benchmark_sor: SOR accumulator-strategy sweep on the standard Poisson problem
// (1D/2D/3D) at the optimal relaxation omega_opt = 2/(1+sin(pi/(m+1))),
// measuring forward error, backward residual, and convergence rate vs the
// theoretical SOR-at-optimum spectral radius omega_opt - 1.
//
//   ./benchmark_sor [1d|2d|3d] [m] [sweeps]              summary CSV on stdout
//   ./benchmark_sor --history [1d|2d|3d] [m] [sweeps]    per-iteration history
#include <mtl/itl/smoother/sor.hpp>
#include "sw/mp_iterative/benchmark/runner.hpp"

using namespace sw::mp_iterative::benchmark;

int main(int argc, char** argv) {
    return run_stationary_benchmark<mtl::itl::smoother::sor>(
        "sor", argc, argv, poisson_sor_optimal_rate);
}
