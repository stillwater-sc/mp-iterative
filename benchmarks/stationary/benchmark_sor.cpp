// benchmark_sor: SOR accumulator-strategy sweep on the standard 1D Poisson
// problem at the optimal relaxation omega_opt = 2/(1+sin(pi/(n+1))), measuring
// forward error, backward residual, and convergence rate vs the theoretical
// SOR-at-optimum spectral radius omega_opt - 1.
//
//   ./benchmark_sor [n] [sweeps]              summary CSV on stdout
//   ./benchmark_sor --history [n] [sweeps]    per-iteration residual history
#include <mtl/itl/smoother/sor.hpp>
#include "sw/mp_iterative/benchmark/runner.hpp"

using namespace sw::mp_iterative::benchmark;

int main(int argc, char** argv) {
    return run_stationary_benchmark<mtl::itl::smoother::sor>(
        "sor", argc, argv, poisson_1d_sor_optimal_rate);
}
