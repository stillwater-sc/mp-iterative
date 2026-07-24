// benchmark_gauss_seidel: Gauss-Seidel accumulator-strategy sweep on the
// standard 1D Poisson problem, measuring forward error, backward residual, and
// convergence rate vs the theoretical GS spectral radius rho_J^2.
//
//   ./benchmark_gauss_seidel [n] [sweeps]              summary CSV on stdout
//   ./benchmark_gauss_seidel --history [n] [sweeps]    per-iteration history
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include "sw/mp_iterative/benchmark/runner.hpp"

using namespace sw::mp_iterative::benchmark;

int main(int argc, char** argv) {
    return run_stationary_benchmark<mtl::itl::smoother::gauss_seidel>(
        "gauss_seidel", argc, argv, poisson_1d_gauss_seidel_rate);
}
