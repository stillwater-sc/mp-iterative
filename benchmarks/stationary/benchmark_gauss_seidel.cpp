// benchmark_gauss_seidel: Gauss-Seidel accumulator-strategy sweep on the
// standard Poisson problem (1D/2D/3D), measuring forward error, backward
// residual, and convergence rate vs the theoretical GS spectral radius rho_J^2.
//
//   ./benchmark_gauss_seidel [1d|2d|3d] [m] [sweeps]              summary CSV
//   ./benchmark_gauss_seidel --history [1d|2d|3d] [m] [sweeps]    history
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include "sw/mp_iterative/benchmark/runner.hpp"

using namespace sw::mp_iterative::benchmark;

int main(int argc, char** argv) {
    return run_stationary_benchmark<mtl::itl::smoother::gauss_seidel>(
        "gauss_seidel", argc, argv, poisson_gauss_seidel_rate);
}
