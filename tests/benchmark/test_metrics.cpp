// Unit tests for the benchmarking harness's shared utilities (issue #6).
// The measurement drivers live under benchmarks/ and are OFF in CI, so this
// test -- built with the normal test suite -- is what keeps the reusable core
// (problems, reference solve, metrics, csv) honest. No external framework;
// returns non-zero on failure.
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "sw/mp_iterative/benchmark/problems.hpp"
#include "sw/mp_iterative/benchmark/reference_solve.hpp"
#include "sw/mp_iterative/benchmark/metrics.hpp"
#include "sw/mp_iterative/benchmark/csv.hpp"

using namespace sw::mp_iterative::benchmark;

namespace {

bool close(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// The double reference solve must actually solve A x = b: residual ~ 0 and,
// for the standard 1D Poisson with b = ones, match the known closed form
// x_i = (i+1)(n-i)/2 (1-based i), the discrete solution of -x'' = 1.
bool reference_solve_is_accurate() {
    const std::size_t n = 25;
    auto A = poisson_1d<double>(n);
    std::vector<double> b(n, 1.0);
    std::vector<double> x = reference_solution(A, b);

    if (residual_2norm(A, x, b) > 1e-10) {
        std::cerr << "reference solve residual too large\n";
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        double exact = (static_cast<double>(i) + 1.0) * (static_cast<double>(n) - static_cast<double>(i)) / 2.0;
        if (!close(x[i], exact, 1e-8)) {
            std::cerr << "reference solve x[" << i << "]=" << x[i] << " != exact " << exact << '\n';
            return false;
        }
    }
    return true;
}

// Forward relative error: zero for identical vectors, and the exact ratio for
// a known perturbation.
bool forward_error_is_correct() {
    std::vector<double> xref = {3.0, 4.0};            // ||xref|| = 5
    if (!close(forward_relative_error(xref, xref), 0.0, 1e-15)) {
        std::cerr << "forward error of identical vectors != 0\n";
        return false;
    }
    std::vector<double> x = {3.0, 4.0 + 5.0};         // ||x - xref|| = 5
    if (!close(forward_relative_error(x, xref), 1.0, 1e-15)) {
        std::cerr << "forward error ratio wrong\n";
        return false;
    }
    return true;
}

// Backward residual on a hand-checkable system.
bool residual_is_correct() {
    auto A = poisson_1d<double>(3);                   // tridiag(-1,2,-1), 3x3
    std::vector<double> b(3, 1.0);
    std::vector<double> zero(3, 0.0);
    // A*0 - b = -b, so ||.||_2 = sqrt(3), ||.||_inf = 1.
    if (!close(residual_2norm(A, zero, b), std::sqrt(3.0), 1e-12)) {
        std::cerr << "residual_2norm wrong\n"; return false;
    }
    if (!close(residual_infnorm(A, zero, b), 1.0, 1e-12)) {
        std::cerr << "residual_infnorm wrong\n"; return false;
    }
    return true;
}

// Convergence-rate estimate on a synthetic geometric history: residual_k = r^k
// should recover rate r; floor is the minimum; iterations-to-tol is exact.
bool convergence_metrics_are_correct() {
    const double r = 0.5;
    std::vector<double> hist;
    for (int k = 0; k < 20; ++k) hist.push_back(std::pow(r, k + 1));   // 0.5, 0.25, ...

    double emp = empirical_convergence_rate(hist);
    if (!close(emp, r, 1e-9)) {
        std::cerr << "empirical rate " << emp << " != " << r << '\n';
        return false;
    }
    if (!close(stagnation_floor(hist), std::pow(r, 20), 1e-15)) {
        std::cerr << "stagnation floor wrong\n"; return false;
    }
    // First index with residual < 0.2: 0.5^k < 0.2 -> k >= 3 (0.125), 1-based = 3.
    if (iterations_to_tolerance(hist, 0.2) != 3) {
        std::cerr << "iterations_to_tolerance wrong: " << iterations_to_tolerance(hist, 0.2) << '\n';
        return false;
    }
    // A flat (already-stagnated) history has no convergence segment -> NaN.
    std::vector<double> flat(10, 1e-8);
    if (!std::isnan(empirical_convergence_rate(flat))) {
        std::cerr << "flat history should give NaN rate\n"; return false;
    }
    return true;
}

// The analytic Poisson rates are ordered as theory predicts:
// SOR@opt < GS < Jacobi (smaller spectral radius = faster).
bool poisson_rates_are_ordered() {
    const std::size_t m = 48;
    double j  = poisson_jacobi_rate(m);
    double gs = poisson_gauss_seidel_rate(m);
    double s  = poisson_sor_optimal_rate(m);
    if (!(s < gs && gs < j && j < 1.0)) {
        std::cerr << "Poisson rates not ordered: sor=" << s << " gs=" << gs << " jac=" << j << '\n';
        return false;
    }
    // GS rate is exactly the square of the Jacobi rate.
    if (!close(gs, j * j, 1e-12)) {
        std::cerr << "GS rate != Jacobi rate squared\n"; return false;
    }
    return true;
}

// Count nonzeros in row r of a compressed2D<double>.
std::size_t nnz_in_row(const mtl::mat::compressed2D<double>& A, std::size_t r) {
    const auto& rp = A.ref_major();
    return rp[r + 1] - rp[r];
}

// The 2D/3D generators have the right dimensions and stencil width, and the
// factory + size helper agree.
bool higher_dim_poisson_structure() {
    const std::size_t m = 5;
    auto A2 = poisson_2d<double>(m);      // 25 x 25, 5-point
    auto A3 = poisson_3d<double>(m);      // 125 x 125, 7-point

    if (A2.num_rows() != m * m || A3.num_rows() != m * m * m) {
        std::cerr << "higher-dim Poisson dimensions wrong\n"; return false;
    }
    if (poisson_matrix_size(poisson_dim::d2, m) != m * m ||
        poisson_matrix_size(poisson_dim::d3, m) != m * m * m) {
        std::cerr << "poisson_matrix_size wrong\n"; return false;
    }
    // The interior cell has the full stencil: 5 nonzeros in 2D, 7 in 3D.
    const std::size_t center2 = (m / 2) * m + (m / 2);
    const std::size_t center3 = ((m / 2) * m + (m / 2)) * m + (m / 2);
    if (nnz_in_row(A2, center2) != 5) { std::cerr << "2D interior nnz != 5\n"; return false; }
    if (nnz_in_row(A3, center3) != 7) { std::cerr << "3D interior nnz != 7\n"; return false; }
    // A corner cell has fewer neighbors: 2D corner = diag + 2 = 3.
    if (nnz_in_row(A2, 0) != 3) { std::cerr << "2D corner nnz != 3\n"; return false; }

    // The factory dispatches to the same operator.
    auto Af = make_poisson<double>(poisson_dim::d2, m);
    if (Af.num_rows() != A2.num_rows() || Af.nnz() != A2.nnz()) {
        std::cerr << "make_poisson(d2) mismatch\n"; return false;
    }
    return true;
}

// The double reference solve works on the 2D operator too (SPD): residual ~ 0.
bool reference_solve_2d() {
    const std::size_t m = 12;
    auto A = poisson_2d<double>(m);
    std::vector<double> b(A.num_rows(), 1.0);
    std::vector<double> x = reference_solution(A, b);
    if (residual_2norm(A, x, b) > 1e-9) {
        std::cerr << "2D reference solve residual too large\n"; return false;
    }
    return true;
}

// CSV writer: header + a row with a comma-containing field is quoted per RFC 4180.
bool csv_quotes_and_formats() {
    csv_writer w({"a", "b", "c"});
    std::ostringstream os;
    w.write_header(os);
    w.field("x,y").number(std::size_t{3}).number(1.5).end_row(os);
    const std::string out = os.str();
    if (out.find("a,b,c\n") != 0) { std::cerr << "csv header wrong: " << out; return false; }
    if (out.find("\"x,y\",3,1.5") == std::string::npos) { std::cerr << "csv row wrong: " << out; return false; }
    return true;
}

} // namespace

int main() {
    int failures = 0;
    if (!reference_solve_is_accurate())      ++failures;
    if (!forward_error_is_correct())         ++failures;
    if (!residual_is_correct())              ++failures;
    if (!convergence_metrics_are_correct())  ++failures;
    if (!poisson_rates_are_ordered())        ++failures;
    if (!higher_dim_poisson_structure())     ++failures;
    if (!reference_solve_2d())               ++failures;
    if (!csv_quotes_and_formats())           ++failures;

    if (failures == 0) std::cout << "test_metrics passed\n";
    return failures == 0 ? 0 : 1;
}
