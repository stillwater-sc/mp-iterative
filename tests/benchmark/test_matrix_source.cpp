// Unit tests for the Matrix Market matrix source (issue #27). The SuiteSparse
// benchmark drivers are OFF in CI and real matrices are never committed, so
// this test exercises the loader + value-type converter against a tiny
// committed SPD fixture (tests/benchmark/data/spd_tridiag4.mtx). No external
// framework; returns non-zero on failure.
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <universal/number/posit/posit.hpp>

#include "sw/mp_iterative/benchmark/matrix_source.hpp"
#include "sw/mp_iterative/benchmark/reference_solve.hpp"
#include "sw/mp_iterative/benchmark/metrics.hpp"

using namespace sw::mp_iterative::benchmark;

#ifndef MPITERATIVE_TEST_DATA_DIR
#error "MPITERATIVE_TEST_DATA_DIR must be defined by CMake"
#endif

namespace {

const std::string kFixture = std::string(MPITERATIVE_TEST_DATA_DIR) + "/spd_tridiag4.mtx";

bool close(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// The loader reads the fixture, expanding symmetric storage: 4x4, diagonal 4,
// off-diagonal -1, so 10 full nonzeros and max 3 per row.
bool loads_and_expands_symmetric() {
    auto A = load_matrix_market(kFixture);
    if (A.num_rows() != 4 || A.num_cols() != 4) {
        std::cerr << "fixture dims wrong: " << A.num_rows() << "x" << A.num_cols() << '\n';
        return false;
    }
    if (A.nnz() != 10) {   // 4 diagonal + 2*3 off-diagonal (symmetric expansion)
        std::cerr << "fixture nnz != 10 (symmetric not expanded?): " << A.nnz() << '\n';
        return false;
    }
    if (max_nnz_per_row(A) != 3) {
        std::cerr << "fixture max nnz/row != 3: " << max_nnz_per_row(A) << '\n';
        return false;
    }
    // Interior rows must carry the full stencil (-1, 4, -1): row 1 has 3 entries.
    const auto& rp = A.ref_major();
    if (rp[2] - rp[1] != 3) { std::cerr << "interior row nnz != 3\n"; return false; }
    return true;
}

// The double reference solve works on the loaded SPD operator: residual ~ 0.
bool reference_solve_on_loaded_matrix() {
    auto A = load_matrix_market(kFixture);
    std::vector<double> b(A.num_rows(), 1.0);
    std::vector<double> x = reference_solution(A, b);
    if (residual_2norm(A, x, b) > 1e-12) {
        std::cerr << "loaded-matrix reference residual too large\n";
        return false;
    }
    return true;
}

// Converting the loaded double operator to a value type preserves the pattern
// and (for a type that represents the small integer entries exactly) the
// values, so the double and converted operators give the same residual.
bool convert_preserves_operator() {
    using Posit = sw::universal::posit<32, 2>;
    auto Ad = load_matrix_market(kFixture);
    auto Ap = convert_matrix<Posit>(Ad);

    if (Ap.num_rows() != Ad.num_rows() || Ap.nnz() != Ad.nnz()) {
        std::cerr << "convert changed dims/nnz\n"; return false;
    }
    // Entries are exact small integers (4, -1), exactly representable in posit32.
    const auto& dp = Ap.ref_data();
    const auto& dd = Ad.ref_data();
    for (std::size_t k = 0; k < Ad.nnz(); ++k)
        if (!close(static_cast<double>(dp[k]), dd[k], 0.0)) {
            std::cerr << "convert changed entry " << k << '\n'; return false;
        }
    return true;
}

} // namespace

int main() {
    int failures = 0;
    if (!loads_and_expands_symmetric())     ++failures;
    if (!reference_solve_on_loaded_matrix()) ++failures;
    if (!convert_preserves_operator())       ++failures;

    if (failures == 0) std::cout << "test_matrix_source passed\n";
    return failures == 0 ? 0 : 1;
}
