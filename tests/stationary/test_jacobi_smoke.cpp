// mp-iterative smoke test: verify the MTL5 + Universal composition builds and
// that Jacobi iteration solves a small diagonally dominant sparse system in
// each number type. Returns non-zero on failure (no external test framework,
// matching the repo's lightweight style).
#include <cmath>
#include <cstddef>
#include <iostream>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

// Universal number types: the iteration must converge in these through the
// MTL5 + Universal composition.
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T>
bool jacobi_ok(double tol) {
    constexpr std::size_t n = 4;
    using Sparse = mtl::mat::compressed2D<T>;
    Sparse A(n, n);
    {
        mtl::mat::inserter<Sparse> ins(A);
        // Strictly diagonally dominant -> Jacobi converges. Exact solution: all-ones.
        ins[0][0] << T(4); ins[0][1] << T(1); ins[0][3] << T(1);
        ins[1][0] << T(1); ins[1][1] << T(5); ins[1][2] << T(2);
        ins[2][1] << T(2); ins[2][2] << T(6); ins[2][3] << T(1);
        ins[3][0] << T(1); ins[3][2] << T(1); ins[3][3] << T(4);
    }
    mtl::vec::dense_vector<T> b = {T(6), T(8), T(9), T(6)};
    mtl::vec::dense_vector<T> x(n, T(0)), xn(n, T(0));

    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t iter = 0; iter < 50; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            T sigma(0), diag(0);
            for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
                if (ci[k] == i) diag = dat[k];
                else            sigma += dat[k] * x(static_cast<int>(ci[k]));
            }
            xn(static_cast<int>(i)) = (b(static_cast<int>(i)) - sigma) / diag;
        }
        x = xn;
    }

    // residual ||A*x - b||_inf in double
    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        r = std::max(r, std::abs(ax - static_cast<double>(b(static_cast<int>(i)))));
    }
    return r < tol;
}

} // namespace

int main() {
    int failures = 0;

    if (!jacobi_ok<double>(1e-10)) { std::cerr << "double Jacobi solve failed\n"; ++failures; }
    if (!jacobi_ok<float>(1e-4))   { std::cerr << "float Jacobi solve failed\n";  ++failures; }

    // Low precision: the iteration stagnates at the rounding floor of the type.
    if (!jacobi_ok<sw::universal::cfloat<16, 5>>(1e-1)) {
        std::cerr << "cfloat<16,5> Jacobi solve failed\n"; ++failures;
    }
    if (!jacobi_ok<sw::universal::posit<16, 2>>(1e-1)) {
        std::cerr << "posit<16,2> Jacobi solve failed\n"; ++failures;
    }

    if (failures == 0) std::cout << "mp-iterative smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
