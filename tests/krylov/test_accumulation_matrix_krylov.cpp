// mp-iterative Krylov test: the configuration-3 (quire) rows of the accumulation
// validation matrix, stillwater-sc/mtl5#586 (issue #47). Each case names the MTL5
// tag it mirrors (mtl5 tests/unit/operation/test_accumulation_solvers_mt.cpp):
//
//   [586:krylov]                 CG and GMRES, every configuration
//   [586:krylov][586:determinism] CG with the quire, threaded vs serial
//
// The same MTL5 solver runs under each accumulation configuration at one element
// type T, so the comparison isolates the accumulation:
//
//   1.  plain        Accumulator = T                       product rounds, then the sum
//   2.  FMA          mtl::math::fma_accumulator<T>         one rounding per term
//   2c. compensated  mtl::math::compensated_accumulator<T> Dot2 (float only: the
//                                                          error-free transformations
//                                                          need IEEE arithmetic)
//   3.  quire        sw::universal::quire<T>               one rounding at the end
//
// at T = float (the MTL5 row, with Universal's IEEE quire in place of MTL5's
// Kulisch stand-in) and T = posit<32,2> (where the CG result of mtl5#237/#238
// was found). The accumulator covers CG's and GMRES's dot products and the
// SpMV; the updates of x, r and p are in T whatever the accumulator.
//
// The systems are MTL5's: a 5-point Laplacian on a 20 x 20 grid, symmetrically
// scaled D L D with spread 1 and 10 (CG), and upwind convection-diffusion,
// Peclet 0.3 (GMRES(20)). Here also spread 1000, a harder CG case. The true
// relative residual ||b - A x|| / ||b|| is measured exactly: every A and x entry
// converted to double (exact for float and posit<32,2>) and each residual entry
// summed in a binary64 Kulisch accumulator, mtl::math::exact_accumulator<double>.
//
// Measured (true relative residual, iterations; recurrence tolerance 1e-6):
//
//   float: NO ORDERING, as in MTL5. Every configuration converges on every
//   system, spread 1 to 1e5 and GMRES; residuals within 2.5x of each other,
//   iteration counts within a few percent. The float updates set the limit.
//
//   posit<32,2>, CG on m x m grids:        plain           fma             quire
//     m = 20, spread 1                     8.4e-7   48     8.4e-7   48     8.4e-7   48
//     m = 20, spread 1e4                   3.0e-5  152     2.4e-5  120     1.8e-5  114
//     m = 30, spread 1e4                   1.3e-2 4000     3.8e-5  232     3.4e-5   50
//     m = 20, spread 3e4                   NaR     153     2.2e36  255     4.9e-5  150
//     m = 30, spread 1e5                   NaR       5     2.2e36  124     2.5e-4  131
//
//   Well scaled, posit behaves as float does. Badly scaled, the QUIRE MATTERS:
//   at spread 3e4 and up (m = 20, 30; and m = 16 at 1e5) plain posit arithmetic
//   turns NaR and fma saturates near maxpos, while the quire converges. The
//   entries reach 4e10 at spread 1e5, so CG's inner products reach magnitudes
//   where a posit<32,2> keeps only a few fraction bits; plain rounds each
//   product there and fma each partial sum, and the quire neither. That is the
//   tapered precision mtl5#586 asked about, and it is what float, with the same
//   precision at every magnitude, never shows. In between (spread 1e3 - 1e4)
//   the quire's residual is lowest or near it but the iteration counts do not
//   order (m = 16, spread 1e3: plain 97, quire 126), so those rows assert only
//   convergence. The posit true residuals stay above the 1e-6 tolerance even
//   with the quire: the updates of x, in posit, set that limit.
//
//   GMRES(20) on convection-diffusion: no ordering in either type.
//
//   Determinism: CG with the quire is bit-identical threaded (4 workers) and
//   serial, in float and in posit<32,2>; plain float is not.
//
// Built with -ffp-contract=off (tests/krylov/CMakeLists.txt): a compiler that
// contracts acc += a*b into an FMA turns configuration 1 into configuration 2.
//
// Sets MTL5_NUM_THREADS before the thread pool's first use. Returns non-zero on
// failure.
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <type_traits>

#include <universal/number/posit/posit.hpp>
#include <universal/number/quire/quire.hpp>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

using posit32 = sw::universal::posit<32, 2>;
template <typename T> using quire_acc = sw::universal::quire<T>;
template <typename T> using fma_acc   = mtl::math::fma_accumulator<T>;
template <typename T> using comp_acc  = mtl::math::compensated_accumulator<T>;
template <typename T> using smat      = mtl::mat::compressed2D<T>;
template <typename T> using vec       = mtl::vec::dense_vector<T>;

static_assert(mtl::math::mergeable_accumulator<quire_acc<float>, float>);
static_assert(mtl::math::mergeable_accumulator<quire_acc<posit32>, posit32>);

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

/// 5-point Laplacian on an m x m grid, symmetrically scaled D L D so that the
/// condition number grows with `spread` (SPD for any spread). Formed in double,
/// rounded once to T.
template <typename T>
smat<T> laplacian_2d(std::size_t m, double spread) {
    const std::size_t n = m * m;
    smat<T> A(n, n);
    auto d = [&](std::size_t i) { return std::pow(spread, static_cast<double>(i % 7) / 6.0); };
    {
        mtl::mat::inserter<smat<T>> ins(A);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const std::size_t r = i * m + j;
                auto put = [&](std::size_t c, double v) { ins[r][c] << T(d(r) * v * d(c)); };
                put(r, 4.0);
                if (i > 0) put(r - m, -1.0);
                if (i + 1 < m) put(r + m, -1.0);
                if (j > 0) put(r - 1, -1.0);
                if (j + 1 < m) put(r + 1, -1.0);
            }
    }
    return A;
}

/// Upwind convection-diffusion on an m x m grid: nonsymmetric.
template <typename T>
smat<T> convection_diffusion(std::size_t m, double peclet) {
    const std::size_t n = m * m;
    smat<T> A(n, n);
    {
        mtl::mat::inserter<smat<T>> ins(A);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const std::size_t r = i * m + j;
                ins[r][r] << T(4.0);
                if (i > 0) ins[r][r - m] << T(-1.0 - peclet);
                if (i + 1 < m) ins[r][r + m] << T(-1.0 + peclet);
                if (j > 0) ins[r][r - 1] << T(-1.0 - peclet);
                if (j + 1 < m) ins[r][r + 1] << T(-1.0 + peclet);
            }
    }
    return A;
}

/// b = A x for a fixed, non-trivial x, each entry summed exactly and rounded to T.
template <typename T>
vec<T> rhs(const smat<T>& A) {
    const std::size_t n = A.num_rows();
    vec<T> b(n);
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v = A.ref_data();
    for (std::size_t r = 0; r < n; ++r) {
        mtl::math::exact_accumulator<double> a{};
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k) {
            const double xc = static_cast<double>(1 + (ci[k] * 7919) % 13) / 13.0;
            accumulator_add_product(a, static_cast<double>(v[k]), xc);
        }
        b(r) = T(accumulator_value(a, std::type_identity<double>{}));
    }
    return b;
}

/// ||b - A x||_2 / ||b||_2, each residual entry computed exactly.
template <typename T>
double rel_residual(const smat<T>& A, const vec<T>& x, const vec<T>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v = A.ref_data();
    double s = 0.0, sb = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        mtl::math::exact_accumulator<double> a{};
        accumulator_add_product(a, static_cast<double>(b(r)), 1.0);
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            accumulator_add_product(a, -static_cast<double>(v[k]), static_cast<double>(x(ci[k])));
        const double e = accumulator_value(a, std::type_identity<double>{});
        const double br = static_cast<double>(b(r));
        s += e * e;
        sb += br * br;
    }
    return std::sqrt(s / sb);
}

struct outcome {
    double residual;
    int iterations;
};

template <typename Acc, typename T>
outcome run_cg(const smat<T>& A, int max_iter) {
    const vec<T> b = rhs(A);
    vec<T> x(A.num_rows(), T(0));
    mtl::itl::pc::identity<smat<T>> I(A);
    mtl::itl::basic_iteration<T> it(b, max_iter, T(1e-6));
    mtl::itl::cg<Acc>(A, x, b, I, it);
    return {rel_residual(A, x, b), it.iterations()};
}

template <typename Acc, typename T>
outcome run_gmres(const smat<T>& A, int max_iter) {
    const vec<T> b = rhs(A);
    vec<T> x(A.num_rows(), T(0));
    mtl::itl::pc::identity<smat<T>> I(A);
    mtl::itl::basic_iteration<T> it(b, max_iter, T(1e-6));
    mtl::itl::gmres<Acc>(A, x, b, I, it, 20);
    return {rel_residual(A, x, b), it.iterations()};
}

void print(const char* what, const char* type, const char* acc, outcome o) {
    std::printf("[586:krylov] %-27s %-8s %-12s true residual %9.2e  iterations %4d\n", what, type, acc,
                o.residual, o.iterations);
}

constexpr int max_iter = 4000;

/// One system under every configuration available for T, printed. Index 2
/// (compensated) is float-only; for posit it stays {0, 0} and is not printed.
template <typename T, typename Solve>
std::array<outcome, 4> row(const char* what, const char* type, Solve solve) {
    std::array<outcome, 4> o{};
    o[0] = solve(T{});
    o[1] = solve(fma_acc<T>{});
    if constexpr (std::is_same_v<T, float>) o[2] = solve(comp_acc<T>{});
    o[3] = solve(quire_acc<T>{});
    const char* names[4] = {"plain", "fma", "compensated", "quire"};
    for (int c = 0; c < 4; ++c)
        if (c != 2 || std::is_same_v<T, float>) print(what, type, names[c], o[c]);
    return o;
}

template <typename T>
std::array<outcome, 4> cg_row(const char* type, std::size_t m, double spread) {
    const auto A = laplacian_2d<T>(m, spread);
    char what[64];
    std::snprintf(what, sizeof(what), "CG, m = %zu, spread %g", m, spread);
    return row<T>(what, type, [&](auto acc) { return run_cg<decltype(acc)>(A, max_iter); });
}

template <typename T>
std::array<outcome, 4> gmres_row(const char* type) {
    const auto A = convection_diffusion<T>(20, 0.3);
    return row<T>("GMRES(20), conv-diff 0.3", type,
                  [&](auto acc) { return run_gmres<decltype(acc)>(A, max_iter); });
}

bool converged(outcome o, double tol) { return o.residual < tol && o.iterations < max_iter; }

/// Failed: a NaR/NaN or saturated iterate, or no convergence.
bool failed(outcome o) { return !(o.residual < 1e-2); }

// ---------------------------------------------------------------------------
// [586:krylov]
// ---------------------------------------------------------------------------

/// float: no ordering, as in MTL5. Every configuration converges, to a true
/// residual at the 1e-6 recurrence tolerance, on every system up to spread 1e5.
void krylov_float() {
    auto all_converge = [](const std::array<outcome, 4>& o, const std::string& what) {
        for (int c = 0; c < 4; ++c)
            check(converged(o[c], 4e-6), "[586:krylov] float " + what + ": configuration " +
                                             std::to_string(c) + " converges");
    };
    for (double spread : {1.0, 10.0, 1e3, 1e4, 1e5})
        all_converge(cg_row<float>("float", 20, spread), "CG spread " + std::to_string(spread));
    all_converge(gmres_row<float>("float"), "GMRES");
}

/// posit<32,2>: the quire matters once the entries leave posit's region of
/// full precision.
void krylov_posit() {
    using T = posit32;
    // Well scaled: every configuration converges, as in float.
    auto all_converge = [](const std::array<outcome, 4>& o, double tol, const std::string& what) {
        for (int c : {0, 1, 3})
            check(converged(o[c], tol), "[586:krylov] posit32 " + what + ": configuration " +
                                            std::to_string(c) + " converges");
    };
    all_converge(cg_row<T>("posit32", 20, 1.0), 4e-6, "CG spread 1");
    all_converge(cg_row<T>("posit32", 20, 10.0), 4e-6, "CG spread 10");
    all_converge(gmres_row<T>("posit32"), 4e-6, "GMRES");
    // In between: each converges, the quire's residual lowest or near it, but
    // the iteration counts do not order (m = 16, spread 1e3: plain 97, quire
    // 126). Asserted only to converge, to the looser true residual that the
    // posit updates of x leave.
    all_converge(cg_row<T>("posit32", 20, 1e3), 1e-4, "CG spread 1e3");
    all_converge(cg_row<T>("posit32", 20, 1e4), 1e-4, "CG spread 1e4");
    // Badly scaled: the inner products reach magnitudes where posit<32,2> keeps
    // only a few fraction bits. Plain arithmetic turns NaR, fma saturates near
    // maxpos; the quire holds every product exactly and converges.
    const std::pair<std::size_t, double> bad[] = {{20, 3e4}, {30, 3e4}, {16, 1e5}, {20, 1e5}, {30, 1e5}};
    for (auto [m, spread] : bad) {
        const auto o = cg_row<T>("posit32", m, spread);
        const std::string what = "CG m " + std::to_string(m) + " spread " + std::to_string(spread);
        check(failed(o[0]), "[586:krylov] posit32 " + what + ": plain fails");
        check(failed(o[1]), "[586:krylov] posit32 " + what + ": fma fails");
        check(converged(o[3], 1e-3), "[586:krylov] posit32 " + what + ": quire converges");
    }
}

// ---------------------------------------------------------------------------
// [586:krylov][586:determinism]
// ---------------------------------------------------------------------------

} // namespace

// The serial twins of the accumulators: the same arithmetic without
// accumulator_merge, so the reductions do not split across the pool.
namespace serial_ns {

template <typename T> struct plain { T s; };
template <typename T> void accumulator_clear(plain<T>& a) { a.s = T(0); }
template <typename T, std::same_as<T> V> void accumulator_assign(plain<T>& a, const V& v) { a.s = v; }
template <typename T, std::same_as<T> V> void accumulator_add_product(plain<T>& a, const V& m, const V& v) {
    a.s += m * v;
}
template <typename T, typename R> R accumulator_value(const plain<T>& a, std::type_identity<R>) {
    return static_cast<R>(a.s);
}

template <typename T> struct quire { sw::universal::quire<T> q; };
template <typename T> void accumulator_clear(quire<T>& a) { accumulator_clear(a.q); }
template <typename T, std::same_as<T> V> void accumulator_assign(quire<T>& a, const V& v) {
    accumulator_assign(a.q, v);
}
template <typename T, std::same_as<T> V> void accumulator_add_product(quire<T>& a, const V& m, const V& v) {
    accumulator_add_product(a.q, m, v);
}
template <typename T, typename R> R accumulator_value(const quire<T>& a, std::type_identity<R> t) {
    return accumulator_value(a.q, t);
}

} // namespace serial_ns

static_assert(!mtl::math::mergeable_accumulator<serial_ns::quire<float>, float>);
static_assert(!mtl::math::mergeable_accumulator<serial_ns::plain<float>, float>);

namespace {

/// CG's dot products split across the pool when the accumulator can merge.
/// The quire's merge is exact, so the whole iteration -- every x, r, p --
/// matches the serial run bit for bit. 160000 unknowns: more than two
/// 65536-term chunks, so plain float splits too and its trajectory moves.
template <typename T>
void determinism(const char* type) {
    const auto A = laplacian_2d<T>(400, 1.0);
    const vec<T> b = rhs(A);
    auto run = [&](auto tag) {
        using Acc = decltype(tag);
        vec<T> x(A.num_rows(), T(0));
        mtl::itl::pc::identity<smat<T>> I(A);
        mtl::itl::basic_iteration<T> it(b, 10, T(0));   // exactly 10 iterations
        mtl::itl::cg<Acc>(A, x, b, I, it);
        return x;
    };
    auto same = [](const vec<T>& u, const vec<T>& v) {
        for (std::size_t i = 0; i < u.size(); ++i)
            if (std::memcmp(&u(i), &v(i), sizeof(T)) != 0) return false;
        return true;
    };
    const bool threaded = mtl::detail::thread_pool::instance().size() >= 2;
    const std::string t = type;
    check(same(run(quire_acc<T>{}), run(serial_ns::quire<T>{})),
          "[586:determinism] " + t + " CG with the quire: threaded equals serial, bit for bit");
    if constexpr (std::is_same_v<T, float>) {
        const bool plain_same = same(run(T{}), run(serial_ns::plain<T>{}));
        check(threaded ? !plain_same : plain_same,
              "[586:determinism] float CG, plain: threaded regroups the dot products");
    }
    std::printf("[586:determinism] %s CG, 160000 unknowns, %u workers: quire threaded == serial\n", type,
                mtl::detail::thread_pool::instance().size());
}

} // namespace

int main() {
    krylov_float();
    krylov_posit();
    determinism<float>("float");
    determinism<posit32>("posit32");
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_accumulation_matrix_krylov: PASS\n";
    return 0;
}
