#pragma once
// mp-iterative -- accumulator-strategy selection helpers.
//
// Shared composition-layer utilities for choosing, per value type, which of the
// three accumulation strategies are available when running an accumulator-aware
// MTL5 kernel (smoother or Krylov solver):
//
//   naive -- accumulate in the value type itself (Accumulator = void)
//   fma   -- fused multiply-add, one rounding per term (mtl5 #259); needs an
//            ADL-found fma (std::fma for built-ins; Universal provides one for
//            posit and cfloat)
//   quire -- exact super-accumulation, single rounding at the end. Quires exist
//            for any fixed-size arithmetic (integer, fixpnt, posit, lns,
//            cfloat); the composition layer currently specializes
//            accumulator_traits for the posit quire (see
//            mtl/math/quire_accumulator.hpp), with more to follow.
//
// These are the capability predicates the demo applications and the
// benchmarking harness use to gate columns; centralized here so the gating
// logic lives in one place.

#include <concepts>
#include <type_traits>

#include <universal/number/posit/posit.hpp>

namespace sw::mp_iterative {

/// The quire super-accumulator type for a value type, or `void` when the
/// composition layer has no accumulator_traits specialization for it yet.
/// Any fixed-size arithmetic admits a quire; posit is the first wired up.
template <typename T>
struct quire_of { using type = void; };

template <unsigned nbits, unsigned es>
struct quire_of<sw::universal::posit<nbits, es>> {
    using type = sw::universal::quire<sw::universal::posit<nbits, es>>;
};

template <typename T>
using quire_of_t = typename quire_of<T>::type;

/// True when the composition layer can build a quire super-accumulator for T.
template <typename T>
inline constexpr bool has_quire = !std::is_void_v<quire_of_t<T>>;

/// True when T has a fused multiply-add usable by mtl::math::fma_accumulator:
/// std::fma for built-in floating point, or an ADL-found fma for a custom
/// arithmetic type (Universal supplies one for posit and cfloat).
template <typename T>
concept has_fused_multiply_add =
    std::is_floating_point_v<T> ||
    requires (T a) { { fma(a, a, a) } -> std::convertible_to<T>; };

} // namespace sw::mp_iterative
