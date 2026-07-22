#pragma once
// mp-iterative -- mixed-precision iterative methods (MTL5 + Universal)
//
// Header-only composition layer. As shared utilities emerge (Krylov solver
// harnesses, mixed-precision preconditioner adapters, convergence trackers),
// they live under sw::mp_iterative. For now this header carries only version
// metadata.

namespace sw::mp_iterative {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

} // namespace sw::mp_iterative
