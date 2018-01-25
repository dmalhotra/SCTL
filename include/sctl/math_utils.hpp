#ifndef _MATH_UTILS_
#define _MATH_UTILS_

#include SCTL_INCLUDE(common.hpp)

#include <cmath>
#include <ostream>

namespace SCTL_NAMESPACE {

template <class Real> Real atoreal(const char* str);

template <class Real> inline Real const_pi() { return (Real)3.1415926535897932384626433832795028841L; }

template <class Real> inline Real const_e() { return (Real)2.7182818284590452353602874713526624977L; }

template <class Real> inline Real fabs(const Real a) { return (Real)::fabs(a); }

template <class Real> inline Real sqrt(const Real a) { return (Real)::sqrt(a); }

template <class Real> inline Real sin(const Real a) { return (Real)::sin(a); }

template <class Real> inline Real cos(const Real a) { return (Real)::cos(a); }

template <class Real> inline Real exp(const Real a) { return (Real)::exp(a); }

template <class Real> inline Real log(const Real a) { return (Real)::log(a); }

template <class Real> inline Real pow(const Real b, const Real e) { return (Real)::pow(b, e); }

template <Integer N, class T> constexpr T pow(const T& x) { return N > 1 ? x * pow<(N - 1) * (N > 1)>(x) : N < 0 ? T(1) / pow<(-N) * (N < 0)>(x) : N == 1 ? x : T(1); }

}  // end namespace

#ifdef SCTL_QUAD_T

namespace SCTL_NAMESPACE {

typedef SCTL_QUAD_T QuadReal;

inline std::ostream& operator<<(std::ostream& output, const QuadReal q_);

}

inline std::ostream& operator<<(std::ostream& output, const SCTL_QUAD_T q_);

#endif  // SCTL_QUAD_T

#include SCTL_INCLUDE(math_utils.txx)

#endif  //_MATH_UTILS_HPP_
