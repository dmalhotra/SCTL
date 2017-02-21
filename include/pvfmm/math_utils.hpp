#ifndef _MATH_UTILS_
#define _MATH_UTILS_

#include <pvfmm/common.hpp>

#include <cmath>
#include <ostream>

namespace pvfmm {

template <class Real> inline Real const_pi() { return 3.1415926535897932384626433832795028841; }

template <class Real> inline Real const_e() { return 2.7182818284590452353602874713526624977; }

template <class Real> inline Real fabs(const Real f) { return ::fabs(f); }

template <class Real> inline Real sqrt(const Real a) { return ::sqrt(a); }

template <class Real> inline Real sin(const Real a) { return ::sin(a); }

template <class Real> inline Real cos(const Real a) { return ::cos(a); }

template <class Real> inline Real exp(const Real a) { return ::exp(a); }

template <class Real> inline Real log(const Real a) { return ::log(a); }

template <class Real> inline Real pow(const Real b, const Real e) { return ::pow(b, e); }

template <Integer N, class T> constexpr T pow(const T& x) { return N > 1 ? x * pow<(N - 1) * (N > 1)>(x) : N < 0 ? T(1) / pow<(-N) * (N < 0)>(x) : N == 1 ? x : T(1); }

}  // end namespace

#ifdef PVFMM_QUAD_T

namespace pvfmm {

typedef PVFMM_QUAD_T QuadReal;

QuadReal atoquad(const char* str);
}

std::ostream& operator<<(std::ostream& output, const pvfmm::QuadReal q_);

#endif  // PVFMM_QUAD_T

#include <pvfmm/math_utils.txx>

#endif  //_MATH_UTILS_HPP_
