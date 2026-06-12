// Per-function tests for sctl/math_utils.{hpp,txx}.
//
// Covers every free function in the public surface plus the QuadReal arithmetic
// wrapper. Cross-checks against <cmath> for `double` Real where applicable;
// uses known algebraic identities (pythagorean / pow round-trip) for the rest.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "sctl/common.hpp"
#include "sctl/math_utils.hpp"
#include "sctl/math_utils.txx"

#include "test-utils.hpp"

using sctl::Integer;
using sctl::Long;

int main() {
  using R = double;
  const R tol = R(1e-12);
  const R pi  = R(3.14159265358979323846);

  // --- significant_bits ---
  // SCTL counts the bits b such that 2^-b is the largest power of 2 with (2^-b + 1 != 1).
  // For IEEE-754 float that's 23 (mantissa bits, excluding the implicit leading 1);
  // for double, 52. Some implementations return one more (counting the implicit bit).
  std::printf("significant_bits:\n");
  {
    const Integer fb = sctl::significant_bits<float>();
    const Integer db = sctl::significant_bits<double>();
    CHECK(fb >= 23 && fb <= 24);
    CHECK(db >= 52 && db <= 53);
  }

  // --- machine_eps ---
  // SCTL defines machine_eps<Real>() = 2^-(significant_bits+1) — the half-step. So
  // 1 + eps may round to 1, but 1 + 2*eps definitely bumps it.
  std::printf("machine_eps:\n");
  {
    const R eps = sctl::machine_eps<R>();
    const Integer bits = sctl::significant_bits<R>();
    CHECK(eps > R(0));
    CHECK(R(1) + R(2) * eps != R(1));
    R expected = R(1);
    for (Integer i = 0; i < bits + 1; ++i) expected *= R(0.5);
    CHECK(test_utils::approx_eq(eps, expected, R(1e-30)));
  }

  // --- const_pi / const_e / const_ln2 ---
  std::printf("const_*:\n");
  {
    CHECK(test_utils::approx_eq(sctl::const_pi <R>(), R(M_PI),  tol));
    CHECK(test_utils::approx_eq(sctl::const_e  <R>(), R(M_E),   tol));
    CHECK(test_utils::approx_eq(sctl::const_ln2<R>(), R(M_LN2), tol));
  }

  // --- isinf / isnan ---
  std::printf("isinf / isnan:\n");
  {
    const R inf = std::numeric_limits<R>::infinity();
    const R nan = std::numeric_limits<R>::quiet_NaN();
    CHECK( sctl::isinf<R>( inf));
    CHECK( sctl::isinf<R>(-inf));
    CHECK(!sctl::isinf<R>(R(0)));
    CHECK( sctl::isnan<R>( nan));
    CHECK(!sctl::isnan<R>(R(0)));
  }

  // --- fabs / trunc / round / floor / ceil ---
  std::printf("fabs / trunc / round / floor / ceil:\n");
  {
    for (R x : {R(-2.7), R(-1.5), R(-0.1), R(0), R(0.1), R(1.5), R(2.7)}) {
      CHECK(test_utils::approx_eq(sctl::fabs <R>(x), std::fabs (x), tol));
      CHECK(test_utils::approx_eq(sctl::trunc<R>(x), std::trunc(x), tol));
      CHECK(test_utils::approx_eq(sctl::round<R>(x), std::round(x), tol));
      CHECK(test_utils::approx_eq(sctl::floor<R>(x), std::floor(x), tol));
      CHECK(test_utils::approx_eq(sctl::ceil <R>(x), std::ceil (x), tol));
    }
  }

  // --- sqrt / hypot ---
  std::printf("sqrt / hypot:\n");
  {
    for (R x : {R(0), R(0.5), R(1), R(2), R(100)}) {
      CHECK(test_utils::approx_eq(sctl::sqrt<R>(x), std::sqrt(x), tol));
    }
    CHECK(test_utils::approx_eq(sctl::hypot<R>(R(3), R(4)), R(5), tol));
    CHECK(test_utils::approx_eq(sctl::hypot<R>(R(0), R(0)), R(0), tol));
    // pythagorean identity
    CHECK(test_utils::approx_eq(sctl::hypot<R>(R(1), R(1)), sctl::sqrt<R>(R(2)), tol));
  }

  // --- trig ---
  std::printf("sin / cos / tan / asin / acos / atan / atan2:\n");
  {
    for (R x : {R(0), R(pi/6), R(pi/4), R(pi/3), R(pi/2)}) {
      CHECK(test_utils::approx_eq(sctl::sin<R>(x), std::sin(x), tol));
      CHECK(test_utils::approx_eq(sctl::cos<R>(x), std::cos(x), tol));
    }
    for (R x : {R(0), R(pi/6), R(pi/4), R(pi/3)}) {
      CHECK(test_utils::approx_eq(sctl::tan<R>(x), std::tan(x), tol));
    }
    for (R x : {R(-1), R(-0.5), R(0), R(0.5), R(1)}) {
      CHECK(test_utils::approx_eq(sctl::asin<R>(x), std::asin(x), tol));
      CHECK(test_utils::approx_eq(sctl::acos<R>(x), std::acos(x), tol));
    }
    for (R x : {R(-10), R(-1), R(0), R(1), R(10)}) {
      CHECK(test_utils::approx_eq(sctl::atan<R>(x), std::atan(x), tol));
    }
    CHECK(test_utils::approx_eq(sctl::atan2<R>(R(1), R(0)), R(pi/2), tol));
    CHECK(test_utils::approx_eq(sctl::atan2<R>(R(1), R(1)), R(pi/4), tol));
    // sin^2 + cos^2 = 1
    for (R x : {R(0.3), R(1.1), R(2.5)}) {
      const R s = sctl::sin<R>(x), c = sctl::cos<R>(x);
      CHECK(test_utils::approx_eq(s*s + c*c, R(1), tol));
    }
    // asin(sin(x)) = x for x in [-pi/2, pi/2]
    for (R x : {R(-1.2), R(-0.4), R(0), R(0.4), R(1.2)}) {
      CHECK(test_utils::approx_eq(sctl::asin<R>(sctl::sin<R>(x)), x, tol));
    }
  }

  // --- fmod ---
  std::printf("fmod:\n");
  {
    CHECK(test_utils::approx_eq(sctl::fmod<R>(R(5.5), R(2)), R(1.5), tol));
    CHECK(test_utils::approx_eq(sctl::fmod<R>(R(-5.5), R(2)), R(-1.5), tol));
  }

  // --- exp / log / log2 ---
  std::printf("exp / log / log2:\n");
  {
    for (R x : {R(-2), R(-0.5), R(0), R(0.5), R(2), R(5)}) {
      CHECK(test_utils::approx_eq(sctl::exp<R>(x), std::exp(x), tol * std::fabs(std::exp(x))));
    }
    for (R x : {R(0.5), R(1), R(2), R(10), R(100)}) {
      CHECK(test_utils::approx_eq(sctl::log <R>(x), std::log (x), tol));
      CHECK(test_utils::approx_eq(sctl::log2<R>(x), std::log2(x), tol));
    }
    // log(exp(x)) = x
    for (R x : {R(-1), R(0), R(1), R(3)}) {
      CHECK(test_utils::approx_eq(sctl::log<R>(sctl::exp<R>(x)), x, tol));
    }
    // log2 of powers of 2
    CHECK(test_utils::approx_eq(sctl::log2<R>(R(1)),     R(0), tol));
    CHECK(test_utils::approx_eq(sctl::log2<R>(R(8)),     R(3), tol));
    CHECK(test_utils::approx_eq(sctl::log2<R>(R(1024)), R(10), tol));
  }

  // --- pow (runtime exponent) ---
  std::printf("pow (runtime):\n");
  {
    CHECK(test_utils::approx_eq(sctl::pow<R, R>(R(2), R(10)), R(1024), tol));
    CHECK(test_utils::approx_eq(sctl::pow<R, R>(R(2), R(0.5)), std::sqrt(R(2)), tol));
    CHECK(test_utils::approx_eq(sctl::pow<R, R>(R(2), R(-2)), R(0.25), tol));
    // Integer exponent overload via int
    CHECK(test_utils::approx_eq(sctl::pow<R, int>(R(3), 4), R(81), tol));
  }

  // --- pow<Long e>(b): compile-time integer exponent ---
  std::printf("pow<e>(b) compile-time:\n");
  {
    CHECK(sctl::pow<0, R>(R(2)) == R(1));
    CHECK(sctl::pow<1, R>(R(2)) == R(2));
    CHECK(sctl::pow<5, R>(R(2)) == R(32));
    CHECK(sctl::pow<10, Long>(Long(2)) == Long(1024));
    CHECK(sctl::pow<3, Long>(Long(5)) == Long(125));
    // x^0 = 1 even for x=0 in this convention
    CHECK(sctl::pow<0, R>(R(0)) == R(1));
  }

  // --- atoreal ---
  std::printf("atoreal:\n");
  {
    CHECK(test_utils::approx_eq(sctl::atoreal<R>("3.14"),     R(3.14),    tol));
    CHECK(test_utils::approx_eq(sctl::atoreal<R>("-2.5"),     R(-2.5),    tol));
    CHECK(test_utils::approx_eq(sctl::atoreal<R>("1e3"),      R(1000),    tol));
    CHECK(test_utils::approx_eq(sctl::atoreal<R>("0"),        R(0),       tol));
  }

#ifdef SCTL_QUAD_T
  // --- QuadReal arithmetic / comparison / I/O ---
  std::printf("QuadReal:\n");
  {
    using Q = sctl::QuadReal;
    const Q a(R(2));
    const Q b(R(3));
    CHECK((double)(a + b) == R(5));
    CHECK((double)(b - a) == R(1));
    CHECK((double)(a * b) == R(6));
    CHECK((double)(b / a) == R(1.5));
    CHECK((double)(-a)    == R(-2));
    CHECK(a < b);
    CHECK(b > a);
    CHECK(a <= a);
    CHECK(a >= a);
    CHECK(a == a);
    CHECK(a != b);
    // Heterogeneous: QuadRealType op QuadReal via friend operators.
    CHECK((double)(SCTL_QUAD_T(1) + a) == R(3));
    CHECK((double)(SCTL_QUAD_T(6) / a) == R(3));
    CHECK((SCTL_QUAD_T(2) == a));
    CHECK((SCTL_QUAD_T(2) <= a));
    // std::math via QuadReal (fabs / sqrt round-trip)
    Q q(R(-3.5));
    CHECK((double)sctl::fabs<Q>(q) == R(3.5));
    CHECK(test_utils::approx_eq((double)sctl::sqrt<Q>(Q(R(9))), R(3), tol));
  }
#endif

  TEST_SUMMARY_RETURN();
}
