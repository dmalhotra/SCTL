// Tests for sctl/function-table.hpp: the error against the function at random points and at the ends, the Vec
// evaluation against the scalar one, x outside [lb, ub] and NaN, for float and double at several vector lengths.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "sctl.hpp"

#include "test-utils.hpp"

using sctl::Integer;
using sctl::Long;

template <class Real, Integer N, class Fn> static void check_table(const char* name, const Fn& f, const double lb, const double ub, const Integer digits, const Integer degree = -1) {
  using VecR = sctl::Vec<Real,N>;
  const double eps = (double)sctl::machine_eps<Real>();
  const sctl::FunctionTable<Real> T([&f](const Real x) { return f((double)x); }, (Real)lb, (Real)ub, digits, degree);
  if (degree >= 0) CHECK(T.Degree() == degree);
  CHECK(T.Pieces() > 0);

  std::vector<Real> x(4096);
  { // random points in [lb, ub] and both ends
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> u(lb, ub);
    for (auto& xi : x) xi = (Real)u(gen);
    x[0] = (Real)lb;
    x[1] = (Real)ub;
  }
  double f_max = 0;
  for (const Real xi : x) f_max = std::max(f_max, std::fabs(f((double)xi)));

  double err = 0; // largest error against f
  double err_vs = 0; // largest difference of the Vec and scalar evaluations
  for (Long i = 0; i < (Long)x.size(); i += N) {
    const VecR y = T(VecR::Load(&x[i]));
    for (Integer k = 0; k < N; k++) {
      err = std::max(err, std::fabs((double)y[k] - f((double)x[i + k])));
      err_vs = std::max(err_vs, std::fabs((double)(y[k] - T(x[i + k]))));
    }
  }
  const double tol = std::max(std::pow(10.0, -(double)digits), 8 * eps);
  CHECK(err <= 4 * tol * f_max);
  CHECK(err_vs <= 64 * eps * f_max);

  { // outside [lb, ub]: the polynomial of the nearest piece, as for the scalar evaluation; NaN for NaN
    const Real w = (Real)((ub - lb) * 1e-6);
    alignas(sizeof(VecR)) Real xo[N];
    for (Integer k = 0; k < N; k++) xo[k] = (k % 3 == 0 ? (Real)lb - w : (k % 3 == 1 ? (Real)ub + w : (Real)NAN));
    const VecR y = T(VecR::LoadAligned(xo));
    for (Integer k = 0; k < N; k++) {
      const Real s = T(xo[k]);
      if (k % 3 == 2) {
        CHECK(std::isnan(y[k]) && std::isnan(s));
      } else {
        CHECK(std::isfinite(y[k]) && std::fabs((double)(y[k] - s)) <= 64 * eps * std::max((double)std::fabs(s), f_max));
      }
    }
  }
  std::printf("  %-6s %-5s N = %2ld, digits %2ld: degree %2ld, %5ld pieces, error %.1e of max|f|\n", name, sizeof(Real) == 4 ? "float" : "double", (long)N, (long)digits, (long)T.Degree(), (long)T.Pieces(), err / f_max);
}

template <class Real, Integer N> static void check_functions(const Integer digits) {
  check_table<Real,N>("exp", [](const double x) { return std::exp(x); }, 0, 10, digits);
  check_table<Real,N>("sin", [](const double x) { return std::sin(x); }, -10, 10, digits);
  check_table<Real,N>("log", [](const double x) { return std::log(x); }, 0.1, 100, digits);
  check_table<Real,N>("runge", [](const double x) { return 1 / (1 + 25 * x * x); }, -1, 1, digits);
  check_table<Real,N>("sqrt", [](const double x) { return std::sqrt(x); }, 1e-6, 1, digits); // graded: two levels
}

int main() {
  std::printf("FunctionTable<double> :\n");
  for (const Integer digits : {6, 12, 15}) check_functions<double,8>(digits);
  check_functions<double,4>(9);
  check_functions<double,2>(9);
  for (const Integer degree : {3, 15, 31}) {
    check_table<double,8>("runge", [](const double x) { return 1 / (1 + 25 * x * x); }, -1, 1, 12, degree);
  }

  std::printf("FunctionTable<float> :\n");
  for (const Integer digits : {3, 6}) check_functions<float,16>(digits);
  check_functions<float,8>(6);
  check_functions<float,4>(6);

  TEST_SUMMARY_RETURN();
}
