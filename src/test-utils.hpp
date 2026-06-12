// Shared CHECK macro and small helpers used by the per-function SCTL unit tests.
//
// Convention:
//   - One src/test-<component>.cpp per component.
//   - Each test increments a static `failures` counter via the CHECK macro and
//     prints `PASS (N check(s) failed)` at the end (return 0 on PASS).
//   - Mirrors the pattern from src/test-nodemid-vs-morton.cpp.

#ifndef _SCTL_TEST_UTILS_HPP_
#define _SCTL_TEST_UTILS_HPP_

#include <cmath>
#include <cstdio>

namespace test_utils {

static int failures = 0;

template <class T> static inline bool approx_eq(T a, T b, T tol = T(1e-12)) {
  using std::fabs;
  const T diff = fabs(a - b);
  const T mag  = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
  return diff <= tol || diff <= tol * mag;
}

}  // namespace test_utils

// Print a FAIL line + bump the failure counter when the expression is false.
// Variadic so that expressions containing commas (e.g. template instantiations
// with multiple type/value args) work without extra parentheses.
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { \
      std::printf("  FAIL @%d: %s\n", __LINE__, #__VA_ARGS__); \
      ++test_utils::failures; \
    } \
  } while (0)

// Boilerplate for `main`'s final summary line. Returns 0 if all checks passed.
#define TEST_SUMMARY_RETURN() do { \
    std::printf("\n%s (%d check(s) failed)\n", \
                test_utils::failures == 0 ? "PASS" : "FAIL", test_utils::failures); \
    return test_utils::failures == 0 ? 0 : 1; \
  } while (0)

#endif  // _SCTL_TEST_UTILS_HPP_
