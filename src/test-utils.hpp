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
#ifdef _OPENMP
#include <omp.h>
#endif

namespace test_utils {

[[maybe_unused]] static int failures = 0;  // unused in a test that wants only TrimmedOmpTeam

/**
 * An OpenMP team smaller than the one the code asks for, for as long as this object lives.
 *
 * That is what a `num_threads(SCTL_GET_MAX_THREADS())` region meets under OMP_DYNAMIC, or under an
 * OMP_THREAD_LIMIT below OMP_NUM_THREADS: `num_threads` is a request, and work split by thread id
 * over the count that was asked for then leaves the rest undone. Asking for more threads than there
 * are processors, with dynamic adjustment on, gets the same thing without an environment variable --
 * `omp_get_max_threads()` reports what was asked for while the team comes back at the processor
 * count. The previous settings are restored on the way out.
 *
 * `Trimmed()` is false where the runtime does not adjust. A test that meant to check this has then
 * checked nothing, and `Report` prints that.
 */
class TrimmedOmpTeam {
 public:
  TrimmedOmpTeam() {
#ifdef _OPENMP
    dynamic_ = omp_get_dynamic();
    requested_ = omp_get_max_threads();
    omp_set_dynamic(1);
    omp_set_num_threads(4 * omp_get_num_procs());
    asked_ = omp_get_max_threads();
    #pragma omp parallel num_threads(asked_)
    {
      #pragma omp single
      got_ = omp_get_num_threads();
    }
#endif
  }

  ~TrimmedOmpTeam() {
#ifdef _OPENMP
    omp_set_num_threads(requested_);
    omp_set_dynamic(dynamic_);
#endif
  }

  TrimmedOmpTeam(const TrimmedOmpTeam&) = delete;
  TrimmedOmpTeam& operator=(const TrimmedOmpTeam&) = delete;

  bool Trimmed() const { return got_ < asked_; }

  void Report(const char* who) const {
    if (Trimmed()) std::printf("%s with an OpenMP team of %d against the %d asked for\n", who, got_, asked_);
    else std::printf("%s: this runtime does not trim the team, so nothing was checked that way\n", who);
  }

 private:
  int dynamic_ = 0, requested_ = 1, asked_ = 1, got_ = 1;
};

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
