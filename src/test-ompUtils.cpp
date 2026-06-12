// Per-function tests for sctl/ompUtils.{hpp,txx}.
//
// Covers omp_par::merge, omp_par::merge_sort (with and without comparator),
// omp_par::reduce, and omp_par::scan.

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/ompUtils.hpp"
#include "sctl/ompUtils.txx"

#include "test-utils.hpp"

using sctl::Long;

int main() {
  std::mt19937_64 rng(42);

  // --- merge_sort (default less) ---
  std::printf("merge_sort default :\n");
  {
    std::vector<int> v;
    std::uniform_int_distribution<int> U(-1000, 1000);
    for (int i = 0; i < 5000; ++i) v.push_back(U(rng));
    std::vector<int> ref = v;
    std::sort(ref.begin(), ref.end());
    sctl::omp_par::merge_sort(v.begin(), v.end());
    for (size_t i = 0; i < v.size(); ++i) CHECK(v[i] == ref[i]);
  }

  // --- merge_sort with custom comparator (descending) ---
  std::printf("merge_sort comparator :\n");
  {
    std::vector<int> v;
    std::uniform_int_distribution<int> U(-1000, 1000);
    for (int i = 0; i < 5000; ++i) v.push_back(U(rng));
    std::vector<int> ref = v;
    std::sort(ref.begin(), ref.end(), std::greater<int>());
    sctl::omp_par::merge_sort(v.begin(), v.end(), std::greater<int>());
    for (size_t i = 0; i < v.size(); ++i) CHECK(v[i] == ref[i]);
  }

  // --- merge_sort on already-sorted + reverse-sorted + duplicates ---
  std::printf("merge_sort edge cases :\n");
  {
    std::vector<int> a(2000);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int)i;  // already sorted
    sctl::omp_par::merge_sort(a.begin(), a.end());
    for (size_t i = 0; i < a.size(); ++i) CHECK(a[i] == (int)i);

    std::vector<int> b(2000);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (int)(b.size() - i);  // reversed
    sctl::omp_par::merge_sort(b.begin(), b.end());
    for (size_t i = 0; i + 1 < b.size(); ++i) CHECK(b[i] <= b[i + 1]);

    std::vector<int> c(2000, 7);  // all duplicates
    sctl::omp_par::merge_sort(c.begin(), c.end());
    for (int v : c) CHECK(v == 7);
  }

  // --- merge: merging two sorted ranges yields a sorted range ---
  std::printf("merge :\n");
  {
    std::vector<int> A, B;
    std::uniform_int_distribution<int> U(-500, 500);
    for (int i = 0; i < 1000; ++i) A.push_back(U(rng));
    for (int i = 0; i < 1500; ++i) B.push_back(U(rng));
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
    std::vector<int> C(A.size() + B.size());
    sctl::omp_par::merge(A.begin(), A.end(), B.begin(), B.end(), C.begin(),
                         /* p = parallelism */ 4, std::less<int>());
    // C is sorted and contains the multiset union of A and B
    for (size_t i = 0; i + 1 < C.size(); ++i) CHECK(C[i] <= C[i + 1]);
    std::vector<int> R = A;
    R.insert(R.end(), B.begin(), B.end());
    std::sort(R.begin(), R.end());
    for (size_t i = 0; i < C.size(); ++i) CHECK(C[i] == R[i]);
  }

  // --- reduce (sum) ---
  std::printf("reduce :\n");
  {
    std::vector<Long> v(1000);
    std::uniform_int_distribution<Long> U(-100, 100);
    Long ref_sum = 0;
    for (size_t i = 0; i < v.size(); ++i) { v[i] = U(rng); ref_sum += v[i]; }
    Long s = sctl::omp_par::reduce(v.begin(), (Long)v.size());
    CHECK(s == ref_sum);

    // single-element
    std::vector<Long> v1{42};
    CHECK(sctl::omp_par::reduce(v1.begin(), (Long)1) == 42);
  }

  // --- scan (exclusive prefix sum with caller-supplied seed in B[0]) ---
  std::printf("scan :\n");
  {
    std::vector<Long> A(10), B(10);
    for (Long i = 0; i < 10; ++i) A[i] = i + 1;  // 1,2,3,...,10
    B[0] = 0;  // seed
    sctl::omp_par::scan(A.begin(), B.begin(), (Long)10);
    // B[i] = 0 + A[0] + A[1] + ... + A[i-1]
    Long acc = 0;
    for (Long i = 0; i < 10; ++i) {
      CHECK(B[i] == acc);
      if (i < 9) acc += A[i];
    }

    // With a non-zero seed
    B[0] = 100;
    sctl::omp_par::scan(A.begin(), B.begin(), (Long)10);
    acc = 100;
    for (Long i = 0; i < 10; ++i) {
      CHECK(B[i] == acc);
      if (i < 9) acc += A[i];
    }

    // Convert a count array to displacement array (canonical use case)
    std::vector<Long> cnt = {3, 5, 1, 4, 2};
    std::vector<Long> dsp(cnt.size());
    dsp[0] = 0;
    sctl::omp_par::scan(cnt.begin(), dsp.begin(), (Long)cnt.size());
    Long expected[] = {0, 3, 8, 9, 13};
    for (size_t i = 0; i < cnt.size(); ++i) CHECK(dsp[i] == expected[i]);
  }

  TEST_SUMMARY_RETURN();
}
