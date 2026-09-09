// Per-function tests for sctl/ompUtils.{hpp,txx}.
//
// Covers omp_par::merge, omp_par::merge_sort (with and without comparator),
// omp_par::reduce, omp_par::scan, omp_par::sample_sort, omp_par::radix_sort and omp_par::sort.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/morton.hpp"
#include "sctl/morton.txx"
#include "sctl/ompUtils.hpp"
#include "sctl/ompUtils.txx"

#include "test-utils.hpp"

using sctl::Integer;
using sctl::Long;

/** Radix-sortable through its own key, as MortonCode is; `tag` records the input order so a sort
 *  that claims to be stable can be held to it. */
struct RadixRec {
  static constexpr bool IntKeyIsExact = true;
  std::uint64_t k;
  Long tag;
  std::uint64_t GetIntKey() const { return k; }
  bool operator<(const RadixRec& o) const { return k < o.k; }
  bool operator>(const RadixRec& o) const { return o.k < k; }
};

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

    // 4-argument overload writes the seed too, matching the 3-argument form
    for (const Long seed : {(Long)0, (Long)100}) {
      std::vector<Long> C(10, -1);
      sctl::omp_par::scan(A.begin(), C.begin(), (Long)10, seed);
      Long acc_ = seed;
      for (Long i = 0; i < 10; ++i) {
        CHECK(C[i] == acc_);
        if (i < 9) acc_ += A[i];
      }
    }
  }

  // --- scan over a range long enough to take the multi-threaded path ---
  std::printf("scan (parallel path) :\n");
  {
    const Long N = 100 * SCTL_GET_MAX_THREADS() + 1000;
    std::vector<Long> A(N), B(N, -1), ref(N);
    for (Long i = 0; i < N; ++i) A[i] = (i % 7) + 1;
    ref[0] = 5;
    for (Long i = 1; i < N; ++i) ref[i] = ref[i - 1] + A[i - 1];
    sctl::omp_par::scan(A.begin(), B.begin(), N, (Long)5);
    for (Long i = 0; i < N; ++i) CHECK(B[i] == ref[i]);
  }

  // --- sample_sort: out-of-place, in-place, and a size past the parallel threshold ---
  std::printf("sample_sort :\n");
  {
    struct Rec { Long key; char pad[88]; };           // ~96B, like BuildNearList's NodeData
    auto by_key = [](const Rec& a, const Rec& b) { return a.key < b.key; };
    for (const Long N : {(Long)5, (Long)1000, (Long)100 * SCTL_GET_MAX_THREADS() + 4321}) {
      std::vector<Rec> A(N), B(N), ref(N);
      for (Long i = 0; i < N; ++i) A[i].key = ((i * 2654435761u) % 100003);
      ref = A;
      std::stable_sort(ref.begin(), ref.end(), [](const Rec& a, const Rec& b){ return a.key < b.key; });

      sctl::omp_par::sample_sort(A.data(), B.data(), N, by_key);   // out-of-place
      for (Long i = 0; i < N; ++i) CHECK(B[i].key == ref[i].key);
      for (Long i = 1; i < N; ++i) CHECK(B[i-1].key <= B[i].key);

      std::vector<Rec> C = A;
      sctl::omp_par::sample_sort(C.data(), C.data() + N, by_key);  // in-place overload
      for (Long i = 0; i < N; ++i) CHECK(C[i].key == ref[i].key);
    }
  }

  // --- radix_sort: against stable_sort, over the sizes and key widths the passes turn on ---
  std::printf("radix_sort :\n");
  {
    // Sizes straddle 4*2^11, the per-thread share radix_sort caps its team by. Key widths leave the
    // upper passes all-zero, which the six-pass loop has to carry through unchanged.
    for (const Long N : {(Long)0, (Long)1, (Long)2, (Long)17, (Long)8191, (Long)8192, (Long)8193,
                         (Long)100 * SCTL_GET_MAX_THREADS() + 54321}) {
      for (const Integer width : {16, 32, 64}) {
        const std::uint64_t mask = (width == 64 ? ~(std::uint64_t)0 : (((std::uint64_t)1 << width) - 1));
        std::vector<RadixRec> A((size_t)N);
        for (Long i = 0; i < N; ++i) {
          A[i].k = rng() & (i % 5 == 0 ? mask : (std::uint64_t)0xFF);  // plenty of repeated keys
          A[i].tag = i;
        }
        std::vector<RadixRec> ref = A;
        std::stable_sort(ref.begin(), ref.end(), [](const RadixRec& a, const RadixRec& b) { return a.k < b.k; });

        std::vector<RadixRec> B = A;
        sctl::omp_par::radix_sort(B.data(), N, [](const RadixRec& r) { return r.k; });
        for (Long i = 0; i < N; ++i) CHECK(B[i].k == ref[i].k);
        for (Long i = 0; i < N; ++i) CHECK(B[i].tag == ref[i].tag);  // stable: ties keep their order
      }
    }
  }

  // --- omp_par::sort: both overloads, and that it picks the radix path only where it may ---
  std::printf("omp_par::sort :\n");
  {
    static_assert(sctl::omp_par::is_radix_sortable<RadixRec>::value, "RadixRec declares an exact key");
    static_assert(!sctl::omp_par::is_radix_sortable<Long>::value, "a plain integer has no GetIntKey");

    for (const Long N : {(Long)0, (Long)1, (Long)3000, (Long)100 * SCTL_GET_MAX_THREADS() + 54321}) {
      std::vector<RadixRec> A((size_t)N);
      for (Long i = 0; i < N; ++i) {
        A[i].k = rng();
        A[i].tag = i;
      }
      std::vector<RadixRec> ref = A;
      std::sort(ref.begin(), ref.end());

      std::vector<RadixRec> B = A;  // in place
      sctl::omp_par::sort(B.data(), N);
      for (Long i = 0; i < N; ++i) CHECK(B[i].k == ref[i].k);

      std::vector<RadixRec> C((size_t)N);  // into a separate range, leaving the input alone
      sctl::omp_par::sort(sctl::Ptr2ConstItr<RadixRec>(A.data(), std::max<Long>(N, 1)),
                          sctl::Ptr2Itr<RadixRec>(C.data(), std::max<Long>(N, 1)), N);
      for (Long i = 0; i < N; ++i) CHECK(C[i].k == ref[i].k);
      for (Long i = 0; i < N; ++i) CHECK(A[i].tag == i);

      // The comparator overloads never take the radix path; a reversed order proves it is not taken.
      std::vector<RadixRec> D = A, E((size_t)N);
      sctl::omp_par::sort(D.data(), N, std::greater<RadixRec>());
      for (Long i = 0; i < N; ++i) CHECK(D[i].k == ref[N - 1 - i].k);
      sctl::omp_par::sort(sctl::Ptr2ConstItr<RadixRec>(A.data(), std::max<Long>(N, 1)),
                          sctl::Ptr2Itr<RadixRec>(E.data(), std::max<Long>(N, 1)), N, std::greater<RadixRec>());
      for (Long i = 0; i < N; ++i) CHECK(E[i].k == ref[N - 1 - i].k);
    }
  }

  // --- MortonCode, the type the radix path exists for: it must agree with operator< ---
  std::printf("omp_par::sort (MortonCode) :\n");
  {
    constexpr Integer DIM = 3;
    using MC = sctl::MortonCode<DIM>;
    if constexpr (sctl::omp_par::is_radix_sortable<MC>::value) {  // false once DIM*(MAX_DEPTH+1) > 64
      const Long N = 200000;
      std::vector<MC> A((size_t)N);
      for (Long i = 0; i < N; ++i) {
        double c[DIM];
        for (Integer k = 0; k < DIM; ++k) c[k] = (double)(rng() % 1000000) / 1000000.0;
        A[i] = MC(c);
      }
      std::vector<MC> ref = A, B = A;
      std::sort(ref.begin(), ref.end());
      sctl::omp_par::sort(B.data(), N);
      for (Long i = 0; i < N; ++i) CHECK(!(B[i] < ref[i]) && !(ref[i] < B[i]));
    }
  }

  TEST_SUMMARY_RETURN();
}
