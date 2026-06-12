// Per-function tests for sctl/mem_mgr.{hpp,txx}.
//
// Covers the user-facing API:
//   - aligned_new<T>(n) / aligned_delete(p): default and via custom MemoryManager.
//   - MemoryManager(N): construct a pooled allocator backed by an N-byte buffer.
//   - MemoryManager::glbMemMgr(): the process-wide default pool.
//   - Round-trip writes; per-allocation alignment; non-overlap of separate allocs.

#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/mem_mgr.hpp"
#include "sctl/mem_mgr.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Iterator;
using sctl::MemoryManager;

template <class T> static inline bool is_aligned(const void* p, std::size_t a) {
  return ((std::uintptr_t)p & (a - 1)) == 0;
}

int main() {
  // --- glbMemMgr() returns a stable reference ---
  std::printf("glbMemMgr stable :\n");
  {
    const MemoryManager& m1 = MemoryManager::glbMemMgr();
    const MemoryManager& m2 = MemoryManager::glbMemMgr();
    CHECK(&m1 == &m2);
  }

  // --- aligned_new / aligned_delete round-trip on built-ins ---
  std::printf("aligned_new int :\n");
  {
    const Long N = 17;
    Iterator<int> p = sctl::aligned_new<int>(N);
    CHECK(is_aligned<int>(&p[0], SCTL_MEM_ALIGN));
    for (Long i = 0; i < N; ++i) p[i] = (int)(i * 7 - 3);
    for (Long i = 0; i < N; ++i) CHECK(p[i] == (int)(i * 7 - 3));
    sctl::aligned_delete(p);
  }
  std::printf("aligned_new double :\n");
  {
    const Long N = 33;
    Iterator<double> p = sctl::aligned_new<double>(N);
    CHECK(is_aligned<double>(&p[0], SCTL_MEM_ALIGN));
    for (Long i = 0; i < N; ++i) p[i] = (double)i / 8.0;
    for (Long i = 0; i < N; ++i) CHECK(p[i] == (double)i / 8.0);
    sctl::aligned_delete(p);
  }

  // --- aligned_new<T>(1) (the default size) ---
  std::printf("aligned_new<T>() default :\n");
  {
    Iterator<long> p = sctl::aligned_new<long>();
    *p = 123456789L;
    CHECK(*p == 123456789L);
    sctl::aligned_delete(p);
  }

  // --- aligned_new on a custom-sized MemoryManager pool ---
  std::printf("custom MemoryManager :\n");
  {
    MemoryManager local(1024 * 1024);  // 1 MB pool
    Iterator<double> a = sctl::aligned_new<double>(100, &local);
    Iterator<int>    b = sctl::aligned_new<int>   (200, &local);
    CHECK(is_aligned<double>(&a[0], SCTL_MEM_ALIGN));
    CHECK(is_aligned<int>   (&b[0], SCTL_MEM_ALIGN));
    for (Long i = 0; i < 100; ++i) a[i] = (double)i;
    for (Long i = 0; i < 200; ++i) b[i] = (int)(i + 1000);
    for (Long i = 0; i < 100; ++i) CHECK(a[i] == (double)i);
    for (Long i = 0; i < 200; ++i) CHECK(b[i] == (int)(i + 1000));
    sctl::aligned_delete(b, &local);
    sctl::aligned_delete(a, &local);
  }

  // --- multiple allocations on the global pool are non-overlapping ---
  std::printf("non-overlapping allocs :\n");
  {
    const Long K = 10;
    const Long n = 32;
    std::vector<Iterator<char>> ptrs;
    ptrs.reserve(K);
    for (Long k = 0; k < K; ++k) ptrs.push_back(sctl::aligned_new<char>(n));
    // pairwise non-overlap
    for (Long i = 0; i < K; ++i) {
      for (Long j = i + 1; j < K; ++j) {
        const char* a_beg = &ptrs[i][0]; const char* a_end = a_beg + n;
        const char* b_beg = &ptrs[j][0]; const char* b_end = b_beg + n;
        const bool overlap = !(a_end <= b_beg || b_end <= a_beg);
        CHECK(!overlap);
      }
    }
    // free in reverse order
    for (Long k = K - 1; k >= 0; --k) sctl::aligned_delete(ptrs[k]);
  }

  // --- alignment check across a range of sizes ---
  std::printf("alignment across sizes :\n");
  {
    for (Long n : {Long(1), Long(2), Long(7), Long(8), Long(15), Long(16), Long(127),
                   Long(128), Long(1023), Long(1024), Long(8192)}) {
      Iterator<double> p = sctl::aligned_new<double>(n);
      CHECK(is_aligned<double>(&p[0], SCTL_MEM_ALIGN));
      // Write to first and last (same slot when n==1) to verify the allocation
      // honors the requested size and the memory is writable.
      p[0] = (double)n;
      CHECK(p[0] == (double)n);
      if (n > 1) {
        p[n - 1] = (double)(-n);
        CHECK(p[n - 1] == (double)(-n));
      }
      sctl::aligned_delete(p);
    }
  }

  // --- non-trivial type: ctor invocation verified via default member init ---
  // aligned_new<T>(N) on a non-trivial T calls placement-new with `T()` on each
  // slot; we confirm this via the default member initializer (x = 42) which is
  // only applied by the constructor. Note: aligned_new uses an OpenMP
  // parallel-for so a counter-style check would race; checking the per-element
  // post-state is race-free.
  std::printf("non-trivial type ctor :\n");
  {
    struct S { int x = 42; };
    Iterator<S> p = sctl::aligned_new<S>(10);
    for (Long i = 0; i < 10; ++i) CHECK(p[i].x == 42);
    sctl::aligned_delete(p);
  }

  TEST_SUMMARY_RETURN();
}
