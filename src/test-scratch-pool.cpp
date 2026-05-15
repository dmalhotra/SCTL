#include "sctl.hpp"
#include <omp.h>
#include <iostream>

using namespace sctl;

static void test_basic() {
  ScratchPool pool;
  double* first_addr;
  {
    ScratchBuf<double> buf(10, 0, pool);
    SCTL_ASSERT(buf.Dim() == 10);
    for (Long i = 0; i < buf.Dim(); ++i) buf[i] = (double)i;
    for (Long i = 0; i < buf.Dim(); ++i) SCTL_ASSERT(buf[i] == (double)i);
    first_addr = &buf[0];
  }
  SCTL_ASSERT(pool.DebugLiveCount(0) == 0);
  SCTL_ASSERT(pool.DebugChunkCount(0) == 1);

  // Second allocation reuses the same address (stack popped).
  {
    ScratchBuf<double> buf(10, 0, pool);
    SCTL_ASSERT(&buf[0] == first_addr);
  }
  std::cout << "test_basic OK\n";
}

static void test_growth_and_shrink() {
  ScratchPool pool;
  {
    ScratchBuf<char> a(SCTL_SCRATCH_POOL_INIT_BYTES / 2, 0, pool);
    SCTL_ASSERT(pool.DebugChunkCount(0) == 1);
    {
      ScratchBuf<char> b(SCTL_SCRATCH_POOL_INIT_BYTES, 0, pool);
      SCTL_ASSERT(pool.DebugChunkCount(0) == 2);
    }
    // After b drops: head was the growth chunk; we retain it. Older chunk
    // is still live because a is still in scope.
    SCTL_ASSERT(pool.DebugChunkCount(0) == 2);
  }
  // After a drops: original (smaller) chunk freed; only the bigger one remains.
  SCTL_ASSERT(pool.DebugChunkCount(0) == 1);
  SCTL_ASSERT(pool.DebugLiveCount(0) == 0);
  std::cout << "test_growth_and_shrink OK\n";
}

static void test_view() {
  ScratchPool pool;
  ScratchBuf<double> buf(8, 0, pool);
  Vector<double> v(buf);
  SCTL_ASSERT(v.Dim() == 8);
  for (Long i = 0; i < 8; ++i) v[i] = i * 2.5;
  for (Long i = 0; i < 8; ++i) SCTL_ASSERT(buf[i] == i * 2.5);
  SCTL_ASSERT(&v[0] == &buf[0]);
  std::cout << "test_view OK\n";
}

static void test_range_for() {
  ScratchPool pool;
  ScratchBuf<int> buf(6, 0, pool);
  int k = 0;
  for (auto it = buf.begin(); it != buf.end(); ++it) *it = k++;
  k = 0;
  for (auto it = buf.begin(); it != buf.end(); ++it) SCTL_ASSERT(*it == k++);
  SCTL_ASSERT(k == 6);
  std::cout << "test_range_for OK\n";
}

static void test_lifo_nested() {
  ScratchPool pool;
  {
    ScratchBuf<int> a(100, 0, pool);
    int* a_end = &a[99];
    {
      ScratchBuf<int> b(50, 0, pool);
      SCTL_ASSERT(&b[0] > a_end);
    }
    {
      ScratchBuf<int> c(30, 0, pool);
      // After b freed, c reuses the same region.
      SCTL_ASSERT(&c[0] > a_end);
    }
  }
  std::cout << "test_lifo_nested OK\n";
}

static void test_default_ctor() {
  // 1-arg ctor uses Instance() + omp_get_thread_num() (which is 0 outside parallel)
  {
    ScratchBuf<double> buf(5);
    for (Long i = 0; i < 5; ++i) buf[i] = i;
    SCTL_ASSERT(buf[3] == 3.0);
  }
  // 2-arg ctor — explicit tid, Instance().
  {
    ScratchBuf<double> buf(5, 0);
    for (Long i = 0; i < 5; ++i) buf[i] = i + 100;
    SCTL_ASSERT(buf[3] == 103.0);
  }
  std::cout << "test_default_ctor OK\n";
}

static void test_multithread() {
  ScratchPool pool;
  const int iters = 5000;
  int global_ok = 1;
  #pragma omp parallel reduction(&: global_ok)
  {
    int tid = omp_get_thread_num();
    for (int it = 0; it < iters; ++it) {
      Long n = 32 + ((it + tid * 7) % 200);
      ScratchBuf<double> buf(n, tid, pool);
      for (Long i = 0; i < n; ++i) buf[i] = (double)(tid * 1000 + it + i);
      for (Long i = 0; i < n; ++i) {
        if (buf[i] != (double)(tid * 1000 + it + i)) { global_ok = 0; break; }
      }
    }
  }
  SCTL_ASSERT(global_ok);
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    SCTL_ASSERT(pool.DebugLiveCount(tid) == 0);
  }
  std::cout << "test_multithread OK\n";
}

int main() {
  test_basic();
  test_growth_and_shrink();
  test_view();
  test_range_for();
  test_lifo_nested();
  test_default_ctor();
  test_multithread();
  std::cout << "all tests passed\n";
  return 0;
}
