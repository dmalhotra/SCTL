// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include "sctl.hpp"
#include <omp.h>
#include <iostream>

using namespace sctl;

static void test_basic() {
  ScratchPool pool;
  double* first_addr;
  {
    ScratchBuf<double> buf(10, pool);
    SCTL_ASSERT(buf.size() == 10);
    for (Long i = 0; i < buf.size(); ++i) buf[i] = (double)i;
    for (Long i = 0; i < buf.size(); ++i) SCTL_ASSERT(buf[i] == (double)i);
    first_addr = &buf[0];
  }
  SCTL_ASSERT(pool.DebugLiveCount() == 0);
  SCTL_ASSERT(pool.DebugChunkCount() == 1);

  // Second allocation reuses the same address (stack popped).
  {
    ScratchBuf<double> buf(10, pool);
    SCTL_ASSERT(&buf[0] == first_addr);
  }
  std::cout << "test_basic OK\n";
}

static void test_growth_and_shrink() {
  ScratchPool pool;
  {
    ScratchBuf<char> a(SCTL_SCRATCH_POOL_INIT_BYTES / 2, pool);
    SCTL_ASSERT(pool.DebugChunkCount() == 1);
    {
      ScratchBuf<char> b(SCTL_SCRATCH_POOL_INIT_BYTES, pool);
      SCTL_ASSERT(pool.DebugChunkCount() == 2);
    }
    // After b drops: head was the growth chunk; we retain it. Older chunk
    // is still live because a is still in scope.
    SCTL_ASSERT(pool.DebugChunkCount() == 2);
  }
  // After a drops: original (smaller) chunk freed; only the bigger one remains.
  SCTL_ASSERT(pool.DebugChunkCount() == 1);
  SCTL_ASSERT(pool.DebugLiveCount() == 0);
  std::cout << "test_growth_and_shrink OK\n";
}

// A zero-byte buffer keeps its chunk while it is alive, so a larger buffer after it does not free it
static void test_zero_bytes() {
  ScratchPool pool;
  {
    ScratchBuf<char> empty(0, pool);
    SCTL_ASSERT(empty.size() == 0);
    {
      ScratchBuf<char> big(2 * SCTL_SCRATCH_POOL_INIT_BYTES, pool);
      SCTL_ASSERT(pool.DebugChunkCount() == 2);
    }
  }
  SCTL_ASSERT(pool.DebugChunkCount() == 1);
  SCTL_ASSERT(pool.DebugLiveCount() == 0);
  std::cout << "test_zero_bytes OK\n";
}

static void test_view() {
  ScratchPool pool;
  ScratchBuf<double> buf(8, pool);
  Vector<double> v(buf);
  SCTL_ASSERT(v.Dim() == 8);
  for (Long i = 0; i < 8; ++i) v[i] = i * 2.5;
  for (Long i = 0; i < 8; ++i) SCTL_ASSERT(buf[i] == i * 2.5);
  SCTL_ASSERT(&v[0] == &buf[0]);
  std::cout << "test_view OK\n";
}

static void test_range_for() {
  ScratchPool pool;
  ScratchBuf<int> buf(6, pool);
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
    ScratchBuf<int> a(100, pool);
    int* a_end = &a[99];
    {
      ScratchBuf<int> b(50, pool);
      SCTL_ASSERT(&b[0] > a_end);
    }
    {
      ScratchBuf<int> c(30, pool);
      // After b freed, c reuses the same region.
      SCTL_ASSERT(&c[0] > a_end);
    }
  }
  std::cout << "test_lifo_nested OK\n";
}

static void test_default_ctor() {
  // 1-arg ctor uses Instance() (thread_local).
  ScratchBuf<double> buf(5);
  for (Long i = 0; i < 5; ++i) buf[i] = i;
  SCTL_ASSERT(buf[3] == 3.0);
  std::cout << "test_default_ctor OK\n";
}

static void test_multithread() {
  // Each thread uses its own thread_local pool via Instance(). No shared
  // pool object is involved; per-thread pools are independent and lock-free.
  const int iters = 5000;
  int global_ok = 1;
  #pragma omp parallel reduction(&: global_ok)
  {
    int tid = omp_get_thread_num();
    for (int it = 0; it < iters; ++it) {
      Long n = 32 + ((it + tid * 7) % 200);
      ScratchBuf<double> buf(n);
      for (Long i = 0; i < n; ++i) buf[i] = (double)(tid * 1000 + it + i);
      for (Long i = 0; i < n; ++i) {
        if (buf[i] != (double)(tid * 1000 + it + i)) { global_ok = 0; break; }
      }
    }
  }
  SCTL_ASSERT(global_ok);
  // After the parallel region: each surviving worker thread's pool should
  // be drained (all ScratchBufs above were RAII-scoped inside the loop).
  #pragma omp parallel reduction(&: global_ok)
  {
    if (ScratchPool::Instance().DebugLiveCount() != 0) global_ok = 0;
  }
  SCTL_ASSERT(global_ok);
  std::cout << "test_multithread OK\n";
}

static void test_request_resize() {
  sctl::ScratchPool pool;
  { // grows in place, so what is already stored stays where it is
    sctl::ScratchBuf<long> b(1000, pool);
    for (long i = 0; i < 1000; i++) b[i] = i;
    const long* before = &b[0];
    const long got = b.RequestResize(5000);
    SCTL_ASSERT(got == b.size() && got >= 1000);
    SCTL_ASSERT(&b[0] == before);
    for (long i = 0; i < 1000; i++) SCTL_ASSERT(b[i] == i);
    for (long i = 1000; i < got; i++) b[i] = i;          // the new room is usable
    for (long i = 0; i < got; i++) SCTL_ASSERT(b[i] == i);
  }
  { // refused below the top of the pool, and refused for a shrink
    sctl::ScratchBuf<long> a(1000, pool);
    sctl::ScratchBuf<long> b(10, pool);
    SCTL_ASSERT(a.RequestResize(5000) == 1000);          // `b` sits above `a`
    SCTL_ASSERT(b.RequestResize(2000) == 2000);          // `b` is the top one
    SCTL_ASSERT(b.RequestResize(1) == 2000);
  }
  const long n = 4 * SCTL_SCRATCH_POOL_INIT_BYTES / sizeof(long);  // past the first chunk
  { // past the room of its chunk, it moves to a larger one with what it holds; the first chunk, then empty, is released
    sctl::ScratchBuf<long> b(1000, pool);
    for (long i = 0; i < 1000; i++) b[i] = i;
    SCTL_ASSERT(b.RequestResize(n) == n);
    for (long i = 0; i < 1000; i++) SCTL_ASSERT(b[i] == i);
    for (long i = 1000; i < n; i++) b[i] = i;            // the new room is usable
    SCTL_ASSERT(pool.DebugChunkCount() == 1);
  }
  { // the pool kept the larger chunk: a buffer of the next round grows in place, the pool allocating nothing
    sctl::ScratchBuf<long> b(1000, pool);
    const long* before = &b[0];
    SCTL_ASSERT(b.RequestResize(n) == n && &b[0] == before);
    SCTL_ASSERT(pool.DebugChunkCount() == 1);
  }
  { // the top-most buffer grows also below an empty head chunk (one that a buffer above it took, and freed)
    sctl::ScratchBuf<long> a(1000, pool);
    for (long i = 0; i < 1000; i++) a[i] = i;
    { sctl::ScratchBuf<long> big(4 * n, pool); }            // a new chunk, empty again
    SCTL_ASSERT(pool.DebugChunkCount() == 2);
    SCTL_ASSERT(a.RequestResize(2 * n) == 2 * n);         // into the empty head, which holds it
    for (long i = 0; i < 1000; i++) SCTL_ASSERT(a[i] == i);
    SCTL_ASSERT(pool.DebugChunkCount() == 1);             // the chunk it left, then empty, released
  }
  { // without copy: the size, and nothing else promised
    sctl::ScratchBuf<long> b(1000, pool);
    SCTL_ASSERT(b.RequestResize(16 * n, false) == 16 * n);
    for (long i = 0; i < 16 * n; i++) b[i] = i;
  }
  std::cout << "test_request_resize OK\n";
}

static void test_reserve() {
  sctl::ScratchPool pool;
  const sctl::Long want = 8 << 20;                       // past the initial chunk, so a bigger one is needed
  { // reserving changes the pool, not the buffer it is asked through
    sctl::ScratchBuf<char> b(1024, pool);
    b.Reserve(want);
    SCTL_ASSERT(b.size() == 1024);
  }
  { // the reserved size is there now: a buffer grows into it in place, the pool allocating nothing
    sctl::ScratchBuf<char> b(1024, pool);
    const sctl::Long chunks = pool.DebugChunkCount();
    const char* before = &b[0];
    SCTL_ASSERT(b.RequestResize(want) == want && &b[0] == before);
    SCTL_ASSERT(pool.DebugChunkCount() == chunks);
  }
  { // a size the pool can serve as it stands costs nothing
    sctl::ScratchBuf<char> b(1024, pool);
    const sctl::Long chunks = pool.DebugChunkCount();
    b.Reserve(1024);
    SCTL_ASSERT(pool.DebugChunkCount() == chunks);
  }
  std::cout << "test_reserve OK\n";
}

int main() {
  test_basic();
  test_growth_and_shrink();
  test_zero_bytes();
  test_view();
  test_range_for();
  test_lifo_nested();
  test_default_ctor();
  test_multithread();
  test_request_resize();
  test_reserve();
  std::cout << "all tests passed\n";
  return 0;
}
