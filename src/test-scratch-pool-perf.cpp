// Performance experiments for sctl::ScratchPool / sctl::ScratchBuf<T>.
// Compile in release mode (the default Makefile target). Timing via the
// platform's high-resolution tick counter: __rdtscp on x86 (invariant TSC),
// CNTVCT_EL0 on AArch64, std::chrono::steady_clock elsewhere. Units are
// "ticks", not necessarily CPU cycles — Apple Silicon's CNTVCT_EL0 runs at
// 24 MHz, so absolute numbers are not comparable across architectures, but
// relative comparisons within a single run are still meaningful.
//
// We deliberately exclude the chunk-growth path: the pool is pre-warmed
// to its peak working-set size, so all measured allocations are pure
// pointer bumps inside the head chunk.

#include "sctl.hpp"

#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>           // __rdtscp
#elif !defined(__aarch64__)
#include <chrono>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sctl;

// ---------------------------------------------------------------------------
// Timing primitives
// ---------------------------------------------------------------------------

static inline uint64_t rdtscp_serialized() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  unsigned aux;
  asm volatile("" ::: "memory");
  uint64_t t = __rdtscp(&aux);
  asm volatile("" ::: "memory");
  return t;
#elif defined(__aarch64__)
  uint64_t t;
  asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(t) :: "memory");
  return t;
#else
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

struct Stats {
  uint64_t median;
  uint64_t p99;
};

static Stats summarize(std::vector<uint64_t>& v) {
  std::sort(v.begin(), v.end());
  Stats s;
  s.median = v[v.size() / 2];
  s.p99    = v[(v.size() * 99) / 100];
  return s;
}

// ---------------------------------------------------------------------------
// Experiment 1: single-thread combined alloc + free, varying size.
// ---------------------------------------------------------------------------

template <class Op>
static Stats measure(Op op, int iters) {
  // Warmup: get caches/branch predictors hot, exercise the lazy-init path.
  for (int i = 0; i < 1000; ++i) op();

  std::vector<uint64_t> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    uint64_t a = rdtscp_serialized();
    op();
    uint64_t b = rdtscp_serialized();
    samples.push_back(b - a);
  }
  return summarize(samples);
}

static void experiment_1_sizes() {
  printf("\n=== Experiment 1: single-thread alloc+free vs size ===\n");
  printf("%-12s  %14s  %14s  %14s  %14s\n",
         "n (doubles)", "ScratchBuf", "sctl::Vector", "std::vector", "malloc");
  printf("%-12s  %14s  %14s  %14s  %14s\n",
         "", "(cycles)", "(cycles)", "(cycles)", "(cycles)");

  const int iters = 100000;
  const Long sizes[] = {8, 64, 512, 4096, 32768};

  for (Long n : sizes) {
    auto s_scratch = measure([n] {
      ScratchBuf<double> buf(n);
      asm volatile("" : : "r"(&buf[0]) : "memory");
    }, iters);

    auto s_sctl_vec = measure([n] {
      Vector<double> v(n);
      asm volatile("" : : "r"(&v[0]) : "memory");
    }, iters);

    auto s_std_vec = measure([n] {
      std::vector<double> v(n);
      asm volatile("" : : "r"(v.data()) : "memory");
    }, iters);

    auto s_malloc = measure([n] {
      void* p = std::malloc(n * sizeof(double));
      asm volatile("" : : "r"(p) : "memory");
      std::free(p);
    }, iters);

    printf("%-12ld  %14lu  %14lu  %14lu  %14lu\n",
           (long)n,
           (unsigned long)s_scratch.median,
           (unsigned long)s_sctl_vec.median,
           (unsigned long)s_std_vec.median,
           (unsigned long)s_malloc.median);
  }
}

// ---------------------------------------------------------------------------
// Experiment 2: multi-thread scaling.
// ---------------------------------------------------------------------------

static void experiment_2_threads() {
  printf("\n=== Experiment 2: multi-thread alloc+free (n=512, per-thread median) ===\n");
  printf("%-8s  %14s  %14s  %14s\n", "threads", "ScratchBuf", "sctl::Vector", "std::vector");
  printf("%-8s  %14s  %14s  %14s\n", "", "(cycles)", "(cycles)", "(cycles)");

  const Long n = 512;
  const int iters = 50000;
  int max_threads = omp_get_max_threads();
  int thread_counts[] = {1, 2, 4, std::min(8, max_threads)};

  for (int T : thread_counts) {
    if (T > max_threads) continue;

    std::vector<uint64_t> per_thread_scratch(T);
    std::vector<uint64_t> per_thread_sctlvec(T);
    std::vector<uint64_t> per_thread_stdvec(T);

    auto bench_per_thread = [&](auto op, std::vector<uint64_t>& out) {
      #pragma omp parallel num_threads(T)
      {
        int tid = omp_get_thread_num();
        for (int i = 0; i < 500; ++i) op();              // warmup
        std::vector<uint64_t> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; ++i) {
          uint64_t a = rdtscp_serialized();
          op();
          uint64_t b = rdtscp_serialized();
          samples.push_back(b - a);
        }
        out[tid] = summarize(samples).median;
      }
    };

    bench_per_thread([]{
      ScratchBuf<double> buf(n);
      asm volatile("" : : "r"(&buf[0]) : "memory");
    }, per_thread_scratch);

    bench_per_thread([]{
      Vector<double> v(n);
      asm volatile("" : : "r"(&v[0]) : "memory");
    }, per_thread_sctlvec);

    bench_per_thread([]{
      std::vector<double> v(n);
      asm volatile("" : : "r"(v.data()) : "memory");
    }, per_thread_stdvec);

    auto median_of = [](std::vector<uint64_t>& v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    printf("%-8d  %14lu  %14lu  %14lu\n", T,
           (unsigned long)median_of(per_thread_scratch),
           (unsigned long)median_of(per_thread_sctlvec),
           (unsigned long)median_of(per_thread_stdvec));
  }
}

// ---------------------------------------------------------------------------
// Experiment 3: nested allocations (LIFO depth).
// ---------------------------------------------------------------------------

template <int K>
struct NestedAllocFree {
  static void run() {
    if constexpr (K == 0) {
      asm volatile("" ::: "memory");
    } else {
      ScratchBuf<double> buf(128);
      asm volatile("" : : "r"(&buf[0]) : "memory");
      NestedAllocFree<K-1>::run();
    }
  }
};

static void experiment_3_nesting() {
  printf("\n=== Experiment 3: nested LIFO depth (n=128 per level, alloc+free pair) ===\n");
  printf("%-12s  %16s\n", "depth", "median cycles");
  printf("%-12s  %16s\n", "", "(total / depth)");

  const int iters = 50000;

  {
    auto s = measure([] { NestedAllocFree<1>::run(); }, iters);
    printf("%-12d  %16lu\n", 1, (unsigned long)s.median / 1);
  }
  {
    auto s = measure([] { NestedAllocFree<4>::run(); }, iters);
    printf("%-12d  %16lu\n", 4, (unsigned long)s.median / 4);
  }
  {
    auto s = measure([] { NestedAllocFree<16>::run(); }, iters);
    printf("%-12d  %16lu\n", 16, (unsigned long)s.median / 16);
  }
}

// ---------------------------------------------------------------------------

int main() {
  // Pre-warm the global pool's head chunk so growth never fires during
  // measurement. 1 MB is the default chunk size; allocating it forces the
  // chunk into existence with pages faulted in.
  {
    ScratchBuf<char> warmup(SCTL_SCRATCH_POOL_INIT_BYTES);
    asm volatile("" : : "r"(&warmup[0]) : "memory");
  }

  printf("# ScratchPool performance microbenchmarks (release mode)\n");
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  printf("# Timer: __rdtscp (CPU cycles, invariant TSC).\n");
#elif defined(__aarch64__)
  printf("# Timer: CNTVCT_EL0 (AArch64 virtual counter ticks; e.g. 24 MHz on Apple Silicon).\n");
#else
  printf("# Timer: std::chrono::steady_clock (nanoseconds).\n");
#endif
  printf("# Methodology: 100k iters (per cell), median reported, after 1k-iter warmup.\n");
  printf("# Chunk-growth path is excluded by pre-warming to %lld bytes.\n",
         (long long)SCTL_SCRATCH_POOL_INIT_BYTES);

  experiment_1_sizes();
  experiment_2_threads();
  experiment_3_nesting();

  return 0;
}
