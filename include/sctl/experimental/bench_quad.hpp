#ifndef _SCTL_BENCH_QUAD_HPP_
#define _SCTL_BENCH_QUAD_HPP_

// Lightweight, opt-in phase timers for the QuadElemList self/near hot loops.
//
// All instrumentation is gated on the BENCH_QUAD macro. When BENCH_QUAD is NOT
// defined the BENCH_TIC/BENCH_TOC/BENCH_FLOPS/BENCH_NEAR macros expand to nothing,
// so the instrumented translation units compile byte-identical to the
// uninstrumented build (zero runtime cost). The Reset()/Report() helpers always
// exist so the benchmark driver links in both builds.
//
// Build instrumented:   make BENCH=1 bin/bench-quad-interac
// Run single-threaded:  OMP_NUM_THREADS=1 ./bin/bench-quad-interac

#include <array>
#include <cstdio>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
#endif

namespace sctl {
namespace bench {

// Phases of the self/near setup. Each phase includes the allocation of the
// temporaries it consumes (per-call malloc/free churn shows up inside its owning
// phase rather than as a separate line). NumPhases must stay last.
enum class Phase {
  InterpBuild = 0, // adaptive Mu_local/Mv_local (LagrangeInterp + GEMM + Transpose)
  GeomTensor,      // coord_shift + 3 geometry EvalTensorProduct calls (6 GEMMs)
  Assembly,        // Xsrc/Xnsrc/wq alloc + per-node normal/area/weight loop
  KernelEval,      // ker.KernelMatrix
  KernelWeight,    // KWc alloc + weighting loop
  Projection,      // projection EvalTensorProduct + scatter (2 GEMMs)
  QuadtreeBuild,   // BuildNearLeaves (adaptive near only)
  ClosestPoint,    // GetClosestPoint Newton/grid search (RectPolar near only)
  ClosestNode,     // GetClosestNode brute-force seed (adaptive near only)
  NumPhases
};

inline constexpr int kNumPhases = static_cast<int>(Phase::NumPhases);
inline constexpr int kMaxThreads = 256;

inline const char* PhaseName(Phase p) {
  switch (p) {
    case Phase::InterpBuild:   return "InterpBuild";
    case Phase::GeomTensor:    return "GeomTensor";
    case Phase::Assembly:      return "Assembly";
    case Phase::KernelEval:    return "KernelEval";
    case Phase::KernelWeight:  return "KernelWeight";
    case Phase::Projection:    return "Projection";
    case Phase::QuadtreeBuild: return "QuadtreeBuild";
    case Phase::ClosestPoint:  return "ClosestPoint";
    case Phase::ClosestNode:   return "ClosestNode";
    default:                   return "?";
  }
}

// Per-thread row of [time, count] keyed by phase, plus a running count of the
// "real" GEMM flops actually issued (interpolation/projection tensor products);
// padded to avoid false sharing.
struct PhaseRow {
  double t[kNumPhases];
  long   n[kNumPhases];
  double gemm_flops;
  long   near_leaves;    // total quadtree leaves summed over adaptive-near targets
  long   near_targets;   // number of adaptive-near NearInteracBlockGraded calls
  long   near_max_depth; // deepest subdivision reached across those targets
  char   pad[64];
};

// Single global table; index by OpenMP thread id. Defined inline (C++17) so the
// header can be included by multiple TUs without a separate .cpp.
inline std::array<PhaseRow, kMaxThreads>& Table() {
  static std::array<PhaseRow, kMaxThreads> table{};
  return table;
}

inline double Wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

inline int ThreadId() {
#ifdef _OPENMP
  const int t = omp_get_thread_num();
  return (t < kMaxThreads ? t : 0);
#else
  return 0;
#endif
}

inline void Accum(Phase p, double dt) {
  PhaseRow& row = Table()[ThreadId()];
  row.t[static_cast<int>(p)] += dt;
  row.n[static_cast<int>(p)] += 1;
}

inline void AccumFlops(double f) {
  Table()[ThreadId()].gemm_flops += f;
}

inline void AccumNear(long nleaf, long depth) {
  PhaseRow& row = Table()[ThreadId()];
  row.near_leaves += nleaf;
  row.near_targets += 1;
  if (depth > row.near_max_depth) row.near_max_depth = depth;
}

inline double TotalFlops() {
  double f = 0;
  for (int th = 0; th < kMaxThreads; th++) f += Table()[th].gemm_flops;
  return f;
}

inline void Reset() {
  Table() = std::array<PhaseRow, kMaxThreads>{};
}

// Reduce across threads and print a per-phase breakdown. `label` tags the block;
// `outer_seconds` is the wall time of the whole timed region (for a "measured
// vs. total" coverage check).
inline void Report(const std::string& label, double outer_seconds = 0) {
  double t[kNumPhases] = {};
  long   n[kNumPhases] = {};
  long   near_leaves = 0, near_targets = 0, near_max_depth = 0;
  for (int th = 0; th < kMaxThreads; th++) {
    for (int p = 0; p < kNumPhases; p++) {
      t[p] += Table()[th].t[p];
      n[p] += Table()[th].n[p];
    }
    near_leaves += Table()[th].near_leaves;
    near_targets += Table()[th].near_targets;
    if (Table()[th].near_max_depth > near_max_depth) near_max_depth = Table()[th].near_max_depth;
  }
  double sum = 0;
  for (int p = 0; p < kNumPhases; p++) sum += t[p];

#ifdef BENCH_QUAD
  std::printf("  [bench] %s  (summed over threads; phase total = %.4g s)\n", label.c_str(), sum);
  std::printf("    %-14s %12s %10s %14s\n", "phase", "time[s]", "calls", "%phase");
  for (int p = 0; p < kNumPhases; p++) {
    const double pct = (sum > 0 ? 100.0 * t[p] / sum : 0.0);
    std::printf("    %-14s %12.5e %10ld %13.1f%%\n",
                PhaseName(static_cast<Phase>(p)), t[p], n[p], pct);
  }
  const double gflops = TotalFlops() * 1e-9;
  std::printf("    %-14s %12.5e GFLOP issued in tensor GEMMs\n", "gemm_flops", gflops);
  if (near_targets > 0) {
    std::printf("    %-14s %12.3f leaves/target  (max_depth=%ld over %ld adaptive-near targets)\n",
                "near_quadtree", (double)near_leaves / near_targets, near_max_depth, near_targets);
  }
  if (outer_seconds > 0) {
    std::printf("    %-14s %12.5e   (phase sum is %.1f%% of outer wall time)\n",
                "outer", outer_seconds, 100.0 * sum / outer_seconds);
    std::printf("    %-14s %12.4f GFLOP/s (true GEMM throughput vs outer wall)\n",
                "gemm_f/s", gflops / outer_seconds);
  }
#else
  (void)t; (void)n; (void)sum; (void)label; (void)outer_seconds;
  std::printf("  [bench] %s: instrumentation disabled (rebuild with -DBENCH_QUAD)\n", label.c_str());
#endif
}

} // namespace bench
} // namespace sctl

// Hot-path macros: real code only under BENCH_QUAD, otherwise nothing.
#ifdef BENCH_QUAD
#define BENCH_TIC(phase) const double _bench_t0_##phase = ::sctl::bench::Wtime()
#define BENCH_TOC(phase) ::sctl::bench::Accum(::sctl::bench::Phase::phase, ::sctl::bench::Wtime() - _bench_t0_##phase)
#define BENCH_FLOPS(n) ::sctl::bench::AccumFlops((double)(n))
#define BENCH_NEAR(nleaf, depth) ::sctl::bench::AccumNear((long)(nleaf), (long)(depth))
#else
#define BENCH_TIC(phase) ((void)0)
#define BENCH_TOC(phase) ((void)0)
#define BENCH_FLOPS(n) ((void)0)
#define BENCH_NEAR(nleaf, depth) ((void)0)
#endif

#endif // _SCTL_BENCH_QUAD_HPP_
