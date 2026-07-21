// Timing benchmark with sort/rest breakdown. Compares three linear-tree implementations:
//   1. sctl::Tree<DIM>::UpdateRefinement                   (CPU, distributed-capable)
//   2. gpu_tree::GPUTree<Real,DIM>::buildTree              (CPU std::vector)
//   3. gpu_tree::GPUTree<Real,DIM>::buildTree              (GPU thrust::device_vector)
//
// For each: total time, isolated sort time (Morton-encode + sort on the same N codes), and
// "rest" = total - sort.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 15
#endif

#include <sctl/common.hpp>
#include <sctl/comm.hpp>
#include <sctl/comm.txx>
#include <sctl/iterator.hpp>
#include <sctl/iterator.txx>
#include <sctl/vector.hpp>
#include <sctl/vector.txx>
#include <sctl/morton.hpp>
#include <sctl/morton.txx>
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include <sctl/ompUtils.hpp>
#include <sctl/ompUtils.txx>
#include <sctl/scratch_pool.hpp>
#include <sctl/scratch_pool.txx>

#include "sctl/experimental/gpu-tree.hpp"

using Real = double;
constexpr int kDim = 3;
using GMorton = sctl::MortonCode<kDim>;
using GNode   = sctl::Morton<kDim>;
using GPUTree = gpu_tree::GPUTree<Real, kDim>;
using GLong   = gpu_tree::Long;
using SMorton = sctl::Morton<kDim>;
using SLong   = sctl::Long;

template <class V> static double ms(V t0, V t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Note: the chunked sctl-style parallel walk that used to live here has been moved into
// `gpu_tree::GPUTree::buildTree` as its CPU implementation (`detail::buildTreeCpuChunked`
// in gpu-tree.txx). The "gpu_tree::GPUTree (CPU)" row below now exercises that code path.

// Run fn() once as warmup (untimed), then time best-of-nruns. Each invocation of fn returns
// elapsed ms for that run.
template <class Fn> static double best_of_with_warmup(int nruns, Fn fn) {
  (void)fn();  // warmup
  double best = 1e30;
  for (int i = 0; i < nruns; ++i) best = std::min(best, fn());
  return best;
}

int main(int argc, char** argv) {
  const GLong N      = (argc > 1) ? std::stoll(argv[1]) : 1'000'000;
  const GLong M      = (argc > 2) ? std::stoll(argv[2]) : 4;
  const int   nruns  = (argc > 3) ? std::stoi(argv[3])  : 3;

  std::printf("=== gpu_tree vs sctl::Tree (\"rest\" only — sort time excluded) ===\n");
  std::printf("N=%lld  M=%lld  DIM=%d  MAX_DEPTH=%d  nruns=%d (best-of, 1 warmup discarded)\n\n",
              (long long)N, (long long)M, kDim, (int)gpu_tree::MAX_DEPTH, nruns);

  // Random coords (identical for all).
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<Real> U(0.0, 1.0);
  std::vector<Real> coord_std(N * kDim);
  for (Real& x : coord_std) x = U(rng);
  sctl::Vector<Real> coord_sctl(N * kDim);
  for (GLong i = 0; i < N * kDim; ++i) coord_sctl[i] = coord_std[i];

  struct Row { const char* name; double total; double sort; double rest; size_t size; };
  std::vector<Row> rows;

  // ===== sctl::Tree (UpdateRefinement on an empty tree, CPU) =====
  // Re-added: tree.txx now uses the chunked parallel walk, so this row is no longer
  // multi-second at large N and is a useful apples-to-apples vs gpu_tree::GPUTree (CPU).
  {
    sctl::Comm comm = sctl::Comm::Self();
    // Match what UpdateRefinement actually does internally (MortonCode, 8 B/elem) — NOT
    // sctl::Morton (16 B). Sorting Morton would over-state the sort cost by ~2x.
    sctl::Vector<sctl::MortonCode<kDim>> pt_mid_pre(N);
    for (GLong i = 0; i < N; ++i) pt_mid_pre[i] = sctl::MortonCode<kDim>(&coord_sctl[i * kDim]);
    const double t_sort = best_of_with_warmup(nruns, [&] {
      sctl::Vector<sctl::MortonCode<kDim>> pt = pt_mid_pre, sorted;
      auto t0 = std::chrono::steady_clock::now();
      comm.HyperQuickSort(pt, sorted);
      auto t1 = std::chrono::steady_clock::now();
      return ms(t0, t1);
    });
    size_t sz = 0;
    const double t_total = best_of_with_warmup(nruns, [&] {
      sctl::Tree<kDim> tree;  // empty: fresh build every call (matches the "empty" semantics)
      auto t0 = std::chrono::steady_clock::now();
      tree.UpdateRefinement(coord_sctl, M);
      auto t1 = std::chrono::steady_clock::now();
      sz = tree.GetNodeMID().Dim();
      return ms(t0, t1);
    });
    rows.push_back({"sctl::Tree::UpdateRefinement (empty, CPU)", t_total, t_sort, t_total - t_sort, sz});
  }

  // ===== gpu_tree CPU ===============================================
  // (This is the same chunked sctl-style parallel walk that used to live in
  // `buildTreeSctlStyleParallel`; now lives in `gpu_tree::detail::buildTreeCpuChunked`.)
  {
    std::vector<GMorton> pt_mid_pre(N);
    for (GLong i = 0; i < N; ++i) pt_mid_pre[i] = GMorton(&coord_std[i * kDim]);

    const double t_sort = best_of_with_warmup(nruns, [&] {
      std::vector<GMorton> pt = pt_mid_pre;
      auto t0 = std::chrono::steady_clock::now();
      sctl::omp_par::merge_sort(pt.begin(), pt.end());
      auto t1 = std::chrono::steady_clock::now();
      return ms(t0, t1);
    });
    size_t sz = 0;
    std::vector<GNode> tree;  // reused
    const double t_total = best_of_with_warmup(nruns, [&] {
      auto t0 = std::chrono::steady_clock::now();
      GPUTree::buildTree(tree, coord_std, M);
      auto t1 = std::chrono::steady_clock::now();
      sz = tree.size();
      return ms(t0, t1);
    });
    rows.push_back({"gpu_tree::GPUTree (CPU std::vector)", t_total, t_sort, t_total - t_sort, sz});
  }

  // ===== gpu_tree GPU ===============================================
  {
    thrust::device_vector<Real> coord_d(coord_std.begin(), coord_std.end());
    cudaDeviceSynchronize();
    thrust::device_vector<GMorton> pt_pre(N);
    gpu_tree::detail::MakeMortonFunctor<Real, kDim> mk{thrust::raw_pointer_cast(coord_d.data())};
    thrust::transform(thrust::counting_iterator<GLong>(0),
                      thrust::counting_iterator<GLong>(N),
                      pt_pre.begin(), mk);
    cudaDeviceSynchronize();

    const double t_sort = best_of_with_warmup(nruns, [&] {
      thrust::device_vector<GMorton> pt = pt_pre;
      cudaDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      thrust::sort(pt.begin(), pt.end());
      cudaDeviceSynchronize();
      auto t1 = std::chrono::steady_clock::now();
      return ms(t0, t1);
    });
    size_t sz = 0;
    thrust::device_vector<GNode> tree_d;  // reused
    const double t_total = best_of_with_warmup(nruns, [&] {
      cudaDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      GPUTree::buildTree(tree_d, coord_d, M);
      cudaDeviceSynchronize();
      auto t1 = std::chrono::steady_clock::now();
      sz = tree_d.size();
      return ms(t0, t1);
    });
    rows.push_back({"gpu_tree::GPUTree (GPU device_vector)", t_total, t_sort, t_total - t_sort, sz});
  }

  // Print: rest only, side-by-side.
  std::printf("%-50s %10s   %10s   %10s   %s\n",
              "implementation", "rest ms", "(sort ms)", "(total ms)", "size");
  std::printf("%-50s %10s   %10s   %10s   %s\n",
              "--------------------------------------------------",
              "----------", "----------", "----------", "--------");
  for (const auto& r : rows) {
    std::printf("%-50s %10.2f   %10.2f   %10.2f   %zu\n",
                r.name, r.rest, r.sort, r.total, r.size);
  }
  std::printf("\n");
  return 0;
}
