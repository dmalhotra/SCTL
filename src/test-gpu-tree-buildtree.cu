// Verifies GPUTree::buildTree compile-time dispatch:
//   thrust::device_vector  -> GPU path (thrust)
//   std::vector            -> CPU path (omp_par, chunked walk)
// Runs each path on N random particles, with and without sort_scatter_index, and checks the
// output leaf node sequence is sorted in (code, depth) lex order and the scatter index (when
// requested) is a permutation of [0..N).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 20
#endif

#include "sctl/experimental/gpu-tree.hpp"

using Real    = double;
constexpr int kDim = 3;
using NodeMID = sctl::Morton<kDim>;
using GPUTree = gpu_tree::GPUTree<Real, kDim>;
using Long    = gpu_tree::Long;

template <class V> double ms(V t0, V t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Leaf nodes are sorted lex by (code, depth).
static bool leaves_sorted(const std::vector<NodeMID>& v) {
  for (size_t i = 1; i < v.size(); ++i) {
    const NodeMID& a = v[i - 1];
    const NodeMID& b = v[i];
    if (b.mid < a.mid) return false;
    const bool codes_equal = !(a.mid < b.mid) && !(b.mid < a.mid);
    if (codes_equal && a.depth > b.depth) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  Long N = (argc > 1) ? std::stoll(argv[1]) : 1'000'000;
  const Long M = 4;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<Real> U(0.0, 1.0);
  std::vector<Real> coord_h(N * kDim);
  for (Real& x : coord_h) x = U(rng);

  // --- CPU path ---------------------------------------------------------
  {
    std::vector<Real>    coord = coord_h;
    std::vector<NodeMID> tree;
    std::vector<Long>    idx;

    auto t0 = std::chrono::steady_clock::now();
    GPUTree::buildTree(tree, coord, M);
    auto t1 = std::chrono::steady_clock::now();
    GPUTree::buildTree(tree, coord, M, &idx);
    auto t2 = std::chrono::steady_clock::now();

    bool sorted_ok = leaves_sorted(tree);
    bool idx_ok    = (idx.size() == (size_t)N);
    if (idx_ok) {
      std::vector<int> seen(N, 0);
      for (Long v : idx) if (v < 0 || v >= N || seen[v]++) { idx_ok = false; break; }
    }
    std::printf("CPU (std::vector):  buildTree(tree)     %.2f ms  sorted=%s\n", ms(t0, t1), sorted_ok ? "ok" : "FAIL");
    std::printf("CPU (std::vector):  buildTree(tree+idx) %.2f ms  idx=%s\n",    ms(t1, t2), idx_ok    ? "ok" : "FAIL");
  }

  // --- GPU path ---------------------------------------------------------
  {
    thrust::device_vector<Real>    coord(coord_h.begin(), coord_h.end());
    thrust::device_vector<NodeMID> tree;
    thrust::device_vector<Long>    idx;

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    GPUTree::buildTree(tree, coord, M);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    GPUTree::buildTree(tree, coord, M, &idx);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    std::vector<NodeMID> tree_h(tree.size());
    thrust::copy(tree.begin(), tree.end(), tree_h.begin());
    thrust::host_vector<Long> idx_h = idx;
    bool sorted_ok = leaves_sorted(tree_h);
    bool idx_ok    = (idx_h.size() == (size_t)N);
    if (idx_ok) {
      std::vector<int> seen(N, 0);
      for (Long v : idx_h) if (v < 0 || v >= N || seen[v]++) { idx_ok = false; break; }
    }
    std::printf("GPU (device_vector): buildTree(tree)     %.2f ms  sorted=%s\n", ms(t0, t1), sorted_ok ? "ok" : "FAIL");
    std::printf("GPU (device_vector): buildTree(tree+idx) %.2f ms  idx=%s\n",    ms(t1, t2), idx_ok    ? "ok" : "FAIL");
  }

  return 0;
}
