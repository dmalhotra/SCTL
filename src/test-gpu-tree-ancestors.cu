// Verifies Morton::CommonAncestor, Morton::Ancestor, NodeMID::Next, and the
// leaf-derivation step of GPUTree::buildTree.
//
// 1) Spot-check CommonAncestor and Next on hand-picked inputs.
// 2) Call buildTree on random coords for both CPU and GPU paths, and check:
//      - depth never exceeds MAX_DEPTH
//      - no two consecutive entries are exactly equal (post-unique)
//      - the (code, depth) sequence is non-decreasing lex (the property
//        the algorithm relies on for unique to yield global dedup)
//      - for sampled pairs, pt[i+M+1].Ancestor(d_common+1) appears in the
//        dedup'd output
//      - the CPU and GPU outputs agree element-for-element

#include <algorithm>
#include <cstdio>
#include <cstdint>
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
using Morton  = sctl::MortonCode<kDim>;
using NodeMID = sctl::Morton<kDim>;
using GPUTree = gpu_tree::GPUTree<Real, kDim>;
using Long    = gpu_tree::Long;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL @%d: %s\n", __LINE__, #cond); ++failures; } } while (0)

int main(int argc, char** argv) {
  const Long N = (argc > 1) ? std::stoll(argv[1]) : 100'000;
  const Long M = (argc > 2) ? std::stoll(argv[2]) : 4;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<Real> U(0.0, 1.0);
  std::vector<Real> coord(N * kDim);
  for (Real& x : coord) x = U(rng);

  // Sorted leaf-level Morton IDs (host) — used for spot checks and the
  // sampled-pair formula. buildTree below regenerates these internally.
  std::vector<Morton> pt_h(N);
  for (Long i = 0; i < N; ++i) pt_h[i] = Morton(&coord[i * kDim]);
  std::sort(pt_h.begin(), pt_h.end());

  // 1) Spot checks.
  std::printf("Spot checks:\n");
  {
    const NodeMID a = pt_h[0].CommonAncestor(pt_h[0]);
    CHECK(a.depth == gpu_tree::MAX_DEPTH);
  }
  for (int k = 0; k < 5 && k + 1 < N; ++k) {
    if (!(pt_h[k] < pt_h[k + 1]) && !(pt_h[k + 1] < pt_h[k])) continue;  // skip exact duplicates
    const NodeMID a = pt_h[k].CommonAncestor(pt_h[k + 1]);
    CHECK(a.depth <= gpu_tree::MAX_DEPTH - 1);
  }

  // 1b) Next() spot check.
  {
    NodeMID a = pt_h[0].CommonAncestor(pt_h[0]);
    NodeMID b = a.Next();
    CHECK(b.depth <= a.depth);
    CHECK(a.mid < b.mid);
  }
  if (N > 1) {
    NodeMID a = pt_h[0].CommonAncestor(pt_h[N - 1]);
    NodeMID b = a.Next();
    CHECK(b.depth <= a.depth);
    if (a.depth > 0) CHECK(a.mid < b.mid);
  }

  // 2) CPU path.
  std::printf("CPU path:\n");
  std::vector<NodeMID> leaves_cpu;
  {
    GPUTree::buildTree(leaves_cpu, coord, M);

    for (const NodeMID& a : leaves_cpu) CHECK(a.depth <= gpu_tree::MAX_DEPTH);

    for (size_t i = 1; i < leaves_cpu.size(); ++i) CHECK(leaves_cpu[i] != leaves_cpu[i - 1]);

    // Sortedness: lex by (code, depth).
    for (size_t i = 1; i < leaves_cpu.size(); ++i) {
      const NodeMID& a = leaves_cpu[i - 1];
      const NodeMID& b = leaves_cpu[i];
      CHECK(!(b.mid < a.mid));
      const bool codes_equal = !(a.mid < b.mid) && !(b.mid < a.mid);
      if (codes_equal) CHECK(a.depth <= b.depth);
    }

    // For sampled pairs, `pt[i+M+1].Ancestor(d_common+1)` (clamped to MAX_DEPTH) must appear in the output.
    const Long N_pairs = N - M;
    for (Long i = 0; i < N_pairs; i += std::max<Long>(1, N_pairs / 32)) {
      uint8_t d = pt_h[i].CommonAncestor(pt_h[i + M]).depth;
      if (d < gpu_tree::MAX_DEPTH) ++d;
      const NodeMID expected = pt_h[i + M].Ancestor(d);
      bool found = false;
      for (const NodeMID& b : leaves_cpu) if (b == expected) { found = true; break; }
      CHECK(found);
    }
    std::printf("  %zu leaf nodes (from %ld pairs)\n", leaves_cpu.size(), (long)N_pairs);
  }

  // 3) GPU path.
  std::printf("GPU path:\n");
  thrust::device_vector<NodeMID> leaves_gpu_d;
  {
    thrust::device_vector<Real> coord_d(coord.begin(), coord.end());
    GPUTree::buildTree(leaves_gpu_d, coord_d, M);
  }
  thrust::host_vector<NodeMID> leaves_gpu(leaves_gpu_d.begin(), leaves_gpu_d.end());

  // 4) CPU and GPU agree.
  CHECK(leaves_cpu.size() == leaves_gpu.size());
  if (leaves_cpu.size() == leaves_gpu.size()) {
    for (size_t i = 0; i < leaves_cpu.size(); ++i) CHECK(leaves_cpu[i] == leaves_gpu[i]);
  }
  std::printf("  %zu leaf nodes from GPU path\n", leaves_gpu.size());

  std::printf("\n%s (%d check(s) failed)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
