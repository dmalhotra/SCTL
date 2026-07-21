// Compare gpu_tree::GPUTree full linear tree vs sctl::PtTree node sequence on the same
// random particle distribution. Reports:
//   - sizes of each tree's node list,
//   - whether the leaf sequences match,
//   - whether the full DFS pre-order sequences match,
//   - any structural differences.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

#include "sctl/experimental/gpu-tree.hpp"

using Real = double;
constexpr int kDim = 3;
using GMorton = sctl::MortonCode<kDim>;
using GNode   = sctl::Morton<kDim>;
using GPUTree = gpu_tree::GPUTree<Real, kDim>;
using GLong   = gpu_tree::Long;

using SMorton = sctl::Morton<kDim>;
using STree   = sctl::Tree<kDim>;
using SPtTree = sctl::PtTree<Real, kDim>;
using SLong   = sctl::Long;

// Compare two depth values across the gpu and sctl Morton types.
static bool same_node(const GNode& g, const SMorton& s) {
  if (g.depth != s.Depth()) return false;
  const std::array<Real, kDim> gc = g.Coord<Real>();
  std::array<Real, kDim> sc; s.Coord(sc);
  for (int d = 0; d < kDim; ++d) if (gc[d] != sc[d]) return false;
  return true;
}

// Extract leaves only from the gpu_tree full linear tree.
// A node is a leaf iff it has no descendant in the next position (the next entry isn't
// a strict descendant under it).
static std::vector<GNode> leaves_of(const std::vector<GNode>& tree) {
  std::vector<GNode> out;
  out.reserve(tree.size());
  for (size_t i = 0; i < tree.size(); ++i) {
    const GNode& n = tree[i];
    const bool has_descendant = (i + 1 < tree.size()) && n.isAncestor(tree[i + 1]);
    if (!has_descendant) out.push_back(n);
  }
  return out;
}

// Independently compute GPUTree's anchor leaves (Step 3 of buildTree, but exposed here so
// we can compare them against sctl::Tree's leaves directly without the linearizeTree fill-in).
static std::vector<GNode> compute_gpu_anchors(const std::vector<Real>& coord, GLong M) {
  const GLong N = static_cast<GLong>(coord.size()) / kDim;
  std::vector<GMorton> pt(N);
  for (GLong i = 0; i < N; ++i) pt[i] = GMorton(&coord[i * kDim]);
  std::sort(pt.begin(), pt.end());

  std::vector<GNode> anchors;
  const GLong N_pairs = (N > M + 1) ? (N - M - 1) : 0;
  anchors.reserve(N_pairs);
  for (GLong i = 0; i < N_pairs; ++i) {
    uint8_t d = pt[i].CommonAncestor(pt[i + M + 1]).depth;
    if (d < gpu_tree::MAX_DEPTH) ++d;
    const GNode leaf = pt[i + M + 1].Ancestor(d);
    if (anchors.empty() || !(leaf == anchors.back())) anchors.push_back(leaf);
  }
  return anchors;
}

int main(int argc, char** argv) {
  const GLong N = (argc > 1) ? std::stoll(argv[1]) : 100'000;
  const GLong M = (argc > 2) ? std::stoll(argv[2]) : 4;

  std::printf("N=%lld M=%lld DIM=%d MAX_DEPTH=%d\n",
              (long long)N, (long long)M, kDim, (int)gpu_tree::MAX_DEPTH);
  std::printf("sctl MAX_DEPTH=%d\n", (int)sctl::Morton<kDim>::MAX_DEPTH);

  // Generate identical coordinate arrays.
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<Real> U(0.0, 1.0);
  std::vector<Real> coord_std(N * kDim);
  for (Real& x : coord_std) x = U(rng);

  // GPU tree: full linear tree on CPU container path.
  std::vector<GNode> gpu_tree;
  GPUTree::buildTree(gpu_tree, coord_std, M);
  std::printf("gpu_tree::GPUTree    size=%zu\n", gpu_tree.size());

  // SCTL tree.
  sctl::Vector<Real> coord_sctl(N * kDim);
  for (GLong i = 0; i < N * kDim; ++i) coord_sctl[i] = coord_std[i];
  SPtTree pt_tree;
  pt_tree.AddParticles("pt", coord_sctl);
  pt_tree.UpdateRefinement(coord_sctl, M);
  const auto& sctl_mid  = pt_tree.GetNodeMID();
  const auto& sctl_attr = pt_tree.GetNodeAttr();
  std::printf("sctl::PtTree         size=%lld  (incl. ghost/non-leaf)\n", (long long)sctl_mid.Dim());

  // Strip ghost nodes from sctl output (single-rank Comm::Self should still flag local-only as
  // Ghost=0; this is defensive). Also retain only depth, code for comparison.
  std::vector<SMorton> sctl_nodes;
  sctl_nodes.reserve(sctl_mid.Dim());
  for (SLong i = 0; i < sctl_mid.Dim(); ++i) {
    if (sctl_attr[i].Ghost) continue;
    sctl_nodes.push_back(sctl_mid[i]);
  }
  std::printf("sctl::PtTree (no ghost) size=%zu\n", sctl_nodes.size());

  // GPUTree anchors (Step 3 only).
  std::vector<GNode> gpu_anchors = compute_gpu_anchors(coord_std, M);

  // sctl::Tree leaves, partitioned into non-empty and empty.
  // SCTL's UpdateRefinement constructs a "complete" linear tree that emits a leaf for every
  // morton-region between particle clusters, INCLUDING empty leaves. Filter to non-empty by
  // counting particles in each leaf's morton range against the sorted pt_mid array.
  std::vector<SMorton> pt_mid(N);
  for (GLong i = 0; i < N; ++i) pt_mid[i] = SMorton(coord_sctl.begin() + i * kDim);
  std::sort(pt_mid.begin(), pt_mid.end());

  std::vector<SMorton> sctl_leaves_all, sctl_leaves_nonempty, sctl_leaves_empty;
  for (SLong i = 0; i < sctl_mid.Dim(); ++i) {
    if (sctl_attr[i].Ghost || !sctl_attr[i].Leaf) continue;
    sctl_leaves_all.push_back(sctl_mid[i]);
    const SMorton lo = sctl_mid[i];
    const SMorton hi = lo.Next();
    const auto lo_it = std::lower_bound(pt_mid.begin(), pt_mid.end(), lo);
    const auto hi_it = std::lower_bound(pt_mid.begin(), pt_mid.end(), hi);
    if (hi_it > lo_it) sctl_leaves_nonempty.push_back(lo);
    else               sctl_leaves_empty.push_back(lo);
  }
  std::printf("gpu anchors (non-empty leaves)        = %zu\n", gpu_anchors.size());
  std::printf("sctl leaves total                     = %zu\n", sctl_leaves_all.size());
  std::printf("sctl leaves non-empty                 = %zu\n", sctl_leaves_nonempty.size());
  std::printf("sctl leaves empty                     = %zu\n", sctl_leaves_empty.size());

  // Are gpu_anchors a subset of sctl_leaves_nonempty? (in Morton order)
  std::printf("\n--- gpu_anchors ⊆ sctl non-empty leaves (in Morton order)? ---\n");
  {
    size_t si = 0, missing = 0;
    for (size_t gi = 0; gi < gpu_anchors.size(); ++gi) {
      while (si < sctl_leaves_nonempty.size() && !same_node(gpu_anchors[gi], sctl_leaves_nonempty[si])) ++si;
      if (si == sctl_leaves_nonempty.size()) { ++missing; if (missing < 5) std::printf("  gpu_anchor[%zu] depth=%d not in sctl non-empty leaves\n", gi, (int)gpu_anchors[gi].depth); break; }
      ++si;
    }
    std::printf("gpu_anchors not in sctl non-empty leaves: %zu / %zu\n", missing, gpu_anchors.size());
  }

  // Linear-tree-derived terminals.
  std::vector<GNode> gpu_terminals = leaves_of(gpu_tree);
  std::printf("\n--- linearizeTree terminals vs sctl leaves ---\n");
  std::printf("gpu linearizeTree terminals = %zu  sctl leaves (incl empty) = %zu  sctl non-empty leaves = %zu\n",
              gpu_terminals.size(), sctl_leaves_all.size(), sctl_leaves_nonempty.size());

  // Full-tree comparison (likely diverges: gpu emits Next-encountered siblings; sctl emits
  // only nodes on the ancestor-chain of leaves).
  std::printf("\n--- Full tree match? ---\n");
  std::printf("gpu_tree size=%zu  sctl_nodes size=%zu\n", gpu_tree.size(), sctl_nodes.size());

  // Subset check: every sctl node should appear in gpu_tree (in order).
  std::printf("\n--- sctl-nodes ⊆ gpu_tree (in order)? ---\n");
  {
    size_t gi = 0;
    size_t missing = 0;
    for (size_t si = 0; si < sctl_nodes.size(); ++si) {
      while (gi < gpu_tree.size() && !same_node(gpu_tree[gi], sctl_nodes[si])) ++gi;
      if (gi == gpu_tree.size()) { ++missing; }
      else { ++gi; }
    }
    std::printf("sctl nodes missing from gpu_tree: %zu / %zu\n", missing, sctl_nodes.size());
  }

  // Depth-by-depth breakdown of nodes appearing only in gpu_tree.
  std::printf("\n--- gpu_tree nodes NOT in sctl_nodes (by depth) ---\n");
  {
    std::vector<int> extra_by_depth(gpu_tree::MAX_DEPTH + 1, 0);
    size_t si = 0;
    for (size_t gi = 0; gi < gpu_tree.size(); ++gi) {
      bool found = false;
      while (si < sctl_nodes.size() && sctl_nodes[si].Depth() <= gpu_tree[gi].depth) {
        if (same_node(gpu_tree[gi], sctl_nodes[si])) { found = true; ++si; break; }
        // sctl_nodes are sorted; advance si if it's lex-less than gpu_tree[gi]
        // (using a simple coord-by-coord compare via Coord()).
        const auto gc = gpu_tree[gi].Coord<Real>();
        std::array<Real, kDim> sc; sctl_nodes[si].Coord(sc);
        bool s_less = false;
        for (int d = 0; d < kDim; ++d) {
          if (sc[d] < gc[d]) { s_less = true; break; }
          if (sc[d] > gc[d]) { break; }
        }
        if (s_less) ++si;
        else break;
      }
      if (!found) extra_by_depth[gpu_tree[gi].depth]++;
    }
    GLong total_extra = 0;
    for (int d = 0; d <= gpu_tree::MAX_DEPTH; ++d) {
      if (extra_by_depth[d] > 0) {
        std::printf("  depth %2d: %d extra gpu_tree nodes (not in sctl_nodes)\n", d, extra_by_depth[d]);
        total_extra += extra_by_depth[d];
      }
    }
    std::printf("gpu_tree extras: %lld (gpu - sctl-nodes-found)\n", (long long)total_extra);
  }

  // First 20 nodes side by side (to debug structural differences).
  std::printf("\n--- First 20 nodes (gpu_tree | sctl) ---\n");
  for (size_t i = 0; i < 20 && i < std::min(gpu_tree.size(), sctl_nodes.size()); ++i) {
    const auto gc = gpu_tree[i].Coord<Real>();
    std::array<Real, kDim> sc; sctl_nodes[i].Coord(sc);
    std::printf("  [%2zu] gpu d=%d (%.4f,%.4f,%.4f) | sctl d=%d (%.4f,%.4f,%.4f) %s\n",
                i, (int)gpu_tree[i].depth, gc[0], gc[1], gc[2],
                (int)sctl_nodes[i].Depth(), sc[0], sc[1], sc[2],
                same_node(gpu_tree[i], sctl_nodes[i]) ? "" : "DIFF");
  }

  // Per-depth node count comparison.
  std::printf("\n--- Node count by depth (gpu_tree vs sctl) ---\n");
  {
    std::vector<GLong> gpu_by_depth(gpu_tree::MAX_DEPTH + 2, 0);
    std::vector<GLong> sctl_by_depth(gpu_tree::MAX_DEPTH + 2, 0);
    for (const auto& n : gpu_tree) gpu_by_depth[n.depth]++;
    for (const auto& n : sctl_nodes) sctl_by_depth[n.Depth()]++;
    std::printf("  depth | gpu_tree | sctl | sctl - gpu\n");
    for (int d = 0; d <= gpu_tree::MAX_DEPTH + 1; ++d) {
      if (gpu_by_depth[d] || sctl_by_depth[d])
        std::printf("  %5d | %8lld | %4lld | %+lld\n", d,
                    (long long)gpu_by_depth[d], (long long)sctl_by_depth[d],
                    (long long)(sctl_by_depth[d] - gpu_by_depth[d]));
    }
  }

  std::printf("\nDone (comparison-only test; structural differences are expected).\n");
  return 0;
}
