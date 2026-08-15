// Experimental single-rank, header-only Morton-order tree (host + device).
// Defines GPUTree; MortonCode and Morton (the node type) live in sctl/morton.hpp.

#ifndef _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
#define _SCTL_EXPERIMENTAL_GPU_TREE_HPP_

#include "sctl/morton.hpp"

namespace sctl { class Comm; }

namespace gpu_tree {

using sctl::Integer;
using sctl::Long;
using sctl::Morton;
using sctl::MortonCode;
using sctl::MAX_DEPTH;

template <class Real, Integer DIM> class GPUTree;

/**
 * Single-rank, header-only Morton-order tree (experimental). Same output as
 * `sctl::Tree::UpdateRefinement` whether the container is a host or device vector.
 */
template <class Real, Integer DIM> class GPUTree {
  static_assert(DIM > 0, "GPUTree: DIM must be positive");

 public:
  /**
   * Build a Morton-order linear tree from particle coordinates.
   *
   * @param[out] tree Full linear tree (sorted in `(code, depth)` lex order; root at index 0).
   * @param[in] coord AoS-packed coordinates of length `N*DIM`, each in [0,1)^DIM.
   * @param[in] M Maximum number of particles per leaf box.
   * @param[out] sort_scatter_index Optional: pre-sort index of the particle that ends up at
   *             position `i` of the sorted Morton array. Pass `nullptr` to discard.
   */
  template <template <class...> class DeviceVector>
  static void buildTree(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M = 1, DeviceVector<Long>* sort_scatter_index = nullptr);

  /**
   * Build a Morton-order linear tree from pre-sorted Morton codes.
   *
   * @param[out] tree Full linear tree.
   * @param[in] pt_mid Morton codes sorted in `(code, depth)` lex order.
   * @param[in] M Maximum number of particles per leaf box.
   */
  template <template <class...> class DeviceVector>
  static void buildTreeFromSortedMorton(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M = 1);

  /**
   * Distributed variant: build the global Morton-order linear tree across the ranks of
   * `comm` via a device-buffer sample sort (one Alltoallv; CUDA-aware MPI required for
   * device vectors). On return each rank holds a contiguous slice of the global tree;
   * the concatenation over ranks equals the single-rank `buildTree` output.
   */
  template <template <class...> class DeviceVector>
  static void buildTreeDist(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M, const sctl::Comm& comm, bool balance21 = false, Integer halo_size = -1, Long* owned_range = nullptr);
};

}  // namespace gpu_tree

#include "sctl/experimental/gpu-tree.txx"

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
