// Experimental Morton-order tree (header-only, host + device, single-rank or distributed).
// Defines GPUTree; MortonCode and Morton (the node type) live in sctl/morton.hpp.

#ifndef _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
#define _SCTL_EXPERIMENTAL_GPU_TREE_HPP_

#include "sctl/morton.hpp"
#include "sctl/comm.hpp"  // for the Comm::Self() default of buildTreeDist

namespace gpu_tree {

using sctl::Integer;
using sctl::Long;
using sctl::Comm;
using sctl::Morton;
using sctl::MortonCode;
using sctl::MAX_DEPTH;

template <class Real, Integer DIM> class GPUTree;

/**
 * Morton-order linear tree (experimental, header-only, host or device vectors). Same output as
 * `sctl::Tree::UpdateRefinement`. `buildTreeDist` is the single entry point; its default
 * `Comm::Self()` gives the single-rank build.
 */
template <class Real, Integer DIM> class GPUTree {
  static_assert(DIM > 0, "GPUTree: DIM must be positive");

 public:
  /**
   * Build the global Morton-order linear tree from particle coordinates, distributed across the
   * ranks of `comm` (default `Comm::Self()` = single-rank build). Each rank returns a contiguous
   * slice; the concatenation over ranks equals the single-rank output. The np>1 path uses a
   * device-buffer sample sort (one Alltoallv; CUDA-aware MPI required for device vectors).
   *
   * @param[out] tree Full linear tree slice (sorted in `(code, depth)` lex order; root at index 0).
   * @param[in] coord AoS-packed coordinates of length `Nloc*DIM`, each in [0,1)^DIM.
   * @param[in] M Maximum number of particles per leaf box.
   * @param[in] comm Communicator to distribute the build across.
   * @param[in] balance21 Apply 2:1 balance refinement.
   * @param[in] halo_size Ghost-layer width; <0 skips ghost nodes.
   * @param[out] owned_range Optional: this rank's [begin,end) slice within the returned tree.
   * @param[out] sort_scatter_index Optional: for the particle at this rank's sorted position `i`,
   *             its index in the global (rank-concatenated) input order; over all ranks these form
   *             a permutation of [0, Nglob). Pass `nullptr` to discard.
   */
  template <template <class...> class DeviceVector>
  static void buildTreeDist(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M = 1, const Comm& comm = Comm::Self(), bool balance21 = false, Integer halo_size = -1, Long* owned_range = nullptr, DeviceVector<Long>* sort_scatter_index = nullptr);
};

}  // namespace gpu_tree

#include "sctl/experimental/gpu-tree.txx"

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
