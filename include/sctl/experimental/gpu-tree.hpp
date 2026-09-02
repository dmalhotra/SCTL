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

namespace detail {
// Blocks template deduction: the optional out-params below must not take part in deducing
// `DeviceVector` -- `tree` and `coord` already fix it -- or passing `nullptr` for one of them
// fails the whole call.
template <class T> struct no_deduce { using type = T; };
template <class T> using no_deduce_t = typename no_deduce<T>::type;
}  // namespace detail

/**
 * Morton-order linear tree (experimental, header-only, host or device vectors). Same output as
 * `sctl::Tree::UpdateRefinement`. `buildTreeDist` is the single entry point; its default
 * `Comm::Self()` gives the single-rank build.
 */
template <class Real, Integer DIM> class GPUTree {
  static_assert(DIM > 0, "GPUTree: DIM must be positive");

 public:
  /** Per-node flags; same meaning and layout as `sctl::Tree`'s. */
  struct NodeAttr {
    unsigned char Leaf : 1,   ///< no child of this node is in the tree
                  Ghost : 1;  ///< outside this rank's owned range
  };

  /**
   * Per-node connectivity, same meaning and layout as `sctl::Tree`'s. Indices are into `tree`;
   * `-1` means the node is not in the tree (no parent at the root, no children at a leaf, no
   * same-level neighbor across a domain face or where the tree is coarser).
   */
  struct NodeLists {
    Long p2n;                              ///< index among the parent's children, -1 at the root
    Long parent;                           ///< index of the parent, -1 at the root
    Long child[1 << DIM];                  ///< indices of the children
    Long nbr[sctl::pow<DIM, Integer>(3)];  ///< indices of the same-level neighbors
  };

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
   * @param[in] periodicity Axes on which the domain wraps; neighbors cross those faces.
   * @param[in] halo_size Ghost-layer width; <0 adds no neighbor nodes. Either way the returned
   *            list is a full-domain complete tree on every rank (coarse outside the halo), so use
   *            `owned_range` to recover this rank's own nodes.
   * @param[out] owned_range Optional: this rank's [begin,end) slice within the returned tree.
   * @param[out] sort_scatter_index Optional: for the particle at this rank's sorted position `i`,
   *             its index in the global (rank-concatenated) input order; over all ranks these form
   *             a permutation of [0, Nglob). Pass `nullptr` to discard.
   * @param[out] partition Optional: `comm.Size()` entries, the first node owned by each rank, so a
   *             caller can route its own data the same way the build did. Caller-allocated.
   * @param[out] node_attr Optional: per-node `Leaf`/`Ghost` flags, one per entry of `tree`.
   * @param[out] node_lists Optional: per-node parent/child/neighbor indices. Costs one binary
   *             search per link, so it is built only when asked for.
   */
  template <template <class...> class DeviceVector>
  static void buildTreeDist(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M = 1, const Comm& comm = Comm::Self(), bool balance21 = false, sctl::Periodicity periodicity = sctl::Periodicity::NONE, Integer halo_size = -1, Long* owned_range = nullptr, detail::no_deduce_t<DeviceVector<Long>>* sort_scatter_index = nullptr, Morton<DIM>* partition = nullptr, detail::no_deduce_t<DeviceVector<NodeAttr>>* node_attr = nullptr, detail::no_deduce_t<DeviceVector<NodeLists>>* node_lists = nullptr);
};

}  // namespace gpu_tree

#include "sctl/experimental/gpu-tree.txx"

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
