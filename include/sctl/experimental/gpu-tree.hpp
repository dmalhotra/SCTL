/**
 * @file gpu-tree.hpp
 * Definition of the GPUTree class (experimental).
 *
 * Header-only, host + device, single-rank or distributed. MortonCode and Morton (the node type)
 * live in sctl/morton.hpp.
 */

#ifndef _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
#define _SCTL_EXPERIMENTAL_GPU_TREE_HPP_

#include <map>
#include <string>
#include <vector>

#include "sctl/morton.hpp"
#include "sctl/comm.hpp"    // Comm::Self() default, and the partition helpers the data layer uses
#include "sctl/vector.hpp"  // per-node counts stay host-side, as in sctl::Tree

namespace gpu_tree {

using sctl::Integer;
using sctl::Long;
using sctl::Comm;
using sctl::Morton;
using sctl::MortonCode;
using sctl::MAX_DEPTH;

template <class Real, Integer DIM, template <class...> class DevVec> class GPUTree;

namespace detail_ptTree {
template <template <class...> class DeviceVector> struct PtScatter;
}  // namespace detail_ptTree

namespace detail {
// Blocks template deduction: the optional out-params below must not take part in deducing
// `DeviceVector` -- `tree` and `coord` already fix it -- or passing `nullptr` for one of them
// fails the whole call.
template <class T> struct no_deduce { using type = T; };
template <class T> using no_deduce_t = typename no_deduce<T>::type;
}  // namespace detail

/**
 * Class template representing a Morton-order linear tree, built on the device.
 *
 * The node set is the same as `sctl::Tree`'s for the same inputs. Two ways to use it: the stateful
 * interface below mirrors `sctl::Tree` -- the object retains the tree, the partition and any named
 * per-node data, and `UpdateRefinement` carries that data onto the new nodes -- while the static
 * `buildTreeDist` builds a tree into caller-owned vectors and keeps no state. As in `sctl::Tree`,
 * `UpdateRefinement` is a full rebuild; the retained tree serves only to remap the data.
 *
 * @tparam Real Data type for the particle coordinates.
 * @tparam DIM Number of spatial dimensions.
 * @tparam DevVec Vector template holding the retained state (e.g. `thrust::device_vector`). Only
 * the stateful interface uses it; `buildTreeDist` deduces its own from its arguments.
 */
template <class Real, Integer DIM, template <class...> class DevVec = std::vector> class GPUTree {
  static_assert(DIM > 0, "GPUTree: DIM must be positive");

 public:

  /**
   * Structure for storing attributes of a tree node.
   */
  struct NodeAttr {
    unsigned char Leaf : 1,   ///< no child of this node is in the tree
                  Ghost : 1;  ///< outside this rank's owned range
  };

  /**
   * Structure for storing lists of nodes (children, parent, neighbors).
   *
   * One array per kind rather than an array of structs: that is how the build produces them and how
   * a GPU consumer reads them -- a thread per node touching one field gives coalesced access either
   * way, and no copy into a packed struct is needed. This is the one place the interface
   * deliberately departs from `sctl::Tree`. Indices are into the node list; `-1` means the node is
   * not in the tree (no parent at the root, no children at a leaf, no same-level neighbor across a
   * domain face or where the tree is coarser). `p2n` is not stored: it is `GetNodeMID()[i].Path2Node()`.
   */
  template <template <class...> class DeviceVector> struct NodeLists {
    DeviceVector<Long> parent;  ///< `N`: index of the parent
    DeviceVector<Long> child;   ///< `N * 2^DIM`, row-major: indices of the children
    DeviceVector<Long> nbr;     ///< `N * 3^DIM`, row-major: indices of the same-level neighbors
  };

  /**
   * @return The number of spatial dimensions.
   */
  static constexpr Integer Dim() { return DIM; }

  /**
   * Constructs a GPUTree object.
   *
   * @param comm_ Communicator.
   */
  explicit GPUTree(const Comm& comm_ = Comm::Self());

  /**
   * @return Vector of Morton IDs partitioning the processor domains.
   */
  const sctl::Vector<Morton<DIM>>& GetPartitionMID() const { return mins_; }

  /**
   * @return Vector of Morton IDs of tree nodes.
   */
  const DevVec<Morton<DIM>>& GetNodeMID() const { return node_mid_; }

  /**
   * @return Vector of attributes of tree nodes.
   */
  const DevVec<NodeAttr>& GetNodeAttr() const { return node_attr_; }

  /**
   * @return Node-lists of tree nodes.
   */
  const NodeLists<DevVec>& GetNodeLists() const { return node_lists_; }

  /**
   * @return The communicator.
   */
  const Comm& GetComm() const { return comm_; }

  /**
   * This rank's own nodes within `GetNodeMID()`; the rest are ghosts. Equivalent to scanning
   * `GetNodeAttr()` for the non-ghost run, which the build already knows.
   *
   * @param[out] begin Index of this rank's first owned node.
   * @param[out] end One past its last owned node.
   */
  void GetOwnedRange(Long& begin, Long& end) const { begin = owned_begin_; end = owned_end_; }

  /**
   * Update tree refinement and repartition node data among the new tree nodes.
   *
   * @param[in] coord Particle coordinates (in [0,1]^dim stored in AoS order) that describe the new tree refinement.
   * @param[in] M Maximum number of particles per tree node.
   * @param[in] balance21 Whether to do level-restriction (2:1 balance refinement).
   * @param[in] periodicity Per-axis periodicity bitmask (e.g. `Periodicity::X | Periodicity::Y`, or `all_periodic(DIM)`).
   * @param[in] halo_size 2^halo_size neighboring boxes will be included in the halo region. Default value of -1 means no halo region.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void UpdateRefinement(const DevVec<Real>& coord, Long M = 1, bool balance21 = 0, sctl::Periodicity periodicity = sctl::Periodicity::NONE, Integer halo_size = -1);

  /**
   * Add named data to the tree nodes.
   *
   * @param[in] name Name for the data. Must not already exist on this tree.
   * @param[in] data Contiguous data for all nodes, concatenated in node
   * order. Must satisfy `data.size() == dof * sum(cnt)` for some `dof >= 0`.
   * @param[in] cnt Number of data elements per node (length = number of tree nodes). Host-side, as
   * in `sctl::Tree`; the payload itself stays on the device and is never copied to the host.
   *
   * @note Collective; must be called from all processes.
   *
   * @warning Storage is type-erased. `GetData<U>` for this `name` only
   * round-trips when `U` matches the `ValueType` used here.
   */
  template <class ValueType> void AddData(const std::string& name, const DevVec<ValueType>& data, const sctl::Vector<Long>& cnt);

  /**
   * Get node data.
   *
   * @param[out] data Copy of the stored data for this name. Unlike `sctl::Tree::GetData`, which
   * returns a non-owning view, this copies: the payload lives in a device vector the tree owns and
   * resizes on every refinement.
   * @param[out] cnt Number of data elements per node (length = number of tree nodes).
   * @param[in] name Name of the data.
   *
   * @warning `ValueType` must match the type used in the corresponding
   * `AddData`; otherwise the bytes are silently reinterpreted.
   */
  template <class ValueType> void GetData(DevVec<ValueType>& data, sctl::Vector<Long>& cnt, const std::string& name) const;

  /**
   * Delete data from the tree nodes.
   *
   * @param[in] name Name of the data.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void DeleteData(const std::string& name);

  /**
   * Write VTK visualization.
   *
   * @param[in] fname Filename for the output.
   * @param[in] show_ghost Whether to show ghost nodes.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void WriteTreeVTK(std::string fname, bool show_ghost = false) const;

  /**
   * Build the global Morton-order linear tree from particle coordinates, distributed across the
   * ranks of `comm` (default `Comm::Self()` = single-rank build). Each rank returns a contiguous
   * slice; the concatenation over ranks equals the single-rank output. The np>1 path uses a
   * device-buffer sample sort (one Alltoallv; CUDA-aware MPI required for device vectors).
   *
   * This is the stateless form: it keeps nothing, so it cannot carry node data across a rebuild.
   * Use the stateful interface above for that.
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
   * @param[out] node_lists Optional: per-node parent/child/neighbor indices. Roughly 288 bytes per
   *             node and a third of the build's time, so it is produced only when asked for.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  template <template <class...> class DeviceVector>
  static void buildTreeDist(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M = 1, const Comm& comm = Comm::Self(), bool balance21 = false, sctl::Periodicity periodicity = sctl::Periodicity::NONE, Integer halo_size = -1, Long* owned_range = nullptr, detail::no_deduce_t<DeviceVector<Long>>* sort_scatter_index = nullptr, Morton<DIM>* partition = nullptr, detail::no_deduce_t<DeviceVector<NodeAttr>>* node_attr = nullptr, detail::no_deduce_t<NodeLists<DeviceVector>>* node_lists = nullptr);

 private:

  /** Per-new-node `[range[i], range[i+1])` into `old_mid`, the old nodes each new node absorbs. */
  static void remapRanges(sctl::Vector<Long>& range, const sctl::Vector<Morton<DIM>>& old_mid, const sctl::Vector<Morton<DIM>>& new_mid);

  sctl::Vector<Morton<DIM>> mins_;
  DevVec<Morton<DIM>> node_mid_;
  DevVec<NodeAttr> node_attr_;
  NodeLists<DevVec> node_lists_;
  Long owned_begin_ = 0, owned_end_ = 0;

  std::map<std::string, DevVec<char>> node_data_;  ///< payload, type-erased to bytes
  std::map<std::string, sctl::Vector<Long>> node_cnt_;

  Comm comm_;
};

/**
 * Class template representing a point tree in a specified dimension, built on the device.
 *
 * Particle groups are named, sorted into the tree's node order, and their data follows the tree
 * through `UpdateRefinement`. `GetParticleData` scatters back to the caller's original ordering.
 *
 * @tparam Real Data type for the coordinates and values of points.
 * @tparam DIM Dimensionality of the point tree.
 * @tparam DevVec Vector template holding the retained state (e.g. `thrust::device_vector`).
 * @tparam BaseTree Base class for the point tree. Defaults to GPUTree<Real,DIM,DevVec>.
 */
template <class Real, Integer DIM, template <class...> class DevVec = std::vector, class BaseTree = GPUTree<Real, DIM, DevVec>>
class PtTree : public BaseTree {
 public:

  /**
   * Constructor for PtTree.
   *
   * @param comm Communication object for distributed computing. Defaults to Comm::Self().
   */
  explicit PtTree(const Comm& comm = Comm::Self());

  /**
   * Update refinement of the point tree based on given coordinates.
   *
   * @param coord Coordinates of the points.
   * @param M Maximum number of points per box for refinement.
   * @param balance21 Flag indicating whether to construct a level-restricted
   *        tree with neighboring boxes within one level of each other.
   * @param periodicity Per-axis periodicity bitmask (e.g. `Periodicity::X | Periodicity::Y`, or `all_periodic(DIM)`).
   * @param[in] halo_size 2^halo_size neighboring boxes will be included in the halo region
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void UpdateRefinement(const DevVec<Real>& coord, Long M = 1, bool balance21 = 0, sctl::Periodicity periodicity = sctl::Periodicity::NONE, Integer halo_size = -1);

  /**
   * Add particles to the point tree.
   *
   * @param name Name of the particle group.
   * @param coord Coordinates of the particles.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void AddParticles(const std::string& name, const DevVec<Real>& coord);

  /**
   * Add particle data to the point tree.
   *
   * @param data_name Name of the data. Must not already exist.
   * @param particle_name Name of an existing particle group from `AddParticles`.
   * @param data Local data values, sized `dof * Nlocal[particle_name]` for
   * some implicit `dof`. Reordered to match the particle group.
   *
   * @note Collective; must be called from all processes.
   */
  void AddParticleData(const std::string& data_name, const std::string& particle_name, const DevVec<Real>& data);

  /**
   * Get particle data from the point tree. The data scattered back to
   * the original ordering of the particles.
   *
   * @param data Vector to store the data values.
   * @param data_name Name of the data.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void GetParticleData(DevVec<Real>& data, const std::string& data_name) const;

  /**
   * Delete particle data from the point tree. Deleting a particle group also deletes every data
   * set attached to it.
   *
   * @param data_name Name of the data to delete.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void DeleteParticleData(const std::string& data_name);

  /**
   * Write particle data to a VTK file.
   *
   * @param fname Filename for the VTK file.
   * @param data_name Name of the data to write.
   * @param show_ghost Flag indicating whether to include ghost particles in the visualization.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void WriteParticleVTK(std::string fname, std::string data_name, bool show_ghost = false) const;

 private:

  /** `dof` deduced globally as `sum(ndata)/sum(nitem)`, as in sctl::Tree. */
  Long globalDof(Long ndata, Long nitem) const;

  /** Sort a group into the tree's node order and record the movement so data can follow it. */
  void sortGroup(const std::string& name, const DevVec<Real>& coord);

  /** Particles of `name` falling in each node of `GetNodeMID()`. */
  void nodeCounts(const std::string& name, sctl::Vector<Long>& cnt) const;

  std::map<std::string, Long> Nlocal_;                                 ///< particles this rank was given
  std::map<std::string, DevVec<Morton<DIM>>> pt_mid_;                  ///< per group, in tree order
  std::map<std::string, detail_ptTree::PtScatter<DevVec>> scatter_;
  std::map<std::string, std::string> data_pt_name_;                    ///< data name -> particle group
};

}  // namespace gpu_tree

#include "sctl/experimental/gpu-tree.txx"

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
