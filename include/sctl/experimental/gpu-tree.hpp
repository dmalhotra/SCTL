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
#include <set>
#include <string>

#include "sctl/morton.hpp"
#include "sctl/comm.hpp"    // Comm::Self() default, and the partition helpers the data layer uses
#include "sctl/vector.hpp"  // per-node counts stay host-side, as in sctl::Tree
#include "sctl/experimental/gpu-vector.hpp"  // HostVector, DeviceVector, DataView
#include "sctl/experimental/sort-scatter.hpp"

namespace gpu_tree {

using sctl::Integer;
using sctl::Long;
using sctl::Comm;
using sctl::Morton;
using sctl::MortonCode;
using sctl::MAX_DEPTH;

/**
 * Morton-order linear tree on a thrust backend, with the node set of `sctl::Tree` for the same
 * inputs. The object retains the tree, the partition and any named per-node data;
 * `UpdateRefinement` is a full rebuild, as in `sctl::Tree`, that remaps the data onto the new nodes.
 *
 * The one input on which the two node sets differ is a box where more than `M` particles share a
 * single Morton code: no refinement separates them, so the box stays over `M` however deep it goes,
 * and the two libraries stop at different depths -- this one at `MAX_DEPTH`, `sctl::Tree` as soon as
 * the run is alone in a box. Both are leaves that splitting cannot make smaller, so neither is
 * wrong; only the node count differs, and only for such runs.
 *
 * @tparam Real Data type for the particle coordinates.
 * @tparam DIM Number of spatial dimensions.
 * @tparam DevVec Container template for the tree and its data: `HostVector` or `DeviceVector`.
 */
template <class Real, Integer DIM, template <class...> class DevVec = HostVector> class GPUTree {
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
   * Node lists, one array per kind (structure of arrays, unlike `sctl::Tree`): how the build
   * produces them and how a kernel reads them. Indices are into the node list; `-1` means absent
   * (no parent at the root, no children at a leaf, no same-level neighbor across a domain face or
   * where the tree is coarser). `p2n` is `GetNodeMID()[i].Path2Node()`.
   */
  template <template <class...> class Vec> struct NodeLists {
    Vec<Long> parent;  ///< `N`: index of the parent
    Vec<Long> child;   ///< `N * 2^DIM`, row-major: indices of the children
    Vec<Long> nbr;     ///< `N * 3^DIM`, row-major: indices of the same-level neighbors
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
  void GetOwnedRange(Long& begin, Long& end) const {
    begin = owned_begin_;
    end = owned_end_;
  }

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
   * @warning Storage is type-erased. `GetData` with a `View<U>` for this `name` only
   * round-trips when `U` matches the `ValueType` used here.
   */
  template <class ValueType> void AddData(const std::string& name, const DevVec<ValueType>& data, const sctl::Vector<Long>& cnt);

  /**
   * Add named data without values: `cnt[i] * dof` unwritten elements for node i, to be filled in
   * place through the view `GetData` fills. No data moves; the ranks only agree on `dof`.
   *
   * @note Collective; must be called from all processes.
   */
  template <class ValueType> void AddData(const std::string& name, Long dof, const sctl::Vector<Long>& cnt);

  /** Non-owning view of a data set's storage on this backend; see `DataView`. */
  template <class T> using View = DataView<T, DevVec>;

  /**
   * Get node data as views of the stored buffers, as `sctl::Tree::GetData` does: `data` over the
   * payload, `cnt` over the per-node counts. Neither owns; both are valid until the set is
   * reallocated (`UpdateRefinement`, `Broadcast`, `ReduceBroadcast`, `DeleteData`). A const tree yields a
   * const view.
   *
   * @param[out] data View of the stored data for this name.
   * @param[out] cnt View of the number of data items per node (length = number of tree nodes); not to be modified.
   * @param[in] name Name of the data.
   *
   * @warning `ValueType` must match the type used in the corresponding
   * `AddData`; otherwise the bytes are silently reinterpreted.
   */
  template <class ValueType> void GetData(View<ValueType>& data, sctl::Vector<Long>& cnt, const std::string& name);
  template <class ValueType> void GetData(View<const ValueType>& data, sctl::Vector<Long>& cnt, const std::string& name) const;

  /**
   * Delete data from the tree nodes.
   *
   * @param[in] name Name of the data.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void DeleteData(const std::string& name);

  /**
   * Sum the partial values of nodes shared between processors, then fill the ghost copies with the
   * result. The ghost nodes themselves come from `UpdateRefinement`'s halo; only their data changes.
   *
   * @param[in] name Name of the data.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  template <class ValueType> void ReduceBroadcast(const std::string& name);

  /**
   * Fill the ghost copies of nodes with their owner's data. The ghost nodes themselves come from
   * `UpdateRefinement`'s halo; only their data changes.
   *
   * @param[in] name Name of the data.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  template <class ValueType> void Broadcast(const std::string& name);

  /**
   * Write VTK visualization.
   *
   * @param[in] fname Filename for the output.
   * @param[in] show_ghost Whether to show ghost nodes.
   *
   * @note This is a collective operation and must be called from all processes in the communicator.
   */
  void WriteTreeVTK(std::string fname, bool show_ghost = false) const;

 protected:

  /**
   * The stored buffers themselves, for a derived class that has to reallocate a data set in place
   * (`PtTree::UpdateRefinement` swaps in the re-cut payload), which the view from `GetData` cannot
   * do. `sctl::Tree` exposes the same seam as `GetData_`.
   */
  DevVec<char>& NodeData_(const std::string& name) { return node_data_.at(name); }
  const DevVec<char>& NodeData_(const std::string& name) const { return node_data_.at(name); }
  sctl::Vector<Long>& NodeCnt_(const std::string& name) { return node_cnt_.at(name); }

  std::set<std::string> data_moved_by_derived_;  ///< payloads a derived class moves itself after a rebuild; UpdateRefinement skips them

 private:

  /**
   * The build step of `UpdateRefinement`, into caller-owned vectors: `tree` is this rank's full
   * linear tree slice (`(code, depth)` order, complete over the domain, coarse outside the halo),
   * `owned_range` its own [begin,end) within it, `partition` the first node of each rank (np
   * entries, caller-allocated), `node_attr` the Leaf/Ghost flags, `node_lists` the connectivity,
   * `user_mid`/`user_cnt` the halo send list and its per-rank counts. Each output pointer may be
   * null. Collective. When all particles fit one leaf, rank 0 holds the root alone and every other
   * rank an empty tree.
   */
  static void buildTreeDist(DevVec<Morton<DIM>>& tree, const DevVec<Real>& coord, Long M, const Comm& comm, bool balance21, sctl::Periodicity periodicity, Integer halo_size, Long* owned_range, Morton<DIM>* partition, DevVec<NodeAttr>* node_attr, NodeLists<DevVec>* node_lists, DevVec<Morton<DIM>>* user_mid, sctl::Vector<Long>* user_cnt);

  /** The view behind both `GetData` overloads; `VT` may be const. */
  template <class VT> void dataView(View<VT>& data, sctl::Vector<Long>& cnt, const std::string& name) const;

  /** Storage for a new data set of `bytes`, with its per-node counts; both `AddData` overloads end here. */
  void addData_(const std::string& name, Long bytes, const sctl::Vector<Long>& cnt);



  sctl::Vector<Morton<DIM>> mins_;
  DevVec<Morton<DIM>> node_mid_;
  DevVec<Morton<DIM>> user_mid_;   ///< halo send list: my nodes, grouped by the rank that ghosts them
  sctl::Vector<Long> user_cnt_;    ///< np entries: how many of them go to each rank
  mutable sctl::Vector<Morton<DIM>> node_mid_host_;  ///< node_mid_ on the host; a device backend only
  mutable sctl::Vector<Morton<DIM>> user_mid_host_;  ///< user_mid_ likewise
  mutable bool host_mid_stale_ = true;               ///< whether those two still describe the tree
  DevVec<NodeAttr> node_attr_;
  NodeLists<DevVec> node_lists_;
  Long owned_begin_ = 0, owned_end_ = 0;

  std::map<std::string, DevVec<char>> node_data_;  ///< payload, type-erased to bytes
  std::map<std::string, sctl::Vector<Long>> node_cnt_;

  Comm comm_;

  /**
   * The node list and the halo send list, in host memory, where Broadcast and the VTK writer read
   * them. Neither changes except at a refinement, so a device backend copies them once and then
   * reuses them, and a host backend hands back its own storage and copies nothing. Taken on first
   * use rather than at the refinement, so a tree whose data never crosses ranks pays nothing.
   *
   * @note Not safe against concurrent first calls, as with the collectives that read them.
   */
  sctl::ConstIterator<Morton<DIM>> hostNodeMID() const;
  sctl::ConstIterator<Morton<DIM>> hostUserMID() const;
  void fillHostMID() const;
};

/**
 * Class template representing a point tree in a specified dimension, built on the device.
 *
 * Particle groups are named, sorted into the tree's node order, and their data follows the tree
 * through `UpdateRefinement`. `GetParticleData` scatters back to the caller's original ordering.
 *
 * @tparam Real Data type for the coordinates and values of points.
 * @tparam DIM Dimensionality of the point tree.
 * @tparam DevVec Container template for the tree and its data: `HostVector` or `DeviceVector`.
 * @tparam BaseTree Base class for the point tree. Defaults to GPUTree<Real,DIM,DevVec>.
 */
template <class Real, Integer DIM, template <class...> class DevVec = HostVector, class BaseTree = GPUTree<Real, DIM, DevVec>>
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
   * Add particle data without values: `dof` unwritten values per particle of `particle_name`, in
   * the group's tree order, to be filled in place through the view `GetData` fills;
   * `GetParticleData` maps that order back to the caller's. Local, no communication.
   */
  void AddParticleData(const std::string& data_name, const std::string& particle_name, Long dof);

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

  /**
   * Minimal example on `Comm::World()`: build a tree over random particles, attach a value per
   * particle, scale it in place through a view of the node data, and read it back in the caller's
   * order. Validation against `sctl::Tree` lives in `src/test-gpu-tree.cu`.
   */
  static void test();

 private:

  /** Particles of `name` falling in each node of `GetNodeMID()`. */
  void nodeCounts(const std::string& name, sctl::Vector<Long>& cnt) const;

  sctl::Vector<MortonCode<DIM>> partition_codes_;  ///< partition mins as codes: the SortScatter splitters; set with each partition
  std::map<std::string, SortScatter<MortonCode<DIM>, DevVec>> groups_;  ///< per particle group: codes in tree order, maps to/from caller order
  std::map<std::string, std::string> data_pt_name_;                    ///< data name -> particle group
};

}  // namespace gpu_tree

#include "sctl/experimental/gpu-tree.txx"

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_HPP_
