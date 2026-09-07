/**
 * @file tree.hpp
 * Definition of Tree and PtTree classes.
 */

#ifndef _SCTL_TREE_HPP_
#define _SCTL_TREE_HPP_

#include <map>                  // for map
#include <set>                  // for set
#include <string>               // for basic_string, string

#include "sctl/common.hpp"        // for Long, Integer, sctl
#include "sctl/comm.hpp"          // for Comm
#include "sctl/comm.txx"          // for Comm::Self
#include "sctl/math_utils.txx"    // for pow
#include "sctl/sort-scatter.hpp"  // for SortScatter
#include "sctl/sort-scatter.txx"  // for SortScatter::Init
#include "sctl/vector.hpp"        // for Vector

namespace sctl {

template <Integer DIM> class Morton;
template <Integer DIM> class MortonCode;

/**
 * Class template representing a tree data structure.
 *
 * @tparam DIM Number of spatial dimensions.
 */
template <Integer DIM> class Tree {
  public:

    /**
     * Structure for storing attributes of a tree node.
     */
    struct NodeAttr {
      unsigned char Leaf : 1, Ghost : 1;
    };

    /**
     * Structure for storing lists of nodes (children, parent, neighbors).
     */
    struct NodeLists {
      Long p2n;                             ///< path-to-node: id among the siblings
      Long parent;                          ///< index of the parent node
      Long child[1 << DIM];                 ///< index of the children
      Long nbr[sctl::pow<DIM,Integer>(3)];  ///< index of the neighbors at the same level
    };

    /**
     * @return The number of spatial dimensions.
     */
    static constexpr Integer Dim();

    /**
     * Constructs a Tree object.
     *
     * @param comm_ Communicator.
     */
    Tree(const Comm& comm_ = Comm::Self());

    /**
     * Destroys the Tree object.
     */
    ~Tree();

    /**
     * @return Vector of Morton IDs partitioning the processor domains.
     */
    const Vector<Morton<DIM>>& GetPartitionMID() const;

    /**
     * @return Vector of Morton IDs of tree nodes.
     */
    const Vector<Morton<DIM>>& GetNodeMID() const;

    /**
     * @return Vector of attributes of tree nodes.
     */
    const Vector<NodeAttr>& GetNodeAttr() const;

    /**
     * @return Vector of node-lists of tree nodes.
     */
    const Vector<NodeLists>& GetNodeLists() const;

    /**
     * @return The communicator.
     */
    const Comm& GetComm() const;

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
    template <class Real> void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, Periodicity periodicity = Periodicity::NONE, Integer halo_size = -1);

    /**
     * Add named data to the tree nodes.
     *
     * @param[in] name Name for the data. Must not already exist on this tree.
     * @param[in] data Contiguous data for all nodes, concatenated in node
     * order. Must satisfy `data.Dim() == dof * sum(cnt)` for some `dof >= 0`.
     * @param[in] cnt Number of data elements per node (length = number of tree nodes).
     *
     * @note Collective; must be called from all processes.
     *
     * @warning Storage is type-erased. `GetData<U>` for this `name` only
     * round-trips when `U` matches the `ValueType` used here.
     */
    template <class ValueType> void AddData(const std::string& name, const Vector<ValueType>& data, const Vector<Long>& cnt);

    /**
     * Add named data without values: `cnt[i] * dof` unwritten elements for node i, to be filled in
     * place through the view `GetData` returns. Local, no communication.
     *
     * @param[in] name Name for the data. Must not already exist on this tree.
     * @param[in] dof Elements per data item; must agree across processes.
     * @param[in] cnt Number of data items per node (length = number of tree nodes).
     */
    template <class ValueType> void AddData(const std::string& name, Long dof, const Vector<Long>& cnt);

    /**
     * Get node data.
     *
     * @param[out] data Non-owning view of the tree's internal buffer for this
     * data. Must not be resized; in-place mutation aliases the stored data. A
     * const tree fills a `Vector<const ValueType>`.
     * @param[out] cnt Number of data elements per node (length = number of tree nodes). Non-owning view; must not be modified.
     * @param[in] name Name of the data.
     *
     * @warning `ValueType` must match the type used in the corresponding
     * `AddData`; otherwise the bytes are silently reinterpreted.
     */
    template <class ValueType> void GetData(Vector<ValueType>& data, Vector<Long>& cnt, const std::string& name);
    template <class ValueType> void GetData(Vector<const ValueType>& data, Vector<Long>& cnt, const std::string& name) const;

    /**
     * Reduce data on nodes shared between processors and then broadcast the halo/ghost node data. The resulting tree
     * will have ghost nodes added to the tree.
     *
     * @param[in] name Name of the data.
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    template <class ValueType> void ReduceBroadcast(const std::string& name);

    /**
     * Broadcast the halo/ghost node data. The resulting tree will have ghost nodes added to the tree.
     *
     * @param[in] name Name of the data.
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    template <class ValueType> void Broadcast(const std::string& name);

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

  protected:

    void GetData_(Iterator<Vector<char>>& data, Iterator<Vector<Long>>& cnt, const std::string& name);

    /** Exclusive scan of `cnt` into `dsp`, returning the total the scan already accumulated. */
    static Long scan(Vector<Long>& dsp, const Vector<Long>& cnt);

    std::set<std::string> data_moved_by_derived;  ///< payloads a derived class moves itself after a rebuild; UpdateRefinement skips them

  private:

    Vector<Morton<DIM>> mins;
    Vector<Morton<DIM>> node_mid;
    Vector<NodeAttr> node_attr;
    Vector<NodeLists> node_lst;

    std::map<std::string, Vector<char>> node_data;
    std::map<std::string, Vector<Long>> node_cnt;

    Vector<Morton<DIM>> user_mid;
    Vector<Long> user_cnt;

    Comm comm;
};

/**
 * Class template representing a point tree in a specified dimension.
 *
 * @tparam Real Data type for the coordinates and values of points.
 * @tparam DIM Dimensionality of the point tree.
 * @tparam BaseTree Base class for the point tree. Defaults to Tree<DIM>.
 */
template <class Real, Integer DIM, class BaseTree = Tree<DIM>> class PtTree : public BaseTree {
  public:

    /**
     * Constructor for PtTree.
     *
     * @param comm Communication object for distributed computing. Defaults to Comm::Self().
     */
    PtTree(const Comm& comm = Comm::Self());

    /**
     * Destructor for PtTree.
     */
    ~PtTree();

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
    void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, Periodicity periodicity = Periodicity::NONE, Integer halo_size = -1);

    /**
     * Add particles to the point tree.
     *
     * @param name Name of the particle group.
     * @param coord Coordinates of the particles.
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    void AddParticles(const std::string& name, const Vector<Real>& coord);

    /**
     * Add particle data to the point tree.
     *
     * @param data_name Name of the data. Must not already exist.
     * @param particle_name Name of an existing particle group from `AddParticles`.
     * @param data Local data values, `dof` per local particle of `particle_name` for some
     * implicit `dof`. Reordered to match the particle group.
     *
     * @note Collective; must be called from all processes.
     */
    void AddParticleData(const std::string& data_name, const std::string& particle_name, const Vector<Real>& data);

    /**
     * Add particle data without values: `dof` unwritten values per particle of `particle_name`, in
     * the group's tree order, to be filled in place through the view `GetData` returns.
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
    void GetParticleData(Vector<Real>& data, const std::string& data_name) const;

    /**
     * Delete particle data from the point tree.
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
     * Example function demonstrating usage of the PtTree class.
     *
     * This function creates a PtTree object, adds particles, performs tree manipulation, and generates visualization.
     */
    static void test();

  private:

    void SetPartitionCodes();  ///< partition_codes from GetPartitionMID()

    Vector<MortonCode<DIM>> partition_codes;  ///< partition mins as codes: the SortScatter splitters; set with each partition
    std::map<std::string, SortScatter<MortonCode<DIM>>> groups;  ///< per particle group: codes in tree order, maps to/from caller order
    struct PtData { std::string particle_name; Long dof; };
    std::map<std::string, PtData> pt_data;  ///< particle group and dof of each particle data set
};

}

#endif // _SCTL_TREE_HPP_
