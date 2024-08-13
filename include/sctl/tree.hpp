/**
 * @file tree.hpp
 * Definition of Tree and PtTree classes.
 */

#ifndef _SCTL_TREE_HPP_
#define _SCTL_TREE_HPP_

#include <map>                  // for map
#include <string>               // for basic_string, string

#include "sctl/common.hpp"      // for Long, Integer, sctl
#include "sctl/comm.hpp"        // for Comm
#include "sctl/comm.txx"        // for Comm::Self
#include "sctl/math_utils.txx"  // for pow
#include "sctl/vector.hpp"      // for Vector

namespace sctl {

template <Integer DIM> class Morton;

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
     * @param[in] periodic Whether the tree is periodic across the faces of the cube.
     * @param[in] halo_size 2^halo_size neighboring boxes will be included in the halo region
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    template <class Real> void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, bool periodic = 0, Integer halo_size = -1);

    /**
     * Add named data to the tree nodes.
     *
     * @param[in] name Name for the data.
     * @param[in] data Vector containing the contiguous data for all nodes.
     * @param[in] cnt Vector of length equal to number of tree nodes, giving the number of data elements per node.
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    template <class ValueType> void AddData(const std::string& name, const Vector<ValueType>& data, const Vector<Long>& cnt);

    /**
     * Get node data.
     *
     * @param[out] data Vector containing the contiguous data of all nodes. The vector does not own the memory, and
     * therefore must not be modified or resized. (Technically, data may be modified in-place, but it violates const
     * correctness).
     * @param[out] cnt Vector of length equal to number of tree nodes, giving the number of data elements per node. The
     * vector does not own the memory and must not be modified.
     * @param[in] name Name of the data.
     */
    template <class ValueType> void GetData(Vector<ValueType>& data, Vector<Long>& cnt, const std::string& name) const;

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

    static void scan(Vector<Long>& dsp, const Vector<Long>& cnt);

    template <typename A, typename B> struct SortPair {
      int operator<(const SortPair<A, B> &p1) const { return key < p1.key; }
      A key;
      B data;
    };

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
     * @param periodic Flag indicating periodic boundary conditions.
     * @param[in] halo_size 2^halo_size neighboring boxes will be included in the halo region
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, bool periodic = 0, Integer halo_size = -1);

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
     * @param data_name Name of the data.
     * @param particle_name Name of the particle group.
     * @param data Data values associated with the particles.
     *
     * @note This is a collective operation and must be called from all processes in the communicator.
     */
    void AddParticleData(const std::string& data_name, const std::string& particle_name, const Vector<Real>& data);

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

    std::map<std::string, Long> Nlocal;                    ///< Number of local particles for each group.
    std::map<std::string, Vector<Morton<DIM>>> pt_mid;     ///< Morton indices for each particle group.
    std::map<std::string, Vector<Long>> scatter_idx;       ///< Scatter indices for each particle group.
    std::map<std::string, std::string> data_pt_name;       ///< Mapping of data name to particle name.
};

}

#endif // _SCTL_TREE_HPP_
