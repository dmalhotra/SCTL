#ifndef _SCTL_TREE_
#define _SCTL_TREE_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(morton.hpp)
#include SCTL_INCLUDE(vtudata.hpp)
#include SCTL_INCLUDE(ompUtils.hpp)

#include <string>
#include <vector>
#include <algorithm>

namespace SCTL_NAMESPACE {

template <Integer DIM> class Tree {
  public:

    struct NodeAttr {
      unsigned char Leaf : 1, Ghost : 1;
    };

    struct NodeLists {
      Long p2n;                             // path-to-node: id among the siblings
      Long parent;                          // index of the parent node
      Long child[1 << DIM];                 // index of the children
      Long nbr[sctl::pow<DIM,Integer>(3)];  // index of the neighbors at the same level
    };

    /**
     * @return the number of spatial dimensions.
     */
    static constexpr Integer Dim();

    Tree(const Comm& comm_ = Comm::Self());

    ~Tree();

    /**
     * @return vector of Morton IDs partitioning the processor domains.
     */
    const Vector<Morton<DIM>>& GetPartitionMID() const;

    /**
     * @return vector of Morton IDs of tree nodes.
     */
    const Vector<Morton<DIM>>& GetNodeMID() const;

    /**
     * @return vector of attributes of tree nodes.
     */
    const Vector<NodeAttr>& GetNodeAttr() const;

    /**
     * @return vector of node-lists of tree nodes.
     */
    const Vector<NodeLists>& GetNodeLists() const;

    /**
     * @return the communicator.
     */
    const Comm& GetComm() const;

    /**
     * Update tree refinement and repartition node data among the new tree nodes.
     *
     * @param[in] coord particle coordinates (in [0,1]^dim stored in AoS order) that describe the new tree refinement.
     *
     * @param[in] M maximum number of particles per tree node.
     *
     * @param[in] balance21 whether to do level-restriction (2:1 balance refinement).
     *
     * @param[in] periodic whether the tree is periodic across the faces of the cube.
     *
     * @note this is a collective operation.
     */
    template <class Real> void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, bool periodic = 0);

    /**
     * Add named data to the tree nodes.
     *
     * @param[in] name name for the data.
     *
     * @param[in] data vector containing the contiguous data for all nodes.
     *
     * @param[in] cnt vector of length equal to number of tree nodes, giving the number of data elements per node.
     *
     * @note this is a collective operation.
     */
    template <class ValueType> void AddData(const std::string& name, const Vector<ValueType>& data, const Vector<Long>& cnt);

    /**
     * Get node data.
     *
     * @param[out] data vector containing the contiguous data of all nodes. The vector does not own the memory, and
     * therefore must not be modified or resized. (Technically, data may be modified in-place, but it violates const
     * correctness).
     *
     * @param[out] cnt vector of length equal to number of tree nodes, giving the number of data elements per node.  The
     * vector does not own the memory and must not be modified.
     *
     * @param[in] name name of the data
     */
    template <class ValueType> void GetData(Vector<ValueType>& data, Vector<Long>& cnt, const std::string& name) const;

    /**
     * @note this is a collective operation.
     */
    template <class ValueType> void ReduceBroadcast(const std::string& name);

    /**
     * @note this is a collective operation.
     */
    template <class ValueType> void Broadcast(const std::string& name);

    /**
     * Delete data from the tree nodes.
     *
     * @param[in] name name of the data.
     *
     * @note this is a collective operation.
     */
    void DeleteData(const std::string& name);

    /**
     * Write VTK visualization.
     *
     * @param[in] fname filename for the output.
     *
     * @param[in] show_ghost whether to show ghost nodes.
     *
     * @note this is a collective operation.
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

template <class Real, Integer DIM, class BaseTree = Tree<DIM>> class PtTree : public BaseTree {
  public:

    PtTree(const Comm& comm = Comm::Self());

    ~PtTree();

    /**
     * @note this is a collective operation.
     */
    void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, bool periodic = 0);

    /**
     * @note this is a collective operation.
     */
    void AddParticles(const std::string& name, const Vector<Real>& coord);

    /**
     * @note this is a collective operation.
     */
    void AddParticleData(const std::string& data_name, const std::string& particle_name, const Vector<Real>& data);

    /**
     * @note this is a collective operation.
     */
    void GetParticleData(Vector<Real>& data, const std::string& data_name) const;

    /**
     * @note this is a collective operation.
     */
    void DeleteParticleData(const std::string& data_name);

    /**
     * @note this is a collective operation.
     */
    void WriteParticleVTK(std::string fname, std::string data_name, bool show_ghost = false) const;

    static void test() {
      Long N = 100000;
      Vector<Real> X(N*DIM), f(N);
      for (Long i = 0; i < N; i++) { // Set coordinates (X), and values (f)
        f[i] = 0;
        for (Integer k = 0; k < DIM; k++) {
          X[i*DIM+k] = pow<3>(drand48()*2-1.0)*0.5+0.5;
          f[i] += X[i*DIM+k]*k;
        }
      }

      PtTree<Real,DIM> tree;
      tree.AddParticles("pt", X);
      tree.AddParticleData("pt-value", "pt", f);
      tree.UpdateRefinement(X, 1000); // refine tree with max 1000 points per box.

      { // manipulate tree node data
        const auto& node_lst = tree.GetNodeLists(); // Get interaction lists
        //const auto& node_mid = tree.GetNodeMID();
        //const auto& node_attr = tree.GetNodeAttr();

        // get point values and count for each node
        Vector<Real> value;
        Vector<Long> cnt, dsp;
        tree.GetData(value, cnt, "pt-value");

        // compute the dsp (the point offset) for each node
        dsp.ReInit(cnt.Dim()); dsp = 0;
        omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());

        Long node_idx = 0;
        for (Long i = 0; i < cnt.Dim(); i++) { // find the tree node with maximum points
          if (cnt[node_idx] < cnt[i]) node_idx = i;
        }

        for (Long j = 0; j < cnt[node_idx]; j++) { // for this node, set all pt-value to -1
          value[dsp[node_idx]+j] = -1;
        }

        for (const Long nbr_idx : node_lst[node_idx].nbr) { // loop over the neighbors and set pt-value to 2
          if (nbr_idx >= 0 && nbr_idx != node_idx) {
            for (Long j = 0; j < cnt[nbr_idx]; j++) {
              value[dsp[nbr_idx]+j] = 2;
            }
          }
        }
      }

      // Generate visualization
      tree.WriteParticleVTK("pt", "pt-value");
      tree.WriteTreeVTK("tree");
    }

  private:

    std::map<std::string, Long> Nlocal;
    std::map<std::string, Vector<Morton<DIM>>> pt_mid;
    std::map<std::string, Vector<Long>> scatter_idx;
    std::map<std::string, std::string> data_pt_name;
};

}

#include SCTL_INCLUDE(tree.txx)

#endif //_SCTL_TREE_
