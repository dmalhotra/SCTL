#ifndef _SCTL_QUAD_ELEMENT_HPP_
#define _SCTL_QUAD_ELEMENT_HPP_

#include <string>
#include <utility>
#include <vector>
#include <sctl.hpp>

namespace sctl {

  class VTUData;
  template <class ValueType> class Matrix;

  namespace detail_quadelem {
    template <class Real> struct Access;
  }

  /**
   * Implements the abstract class ElementListBase (in SCTL) for a list of
   * quadrilateral surface elements in 3D. Each element is a tensor-product
   * polynomial map from the parameter square [0,1]x[0,1] to the surface,
   * interpolating the element's order x order nodes, which lie at the
   * parameter points ParamNodes(order) in each of the directions 'u' and 'v'.
   *
   * @see ElementListBase
   */
  template <class Real> class QuadElemList : public ElementListBase<Real> {

      friend struct detail_quadelem::Access<Real>;

    public:
      /**
       * Constructor
       */
      QuadElemList() {}

      /**
       * Construct the element list from the coordinates of the element nodes.
       *
       * @param[in] order number of nodes in each parameter direction, the same
       * for every element.
       *
       * @param[in] coord coordinates of the nodes, element by element, in the
       * order {x1,y1,z1,...,xn,yn,zn}. Within an element, node i*order+j is at
       * the parameter point (u_i, v_j), where u and v are ParamNodes(order).
       */
      template <class ValueType> QuadElemList(const Integer order, const Vector<ValueType>& coord);

      /**
       * Initialize the element list from the coordinates of the element nodes.
       *
       * @param[in] order number of nodes in each parameter direction, the same
       * for every element.
       *
       * @param[in] coord coordinates of the nodes, element by element, in the
       * order {x1,y1,z1,...,xn,yn,zn}. Within an element, node i*order+j is at
       * the parameter point (u_i, v_j), where u and v are ParamNodes(order).
       */
      template <class ValueType> void Init(const Integer order, const Vector<ValueType>& coord);

      /**
       * Destructor
       */
      virtual ~QuadElemList() {}

      /**
       * Return the number of elements in the list.
       */
      Long Size() const override;

      /**
       * Return the number of nodes in each parameter direction of an element.
       */
      Integer Order() const;

      /**
       * Get geometry data for an element on a tensor-product grid of
       * parameter values 'u' and 'v'. Output point a*Nv+b, with
       * Nv = v_param.Dim(), is at the parameter point (u_param[a], v_param[b]).
       *
       * @param[out] X (optional) coordinates of the surface points in AoS order.
       *
       * @param[out] Xn (optional) unit normal, in the direction of the cross
       * product of dX_du and dX_dv (AoS order).
       *
       * @param[out] Xa (optional) differential area-element, the length of the
       * cross product of dX_du and dX_dv.
       *
       * @param[out] dX_du (optional) derivative of X with respect to 'u' (AoS order).
       *
       * @param[out] dX_dv (optional) derivative of X with respect to 'v' (AoS order).
       *
       * @param[in] u_param vector of 'u' values (in the range [0,1]).
       *
       * @param[in] v_param vector of 'v' values (in the range [0,1]).
       *
       * @param[in] elem_idx index of the element whose geometry is requested.
       */
      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx) const;

      /**
       * Returns the position and normals of the surface nodal points for each
       * element.
       *
       * @see ElementListBase::GetNodeCoord()
       */
      void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const override;

      /**
       * Given an accuracy tolerance, returns the quadrature node positions,
       * the normals at the nodes, the weights and the cut-off distance from
       * the nodes for computing the far-field potential from the surface (at
       * target points beyond the cut-off distance). The quadrature nodes are
       * the element nodes.
       *
       * @see ElementListBase::GetFarFieldNodes()
       */
      void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const override;

      /**
       * Compute self-interaction operator for each element, with the
       * quadrature scheme set by SetQuadScheme(). The element order must be
       * one of {4, 8, 12, 16, 20}.
       *
       * @see ElementListBase::SelfInterac()
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const Real tol, const bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Compute near-interaction operator for a given element-idx and each
       * target, with the quadrature scheme set by SetQuadScheme(). The element
       * order must be one of {4, 8, 12, 16, 20}.
       *
       * @see ElementListBase::NearInterac()
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /**
       * Quadrature scheme for the self-interactions (targets at the element
       * nodes) and the near-interactions.
       *
       * - TensorProduct: self, a tensor-product rule of Gauss-Legendre panels
       *   halving toward the target node, with Alpert panels corrected for the
       *   log singularity at the node along 'v'; near, Gauss-Legendre segments
       *   graded toward the target's closest point on the element.
       *
       * - Duffy: self, the element split into four triangles joining the target
       *   node to the edges, each integrated with a Duffy transform; near, the
       *   element split at the target's closest point into rectangles refined
       *   dyadically toward it.
       *
       * - Hedgehog: self, the potential at five proxy points along the normal
       *   at the target node, computed with the near rule of Duffy and
       *   extrapolated to the surface; near, as for Duffy.
       *
       * At a target on the surface, TensorProduct and Duffy return the
       * principal value and Hedgehog the limit from the side of the normal;
       * for a double-layer kernel the two differ by the jump, half the density.
       */
      enum class QuadScheme { TensorProduct, Duffy, Hedgehog };

      /**
       * Set the quadrature scheme for the self- and near-interactions. The
       * default is QuadScheme::Duffy.
       *
       * @param[in] s the quadrature scheme.
       */
      void SetQuadScheme(const QuadScheme s) {
        scheme_ = s;
      }

      /**
       * Returns the parameter values of the element nodes in each direction.
       *
       * @param[in] order the number of nodes in each parameter direction.
       *
       * @return the Gauss-Legendre nodes in the interval [0,1].
       */
      static const Vector<Real>& ParamNodes(const Integer order);

      /**
       * Write elements to file. Each process writes its elements to the file
       * named fname followed by its rank as a six-digit number, with one line
       * per node holding its coordinates; the line of the first node of each
       * element also holds the element order.
       *
       * @param[in] fname the filename.
       *
       * @param[in] comm the communicator.
       */
      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      /**
       * Read elements from the files written by Write(). All elements must
       * have the same order.
       *
       * @param[in] fname the filename.
       *
       * @param[in] comm the communicator.
       *
       * @note this is a collective operation.
       */
      template <class ValueType> void Read(const std::string& fname, const Comm& comm = Comm::Self());

      /**
       * Get the VTU (Visualization Toolkit for Unstructured grids) data for
       * one or all elements. Each element is appended as a grid of
       * quadrilateral cells whose vertices are at the parameter values
       * {0, ParamNodes(order), 1} in each direction.
       *
       * @param[out] vtu_data the VTU data, to which the elements are appended.
       *
       * @param[in] F (optional) the data values at the nodes in AoS order,
       * for all elements, or for the element elem_idx if it is not -1.
       *
       * @param[in] elem_idx index of the element, or -1 for all elements.
       */
      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F = Vector<Real>(), const Long elem_idx = -1) const;

      /**
       * Write VTU data to file.
       *
       * @param[in] fname the filename.
       *
       * @param[in] F (optional) the data values at the nodes in AoS order,
       * with any number of values per node.
       *
       * @param[in] comm the communicator.
       */
      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      /**
       * Copy the element-list, possibly to a different precision (ValueType).
       *
       * @param[out] elem_lst the element-list that receives the copy.
       */
      template <class ValueType> void Copy(QuadElemList<ValueType>& elem_lst) const;

      template<typename> friend class QuadElemList;

      template<typename> friend struct QuadElemTestAccess;

    private:

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord; // node coordinates; per element, x of all nodes, then y, then z
      Vector<Real> dcoord_du, dcoord_dv; // derivatives of coord along 'u' and 'v', in the same layout
      Vector<Real> X_node, Xn_node; // node positions and unit normals in AoS order
      Vector<Long> node_cnt; // number of nodes of each element
      QuadScheme scheme_ = QuadScheme::Duffy;
  };

}

#endif
