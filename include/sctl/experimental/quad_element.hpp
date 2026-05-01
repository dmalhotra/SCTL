#ifndef _SCTL_QUAD_ELEMENT_HPP_
#define _SCTL_QUAD_ELEMENT_HPP_

#include <string>
#include <sctl.hpp>

namespace sctl {

  class VTUData;
  template <class ValueType> class Matrix;

  /**
   * Implements the abstract class ElementListBase for list of high-order
   * quadrilateral surface elements represented on tensor-product Gauss-Legendre
   * nodes.
   *
   * Each element uses a single order N and is described by N x N nodes on the
   * reference square [0,1] x [0,1], ordered lexicographically in (u,v) with u
   * as the slow index and v as the fast index.
   *
   * @see ElementListBase
   */
  template <class Real> class QuadElemList : public ElementListBase<Real> {
      static constexpr Integer COORD_DIM = 3;

    public:

      /**
       * Constructor.
       */
      QuadElemList() {}

      /**
       * Construct the element list from nodal coordinates.
       *
       * @param[in] order polynomial order of each element.
       *
       * @param[in] coord coordinates of the tensor-product Gauss-Legendre
       * nodes in AoS order {x1,y1,z1,...,xn,yn,zn}.
       */
      template <class ValueType> QuadElemList(Integer order, const Vector<ValueType>& coord);

      /**
       * Initialize list of elements from nodal coordinates.
       *
       * @param[in] order polynomial order of each element.
       *
       * @param[in] coord coordinates of the tensor-product Gauss-Legendre
       * nodes in AoS order {x1,y1,z1,...,xn,yn,zn}.
       */
      template <class ValueType> void Init(Integer order, const Vector<ValueType>& coord);

      /**
       * Destructor.
       */
      virtual ~QuadElemList() {}

      /**
       * Return the number of elements in the list.
       */
      Long Size() const override;

      /**
       * Return the polynomial order of the elements.
       */
      Integer Order() const;

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
       * the nodes for computing the far-field potential from the surface.
       *
       * @see ElementListBase::GetFarFieldNodes()
       */
      void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const override;

      /**
       * Compute the self-interaction operator matrix for each element.
       *
       * @see ElementListBase::SelfInterac()
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Compute the near-interaction operator matrix for a given element and
       * each target point.
       *
       * @see ElementListBase::NearInterac()
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /**
       * Returns the reference-space Gauss-Legendre nodes for a given order.
       *
       * @param[in] Order the polynomial order of the element.
       *
       * @return the location of the discretization nodes in [0,1].
       */
      static const Vector<Real>& ParamNodes(const Integer Order);

      /**
       * Write elements to file.
       *
       * @param[in] fname the filename.
       *
       * @param[in] comm the communicator.
       */
      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      /**
       * Read elements from file.
       *
       * @param[in] fname the filename.
       *
       * @param[in] comm the communicator.
       */
      template <class ValueType> void Read(const std::string& fname, const Comm& comm = Comm::Self());

      /**
       * Get geometry data for an element on a tensor-product grid of parameter
       * values 'u' and 'v'.
       *
       * @param[out] X (optional) coordinates of the surface points in AoS order.
       *
       * @param[out] Xn (optional) surface normal of the points in AoS order.
       *
       * @param[out] Xa (optional) differential area-element at the surface points.
       *
       * @param[out] dX_du (optional) the surface-gradient in 'u'-direction (AoS order).
       *
       * @param[out] dX_dv (optional) the surface-gradient in 'v'-direction (AoS order).
       *
       * @param[in] u_param vector of 'u' values in [0,1].
       *
       * @param[in] v_param vector of 'v' values in [0,1].
       *
       * @param[in] elem_idx index of the element whose geometry is requested.
       */
      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx) const;

      /**
       * Get the VTU (Visualization Toolkit for Unstructured grids) data for
       * one or all elements.
       */
      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F = Vector<Real>(), const Long elem_idx = -1) const;

      /**
       * Write VTU data to file.
       *
       * @param[in] fname the filename.
       *
       * @param[in] F the data values at each surface discretization node in
       * AoS order {Ux1,Uy1,Uz1,...,Uxn,Uyn,Uzn}.
       *
       * @param[in] comm the communicator.
       */
      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      /**
       * Create a copy of the element-list possibly with a different precision.
       *
       * @param[in] elem_lst input element-list
       */
      template <class ValueType> void Copy(QuadElemList<ValueType>& elem_lst) const;

      template<typename> friend class QuadElemList;

    private:

      template <class ValueType> static void EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv);

      void BuildDerivativeCache();

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
