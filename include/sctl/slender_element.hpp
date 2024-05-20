#ifndef _SCTL_SLENDER_ELEMENT_HPP_
#define _SCTL_SLENDER_ELEMENT_HPP_

#include <string>                      // for string

#include "sctl/common.hpp"             // for Long, Integer, SCTL_NAMESPACE
#include SCTL_INCLUDE(boundary_integral.hpp)  // for ElementListBase
#include SCTL_INCLUDE(comm.hpp)               // for Comm
#include SCTL_INCLUDE(comm.txx)               // for Comm::Self
#include SCTL_INCLUDE(vector.hpp)             // for Vector
#include SCTL_INCLUDE(vector.txx)             // for Vector::~Vector<ValueType>

namespace SCTL_NAMESPACE {

  class VTUData;
  template <class ValueType> class Matrix;

  /**
   * Implements the abstract class ElementListBase for list of slender boundary
   * elements with circular cross-section.
   *
   * @see ElementListBase
   */
  template <class Real> class SlenderElemList : public ElementListBase<Real> {
      static constexpr Integer FARFIELD_UPSAMPLE = 1;
      static constexpr Integer COORD_DIM = 3;

      static constexpr Integer ModalUpsample = 1; // toroidal quadrature order is FourierModes+ModalUpsample

    public:

      /**
       * Constructor
       */
      SlenderElemList() {}

      /**
       * Construct the element list from centerline coordinates and
       * cross-sectional radius evaluated the panel discretization nodes.
       *
       * @param[in] cheb_order vector of Chebyshev order of the elements.
       *
       * @param[in] fourier_order vector of Fourier order of the elements.
       *
       * @param[in] coord coordinates of the centerline discretization nodes in
       * the order {x1,y1,z1,...,zn,yn,zn}.
       *
       * @param[in] radius cross-sectional radius at the centerline
       * discretization nodes.
       *
       * @param[in] orientation optional orientation vector at the centerline
       * discretization nodes in the order {ex1,ey1,ez1,...,ezn,eyn,ezn}.
       */
      template <class ValueType> SlenderElemList(const Vector<Long>& cheb_order, const Vector<Long>& fourier_order, const Vector<ValueType>& coord, const Vector<ValueType>& radius, const Vector<ValueType>& orientation = Vector<ValueType>());

      /**
       * Initialize list of elements from centerline coordinates and
       * cross-sectional radius evaluated the panel discretization nodes.
       *
       * @param[in] cheb_order vector of Chebyshev order of the elements.
       *
       * @param[in] fourier_order vector of Fourier order of the elements.
       *
       * @param[in] coord coordinates of the centerline discretization nodes in
       * the order {x1,y1,z1,...,zn,yn,zn}.
       *
       * @param[in] radius cross-sectional radius at the centerline
       * discretization nodes.
       *
       * @param[in] orientation optional orientation vector at the centerline
       * discretization nodes in the order {ex1,ey1,ez1,...,ezn,eyn,ezn}.
       */
      template <class ValueType> void Init(const Vector<Long>& cheb_order, const Vector<Long>& fourier_order, const Vector<ValueType>& coord, const Vector<ValueType>& radius, const Vector<ValueType>& orientation = Vector<ValueType>());

      /**
       * Destructor
       */
      virtual ~SlenderElemList() {}

      /**
       * Return the number of elements in the list.
       */
      Long Size() const override;

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
       * target points beyond the cut-off distance).
       *
       * @see ElementListBase::GetFarFieldNodes()
       */
      void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const override;

      /**
       * Interpolates the density from surface node points to far-field
       * quadrature node points.
       *
       * @see ElementListBase::GetFarFieldDensity()
       */
      void GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const override;

      /**
       * Apply the transpose of the GetFarFieldDensity() operator applied to
       * the column-vectors of Min and the result is returned in Mout.
       *
       * @see ElementListBase::FarFieldDensityOperatorTranspose()
       */
      void FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const override;

      /**
       * Compute self-interaction operator for each element.
       *
       * @see ElementListBase::SelfInterac()
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Compute near-interaction operator for a given element-idx and each target.
       *
       * @see ElementListBase::NearInterac()
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /**
       * Returns the Chebyshev node points for a given order.
       *
       * @param[in] the Chebyshev order of the panel.
       *
       * @return the location of the discretization nodes for a panel in the
       * interval [0,1].
       */
      static const Vector<Real>& CenterlineNodes(const Integer Order);

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
       * values 's' and 'theta'.
       *
       * @param[out] X (optional) coordinates of the surface points in AoS order.
       *
       * @param[out] Xn (optional) surface normal of the points in AoS order.
       *
       * @param[out] Xa (optional) differential area-element at the surface points.
       *
       * @param[out] dX_ds (optional) the surface-gradient in 's'-direction (AoS order).
       *
       * @param[out] dX_dt (optional) the surface-gradient in 'theta'-direction (AoS order).
       *
       * @param[in] s_param vector of 's' values (in the range [0-1]).
       *
       * @param[in] sin_theta vector of sin(theta) values.
       *
       * @param[in] cos_theta vector of cos(theta) values.
       *
       * @param[in] elem_idx index of the element whose geometry is requested.
       */
      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_ds, Vector<Real>* dX_dt, const Vector<Real>& s_param, const Vector<Real>& sin_theta, const Vector<Real>& cos_theta, const Long elem_idx) const;

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
       * the order AoS order {Ux1,Uy1,Uz1,...,Uxn,Uyn,Uzn}.
       *
       * @param[in] comm the communicator.
       */
      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      /**
       * Test example for Laplace double-layer kernel.
       */
      template <class Kernel> static void test(const Comm& comm = Comm::Self(), Real tol = 1e-10);

      /**
       * Test example for Green's identity with Laplace kernel.
       */
      static void test_greens_identity(const Comm& comm = Comm::Self(), Real tol = 1e-10);

      /**
       * Create a copy of the element-list possibly from a different a
       * different precision (ValueType).
       *
       * @param[in] elem_lst input element-list
       */
      template <class ValueType> void Copy(SlenderElemList<ValueType>& elem_lst) const;

      template<typename> friend class SlenderElemList;

    private:

      template <class Kernel> Matrix<Real> SelfInteracHelper_(const Kernel& ker, const Long elem_idx, const Real tol) const; // constant radius
      template <Integer digits, bool trg_dot_prod, class Kernel> Matrix<Real> SelfInteracHelper(const Kernel& ker, const Long elem_idx) const;

      template <Integer digits, bool trg_dot_prod, class Kernel> void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx) const;

      Vector<Real> radius, coord, e1;
      Vector<Long> cheb_order, fourier_order, elem_dsp;

      Vector<Real> dr, dx, d2x; // derived quantities
  };

}

#endif // _SCTL_SLENDER_ELEMENT_HPP_
