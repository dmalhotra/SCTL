// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _SCTL_BOUNDARY_INTEGRAL_HPP_
#define _SCTL_BOUNDARY_INTEGRAL_HPP_

#include <cstddef>               // for size_t
#include <map>                   // for map
#include <memory>                // for unique_ptr
#include <string>                // for basic_string, string

#include "sctl/common.hpp"       // for Long, Integer, sctl
#include "sctl/comm.hpp"         // for Comm
#include "sctl/comm.txx"         // for Comm::Self
#include "sctl/fmm-wrapper.hpp"  // for ParticleFMM
#include "sctl/sort-scatter.hpp" // for SortScatter
#include "sctl/vector.hpp"       // for Vector

namespace sctl {

  template <class ValueType> class Matrix;

  /**
   * Working storage for building near lists, reused from one call to the next so that its memory is not requested
   * again. Its contents are private and need no initialization. Defined in boundary_integral.txx.
   */
  template <class Real, Integer COORD_DIM=3> class NearListWork;

  /**
   * Given N target points and K elements (each made of a set of source nodes with given radius that describes its near
   * region), return vectors containing the set of near targets (coordinates and normals) for each element.
   *
   * @param[in] Xtrg vector of length N*COORD_DIM, containing the target coordinates in AoS order {x1,y1,z1,..., xn,yn,zn}.
   *
   * @param[in] Xn_trg vector of target normals in AoS order (can be empty).
   *
   * @param[in] Xsrc vector of all source node coordinates in AoS order.
   *
   * @param[in] src_radius vector of all source node radii.
   *
   * @param[in] src_elem_nds_cnt vector of length K, containing the number of source nodes in each element.
   *
   * @param[in] src_elem_nds_dsp vector of length K, containing the offset to the first source node of each element.
   *
   * @param[in] comm MPI communicator.
   *
   *
   * @param[out] Xtrg_near vector containing the target coordinates near each element (in AoS order).
   *
   * @param[out] Xn_trg_near vector containing the normal vector at each target near each element (in AoS order).
   *
   * @param[out] near_elem_cnt vector of length K, containing the number of near target points (in Xtrg_near) for each
   * element.
   *
   * @param[out] near_elem_dsp vector of length K, containing the offset of the first near target point (in Xtrg_near)
   * for each element.
   *
   * @param[out] near_sort_scatter the target index of each target point in the near list (Xtrg_near), sorted and cut at
   * the first target of each process. Once the near potential has been computed for each pair of element and its near
   * targets, its ScatterForward() moves the potentials to the order of the targets (Xtrg).
   *
   * @param[out] near_trg_cnt, near_trg_dsp vectors of length N.  After moving the potential for element-target pairs
   * with near_sort_scatter, the potential at the i-th target is given by summing over near_trg_cnt[i] array elements
   * starting at index near_trg_dsp[i].
   *
   * @param[in,out] work (optional) working storage; its contents on input are not used, and its storage is reused from
   * one call to the next. If null, temporary storage is used.
   *
   * @note this is a collective operation.
   */
  template <class Real, Integer COORD_DIM=3> void BuildNearList(Vector<Real>& Xtrg_near, Vector<Real>& Xn_trg_near, Vector<Long>& near_elem_cnt, Vector<Long>& near_elem_dsp, SortScatter<Long>& near_sort_scatter, Vector<Long>& near_trg_cnt, Vector<Long>& near_trg_dsp, const Vector<Real>& Xtrg, const Vector<Real>& Xn_trg, const Vector<Real>& Xsrc, const Vector<Real>& src_radius, const Vector<Long>& src_elem_nds_cnt, const Vector<Long>& src_elem_nds_dsp, const Comm& comm, NearListWork<Real,COORD_DIM>* work = nullptr);

  /**
   * Abstract base class for an element-list. In addition to the functions
   * declared in this base class, the derived class must be copy-constructible.
   */
  template <class Real> class ElementListBase {

    public:

      /**
       * Virtual destructor.
       */
      virtual ~ElementListBase() {}

      /**
       * Return the number of elements in the list.
       */
      virtual Long Size() const = 0;

      /**
       * Returns the position and normals of the surface nodal points for each
       * element.
       *
       * @param[out] X the position of the node points in array-of-struct
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[out] Xn the normal vectors of the node points in
       * array-of-struct order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[out] element_wise_node_cnt the number of node points
       * belonging to each element.
       */
      virtual void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const = 0;

      /**
       * Given an accuracy tolerance, returns the quadrature node positions,
       * the normals at the nodes, the weights and the cut-off distance from
       * the nodes for computing the far-field potential from the surface (at
       * target points beyond the cut-off distance).
       *
       * @param[out] X the position of the quadrature node points in
       * array-of-struct order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[out] Xn the normal vectors at the quadrature node points in
       * array-of-struct order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[out] wts the weights corresponding to each quadrature node.
       *
       * @param[out] dist_far the cut-off distance from each quadrature node
       * such that quadrature rule will be accurate to the specified tolerance
       * for target points further away than this distance.
       *
       * @param[out] element_wise_node_cnt the number of quadrature node
       * points belonging to each element.
       *
       * @param[in] tol the accuracy tolerance.
       */
      virtual void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const = 0;

      /**
       * Interpolates the density from surface node points to far-field
       * quadrature node points. If Fout is empty, then the density is assumed
       * to be Fin.
       *
       * @param[out] Fout the interpolated density at far-field quadrature
       * nodes in array-of-struct order.
       *
       * @param[in] Fin the input density at surface node points in
       * array-of-struct order.
       */
      virtual void GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const;

      /**
       * Apply the transpose of the GetFarFieldDensity() operator for a single
       * element applied to the column-vectors of Min and the result is
       * returned in Mout. If Mout is empty on return, then identity operator is assumed.
       *
       * @param[out] Mout the output matrix where the column-vectors are the
       * result of the application of the transpose operator. If it has the
       * dimensions of the result on input, the result is written to its
       * storage.
       *
       * @param[in] Min the input matrix whose column-vectors are
       * multiplied by the transpose operator.
       *
       * @param[in] elem_idx the index of the element.
       */
      virtual void FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const;

      /**
       * Compute self-interaction operator matrix for each element.
       *
       * @note this function must not be implemented if EvalNearInterac() is implemented and handles
       * singular interactions.
       *
       * @param[out] M_lst the vector of all self-interaction matrices
       * (in row-major format).
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] tol the accuracy tolerance.
       *
       * @param[in] trg_dot_prod whether to compute dot product of the potential with the target-normal vector.
       *
       * @param[in] self pointer to element-list object.
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Compute near-interaction operator matrix for a given element-idx and each target. This does
       * not include on-surface singular interactions, which should be handled by SelfInterac().
       *
       * @note this function must not be implemented if EvalNearInterac() is implemented and handles
       * near-singular interactions.
       *
       * @param[out] M the near-interaction matrix (in row-major format).
       *
       * @param[in] Xt the position of the target points in array-of-struct
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[in] normal_trg the normal at the target points in array-of-struct
       * order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] tol the accuracy tolerance.
       *
       * @param[in] elem_idx the index of the source element.
       *
       * @param[in] self pointer to element-list object.
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /**
       * Evaluate singular and/or near-singular interactions on the fly. This is an alternative to
       * constructing the operator matrix using SelfInterac() and NearInterac().
       *
       * @note SelfInterac() and/or NearInterac() must not be implemented if this function this
       * function handles singular and/or near-singular interactions.
       *
       * @note If MatrixFree() is defined to return true, then subtracting the incorrect near-field
       * contribution must also be handled by this function.
       *
       * @param[in,out] u the potential vector to which the contribution from singular and/or
       * near-singular interactions will be added. The data is in array-of-struct format in the
       * order of the target points.
       *
       * @param[in] f the input density at surface discretization nodes in array-of-struct order.
       *
       * @param[in] Xt the position of the target points in array-of-struct
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[in] normal_trg the normal at the target points in array-of-struct
       * order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] tol the accuracy tolerance.
       *
       * @param[in] elem_idx the index of the source element.
       *
       * @param[in] self pointer to element-list object.
       */
      template <class Kernel> static void EvalNearInterac(Vector<Real>& u, const Vector<Real>& f, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /**
       * Returns a boolean value indicating whether the local corrections are computed in a
       * matrix-free way.
       *
       * If true, then SelfInterac() and NearInterac() will not be used and the near corrections
       * must be handled entirely by EvalNearInterac() on the fly. EvalNearInterac() must compute
       * the correct singular and near-singular interactions and also subtract the incorrect
       * near-field contribution due to the direct sum in far-field computation.
       *
       * If false, the local correction matrix to subtract the incorrect near-field contribution due
       * to the direct sum in will be constructed. If SelfInterac() and/or NearInterac() are
       * implemented, then their results will be added to the local correction matrix.
       * EvalNearInterac() will also be called if implemented and can be used to handle singular
       * and/or near-singular interactions when SelfInterac() and/or NearInterac() are not
       * implemented.
       */
      virtual bool MatrixFree() const;
  };

  /**
   * Implements parallel computation of boundary integrals.
   *
   * Every method except the getters (GetComm, GetElemList, GetPeriodicity, GetPeriodLength, GetAccuracy) is collective
   * and must be called on every process in the same order; methods that take settings (SetPeriodicity, SetAccuracy,
   * SetFMMKer) must get the same values on every process.
   *
   * @tparam Real Data type for real values.
   * @tparam Kernel The kernel function.
   *
   * @see kernel_functions.hpp
   */
  template <class Real, class Kernel> class BoundaryIntegralOp {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1 = Kernel::TrgDim();
      static constexpr Integer COORD_DIM = Kernel::CoordDim();

    public:

      BoundaryIntegralOp() = delete;
      BoundaryIntegralOp(const BoundaryIntegralOp&) = delete;
      BoundaryIntegralOp& operator= (const BoundaryIntegralOp&) = delete;

      /**
       * Constructor
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] trg_normal_dot_prod whether to compute dot-product of the
       * potential with the target normal vector.
       *
       * @param[in] comm the MPI communicator.
       */
      explicit BoundaryIntegralOp(const Kernel& ker, bool trg_normal_dot_prod = false, const Comm& comm = Comm::Self());

      /**
       * Destructor
       */
      ~BoundaryIntegralOp();

      /**
       * Get the MPI communicator.
       */
      const Comm& GetComm() const;

      /**
       * Set periodicity.
       *
       * @param[in] periodicity periodicity type (NONE, X, XY, XYZ).
       *
       * @param[in] period_length length of the periodic box in each dimension.
       * Must be positive if periodicity is not NONE.
       *
       * @remark Periodicity only supported in 3D and with PVFMM.
       */
      void SetPeriodicity(Periodicity periodicity, Real period_length = 0);

      /**
       * Get periodicity.
       */
      Periodicity GetPeriodicity() const;

      /**
       * Get period length.
       */
      Real GetPeriodLength() const;

      /**
       * Specify quadrature accuracy tolerance.
       */
      void SetAccuracy(Real tol);

      /**
       * Get quadrature accuracy tolerance.
       */
      Real GetAccuracy() const;

      /**
       * Set kernel functions for FMM translation operators (@see pvfmm.org).
       *
       * @param[in] k_s2m source-to-multipole kernel.
       * @param[in] k_s2l source-to-local kernel.
       * @param[in] k_s2t source-to-target kernel.
       * @param[in] k_m2m multipole-to-multipole kernel.
       * @param[in] k_m2l multipole-to-local kernel.
       * @param[in] k_m2t multipole-to-target kernel.
       * @param[in] k_l2l local-to-local kernel.
       * @param[in] k_l2t local-to-target kernel.
       * @param[in] m2l_vol_poten evaluator for analytical potential from a uniform volume source density (for k_m2l).
       * @param[in] m2t_vol_poten evaluator for analytical potential from a uniform volume source density (for k_m2t).
       */
      template <class KerS2M, class KerS2L, class KerS2T, class KerM2M, class KerM2L, class KerM2T, class KerL2L, class KerL2T> void SetFMMKer(const KerS2M& k_s2m, const KerS2L& k_s2l, const KerS2T& k_s2t, const KerM2M& k_m2m, const KerM2L& k_m2l, const KerM2T& k_m2t, const KerL2L& k_l2l, const KerL2T& k_l2t, const typename ParticleFMM<Real,COORD_DIM>::VolPotenT m2l_vol_poten = {}, const typename ParticleFMM<Real,COORD_DIM>::VolPotenT m2t_vol_poten = {});

      /**
       * Add an element-list. A copy of elem_lst is stored; it replaces any element-list of the same name, and is
       * copy-assigned into it when that one has the same type, so that its storage is reused.
       *
       * @param[in] elem_lst an object (of type ElemLstType, derived from the
       * base class ElementListBase) that contains the description of a list of
       * elements.
       *
       * @param[in] name a string name for this element list.
       */
      template <class ElemLstType> void AddElemList(const ElemLstType& elem_lst, const std::string& name = ElemLstTypeName<ElemLstType>());

      /**
       * Get const reference to an element-list.
       *
       * @param[in] name name of the element-list to return.
       *
       * @return const reference to the element-list.
       */
      template <class ElemLstType> const ElemLstType& GetElemList(const std::string& name = ElemLstTypeName<ElemLstType>()) const;

      /**
       * Extract the section of the input vector V corresponding to the nodes of the element with index elem_idx in the
       * element-list with the given name and return it in Ve. If elem_idx is -1, then extract the section corresponding
       * to all elements in the list. The vector Ve does not own the data and is just a view into the input vector V.
       * Ve must not be resized and modifying Ve will modify the corresponding entries in V.
       *
       * @param[out] Ve the output vector containing the section of the input vector V corresponding to the nodes of the
       * element with index elem_idx in the element-list with the given name.
       *
       * @param[in] V the input vector corresponding to data at discretization nodes for all element lists.
       *
       * @param[in] name name of the element-list.
       *
       * @param[in] elem_idx the index of the element for which the section of the input vector V is to be extracted. If
       * elem_idx is -1, then the section corresponding to all elements in the list is extracted.
       */
      void GetElemSubArray(Vector<Real>& Ve, Vector<Real>& V, const std::string& name, const Long elem_idx) const;

      /**
       * Extract the section of the input vector V corresponding to the nodes of the element with index elem_idx in the
       * element-list with the given name and return it in Ve. If elem_idx is -1, then extract the section corresponding
       * to all elements in the list. The vector Ve does not own the data and is just a view into the input vector V. Ve
       * must not be resized and modifying Ve will modify the corresponding entries in V.
       *
       * @tparam[in] ElemLstType the type of the element-list for which the section of the input vector V is to be
       * extracted.
       *
       * @param[out] Ve the output vector containing the section of the input vector V corresponding to the nodes of the
       * element with index elem_idx in the element-list identified by the template parameter ElemLstType.
       *
       * @param[in] V the input vector corresponding to data at discretization nodes for all element lists.
       *
       * @param[in] elem_idx the index of the element for which the section of the input vector V is to be extracted. If
       * elem_idx is -1, then the section corresponding to all elements in the list is extracted.
       *
       * @note the element-list is identified by the template parameter ElemLstType, so the name of the element-list is
       * not needed as an argument. The name of the element-list is generated from the typeid hash of the ElemLstType,
       * so there can only be one element-list for each ElemLstType.
       */
      template <class ElemLstType> void GetElemSubArray(Vector<Real>& Ve, Vector<Real>& V, const Long elem_idx = -1) const;

      /**
       * Delete an element-list.
       *
       * @param[in] name name of the element-list to delete.
       */
      void DeleteElemList(const std::string& name);

      /**
       * Delete an element-list.
       */
      template <class ElemLstType> void DeleteElemList();

      /**
       * Set target point coordinates.
       *
       * @param[in] Xtrg the coordinates of target points in array-of-struct
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       */
      void SetTargetCoord(const Vector<Real>& Xtrg);

      /**
       * Set target point normals.
       *
       * @param[in] Xn_trg the normal vectors at the target points in array-of-struct
       * order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       */
      void SetTargetNormal(const Vector<Real>& Xn_trg);

      /**
       * Get local dimension of the boundary integral operator. Dim(0) is the
       * input dimension and Dim(1) is the output dimension.
       */
      Long Dim(Integer k) const;

      /**
       * Setup the boundary integral operator.
       */
      void Setup() const;

      /**
       * Clear setup data.
       */
      void ClearSetup() const;

      /**
       * Evaluate the boundary integral operator.
       *
       * @param[out] U the potential computed at each target point in
       * array-of-struct order.
       *
       * @param[in] F the charge density at each surface discretization node in
       * array-of-struct order.
       */
      void ComputePotential(Vector<Real>& U, const Vector<Real>& F) const;

      /**
       * Scale input vector by sqrt of the area of the element.
       */
      void SqrtScaling(Vector<Real>& U) const; // TODO: replace by sqrt of surface quadrature weights (not sure if it makes a difference though)

      /**
       * Scale input vector by inv-sqrt of the area of the element.
       */
      void InvSqrtScaling(Vector<Real>& U) const; // TODO: replace by inv-sqrt of surface quadrature weights (not sure if it makes a difference though)

    private:

      template <class ElemLstType> static const std::string& ElemLstTypeName(); // default name of an element-list of type ElemLstType

      void CheckInStep() const; // with SCTL_MEMDEBUG, aborts if the setup state differs between processes

      void SetupBasic() const;
      void SetupFar() const;
      void SetupSelf() const;
      void SetupNear() const;
      void SetupElemArea() const;

      void ComputeFarField(Vector<Real>& U, const Vector<Real>& F) const;

      /**
       * Call fn(K) with the far-field quadrature matrix K from the nodes of an element to the targets Xt. K is
       * source-major, (element node dof) x (target dof), and valid only during the call.
       *
       * @param[in] ker_far the kernel matrix function of the far-field evaluation (fmm.KernelMatrixS2T()).
       * @param[in] elem_lst_idx index of the element-list.
       * @param[in] elem_idx index of the element (over all element-lists).
       * @param[in] Xt target coordinates in array-of-struct order.
       * @param[in] Xn target normals in array-of-struct order (used if trg_normal_dot_prod_ is set).
       * @param[in] fn the function called with K.
       */
      template <class KerFar, class Fn> void FarFieldMatrix(const KerFar& ker_far, const Long elem_lst_idx, const Long elem_idx, const Vector<Real>& Xt, const Vector<Real>& Xn, const Fn& fn) const;

      void ComputeNearInterac(Vector<Real>& U, const Vector<Real>& F) const;

      struct ElemLstData {
        void (*SelfInterac)(Vector<Matrix<Real>>&, const Kernel&, Real, bool, const ElementListBase<Real>*);
        void (*NearInterac)(Matrix<Real>&, const Vector<Real>&, const Vector<Real>&, const Kernel&, Real, const Long, const ElementListBase<Real>*);
        void (*EvalNearInterac)(Vector<Real>&, const Vector<Real>&, const Vector<Real>&, const Vector<Real>&, const Kernel&, Real, const Long, const ElementListBase<Real>*);
      };
      std::map<std::string,std::unique_ptr<ElementListBase<Real>>> elem_lst_map;
      std::map<std::string,ElemLstData> elem_data_map;
      Vector<Real> Xt; // User specified position of target points
      Vector<Real> Xnt; // User specified normal at target points
      Periodicity periodicity_ = Periodicity::NONE;
      Real period_length_ = 0;
      Real tol_;
      Kernel ker_;
      std::size_t fmm_ker_id; // identifies the types of the FMM kernels
      bool trg_normal_dot_prod_;
      Comm comm_;

      mutable bool setup_flag;
      mutable Vector<std::string> elem_lst_name; // name of each element-list (size=Nlst)
      mutable Vector<Long> elem_lst_cnt, elem_lst_dsp; // cnt and dsp of elements for each elem_lst (size=Nlst)
      mutable Vector<Long> elem_nds_cnt, elem_nds_dsp; // cnt and dsp of nodes for each element (size=Nelem)
      mutable Vector<Real> Xsurf; // Position of surface node points (target points for on-surface evaluation)
      mutable Vector<Real> Xn_surf; // Normal at surface node points (normal vector for on-surface evaluation)
      mutable Vector<Real> Xtrg; // Position of target points
      mutable Vector<Real> Xn_trg; // Normal vector at target points
      mutable Vector<Vector<Real>> Xsurf_lst, Xn_surf_lst; // Xsurf and Xn_surf of each element-list (size=Nlst), kept so that their storage is reused
      mutable Vector<Vector<Long>> elem_nds_cnt_lst; // elem_nds_cnt of each element-list (size=Nlst)

      mutable bool setup_far_flag;
      mutable ParticleFMM<Real,COORD_DIM> fmm;
      mutable Vector<Long> elem_nds_cnt_far, elem_nds_dsp_far; // cnt and dsp of far-nodes for each element (size=Nelem)
      mutable Vector<Real> X_far, Xn_far, wts_far; // position, normal and weights for far-field quadrature
      mutable Vector<Real> dist_far; // minimum distance of target points for far-field evaluation
      mutable Vector<Real> F_far; // pre-allocated memory for density in far-field evaluation
      mutable Vector<Vector<Real>> X_far_lst, Xn_far_lst, wts_far_lst, dist_far_lst; // far-field nodes of each element-list (size=Nlst), kept so that their storage is reused
      mutable Vector<Vector<Long>> elem_nds_cnt_far_lst; // elem_nds_cnt_far of each element-list (size=Nlst)

      mutable Vector<Real> elem_area; // area of each element: sum of its far-field quadrature weights for tolerance 1 (size=Nelem, empty until SetupElemArea())
      mutable Vector<Real> area_X, area_Xn, area_wts, area_dist; // far-field nodes, normals, weights and distances for tolerance 1 of one element-list, kept so that their storage is reused
      mutable Vector<Long> area_cnt; // number of those far-field nodes of each element of that element-list

      mutable bool setup_near_flag;
      mutable Vector<Real> Xtrg_near; // position of near-interaction target points sorted by element (size=Nnear*COORD_DIM)
      mutable Vector<Real> Xn_trg_near; // normal at near-interaction target points sorted by element (size=Nnear*COORD_DIM)
      mutable SortScatter<Long> near_sort_scatter; // the trg-idx of each near-interaction: moves near-interactions sorted by elem-idx to sorted by trg-idx
      mutable Vector<Long> near_trg_cnt, near_trg_dsp; // cnt and dsp of near-interactions for each target (size=Ntrg)
      mutable Vector<Long> near_elem_cnt, near_elem_dsp; // cnt and dsp of near-interaction for each element (size=Nelem)
      mutable NearListWork<Real,COORD_DIM> near_list_work; // working storage for building the near list, kept between setups so that its storage is reused
      mutable Vector<Long> K_near_cnt, K_near_dsp; // cnt and dsp of element wise near-interaction matrix (size=Nelem)
      mutable Vector<Real> K_near;
      // Near-eval work units: each is a contiguous target range within one element. Elements are
      // subdivided only when there are too few to fill the thread team (size=Nblk_near)
      mutable Vector<Long> near_blk_elem, near_blk_t0, near_blk_cnt;

      mutable bool setup_self_flag;
      mutable Vector<Matrix<Real>> K_self; // self-interaction matrix for each element (size=Nelem)
      mutable Vector<Real> K_self_data; // storage of the K_self matrices
  };

}

#endif // _SCTL_BOUNDARY_INTEGRAL_HPP_
