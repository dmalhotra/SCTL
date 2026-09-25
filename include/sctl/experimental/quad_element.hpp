#ifndef _SCTL_QUAD_ELEMENT_HPP_
#define _SCTL_QUAD_ELEMENT_HPP_

#include <string>
#include <utility>
#include <vector>
#include <sctl.hpp>

namespace sctl {

  class VTUData;
  template <class ValueType> class Matrix;

  /**
   * High-order quadrilateral surface elements on tensor-product Gauss-Legendre
   * nodes (order N => N x N nodes on [0,1]^2, lexicographic in (u,v), u slow).
   * @see ElementListBase
   */
  template <class Real> class QuadElemList : public ElementListBase<Real> {
      static constexpr Integer COORD_DIM = 3;

    public:
      QuadElemList() {}

      /**
       * Construct from nodal coordinates.
       * @param[in] order polynomial order of each element.
       * @param[in] coord node coords, AoS {x1,y1,z1,...,xn,yn,zn}.
       * @param[in] comm communicator. `coord` holds the whole mesh on every rank; each keeps its
       * own contiguous element slice. TODO: a distributed code should not replicate the mesh.
       */
      template <class ValueType> QuadElemList(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      /**
       * Initialize from nodal coordinates.
       * @param[in] order polynomial order of each element.
       * @param[in] coord node coords, AoS {x1,y1,z1,...,xn,yn,zn}.
       * @param[in] comm communicator. `coord` holds the whole mesh on every rank; each keeps its
       * own contiguous element slice. TODO: a distributed code should not replicate the mesh.
       */
      template <class ValueType> void Init(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      virtual ~QuadElemList() {}

      /** Number of elements. */
      Long Size() const override;

      /** Polynomial order of the elements. */
      Integer Order() const;

      /**
       * Element geometry on a tensor-product (u,v) parameter grid.
       * @param[out] X,Xn,Xa (optional) AoS position, normal, area-element.
       * @param[out] dX_du,dX_dv (optional) AoS surface-gradients in u,v.
       * @param[in] u_param,v_param parameter values in [0,1].
       * @param[in] elem_idx element index.
       * @param[in] origin (optional, COORD_DIM reals) subtracted from nodes before
       * interpolation so X is target-relative and cancellation-free for nearby targets.
       */
      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx, const Vector<Real>* origin = nullptr) const;

      /**
       * Position and normals of the surface nodal points per element.
       * @see ElementListBase::GetNodeCoord()
       */
      void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const override;

      /**
       * Far-field quadrature nodes, normals, weights and cut-off distances for a tolerance.
       * @see ElementListBase::GetFarFieldNodes()
       */
      void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const override;

      /**
       * Self-interaction operator matrix per element.
       * @see ElementListBase::SelfInterac()
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Near-interaction operator matrix for an element and each target.
       * @see ElementListBase::NearInterac()
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);


      /**
       * Near/self singular-quadrature scheme. The tolerance drives all three.
       *   Adaptive  foot-graded separable-tensor near + centered graded/Alpert self (default).
       *   Duffy     split-at-foot near + Duffy edge-collapsed, sinh-substituted self.
       *   Hedgehog  Duffy's split-at-foot near, with the self block built by line-QBX: a short
       *             line of proxy points along the outward normal, integrated with that near
       *             scheme and extrapolated back to the surface.
       *
       * CAVEAT for Hedgehog: extrapolating from off-surface proxies gives the ONE-SIDED limit,
       * not the principal value the other two return. The two differ by half the jump, so a
       * caller mixing the conventions must correct for it.
       */
      enum class QuadScheme { Adaptive, Duffy, Hedgehog };

      /**
       * Set the singular-quadrature scheme.
       * @param[in] s scheme (see QuadScheme; default Adaptive).
       */
      void SetQuadScheme(QuadScheme s) {
        scheme_ = s;
      }

      /** Reference-space Gauss-Legendre nodes in [0,1]. */
      static const Vector<Real>& ParamNodes(const Integer Order);

      /** Write elements to file. */
      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      /** Read elements from file, partitioned across `comm` as in Init. */
      template <class ValueType> void Read(const std::string& fname, const Comm& comm = Comm::Self());


      /** VTU data for one (elem_idx) or all elements. */
      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F = Vector<Real>(), const Long elem_idx = -1) const;

      /**
       * Write VTU data to file.
       * @param[in] F nodal data, AoS {Ux1,Uy1,Uz1,...}.
       */
      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      /** Copy the element-list, possibly at a different precision. */
      template <class ValueType> void Copy(QuadElemList<ValueType>& elem_lst) const;

      template<typename> friend class QuadElemList;

      // Grants unit tests access to the private helpers below; defined in test-quad-elem.cpp.
      template<typename> friend struct QuadElemTestAccess;

    private:

      // ============ Foot finding: the geometry both near schemes split at ============

      // Runtime order is dispatched to a compile-time `order` (switch {4..20}), which bounds every
      // inner loop. `digits` is a runtime value throughout: it only selects a cached rule.

      // Single-point position (target-centered by `origin` when non-null) and, when the
      // pointers are non-null, the tangents dXu/dXv. Allocation-free -- the Lagrange bases are
      // built on the stack.
      void EvalPoint(Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Long elem_idx, const Vector<Real>* origin) const;

      // Closest nodal-grid point to Xtrg (brute force); seeds GetClosestPoint. Returns the distance.
      Real GetClosestNode(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) const;

      // Closest point on the patch over (u,v) in [0,1]^2: GetClosestNode seed, then active-set
      // Gauss-Newton with a grid-search fallback. Returns the distance. This is the foot the near
      // scheme splits at. A coordinate pinned at a bound by an outward gradient is held FIXED and
      // the step solved in the free subspace only, so metric coupling cannot contaminate the
      // surviving component -- edge/corner feet converge instead of falling back.
      // n_iter/used_fallback (optional) report the iteration count and whether Newton stalled.
      Real GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) const;

      // ============================ Duffy and Hedgehog schemes ============================

      // One near leaf cell: accumulate its tensor-product quadrature (weights wu (x) wv) into the
      // channel-major accumulator acc_cm. Non-empty normal_trg contracts with the target normal.
      // src_nodal is the target-shifted nodal slab, so the kernel target is the origin.
      template <Integer order, class Kernel> static void IntegrateCell(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD,
                                                                        const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT,
                                                                        const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());

      // Duffy edge-collapsed self scheme (QuadScheme::Duffy). The panel is split
      // at the target (u0,v0) into four triangles, each parametrised P(s,t) = (u0,v0) + s*c(t) with
      // |det| = s*|a x b|. The s factor cancels the 1/r singularity, so s takes a plain GL rule and t a
      // sinh-substituted rule concentrated at the foot of the perpendicular. Only the t-rule depends on
      // the metric and tolerance; everything below is fixed by (order, ti, tj, tri). `digits` is runtime.
      template <Integer order, class Kernel> static void SelfInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      // Split-at-foot near: the element is split at the foot so every refinement grades toward an
      // ENDPOINT, making the graded intervals depend only on the level. See detail_duffy.
      template <Integer order, class Kernel> static void NearInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());

      // ============================ Adaptive scheme ============================

      // Cell size above which the u sweep is blocked to keep the intermediates cache-resident.
      static constexpr Long MaxUnblockedPts = 16384;


      // TODO: measure this against the plain form above. It carries nine optional arguments, an
      // internal target-shift, and a u-blocked sweep, none of which the plain form has. Establish
      // whether any of that earns its complexity -- in particular whether the u-blocking is worth
      // keeping at all (it only runs above MaxUnblockedPts points, which only the Adaptive grids
      // reach) -- and if not, drop this overload and move the callers to the plain form.
      template <Integer order, class Kernel> static void IntegratePanel(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx,
                                                                        const Vector<Real>& Xtrg, const Vector<Real>& normal_trg,
                                                                        const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr,
                                                                        const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr,
                                                                        const Vector<Real>* src_nodal = nullptr, const Matrix<Real>* MuD_pre = nullptr, const Real nrm_sign = 1,
                                                                        Vector<Real>* acc_cm = nullptr);


      // ---- Adaptive self ----


      // Per-target singular self-interaction block at (u0,v0): graded u-refinement + 1D log rule in v.
      template <Integer order, class Kernel> static void SelfInteracBlockAdaptive(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);


      // ---- Adaptive near ----

      // Foot (u*,v*), off-surface distance, and depth cap from GetClosestPoint (the FOOT, not the
      // nearest node). h_param (optional): off-surface distance in parameter units.
      static Integer FootAndDepth(Real& ustar, Real& vstar, Real& dist, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse, Real* h_param = nullptr);

      // Foot-graded tensor rule over the whole panel (u_param x v_param, weights wu (x) wv).
      static Integer BuildNearTensorRule(Vector<Real>& u_param, Vector<Real>& wu, Vector<Real>& v_param, Vector<Real>& wv,
                                         Vector<Real>* useg, Vector<Long>* useg_depth, Vector<Real>* vseg, Vector<Long>* vseg_depth,
                                         const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg,
                                         const Real b_ellipse, const Vector<Real>& qnds, const Vector<Real>& qwts);

      template <Integer order, class Kernel> static void NearInteracBlockAdaptive(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      // ============================ Entry-point dispatch ============================

      template <Integer order, class Kernel> static void SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits);
      template <Integer order, class Kernel> static void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
      QuadScheme scheme_ = QuadScheme::Adaptive;
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
