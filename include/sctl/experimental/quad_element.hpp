#ifndef _SCTL_QUAD_ELEMENT_HPP_
#define _SCTL_QUAD_ELEMENT_HPP_

#include <string>
#include <utility>
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
       * @param[in] comm communicator. When comm.Size() > 1, `coord` is assumed to
       * hold the full (globally-replicated) mesh and only this rank's contiguous
       * element slice is kept; with the default single-process comm the whole mesh
       * is used.
       *
       * TODO: Fix this: in a distributed code, a global array should never all be on a single process.
       */
      template <class ValueType> QuadElemList(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      /**
       * Initialize from nodal coordinates.
       * @param[in] order polynomial order of each element.
       * @param[in] coord node coords, AoS {x1,y1,z1,...,xn,yn,zn}.
       * @param[in] comm communicator. When comm.Size() > 1, `coord` is assumed to
       * hold the full (globally-replicated) mesh and only this rank's contiguous
       * element slice is kept; with the default single-process comm the whole mesh
       * is used.
       *
       * TODO: Fix this: in a distributed code, a global array should never all be on a single process.
       */
      template <class ValueType> void Init(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      virtual ~QuadElemList() {}

      /** Number of elements. */
      Long Size() const override;

      /** Polynomial order of the elements. */
      Integer Order() const;

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

      /// Hedgehog / line-QBX near interaction.  Xt_proxy holds p targets on one line (closest
      /// first); wts holds the 1D extrapolation weights to the actual target.  All p proxies
      /// share one foot, one subdivision and one geometry pass; the weights are applied at the
      /// kernel matrix, so only the kernel evaluation scales with p.  M is a single target block.
      template <class Kernel> static void NearInteracHedgehog(Matrix<Real>& M, const Vector<Real>& Xt_proxy, const Vector<Real>& wts, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /** Reference-space Gauss-Legendre nodes in [0,1]. */
      static const Vector<Real>& ParamNodes(const Integer Order);

      /** Write elements to file. */
      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      /** Read elements from file, partitioned across `comm` as in Init. */
      template <class ValueType> void Read(const std::string& fname, const Comm& comm = Comm::Self());

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

      // Grants unit tests access to the private helpers below; defined in unit-test-quad-element.cpp.
      template<typename> friend struct QuadElemTestAccess;

    private:

      // Contiguous element range [i0,i1) owned by this rank under a linear partition of
      // Nelem_total elements; the full range for a single-process comm. Used by Init and Read.
      static void PartitionRange(Long Nelem_total, const Comm& comm, Long& i0, Long& i1);

      // Tensor-product contraction of a component-major SoA slab; used by GetGeom and GetVTUData.
      template <class ValueType> static void EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv);

      // Nodal d/du, d/dv of a component-major SoA coord slab. Init builds the absolute
      // dcoord_du/dv cache with it; GetGeom uses it on target-shifted coords.
      static void NodalDerivs(const Vector<Real>& coord_slab, const Integer order, Vector<Real>& du_slab, Vector<Real>& dv_slab);


      // ============================ SelfInterac and NearInterac ============================

      // Cached 1D nodal differentiation matrix D (order x order) on the GL nodes,
      // D[i][a] = L_i'(node_a); D . LuV turns a value-interp operator into a deriv one.
      static const Matrix<Real>& DiffMat(const Integer order);
      template <Integer order> static const Matrix<Real>& DiffMat() { return DiffMat(order); }

      // Runtime order is dispatched to a compile-time `order` (switch {4..48}), which bounds
      // every inner loop. `digits` is a runtime value throughout: it only selects a cached rule.
      // Accuracy levels the type can express: digits10 ~ significand bits * log10(2)
      // (30103/100000). 7 for float, 16 for double, 19 for long double, 34 for __float128.
      static constexpr Integer MaxDigits = 1 + GetSigBits<Real>::value()*30103/100000;
      // Largest d with tol <= 10^-d, where 10^-d is repeated multiplication of 0.1, NOT the
      // literal 1e-d -- the two differ in the last bits and so pick different d at exact powers.
      static Integer DigitsFromTol(const Real tol);


      // ============================ SelfInterac only ============================

      // Duffy edge-collapsed self scheme: the panel is split at the target (u0,v0) into four
      // triangles, each parametrised as P(s,t) = (u0,v0) + s*c(t) with |det| = s*|a x b|. The
      // s factor cancels the 1/r singularity, so s takes a plain GL rule and t a rule graded
      // toward the foot of the perpendicular. Only the t-rule depends on the metric and the
      // tolerance; everything below is fixed by (order, ti, tj, tri).
      struct DuffyTri {
        bool swap_ab = false;    // collapsed (s-only) coordinate is u => local (alpha,beta) = (v,u)
        Real nsign = 1;          // restores the sign of dX/du x dX/dv
        Real J0 = 0;             // |a x b|
        Matrix<Real> WbC;        // (order x 2*ns) = [Wb | Wb'], collapsed direction at the s-nodes
        Matrix<Real> WbT;        // (ns x order), adjoint of the value half
        Vector<Matrix<Real>> MiC, MiT;       // ns entries: (order x 2*order) = [Mi | Mi'], and (order x order)
      };
      struct DuffySelfTable {
        Integer ns = 0;
        Vector<Real> sn, sw;
        std::vector<DuffyTri> tri;   // 4*order*order entries, indexed (ti*order + tj)*4 + tri
      };
      template <Integer order> static const DuffySelfTable& DuffyTable();
      static Integer DuffyTOrder(const Integer digits, const Integer order, const Integer kdim0);
      template <Integer order, class Kernel> static void SelfInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);
      template <Integer order, class Kernel> static void SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits);


      // ======================= NearInterac only =======================

      // Single-point position (target-centered by `origin` when non-null) and, when the
      // pointers are non-null, the tangents dXu/dXv. Allocation-free -- the Lagrange bases are
      // built on the stack. Called many times per target by GetClosestPoint.
      void EvalPoint(Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Long elem_idx, const Vector<Real>* origin) const;

      // Closest nodal-grid point to Xtrg (brute force); seeds GetClosestPoint. Returns the distance.
      Real GetClosestNode(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) const;

      // Closest point on the patch over (u,v) in [0,1]^2: GetClosestNode seed, then Gauss-Newton
      // with a grid-search fallback. Returns the distance. This is the foot the near scheme
      // splits at. n_iter/used_fallback (optional) report the iteration count and whether Newton
      // stalled.
      Real GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) const;

      // One near leaf cell: accumulate its tensor-product quadrature (weights wu (x) wv) into the
      // channel-major accumulator acc_cm. Non-empty normal_trg contracts with the target normal.
      // src_nodal is the target-shifted nodal slab, so the kernel target is the origin.
      template <Integer order, class Kernel> static void IntegrateBlock(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD,
                                                                        const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT,
                                                                        const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());
      // Split-at-foot near scheme. Splitting the element at the foot makes every refinement
      // grade toward an ENDPOINT, so in normalized sub-element coordinates the graded intervals
      // depend only on the level and their operators precompute once per `order`. Per side,
      // grading toward the foot at x=1:
      //   shell_k = [1-2^-k, 1-2^-(k+1)]    the half of core_k away from the foot
      //   core_k  = [1-2^-k, 1]             the half touching it
      // The sub-elements are anisotropic, so the corner cell is bisected along its longer
      // PHYSICAL dimension only (parameter extent x surface speed), one split at a time, until
      // that dimension is admissible against the target distance -- which keeps the aspect ratio
      // from propagating to every descendant. Each split emits one leaf, and the u- and v-levels
      // advance independently, so every interval stays shell_k / core_k at some level.

      // Per-cell GL order and admissibility constant. b_ellipse is the end-foot Bernstein reach,
      // weaker than the semi-major reach by a^2/b^2 ~ 1.9x because the foot lands on a cell
      // endpoint.
      static void NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder);
      static Integer NearQuadOrder(const Integer digits);
      static Real NearBEllipse(const Integer digits);

      // ---- on-surface hedgehog (line-QBX) ----
      //
      // With this on, SelfInterac evaluates each on-surface target by placing a short line of
      // proxy points along the normal, integrating there with the ordinary near scheme, and
      // extrapolating back to the surface. Off it, the Duffy edge-collapsed scheme is used.
      // NearInterac is unaffected either way -- the near case is still the split scheme.
      static constexpr bool UseHedgehogSelf = false;
      // Proxies sit at rmin*ratio^(j/(p-1)), j = 0..p-1, so the line spans one factor of `ratio`.
      // Extrapolating to the surface amplifies the quadrature error by sum|w|, which for a
      // geometric line depends only on (p, ratio) and NOT on rmin -- see HedgehogWeights. p=5 with
      // ratio 4 gives 61; going to p=6 costs 226 and p=4 gives 17 but extrapolates far worse.
      static constexpr Integer HedgehogNumProxy = 5;
      static constexpr Integer HedgehogRatio = 4;
      // Lagrange extrapolation weights to zero distance. Scale free: r enters only through the
      // ratios r_k/r_j, so one vector serves every node. Returns sum|w|, the error amplification.
      static Real HedgehogWeights(Vector<Real>& w);
      // Innermost proxy distance for a node whose distance to the element edge is `edge_dist`.
      // Truncation of the extrapolation falls like the sixth power of rmin for this family, so
      // rmin ~ tol^(1/6); the constant is calibrated on the flat panel against the exact single
      // layer at a tenth of the requested tolerance, and is within 5% across orders 8, 12 and 16.
      static Real HedgehogRmin(const Integer digits, const Real edge_dist);

      // One graded interval in normalized sub-element coordinates:
      //   T  (order x q)   sub-element nodes -> this interval's GL nodes
      //   dT (order x q)   d/dx of the above, x = the sub-element's normalized coordinate
      //   TT (q x order)   T^T, for the projection
      //   TD (2q x order)  [T^T ; dT^T] stacked, so value+derivative come from ONE GEMM
      // All four are built from the offset from the foot rather than from absolute position, so the
      // interpolated source-to-target vector is a sum of terms that each vanish with the offset
      // instead of a cancelling sum of panel-scale ones -- its error is then relative rather than
      // absolute, which is what a 1/r^k kernel amplifies.
      struct GradeRule { Vector<Real> nds, w; Matrix<Real> T, dT, TT, TD; Real a, b; };
      // Refinement bottoms out where 1-2^-k stops being distinct from 1, i.e. at the mantissa width.
      static constexpr Integer MaxNearLvl = GetSigBits<Real>::value();  // flat index: shell_k -> k, core_k -> MaxNearLvl + k
      static constexpr Integer NearMaxQuadOrder = 60;
      // Accuracy-independent: one static per `order`, holding every rung the corner-angle
      // correction can select (each multiple of 4, plus each NearQuadOrder(d)).
      template <Integer order> static const Vector<GradeRule>& NearGradeTable(const Integer q);
      template <Integer order, class Kernel> static void NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());
      template <Integer order, class Kernel> static void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());
      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
