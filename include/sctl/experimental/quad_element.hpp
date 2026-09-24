#ifndef _SCTL_QUAD_ELEMENT_HPP_
#define _SCTL_QUAD_ELEMENT_HPP_

#include <string>
#include <utility>
#include <vector>
#include <sctl.hpp>

namespace sctl {

  class VTUData;
  template <class ValueType> class Matrix;

  // Kernels declare SingularOrder in kernel_functions.hpp. Anything that does not gets 2, the
  // conservative side: tighter parameters, so more cost and never less accuracy.
  template <class Kernel, class = void> struct KernelSingularOrder {
    static constexpr Integer value = 2;
  };
  template <class Kernel> struct KernelSingularOrder<Kernel, std::void_t<decltype(Kernel::SingularOrder())>> {
    static constexpr Integer value = Kernel::SingularOrder();
  };

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

      /**
       * Near/self singular-quadrature scheme: Adaptive (foot-graded separable-tensor near +
       * centered graded/Alpert self, default), or Duffy (Adaptive's split-at-foot near paired
       * with a Duffy edge-collapsed, sinh-substituted self). The tolerance drives both.
       */
      enum class QuadScheme { Adaptive, Duffy };

      /**
       * Set the singular-quadrature scheme.
       * @param[in] s scheme (see QuadScheme; default Adaptive).
       * @param[in] max_depth dyadic-refinement depth cap; must be one of {4,8,12,30}.
       */
      void SetQuadScheme(QuadScheme s, Integer max_depth = 30) {
        SCTL_ASSERT_MSG(max_depth == 4 || max_depth == 8 || max_depth == 12 || max_depth == 30, "Adaptive max_depth must be one of {4,8,12,30}.");
        scheme_ = s; max_depth_ = max_depth;
      }

      /// Build-time switch for the on-surface hedgehog (line-QBX) self scheme. With it on,
      /// SelfInterac evaluates each on-surface target by placing a short line of proxy points
      /// along the outward normal, integrating there with the ordinary near scheme, and
      /// extrapolating back to the surface; off, the Duffy edge-collapsed scheme is used.
      /// NearInterac is unaffected either way.
      ///
      /// CAVEAT: extrapolating from off-surface proxies gives the ONE-SIDED limit, not the
      /// principal value the Duffy path returns. The two differ by half the jump, so a caller
      /// mixing the conventions must correct for it.
      static constexpr bool UseHedgehogSelf = false;

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

      // Grants unit tests access to the private helpers below; defined in test-quad-elem.cpp.
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

      // Duffy edge-collapsed self scheme (QuadScheme::Duffy). The panel is split
      // at the target (u0,v0) into four triangles, each parametrised P(s,t) = (u0,v0) + s*c(t) with
      // |det| = s*|a x b|. The s factor cancels the 1/r singularity, so s takes a plain GL rule and t a
      // sinh-substituted rule concentrated at the foot of the perpendicular. Only the t-rule depends on
      // the metric and tolerance; everything below is fixed by (order, ti, tj, tri). `digits` is runtime.
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

      // Closest point on the patch over (u,v) in [0,1]^2: GetClosestNode seed, then active-set
      // Gauss-Newton with a grid-search fallback. Returns the distance. This is the foot the near
      // scheme splits at. A coordinate pinned at a bound by an outward gradient is held FIXED and
      // the step solved in the free subspace only, so metric coupling cannot contaminate the
      // surviving component -- edge/corner feet converge instead of falling back.
      // n_iter/used_fallback (optional) report the iteration count and whether Newton stalled.
      Real GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) const;

      // One near leaf cell: accumulate its tensor-product quadrature (weights wu (x) wv) into the
      // channel-major accumulator acc_cm. Non-empty normal_trg contracts with the target normal.
      // src_nodal is the target-shifted nodal slab, so the kernel target is the origin.
      template <Integer order, class Kernel> static void IntegrateBlock(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD,
                                                                        const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT,
                                                                        const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>());

      // Accumulate a tensor-product quadrature (weights wu (x) wv) on elem_idx against Xtrg into
      // M_acc. Non-empty normal_trg contracts with the target normal. Every _pre operator is
      // optional and replaces the build from param: Mv/dMv and Mu/dMu are the v- and u-interps,
      // MuD is [T^T; dT^T] stacked so value and derivative come from one GEMM. src_nodal is a
      // caller-supplied target-shifted nodal slab, so the kernel target is the origin. nrm_sign
      // flips the source normal for mirrored sub-elements. acc_cm is a channel-major accumulator
      // added into with beta=1, so the caller transposes to node-major once per target, not per cell.
      // Cell size above which the u sweep is blocked to keep the intermediates cache-resident.
      // Only the Adaptive grids get near it; the Duffy paths never reach this routine.
      static constexpr Long UBlkPts = 16384;


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
      // endpoint. QuadParams, which the Adaptive self path still uses, pins rho = 2.5 and the
      // semi-major reach instead.
      // VALIDITY: calibrated and validated on the twisted unit sphere for twist <= pi/3
      // (element anisotropy <= ~4.2). Beyond that the near rule needs a higher GL order than
      // this gives; do not rely on it past pi/3 without re-checking accuracy.
      static void NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder);
      static Integer NearQuadOrder(const Integer digits);
      static Real NearBEllipse(const Integer digits);

      // ---- on-surface hedgehog (line-QBX) ----
      //
      // Proxy offsets along the normal, normalized so the innermost is 1; each node scales by its
      // own rmin. sum|w| below depends only on this shape, not on rmin, and floors the accuracy.
      static const Vector<Real>& HedgehogProxyOffsets();
      // Extrapolation weights to zero offset; returns sum|w|.
      static Real HedgehogWeights(Vector<Real>& w);
      // Innermost proxy offset as a multiple of the node's distance to the element edge, and the
      // digits to ask at the proxies. Both tuned per singularity order against Duffy on the sphere.
      static Real HedgehogRminCoeff(const Integer digits, const Integer sing_order);
      static Integer HedgehogNearDigits(const Integer digits, const Integer sing_order);

      // One graded interval, in NORMALIZED sub-element coordinates. dT/TT/TD are precomputed
      // here (not per target) because the split-at-foot scheme feeds sub-element NODAL coords
      // into the cell quadrature, so these operators no longer depend on (u*,v*).
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

      // ============================ Adaptive scheme only ============================

      // TODO: measure this against the plain form above. It carries nine optional arguments, an
      // internal target-shift, and a u-blocked sweep, none of which the plain form has. Establish
      // whether any of that earns its complexity -- in particular whether the u-blocking is worth
      // keeping at all (it only runs above SCTL_UBLK_PTS points, which only the Adaptive grids
      // reach) -- and if not, drop this overload and move the callers to the plain form.
      template <Integer order, class Kernel> static void IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx,
                                                                        const Vector<Real>& Xtrg, const Vector<Real>& normal_trg,
                                                                        const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr,
                                                                        const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr,
                                                                        const Vector<Real>* src_nodal = nullptr, const Matrix<Real>* MuD_pre = nullptr, const Real nrm_sign = 1,
                                                                        Vector<Real>* acc_cm = nullptr);



      // Bernstein-ellipse parameter + per-panel GL order from tolerance (shared by adaptive schemes).
      static void QuadParams(const Real tol, Real& b_ellipse, Integer& QuadOrder);

      // Per-panel GL order and Bernstein parameter for `digits`, i.e. QuadParams at 10^-digits,
      // tabulated once over MaxDigits.
      static Integer DigitsQuadOrder(const Integer digits);
      static Real DigitsBEllipse(const Integer digits);

      // Per-panel GL rule (nodes,weights) for `digits`, built once (ComputeNdsWts is an uncached
      // O(N^2) Newton solve). Consumed by the foot-graded tensor near (NearInteracBlockGraded).
      static const std::pair<Vector<Real>, Vector<Real>>& DigitsGLRule(const Integer digits);

      // Number of geometric grading levels (per side) toward v0 in the composite Alpert v-rule,
      // as a function of requested accuracy.
      static Integer VLevelsForDigits(const Integer digits);




      // ============================ SelfInterac only ============================

      // 1D value + derivative interpolation from order GL nodes to `param`:
      // M[i][a] = L_i(param[a]) (order x N), dM = DiffMat<order> . M.
      template <Integer order> static void BuildInterp1D(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& param);

      // 1D quadrature rule (param, w) + value/derivative interp operators (M, dM = order x N).
      struct NodeRuleData { Vector<Real> param, w; Matrix<Real> M, dM, MT, dMT; };

      // L_i(u0+d) with the vanishing factor formed as `d` itself, never as a subtraction of
      // absolute coordinates. dM = DiffMat . M as usual.
      template <Integer order> static void LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& delta, const Integer ti);

      // Geometric panels marching outward from u0 to each end; `levels`+1 panels per side.
      static void BuildCenteredGraded1D(Vector<Real>& delta, Vector<Real>& w, const Real u0, const Integer levels, const Vector<Real>& qnds, const Vector<Real>& qwts);

      // Outward-graded log-singular 1D rule, emitted as offsets `delta` from v0 (singular node at
      // offset exactly 0) so the innermost panels keep full relative precision.
      static void LogSingularQuad1DCentered(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder);

      template <Integer order> static const NodeRuleData& CenteredURule(const Integer ti, const Integer levels, const Integer digits);
      template <Integer order> static const NodeRuleData& CenteredVRule(const Integer tj, const Integer digits);

      // Per-target singular self-interaction block at (u0,v0): graded u-refinement + 1D log rule in v.
      template <Integer order, class Kernel> static void SelfInteracBlockAdaptive(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      // ---- Foot-graded separable-tensor near (QuadScheme::Adaptive) ----
      // THE production near path for the Adaptive scheme. Grade [0,1] toward u* and
      // toward v* INDEPENDENTLY -- split each side AT the foot (u*,v*), grade geometrically outward
      // (BuildFootGraded1DSegments) -- then take the FULL TENSOR PRODUCT and integrate the whole
      // panel with ONE IntegrateBlock: no quadtree, no per-leaf loop, no interval dedup. Holds its
      // accuracy across parametric shear, where the isotropic-quadtree split-at-foot rule it
      // replaced lost ~2-3 orders on smooth geometry. Requires dist > 0 (foot on a cell
      // boundary), so it must never serve the self path.
      // One GL rule per segment, concatenated in segment order (contiguous runs -> contiguous slices).
      static void ExpandSegments(Vector<Real>& param, Vector<Real>& w, const Vector<Real>& seg, const Vector<Real>& qnds, const Vector<Real>& qwts);

      // Split [0,1] at `center`, grade geometrically outward on each side; innermost segment touches
      // the foot (admissible only under the off-surface effective distance -> near targets only).
      static void BuildFootGraded1DSegments(Vector<Real>& seg, Vector<Long>& seg_depth, const Real center, const Real b_ellipse, const Real w_min);

      // Foot (u*,v*), off-surface distance, and depth cap from GetClosestPoint (the FOOT, not the
      // nearest node). h_param (optional): off-surface distance in parameter units.
      static Integer NearFootAndDepth(Real& ustar, Real& vstar, Real& dist, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse, const Integer max_depth, Real* h_param = nullptr);

      // Foot-graded tensor rule over the whole panel (u_param x v_param, weights wu (x) wv).
      static Integer BuildNearTensorRule(Vector<Real>& u_param, Vector<Real>& wu, Vector<Real>& v_param, Vector<Real>& wv,
                                         Vector<Real>* useg, Vector<Long>* useg_depth, Vector<Real>* vseg, Vector<Long>* vseg_depth,
                                         const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg,
                                         const Real b_ellipse, const Vector<Real>& qnds, const Vector<Real>& qwts, const Integer max_depth);
      template <Integer order, class Kernel> static void NearInteracBlockGraded(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
      QuadScheme scheme_ = QuadScheme::Adaptive;
      Integer max_depth_ = 30;
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
