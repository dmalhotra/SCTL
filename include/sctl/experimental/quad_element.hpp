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

      /**
       * Near/self singular-quadrature scheme: Adaptive (dyadic subdivision +
       * Alpert log correction, default), RectPolar (Bruno-2018 change of var),
       * Hybrid (Adaptive near + RectPolar self), or LineQBX (Lu 2019 sec.3.1
       * line-QBX near). For Hybrid the near phase is tolerance-driven (like
       * Adaptive) while the self phase uses the RectPolar COV knobs (`q`,
       * `cov_order`/Nbeta) passed to SetQuadScheme. LineQBX affects the near
       * phase only; its self phase falls back to the Adaptive scheme and its
       * near knobs are set via SetLineQBXParams. Duffy is Adaptive's split-at-foot
       * near paired with a Duffy edge-collapsed (sinh-substituted) self, ported from
       * upstream; opt in explicitly, the tolerance drives it like Adaptive.
       */
      enum class QuadScheme { Adaptive, RectPolar, Hybrid, LineQBX, Duffy };

      /** Constructor. */
      QuadElemList() {}

      /**
       * Construct from nodal coordinates.
       * @param[in] order polynomial order of each element.
       * @param[in] coord node coords, AoS {x1,y1,z1,...,xn,yn,zn}.
       * @param[in] comm communicator. When comm.Size() > 1, `coord` is assumed to
       * hold the full (globally-replicated) mesh and only this rank's contiguous
       * element slice is kept; with the default single-process comm the whole mesh
       * is used.
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
       */
      template <class ValueType> void Init(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      /** Destructor. */
      virtual ~QuadElemList() {}

      /** Number of elements. */
      Long Size() const override;

      /** Polynomial order of the elements. */
      Integer Order() const;


      /** Singular-quadrature scheme used by SelfInterac/NearInterac. */
      QuadScheme Scheme() const;

      /** True if the near phase uses the RectPolar COV (RectPolar scheme only). */
      bool NearUsesRectPolar() const { return scheme_ == QuadScheme::RectPolar; }

      /** True if the near phase uses the line-QBX scheme (LineQBX only). */
      bool NearUsesLineQBX() const { return scheme_ == QuadScheme::LineQBX; }

      /** True if the self phase uses the RectPolar COV (RectPolar or Hybrid). */
      bool SelfUsesRectPolar() const { return scheme_ == QuadScheme::RectPolar || scheme_ == QuadScheme::Hybrid; }

      /** True if the self phase uses the Duffy edge-collapsed scheme (Duffy only). */
      bool SelfUsesDuffy() const { return scheme_ == QuadScheme::Duffy; }

      /**
       * Set the singular-quadrature scheme.
       * @param[in] s scheme (Adaptive, RectPolar, or Hybrid).
       * @param[in] q derivative-flattening parameter for RectPolar (ignored for Adaptive).
       * @param[in] cov_order RectPolar GL points per direction (Nbeta, Bruno 2018);
       * decoupled from field order. 0 falls back to the tolerance-derived order.
       * @param[in] max_depth adaptive dyadic-refinement depth cap for the Adaptive scheme
       * (self + near) and the near phase of Hybrid; must be one of {4,8,12,30}. Ignored by
       * the RectPolar self/near phases.
       */
      void SetQuadScheme(QuadScheme s, Integer q = 6, Integer cov_order = 0, Integer max_depth = 30) {
        SCTL_ASSERT_MSG(max_depth == 4 || max_depth == 8 || max_depth == 12 || max_depth == 30, "Adaptive max_depth must be one of {4,8,12,30}.");
        scheme_ = s; cov_q_ = q; cov_order_ = cov_order; max_depth_ = max_depth;
      }

      /**
       * Set the line-QBX / "hedgehog" near-quadrature parameters (LineQBX scheme only; Lu 2019
       * sec.3.1). Check points are placed at heights R + i*r (i=0..p, units of the local patch size
       * L = sqrt(patch area)) along the patch normal through the target, the potential is evaluated
       * there with a 4^eta-subpaneled up_order GL rule (0 => 2*order), and extrapolated (degree-p
       * polynomial) to the target.
       *
       * NOTE: accurate only when the target is FAR FROM PANEL SEAMS (edges/corners); near a seam the
       * per-pair check-point line cannot resolve the adjacent panel's edge singularity (~5e-3 floor).
       *
       * Defaults R=r=0.02L, p=16, eta=2, up=72 target deep-near ~1e-10 for panel-interior targets
       * (verified vs a RectPolar gold at d=1e-4). Accuracy is a U-curve in R and p; up_order/eta only
       * need to resolve the check points at the chosen R. Paper's cheap ~1e-2 setting: R=r=0.15L, p=8,
       * eta=1, up=0.
       */
      void SetLineQBXParams(Real R_h = 0.02, Real r_h = 0.02, Integer p = 16, Integer up_order = 72, Integer eta = 2) {
        qbx_R_ = R_h; qbx_r_ = r_h; qbx_p_ = p; qbx_up_order_ = up_order; qbx_eta_ = eta;
      }

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
       * Reference-space Gauss-Legendre nodes in [0,1] for a given order.
       * @param[in] Order polynomial order of the element.
       */
      static const Vector<Real>& ParamNodes(const Integer Order);

      /**
       * Equidistant tensor grid of Nelem_perside panels of GL nodes in [0,1] (z left zero).
       * @param[in] Order polynomial order of the element.
       * @param[in] Nelem_perside panels per direction, split equally.
       */
      static const Vector<Real>& ParamGrid(const Integer Order, const Integer Nelem_perside);

      /**
       * Write elements to file.
       * @param[in] fname filename.
       * @param[in] comm communicator.
       */
      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      /**
       * Read elements from file.
       * @param[in] fname filename.
       * @param[in] comm communicator.
       */
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

      /**
       * Closest discretization NODE on elem_idx to Xtrg (brute-force over the nodal
       * grid; see GetClosestPoint for the true closest patch point).
       * @param[out] ustar,vstar parameters of the closest node in [0,1].
       * @param[in] elem_idx element index.
       * @param[in] Xtrg target coordinates (COORD_DIM reals).
       * @return distance from target to the closest node.
       */
      Real GetClosestNode(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) const;

      /**
       * Closest POINT on patch elem_idx to Xtrg over (u,v) in [0,1]^2 (GetClosestNode
       * seed, then Gauss-Newton with grid-search fallback).
       * @param[out] ustar,vstar parameters of the closest point in [0,1].
       * @param[in] elem_idx element index.
       * @param[in] Xtrg target coordinates (COORD_DIM reals).
       * @param[out] n_iter (optional) number of Gauss-Newton iterations executed.
       * @param[out] used_fallback (optional) true if Newton stalled and the grid-search fallback ran.
       * @return distance from target to the closest point.
       */
      Real GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) const;

      // Active-set variant of GetClosestPoint (upstream): a coordinate pinned at a bound by an
      // outward gradient is held FIXED and the reduced Gauss-Newton step is solved in the free
      // subspace only, so the metric coupling F cannot contaminate the surviving component with
      // the constrained gradient. Used only by the Duffy near path; the shared GetClosestPoint
      // (and hence the Adaptive/Hybrid near) is left unchanged.
      Real GetClosestPointAS(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) const;

      /** VTU data for one (elem_idx) or all elements. */
      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F = Vector<Real>(), const Long elem_idx = -1) const;

      /**
       * Write VTU data to file.
       * @param[in] fname filename.
       * @param[in] F nodal data, AoS {Ux1,Uy1,Uz1,...}.
       * @param[in] comm communicator.
       */
      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      /**
       * Visualize the adaptive near-interaction quadtree (off-surface target):
       * writes `<fname>` (per-leaf GL nodes + VTK_QUAD leaf outlines, colored by
       * depth) and `<fname>-target`.
       * @param[in] fname output filename prefix.
       * @param[in] elem_idx source element index.
       * @param[in] Xtrg off-surface target coords (COORD_DIM reals).
       * @param[in] tol accuracy tolerance (match the BIO's SetAccuracy).
       * @param[in] comm communicator.
       */
      void WriteNearInteracVTK(const std::string& fname, const Long elem_idx, const Vector<Real>& Xtrg, const Real tol, const Comm& comm = Comm::Self()) const;

      /**
       * Visualize the on-surface self-interaction structure at (u0,v0) (graded u x
       * Alpert v): writes `<fname>` (quadrature node cloud) and `<fname>-singpt`.
       * @param[in] fname output filename prefix.
       * @param[in] elem_idx source element index.
       * @param[in] u0,v0 on-surface target parameters in [0,1].
       * @param[in] tol accuracy tolerance (match the BIO's SetAccuracy).
       * @param[in] comm communicator.
       */
      void WriteSelfInteracVTK(const std::string& fname, const Long elem_idx, const Real u0, const Real v0, const Real tol, const Comm& comm = Comm::Self()) const;

      /**
       * Visualize the rectangular-polar (Scheme 2) grid for an off-surface target:
       * writes `<fname>` (warped Nbeta x Nbeta VTK_QUAD mesh) and `<fname>-target`.
       * @param[in] fname output filename prefix.
       * @param[in] elem_idx source element index.
       * @param[in] Xtrg off-surface target coords (COORD_DIM reals).
       * @param[in] Nbeta nodes per direction to draw (keep modest, e.g. 30-60).
       * @param[in] comm communicator.
       */
      void WriteNearInteracRPVTK(const std::string& fname, const Long elem_idx, const Vector<Real>& Xtrg, const Integer Nbeta = 48, const Comm& comm = Comm::Self()) const;

      /**
       * Visualize the rectangular-polar (Scheme 2) grid for an on-surface target at
       * (u0,v0): writes `<fname>` (warped Nbeta x Nbeta VTK_QUAD mesh) and `<fname>-singpt`.
       * @param[in] fname output filename prefix.
       * @param[in] elem_idx source element index.
       * @param[in] u0,v0 on-surface target parameters in [0,1].
       * @param[in] Nbeta nodes per direction to draw (keep modest, e.g. 30-60).
       * @param[in] comm communicator.
       */
      void WriteSelfInteracRPVTK(const std::string& fname, const Long elem_idx, const Real u0, const Real v0, const Integer Nbeta = 48, const Comm& comm = Comm::Self()) const;

      /**
       * Copy the element-list, possibly at a different precision.
       * @param[in] elem_lst input element-list.
       */
      template <class ValueType> void Copy(QuadElemList<ValueType>& elem_lst) const;

      template<typename> friend class QuadElemList;

      // Grants unit tests access to the private helpers below; defined in unit-test-quad-element.cpp.
      template<typename> friend struct QuadElemTestAccess;

    private:

      // Contiguous element range [i0,i1) owned by this rank when a global mesh of
      // Nelem_total elements is linearly partitioned across comm. Shared by Init
      // (in-memory construction) and Read (file load). With a single-process comm
      // this returns the full range [0, Nelem_total).
      static void PartitionRange(Long Nelem_total, const Comm& comm, Long& i0, Long& i1);

      template <class ValueType> static void EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv);
      
      void BuildDerivativeCache();

      // Nodal d/du, d/dv of a component-major SoA coord slab (order x order grid).
      // Shared by BuildDerivativeCache (absolute) and GetGeom (target-shifted).
      static void NodalDerivs(const Vector<Real>& coord_slab, const Integer order, Vector<Real>& du_slab, Vector<Real>& dv_slab);

      // Allocation-free single-point geometry evaluator: writes position X[COORD_DIM]
      // (target-centered by `origin` when non-null) and, when the pointers are non-null,
      // the tangents dXu/dXv[COORD_DIM] at parameter (u,v) on elem_idx. Builds the
      // order-length Lagrange bases on the stack and contracts against the cached nodal
      // coords -- no Matrix alloc / Transpose, unlike GetGeom. Used by the closest-point
      // search where it is called many times per target.
      void EvalPoint(Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Long elem_idx, const Vector<Real>* origin) const;

      // Cached 1D nodal differentiation matrix D (order x order) on the GL nodes,
      // D[i][a] = L_i'(node_a); D . LuV turns a value-interp operator into a deriv one.
      static const Matrix<Real>& DiffMat(const Integer order);
      template <Integer order> static const Matrix<Real>& DiffMat() { return DiffMat(order); }

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
      // Offset-valued counterpart of LogSingularQuad1D (which is already outward-graded).
      static void LogSingularQuad1DCentered(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder);
      template <Integer order, Integer digits> static const NodeRuleData& CenteredURule(const Integer ti, const Integer levels);
      template <Integer order, Integer digits> static const NodeRuleData& CenteredVRule(const Integer tj);

      // GL rule (nodes, weights) on [0,1] for compile-time count Nbeta (RP uses Nbeta>>50,
      // beyond LegQuadRule's cache); function-local static, runtime value via dispatch over {128,256,512}.
      template <Integer Nbeta> static const std::pair<Vector<Real>, Vector<Real>>& GLRuleNbeta();
      static const std::pair<Vector<Real>, Vector<Real>>& GLRuleNbetaDispatch(const Integer Nbeta);

      // Preloaded self-RP change-of-variable rule for on-surface node k (singularity at nds[k]),
      // serving both u (k=ti) and v (k=tj). Build-once static; dispatch over q in {6,10}, Nbeta in {128,256,512}.
      template <Integer order, Integer Nbeta, Integer q> static const NodeRuleData& RPSelfRule(const Integer k);
      template <Integer order> static const NodeRuleData& RPSelfRuleDispatch(const Integer k, const Integer q, const Integer Nbeta);

      // Bernstein-ellipse parameter + per-panel GL order from tolerance (shared by adaptive schemes).
      static void QuadParams(const Real tol, Real& b_ellipse, Integer& QuadOrder);

      // Compile-time per-panel GL order / Bernstein parameter for `digits` (QuadParams at 10^-digits);
      // near/self map runtime tolerance to compile-time `digits` (CSBQ-style).
      template <Integer digits> static Integer DigitsQuadOrder();
      template <Integer digits> static Real DigitsBEllipse();

      // Number of geometric grading levels (per side) toward v0 in the composite Alpert v-rule,
      // as a function of requested accuracy. Runtime core + compile-time `digits` wrapper.
      static Integer VLevelsForDigits(const Integer digits);
      template <Integer digits> static Integer DigitsVLevels();

      // Default RectPolar Nbeta (GL points per direction) for `digits`, used when cov_order_==0.
      // Worst-case-calibrated ladder (theta=pi twist sphere, Nbeta_sweep.txt); returns a value
      // in {128,256,384,512} (the GLRuleNbetaDispatch/RPSelfRuleDispatch ladders).
      static Integer NbetaForDigits(const Integer digits);

      // Accumulate a tensor-product quadrature (u_param x v_param, weights wu (x) wv) on
      // elem_idx against target Xtrg into M_acc; normal_trg != null enables target-normal contraction.
      // Mv_pre/dMv_pre, Mu_pre/dMu_pre (optional): precomputed v/u interp operators (order x N) used in
      // place of building from param (self supplies Alpert v; self-RP supplies both; near/Adaptive leave null).
      // src_nodal (optional): caller-supplied target-shifted nodal slab (COORD_DIM*order x order,
      // component-major) for this sub-element, bypassing the internal coord_shift build (near split
      // supplies it). MuD_pre (optional): stacked [T^T; dT^T] (2q x order) so value+derivative come
      // from one GEMM. nrm_sign: flips the source normal (mirrored sub-elements). acc_cm (optional):
      // channel-major accumulator (C*nnode) added into via beta=1, so the caller transposes to
      // node-major once per target instead of per cell.
      template <Integer order, class Kernel> static void IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx,
                                                                        const Vector<Real>& Xtrg, const Vector<Real>& normal_trg,
                                                                        const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr,
                                                                        const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr,
                                                                        const Vector<Real>* src_nodal = nullptr, const Matrix<Real>* MuD_pre = nullptr, const Real nrm_sign = 1,
                                                                        Vector<Real>* acc_cm = nullptr);

      // Accuracy/order-templated impls of NearInterac/SelfInterac: entry points dispatch runtime
      // order to compile-time `order` (switch {4..48}) and tolerance to `digits` (if-else), CSBQ-style.
      template <Integer order, class Kernel> static void SelfInteracDispatchDigits(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);
      template <Integer order, class Kernel> static void NearInteracDispatchDigits(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);
      template <Integer digits, Integer order, class Kernel> static void SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self);
      template <Integer digits, Integer order, class Kernel> static void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self);

      // Near-only knobs, independent of the self path (SCTL_QUAD_ORDER drives both). For tuning
      // the near heuristic while self is held at a much tighter tolerance.
      //   SCTL_NEAR_QORDER   per-cell GL order        (default from NearRhoRule)
      //   SCTL_NEAR_BELLIPSE admissibility constant   (default from NearRhoRule)
      // VALIDITY: calibrated and validated on the twisted unit sphere for twist <= pi/3
      // (element anisotropy <= ~4.2). Beyond that the near rule needs a higher GL order than
      // this gives; do not rely on it past pi/3 without re-checking accuracy.
      // Tolerance-dependent rho + the end-foot Bernstein reach the split-at-(u0,v0) geometry
      // needs. QuadParams (still used by self) pins rho = 2.5 and the semi-major reach, which
      // over-refines near by a^2/b^2 ~ 1.9x.
      static void NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder);
      template <Integer digits> static Integer NearQuadOrder();
      template <Integer digits> static Real NearBEllipse();
      //   SCTL_NEAR_MAXLVL   near-only level cap (0 => use max_depth_). Near-touching targets
      //   (a neighbouring patch's node, foot distance ~0) refine to the cap regardless of the
      //   admissibility constant, so the cap -- not b_ellipse -- controls their error.
      static Integer NearMaxLvlOverride();
      // One graded interval, in NORMALIZED sub-element coordinates. dT/TT/TD are precomputed
      // here (not per target) because the split-at-foot scheme feeds sub-element NODAL coords
      // into the cell quadrature, so these operators no longer depend on (u*,v*).
      //   T  (order x q)   sub-element nodes -> this interval's GL nodes
      //   dT (order x q)   d/dx of the above, x = the sub-element's normalized coordinate
      //   TT (q x order)   T^T, for the projection
      //   TD (2q x order)  [T^T ; dT^T] stacked, so value+derivative come from ONE GEMM
      struct GradeRule { Vector<Real> nds, w; Matrix<Real> T, dT, TT, TD; Real a, b; };
      // Flat index: shell_k -> k, core_k -> MaxNearLvl + k.
      static constexpr Integer MaxNearLvl = 31;
      template <Integer order, Integer digits> static const Vector<GradeRule>& NearGradeTable();
      template <Integer digits, Integer order, class Kernel> static void NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // ---- Upstream-ported near path (QuadScheme::Duffy only) ----
      // Same split-at-foot geometry as NearInteracBlockSplit, but with the upstream additions that
      // let the near rule hold accuracy under strong parametric shear: (1) a corner-angle bump to
      // the per-target GL order (the acute tangent angle at the foot sets how much the element
      // wraps the target, which the parameter-space admissibility test cannot see), and (2) a
      // deeper refinement ladder (MaxNearLvlCM = mantissa width, vs the 31 above). `digits` is
      // runtime here, so the grade table is keyed on the runtime GL order q.
      static constexpr Integer MaxNearLvlCM = GetSigBits<Real>::value();  // shell_k -> k, core_k -> MaxNearLvlCM + k
      static constexpr Integer NearMaxQuadOrderCM = 60;
      static constexpr Integer MaxDigitsCM = 1 + GetSigBits<Real>::value()*30103/100000;
      static Integer NearQuadOrderRt(const Integer digits);   // runtime counterpart of NearQuadOrder<digits>
      static Real NearBEllipseRt(const Integer digits);       // runtime counterpart of NearBEllipse<digits>
      template <Integer order> static const Vector<GradeRule>& NearGradeTableQ(const Integer q);
      template <Integer order, class Kernel> static void IntegrateNearCM(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker,
                                                                         const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD,
                                                                         const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT,
                                                                         const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm);
      template <Integer order, class Kernel> static void NearInteracBlockSplitDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      // Per-target singular self-interaction block at (u0,v0): graded u-refinement + 1D log rule in v.
      template <Integer digits, Integer order, class Kernel> static void SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // Duffy edge-collapsed self scheme (QuadScheme::Duffy; ported from upstream). The panel is split
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

      // Rectangular-polar (Bruno-2018) change-of-variable 1D rule on [0,1]: maps {gl_nds,gl_wts}
      // via eta_alpha to cluster toward the singularity (alpha=2*sing-1) with vanishing weight;
      // `q` flattens derivatives up to order q-1.
      static void RectPolarNodes1D(Vector<Real>& nodes, Vector<Real>& wts, const Real alpha, const Integer q, const Vector<Real>& gl_nds, const Vector<Real>& gl_wts);

      // Shared core for WriteNear/SelfInteracRPVTK: warped Nbeta x Nbeta CoV grid clustered toward (ustar,vstar).
      void WriteRectPolarGridVTK(const std::string& fname, const Long elem_idx, const Real ustar, const Real vstar, const Integer Nbeta) const;

      // RP counterparts of NearInteracBlock/SelfInteracBlock; quadrature size is cov_order_ if set,
      // else the tol-derived `nbeta_default` the caller passes (NbetaForDigits(digits)).
      template <Integer order, class Kernel> static void NearInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer nbeta_default);
      template <Integer order, class Kernel> static void SelfInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer nbeta_default);

      // Line-QBX / hedgehog near-interaction block (Lu 2019 sec.3.1); knobs from SetLineQBXParams.
      // Accurate only for panel-interior targets (see SetLineQBXParams seam caveat).
      template <Integer order, class Kernel> static void NearInteracBlockQBX(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // Batched Line-QBX / hedgehog near-interaction for ALL near targets of one element (the
      // production path; NearInteracBlockQBX above is the equivalent per-target reference). The
      // upsampled source-patch geometry and the sub-panel interp rules are target-independent, so
      // they are precomputed ONCE per element and reused across every target and check point; only
      // the kernel evaluation and the (accumulate-then-project) contraction remain per check point.
      // Writes the full nnode*KDIM0 x Ntrg*KDIM1_out operator M (same layout as NearInteracHelper).
      template <Integer order, class Kernel> static void NearInteracBatchedQBX(Matrix<Real>& M, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker);

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
      QuadScheme scheme_ = QuadScheme::Adaptive;
      Integer cov_q_ = 6;
      Integer cov_order_ = 0;
      Integer max_depth_ = 30;
      // Line-QBX near knobs (LineQBX scheme only); see SetLineQBXParams. Defaults target deep-near
      // accuracy ~1e-10 (verified vs a RectPolar gold at d=1e-4). L = sqrt(patch area).
      Real qbx_R_ = 0.02;         // first check-point distance in units of patch size L
      Real qbx_r_ = 0.02;         // check-point spacing in units of L
      Integer qbx_p_ = 16;        // extrapolant degree (p+1 check points)
      Integer qbx_up_order_ = 72; // per-subpatch smooth-rule GL order; 0 => 2*order
      Integer qbx_eta_ = 2;       // sub-paneling level: 4^eta = (2^eta)^2 subpatches
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
