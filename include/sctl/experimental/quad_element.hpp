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
      static Integer SelfQuadOrderProbe(const Real tol); // resolved self GL order (sweeps/reporting)
      static Integer SelfLevelsProbe(const Real tol);    // resolved self u-levels (sweeps/reporting)

      /**
       * Near/self singular-quadrature scheme: Adaptive (dyadic subdivision +
       * Alpert log correction, default), RectPolar (Bruno-2018 change of var),
       * Hybrid (Adaptive near + RectPolar self), or LineQBX (Lu 2019 sec.3.1
       * line-QBX near). For Hybrid the near phase is tolerance-driven (like
       * Adaptive) while the self phase uses the RectPolar COV knobs (`q`,
       * `cov_order`/Nbeta) passed to SetQuadScheme. LineQBX affects the near
       * phase only; its self phase falls back to the Adaptive scheme and its
       * near knobs are set via SetLineQBXParams.
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

      /** True if the self phase uses the Duffy edge-collapsed scheme. */
      bool SelfUsesDuffy() const { return scheme_ == QuadScheme::Duffy; }

      /**
       * Set the singular-quadrature scheme.
       * @param[in] s scheme (Adaptive, RectPolar, or Hybrid).
       * @param[in] q derivative-flattening parameter for RectPolar (ignored for Adaptive).
       * @param[in] cov_order RectPolar GL points per direction (Nbeta, Bruno 2018);
       * decoupled from field order. 0 falls back to the tolerance-derived order.
       * @param[in] max_depth number of dyadic grading levels toward the singularity for the
       * Adaptive scheme (self + near) and the near phase of Hybrid; any value in [1,40]. The
       * self rule uses it as an exact level count (2*(max_depth+1) panels); the near quadtree
       * still treats it as a depth cap. Ignored by the RectPolar self/near phases.
       */
      void SetQuadScheme(QuadScheme s, Integer q = 6, Integer cov_order = 0, Integer max_depth = 30) {
        // Centered rules take `max_depth` as an exact runtime level count; the legacy
        // templated path snaps to the nearest of {4,8,12,30}.
        SCTL_ASSERT_MSG(max_depth >= 1 && max_depth <= 40, "Adaptive max_depth must be in [1,40].");
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
      // Preloaded composite/graded Alpert log-singular v-rule for self-interaction at
      // v0=ParamNodes(order)[tj]; geometry-independent, cached once per (order, digits, tj).
      // Grading levels toward v0 grow with `digits` (DigitsVLevels), so v-accuracy tracks tol.
      template <Integer order, Integer digits> static const NodeRuleData& SelfVRule(const Integer tj);

      // Preloaded self-interaction graded u-rule for u0=ParamNodes(order)[ti]. The subdivision
      // is geometry-independent (scale-invariance), so fixed by (order, ti, digits, max_depth) and
      // cached once. Runtime max_depth maps to the compile-time template via SelfURuleDispatch.
      template <Integer order, Integer digits, Integer max_depth> static const NodeRuleData& SelfURule(const Integer ti);
      template <Integer order, Integer digits> static const NodeRuleData& SelfURuleDispatch(const Integer ti, const Integer max_depth);

      // --- Centered rules (SCTL_CENTERED_RULES=1) ------------------------------------
      // DEFAULT ON (set SCTL_CENTERED_RULES=0 for the legacy bisection rules).
      // Built OUTWARD from the singular node so it lands on a panel ENDPOINT (rather than
      // strictly inside the innermost leaf of a top-down bisection), and stored as OFFSETS
      // from it. Offsets keep full relative precision in the innermost panels: the absolute
      // form `a0 + len*qnds` rounds to ulp(a0)~1e-16, leaving only ~7 digits of the distance
      // to the singularity at 30 levels. `levels` is a runtime value here, not a template.
      static bool UseCenteredRules();
      // L_i(u0+d) with the vanishing factor formed as `d` itself, never as a subtraction of
      // absolute coordinates. dM = DiffMat . M as usual.
      template <Integer order> static void LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& delta, const Integer ti);
      // Geometric panels marching outward from u0 to each end; `levels`+1 panels per side.
      static void BuildCenteredGraded1D(Vector<Real>& delta, Vector<Real>& w, const Real u0, const Integer levels, const Vector<Real>& qnds, const Vector<Real>& qwts);
      // Offset-valued counterpart of LogSingularQuad1D (which is already outward-graded).
      static void LogSingularQuad1DCentered(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder);
      template <Integer order, Integer digits> static const NodeRuleData& CenteredURule(const Integer ti, const Integer levels, const Integer kid = -1);

      // ---- Duffy edge-collapsed self scheme ----
      // The panel is split at (u0,v0) into four quads with one edge collapsed onto the
      // target. P(s,t) = (u0,v0) + s*c(t) has |det| = s*|a x b|, so the 1/r singularity is
      // removed by the Jacobian: s needs only a plain GL rule and t a rule graded toward the
      // foot of the perpendicular. Everything but the t-rule is fixed by (order,digits,ti,tj,tri).
      struct DuffyTri {
        bool swap_ab = false;    // collapsed (s-only) coordinate is u => local (alpha,beta) = (v,u)
        Real nsign = 1;          // restores the sign of dX/du x dX/dv
        Real J0 = 0;             // |a x b|
        Real tstarI = 0, ddI = 0, Llen = 0;  // parameter-space foot and width; metric-corrected per target
        Matrix<Real> WbC;        // (order x 2*ns) = [Wb | Wb'], collapsed direction at the s-nodes
        Matrix<Real> WbT;        // (ns x order), adjoint of the value half
        Vector<Matrix<Real>> MiC, MiT;       // ns entries: (order x 2*order) = [Mi | Mi'], and (order x order)
      };
      struct DuffySelfTable {
        Integer ns = 0;
        Vector<Real> sn, sw;
        std::vector<DuffyTri> tri;   // 4*order*order entries, indexed (ti*order + tj)*4 + tri
      };
      template <Integer order, Integer digits> static const DuffySelfTable& DuffyTable();
      template <Integer digits> static constexpr Integer DuffySOrderDelta();
      template <Integer digits> static Integer DuffyTOrder(const Integer order, const Integer kdim0);
      template <Integer digits, Integer order, class Kernel> static void SelfInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);
      template <Integer order, Integer digits> static const NodeRuleData& CenteredVRule(const Integer tj, const Integer kid = -1);

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
      template <Integer order, class Kernel> static void IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, 
                                                                        const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, 
                                                                        const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker, 
                                                                        const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr,
                                                                        const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr,
                                                                        const Vector<Real>* src_nodal = nullptr, const Matrix<Real>* MuD_pre = nullptr, const Real nrm_sign = 1,
                                                                        Vector<Real>* acc_cm = nullptr);

      // Geometry-independent graded 1D GL rule on [0,1], refined toward `center` until
      // admissible or `max_depth`. Returns nodes `param`, weights `w`.
      static void BuildGraded1D(Vector<Real>& param, Vector<Real>& w, const Real center, const Real b_ellipse, const Vector<Real>& qnds, const Vector<Real>& qwts, const Integer max_depth);

      // Dyadic subdivision underlying BuildGraded1D: leaf segments ({a0,a1} each in `seg`) + depths.
      static void BuildGraded1DSegments(Vector<Real>& seg, Vector<Long>& seg_depth, const Real center, const Real b_ellipse, const Integer max_depth);

      // Composite/graded Alpert rule on [0,1] for a log singularity at interior v0: split at
      // v0, grade `Lvl` geometric panels per side toward v0. The v0-touching panel uses the
      // Alpert log endpoint correction; smooth panels use a `QuadOrder` Gauss-Legendre rule.
      // Tables: alpert_quadr.cpp.
      static void LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder);

      // Adaptive 2D quadtree underlying NearInteracBlock: leaf rectangles (4 reals {u0,u1,v0,v1}
      // each in `leaf_box`) + depths, graded toward the closest point to Xtrg. Shared with WriteNearInteracVTK.
      static void BuildNearLeaves(Vector<Real>& leaf_box, Vector<Long>& leaf_depth, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse, const Integer max_depth);

      // Accuracy/order-templated impls of NearInterac/SelfInterac: entry points dispatch runtime
      // order to compile-time `order` (switch {4..48}) and tolerance to `digits` (if-else), CSBQ-style.
      template <Integer order, class Kernel> static void SelfInteracDispatchDigits(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);
      template <Integer order, class Kernel> static void NearInteracDispatchDigits(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);
      template <Integer digits, Integer order, class Kernel> static void SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self);
      template <Integer digits, Integer order, class Kernel> static void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self);

      // Per-target adaptive 2D quadtree near-interaction block (off-surface target).
      static Integer NearOrderFromMetric(const Real* dXu, const Real* dXv, const Integer q_iso);

      // Leaf-batched equivalent of NearInteracBlock: same quadtree/interp-cache, but the per-leaf
      // geometry/kernel/projection GEMMs are batched across leaves (interval-grouped, separable,
      // bit-for-bit identical to the per-leaf path). This is the production near path; NearInteracBlock
      // is retained as the reference for the bit-for-bit gate (QuadElemTestAccess::CompareNearBlocks).

      // Per-target singular self-interaction block at (u0,v0): graded u-refinement + 1D log rule in v.
      template <Integer digits, Integer order, class Kernel> static void SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

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

      // --- Split-at-foot near scheme (default; SCTL_NEAR_SPLIT=0 for the old quadtree) ---
      // The bisection quadtree leaves the foot mid-cell and needs a position-dependent
      // interpolation operator per leaf interval (~350 matrices rebuilt per target). Splitting
      // the element AT the foot makes every refinement grade toward an ENDPOINT, so in
      // normalized sub-element coordinates the graded intervals depend only on the level and
      // their operators precompute once per (order,digits).
      //
      // Per side the normalized intervals are, grading toward the foot at x=1:
      //   shell_k = [1-2^-k, 1-2^-(k+1)]    the half of core_k away from the foot
      //   core_k  = [1-2^-k, 1]             the half touching it
      // Splitting at (u*,v*) leaves each sub-element ANISOTROPIC, and quadrisection would pass
      // that aspect ratio to every descendant -- giving cells still long in one direction while
      // already close to the target in the other (inadmissible). So the corner cell is bisected
      // along its longer PHYSICAL dimension only (parameter extent x surface speed), one split at
      // a time, until that dimension is admissible against the target distance. Each split emits
      // one leaf; the u- and v-levels advance independently, so every interval remains
      // shell_k / core_k at some level and its operators stay precomputed.
      // Near-only knobs, independent of the self path (SCTL_QUAD_ORDER drives both). For tuning
      // the near heuristic while self is held at a much tighter tolerance.
      //   SCTL_NEAR_QORDER   per-cell GL order        (default DigitsQuadOrder<digits>)
      //   SCTL_NEAR_BELLIPSE admissibility constant   (default DigitsBEllipse<digits>)
      // VALIDITY: calibrated and validated on the twisted unit sphere for twist <= pi/3
      // (element anisotropy <= ~4.2). At twist pi/2 (anisotropy 6.6) the near rule needs a
      // higher GL order than this gives -- measured q=20 vs 15 at digits=13 -- and the driver is
      // the parameterisation's own analyticity, not element size or local shear (the refinement
      // test b_ellipse*max(hu,hv) <= dist is scale-invariant, and keying q on the per-target
      // spd_v/spd_u ratio was measured to change nothing). Do not rely on this past pi/3.
      // Near-only quadrature rule: tolerance-dependent rho, and the end-foot Bernstein reach
      // that the split-at-(u0,v0) geometry actually needs. QuadParams (still used by self) pins
      // rho = 2.5 and the semi-major reach, which over-refines near by a^2/b^2 ~ 1.9x.
      // Near default; SCTL_NEAR_QORDER / SCTL_NEAR_BELLIPSE still override.
      static void NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder);
      // Self quadrature heuristic, from a JOINT (L,q_u) sweep at order 12 / 4x4 panels per face.
      // The error is NOT separable in L and q_u: q_u sets the PREFACTOR, not a floor --
      //     err ~= C(q) * 2^-L,   C(q) ~= 2.5e-3 * (q/4)^-1.85
      // (measured C: 2.5e-3, 1.19e-3, 6.98e-4, 4.56e-4, 3.22e-4 at q = 4,6,8,10,12; C*2^L is
      // constant to ~5% down each column). The single genuine floor is q_u=4 saturating near
      // 2.4e-8, so q_u=4 cannot reach 1e-8 at any depth. Pick q_u from that threshold and the
      // smallest L clearing tol with a 2x margin.
      //
      // q_u SATURATES, and it is the binding limit at tight tolerance -- adding levels does not
      // help once it does. Measured on order-12, 12x12 panels/face, 124k nodes:
      //   q_u=4 -> ~2.4e-8      q_u=6 -> 2.56e-12 (11.6 digits)     q_u=8 -> 5.47e-14 (13.3 digits)
      // Long double gives the SAME 2.57e-12 floor at q_u=6, so precision is not the limit there --
      // it is the u-rule order. q_u=8 is also CHEAPER than the shipped QuadParams order (18 at
      // digits=13): 13.0s vs 17.6s setup, because a higher q_u needs fewer levels.
      // Only at q_u=8 does double precision bite, near 5e-14, and the residual is then dominated
      // by the far-field direct N^2 sum (1.07e-14) rather than any singular quadrature (all <=1.8e-15).
      // Coarser surfaces floor earlier: at 4x4 panels/face SL floors near 1.5e-11 and DL near 1.3e-9.
      // These now drive self instead of QuadParams' order and the caller's max_depth.
      // SCTL_SELF_LVL / SCTL_QUAD_ORDER override for sweeps.
      template <Integer digits> static Integer SelfLevels();
      template <Integer digits> static Integer SelfQuadOrder();
      template <Integer digits> static Integer NearQuadOrder();
      template <Integer digits> static Real NearBEllipse();
      //   SCTL_NEAR_MAXLVL   near-only level cap (0 => use max_depth_). Near-touching targets
      //   (a neighbouring patch's node, foot distance ~0) refine to the cap regardless of the
      //   admissibility constant, so the cap -- not b_ellipse -- is what controls their error.
      static Integer NearMaxLvlOverride();
      // Normalized rule + operator from the sub-element's order nodes to this interval's nodes.
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
      static constexpr Integer NearMaxQuadOrder = 60;
      template <Integer order, Integer digits> static const Vector<GradeRule>& NearGradeTable(const Integer q_req = 0);
      template <Integer digits, Integer order, class Kernel> static void NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // Line-QBX / hedgehog near-interaction block (Lu 2019 sec.3.1); knobs from SetLineQBXParams.
      // Accurate only for panel-interior targets (see SetLineQBXParams seam caveat).
      template <Integer order, class Kernel> static void NearInteracBlockQBX(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

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
