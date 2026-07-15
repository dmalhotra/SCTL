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

      /**
       * Near/self singular-quadrature scheme: Adaptive (dyadic subdivision +
       * Alpert log correction, default), RectPolar (Bruno-2018 change of var),
       * or Hybrid (Adaptive near + RectPolar self). For Hybrid the near phase is
       * tolerance-driven (like Adaptive) while the self phase uses the RectPolar
       * COV knobs (`q`, `cov_order`/Nbeta) passed to SetQuadScheme.
       */
      enum class QuadScheme { Adaptive, RectPolar, Hybrid };

      /** Constructor. */
      QuadElemList() {}

      /**
       * Construct from nodal coordinates.
       * @param[in] order polynomial order of each element.
       * @param[in] coord node coords, AoS {x1,y1,z1,...,xn,yn,zn}.
       */
      template <class ValueType> QuadElemList(Integer order, const Vector<ValueType>& coord);

      /**
       * Initialize from nodal coordinates.
       * @param[in] order polynomial order of each element.
       * @param[in] coord node coords, AoS {x1,y1,z1,...,xn,yn,zn}.
       */
      template <class ValueType> void Init(Integer order, const Vector<ValueType>& coord);

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

      /** True if the self phase uses the RectPolar COV (RectPolar or Hybrid). */
      bool SelfUsesRectPolar() const { return scheme_ == QuadScheme::RectPolar || scheme_ == QuadScheme::Hybrid; }

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

      // Accumulate a tensor-product quadrature (u_param x v_param, weights wu (x) wv) on
      // elem_idx against target Xtrg into M_acc; normal_trg != null enables target-normal contraction.
      // Mv_pre/dMv_pre, Mu_pre/dMu_pre (optional): precomputed v/u interp operators (order x N) used in
      // place of building from param (self supplies Alpert v; self-RP supplies both; near/Adaptive leave null).
      template <Integer order, class Kernel> static void IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, 
                                                                        const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, 
                                                                        const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker, 
                                                                        const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr,
                                                                        const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr);

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
      template <Integer digits, Integer order, class Kernel> static void NearInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // Leaf-batched equivalent of NearInteracBlock: same quadtree/interp-cache, but the per-leaf
      // geometry/kernel/projection GEMMs are batched across leaves (interval-grouped, separable,
      // bit-for-bit identical to the per-leaf path). This is the production near path; NearInteracBlock
      // is retained as the reference for the bit-for-bit gate (QuadElemTestAccess::CompareNearBlocks).
      template <Integer digits, Integer order, class Kernel> static void NearInteracBlockBatched(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // Per-target singular self-interaction block at (u0,v0): graded u-refinement + 1D log rule in v.
      template <Integer digits, Integer order, class Kernel> static void SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      // Rectangular-polar (Bruno-2018) change-of-variable 1D rule on [0,1]: maps {gl_nds,gl_wts}
      // via eta_alpha to cluster toward the singularity (alpha=2*sing-1) with vanishing weight;
      // `q` flattens derivatives up to order q-1.
      static void RectPolarNodes1D(Vector<Real>& nodes, Vector<Real>& wts, const Real alpha, const Integer q, const Vector<Real>& gl_nds, const Vector<Real>& gl_wts);

      // Shared core for WriteNear/SelfInteracRPVTK: warped Nbeta x Nbeta CoV grid clustered toward (ustar,vstar).
      void WriteRectPolarGridVTK(const std::string& fname, const Long elem_idx, const Real ustar, const Real vstar, const Integer Nbeta) const;

      // RP counterparts of NearInteracBlock/SelfInteracBlock; quadrature size set by cov_order_, so not templated on `digits`.
      template <Integer order, class Kernel> static void NearInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);
      template <Integer order, class Kernel> static void SelfInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker);

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
      QuadScheme scheme_ = QuadScheme::Adaptive;
      Integer cov_q_ = 6;
      Integer cov_order_ = 0;
      Integer max_depth_ = 30;
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
