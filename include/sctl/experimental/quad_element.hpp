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

      // Closest discretization NODE on elem_idx to Xtrg (brute force over the nodal grid);
      // seeds GetClosestPoint. Returns the distance, (ustar,vstar) the node's parameters.
      Real GetClosestNode(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) const;

      // Closest POINT on patch elem_idx to Xtrg over (u,v) in [0,1]^2: GetClosestNode seed, then
      // Gauss-Newton with a grid-search fallback. Returns the distance. This is the FOOT that the
      // near scheme splits at, and where it reads the metric to pick the per-target GL order.
      // n_iter/used_fallback (optional) report the Newton iteration count and whether it stalled.
      Real GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) const;

      // Cached 1D nodal differentiation matrix D (order x order) on the GL nodes,
      // D[i][a] = L_i'(node_a); D . LuV turns a value-interp operator into a deriv one.
      static const Matrix<Real>& DiffMat(const Integer order);
      template <Integer order> static const Matrix<Real>& DiffMat() { return DiffMat(order); }

      // 1D value + derivative interpolation from order GL nodes to `param`:
      // M[i][a] = L_i(param[a]) (order x N), dM = DiffMat<order> . M.
      template <Integer order> static void BuildInterp1D(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& param);

      // 1D quadrature rule (param, w) + value/derivative interp operators (M, dM = order x N).
      struct NodeRuleData { Vector<Real> param, w; Matrix<Real> M, dM, MT, dMT; };

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
      template <Integer order> static const DuffySelfTable& DuffyTable(const Integer digits);
      static Integer DuffyTOrder(const Integer digits, const Integer order, const Integer kdim0);
      template <Integer order, class Kernel> static void SelfInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      // Accumulate a tensor-product quadrature (u_param x v_param, weights wu (x) wv) on
      // elem_idx against target Xtrg into M_acc; normal_trg != null enables target-normal contraction.
      // Mv_pre/dMv_pre, Mu_pre/dMu_pre (optional): precomputed v/u interp operators (order x N) used
      // in place of building from param.
      template <Integer order, class Kernel> static void IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx,
                                                                        const Vector<Real>& Xtrg, const Vector<Real>& normal_trg,
                                                                        const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker,
                                                                        const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr,
                                                                        const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr,
                                                                        const Vector<Real>* src_nodal = nullptr, const Matrix<Real>* MuD_pre = nullptr, const Real nrm_sign = 1,
                                                                        Vector<Real>* acc_cm = nullptr);

      // Order-templated impls of NearInterac/SelfInterac: the entry points dispatch runtime order
      // to compile-time `order` (switch {4..48}), because `order` is the bound of every inner
      // loop. `digits` stays a RUNTIME parameter: it never sizes an array or bounds a loop, it
      // only picks a cached rule (DuffyTable / NearGradeTable / NearQuadOrder / NearBEllipse),
      // and every one of those already returns a runtime value. Templating it produced 16
      // byte-identical instantiations of each function below -- 77% of this file's compile time.
      static constexpr Integer MaxDigits = 16;   // digits in [0, MaxDigits)
      // 10^-d as the old `pow<d,Real>((Real)0.1)` computed it, so the tol -> digits mapping is
      // unchanged bit-for-bit (repeated multiplication, NOT the literal 1e-d).
      static Integer DigitsFromTol(const Real tol);
      template <Integer order, class Kernel> static void SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits);
      template <Integer order, class Kernel> static void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits);

      // --- Split-at-foot near scheme ---
      // Splitting the element AT the foot makes every refinement grade toward an ENDPOINT, so in
      // normalized sub-element coordinates the graded intervals depend only on the level and
      // their operators precompute once per `order`. (A bisection quadtree instead leaves
      // the foot mid-cell and needs a position-dependent interpolation operator per leaf
      // interval -- ~350 matrices rebuilt per target.)
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
      //
      // Per-cell GL order and admissibility constant, both from NearRhoRule: a tolerance-dependent
      // rho plus the end-foot Bernstein reach the split-at-foot geometry actually needs (weaker
      // than the semi-major reach by a^2/b^2 ~ 1.9x, since the foot lands on a cell ENDPOINT).
      // SCTL_NEAR_QORDER / SCTL_NEAR_BELLIPSE override each, for tuning near while self is held
      // at a much tighter tolerance.
      static void NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder);
      static Integer NearQuadOrder(const Integer digits);
      static Real NearBEllipse(const Integer digits);
      // Per-target GL order from the corner skew of the metric at the foot. The isotropic order
      // holds to ~120 degrees; past that the required order grows like 1/(180-theta), which the
      // refinement cannot supply -- the admissibility test b_ellipse*max(hu,hv) <= dist is
      // scale-invariant, so skew is refinement-invariant. Rounded up to a multiple of 4 (the
      // NearGradeTable ladder) and capped at NearMaxQuadOrder. SCTL_NEAR_CK scales it.
      static Integer NearOrderFromMetric(const Real* dXu, const Real* dXv, const Integer q_iso);
      //   SCTL_NEAR_MAXLVL   near-only level cap (0 => MaxNearLvl-1). Near-touching targets
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
      // The ladder itself is accuracy-independent: one static per `order` holds every rung the
      // corner-angle correction can select (each multiple of 4, plus each NearQuadOrder(d)).
      template <Integer order> static const Vector<GradeRule>& NearGradeTable(const Integer q);
      template <Integer order, class Kernel> static void NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits);

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
  };

}

#endif // _SCTL_QUAD_ELEMENT_HPP_
