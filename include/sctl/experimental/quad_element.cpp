#ifndef _SCTL_QUAD_ELEMENT_CPP_
#define _SCTL_QUAD_ELEMENT_CPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sctl.hpp>
#include "sctl/experimental/quad_element.hpp"
#include "sctl/experimental/alpert_quadr.cpp"

namespace sctl {

  namespace detail_quadelem {

    static constexpr Integer COORD_DIM = 3;
    static constexpr Integer MaxTableOrder = 50;
    static constexpr Integer MaxUnblockedPts = 16384;
    static constexpr Integer MaxFusedPts = 4096;
    static_assert(MaxFusedPts <= MaxUnblockedPts, "a small rule must be one block of u-rows");

    /** static accessors to private data of QuadElemList */
    template <class Real> struct Access {
      static const Vector<Real>& Coord(const QuadElemList<Real>& qel) { return qel.coord; }
      static typename QuadElemList<Real>::QuadScheme Scheme(const QuadElemList<Real>& qel) { return qel.scheme_; }
      static const Vector<Real>& XnNode(const QuadElemList<Real>& qel) { return qel.Xn_node; }
      static const Vector<Real>& DCoordDu(const QuadElemList<Real>& qel) { return qel.dcoord_du; }
      static const Vector<Real>& DCoordDv(const QuadElemList<Real>& qel) { return qel.dcoord_dv; }
    };

    template <class Real> static constexpr Integer MaxDigits = 1 + GetSigBits<Real>::value()*30103/100000;

    template <class Real> static constexpr Integer MaxRefineLvl = GetSigBits<Real>::value();
    template <class Real> static constexpr Integer MaxNearRefineLvl = 2*GetSigBits<Real>::value(); // levels of the dyadic near rule: its pieces are stored as offsets from the closest point, so they can be smaller than the rounding of the parameters

    /** returns fname followed by the rank of comm, zero-padded to 6 digits */
    inline std::string RankFileName(const std::string& fname, const Comm& comm) {
      std::stringstream ss;
      ss << fname << std::setfill('0') << std::setw(6) << comm.Rank();
      return ss.str();
    }

    /**
     * Computes out_k = MuT * in_k * Mv for each k-th sub-block, where in={in_0,...,in_{n-1}} and
     * out={out_0,...,out_{n-1}} are flattened arrays.  out is resized if its size differs.
     */
    template <class ValueType> void EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv) {
      const Integer Nu = (Integer)MuT.Dim(0);
      const Integer R  = (Integer)MuT.Dim(1);
      const Integer S  = (Integer)Mv.Dim(0);
      const Integer Nv = (Integer)Mv.Dim(1);
      const Integer ncomp = (Integer)(in.Dim() / (R * S));
      SCTL_ASSERT(in.Dim() == ncomp * R * S);

      const Integer Nout = Nu * Nv;
      if (out.Dim() != ncomp * Nout) out.ReInit(ncomp * Nout);

      ScratchBuf<ValueType> tmp_buf(R * Nv);
      Matrix<ValueType> tmp(R, Nv, tmp_buf.begin(), false);

      for (Integer k = 0; k < ncomp; k++) {
        const Matrix<ValueType> in_(R, S, (Iterator<ValueType>)in.begin() + k * R * S, false);
        Matrix<ValueType> out_(Nu, Nv, out.begin() + k * Nout, false);
        Matrix<ValueType>::GEMM(tmp, in_, Mv);
        Matrix<ValueType>::GEMM(out_, MuT, tmp);
      }
    }

    /** sets D(i,j) to the derivative of the i-th Lagrange basis on nds at nds[j] */
    template <class Real> void LagrangeDiffMat(Matrix<Real>& D, const Vector<Real>& nds) {
      const Integer n = (Integer)nds.Dim();
      Vector<Real> f(n * n);
      f.SetZero();
      for (Integer i = 0; i < n; i++) f[i * n + i] = 1;
      D.ReInit(n, n);
      Vector<Real> df(n * n, D.begin(), false);
      LagrangeInterp<Real>::Derivative(df, f, nds);
    }

    // TODO: test the accuracy of dcoord_du, dcoord_dv in place of DiffMat in the rule builds and the Duffy node metric
    /**
     * Returns an 'order' x 'order' matrix for the given 'order'; entry (i, j) is the derivative of
     * the i-th Lagrange basis function on ParamNodes(order) at j-th node.
     */
    template <class Real> inline const Matrix<Real>& DiffMat(const Integer order) {
      SCTL_ASSERT(0 < order && order <= MaxTableOrder);
      static const Vector<Matrix<Real>> all = []() {
        Vector<Matrix<Real>> D(MaxTableOrder + 1);
        for (Integer n = 2; n <= MaxTableOrder; n++) LagrangeDiffMat(D[n], QuadElemList<Real>::ParamNodes(n));
        return D;
      }();
      return all[order];
    }

    /** returns floor(-log10(tol)), limited to [0, MaxDigits-1] */
    template <class Real> inline Integer DigitsFromTol(const Real tol) {
      for (Integer d = MaxDigits<Real>-1; d > 0; d--) if (tol <= pow<Real,Long>((Real)0.1, (Long)d)) return d;
      return 0;
    }

    /** Gauss-Legendre order, and the minimum ratio b_ellipse of target distance to panel size */
    template <class Real> struct QuadParamSet {
      Real b_ellipse;
      Integer quad_order;
    };

    /**
     * Returns the {b_ellipse, quad_order} pair for each digits, from QuadParams at tolerance
     * 10^-digits; one table per QuadParams function.
     * */
    template <class Real, void (*QuadParams)(Real&, Integer&, Real)> const QuadParamSet<Real>& CachedQuadParams(const Integer digits) {
      static const std::array<QuadParamSet<Real>, MaxDigits<Real>> table = []() {
        std::array<QuadParamSet<Real>, MaxDigits<Real>> t{};
        for (Integer d = 0; d < MaxDigits<Real>; d++) QuadParams(t[d].b_ellipse, t[d].quad_order, pow<Real,Long>((Real)0.1, (Long)d));
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real>);
      return table[digits];
    }

    #ifdef SCTL_QUAD_T
    using PrecompReal = QuadReal;
    #else
    using PrecompReal = long double;
    #endif

    /** 1D rule: weights w; basis values M, derivatives dM at its N points; MTD: rows of M^T, then of dM^T (2N x order) */
    template <class Real> struct QuadRule1D {
      Vector<Real> w;
      Matrix<Real> M, dM, MTD;
    };

    /** evaluates position (minus origin, if given) and tangents dXu, dXv (if non-null) at (u,v) */
    template <class Real> void EvalPoint(Real* X, Real* dXu, Real* dXv, const Vector<Real>& coord, const Vector<Real>& dcoord_du, const Vector<Real>& dcoord_dv, const Integer order, const Real u, const Real v, const Vector<Real>* origin) {
      const Integer nnode = order * order;
      ScratchPool& pool = ScratchPool::Instance(); // looked up once per call, not once per buffer

      ScratchBuf<Real> Luv(2*order, pool); // Luv[i], Luv[order+i]: i-th interpolation weight at u, at v
      { // Interpolation weights at u and at v
        ScratchBuf<Real> L(2*order, pool); // L[2*i], L[2*i+1]
        StaticArray<Real,2> uv{u, v};
        const Vector<Real> trg(2, uv, false);
        Vector<Real> L_(L);
        LagrangeInterp<Real>::Interpolate(L_, QuadElemList<Real>::ParamNodes(order), trg);
        for (Integer i = 0; i < order; i++) {
          Luv[i] = L[2*i];
          Luv[order + i] = L[2*i+1];
        }
      }

      Real x[COORD_DIM] = {0, 0, 0}, xu[COORD_DIM] = {0, 0, 0}, xv[COORD_DIM] = {0, 0, 0};
      // Interpolate the coordinates, and their derivatives if want_d (a template parameter: no branch in the loop)
      const auto sum_nodes = [&coord, &dcoord_du, &dcoord_dv, &Luv, &x, &xu, &xv, order, nnode](const auto want_d) {
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            const Integer p = i*order + j;
            const Real w = Luv[i]*Luv[order + j];
            for (Integer k = 0; k < COORD_DIM; k++) x[k] += coord[k*nnode + p]*w;
            if constexpr (decltype(want_d)::value) {
              for (Integer k = 0; k < COORD_DIM; k++) {
                xu[k] += dcoord_du[k*nnode + p]*w;
                xv[k] += dcoord_dv[k*nnode + p]*w;
              }
            }
          }
        }
      };
      if (dXu || dXv) sum_nodes(std::true_type{});
      else sum_nodes(std::false_type{});
      for (Integer k = 0; k < COORD_DIM; k++) {
        X[k] = (origin ? x[k] - (*origin)[k] : x[k]);
        if (dXu) dXu[k] = xu[k];
        if (dXv) dXv[k] = xv[k];
      }
    }

    /** sets out to coord minus Xtrg, for component-major nodal coordinates */
    template <class Real> void ShiftedElemCoord(Vector<Real>& out, const Vector<Real>& coord, const Vector<Real>& Xtrg) {
      const Integer nnode = (Integer)(coord.Dim() / COORD_DIM);
      if (out.Dim() != COORD_DIM*nnode) out.ReInit(COORD_DIM*nnode);
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Integer p = 0; p < nnode; p++) out[k*nnode + p] = coord[k*nnode + p] - ok;
      }
    }

    /** returns the distance from Xtrg to the nearest node; (ustar, vstar) are its parameters */
    template <class Real> Real GetClosestNode(Real& ustar, Real& vstar, const Vector<Real>& coord, const Integer order, const Vector<Real>& Xtrg) {
      const Integer nnode = order * order;
      Integer seed = 0;
      Real best = -1;
      for (Integer p = 0; p < nnode; p++) {
        Real r2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          const Real d = coord[k*nnode + p] - Xtrg[k];
          r2 += d*d;
        }
        if (best < 0 || r2 < best) {
          best = r2;
          seed = p;
        }
      }

      const auto& nds = QuadElemList<Real>::ParamNodes(order);
      ustar = nds[seed/order];
      vstar = nds[seed%order];
      return sqrt<Real>(best);
    }

    /** returns the distance from Xtrg to the element; (ustar, vstar) are the closest point's parameters, found to about 1% of the distance */
    template <class Real> Real GetClosestPoint(Real& ustar, Real& vstar, const Vector<Real>& coord, const Vector<Real>& dcoord_du, const Vector<Real>& dcoord_dv, const Integer order, const Vector<Real>& Xtrg) {
      const auto dist2_at = [&coord, &dcoord_du, &dcoord_dv, order, &Xtrg](const Real uu, const Real vv) -> Real {
        Real X[COORD_DIM];
        EvalPoint<Real>(X, nullptr, nullptr, coord, dcoord_du, dcoord_dv, order, uu, vv, &Xtrg);
        Real r2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) r2 += X[k]*X[k];
        return r2;
      };

      Real u, v, f;
      { // Start from the nearest node
        const Real f_seed = GetClosestNode(u, v, coord, order, Xtrg);
        f = f_seed * f_seed;
      }

      Real dtol = 0; // rounding error of a computed distance
      {
        Real S = 0;
        for (const auto& c : coord) S = std::max<Real>(S, fabs(c));
        for (Integer k = 0; k < COORD_DIM; k++) S = std::max<Real>(S, fabs(Xtrg[k]));
        dtol = machine_eps<Real>() * 32 * S;
      }

      constexpr Integer max_iter = 30;
      const Real utol = machine_eps<Real>();
      bool converged = false;
      for (Integer it = 0; it < max_iter; it++) { // Projected Newton iterations
        Real X[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM]; // the point relative to the target, and the tangents
        EvalPoint<Real>(X, dXu, dXv, coord, dcoord_du, dcoord_dv, order, u, v, &Xtrg);
        Real E = 0, F = 0, G = 0, gu = 0, gv = 0;
        for (Integer k = 0; k < COORD_DIM; k++) { // Metric and gradient of the squared distance
          const Real r = X[k], a = dXu[k], b = dXv[k];
          E += a*a;
          F += a*b;
          G += b*b;
          gu += r*a;
          gv += r*b;
        }

        Real Pu = gu, Pv = gv;
        { // Project the gradient onto the parameter square
          if      (u <= 0) Pu = std::min<Real>(gu, (Real)0);
          else if (u >= 1) Pu = std::max<Real>(gu, (Real)0);
          if      (v <= 0) Pv = std::min<Real>(gv, (Real)0);
          else if (v >= 1) Pv = std::max<Real>(gv, (Real)0);
        }

        const Real rel_tol = machine_eps<Real>() * 10000; // singular values below rel_tol times the largest are taken as zero; the computed tangents carry rounding errors up to about 1000 eps
        const Real zero_len2 = rel_tol * rel_tol * (E + G); // a tangent with squared length below this is taken as zero
        Real step_u = 0, step_v = 0;
        { // Newton step: the least-squares solution of [dXu dXv] step = X, fixing a coordinate held at its bound
          const bool u_act = ((u <= 0 && gu >= 0) || (u >= 1 && gu <= 0));
          const bool v_act = ((v <= 0 && gv >= 0) || (v >= 1 && gv <= 0));
          if (!u_act && !v_act) { // from the SVD [dXu dXv] = [c1 c2] V^T, V the rotation making the columns c1, c2 orthogonal (one-sided Jacobi)
            Real cs = 1, sn = 0;
            if (F != 0) {
              const Real zeta = (G - E) / (2*F);
              const Real t = (zeta >= 0 ? 1 : -1) / (fabs(zeta) + sqrt<Real>(1 + zeta*zeta));
              cs = 1 / sqrt<Real>(1 + t*t);
              sn = cs * t;
            }
            Real n1 = 0, n2 = 0, y1 = 0, y2 = 0; // squared lengths of c1 and c2, and their dot products with X
            for (Integer k = 0; k < COORD_DIM; k++) {
              const Real c1 = cs*dXu[k] - sn*dXv[k];
              const Real c2 = sn*dXu[k] + cs*dXv[k];
              n1 += c1*c1;
              n2 += c2*c2;
              y1 += c1*X[k];
              y2 += c2*X[k];
            }
            const Real n_min = rel_tol * rel_tol * std::max<Real>(n1, n2);
            y1 = (n1 > n_min ? y1 / n1 : 0);
            y2 = (n2 > n_min ? y2 / n2 : 0);
            step_u = cs*y1 + sn*y2;
            step_v = -sn*y1 + cs*y2;
          } else if (u_act) {
            step_v = (G > zero_len2 ? gv / G : 0);
          } else {
            step_u = (E > zero_len2 ? gu / E : 0);
          }
        }

        Real un = u, vn = v, fn = f;
        bool improved;
        { // Line search along the step, else along the gradient; a step is rejected only if the distance grows by more than its rounding error
          const Real fmax = f + dtol * (2 * sqrt<Real>(f) + dtol);
          const auto line_search = [&dist2_at, u, v, fmax, &un, &vn, &fn](const Real step_u, const Real step_v) {
            Real lambda = 1;
            for (Integer ls = 0; ls < 40; ls++) {
              un = std::min<Real>(1, std::max<Real>(0, u - lambda*step_u));
              vn = std::min<Real>(1, std::max<Real>(0, v - lambda*step_v));
              fn = dist2_at(un, vn);
              if (fn <= fmax) return true;
              lambda *= (Real)0.5;
            }
            return false;
          };
          improved = line_search(step_u, step_v);
          if (!improved) improved = line_search((E > zero_len2 ? Pu / E : 0), (G > zero_len2 ? Pv / G : 0));
        }
        if (!improved) { // Every step grows the distance: a minimum
          converged = true;
          break;
        }

        const Real du = un - u, dv = vn - v;
        const bool small_step = (fabs(du) < utol && fabs(dv) < utol) || (E*du*du + 2*F*du*dv + G*dv*dv < (Real)1e-4 * fn); // the update is below machine epsilon, or moved the point by less than 1% of the distance
        u = un;
        v = vn;
        f = fn;
        if (small_step) {
          converged = true;
          break;
        }
      }

      if (!converged) { // Fall back to a refining grid search
        constexpr Integer K = 8, levels = 25;
        Real u0 = 0, u1 = 1, v0 = 0, v1 = 1;
        for (Integer L = 0; L < levels; L++) {
          for (Integer i = 0; i <= K; i++) {
            const Real ui = u0 + (u1-u0)*i/(Real)K;
            for (Integer j = 0; j <= K; j++) {
              const Real vj = v0 + (v1-v0)*j/(Real)K;
              const Real r2 = dist2_at(ui, vj);
              if (r2 < f) {
                f = r2;
                u = ui;
                v = vj;
              }
            }
          }
          const Real hu = (u1-u0)/K, hv = (v1-v0)/K;
          { // Stop when the grid spacing at the best point is less than 1% of the distance
            Real X[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
            EvalPoint<Real>(X, dXu, dXv, coord, dcoord_du, dcoord_dv, order, u, v, &Xtrg);
            Real dX2 = 0;
            for (Integer k = 0; k < COORD_DIM; k++) dX2 += dXu[k]*dXu[k]*hu*hu + dXv[k]*dXv[k]*hv*hv;
            if (dX2 < (Real)1e-4 * f) break;
          }
          u0 = std::max<Real>(0, u-hu);
          u1 = std::min<Real>(1, u+hu);
          v0 = std::max<Real>(0, v-hv);
          v1 = std::min<Real>(1, v+hv);
          if ((u1-u0) < utol && (v1-v0) < utol) break;
        }
      }

      ustar = u;
      vstar = v;
      return sqrt<Real>(f);
    }

    /** sin of the angle between tangents with metric guu, guv, gvv, at least 1e-6; 1 if a tangent vanishes */
    template <class Real> Real SinTangentAngle(const Real guu, const Real guv, const Real gvv) {
      const Real den = guu*gvv;
      if (!(den > 0)) return 1;
      return std::max<Real>((Real)1e-6, sqrt<Real>(std::max<Real>(0, 1 - guv*guv/den)));
    }

    /** true if K::uKerMatrix takes the source normal */
    template <class K, class VT, class = void> struct UKerNeedsN : std::false_type {};
    template <class K, class VT> struct UKerNeedsN<K, VT, std::void_t<decltype(
        K::template uKerMatrix<0,VT>(std::declval<VT(&)[K::SrcDim()][K::TrgDim()]>(),
          std::declval<const VT(&)[3]>(),
          std::declval<const VT(&)[3]>(),
          (const void*)nullptr))>> : std::true_type {};

    /** same as WeightedKernel, for j in [j0,j1) only, using vector type VecType */
    template <class Real, class Kernel, class VecType, bool HAS_N, bool TRG_DOT>
    static void WeightedKernelVec(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Integer nq, const Integer run, const Integer ldx, const Integer ldo, const Integer j0, const Integer j1, const Real wj, const bool accum, ConstIterator<Real> ntrg, const void* ctx) {
      static constexpr Integer CD = 3;
      static constexpr Integer KD0 = Kernel::SrcDim();
      static constexpr Integer KD1 = Kernel::TrgDim();
      static constexpr Integer KD1o = (TRG_DOT ? KD1/CD : KD1);
      static constexpr Integer C = KD0*KD1o;
      static constexpr Integer digits = (Integer)(TypeTraits<Real>::SigBits*0.3010299957);
      static constexpr Integer VL = VecType::Size();
      const VecType vws(wj * Kernel::template uKerScaleFactor<Real>());
      VecType vXt[CD];
      for (Integer k = 0; k < CD; k++) vXt[k] = VecType(Xt[k]);
      for (Integer qb = 0, blk = 0; qb < nq; qb += run, blk++) {
        for (Integer j = j0; j < j1; j += VL) {
          const Integer q = qb + j;
          VecType r[CD], n[CD], u[KD0][KD1];
          for (Integer k = 0; k < CD; k++) r[k] = vXt[k] - VecType::Load(&Xs[k*ldx+q]);
          if constexpr (HAS_N) {
            for (Integer k = 0; k < CD; k++) n[k] = VecType::Load(&Xn[k*ldx+q]);
            Kernel::template uKerMatrix<digits,VecType>(u, r, n, ctx);
          } else {
            Kernel::template uKerMatrix<digits,VecType>(u, r, ctx);
          }
          const VecType vw = vws * VecType::Load(&wq[q]);
          for (Integer a = 0; a < KD0; a++) {
            for (Integer b = 0; b < KD1o; b++) {
              VecType val;
              if constexpr (TRG_DOT) {
                val = u[a][b*CD+0] * VecType(ntrg[0]);
                for (Integer l = 1; l < CD; l++) val = val + u[a][b*CD+l] * VecType(ntrg[l]);
              } else {
                val = u[a][b];
              }
              const Integer id = blk*C*ldo + (a*KD1o+b)*ldo + j;
              if (accum) (VecType::Load(&out[id]) + val*vw).Store(&out[id]);
              else       (val*vw).Store(&out[id]);
            }
          }
        }
      }
    }

    /** out[(blk*C + c)*ldo + j] = wj*wq*K(Xt-Xs) at point blk*run+j of nq; component k of the point at Xs[k*ldx + point]; dotted with normal_trg if non-empty; added if accum */
    template <class Real, class Kernel> void WeightedKernel(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Integer nq, const Integer run, const Integer ldx, const Integer ldo, const Real wj, const bool accum, const Vector<Real>& normal_trg, const Kernel& ker) {
      static constexpr bool HAS_N = UKerNeedsN<Kernel, Vec<Real,1>>::value;
      using WVec = Vec<Real, DefaultVecLen<Real>()>;
      const Integer jmain = (run/WVec::Size())*WVec::Size();
      const ConstIterator<Real> nt = (normal_trg.Dim() ? normal_trg.begin() : ConstIterator<Real>(NullIterator<Real>()));
      if (normal_trg.Dim()) {
        WeightedKernelVec<Real,Kernel,WVec,        HAS_N,true >(out, Xt, Xs, Xn, wq, nq, run, ldx, ldo,     0, jmain, wj, accum, nt, ker.GetCtxPtr());
        WeightedKernelVec<Real,Kernel,Vec<Real,1>, HAS_N,true >(out, Xt, Xs, Xn, wq, nq, run, ldx, ldo, jmain,   run, wj, accum, nt, ker.GetCtxPtr());
      } else {
        WeightedKernelVec<Real,Kernel,WVec,        HAS_N,false>(out, Xt, Xs, Xn, wq, nq, run, ldx, ldo,     0, jmain, wj, accum, nt, ker.GetCtxPtr());
        WeightedKernelVec<Real,Kernel,Vec<Real,1>, HAS_N,false>(out, Xt, Xs, Xn, wq, nq, run, ldx, ldo, jmain,   run, wj, accum, nt, ker.GetCtxPtr());
      }
    }

    /** adds to acc_cm (ru,rv) integrals of kernel times basis; target at origin, or proxy-weighted sum */
    template <Integer order, class Real, class Kernel> void IntegrateTensorRule(Vector<Real>& acc_cm, const Vector<Real>& src_nodal, const QuadRule1D<Real>& ru, const QuadRule1D<Real>& rv, const Vector<Real>& normal_trg, const Kernel& ker, const Real nrm_sign = 1, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Integer nnode = order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      const Integer C = KDIM0 * KDIM1_out;
      const Integer Nu = (Integer)ru.M.Dim(1);
      const Integer Nv = (Integer)rv.M.Dim(1);
      SCTL_ASSERT(Nu > 0 && Nv > 0);
      ScratchPool& pool = ScratchPool::Instance(); // looked up once per call, not once per buffer

      // Small rules (one block, where the cost per product call dominates) store the coordinates side
      // by side, so that one product along u covers all of them, and project each component into acc_cm
      const bool fused = (Nu*Nv <= MaxFusedPts);
      const Integer ldc = COORD_DIM*Nv;

      ScratchBuf<Real> Cv(COORD_DIM*order*Nv, pool), Cdv(COORD_DIM*order*Nv, pool); // [k][i*Nv + b], or [i][k*Nv + b] for a small rule
      { // Interpolate the coordinates and their v-derivative along v
        ScratchBuf<Real> cs_ik(fused ? COORD_DIM*order*order : 0, pool); // rows (i,k) of src_nodal
        if (fused) {
          for (Integer i = 0; i < order; i++) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              for (Integer j = 0; j < order; j++) cs_ik[(i*COORD_DIM + k)*order + j] = src_nodal[(k*order + i)*order + j];
            }
          }
        }
        const Matrix<Real> cs_all(COORD_DIM*order, order, (fused ? cs_ik.begin() : (Iterator<Real>)src_nodal.begin()), false);
        Matrix<Real> Cv_all (COORD_DIM*order, Nv, Cv.begin(),  false);
        Matrix<Real> Cdv_all(COORD_DIM*order, Nv, Cdv.begin(), false);
        const SmallGEMM<Real, COORD_DIM*order, DynamicSize, order> interp_v(false, COORD_DIM*order, Nv, order);
        interp_v(Cv_all,  cs_all, rv.M);
        interp_v(Cdv_all, cs_all, rv.dM);
      }

      ScratchBuf<Real> Tall(fused ? 0 : Nu*C*order, pool);
      const Integer UBLK = std::max<Integer>(1, std::min<Integer>(Nu, MaxUnblockedPts / Nv));
      for (Integer a0 = 0; a0 < Nu; a0 += UBLK) { // Blocks of u-rows
        const Integer nu = std::min<Integer>(UBLK, Nu - a0);
        const Integer nqb = nu*Nv;

        // The block's coordinates, normals, weights, and tangents (large rule) or later kernel values in
        // the same space, nqb values per component. Consecutive components are an odd number of cache lines
        // apart, and 25 lines modulo 64 (about 0.38 of 4 KB) for components of 4 KB or more, so that no two
        // share an L1 set, lie a multiple of 4 KB apart, or cross 4 KB boundaries together (up to 1.9x slower)
        constexpr Integer LineVals = std::max<Integer>(1, 64/(Integer)sizeof(Real));
        Integer lines = (nqb + LineVals - 1)/LineVals;
        lines = (lines < 64 ? (lines | 1) : lines + (25 - lines % 64 + 64) % 64);
        const Integer ld = lines*LineVals;
        ScratchBuf<Real> pts((2*COORD_DIM + 1 + std::max<Integer>(C, 2*COORD_DIM))*ld, pool);
        const Iterator<Real> Xs = pts.begin(), Xn = Xs + COORD_DIM*ld, wq = Xn + COORD_DIM*ld, KW = wq + ld;
        { // Points, normals and weights of the block
          const Integer nfused = (fused ? nqb : 0);
          ScratchBuf<Real> XdU(2*COORD_DIM*nfused, pool), dV(COORD_DIM*nfused, pool); // row a: [k*Nv + b]; XdU: coordinates, then u-tangents
          const Iterator<Real> dXu = KW, dXv = dXu + COORD_DIM*ld; // large rule: component k at k*ld, before the kernel values
          if (fused) { // Coordinates and u-tangents of all components in one product, v-tangents in another
            const Matrix<Real> MuD_m(2*nu, order, (Iterator<Real>)ru.MTD.begin(), false); // the only block: all of MTD
            const Matrix<Real> MuT_b(nu, order, (Iterator<Real>)ru.MTD.begin(), false);
            const Matrix<Real> Cvc_m(order, ldc, Cv.begin(), false);
            const Matrix<Real> Cdvc_m(order, ldc, Cdv.begin(), false);
            Matrix<Real> XdU_m(2*nu, ldc, XdU.begin(), false);
            Matrix<Real> dV_m(nu, ldc, dV.begin(), false);
            const SmallGEMM<Real, DynamicSize, DynamicSize, order> interp_xdu(false, 2*nu, ldc, order), interp_dv(false, nu, ldc, order);
            interp_xdu(XdU_m, MuD_m, Cvc_m);
            interp_dv(dV_m, MuT_b, Cdvc_m);
          } else { // Interpolate coordinates and tangents along u, one product per coordinate
            const Matrix<Real> MuT_b (nu, order, (Iterator<Real>)ru.MTD.begin() + a0*order, false);
            const Matrix<Real> dMuT_b(nu, order, (Iterator<Real>)ru.MTD.begin() + (Nu + a0)*order, false);
            const SmallGEMM<Real, DynamicSize, DynamicSize, order> interp_u(false, nu, Nv, order);
            for (Integer k = 0; k < COORD_DIM; k++) {
              const Matrix<Real> Cv_k (order, Nv, Cv.begin()  + k*order*Nv, false);
              const Matrix<Real> Cdv_k(order, Nv, Cdv.begin() + k*order*Nv, false);
              Matrix<Real> X_k(nu, Nv, Xs + k*ld, false);
              Matrix<Real> dXu_k(nu, Nv, dXu + k*ld, false);
              Matrix<Real> dXv_k(nu, Nv, dXv + k*ld, false);
              interp_u(X_k,   MuT_b,  Cv_k);
              interp_u(dXu_k, dMuT_b, Cv_k);
              interp_u(dXv_k, MuT_b,  Cdv_k);
            }
          }
          const ConstIterator<Real> Xp = (fused ? (ConstIterator<Real>)XdU.begin() : (ConstIterator<Real>)Xs);
          const ConstIterator<Real> dUp = (fused ? (ConstIterator<Real>)XdU.begin() + nu*ldc : (ConstIterator<Real>)dXu);
          const ConstIterator<Real> dVp = (fused ? (ConstIterator<Real>)dV.begin() : (ConstIterator<Real>)dXv);
          const Integer sk = (fused ? Nv : ld), sa = (fused ? ldc : Nv); // component and u-row strides
          for (Integer a = 0; a < nu; a++) {
            for (Integer b = 0; b < Nv; b++) {
              const Integer q = a*Nv + b;
              const Integer p = a*sa + b;
              const Real du0 = dUp[p], du1 = dUp[p + sk], du2 = dUp[p + 2*sk];
              const Real dv0 = dVp[p], dv1 = dVp[p + sk], dv2 = dVp[p + 2*sk];
              const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
              const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
              const Real inv_area = (area > 0 ? nrm_sign/area : 0);
              if (fused) {
                for (Integer k = 0; k < COORD_DIM; k++) Xs[k*ld + q] = Xp[p + k*sk];
              }
              Xn[0*ld+q] = n0*inv_area;
              Xn[1*ld+q] = n1*inv_area;
              Xn[2*ld+q] = n2*inv_area;
              wq[q] = area*ru.w[a0+a]*rv.w[b];
            }
          }
        }

        const Integer np = std::max<Integer>(1, (Integer)proxy_w.Dim());
        for (Integer j = 0; j < np; j++) { // Weighted kernel, summed over the proxy points
          StaticArray<Real,COORD_DIM> Xtj{0, 0, 0};
          if (proxy_w.Dim()) {
            for (Integer l = 0; l < COORD_DIM; l++) Xtj[l] = proxy_off[j*COORD_DIM+l];
          }
          const Vector<Real> Xtj_v(COORD_DIM, Xtj, false);
          const Real wj = (proxy_w.Dim() ? proxy_w[j] : (Real)1);
          const bool accum = (j > 0);
          WeightedKernel<Real>(KW, Xtj_v.begin(), Xs, Xn, wq, nqb, nqb, ld, ld, wj, accum, normal_trg, ker);
        }

        { // Project onto the v-nodes: into rows (a,c) of Tall, or for a small rule into Tblk and then onto the u-nodes
          ScratchBuf<Real> Tblk(fused ? C*nu*order : 0, pool); // [c][a*order + j]; rows (a,c) as in Tall, through strides known only at run time, made small rules up to 1.7% slower
          const SmallGEMM<Real, DynamicSize, order, DynamicSize> proj_v(false, nu, order, Nv, Nv, order, (fused ? order : C*order));
          for (Integer c = 0; c < C; c++) proj_v((fused ? Tblk.begin() + c*nu*order : Tall.begin() + (a0*C + c)*order), KW + c*ld, rv.MTD.begin());
          if (fused) { // the only block: nu = Nu
            const SmallGEMM<Real, order, order, DynamicSize> proj_u(true, order, order, nu);
            for (Integer c = 0; c < C; c++) proj_u(acc_cm.begin() + c*nnode, ru.M.begin(), Tblk.begin() + c*nu*order);
          }
        }
      }

      if (!fused) { // Project onto the u-nodes and add into acc_cm
        ScratchBuf<Real> Aall(order*C*order, pool);
        const Matrix<Real> T_m(Nu, C*order, Tall.begin(), false);
        Matrix<Real> A_m(order, C*order, Aall.begin(), false);
        const SmallGEMM<Real, order, DynamicSize, DynamicSize> proj_u(false, order, C*order, Nu);
        proj_u(A_m, ru.M, T_m);
        for (Integer c = 0; c < C; c++) {
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) acc_cm[c*nnode + i*order + j] += Aall[(i*C + c)*order + j];
          }
        }
      }
    }

    /** sets M_acc[p][c] to the (ru,rv) integral at the target of kernel component c times basis p; coord_shift are the nodal coordinates relative to the target */
    template <Integer order, class Real, class Kernel> void IntegratePanel(Matrix<Real>& M_acc, const Vector<Real>& coord_shift, const Vector<Real>& normal_trg, const QuadRule1D<Real>& ru, const QuadRule1D<Real>& rv, const Kernel& ker) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(coord_shift.Dim() == COORD_DIM*order*order);
      const Integer nnode = order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      const Integer C = KDIM0 * KDIM1_out;

      ScratchBuf<Real> acc_buf(C*nnode);
      Vector<Real> acc(acc_buf);
      acc.SetZero();
      IntegrateTensorRule<order,Real>(acc, coord_shift, ru, rv, normal_trg, ker);
      for (Integer p = 0; p < nnode; p++) for (Integer c = 0; c < C; c++) M_acc[p][c] = acc[c*nnode + p];
    }

    /** copies src, read as nrow x KDIM1_out, into columns t*KDIM1_out to (t+1)*KDIM1_out-1 of M */
    template <class Real> void ScatterTargetBlock(Matrix<Real>& M, const Matrix<Real>& src, const Long t, const Integer KDIM1_out) {
      const Integer nrow = (Integer)M.Dim(0);
      SCTL_ASSERT(src.Dim(0)*src.Dim(1) == nrow*KDIM1_out);
      const ConstIterator<Real> src_ = src.begin();
      for (Integer r = 0; r < nrow; r++) {
        for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          M[r][t*KDIM1_out+k1] = src_[r*KDIM1_out+k1];
        }
      }
    }

    /** sets M to the matrix from element elem_idx to targets Xt, by near_interac_one_trg per target */
    template <Integer order, class Real, class Kernel, class NearInteracOneTrg>
    void NearInteracTargets(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const QuadElemList<Real>& qel, const Long elem_idx, NearInteracOneTrg near_interac_one_trg) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Integer nnode = order * order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

      const Long Ntrg = Xt.Dim() / COORD_DIM;
      if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != Ntrg*KDIM1_out) M.ReInit(nnode*KDIM0, Ntrg*KDIM1_out);

      const Long offset = elem_idx*COORD_DIM*nnode;
      const Vector<Real> coord(COORD_DIM*nnode, (Iterator<Real>)Access<Real>::Coord(qel).begin() + offset, false);
      const Vector<Real> dcoord_du(COORD_DIM*nnode, (Iterator<Real>)Access<Real>::DCoordDu(qel).begin() + offset, false);
      const Vector<Real> dcoord_dv(COORD_DIM*nnode, (Iterator<Real>)Access<Real>::DCoordDv(qel).begin() + offset, false);
      ScratchBuf<Real> M_acc_buf(nnode*KDIM0*KDIM1_out);
      Matrix<Real> M_acc(nnode, KDIM0*KDIM1_out, M_acc_buf.begin(), false);
      for (Long t = 0; t < Ntrg; t++) {
        const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)normal_trg.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        near_interac_one_trg(M_acc, coord, dcoord_du, dcoord_dv, Xtrg, ntrg);
        ScatterTargetBlock(M, M_acc, t, KDIM1_out);
      }
    }

    /** sets M_lst[e] to the matrix from element e to its nodes, by self_interac_one_trg per node */
    template <Integer order, class Real, class Kernel, class SelfInteracOneTrg>
    void SelfInteracElems(Vector<Matrix<Real>>& M_lst, const bool trg_dot_prod, const QuadElemList<Real>& qel, SelfInteracOneTrg self_interac_one_trg) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      SCTL_ASSERT((Long)M_lst.Dim() == qel.Size());
      const Integer nnode = order * order;
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

      const Integer nrow = nnode*KDIM0;
      const Integer blk = nrow*KDIM1_out; // one target's block of M_acc

      const Long nelem = qel.Size();
      #pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) { // Size each element's matrix
        Matrix<Real>& M = M_lst[elem_idx];
        if (M.Dim(0) != nrow || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nrow, nnode*KDIM1_out);
      }

      const Integer chunk = [KDIM1_out, nelem]() { // Targets per task: whole cache lines of M, halved until each thread has 8 tasks
        const Integer line = SCTL_MEM_ALIGN;
        const Long min_tasks = 8*(Long)SCTL_GET_MAX_THREADS();
        Integer c = line / std::gcd(line, KDIM1_out*(Integer)sizeof(Real));
        while (c > 1 && nelem*((nnode + c - 1) / c) < min_tasks) c /= 2;
        return c;
      }();
      const Integer ntask_elem = (nnode + chunk - 1) / chunk;

      #pragma omp parallel
      {
        ScratchBuf<Real> buf((Long)chunk*blk);
        // Each run of consecutive targets of one element; consecutive tasks take the same run on
        // consecutive elements, so they share its targets' precomputed tables while those are in cache
        #pragma omp for schedule(dynamic)
        for (Long task = 0; task < nelem*ntask_elem; task++) {
          const Long elem_idx = task % nelem;
          const Integer t0 = (Integer)(task / nelem)*chunk;
          const Integer t1 = std::min<Integer>(nnode, t0 + chunk);

          // Coordinates and tangents component by component, normals point by point
          const Long offset = elem_idx*nnode*COORD_DIM;
          const Vector<Real> coord(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::Coord(qel).begin() + offset, false);
          const Vector<Real> dXu(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::DCoordDu(qel).begin() + offset, false);
          const Vector<Real> dXv(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::DCoordDv(qel).begin() + offset, false);
          const Vector<Real> Xnnodes(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::XnNode(qel).begin() + offset, false);

          for (Integer t = t0; t < t1; t++) { // Each target's block into the buffer
            Matrix<Real> M_acc(nnode, KDIM0*KDIM1_out, buf.begin() + (t - t0)*blk, false);
            self_interac_one_trg(M_acc, coord, Xnnodes, dXu, dXv, t / order, t % order);
          }
          { // Copy the run's columns into M, row by row
            Matrix<Real>& M = M_lst[elem_idx];
            for (Integer r = 0; r < nrow; r++) {
              for (Integer t = t0; t < t1; t++) {
                for (Integer k1 = 0; k1 < KDIM1_out; k1++) M[r][t*KDIM1_out + k1] = buf[(t - t0)*blk + r*KDIM1_out + k1];
              }
            }
          }
        }
      }
    }

  }

  namespace detail_dyadic_near {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxNearRefineLvl;
    using detail_quadelem::MaxTableOrder;
    using detail_quadelem::PrecompReal;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::CachedQuadParams;
    using detail_quadelem::EvalPoint;
    using detail_quadelem::GetClosestPoint;
    using detail_quadelem::IntegrateTensorRule;
    using detail_quadelem::LagrangeDiffMat;
    using detail_quadelem::NearInteracTargets;
    using detail_quadelem::ShiftedElemCoord;
    using detail_quadelem::SinTangentAngle;

    template <class Real> void QuadParams(Real& b_ellipse, Integer& quad_order, const Real tol) {
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real d = -log<Real>(tol_)/log<Real>((Real)10);
      const Real rho = std::min<Real>(3, std::max<Real>(2, 2 + (Real)0.25*(d - 6)));
      const Real C = std::max<Real>((Real)1e-3, (15*(rho*rho - 1))/64);
      quad_order = std::max<Integer>(2, (Integer)ceil<Real>(-log<Real>(C*tol_)/log<Real>(rho)*(Real)0.5 + 1));

      const Real a = (rho + 1/rho)/2, b = (rho - 1/rho)/2;
      b_ellipse = b*b/(2*a);
    }

    static constexpr Integer NearMaxQuadOrder = 60;

    /** Returns 'order' values cos^2(pi i/(2(order-1))) for each order: one minus the Chebyshev extreme points sin^2(pi i/(2(order-1))) on [0, 1], computed as cos^2 so that they are accurate where small. */
    template <class Real> static const Vector<Real>& NearSubOffsets(const Integer order) {
      SCTL_ASSERT(1 < order && order <= MaxTableOrder);
      static const Vector<Vector<Real>> all = []() {
        Vector<Vector<Real>> v(MaxTableOrder + 1);
        for (Integer n = 2; n <= MaxTableOrder; n++) {
          v[n].ReInit(n);
          using W = PrecompReal;
          for (Integer i = 0; i < n; i++) {
            const W ch = cos<W>(const_pi<W>()*i/(2*(n-1)));
            v[n][i] = (Real)(ch*ch);
          }
          v[n][0] = 1;
          v[n][n-1] = 0;
        }
        return v;
      }();
      return all[order];
    }

    /** Returns the derivatives of the Lagrange basis on the nodes NearSubOffsets(order), with respect to one minus the offset (order x order), computed in PrecompReal. */
    template <Integer order, class Real> const Matrix<Real>& NearSubDiffMat() {
      static const Matrix<Real> D = []() {
        using W = PrecompReal;
        Vector<W> sub_nds(order);
        for (Integer i = 0; i < order; i++) {
          const W sh = sin<W>(const_pi<W>()*i/(2*(order-1)));
          sub_nds[i] = sh*sh;
        }
        sub_nds[0] = 0;
        sub_nds[order-1] = 1;
        Matrix<W> Dw;
        LagrangeDiffMat(Dw, sub_nds);
        Matrix<Real> D_(order, order);
        for (Integer i = 0; i < order; i++) for (Integer j = 0; j < order; j++) D_[i][j] = (Real)Dw[i][j];
        return D_;
      }();
      return D;
    }

    /** sets r, already sized for nseg*q points, to the q-point Gauss-Legendre rule (nodes qn, weights qw on [0, 1]) on each segment [seg[2i], seg[2i+1]] of offsets from the refined end, with the interpolation matrices from the nodes NearSubOffsets(order); computed in W */
    template <Integer order, class W, class Real> void NearSegmentRule(QuadRule1D<Real>& r, ConstIterator<W> seg, const Integer nseg, const Vector<W>& qn, const Vector<W>& qw) {
      const Integer q = qn.Dim();
      const Integer N = nseg*q;
      ScratchBuf<W> tq_buf(N), T_buf(order*N), dT_buf(order*N);
      Vector<W> tq(tq_buf), Twts(T_buf);
      for (Integer s = 0; s < nseg; s++) {
        const W t_lo = seg[2*s+0], t_hi = seg[2*s+1];
        const W t_w = t_hi - t_lo;
        for (Integer j = 0; j < q; j++) {
          r.w[s*q + j] = (Real)(t_w*qw[j]);
          tq[s*q + j] = t_hi - t_w*qn[j];
        }
      }
      LagrangeInterp<W>::Interpolate(Twts, NearSubOffsets<W>(order), tq);
      const Matrix<W> T(order, N, Twts.begin(), false);
      Matrix<W> dT(order, N, dT_buf.begin(), false);
      Matrix<W>::GEMM(dT, NearSubDiffMat<order,W>(), T);
      for (Integer i = 0; i < order; i++) for (Integer j = 0; j < N; j++) {
        r.M[i][j] = (Real)T[i][j];
        r.dM[i][j] = (Real)dT[i][j];
        r.MTD[j][i] = r.M[i][j];
        r.MTD[N + j][i] = r.dM[i][j];
      }
    }

    /** Returns 2*MaxNearRefineLvl rules for each (order, q), built the first time q is requested: q-point Gauss-Legendre on the dyadic intervals at offsets [2^-(k+1), 2^-k] from the refined end and on the tails [0, 2^-k], each with order x q interpolation matrices. */
    template <Integer order, class Real> const Vector<QuadRule1D<Real>>& NearGradeTable(const Integer q) {
      const auto build = [](const Integer q) {
        using W = PrecompReal;
        Vector<W> qn, qw;
        LegQuadRule<W>::template ComputeNdsWts<W>(&qn, &qw, q);
        Vector<QuadRule1D<Real>> tab(2*MaxNearRefineLvl<Real>);
        for (Integer k = 0; k < MaxNearRefineLvl<Real>; k++) { // Rules on dyadic interval k and on its tail
          const W off_k = pow<W>((W)0.5, k), off_k1 = pow<W>((W)0.5, k+1);
          const StaticArray<W,4> seg{off_k1, off_k, (W)0, off_k};
          for (Integer tail = 0; tail < 2; tail++) {
            QuadRule1D<Real>& r = tab[tail*MaxNearRefineLvl<Real> + k];
            r.w.ReInit(q);
            r.M.ReInit(order, q);
            r.dM.ReInit(order, q);
            r.MTD.ReInit(2*q, order);
            NearSegmentRule<order>(r, (ConstIterator<W>)seg + 2*tail, 1, qn, qw);
          }
        }
        return tab;
      };
      static std::array<std::once_flag, NearMaxQuadOrder+1> built;
      static std::array<Vector<QuadRule1D<Real>>, NearMaxQuadOrder+1> all;
      SCTL_ASSERT(q > 0 && q <= NearMaxQuadOrder);
      std::call_once(built[q], [&build, q]() { all[q] = build(q); });
      return all[q];
    }

    /**
     * The element split at the target's closest point (ustar, vstar) into up to four sub-rectangles (sdu, sdv), whose
     * parameters are the offsets from the closest point divided by the side lengths slen[0][sdu] and slen[1][sdv]: the
     * distance dist of the closest point and the metric (guu, guv, gvv) there, the interpolation from the element to each
     * side's sub-interval (Sf, and its transpose St), and the coordinates of each sub-rectangle at its nodes relative to
     * the target (Xsub). The sub-rectangles have the closest point as a node, so that points near it are found from the
     * small offsets and the small coordinates there, not from the parameters and coordinates of size 1.
     */
    template <Integer order, class Real> struct NearSplit {
      ScratchBuf<Real> Sf, St, Xsub;
      Real ustar, vstar, dist, guu, guv, gvv;
      Real slen[2][2];

      NearSplit(const Vector<Real>& coord, const Vector<Real>& dcoord_du, const Vector<Real>& dcoord_dv, const Vector<Real>& Xtrg, ScratchPool& pool) : Sf(4*order*order, pool), St(4*order*order, pool), Xsub(4*COORD_DIM*order*order, pool) {
        const Integer nnode = order*order;
        ScratchBuf<Real> cs_buf(COORD_DIM*nnode, pool);
        Vector<Real> cs(cs_buf); // relative to the target; the closest point too is found from these, so that it is on the surface that the quadrature integrates
        ShiftedElemCoord(cs, coord, Xtrg);

        { // Closest point, and the metric there
          StaticArray<Real,COORD_DIM> origin{(Real)0, (Real)0, (Real)0};
          dist = GetClosestPoint(ustar, vstar, cs, dcoord_du, dcoord_dv, order, Vector<Real>(COORD_DIM, origin, false));
          Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
          EvalPoint<Real>(Xc, dXu, dXv, coord, dcoord_du, dcoord_dv, order, ustar, vstar, nullptr);
          guu = 0;
          guv = 0;
          gvv = 0;
          for (Integer k = 0; k < COORD_DIM; k++) {
            guu += dXu[k]*dXu[k];
            guv += dXu[k]*dXv[k];
            gvv += dXv[k]*dXv[k];
          }
          slen[0][0] = ustar;
          slen[0][1] = 1 - ustar;
          slen[1][0] = vstar;
          slen[1][1] = 1 - vstar;
        }

        { // Interpolation from the element to each side's sub-interval
          const Vector<Real>& gnds = QuadElemList<Real>::ParamNodes(order);
          const Vector<Real>& soff = NearSubOffsets<Real>(order);
          ScratchBuf<Real> gsh_buf(order, pool), sub_buf(order, pool);
          Vector<Real> gsh(gsh_buf), sub(sub_buf);
          for (Integer d = 0; d < 2; d++) {
            const Real xs = (d ? vstar : ustar);
            for (Integer i = 0; i < order; i++) gsh[i] = gnds[i] - xs;
            for (Integer sd = 0; sd < 2; sd++) {
              if (!(slen[d][sd] > 0)) continue;
              const Real sg = (sd ? slen[d][sd] : -slen[d][sd]);
              for (Integer i = 0; i < order; i++) sub[i] = sg*soff[i];
              Vector<Real> Sf_v(nnode, Sf.begin() + (2*d+sd)*nnode, false);
              LagrangeInterp<Real>::Interpolate(Sf_v, gsh, sub);
              const Matrix<Real> Sf_m(order, order, Sf.begin() + (2*d+sd)*nnode, false);
              Matrix<Real> St_m(order, order, St.begin() + (2*d+sd)*nnode, false);
              for (Integer i = 0; i < order; i++) for (Integer aa = 0; aa < order; aa++) St_m[aa][i] = Sf_m[i][aa];
            }
          }
        }

        { // Coordinates of the sub-rectangles, relative to the target
          ScratchBuf<Real> Av(2*COORD_DIM*nnode, pool);
          const SmallGEMM<Real, COORD_DIM*order, order, order> sub_v;
          for (Integer sdv = 0; sdv < 2; sdv++) {
            if (!(slen[1][sdv] > 0)) continue;
            const Matrix<Real> cs_all(COORD_DIM*order, order, cs.begin(), false);
            const Matrix<Real> Sf_v(order, order, Sf.begin() + (2+sdv)*nnode, false);
            Matrix<Real> A_all(COORD_DIM*order, order, Av.begin() + sdv*COORD_DIM*nnode, false);
            sub_v(A_all, cs_all, Sf_v);
          }
          const SmallGEMM<Real, order, order, order> sub_u;
          for (Integer sdu = 0; sdu < 2; sdu++) {
            if (!(slen[0][sdu] > 0)) continue;
            const Matrix<Real> St_u(order, order, St.begin() + sdu*nnode, false);
            for (Integer sdv = 0; sdv < 2; sdv++) {
              if (!(slen[1][sdv] > 0)) continue;
              for (Integer k = 0; k < COORD_DIM; k++) {
                const Matrix<Real> A_k(order, order, Av.begin() + (sdv*COORD_DIM + k)*nnode, false);
                Matrix<Real> X_k(order, order, Xsub.begin() + ((2*sdu+sdv)*COORD_DIM + k)*nnode, false);
                sub_u(X_k, St_u, A_k);
              }
            }
          }
        }
      }

      /** Returns the coordinates of sub-rectangle (sdu, sdv) at its nodes, relative to the target */
      Vector<Real> SubCoord(const Integer sdu, const Integer sdv) {
        const Integer nsub = COORD_DIM*order*order;
        return Vector<Real>(nsub, Xsub.begin() + (2*sdu+sdv)*nsub, false);
      }

      /** adds to M_acc (nnode x C) the values acc (C x nnode, component-major) at the nodes of sub-rectangle (sdu, sdv), interpolated to the nodes of the element */
      void AddToElem(Matrix<Real>& M_acc, const Vector<Real>& acc, const Integer C, const Integer sdu, const Integer sdv, ScratchPool& pool) {
        const Integer nnode = order*order;
        ScratchBuf<Real> accB(C*nnode, pool), accE(nnode, pool);
        const Matrix<Real> St_v(order, order, St.begin() + (2+sdv)*nnode, false);
        const Matrix<Real> Sf_u(order, order, Sf.begin() + sdu*nnode, false);
        const Matrix<Real> A_all(C*order, order, (Iterator<Real>)acc.begin(), false);
        Matrix<Real> B_all(C*order, order, accB.begin(), false);
        const SmallGEMM<Real, DynamicSize, order, order> map_v(false, C*order, order, order);
        map_v(B_all, A_all, St_v);
        const SmallGEMM<Real, order, order, order> map_u;
        for (Integer c = 0; c < C; c++) {
          const Matrix<Real> B_c(order, order, accB.begin() + c*nnode, false);
          Matrix<Real> E_c(order, order, accE.begin(), false);
          map_u(E_c, Sf_u, B_c);
          for (Integer p = 0; p < nnode; p++) M_acc[p][c] += accE[p];
        }
      }
    };

    /** quad_order: Gauss-Legendre order on each piece, or 0 to choose it from digits and the tangent angle at the closest point */
    template <Integer order, class Real, class Kernel> void NearInteracBlockDyadic(Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& dcoord_du, const Vector<Real>& dcoord_dv, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Integer quad_order = 0, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Integer nnode = order*order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full/COORD_DIM : KDIM1full;
      const Integer C = KDIM0*KDIM1_out;
      ScratchPool& pool = ScratchPool::Instance(); // looked up once per call, not once per buffer
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
      M_acc.SetZero();

      NearSplit<order,Real> split(coord, dcoord_du, dcoord_dv, Xtrg, pool);
      const Real spd_u = sqrt<Real>(split.guu);
      const Real spd_v = sqrt<Real>(split.gvv);
      Integer q_near = quad_order;
      if (q_near <= 0) { // at least the order for orthogonal tangents, raised for skewed tangents (fitted to the smallest passing orders, targets on and off the surface), rounded up to even
        const Real s = SinTangentAngle<Real>(split.guu, split.guv, split.gvv);
        const Real q_iso = (Real)CachedQuadParams<Real, QuadParams<Real>>(digits).quad_order;
        const Real q = std::max<Real>(std::max<Real>(q_iso, 4 + (Real)order/2), ((Real)0.875 + (Real)1.3*digits)/pow<Real>(s, (Real)0.875));
        q_near = 2*(Integer)ceil<Real>(std::min<Real>(q, (Real)NearMaxQuadOrder)/2);
      }

      ScratchBuf<Real> acc_buf(C*nnode, pool);
      Vector<Real> acc(acc_buf);
      const Vector<QuadRule1D<Real>>& tab = NearGradeTable<order,Real>(q_near);
      const auto integrate_piece = [&tab, &normal_trg, &ker, &proxy_off, &proxy_w, &acc, &split](const Integer sdu, const Integer sdv, const Integer iu, const Integer iv) {
        const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
        IntegrateTensorRule<order,Real>(acc, split.SubCoord(sdu, sdv), tab[iu], tab[iv], normal_trg, ker, nsign, proxy_off, proxy_w);
      };
      const Real b_ellipse = CachedQuadParams<Real, QuadParams<Real>>(digits).b_ellipse;
      const Real dist = split.dist;
      const auto refine = [&integrate_piece, dist, b_ellipse](const Integer sdu, const Integer sdv, Real hu, Real hv) {
        Integer ku = 0, kv = 0;
        const bool refine_to_max = !(dist > 0) || isinf<Real>(dist) || isnan<Real>(dist);
        constexpr Integer KMAX = MaxNearRefineLvl<Real>-1;
        while ((refine_to_max || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
          if (hu >= hv && ku < KMAX) {
            integrate_piece(sdu, sdv, ku, MaxNearRefineLvl<Real> + kv);
            ku++;
            hu *= (Real)0.5;
          } else if (kv < KMAX) {
            integrate_piece(sdu, sdv, MaxNearRefineLvl<Real> + ku, kv);
            kv++;
            hv *= (Real)0.5;
          } else {
            integrate_piece(sdu, sdv, ku, MaxNearRefineLvl<Real> + kv);
            ku++;
            hu *= (Real)0.5;
          }
        }
        integrate_piece(sdu, sdv, MaxNearRefineLvl<Real> + ku, MaxNearRefineLvl<Real> + kv);
      };
      for (Integer sdu = 0; sdu < 2; sdu++) { // Integrate each sub-rectangle, refining toward the closest point
        if (!(split.slen[0][sdu] > 0)) continue;
        for (Integer sdv = 0; sdv < 2; sdv++) {
          if (!(split.slen[1][sdv] > 0)) continue;
          acc.SetZero();
          refine(sdu, sdv, split.slen[0][sdu]*spd_u, split.slen[1][sdv]*spd_v);
          split.AddToElem(M_acc, acc, C, sdu, sdv, pool);
        }
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracDyadic(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits) {
      const auto near_interac_one_trg = [&ker, digits](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& dcoord_du, const Vector<Real>& dcoord_dv, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
        NearInteracBlockDyadic<order,Real>(M_acc, coord, dcoord_du, dcoord_dv, Xtrg, ntrg, ker, digits);
      };
      NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel, elem_idx, near_interac_one_trg);
    }

  }

  namespace detail_tensorprod_near {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxNearRefineLvl;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::IntegrateTensorRule;
    using detail_quadelem::NearInteracTargets;
    using detail_quadelem::SinTangentAngle;
    using detail_dyadic_near::NearMaxQuadOrder;
    using detail_dyadic_near::NearSegmentRule;
    using detail_dyadic_near::NearSplit;

    template <Integer order, class Real, class Kernel> void NearInteracTensorProduct(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits) {
      const auto near_interac_one_trg = [&ker, digits](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& dcoord_du, const Vector<Real>& dcoord_dv, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1full = Kernel::TrgDim();
        constexpr Integer MaxSegments = 4096;
        const Integer nnode = order*order;
        const Integer C = KDIM0*((ntrg.Dim() > 0) ? KDIM1full/COORD_DIM : KDIM1full);
        ScratchPool& pool = ScratchPool::Instance(); // looked up once per call, not once per buffer
        if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
        M_acc.SetZero();

        NearSplit<order,Real> split(coord, dcoord_du, dcoord_dv, Xtrg, pool);
        const Real rho = (Real)2.5;
        const Real b_ellipse = (rho + 1/rho)/4;
        Integer quad_order;
        Real w_stop;
        { // Gauss-Legendre order for the tangents at the closest point, and the smallest segment in parameter units
          const Real L_phys = std::max<Real>(sqrt<Real>(split.guu), sqrt<Real>(split.gvv));
          const bool degenerate = !(split.dist > 0) || isinf<Real>(split.dist) || isnan<Real>(split.dist) || !(L_phys > 0);
          const Real h_param = (degenerate ? 0 : split.dist/L_phys);
          const Real w_floor = pow<Real>((Real)0.5, MaxNearRefineLvl<Real>);
          w_stop = std::max<Real>(h_param/b_ellipse, w_floor);

          // at least digits + 1, the order for orthogonal tangents and targets off the surface, raised for skewed tangents (fitted to the smallest passing orders)
          const Real s = SinTangentAngle<Real>(split.guu, split.guv, split.gvv);
          const Real q = std::max<Real>(std::max<Real>((Real)digits + 1, (Real)digits - 8 + (Real)order/2), (1 + (Real)0.75*digits)/sqrt<Real>(s));
          quad_order = (Integer)ceil<Real>(std::min<Real>(q, (Real)NearMaxQuadOrder));
        }

        Integer nseg[2][2] = {{0, 0}, {0, 0}};
        ScratchBuf<Real> seg_buf(4*2*MaxSegments, pool);
        { // Segments of each side, offsets from the closest point in units of the side's length, graded geometrically toward it down to w_stop in parameter units
          const Real r = std::min<Real>((Real)0.9, b_ellipse/(1 + b_ellipse) * (Real)1.05);
          for (Integer d = 0; d < 2; d++) {
            for (Integer sd = 0; sd < 2; sd++) {
              const Real span = split.slen[d][sd];
              if (!(span > 0)) continue;
              const Iterator<Real> seg = seg_buf.begin() + (2*d+sd)*2*MaxSegments;
              Integer n = 0;
              Real w = 1;
              while (w*span > w_stop) {
                SCTL_ASSERT(n + 1 < MaxSegments);
                seg[2*n+0] = w*r;
                seg[2*n+1] = w;
                w *= r;
                n++;
              }
              seg[2*n+0] = 0;
              seg[2*n+1] = w;
              nseg[d][sd] = n + 1;
            }
          }
        }

        ScratchBuf<Real> rule_buf((nseg[0][0] + nseg[0][1] + nseg[1][0] + nseg[1][1])*quad_order*(1 + 4*order), pool);
        const auto rule_view = [&rule_buf, &nseg, quad_order](const Integer d, const Integer sd) { // the rule of side (d, sd) in rule_buf, after those of the sides before it
          Integer offset = 0;
          for (Integer i = 0; i < 2*d+sd; i++) offset += nseg[i/2][i%2]*quad_order*(1 + 4*order);
          const Integer N = nseg[d][sd]*quad_order;
          const Iterator<Real> buf = rule_buf.begin() + offset;
          return QuadRule1D<Real>{Vector<Real>(N, buf, false), Matrix<Real>(order, N, buf + N, false), Matrix<Real>(order, N, buf + N*(1 + order), false), Matrix<Real>(2*N, order, buf + N*(1 + 2*order), false)};
        };
        QuadRule1D<Real> rule[2][2] = {{rule_view(0, 0), rule_view(0, 1)}, {rule_view(1, 0), rule_view(1, 1)}};
        { // Gauss-Legendre rule on the segments of each side, with its interpolation matrices from the sub-rectangle's nodes
          const Vector<Real>& gl_nds = LegQuadRule<Real>::template nds<NearMaxQuadOrder>(quad_order);
          const Vector<Real>& gl_wts = LegQuadRule<Real>::template wts<NearMaxQuadOrder>(quad_order);
          for (Integer d = 0; d < 2; d++) {
            for (Integer sd = 0; sd < 2; sd++) {
              if (nseg[d][sd]) NearSegmentRule<order>(rule[d][sd], (ConstIterator<Real>)seg_buf.begin() + (2*d+sd)*2*MaxSegments, nseg[d][sd], gl_nds, gl_wts);
            }
          }
        }

        ScratchBuf<Real> acc_buf(C*nnode, pool);
        Vector<Real> acc(acc_buf);
        for (Integer sdu = 0; sdu < 2; sdu++) { // Integrate each sub-rectangle
          if (!nseg[0][sdu]) continue;
          for (Integer sdv = 0; sdv < 2; sdv++) {
            if (!nseg[1][sdv]) continue;
            const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
            acc.SetZero();
            IntegrateTensorRule<order,Real>(acc, split.SubCoord(sdu, sdv), rule[0][sdu], rule[1][sdv], ntrg, ker, nsign);
            split.AddToElem(M_acc, acc, C, sdu, sdv, pool);
          }
        }
      };
      NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel, elem_idx, near_interac_one_trg);
    }

  }

  namespace detail_duffy {

    using detail_quadelem::COORD_DIM;

    using detail_quadelem::DiffMat;
    using detail_quadelem::SelfInteracElems;
    using detail_quadelem::ShiftedElemCoord;
    using detail_quadelem::WeightedKernel;

    template <class Real> struct DuffyTri {
      bool swap_ab = false;
      Real nsign = 1;
      Real J0 = 0;
      Integer alpha = 0, beta = 0; // indices of the interpolation matrices along alpha and beta in DuffySelfTable
    };
    /** Interpolation along alpha (the triangle's edge), with its derivative, at each s-node, and the transposes without it */
    template <class Real> struct DuffyAlpha {
      Vector<Matrix<Real>> interp, interp_T;
    };
    /** Interpolation along beta (toward the edge), with its derivative, at the s-nodes, and the transpose without it */
    template <class Real> struct DuffyBeta {
      Matrix<Real> interp, interp_T;
    };
    template <class Real> struct DuffySelfTable {
      Integer ns = 0;
      Vector<Real> sn, sw;
      std::vector<DuffyTri<Real>> tri;
      std::vector<DuffyAlpha<Real>> alpha;
      std::vector<DuffyBeta<Real>> beta;
    };

    /** Returns, for each order, a (2 + ceil(order/2))-point radial rule and, for each of the 4*order^2 (node, triangle) pairs, its Jacobian, orientation and the indices of its interpolation matrices along alpha and beta. Those depend on the triangle and one index of the node only, and are stored once for each: 4*order of each kind. */
    template <Integer order, class Real> const DuffySelfTable<Real>& DuffyTable() {
      static const DuffySelfTable<Real> table = []() {
        DuffySelfTable<Real> tbl;
        const Integer qs = 2 + (order + 1)/2; // fitted to the smallest radial orders meeting the tolerance, at any digits
        tbl.ns = qs;
        LegQuadRule<Real>::ComputeNdsWts(&tbl.sn, &tbl.sw, qs);

        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Matrix<Real>& D = DiffMat<Real>(order);
        tbl.tri.resize((size_t)(4*order*order));
        tbl.alpha.resize((size_t)(4*order));
        tbl.beta.resize((size_t)(4*order));
        const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
        for (Integer ti = 0; ti < order; ti++) for (Integer tj = 0; tj < order; tj++) {
          const Real u0 = nds[ti], v0 = nds[tj];
          for (Integer kt = 0; kt < 4; kt++) { // Triangle joining node (ti, tj) to edge kt
            DuffyTri<Real>& T = tbl.tri[(size_t)((ti*order + tj)*4 + kt)];
            const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
            const Real b[2] = {cu[(kt+1)%4]-u0, cv[(kt+1)%4]-v0};
            const Real e[2] = {b[0]-a[0], b[1]-a[1]};
            T.J0 = a[0]*b[1] - a[1]*b[0];
            SCTL_ASSERT_MSG(T.J0 > 0, "Duffy triangle orientation");
            T.swap_ab = (fabs<Real>(e[0]) < fabs<Real>(e[1]));
            T.nsign = (T.swap_ab ? (Real)-1 : (Real)1);
            const Real alpha0 = (T.swap_ab ? v0 : u0), beta0 = (T.swap_ab ? u0 : v0);
            const Real a_alpha = (T.swap_ab ? a[1] : a[0]), a_beta = (T.swap_ab ? a[0] : a[1]);
            const Real e_alpha = (T.swap_ab ? e[1] : e[0]);
            const Integer i_alpha = (T.swap_ab ? tj : ti), i_beta = (T.swap_ab ? ti : tj); // the index of the node each depends on
            T.alpha = kt*order + i_alpha;
            T.beta = kt*order + i_beta;
            if (i_alpha == 0) { // Interpolation along beta, with its derivative, at the s-nodes, at the first node with this i_beta
              DuffyBeta<Real>& B = tbl.beta[(size_t)T.beta];
              Vector<Real> beta_vals(qs);
              for (Integer i = 0; i < qs; i++) beta_vals[i] = beta0 + tbl.sn[i]*a_beta;
              Matrix<Real> Mbeta(order, qs), dMbeta(order, qs);
              Vector<Real> Mbeta_v(order*qs, Mbeta.begin(), false);
              LagrangeInterp<Real>::Interpolate(Mbeta_v, nds, beta_vals);
              Matrix<Real>::GEMM(dMbeta, D, Mbeta);
              B.interp.ReInit(order, 2*qs);
              for (Integer r = 0; r < order; r++) for (Integer i = 0; i < qs; i++) {
                B.interp[r][i] = Mbeta[r][i];
                B.interp[r][qs+i] = dMbeta[r][i];
              }
              B.interp_T = Mbeta.Transpose();
            }
            if (i_beta == 0) { // Interpolation along alpha, with its derivative, at each s-node, at the first node with this i_alpha
              DuffyAlpha<Real>& A = tbl.alpha[(size_t)T.alpha];
              A.interp.ReInit(qs);
              A.interp_T.ReInit(qs);
              Vector<Real> alpha_vals(order);
              Matrix<Real> Malpha(order, order), dMalpha(order, order);
              Vector<Real> Malpha_v(order*order, Malpha.begin(), false);
              for (Integer i = 0; i < qs; i++) {
                for (Integer k = 0; k < order; k++) alpha_vals[k] = alpha0 + tbl.sn[i]*(a_alpha + nds[k]*e_alpha);
                LagrangeInterp<Real>::Interpolate(Malpha_v, nds, alpha_vals);
                Matrix<Real>::GEMM(dMalpha, D, Malpha);
                A.interp[i].ReInit(order, 2*order);
                for (Integer r = 0; r < order; r++) for (Integer k = 0; k < order; k++) {
                  A.interp[i][r][k] = Malpha[r][k];
                  A.interp[i][r][order+k] = dMalpha[r][k];
                }
                A.interp_T[i] = Malpha.Transpose();
              }
            }
          }
        }
        return tbl;
      }();
      return table;
    }

    template <Integer order, class Real, class Kernel> void SelfInteracBlockDuffy(Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xnnodes, const Integer ti, const Integer tj, const bool trg_dot_prod, const Kernel& ker, const Integer digits) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(coord.Dim() == COORD_DIM*order*order && Xnnodes.Dim() == coord.Dim());
      const Integer nnode = order*order;
      const Integer t = ti*order + tj;
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full/COORD_DIM : KDIM1full;
      const Integer C = KDIM0*KDIM1_out;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
      M_acc.SetZero();

      const DuffySelfTable<Real>& tbl = DuffyTable<order,Real>();
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      ScratchBuf<Real> cs_buf(COORD_DIM*nnode);
      Vector<Real> cs(cs_buf);
      { // Coordinates relative to the target node
        StaticArray<Real,COORD_DIM> Xtrg_buf;
        for (Integer k = 0; k < COORD_DIM; k++) Xtrg_buf[k] = coord[k*nnode + t];
        const Vector<Real> Xtrg(COORD_DIM, Xtrg_buf, false);
        ShiftedElemCoord(cs, coord, Xtrg);
      }

      Real G[4];
      { // Metric of the element at the target node
        const Matrix<Real>& D = DiffMat<Real>(order);
        Real du[COORD_DIM], dv[COORD_DIM];
        for (Integer k = 0; k < COORD_DIM; k++) {
          Real su = 0, sv = 0;
          for (Integer i = 0; i < order; i++) su += cs[k*nnode + i*order + tj]*D[i][ti];
          for (Integer j = 0; j < order; j++) sv += cs[k*nnode + ti*order + j]*D[j][tj];
          du[k] = su;
          dv[k] = sv;
        }
        Real guu = 0, guv = 0, gvv = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          guu += du[k]*du[k];
          guv += du[k]*dv[k];
          gvv += dv[k]*dv[k];
        }
        G[0] = guu;
        G[1] = guv;
        G[2] = guv;
        G[3] = gvv;
      }

      const Integer ns = tbl.ns;
      static constexpr Integer MaxGLOrder = 128; // the largest angular rule
      const Integer nt = std::min<Integer>(MaxGLOrder, std::max<Integer>(order/2, (Integer)ceil<Real>(KDIM0 > 1 ? (Real)4.75*digits - (Real)9.5 + (Real)0.625*order : (Real)3*digits - 6 + (Real)order/4))); // fitted to the smallest angular orders meeting the tolerance
      const Integer nq = ns*nt;
      ScratchBuf<Real> csT(COORD_DIM*nnode);
      { // cs with u and v exchanged, for the triangles with swap_ab
        for (Integer k = 0; k < COORD_DIM; k++) {
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) csT[k*nnode + j*order + i] = cs[k*nnode + i*order + j];
          }
        }
      }
      for (Integer kt = 0; kt < 4; kt++) { // Triangles joining the target node to each edge
        const DuffyTri<Real>& T = tbl.tri[(size_t)((ti*order + tj)*4 + kt)];
        const DuffyAlpha<Real>& Talpha = tbl.alpha[(size_t)T.alpha];
        const DuffyBeta<Real>& Tbeta = tbl.beta[(size_t)T.beta];

        Real tstar, dOverL;
        { // Closest edge point, and its distance over edge length
          const Real u0 = nds[ti], v0 = nds[tj];
          const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
          const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
          const Real e[2] = {cu[(kt+1)%4]-cu[kt], cv[(kt+1)%4]-cv[kt]};
          const Real Me[2] = {G[0]*e[0]+G[1]*e[1], G[2]*e[0]+G[3]*e[1]};
          const Real am = e[0]*Me[0] + e[1]*Me[1];
          Real ts = -(a[0]*Me[0] + a[1]*Me[1])/am;
          ts = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
          const Real c[2] = {a[0]+ts*e[0], a[1]+ts*e[1]};
          const Real d2 = c[0]*(G[0]*c[0]+G[1]*c[1]) + c[1]*(G[2]*c[0]+G[3]*c[1]);
          tstar = ts;
          dOverL = sqrt<Real>(d2)/sqrt<Real>(am);
        }

        ScratchBuf<Real> tw(nt), Tt_buf(order*nt), TtT_buf(nt*order);
        { // Rule along t, graded toward the closest edge point
          const Vector<Real>& qn = LegQuadRule<Real>::template nds<MaxGLOrder>(nt);
          const Vector<Real>& qw = LegQuadRule<Real>::template wts<MaxGLOrder>(nt);
          const auto arcsinh = [](const Real x) { return log<Real>(x + sqrt<Real>(x*x + (Real)1)); };
          ScratchBuf<Real> tn_buf(nt);
          Vector<Real> tn(tn_buf);
          const Real x0 = -arcsinh(tstar/dOverL), x1 = arcsinh(((Real)1-tstar)/dOverL);
          for (Integer i = 0; i < nt; i++) {
            const Real xi = x0 + (x1-x0)*qn[i];
            const Real ex = exp<Real>(xi), iex = (Real)1/ex;
            tn[i] = tstar + dOverL*(ex-iex)/(Real)2;
            tw[i] = dOverL*(ex+iex)/(Real)2*(x1-x0)*qw[i];
          }
          Vector<Real> Tt_v(Tt_buf);
          LagrangeInterp<Real>::Interpolate(Tt_v, nds, tn);
          for (Integer r = 0; r < order; r++) for (Integer j = 0; j < nt; j++) TtT_buf[j*order + r] = Tt_buf[r*nt + j];
        }
        const Matrix<Real> Tt(order, nt, Tt_buf.begin(), false);
        const Matrix<Real> TtT(nt, order, TtT_buf.begin(), false);

        ScratchBuf<Real> Xs(COORD_DIM*nq), Xn(COORD_DIM*nq), wq(nq);
        { // Points, normals and weights of the triangle's rule
          constexpr Integer NR = 3*COORD_DIM;
          ScratchBuf<Real> XdX_buf(ns*NR*nt);
          Matrix<Real> XdX(ns*NR, nt, XdX_buf.begin(), false);
          { // Interpolate positions and tangents to the rule's points
            constexpr Integer NA = 2*COORD_DIM;
            ScratchBuf<Real> Gm_buf(2*COORD_DIM*order*ns);
            ScratchBuf<Real> As_buf(NA*order), Tmp_buf(2*NA*order), HG_buf(ns*NR*order);
            const Matrix<Real> FS(COORD_DIM*order, order, (T.swap_ab ? csT.begin() : cs.begin()), false);
            Matrix<Real> Gm(COORD_DIM*order, 2*ns, Gm_buf.begin(), false);
            Matrix<Real> As(NA, order, As_buf.begin(), false);
            Matrix<Real> Tmp(NA, 2*order, Tmp_buf.begin(), false);
            Matrix<Real> HG(ns*NR, order, HG_buf.begin(), false);
            const SmallGEMM<Real, COORD_DIM*order, DynamicSize, order> interp_beta(false, COORD_DIM*order, 2*ns, order);
            interp_beta(Gm, FS, Tbeta.interp);
            const SmallGEMM<Real, NA, 2*order, order> interp_alpha;
            for (Integer i = 0; i < ns; i++) {
              for (Integer k = 0; k < COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
                As[k][m] = Gm[k*order+m][i];
                As[COORD_DIM+k][m] = Gm[k*order+m][ns+i];
              }
              interp_alpha(Tmp, As, Talpha.interp[i]);
              for (Integer k = 0; k < COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
                HG[i*NR + k][m]               = Tmp[k][m];
                HG[i*NR + COORD_DIM + k][m]   = Tmp[k][order+m];
                HG[i*NR + 2*COORD_DIM + k][m] = Tmp[COORD_DIM+k][m];
              }
            }
            const SmallGEMM<Real, DynamicSize, DynamicSize, order> interp_t(false, ns*NR, nt, order);
            interp_t(XdX, HG, Tt);
          }
          for (Integer i = 0; i < ns; i++) {
            const Real jw = tbl.sn[i]*T.J0*tbl.sw[i];
            for (Integer j = 0; j < nt; j++) {
              const Integer q = i*nt + j;
              const Real a0 = XdX[i*NR+COORD_DIM+0][j], a1 = XdX[i*NR+COORD_DIM+1][j], a2 = XdX[i*NR+COORD_DIM+2][j];
              const Real b0 = XdX[i*NR+2*COORD_DIM+0][j], b1 = XdX[i*NR+2*COORD_DIM+1][j], b2 = XdX[i*NR+2*COORD_DIM+2][j];
              const Real n0 = T.nsign*(a1*b2-a2*b1), n1 = T.nsign*(a2*b0-a0*b2), n2 = T.nsign*(a0*b1-a1*b0);
              const Real ar = sqrt<Real>(n0*n0+n1*n1+n2*n2), ia = (ar > 0 ? (Real)1/ar : (Real)0);
              for (Integer k = 0; k < COORD_DIM; k++) Xs[k*nq+q] = XdX[i*NR+k][j];
              Xn[0*nq+q] = n0*ia;
              Xn[1*nq+q] = n1*ia;
              Xn[2*nq+q] = n2*ia;
              wq[q] = ar*jw*tw[j];
            }
          }
        }

        ScratchBuf<Real> KW_buf(C*nq);
        { // Weighted kernel at the rule's points
          StaticArray<Real,COORD_DIM> Xt0{0,0,0};
          const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);
          const Vector<Real> normal_trg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)Xnnodes.begin() + t*COORD_DIM : NullIterator<Real>()), false);
          WeightedKernel<Real>(KW_buf.begin(), Xt0_v.begin(), Xs.begin(), Xn.begin(), wq.begin(), ns*nt, nt, ns*nt, nt, (Real)1, false, normal_trg, ker);
        }

        { // Project onto the element nodes and add into M_acc
          ScratchBuf<Real> Zall_buf(ns*C*order), Yi_buf(C*order), Yall_buf(C*order*ns), Pc_buf(nnode);
          const Matrix<Real> KW(ns*C, nt, KW_buf.begin(), false);
          Matrix<Real> Zall(ns*C, order, Zall_buf.begin(), false);
          Matrix<Real> Yi(C, order, Yi_buf.begin(), false);
          Matrix<Real> Yall(C*order, ns, Yall_buf.begin(), false);
          Matrix<Real> Pc(order, order, Pc_buf.begin(), false);
          const SmallGEMM<Real, DynamicSize, order, DynamicSize> proj_t(false, ns*C, order, nt);
          proj_t(Zall, KW, TtT);
          const SmallGEMM<Real, DynamicSize, order, order> proj_alpha(false, C, order, order);
          for (Integer i = 0; i < ns; i++) {
            const Matrix<Real> Zi(C, order, (Iterator<Real>)Zall.begin() + i*C*order, false);
            proj_alpha(Yi, Zi, Talpha.interp_T[i]);
            for (Integer c = 0; c < C; c++) for (Integer m = 0; m < order; m++) Yall[c*order+m][i] = Yi[c][m];
          }
          const SmallGEMM<Real, order, order, DynamicSize> proj_beta(false, order, order, ns);
          for (Integer c = 0; c < C; c++) {
            const Matrix<Real> Yc(order, ns, (Iterator<Real>)Yall.begin() + c*order*ns, false);
            proj_beta(Pc, Yc, Tbeta.interp_T);
            for (Integer m = 0; m < order; m++) for (Integer n = 0; n < order; n++)
              M_acc[T.swap_ab ? n*order+m : m*order+n][c] += Pc[m][n];
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void SelfInteracDuffy(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      DuffyTable<order,Real>(); // precomp cache
      const auto self_interac_one_trg = [&ker, digits, trg_dot_prod](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xnnodes, const Vector<Real>&, const Vector<Real>&, const Integer ti, const Integer tj) {
        SelfInteracBlockDuffy<order,Real>(M_acc, coord, Xnnodes, ti, tj, trg_dot_prod, ker, digits);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, self_interac_one_trg);
    }

  }

  namespace detail_hedgehog {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::PrecompReal;
    using detail_dyadic_near::NearMaxQuadOrder;

    using detail_quadelem::SelfInteracElems;
    using detail_quadelem::SinTangentAngle;
    using detail_dyadic_near::NearInteracBlockDyadic;

    template <class Kernel, class = void> struct KernelSingularOrder {
      static constexpr Integer value = 2;
    };
    template <class Kernel> struct KernelSingularOrder<Kernel, std::void_t<decltype(Kernel::SingularOrder())>> {
      static constexpr Integer value = Kernel::SingularOrder();
    };

    template <Integer order, class Real, class Kernel> void SelfInteracHedgehog(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      static constexpr Integer sing_order = KernelSingularOrder<Kernel>::value;
      const Integer near_digits = std::min<Integer>(MaxDigits<Real>-1, digits + (sing_order <= 1 ? 2 : 6));

      static const Vector<Real> proxy_dist = []() { // Proxy distances, in units of rmin
        Vector<Real> v;
        for (Integer j = 0; j < 5; j++) v.PushBack(pow<Real>((Real)4, (Real)j/(Real)4));
        return v;
      }();
      static const Vector<Real> hh_w = []() { // Weights extrapolating the proxy values to distance 0
        using W = PrecompReal;
        const Integer p = (Integer)proxy_dist.Dim();
        Vector<Real> wj(p);
        for (Integer j = 0; j < p; j++) {
          W v = 1;
          for (Integer k = 0; k < p; k++) if (k != j) v *= (0 - (W)proxy_dist[k])/((W)proxy_dist[j] - (W)proxy_dist[k]);
          wj[j] = (Real)v;
        }
        return wj;
      }();
      const Real rmin_coeff = [digits]() {
        const Real c = (Real)0.1 * pow<Real>(pow<Real,Long>((Real)0.1, (Long)digits), (Real)1/(Real)6);
        return (sing_order <= 1 ? c : std::min<Real>(c, (Real)3e-3));
      }();

      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      const auto self_interac_one_trg = [&nds, rmin_coeff, &ker, digits, near_digits, trg_dot_prod](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xnnodes, const Vector<Real>& dXu, const Vector<Real>& dXv, const Integer ti, const Integer tj) {
        const Integer nnode = order*order;
        const Integer t = ti*order + tj;
        ScratchBuf<Real> hh_Xt1_buf(COORD_DIM), hh_off_buf(proxy_dist.Dim()*COORD_DIM);
        Vector<Real> hh_Xt1(hh_Xt1_buf), hh_off(hh_off_buf);
        Integer q_proxy;
        { // Proxy points along the normal, sized by the distance to the nearer edge, and their quadrature order
          Real su2 = 0, sv2 = 0, suv = 0;
          for (Integer k = 0; k < COORD_DIM; k++) {
            su2 += dXu[k*nnode+t]*dXu[k*nnode+t];
            sv2 += dXv[k*nnode+t]*dXv[k*nnode+t];
            suv += dXu[k*nnode+t]*dXv[k*nnode+t];
          }
          const Real s = SinTangentAngle<Real>(su2, suv, sv2);
          { // fitted to the smallest orders meeting the tolerance on skewed elements, rounded up to even
            const Real q = std::max<Real>(4 + (Real)digits + (Real)order/8, ((Real)0.75 + (Real)1.25*digits)/s);
            q_proxy = 2*(Integer)ceil<Real>(std::min<Real>(q, (Real)NearMaxQuadOrder)/2);
          }
          const Real edge_u = std::min<Real>(nds[ti], 1-nds[ti]), edge_v = std::min<Real>(nds[tj], 1-nds[tj]);
          const Real rmin = rmin_coeff * s * std::min<Real>(edge_u*sqrt<Real>(su2), edge_v*sqrt<Real>(sv2));
          for (Integer k = 0; k < COORD_DIM; k++) hh_Xt1[k] = coord[k*nnode + t] + rmin*Xnnodes[t*COORD_DIM+k];
          for (Integer j = 0; j < proxy_dist.Dim(); j++) {
            const Real rj = rmin*proxy_dist[j];
            for (Integer k = 0; k < COORD_DIM; k++) hh_off[j*COORD_DIM+k] = (rj-rmin)*Xnnodes[t*COORD_DIM+k];
          }
        }
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)Xnnodes.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        NearInteracBlockDyadic<order,Real>(M_acc, coord, dXu, dXv, hh_Xt1, ntrg, ker, near_digits, q_proxy, hh_off, hh_w);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, self_interac_one_trg);
    }

  }

  namespace detail_tensorprod_singular {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::MaxRefineLvl;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::DiffMat;
    using detail_quadelem::IntegratePanel;
    using detail_quadelem::SelfInteracElems;
    using detail_quadelem::ShiftedElemCoord;
    using detail_quadelem::SinTangentAngle;

    static constexpr Integer SelfMaxQuadOrder = 60;
    static constexpr Integer SelfMaxLvlV = 12;

    template <Integer order, class Real> void LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MTD, const Vector<Real>& delta, const Integer ti) {
      const Integer N = (Integer)delta.Dim();
      M.ReInit(order, N);
      { // Lagrange basis at the points nds[ti] + delta
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        StaticArray<Real,order> inv_den, off, f;
        for (Integer i = 0; i < order; i++) {
          Real d = 1;
          for (Integer j = 0; j < order; j++) if (j != i) d *= (nds[i]-nds[j]);
          inv_den[i] = 1/d;
        }
        for (Integer j = 0; j < order; j++) off[j] = nds[ti]-nds[j];
        for (Integer a = 0; a < N; a++) {
          for (Integer j = 0; j < order; j++) f[j] = delta[a] + off[j];
          for (Integer i = 0; i < order; i++) {
            Real p = inv_den[i];
            for (Integer j = 0; j < order; j++) if (j != i) p *= f[j];
            M[i][a] = p;
          }
        }
      }
      dM.ReInit(order, N);
      Matrix<Real>::GEMM(dM, DiffMat<Real>(order), M);
      MTD.ReInit(2*N, order);
      for (Integer i = 0; i < order; i++) {
        for (Integer a = 0; a < N; a++) {
          MTD[a][i] = M[i][a];
          MTD[N + a][i] = dM[i][a];
        }
      }
    }

    template <class Real> void BuildCenteredLogSingular1D(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer quad_order) {
      const Integer ord = 16;
      delta.ReInit(0);
      w.ReInit(0);
      const auto add_alpert = [&delta, &w](const Real a, const Real b, const bool log_a, const bool log_b) {
        const auto& L = (log_a ? AlpertQuadRule<Real>::LogCorrection(ord) : AlpertQuadRule<Real>::SmoothCorrection(ord));
        const auto& R = (log_b ? AlpertQuadRule<Real>::LogCorrection(ord) : AlpertQuadRule<Real>::SmoothCorrection(ord));
        const Integer skipL = L.nskip, skipR = R.nskip;
        const Integer N = std::max<Integer>(skipL + skipR + 2, 2 * ord);
        const Integer N1 = N - 1;
        const Real h = (b - a) / (Real)N1;
        for (Integer i = skipL; i <= N1 - skipR; i++) {
          delta.PushBack(a + (Real)i*h);
          w.PushBack(h);
        }
        for (Integer i = 0; i < L.nds.Dim(); i++) {
          delta.PushBack(a + L.nds[i]*h);
          w.PushBack(L.wts[i]*h);
        }
        for (Integer i = 0; i < R.nds.Dim(); i++) {
          delta.PushBack(b - R.nds[i]*h);
          w.PushBack(R.wts[i]*h);
        }
      };
      Vector<Real> gnds, gwts;
      LegQuadRule<Real>::ComputeNdsWts(&gnds, &gwts, quad_order);
      const auto add_gl = [&delta, &w, quad_order, &gnds, &gwts](const Real a, const Real b) {
        const Real len = b - a;
        for (Integer i = 0; i < quad_order; i++) {
          delta.PushBack(a + len*gnds[i]);
          w.PushBack(len*gwts[i]);
        }
      };
      { // Left of v0: halving panels, then an Alpert panel
        const Real Ll = v0;
        Real prev = -Ll;
        for (Integer i = 1; i <= Lvl; i++) {
          const Real bnd = -Ll*pow<Real,Long>((Real)0.5, (Long)i);
          add_gl(prev, bnd);
          prev = bnd;
        }
        add_alpert(prev, (Real)0, false, true);
      }
      { // Right of v0: halving panels, then an Alpert panel
        const Real Lr = (Real)1 - v0;
        Real prev = Lr;
        for (Integer i = 1; i <= Lvl; i++) {
          const Real bnd = Lr*pow<Real,Long>((Real)0.5, (Long)i);
          add_gl(bnd, prev);
          prev = bnd;
        }
        add_alpert((Real)0, prev, true, false);
      }
    }

    /** Returns the rule toward node ti along u for each (order, digits, quad_order), built the first time it is requested: quad_order-point Gauss-Legendre on intervals halving min(MaxRefineLvl, 2*digits+6) times toward the node on each side, with order x N interpolation matrices. */
    template <Integer order, class Real> const QuadRule1D<Real>& CenteredURule(const Integer ti, const Integer digits, const Integer quad_order) {
      const auto build = [](const Integer digits, const Integer quad_order) {
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Integer Lvl = std::min<Integer>(MaxRefineLvl<Real>, 2*digits + 6);
        Vector<Real> qnds, qwts;
        LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, quad_order);
        const auto graded_rule = [Lvl, &qnds, &qwts](Vector<Real>& delta, Vector<Real>& w, const Real u0) {
          const Integer q = (Integer)qnds.Dim();
          const auto side = [&delta, &w, Lvl, q, &qnds, &qwts](const Real span, const Real sgn) {
            Real a = 0;
            for (Integer k = Lvl; k >= 0; k--) {
              const Real b = span * pow<Real>((Real)0.5, (Integer)k);
              const Real len = b - a;
              for (Integer i = 0; i < q; i++) {
                delta.PushBack(sgn*(a + len*qnds[i]));
                w.PushBack(len*qwts[i]);
              }
              a = b;
            }
          };
          side(1-u0, (Real)1);
          side(u0,   (Real)-1);
        };
        Vector<QuadRule1D<Real>> rules(order);
        for (Integer i = 0; i < order; i++) {
          Vector<Real> delta;
          graded_rule(delta, rules[i].w, nds[i]);
          LagrangeAtOffset<order,Real>(rules[i].M, rules[i].dM, rules[i].MTD, delta, i);
        }
        return rules;
      };
      static std::array<std::array<std::once_flag, SelfMaxQuadOrder+1>, MaxDigits<Real>> built;
      static std::array<std::array<Vector<QuadRule1D<Real>>, SelfMaxQuadOrder+1>, MaxDigits<Real>> table;
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real> && quad_order > 0 && quad_order <= SelfMaxQuadOrder);
      std::call_once(built[digits][quad_order], [&build, digits, quad_order]() { table[digits][quad_order] = build(digits, quad_order); });
      return table[digits][quad_order][ti];
    }

    /** Returns the rule toward node tj along v for each (order, Lvl, quad_order), built the first time it is requested: quad_order-point Gauss-Legendre on Lvl panels halving toward the node on each side, then one order-16 Alpert panel per side, log-corrected at the node, with order x N interpolation matrices. */
    template <Integer order, class Real> const QuadRule1D<Real>& CenteredVRule(const Integer tj, const Integer Lvl, const Integer quad_order) {
      const auto build = [](const Integer Lvl, const Integer quad_order) {
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        Vector<QuadRule1D<Real>> rules(order);
        for (Integer j = 0; j < order; j++) {
          Vector<Real> delta;
          BuildCenteredLogSingular1D(delta, rules[j].w, nds[j], Lvl, quad_order);
          LagrangeAtOffset<order,Real>(rules[j].M, rules[j].dM, rules[j].MTD, delta, j);
        }
        return rules;
      };
      static std::array<std::array<std::once_flag, SelfMaxQuadOrder+1>, SelfMaxLvlV+1> built;
      static std::array<std::array<Vector<QuadRule1D<Real>>, SelfMaxQuadOrder+1>, SelfMaxLvlV+1> table;
      SCTL_ASSERT(Lvl > 0 && Lvl <= SelfMaxLvlV && quad_order > 0 && quad_order <= SelfMaxQuadOrder);
      std::call_once(built[Lvl][quad_order], [&build, Lvl, quad_order]() { table[Lvl][quad_order] = build(Lvl, quad_order); });
      return table[Lvl][quad_order][tj];
    }

    template <Integer order, class Real, class Kernel> void SelfInteracTensorProduct(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      const auto self_interac_one_trg = [&ker, digits, trg_dot_prod](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xnnodes, const Vector<Real>& dXu, const Vector<Real>& dXv, const Integer ti, const Integer tj) {
        const Integer nnode = order*order;
        const Integer t = ti*order + tj;
        Integer quad_order, lvl_v;
        { // Gauss-Legendre order and halving levels along v for the tangent angle at the node, fitted to the smallest values meeting the tolerance on skewed elements
          Real guu = 0, guv = 0, gvv = 0;
          for (Integer k = 0; k < COORD_DIM; k++) {
            guu += dXu[k*nnode+t]*dXu[k*nnode+t];
            guv += dXu[k*nnode+t]*dXv[k*nnode+t];
            gvv += dXv[k*nnode+t]*dXv[k*nnode+t];
          }
          const Real s = SinTangentAngle<Real>(guu, guv, gvv);
          const Real q = std::max<Real>((Real)1.5 + (Real)digits/2 + (Real)order/8, ((Real)1.25 + (Real)digits/2)/pow<Real>(s, (Real)0.75));
          quad_order = (Integer)ceil<Real>(std::min<Real>(q, (Real)SelfMaxQuadOrder));
          // capped: levels beyond those needed add rounding error
          const Integer lvl_max = std::min<Integer>(SelfMaxLvlV, std::max<Integer>(7, digits - 4));
          const Real lvl = std::max<Real>((Real)std::max<Integer>(1, digits - 5), (Real)1.5*digits - (Real)6.5 + (Real)2.5*log<Real>(1/s)/log<Real>(2));
          lvl_v = (Integer)ceil<Real>(std::min<Real>(lvl, (Real)lvl_max));
        }
        ScratchBuf<Real> cs_buf(COORD_DIM*nnode);
        Vector<Real> cs(cs_buf);
        { // Coordinates relative to the target node
          StaticArray<Real,COORD_DIM> Xtrg_buf;
          for (Integer k = 0; k < COORD_DIM; k++) Xtrg_buf[k] = coord[k*nnode + t];
          const Vector<Real> Xtrg(COORD_DIM, Xtrg_buf, false);
          ShiftedElemCoord(cs, coord, Xtrg);
        }
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)Xnnodes.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        const QuadRule1D<Real>& ru = CenteredURule<order,Real>(ti, digits, quad_order);
        const QuadRule1D<Real>& rv = CenteredVRule<order,Real>(tj, lvl_v, quad_order);
        IntegratePanel<order,Real>(M_acc, cs, ntrg, ru, rv, ker);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, self_interac_one_trg);
    }

  }

  namespace detail_dispatch {

    using detail_quadelem::Access;
    using detail_quadelem::COORD_DIM;

    using detail_dyadic_near::NearInteracDyadic;
    using detail_tensorprod_near::NearInteracTensorProduct;
    using detail_duffy::SelfInteracDuffy;
    using detail_hedgehog::SelfInteracHedgehog;
    using detail_tensorprod_singular::SelfInteracTensorProduct;
    using detail_quadelem::WeightedKernel;

    /**
     * Returns d such that the kernel, dotted with the target normal if trg_dot_prod, grows like 1/r^d as a
     * source approaches the target along a surface: the target at the origin of a curved model surface,
     * whose normal and curvatures are in no special direction, the sources on it at distance h and 2h in 3
     * directions, each with the surface normal there as its normal; for each kernel entry log2(A(h)/A(2h)),
     * A the largest magnitude over the directions, and d the largest over the entries. A curved surface, as
     * the factor n.r of a double layer vanishes on a plane but is of order r^2 on a curved surface.
     */
    template <class Real, class Kernel> Real SurfaceSingularDegree(const Kernel& ker, const bool trg_dot_prod, const Real h) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1 = Kernel::TrgDim();
      constexpr Integer ndir = 3;
      constexpr Integer nq = 2*ndir;
      const Integer C = KDIM0*(trg_dot_prod ? KDIM1/COORD_DIM : KDIM1);
      const Real ang[ndir] = {(Real)0.3, (Real)2.4, (Real)4.1};
      const Real k11 = (Real)0.7, k12 = (Real)0.3, k22 = (Real)-0.4;

      StaticArray<Real,COORD_DIM> n{(Real)0.36, (Real)-0.48, (Real)0.8}, a, b;
      { // Tangents a, b completing n
        const Real seed[COORD_DIM] = {(Real)0.6, (Real)0.7, (Real)0.2};
        const Real sn = seed[0]*n[0] + seed[1]*n[1] + seed[2]*n[2];
        for (Integer k = 0; k < COORD_DIM; k++) a[k] = seed[k] - sn*n[k];
        const Real na = sqrt<Real>(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        for (Integer k = 0; k < COORD_DIM; k++) a[k] /= na;
        b[0] = n[1]*a[2] - n[2]*a[1];
        b[1] = n[2]*a[0] - n[0]*a[2];
        b[2] = n[0]*a[1] - n[1]*a[0];
      }

      ScratchBuf<Real> Xs(COORD_DIM*nq), Xn(COORD_DIM*nq), wq(nq);
      for (Integer s = 0; s < 2; s++) { // Sources at distance h and 2h, with the normals of the model surface there
        for (Integer i = 0; i < ndir; i++) {
          const Integer q = s*ndir + i;
          const Real x1 = (s+1)*h*cos<Real>(ang[i]), x2 = (s+1)*h*sin<Real>(ang[i]);
          const Real qn = (k11*x1*x1 + 2*k12*x1*x2 + k22*x2*x2)/2;
          const Real pu = k11*x1 + k12*x2, pv = k12*x1 + k22*x2;
          Real tu[COORD_DIM], tv[COORD_DIM], ns[COORD_DIM];
          for (Integer k = 0; k < COORD_DIM; k++) {
            Xs[k*nq + q] = x1*a[k] + x2*b[k] + qn*n[k];
            tu[k] = a[k] + pu*n[k];
            tv[k] = b[k] + pv*n[k];
          }
          ns[0] = tu[1]*tv[2] - tu[2]*tv[1];
          ns[1] = tu[2]*tv[0] - tu[0]*tv[2];
          ns[2] = tu[0]*tv[1] - tu[1]*tv[0];
          const Real nn = sqrt<Real>(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
          for (Integer k = 0; k < COORD_DIM; k++) Xn[k*nq + q] = ns[k]/nn;
          wq[q] = 1;
        }
      }

      ScratchBuf<Real> K(C*nq);
      {
        StaticArray<Real,COORD_DIM> origin{(Real)0, (Real)0, (Real)0};
        const Vector<Real> normal_trg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)n : NullIterator<Real>()), false);
        WeightedKernel<Real>(K.begin(), (ConstIterator<Real>)origin, Xs.begin(), Xn.begin(), wq.begin(), nq, nq, nq, nq, (Real)1, false, normal_trg, ker);
      }

      Real Kmax = 0, d = -1;
      for (Integer i = 0; i < C*nq; i++) Kmax = std::max<Real>(Kmax, fabs(K[i]));
      for (Integer c = 0; c < C; c++) {
        Real A1 = 0, A2 = 0;
        for (Integer i = 0; i < ndir; i++) {
          A1 = std::max<Real>(A1, fabs(K[c*nq + i]));
          A2 = std::max<Real>(A2, fabs(K[c*nq + ndir + i]));
        }
        if (std::max<Real>(A1, A2) > Kmax*(Real)1e-12) d = std::max<Real>(d, log<Real>(A1/A2)/log<Real>((Real)2)); // entries that vanish are skipped
      }
      return d;
    }

    /** prints a warning, once per process for each kernel and trg_dot_prod and only on rank 0 of MPI_COMM_WORLD, when the self-interaction rule of the list's scheme (TensorProduct or Duffy) is wrong for the kernel: SurfaceSingularDegree above 1.5, with h 1e-4 times the size of the first element */
    template <class Real, class Kernel> void WarnStronglySingular(const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel) {
      const auto scheme = Access<Real>::Scheme(qel);
      if (scheme == QuadElemList<Real>::QuadScheme::Hedgehog) return;
      Real size = 1;
      if (qel.Size()) { // diagonal of the bounding box of the first element's nodes
        const Integer nnode = qel.Order()*qel.Order();
        const Vector<Real>& coord = Access<Real>::Coord(qel);
        Real diag2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          Real lo = coord[k*nnode], hi = coord[k*nnode];
          for (Integer p = 1; p < nnode; p++) {
            lo = std::min<Real>(lo, coord[k*nnode + p]);
            hi = std::max<Real>(hi, coord[k*nnode + p]);
          }
          diag2 += (hi - lo)*(hi - lo);
        }
        size = sqrt<Real>(diag2);
      }
      const Real d = SurfaceSingularDegree<Real>(ker, trg_dot_prod, (Real)1e-4*size);
      if (!(d > (Real)1.5)) return;
#ifdef SCTL_HAVE_MPI
      { // rank in MPI_COMM_WORLD, without Comm::World(), whose MPI_Comm_dup is collective
        int rank = 0;
        if (comm_detail::MPIIsActive()) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0) return;
      }
#endif
      static std::array<std::once_flag, 2> warned;
      std::call_once(warned[trg_dot_prod ? 1 : 0], [&ker, trg_dot_prod, scheme, d]() {
        SCTL_WARN("QuadElemList: the self-interaction integrand of " << ker.Name() << (trg_dot_prod ? " dotted with the target normal" : "") << " grows like 1/r^" << std::lround((double)d) << " along the surface, but the " << (scheme == QuadElemList<Real>::QuadScheme::Duffy ? "Duffy" : "TensorProduct") << " rule is accurate only up to 1/r, so the self-interactions are wrong; use QuadScheme::Hedgehog.");
      });
    }

    template <Integer order, class Real, class Kernel> void SelfInteracDispatch(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      switch (Access<Real>::Scheme(qel)) {
        case QuadElemList<Real>::QuadScheme::TensorProduct: return SelfInteracTensorProduct<order,Real>(M_lst, ker, trg_dot_prod, qel, digits);
        case QuadElemList<Real>::QuadScheme::Duffy:         return SelfInteracDuffy<order,Real>(M_lst, ker, trg_dot_prod, qel, digits);
        case QuadElemList<Real>::QuadScheme::Hedgehog:      return SelfInteracHedgehog<order,Real>(M_lst, ker, trg_dot_prod, qel, digits);
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracDispatch(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits) {
      if (Access<Real>::Scheme(qel) == QuadElemList<Real>::QuadScheme::TensorProduct) {
        NearInteracTensorProduct<order,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      } else {
        NearInteracDyadic<order,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      }
    }

  }

  template <class Real> template <class ValueType> QuadElemList<Real>::QuadElemList(const Integer order0, const Vector<ValueType>& coord0) {
    Init(order0, coord0);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Init(const Integer order0, const Vector<ValueType>& coord0) {
    order = order0;
    SCTL_ASSERT(order > 0);
    const Integer nnode_per_elem = order * order;
    const Integer elem_stride = detail_quadelem::COORD_DIM * nnode_per_elem;
    SCTL_ASSERT(coord0.Dim() % elem_stride == 0);
    nelem = coord0.Dim() / elem_stride;

    { // Store the coordinates component by component per element
      coord.ReInit(nelem * elem_stride);
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * elem_stride;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          for (Integer p = 0; p < nnode_per_elem; p++) {
            coord[base + k * nnode_per_elem + p] = (Real)coord0[(elem_idx * nnode_per_elem + p) * detail_quadelem::COORD_DIM + k];
          }
        }
      }
    }
    { // Differentiate the coordinates along u and v
      dcoord_du.ReInit(coord.Dim());
      dcoord_dv.ReInit(coord.Dim());
      const auto transpose_blocks = [order = order, nnode_per_elem](Vector<Real>& out, const Vector<Real>& in) {
        for (Integer k = 0; k < in.Dim() / nnode_per_elem; k++) {
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) out[k * nnode_per_elem + j * order + i] = in[k * nnode_per_elem + i * order + j];
          }
        }
      };

      const auto& nodes = ParamNodes(order);
      const Long nblk = detail_quadelem::COORD_DIM * nelem;
      const Integer nthreads = SCTL_GET_MAX_THREADS();
      const Integer chunk = (Integer)std::max<Long>(1, std::min<Long>(64, (nblk + nthreads - 1) / nthreads));
      const Long nchunk = (nblk + chunk - 1) / chunk;
      #pragma omp parallel for schedule(static)
      for (Long b = 0; b < nchunk; b++) {
        const Long offset = b * chunk * nnode_per_elem;
        const Integer n = (Integer)(std::min(nblk, (b + 1) * chunk) - b * chunk) * nnode_per_elem;
        const Vector<Real> coord_(n, coord.begin() + offset, false);
        { // Differentiate along v, the contiguous index
          Vector<Real> dv_(n, dcoord_dv.begin() + offset, false);
          LagrangeInterp<Real>::Derivative(dv_, coord_, nodes);
        }
        { // Differentiate along u through transposed copies
          ScratchBuf<Real> coordT_buf(n), duT_buf(n);
          Vector<Real> coordT(coordT_buf), duT(duT_buf);
          Vector<Real> du_(n, dcoord_du.begin() + offset, false);
          transpose_blocks(coordT, coord_);
          LagrangeInterp<Real>::Derivative(duT, coordT, nodes);
          transpose_blocks(du_, duT);
        }
      }
    }
    { // Node positions and normals
      X_node.ReInit(nelem * elem_stride);
      Xn_node.ReInit(nelem * elem_stride);
      node_cnt.ReInit(nelem);
      node_cnt = nnode_per_elem;
      const auto& nodes = ParamNodes(order);
      #pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        Vector<Real> X_(elem_stride, X_node.begin() + elem_idx*elem_stride, false);
        Vector<Real> Xn_(elem_stride, Xn_node.begin() + elem_idx*elem_stride, false);
        GetGeom(&X_, &Xn_, nullptr, nullptr, nullptr, nodes, nodes, elem_idx);
      }
    }
  }

  template <class Real> Long QuadElemList<Real>::Size() const {
    return nelem;
  }

  template <class Real> Integer QuadElemList<Real>::Order() const {
    return order;
  }

  template <class Real> void QuadElemList<Real>::GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx) const {
    SCTL_ASSERT(elem_idx >= 0 && elem_idx < nelem);
    const Integer nnode_per_elem = order * order;
    const Integer Nu = (Integer)u_param.Dim();
    const Integer Nv = (Integer)v_param.Dim();
    const Integer N = Nu * Nv;

    { // Size the requested outputs
      if (X && X->Dim() != N * detail_quadelem::COORD_DIM) X->ReInit(N * detail_quadelem::COORD_DIM);
      if (Xn && Xn->Dim() != N * detail_quadelem::COORD_DIM) Xn->ReInit(N * detail_quadelem::COORD_DIM);
      if (Xa && Xa->Dim() != N) Xa->ReInit(N);
      if (dX_du && dX_du->Dim() != N * detail_quadelem::COORD_DIM) dX_du->ReInit(N * detail_quadelem::COORD_DIM);
      if (dX_dv && dX_dv->Dim() != N * detail_quadelem::COORD_DIM) dX_dv->ReInit(N * detail_quadelem::COORD_DIM);
    }

    ScratchBuf<Real> MuT_buf(Nu * order), Mv_buf(order * Nv);
    { // Interpolation matrices from the nodes to u_param and v_param
      ScratchBuf<Real> Mu_buf(order * Nu);
      Vector<Real> Mu_(Mu_buf);
      Vector<Real> Mv_(Mv_buf);
      LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_param);
      LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_param);
      for (Integer i = 0; i < order; i++) for (Integer a = 0; a < Nu; a++) MuT_buf[a * order + i] = Mu_buf[i * Nu + a];
    }
    const Matrix<Real> MuT(Nu, order, MuT_buf.begin(), false);
    const Matrix<Real> Mv(order, Nv, Mv_buf.begin(), false);

    const Long base = elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM;
    if (X) { // Positions
      const Vector<Real> coord_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)coord.begin() + base, false);
      ScratchBuf<Real> X_soa_buf(N * detail_quadelem::COORD_DIM);
      Vector<Real> X_soa(X_soa_buf);
      detail_quadelem::EvalTensorProduct(X_soa, coord_, MuT, Mv);
      for (Integer i = 0; i < N; i++) {
        (*X)[i * detail_quadelem::COORD_DIM + 0] = X_soa[0 * N + i];
        (*X)[i * detail_quadelem::COORD_DIM + 1] = X_soa[1 * N + i];
        (*X)[i * detail_quadelem::COORD_DIM + 2] = X_soa[2 * N + i];
      }
    }
    if (Xn || Xa || dX_du || dX_dv) { // Tangents, and from them normals and area elements
      const Vector<Real> dcoord_du_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_du.begin() + base, false);
      const Vector<Real> dcoord_dv_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_dv.begin() + base, false);
      ScratchBuf<Real> dXdu_soa_buf(N * detail_quadelem::COORD_DIM), dXdv_soa_buf(N * detail_quadelem::COORD_DIM);
      Vector<Real> dXdu_soa(dXdu_soa_buf), dXdv_soa(dXdv_soa_buf);
      detail_quadelem::EvalTensorProduct(dXdu_soa, dcoord_du_, MuT, Mv);
      detail_quadelem::EvalTensorProduct(dXdv_soa, dcoord_dv_, MuT, Mv);
      for (Integer i = 0; i < N; i++) {
        const Real du0 = dXdu_soa[0 * N + i];
        const Real du1 = dXdu_soa[1 * N + i];
        const Real du2 = dXdu_soa[2 * N + i];
        const Real dv0 = dXdv_soa[0 * N + i];
        const Real dv1 = dXdv_soa[1 * N + i];
        const Real dv2 = dXdv_soa[2 * N + i];

        const Real n0 = du1 * dv2 - du2 * dv1;
        const Real n1 = du2 * dv0 - du0 * dv2;
        const Real n2 = du0 * dv1 - du1 * dv0;
        const Real area = sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
        const Real inv_area = (area > 0 ? 1 / area : 0);

        if (Xn) {
          (*Xn)[i * detail_quadelem::COORD_DIM + 0] = n0 * inv_area;
          (*Xn)[i * detail_quadelem::COORD_DIM + 1] = n1 * inv_area;
          (*Xn)[i * detail_quadelem::COORD_DIM + 2] = n2 * inv_area;
        }
        if (Xa) {
          (*Xa)[i] = area;
        }
        if (dX_du) {
          (*dX_du)[i * detail_quadelem::COORD_DIM + 0] = du0;
          (*dX_du)[i * detail_quadelem::COORD_DIM + 1] = du1;
          (*dX_du)[i * detail_quadelem::COORD_DIM + 2] = du2;
        }
        if (dX_dv) {
          (*dX_dv)[i * detail_quadelem::COORD_DIM + 0] = dv0;
          (*dX_dv)[i * detail_quadelem::COORD_DIM + 1] = dv1;
          (*dX_dv)[i * detail_quadelem::COORD_DIM + 2] = dv2;
        }
      }
    }
  }

  template <class Real> void QuadElemList<Real>::GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const {
    if (X) *X = X_node;
    if (Xn) *Xn = Xn_node;
    if (element_wise_node_cnt) *element_wise_node_cnt = node_cnt;
  }

  template <class Real> void QuadElemList<Real>::GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const {
    const Integer nnode_per_elem = order * order;
    const Long Nnode = nelem * nnode_per_elem;
    { // Size the outputs
      if (X.Dim() != Nnode * detail_quadelem::COORD_DIM) X.ReInit(Nnode * detail_quadelem::COORD_DIM);
      if (Xn.Dim() != Nnode * detail_quadelem::COORD_DIM) Xn.ReInit(Nnode * detail_quadelem::COORD_DIM);
      if (wts.Dim() != Nnode) wts.ReInit(Nnode);
      if (dist_far.Dim() != Nnode) dist_far.ReInit(Nnode);
      if (element_wise_node_cnt.Dim() != nelem) element_wise_node_cnt.ReInit(nelem);
      element_wise_node_cnt = nnode_per_elem;
    }
    if (!nelem) return; // nothing to compute, also for a default-constructed list, whose order is 0

    const auto& nodes = ParamNodes(order);
    ScratchBuf<Real> dist_nodes(order);
    { // Parameter distance from each node to the accuracy ellipse
      const Integer n = order;
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real rho = pow<Real>((64 / (15 * tol_)), 1 / (Real)(2 * n));
      const Real a = (rho - 1 / rho) / 4;
      const Real b = (rho + 1 / rho) / 4;
      for (Integer i = 0; i < n; i++) {
        dist_nodes[i] = b - fabs(nodes[i] - (Real)0.5);
        const Real cos_t = 4 * b * (nodes[i] - (Real)0.5);
        if (fabs(cos_t) <= 1) {
          dist_nodes[i] = a * sqrt<Real>(1 + ((a * a) / (b * b) - 1) * cos_t * cos_t);
        }
      }
    }

    const auto& node_wts = LegQuadRule<Real>::wts(order);
    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) { // Nodes, weights and far distances per element
      Vector<Real> X_(nnode_per_elem * detail_quadelem::COORD_DIM, X.begin() + elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM, false);
      Vector<Real> Xn_(nnode_per_elem * detail_quadelem::COORD_DIM, Xn.begin() + elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM, false);
      Vector<Real> wts_(nnode_per_elem, wts.begin() + elem_idx * nnode_per_elem, false);
      Vector<Real> dist_far_(nnode_per_elem, dist_far.begin() + elem_idx * nnode_per_elem, false);

      ScratchBuf<Real> Xa_buf(nnode_per_elem), dXdu_buf(nnode_per_elem * detail_quadelem::COORD_DIM), dXdv_buf(nnode_per_elem * detail_quadelem::COORD_DIM);
      Vector<Real> Xa(Xa_buf), dXdu(dXdu_buf), dXdv(dXdv_buf);
      GetGeom(&X_, &Xn_, &Xa, &dXdu, &dXdv, nodes, nodes, elem_idx);

      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          const Integer p = i * order + j;
          const Real wu = node_wts[i];
          const Real wv = node_wts[j];
          wts_[p] = Xa[p] * wu * wv;

          const Real len_u = sqrt<Real>(dXdu[p * detail_quadelem::COORD_DIM + 0] * dXdu[p * detail_quadelem::COORD_DIM + 0] +
              dXdu[p * detail_quadelem::COORD_DIM + 1] * dXdu[p * detail_quadelem::COORD_DIM + 1] +
              dXdu[p * detail_quadelem::COORD_DIM + 2] * dXdu[p * detail_quadelem::COORD_DIM + 2]);
          const Real len_v = sqrt<Real>(dXdv[p * detail_quadelem::COORD_DIM + 0] * dXdv[p * detail_quadelem::COORD_DIM + 0] +
              dXdv[p * detail_quadelem::COORD_DIM + 1] * dXdv[p * detail_quadelem::COORD_DIM + 1] +
              dXdv[p * detail_quadelem::COORD_DIM + 2] * dXdv[p * detail_quadelem::COORD_DIM + 2]);
          dist_far_[p] = std::max(dist_nodes[i] * len_u, dist_nodes[j] * len_v);
        }
      }
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const Real tol, const bool trg_dot_prod, const ElementListBase<Real>* self) {
    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
    detail_dispatch::WarnStronglySingular<Real>(ker, trg_dot_prod, qel); // also for an empty list, so that rank 0 checks
    if (!qel.Size()) return; // nothing to compute, also for a default-constructed list, whose order is 0
    const Integer order = qel.Order();
    const Integer digits = detail_quadelem::DigitsFromTol<Real>(tol);
    switch (order) {
      case  4: return detail_dispatch::SelfInteracDispatch<4,Real>(M_lst, ker, trg_dot_prod, qel, digits);
      case  8: return detail_dispatch::SelfInteracDispatch<8,Real>(M_lst, ker, trg_dot_prod, qel, digits);
      case 12: return detail_dispatch::SelfInteracDispatch<12,Real>(M_lst, ker, trg_dot_prod, qel, digits);
      case 16: return detail_dispatch::SelfInteracDispatch<16,Real>(M_lst, ker, trg_dot_prod, qel, digits);
      case 20: return detail_dispatch::SelfInteracDispatch<20,Real>(M_lst, ker, trg_dot_prod, qel, digits);
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,12,16,20} for the templated near/self schemes.");
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
    const Integer order = qel.Order();
    const Integer digits = detail_quadelem::DigitsFromTol<Real>(tol);
    switch (order) {
      case  4: return detail_dispatch::NearInteracDispatch<4,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      case  8: return detail_dispatch::NearInteracDispatch<8,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      case 12: return detail_dispatch::NearInteracDispatch<12,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      case 16: return detail_dispatch::NearInteracDispatch<16,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      case 20: return detail_dispatch::NearInteracDispatch<20,Real>(M, Xt, normal_trg, ker, elem_idx, qel, digits);
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,12,16,20} for the templated near/self schemes.");
    }
  }

  template <class Real> const Vector<Real>& QuadElemList<Real>::ParamNodes(const Integer order) {
    return LegQuadRule<Real>::nds(order);
  }

  template <class Real> void QuadElemList<Real>::Write(const std::string& fname, const Comm& comm) const {
    const Integer precision = (Integer)ceil<Real>(-log<Real>(machine_eps<Real>()) / log<Real>((Real)10));
    const Integer width = precision + 8;
    const std::string fname_rank = detail_quadelem::RankFileName(fname, comm);
    std::ofstream file(fname_rank, std::ofstream::out | std::ofstream::trunc);
    SCTL_ASSERT_MSG(file.good(), std::string("Unable to open file for writing: ") + fname_rank);

    { // Header line
      file << "#";
      file << std::setw(width - 1) << "X";
      file << std::setw(width) << "Y";
      file << std::setw(width) << "Z";
      file << std::setw(width) << "ElemOrder";
      file << '\n';
    }
    { // One line per node; order on each element's first
      file << std::scientific << std::setprecision(precision);
      const Integer nnode_per_elem = order * order;
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * detail_quadelem::COORD_DIM * nnode_per_elem;
        for (Integer p = 0; p < nnode_per_elem; p++) {
          for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
            file << std::setw(width) << coord[base + k * nnode_per_elem + p];
          }
          if (!p) file << std::setw(width) << order;
          file << '\n';
        }
      }
    }
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Read(const std::string& fname, const Comm& comm) {
    Vector<ValueType> coord_;
    Vector<Integer> order_markers;
    { // Parse each node line: coordinates, then an optional element order
      const std::string fname_rank = detail_quadelem::RankFileName(fname, comm);
      std::ifstream file(fname_rank, std::ifstream::in);
      SCTL_ASSERT_MSG(file.good(), std::string("Unable to open file for reading: ") + fname_rank);
      std::string line;
      while (std::getline(file, line)) {
        const size_t first_char_pos = line.find_first_not_of(' ');
        if (first_char_pos == std::string::npos || line[first_char_pos] == '#') continue;

        std::istringstream iss(line);
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          ValueType a;
          iss >> a;
          SCTL_ASSERT(!iss.fail());
          coord_.PushBack(a);
        }

        Integer order_;
        if (iss >> order_) {
          order_markers.PushBack(order_);
        } else {
          order_markers.PushBack(-1);
        }
      }
    }

    StaticArray<Integer,2> file_order{(order_markers.Dim() ? order_markers[0] : 0), 0};
    comm.Allreduce(file_order + 0, file_order + 1, 1, CommOp::MAX);
    SCTL_ASSERT(file_order[1] > 0);
    { // Check that every element starts with that order
      const Integer nnode_per_elem = file_order[1] * file_order[1];
      SCTL_ASSERT(order_markers.Dim() % nnode_per_elem == 0);
      const Long Nelem_local = order_markers.Dim() / nnode_per_elem;
      for (Long elem = 0; elem < Nelem_local; elem++) {
        const Long offset = elem * nnode_per_elem;
        SCTL_ASSERT(order_markers[offset] == file_order[1]);
        for (Integer j = 1; j < nnode_per_elem; j++) {
          SCTL_ASSERT(order_markers[offset + j] == file_order[1] || order_markers[offset + j] == -1);
        }
      }
    }
    Init<ValueType>(file_order[1], coord_);
  }

  template <class Real> void QuadElemList<Real>::GetVTUData(VTUData& vtu_data, const Vector<Real>& F, const Long elem_idx) const {
    if (elem_idx == -1) { // Every element, each with its part of F
      const Integer nnode_per_elem = order * order;
      Integer dof = 0;
      Long offset = 0;
      if (F.Dim()) {
        const Long Nnode = nelem * nnode_per_elem;
        dof = (Nnode ? F.Dim() / Nnode : 0);
        SCTL_ASSERT(F.Dim() == Nnode * dof);
      }
      for (Long i = 0; i < nelem; i++) {
        const Vector<Real> F_(nnode_per_elem * dof, (Iterator<Real>)F.begin() + offset, false);
        GetVTUData(vtu_data, F_, i);
        offset += F_.Dim();
      }
      return;
    }

    const Integer Ng = order + 2;
    ScratchBuf<Real> grid_buf(Ng);
    Vector<Real> grid(grid_buf);
    { // Grid of the nodes plus both ends, in each direction
      grid[0] = 0;
      for (Integer i = 0; i < order; i++) grid[i + 1] = ParamNodes(order)[i];
      grid[Ng - 1] = 1;
    }
    ScratchBuf<Real> X_buf(Ng * Ng * detail_quadelem::COORD_DIM);
    Vector<Real> X(X_buf);
    GetGeom(&X, nullptr, nullptr, nullptr, nullptr, grid, grid, elem_idx);

    if (F.Dim()) { // Field values at the grid points
      const Integer nnode_per_elem = order * order;
      const Integer dof = F.Dim() / nnode_per_elem;
      SCTL_ASSERT(F.Dim() == nnode_per_elem * dof);

      ScratchBuf<Real> F_soa_buf(dof * nnode_per_elem);
      Vector<Real> F_soa(F_soa_buf);
      for (Integer p = 0; p < nnode_per_elem; p++) {
        for (Integer k = 0; k < dof; k++) {
          F_soa[k * nnode_per_elem + p] = F[p * dof + k];
        }
      }

      ScratchBuf<Real> M_buf(order * Ng), MT_buf(Ng * order);
      Vector<Real> M_v(M_buf);
      LagrangeInterp<Real>::Interpolate(M_v, ParamNodes(order), grid);
      for (Integer i = 0; i < order; i++) {
        for (Integer a = 0; a < Ng; a++) MT_buf[a * order + i] = M_buf[i * Ng + a];
      }
      const Matrix<Real> M(order, Ng, M_buf.begin(), false);
      const Matrix<Real> MT(Ng, order, MT_buf.begin(), false);

      ScratchBuf<Real> F_grid_buf(dof * Ng * Ng);
      Vector<Real> F_grid(F_grid_buf);
      detail_quadelem::EvalTensorProduct(F_grid, F_soa, MT, M);
      for (Integer p = 0; p < Ng * Ng; p++) {
        for (Integer k = 0; k < dof; k++) vtu_data.value.PushBack((VTUData::VTKReal)F_grid[k * (Ng * Ng) + p]);
      }
    }

    const Long point_offset = vtu_data.coord.Dim() / detail_quadelem::COORD_DIM;
    for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
    for (Integer i = 0; i < Ng - 1; i++) { // Quadrilateral cells of the grid
      for (Integer j = 0; j < Ng - 1; j++) {
        const Long idx = point_offset + i * Ng + j;
        vtu_data.connect.PushBack(idx);
        vtu_data.connect.PushBack(idx + 1);
        vtu_data.connect.PushBack(idx + Ng + 1);
        vtu_data.connect.PushBack(idx + Ng);
        vtu_data.offset.PushBack(vtu_data.connect.Dim());
        vtu_data.types.PushBack(9);
      }
    }
  }

  template <class Real> void QuadElemList<Real>::WriteVTK(const std::string& fname, const Vector<Real>& F, const Comm& comm) const {
    VTUData vtu_data;
    GetVTUData(vtu_data, F);
    vtu_data.WriteVTK(fname, comm);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Copy(QuadElemList<ValueType>& elem_lst) const {
    elem_lst.scheme_ = static_cast<typename QuadElemList<ValueType>::QuadScheme>(static_cast<int>(scheme_));
    if constexpr (significant_bits<ValueType>() > significant_bits<Real>()) {
      if (order > 0) { // to a higher precision: the tangents, nodes and normals again from the coordinates, at that precision
        const Integer nnode = order * order;
        ScratchBuf<Real> X0_buf(coord.Dim());
        Vector<Real> X0(X0_buf); // the coordinates node by node, as Init takes them
        for (Long e = 0; e < nelem; e++) {
          for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
            for (Integer p = 0; p < nnode; p++) X0[(e * nnode + p) * detail_quadelem::COORD_DIM + k] = coord[(e * detail_quadelem::COORD_DIM + k) * nnode + p];
          }
        }
        elem_lst.Init(order, X0);
        return;
      }
    }
    elem_lst.nelem = nelem;
    elem_lst.order = order;

    const auto convert = [](Vector<ValueType>& dst, const Vector<Real>& src) {
      dst.ReInit(src.Dim());
      for (Long i = 0; i < src.Dim(); i++) dst[i] = (ValueType)src[i];
    };
    convert(elem_lst.coord, coord);
    convert(elem_lst.dcoord_du, dcoord_du);
    convert(elem_lst.dcoord_dv, dcoord_dv);
    convert(elem_lst.X_node, X_node);
    convert(elem_lst.Xn_node, Xn_node);
    elem_lst.node_cnt = node_cnt;
  }

}

#endif
