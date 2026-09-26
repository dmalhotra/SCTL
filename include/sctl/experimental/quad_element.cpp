#ifndef _SCTL_QUAD_ELEMENT_CPP_
#define _SCTL_QUAD_ELEMENT_CPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
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
    static constexpr Long MaxUnblockedPts = 16384;

    template <class Real> struct Access {
      static const Vector<Real>& Coord(const QuadElemList<Real>& qel) { return qel.coord; }
      static typename QuadElemList<Real>::QuadScheme Scheme(const QuadElemList<Real>& qel) { return qel.scheme_; }
    };

    template <class Real> static constexpr Integer MaxDigits = 1 + GetSigBits<Real>::value()*30103/100000;

    template <class Real> static constexpr Integer MaxRefineLvl = GetSigBits<Real>::value();

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
      const Integer Nu = MuT.Dim(0);
      const Integer R  = MuT.Dim(1);
      const Integer S  = Mv.Dim(0);
      const Integer Nv = Mv.Dim(1);
      const Long ncomp = in.Dim() / ((Long)R * S);
      SCTL_ASSERT(in.Dim() == ncomp * (Long)R * S);

      const Long Nout = (Long)Nu * Nv;
      if (out.Dim() != ncomp * Nout) out.ReInit(ncomp * Nout);

      ScratchBuf<ValueType> tmp_buf((Long)R * Nv);
      Matrix<ValueType> tmp(R, Nv, tmp_buf.begin(), false);

      for (Long k = 0; k < ncomp; k++) {
        const Matrix<ValueType> in_(R, S, (Iterator<ValueType>)in.begin() + k * (Long)R * S, false);
        Matrix<ValueType> out_(Nu, Nv, out.begin() + k * Nout, false);
        Matrix<ValueType>::GEMM(tmp, in_, Mv);
        Matrix<ValueType>::GEMM(out_, MuT, tmp);
      }
    }

    template <class Real> void NodalDerivs(const Vector<Real>& coord_slab, const Integer order, Vector<Real>& du_slab, Vector<Real>& dv_slab) {
      const Long nnode_per_elem = (Long)order * order;
      const Long ncomp = coord_slab.Dim() / nnode_per_elem;
      SCTL_ASSERT(coord_slab.Dim() == ncomp * nnode_per_elem);
      if (du_slab.Dim() != coord_slab.Dim()) du_slab.ReInit(coord_slab.Dim());
      if (dv_slab.Dim() != coord_slab.Dim()) dv_slab.ReInit(coord_slab.Dim());

      const auto& nodes = QuadElemList<Real>::ParamNodes(order);
      ScratchBuf<Real> line_in_buf(order), line_out_buf(order);
      Vector<Real> line_in(line_in_buf), line_out(line_out_buf);
      for (Long k = 0; k < ncomp; k++) {
        const Long cb = k * nnode_per_elem;
        { // Differentiate along u, one v-node column at a time
          for (Integer j = 0; j < order; j++) {
            for (Integer i = 0; i < order; i++) line_in[i] = coord_slab[cb + i * order + j];
            LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
            for (Integer i = 0; i < order; i++) du_slab[cb + i * order + j] = line_out[i];
          }
        }
        { // Differentiate along v, one u-node row at a time
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) line_in[j] = coord_slab[cb + i * order + j];
            LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
            for (Integer j = 0; j < order; j++) dv_slab[cb + i * order + j] = line_out[j];
          }
        }
      }
    }

    template <class Real> void LagrangeDiffMat(Matrix<Real>& D, const Vector<Real>& nds) {
      const Integer n = nds.Dim();
      Vector<Real> f((Long)n * n);
      f.SetZero();
      for (Integer i = 0; i < n; i++) f[i * n + i] = 1;
      D.ReInit(n, n);
      Vector<Real> df((Long)n * n, D.begin(), false);
      LagrangeInterp<Real>::Derivative(df, f, nds);
    }

    /** Returns an order x order matrix for each order; entry (i, j) is the derivative of the i-th Lagrange basis function on ParamNodes(order) at j-th node. */
    template <class Real> inline const Matrix<Real>& DiffMat(const Integer order) {
      SCTL_ASSERT(0 < order && order <= MaxTableOrder);
      static const Vector<Matrix<Real>> all = []() {
        Vector<Matrix<Real>> D(MaxTableOrder + 1);
        for (Integer n = 2; n <= MaxTableOrder; n++) LagrangeDiffMat(D[n], QuadElemList<Real>::ParamNodes(n));
        return D;
      }();
      return all[order];
    }

    template <class Real> inline Integer DigitsFromTol(const Real tol) {
      for (Integer d = MaxDigits<Real>-1; d > 0; d--) if (tol <= pow<Real,Long>((Real)0.1, (Long)d)) return d;
      return 0;
    }

    template <class Real> struct QuadParamSet {
      Real b_ellipse;
      Integer quad_order;
    };

    /** Returns the {b_ellipse, quad_order} pair for each digits, from QuadParams at tolerance 10^-digits; one table per QuadParams function. */
    template <class Real, void (*QuadParams)(Real, Real&, Integer&)> const QuadParamSet<Real>& QuadParamsForDigits(const Integer digits) {
      static const std::array<QuadParamSet<Real>, MaxDigits<Real>> table = []() {
        std::array<QuadParamSet<Real>, MaxDigits<Real>> t{};
        for (Integer d = 0; d < MaxDigits<Real>; d++) QuadParams(pow<Real,Long>((Real)0.1, (Long)d), t[d].b_ellipse, t[d].quad_order);
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

    template <class Real> struct QuadRule1D {
      Vector<Real> w;
      Matrix<Real> M, dM, MT, dMT;
    };

    template <class Real> void EvalPoint(const QuadElemList<Real>& qel, Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Long elem_idx, const Vector<Real>* origin) {
      const Integer order = qel.Order();
      const Long nnode = (Long)order * order;
      const bool want_d = (dXu || dXv);

      ScratchBuf<Real> Lu(order), Lv(order), dLu(order), dLv(order);
      { // Lagrange basis at u and at v
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const auto interp_at = [&nds, order](ScratchBuf<Real>& L, const Real t) {
          StaticArray<Real,1> tp;
          tp[0] = t;
          const Vector<Real> p(1, tp, false);
          Vector<Real> o(order, L.begin(), false);
          LagrangeInterp<Real>::Interpolate(o, nds, p);
        };
        interp_at(Lu, u);
        interp_at(Lv, v);
      }
      if (want_d) { // Derivatives of the basis
        const Matrix<Real>& D = DiffMat<Real>(order);
        for (Integer i = 0; i < order; i++) {
          Real su = 0, sv = 0;
          for (Integer a = 0; a < order; a++) {
            su += D[i][a]*Lu[a];
            sv += D[i][a]*Lv[a];
          }
          dLu[i] = su;
          dLv[i] = sv;
        }
      }

      Real x[COORD_DIM] = {0, 0, 0}, xu[COORD_DIM] = {0, 0, 0}, xv[COORD_DIM] = {0, 0, 0};
      { // Sum the nodal coordinates against the basis
        const Vector<Real>& coord = Access<Real>::Coord(qel);
        const Long base = elem_idx * nnode * COORD_DIM;
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            const Long p = i*order + j;
            const Real w = Lu[i]*Lv[j];
            for (Integer k = 0; k < COORD_DIM; k++) x[k] += coord[base + k*nnode + p]*w;
            if (want_d) {
              const Real w_du = dLu[i]*Lv[j];
              const Real w_dv = Lu[i]*dLv[j];
              for (Integer k = 0; k < COORD_DIM; k++) {
                xu[k] += coord[base + k*nnode + p]*w_du;
                xv[k] += coord[base + k*nnode + p]*w_dv;
              }
            }
          }
        }
      }
      for (Integer k = 0; k < COORD_DIM; k++) {
        X[k] = (origin ? x[k] - (*origin)[k] : x[k]);
        if (dXu) dXu[k] = xu[k];
        if (dXv) dXv[k] = xv[k];
      }
    }

    template <class Real> void ShiftedElemCoord(Vector<Real>& out, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg) {
      const Long nnode = (Long)qel.Order() * qel.Order();
      const Long base = elem_idx * nnode * COORD_DIM;
      if (out.Dim() != COORD_DIM*nnode) out.ReInit(COORD_DIM*nnode);
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) out[k*nnode + p] = Access<Real>::Coord(qel)[base + k*nnode + p] - ok;
      }
    }

    template <class Real> Real GetClosestNode(const QuadElemList<Real>& qel, Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) {
      const Long nnode = (Long)qel.Order() * qel.Order();
      const Long base = elem_idx * nnode * COORD_DIM;
      Long seed = 0;
      Real best = -1;
      for (Long p = 0; p < nnode; p++) {
        Real r2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          const Real d = Access<Real>::Coord(qel)[base + k*nnode + p] - Xtrg[k];
          r2 += d*d;
        }
        if (best < 0 || r2 < best) {
          best = r2;
          seed = p;
        }
      }

      const auto& nds = QuadElemList<Real>::ParamNodes(qel.Order());
      ustar = nds[seed/qel.Order()];
      vstar = nds[seed%qel.Order()];
      return sqrt<Real>(best);
    }

    template <class Real> Real GetClosestPoint(const QuadElemList<Real>& qel, Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) {
      const auto dist2_at = [&qel, elem_idx, &Xtrg](const Real uu, const Real vv) -> Real {
        Real X[COORD_DIM];
        EvalPoint<Real>(qel, X, nullptr, nullptr, uu, vv, elem_idx, &Xtrg);
        Real r2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) r2 += X[k]*X[k];
        return r2;
      };

      Real u, v, f;
      { // Start from the nearest node
        const Real f_seed = GetClosestNode(qel, u, v, elem_idx, Xtrg);
        f = f_seed * f_seed;
      }

      constexpr Integer max_iter = 30;
      const Real utol = (Real)machine_eps<Real>() * 64;
      const Real gtol = sqrt<Real>(machine_eps<Real>()) * 16;
      bool converged = false;
      for (Integer it = 0; it < max_iter; it++) { // Projected Newton iterations
        Real E = 0, F = 0, G = 0, gu = 0, gv = 0;
        { // Metric and gradient of the squared distance
          Real X[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
          EvalPoint<Real>(qel, X, dXu, dXv, u, v, elem_idx, &Xtrg);
          for (Integer k = 0; k < COORD_DIM; k++) {
            const Real r = X[k], a = dXu[k], b = dXv[k];
            E += a*a;
            F += a*b;
            G += b*b;
            gu += r*a;
            gv += r*b;
          }
        }

        Real Pu = gu, Pv = gv;
        { // Project the gradient onto the parameter square
          if      (u <= 0) Pu = std::min<Real>(gu, (Real)0);
          else if (u >= 1) Pu = std::max<Real>(gu, (Real)0);
          if      (v <= 0) Pv = std::min<Real>(gv, (Real)0);
          else if (v >= 1) Pv = std::max<Real>(gv, (Real)0);
        }
        const bool opt_u = (fabs(Pu) <= gtol * sqrt<Real>(E*f));
        const bool opt_v = (fabs(Pv) <= gtol * sqrt<Real>(G*f));
        if (opt_u && opt_v) {
          converged = true;
          break;
        }

        Real step_u = 0, step_v = 0;
        { // Newton step, fixing a coordinate held at its bound
          const bool u_act = ((u <= 0 && gu >= 0) || (u >= 1 && gu <= 0));
          const bool v_act = ((v <= 0 && gv >= 0) || (v >= 1 && gv <= 0));
          if (!u_act && !v_act) {
            const Real det = E*G - F*F;
            if (fabs(det) > (Real)1e-30 * (E*G + F*F + 1)) {
              step_u = ( G*gu - F*gv) / det;
              step_v = (-F*gu + E*gv) / det;
            } else {
              step_u = gu / (E + (Real)1e-30);
              step_v = gv / (G + (Real)1e-30);
            }
          } else if (u_act) {
            step_v = gv / (G + (Real)1e-30);
          } else {
            step_u = gu / (E + (Real)1e-30);
          }
        }

        Real un = u, vn = v, fn = f;
        bool improved;
        { // Line search along the step, else along the gradient
          const Real c_eps = machine_eps<Real>() * 8;
          const auto line_search = [&dist2_at, u, v, f, c_eps, &un, &vn, &fn](const Real step_u, const Real step_v) {
            Real lambda = 1;
            for (Integer ls = 0; ls < 40; ls++) {
              un = std::min<Real>(1, std::max<Real>(0, u - lambda*step_u));
              vn = std::min<Real>(1, std::max<Real>(0, v - lambda*step_v));
              fn = dist2_at(un, vn);
              if (fn <= f * (1 + c_eps)) return true;
              lambda *= (Real)0.5;
            }
            return false;
          };
          improved = line_search(step_u, step_v);
          if (!improved) improved = line_search(Pu / (E + (Real)1e-30), Pv / (G + (Real)1e-30));
        }
        if (!improved) { // Stop; converged if the gradient or step is negligible
          const Real gtol_stall = sqrt<Real>(machine_eps<Real>()) * 256;
          const bool stationary = (fabs(Pu) <= gtol_stall * sqrt<Real>(E*f)) && (fabs(Pv) <= gtol_stall * sqrt<Real>(G*f));
          const bool tiny = (fabs(step_u) < utol && fabs(step_v) < utol);
          if (it > 0 && (stationary || tiny)) converged = true;
          break;
        }

        const bool small_step = (fabs(un-u) < utol && fabs(vn-v) < utol);
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

    template <class K, class VT, class = void> struct UKerNeedsN : std::false_type {};
    template <class K, class VT> struct UKerNeedsN<K, VT, std::void_t<decltype(
        K::template uKerMatrix<0,VT>(std::declval<VT(&)[K::SrcDim()][K::TrgDim()]>(),
          std::declval<const VT(&)[3]>(),
          std::declval<const VT(&)[3]>(),
          (const void*)nullptr))>> : std::true_type {};

    template <class Real, class Kernel, class VecType, bool HAS_N, bool TRG_DOT>
    static void WeightedKernelVec(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Long nq, const Long run, const Long j0, const Long j1, const Real wj, const bool accum, ConstIterator<Real> ntrg, const void* ctx) {
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
      for (Long qb = 0, blk = 0; qb < nq; qb += run, blk++) {
        for (Long j = j0; j < j1; j += VL) {
          const Long q = qb + j;
          VecType r[CD], n[CD], u[KD0][KD1];
          for (Integer k = 0; k < CD; k++) r[k] = vXt[k] - VecType::Load(&Xs[k*nq+q]);
          if constexpr (HAS_N) {
            for (Integer k = 0; k < CD; k++) n[k] = VecType::Load(&Xn[k*nq+q]);
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
              const Long id = blk*C*run + (Long)(a*KD1o+b)*run + j;
              if (accum) (VecType::Load(&out[id]) + val*vw).Store(&out[id]);
              else       (val*vw).Store(&out[id]);
            }
          }
        }
      }
    }

    template <class Real, class Kernel> void WeightedKernel(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Long nq, const Long run, const Real wj, const bool accum, const Vector<Real>& normal_trg, const Kernel& ker) {
      static constexpr bool HAS_N = UKerNeedsN<Kernel, Vec<Real,1>>::value;
      using WVec = Vec<Real, DefaultVecLen<Real>()>;
      const Long jmain = (run/WVec::Size())*WVec::Size();
      const ConstIterator<Real> nt = (normal_trg.Dim() ? normal_trg.begin() : ConstIterator<Real>(NullIterator<Real>()));
      if (normal_trg.Dim()) {
        WeightedKernelVec<Real,Kernel,WVec,        HAS_N,true >(out, Xt, Xs, Xn, wq, nq, run,     0, jmain, wj, accum, nt, ker.GetCtxPtr());
        WeightedKernelVec<Real,Kernel,Vec<Real,1>, HAS_N,true >(out, Xt, Xs, Xn, wq, nq, run, jmain,   run, wj, accum, nt, ker.GetCtxPtr());
      } else {
        WeightedKernelVec<Real,Kernel,WVec,        HAS_N,false>(out, Xt, Xs, Xn, wq, nq, run,     0, jmain, wj, accum, nt, ker.GetCtxPtr());
        WeightedKernelVec<Real,Kernel,Vec<Real,1>, HAS_N,false>(out, Xt, Xs, Xn, wq, nq, run, jmain,   run, wj, accum, nt, ker.GetCtxPtr());
      }
    }

    template <Integer order, class Real, class Kernel> void IntegrateTensorRule(Vector<Real>& acc_cm, const Vector<Real>& src_nodal, const QuadRule1D<Real>& ru, const QuadRule1D<Real>& rv, const Vector<Real>& normal_trg, const Kernel& ker, const Real nrm_sign = 1, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Long nnode = (Long)order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      const Integer C = KDIM0 * KDIM1_out;
      const Long Nu = ru.M.Dim(1);
      const Long Nv = rv.M.Dim(1);
      if (!Nu || !Nv) return;

      ScratchBuf<Real> Cv(COORD_DIM*order*Nv), Cdv(COORD_DIM*order*Nv);
      { // Interpolate the coordinates and their v-derivative along v
        const Matrix<Real> cs_all(COORD_DIM*order, order, (Iterator<Real>)src_nodal.begin(), false);
        Matrix<Real> Cv_all (COORD_DIM*order, Nv, Cv.begin(),  false);
        Matrix<Real> Cdv_all(COORD_DIM*order, Nv, Cdv.begin(), false);
        Matrix<Real>::GEMM(Cv_all,  cs_all, rv.M);
        Matrix<Real>::GEMM(Cdv_all, cs_all, rv.dM);
      }

      ScratchBuf<Real> Tall(Nu*C*(Long)order);
      const Long UBLK = std::max<Long>(1, std::min<Long>(Nu, MaxUnblockedPts / Nv));
      for (Long a0 = 0; a0 < Nu; a0 += UBLK) { // Blocks of u-rows
        const Long nu = std::min<Long>(UBLK, Nu - a0);
        const Long nqb = nu*Nv;

        ScratchBuf<Real> Xs(COORD_DIM*nqb), Xn(COORD_DIM*nqb), wq(nqb);
        { // Points, normals and weights of the block
          ScratchBuf<Real> dXu(COORD_DIM*nqb), dXv(COORD_DIM*nqb);
          { // Interpolate coordinates and tangents along u
            const Matrix<Real> MuT_b (nu, order, (Iterator<Real>)ru.MT.begin()  + a0*(Long)order, false);
            const Matrix<Real> dMuT_b(nu, order, (Iterator<Real>)ru.dMT.begin() + a0*(Long)order, false);
            for (Integer k = 0; k < COORD_DIM; k++) {
              const Matrix<Real> Cv_k (order, Nv, Cv.begin()  + k*(Long)order*Nv, false);
              const Matrix<Real> Cdv_k(order, Nv, Cdv.begin() + k*(Long)order*Nv, false);
              Matrix<Real> X_k(nu, Nv, Xs.begin() + k*nqb, false);
              Matrix<Real> dXu_k(nu, Nv, dXu.begin() + k*nqb, false);
              Matrix<Real> dXv_k(nu, Nv, dXv.begin() + k*nqb, false);
              Matrix<Real>::GEMM(X_k,   MuT_b,  Cv_k);
              Matrix<Real>::GEMM(dXu_k, dMuT_b, Cv_k);
              Matrix<Real>::GEMM(dXv_k, MuT_b,  Cdv_k);
            }
          }
          for (Long a = 0; a < nu; a++) {
            for (Long b = 0; b < Nv; b++) {
              const Long q = a*Nv + b;
              const Real du0 = dXu[0*nqb+q], du1 = dXu[1*nqb+q], du2 = dXu[2*nqb+q];
              const Real dv0 = dXv[0*nqb+q], dv1 = dXv[1*nqb+q], dv2 = dXv[2*nqb+q];
              const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
              const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
              const Real inv_area = (area > 0 ? nrm_sign/area : 0);
              Xn[0*nqb+q] = n0*inv_area;
              Xn[1*nqb+q] = n1*inv_area;
              Xn[2*nqb+q] = n2*inv_area;
              wq[q] = area*ru.w[a0+a]*rv.w[b];
            }
          }
        }

        ScratchBuf<Real> KW(C*nqb);
        const Long np = std::max<Long>(1, proxy_w.Dim());
        for (Long j = 0; j < np; j++) { // Weighted kernel, summed over the proxy points
          StaticArray<Real,COORD_DIM> Xtj{0, 0, 0};
          if (proxy_w.Dim()) {
            for (Integer l = 0; l < COORD_DIM; l++) Xtj[l] = proxy_off[j*COORD_DIM+l];
          }
          const Vector<Real> Xtj_v(COORD_DIM, Xtj, false);
          const Real wj = (proxy_w.Dim() ? proxy_w[j] : (Real)1);
          const bool accum = (j > 0);
          WeightedKernel<Real>(KW.begin(), Xtj_v.begin(), Xs.begin(), Xn.begin(), wq.begin(), nqb, nqb, wj, accum, normal_trg, ker);
        }

        { // Project onto the v-nodes and store the block's rows
          ScratchBuf<Real> Tblk(C*nu*(Long)order);
          const Matrix<Real> KW_m((Long)C*nu, Nv, KW.begin(), false);
          Matrix<Real> T_m((Long)C*nu, order, Tblk.begin(), false);
          Matrix<Real>::GEMM(T_m, KW_m, rv.MT);
          for (Long a = 0; a < nu; a++) {
            for (Integer c = 0; c < C; c++) {
              for (Integer j = 0; j < order; j++) Tall[((a0 + a)*C + c)*order + j] = Tblk[((Long)c*nu + a)*order + j];
            }
          }
        }
      }

      { // Project onto the u-nodes and add into acc_cm
        ScratchBuf<Real> Aall((Long)order*C*order);
        const Matrix<Real> T_m(Nu, (Long)C*order, Tall.begin(), false);
        Matrix<Real> A_m(order, (Long)C*order, Aall.begin(), false);
        Matrix<Real>::GEMM(A_m, ru.M, T_m);
        for (Integer c = 0; c < C; c++) {
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) acc_cm[(Long)c*nnode + i*order + j] += Aall[((Long)i*C + c)*order + j];
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void IntegratePanel(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const QuadRule1D<Real>& ru, const QuadRule1D<Real>& rv, const Kernel& ker) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      const Integer C = KDIM0 * KDIM1_out;

      ScratchBuf<Real> coord_shift_buf(COORD_DIM*nnode), acc_buf((Long)C*nnode);
      Vector<Real> coord_shift(coord_shift_buf), acc(acc_buf);
      ShiftedElemCoord(coord_shift, qel, elem_idx, Xtrg);
      acc.SetZero();
      IntegrateTensorRule<order,Real>(acc, coord_shift, ru, rv, normal_trg, ker);
      for (Long p = 0; p < nnode; p++) for (Integer c = 0; c < C; c++) M_acc[p][c] += acc[(Long)c*nnode + p];
    }

    template <class Real> void ScatterTargetBlock(Matrix<Real>& M, const Matrix<Real>& src, const Long t, const Integer KDIM1_out) {
      const Long nrow = M.Dim(0);
      SCTL_ASSERT(src.Dim(0)*src.Dim(1) == nrow*KDIM1_out);
      const ConstIterator<Real> src_ = src.begin();
      for (Long r = 0; r < nrow; r++) {
        for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          M[r][t*KDIM1_out+k1] = src_[r*KDIM1_out+k1];
        }
      }
    }

    template <Integer order, class Real, class Kernel, class NearInteracOneTrg>
    void NearInteracTargets(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const QuadElemList<Real>& qel, NearInteracOneTrg near_interac_one_trg) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

      const Long Ntrg = Xt.Dim() / COORD_DIM;
      if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != Ntrg*KDIM1_out) M.ReInit(nnode*KDIM0, Ntrg*KDIM1_out);
      M.SetZero();
      if (!Ntrg) return;

      ScratchBuf<Real> M_acc_buf(nnode*KDIM0*KDIM1_out);
      Matrix<Real> M_acc(nnode, KDIM0*KDIM1_out, M_acc_buf.begin(), false);
      for (Long t = 0; t < Ntrg; t++) {
        const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)normal_trg.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        near_interac_one_trg(M_acc, Xtrg, ntrg);
        ScatterTargetBlock(M, M_acc, t, KDIM1_out);
      }
    }

    template <Integer order, class Real, class Kernel, class SelfInteracOneTrg>
    void SelfInteracElems(Vector<Matrix<Real>>& M_lst, const bool trg_dot_prod, const QuadElemList<Real>& qel, const bool want_tangents, SelfInteracOneTrg self_interac_one_trg) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      SCTL_ASSERT((Long)M_lst.Dim() == qel.Size());
      const Long nnode = (Long)order * order;
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;
      const bool want_normals = (want_tangents || trg_dot_prod);
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);

      #pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < qel.Size(); elem_idx++) {
        ScratchBuf<Real> Xnodes_buf(nnode*COORD_DIM), Xnnodes_buf(want_normals ? nnode*COORD_DIM : 0);
        ScratchBuf<Real> dXu_buf(want_tangents ? nnode*COORD_DIM : 0), dXv_buf(want_tangents ? nnode*COORD_DIM : 0);
        Vector<Real> Xnodes(Xnodes_buf), Xnnodes(Xnnodes_buf), dXu(dXu_buf), dXv(dXv_buf);
        qel.GetGeom(&Xnodes, (want_normals ? &Xnnodes : nullptr), nullptr,
            (want_tangents ? &dXu : nullptr), (want_tangents ? &dXv : nullptr), nds, nds, elem_idx);

        Matrix<Real>& M = M_lst[elem_idx];
        if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nnode*KDIM0, nnode*KDIM1_out);
        M.SetZero();
        for (Integer ti = 0; ti < order; ti++) {
          for (Integer tj = 0; tj < order; tj++) {
            const Long t = ti*order + tj;
            const Vector<Real> Xtrg(COORD_DIM, Xnodes.begin() + t*COORD_DIM, false);
            const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? Xnnodes.begin() + t*COORD_DIM : NullIterator<Real>()), false);
            self_interac_one_trg(M, elem_idx, t, ti, tj, Xtrg, ntrg, Xnodes, Xnnodes, dXu, dXv, KDIM1_out);
          }
        }
      }
    }

  }

  template <class Real> template <class ValueType> QuadElemList<Real>::QuadElemList(const Integer order0, const Vector<ValueType>& coord0) {
    Init(order0, coord0);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Init(const Integer order0, const Vector<ValueType>& coord0) {
    order = order0;
    SCTL_ASSERT(order > 0);
    const Long nnode_per_elem = (Long)order * order;
    const Long elem_stride = detail_quadelem::COORD_DIM * nnode_per_elem;
    SCTL_ASSERT(coord0.Dim() % elem_stride == 0);
    nelem = coord0.Dim() / elem_stride;

    { // Store the coordinates component by component per element
      coord.ReInit(nelem * elem_stride);
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * elem_stride;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          for (Long p = 0; p < nnode_per_elem; p++) {
            coord[base + k * nnode_per_elem + p] = (Real)coord0[(elem_idx * nnode_per_elem + p) * detail_quadelem::COORD_DIM + k];
          }
        }
      }
    }
    { // Differentiate the coordinates along u and v
      dcoord_du.ReInit(coord.Dim());
      dcoord_dv.ReInit(coord.Dim());
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * elem_stride;
        const Vector<Real> coord_(elem_stride, (Iterator<Real>)coord.begin() + base, false);
        Vector<Real> du_(elem_stride, dcoord_du.begin() + base, false);
        Vector<Real> dv_(elem_stride, dcoord_dv.begin() + base, false);
        detail_quadelem::NodalDerivs<Real>(coord_, order, du_, dv_);
      }
    }
    { // Node positions and normals returned by GetNodeCoord
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
    const Long nnode_per_elem = (Long)order * order;
    const Long Nu = u_param.Dim();
    const Long Nv = v_param.Dim();
    const Long N = Nu * Nv;

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
      Vector<Real> Mu_(order * Nu, Mu_buf.begin(), false);
      Vector<Real> Mv_(order * Nv, Mv_buf.begin(), false);
      LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_param);
      LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_param);
      for (Integer i = 0; i < order; i++) for (Long a = 0; a < Nu; a++) MuT_buf[a * order + i] = Mu_buf[i * Nu + a];
    }
    const Matrix<Real> MuT(Nu, order, MuT_buf.begin(), false);
    const Matrix<Real> Mv(order, Nv, Mv_buf.begin(), false);

    const Long base = elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM;
    if (X) { // Positions
      const Vector<Real> coord_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)coord.begin() + base, false);
      ScratchBuf<Real> X_soa_buf(N * detail_quadelem::COORD_DIM);
      Vector<Real> X_soa(X_soa_buf);
      detail_quadelem::EvalTensorProduct(X_soa, coord_, MuT, Mv);
      for (Long i = 0; i < N; i++) {
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
      for (Long i = 0; i < N; i++) {
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
    const Long nnode_per_elem = (Long)order * order;
    const Long Nnode = nelem * nnode_per_elem;
    { // Size the outputs
      if (X.Dim() != Nnode * detail_quadelem::COORD_DIM) X.ReInit(Nnode * detail_quadelem::COORD_DIM);
      if (Xn.Dim() != Nnode * detail_quadelem::COORD_DIM) Xn.ReInit(Nnode * detail_quadelem::COORD_DIM);
      if (wts.Dim() != Nnode) wts.ReInit(Nnode);
      if (dist_far.Dim() != Nnode) dist_far.ReInit(Nnode);
      if (element_wise_node_cnt.Dim() != nelem) element_wise_node_cnt.ReInit(nelem);
      element_wise_node_cnt = nnode_per_elem;
    }

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
          const Long p = i * order + j;
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

  namespace detail_dyadic_near {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::MaxRefineLvl;
    using detail_quadelem::MaxTableOrder;
    using detail_quadelem::PrecompReal;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::EvalPoint;
    using detail_quadelem::GetClosestPoint;
    using detail_quadelem::IntegrateTensorRule;
    using detail_quadelem::LagrangeDiffMat;
    using detail_quadelem::NearInteracTargets;
    using detail_quadelem::QuadParamsForDigits;
    using detail_quadelem::ShiftedElemCoord;

    template <class Real> void QuadParams(const Real tol, Real& b_ellipse, Integer& quad_order) {
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real d = -log<Real>(tol_)/log<Real>((Real)10);
      const Real rho = std::min<Real>(3, std::max<Real>(2, 2 + (Real)0.25*(d - 6)));
      const Real C = std::max<Real>((Real)1e-3, (15*(rho*rho - 1))/64);
      quad_order = std::max<Integer>(2, (Integer)ceil<Real>(-log<Real>(C*tol_)/log<Real>(rho)*(Real)0.5 + 1));

      const Real a = (rho + 1/rho)/2, b = (rho - 1/rho)/2;
      b_ellipse = b*b/(2*a);
    }

    /** Returns the Gauss-Legendre order of each piece for each digits, from the dyadic QuadParams. */
    template <class Real> inline Integer QuadOrder(const Integer digits) {
      return QuadParamsForDigits<Real, QuadParams<Real>>(digits).quad_order;
    }

    /** Returns b_ellipse for each digits, from the dyadic QuadParams; pieces larger than dist/b_ellipse are split. */
    template <class Real> inline Real BEllipse(const Integer digits) {
      return QuadParamsForDigits<Real, QuadParams<Real>>(digits).b_ellipse;
    }

    template <class Real> struct GradeRule : QuadRule1D<Real> {
      Real a, b;
    };

    static constexpr Integer NearMaxQuadOrder = 60;

    /** Returns 'order' nodes on [0, 1] for each order: sin^2(pi i/(2(order-1))), the Chebyshev extreme points. */
    template <class Real> static const Vector<Real>& NearSubNodes(const Integer order) {
      SCTL_ASSERT(1 < order && order <= MaxTableOrder);
      static const Vector<Vector<Real>> all = []() {
        Vector<Vector<Real>> v(MaxTableOrder + 1);
        for (Integer n = 2; n <= MaxTableOrder; n++) {
          v[n].ReInit(n);
          using W = PrecompReal;
          for (Integer i = 0; i < n; i++) {
            const W sh = sin<W>(const_pi<W>()*i/(2*(n-1)));
            v[n][i] = (Real)(sh*sh);
          }
          v[n][0] = 0;
          v[n][n-1] = 1;
        }
        return v;
      }();
      return all[order];
    }

    /** Returns 1.0 - NearSubNodes(order) for each 'order', computed as cos^2 so that it is accurate where it is small. */
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

    /** Returns an order x order matrix for each 'order'; entry (i, j) is the derivative of the i-th Lagrange basis function on NearSubNodes(order) at j-th node. */
    template <class Real> static const Matrix<Real>& NearSubDiffMat(const Integer order) {
      SCTL_ASSERT(1 < order && order <= MaxTableOrder);
      static const Vector<Matrix<Real>> all = []() {
        Vector<Matrix<Real>> D(MaxTableOrder + 1);
        for (Integer n = 2; n <= MaxTableOrder; n++) LagrangeDiffMat(D[n], NearSubNodes<Real>(n));
        return D;
      }();
      return all[order];
    }

    /** Returns 2*MaxRefineLvl rules for each (order, q): q-point Gauss-Legendre on the dyadic intervals [1-2^-k, 1-2^-(k+1)] and tails [1-2^-k, 1], each with order x q interpolation matrices. */
    template <Integer order, class Real> const Vector<GradeRule<Real>>& NearGradeTable(const Integer q) {
      const auto build = [](const Integer q) {
        using W = PrecompReal;
        const Vector<W>& sig = NearSubOffsets<W>(order);
        const Matrix<W>& Dsub = NearSubDiffMat<W>(order);
        Vector<W> qn, qw;
        LegQuadRule<W>::template ComputeNdsWts<W>(&qn, &qw, q);
        Vector<GradeRule<Real>> tab(2*MaxRefineLvl<Real>);
        Vector<W> tq(q), Twts((Long)order*q);
        Matrix<W> dT(order, q);
        const auto fill = [&qn, &qw, q, &tq, &Twts, &sig, &Dsub, &dT](GradeRule<Real>& r, const Real a, const Real b) {
          r.a = a;
          r.b = b;
          const W aw = (W)a, w = (W)b - (W)a;
          r.w.ReInit(q);
          for (Integer i = 0; i < q; i++) r.w[i] = (Real)(w*qw[i]);
          const W t_hi = (W)1 - aw, t_w = t_hi - ((W)1 - (W)b);
          for (Integer j = 0; j < q; j++) tq[j] = t_hi - t_w*qn[j];
          LagrangeInterp<W>::Interpolate(Twts, sig, tq);
          const Matrix<W> T(order, q, Twts.begin(), false);
          Matrix<W>::GEMM(dT, Dsub, T);
          r.M.ReInit(order, q);
          r.dM.ReInit(order, q);
          r.MT.ReInit(q, order);
          r.dMT.ReInit(q, order);
          for (Integer i = 0; i < order; i++) for (Integer j = 0; j < q; j++) {
            r.M[i][j] = (Real)T[i][j];
            r.dM[i][j] = (Real)dT[i][j];
            r.MT[j][i] = r.M[i][j];
            r.dMT[j][i] = r.dM[i][j];
          }
        };
        for (Integer k = 0; k < MaxRefineLvl<Real>; k++) { // Rules on dyadic interval k and on its tail
          const Real lo = 1 - pow<Real>((Real)0.5, k), hi = 1 - pow<Real>((Real)0.5, k+1);
          fill(tab[k], lo, hi);
          fill(tab[MaxRefineLvl<Real> + k], lo, (Real)1);
        }
        return tab;
      };
      static const std::vector<Vector<GradeRule<Real>>> all = [&build]() {
        std::vector<Vector<GradeRule<Real>>> t(NearMaxQuadOrder+1);
        for (Integer q = 4; q <= NearMaxQuadOrder; q += 4) t[q] = build(q);
        for (Integer d = 0; d < MaxDigits<Real>; d++) {
          const Integer qi = QuadOrder<Real>(d);
          if (qi > 0 && qi <= NearMaxQuadOrder && t[qi].Dim() == 0) t[qi] = build(qi);
        }
        return t;
      }();
      SCTL_ASSERT(q > 0 && q <= NearMaxQuadOrder && all[q].Dim());
      return all[q];
    }

    template <Integer order, class Real, class Kernel> void NearInteracBlockDyadic(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Long nnode = (Long)order*order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full/COORD_DIM : KDIM1full;
      const Integer C = KDIM0*KDIM1_out;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
      M_acc.SetZero();

      Real ustar, vstar;
      const Real dist = GetClosestPoint(qel, ustar, vstar, elem_idx, Xtrg);
      const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};

      Real spd_u, spd_v;
      Integer q_near;
      { // Speeds, and the quadrature order raised for skewed tangents
        Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
        EvalPoint<Real>(qel, Xc, dXu, dXv, ustar, vstar, elem_idx, nullptr);
        Real guu = 0, gvv = 0, guv = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          guu += dXu[k]*dXu[k];
          gvv += dXv[k]*dXv[k];
          guv += dXu[k]*dXv[k];
        }
        spd_u = sqrt<Real>(guu);
        spd_v = sqrt<Real>(gvv);

        const Integer q_iso = QuadOrder<Real>(digits);
        q_near = q_iso;
        const Real den = sqrt<Real>(guu*gvv);
        if (den > 0) {
          const Real c = std::min<Real>(1, fabs<Real>(guv)/den);
          const Real phi = acos<Real>(c)*180/const_pi<Real>();
          const Real Ck = 400;
          const Real f = std::max<Real>(1, Ck/(10*std::max<Real>((Real)1e-3, phi)));
          if (f > 1) {
            Integer q = (Integer)ceil<Real>(f*q_iso);
            q = ((q + 3)/4)*4;
            q_near = std::min<Integer>(NearMaxQuadOrder, std::max<Integer>(q_iso, q));
          }
        }
      }

      ScratchBuf<Real> Sf_buf(4*nnode), St_buf(4*nnode);
      { // Interpolation from the element to each side's sub-interval
        const Vector<Real>& gnds = QuadElemList<Real>::ParamNodes(order);
        const Vector<Real>& soff = NearSubOffsets<Real>(order);
        ScratchBuf<Real> gsh_buf(order), sub_buf(order);
        Vector<Real> gsh(gsh_buf), sub(sub_buf);
        for (Integer d = 0; d < 2; d++) {
          const Real xs = (d ? vstar : ustar);
          for (Integer i = 0; i < order; i++) gsh[i] = gnds[i] - xs;
          for (Integer sd = 0; sd < 2; sd++) {
            if (!(slen[d][sd] > 0)) continue;
            const Real sg = (sd ? slen[d][sd] : -slen[d][sd]);
            for (Integer i = 0; i < order; i++) sub[i] = sg*soff[i];
            Vector<Real> Sf_v(nnode, Sf_buf.begin() + (2*d+sd)*nnode, false);
            LagrangeInterp<Real>::Interpolate(Sf_v, gsh, sub);
            const Matrix<Real> Sf_m(order, order, Sf_buf.begin() + (2*d+sd)*nnode, false);
            Matrix<Real> St_m(order, order, St_buf.begin() + (2*d+sd)*nnode, false);
            for (Integer i = 0; i < order; i++) for (Integer aa = 0; aa < order; aa++) St_m[aa][i] = Sf_m[i][aa];
          }
        }
      }

      ScratchBuf<Real> Xsub_buf(4*COORD_DIM*nnode);
      { // Coordinates of the sub-rectangles, relative to the target
        ScratchBuf<Real> cs_buf(COORD_DIM*nnode);
        Vector<Real> cs(cs_buf);
        ShiftedElemCoord(cs, qel, elem_idx, Xtrg);
        ScratchBuf<Real> Av(2*COORD_DIM*nnode);
        for (Integer sdv = 0; sdv < 2; sdv++) {
          if (!(slen[1][sdv] > 0)) continue;
          const Matrix<Real> cs_all(COORD_DIM*order, order, cs.begin(), false);
          const Matrix<Real> Sf_v(order, order, Sf_buf.begin() + (2+sdv)*nnode, false);
          Matrix<Real> A_all(COORD_DIM*order, order, Av.begin() + sdv*COORD_DIM*nnode, false);
          Matrix<Real>::GEMM(A_all, cs_all, Sf_v);
        }
        for (Integer sdu = 0; sdu < 2; sdu++) {
          if (!(slen[0][sdu] > 0)) continue;
          const Matrix<Real> St_u(order, order, St_buf.begin() + sdu*nnode, false);
          for (Integer sdv = 0; sdv < 2; sdv++) {
            if (!(slen[1][sdv] > 0)) continue;
            for (Integer k = 0; k < COORD_DIM; k++) {
              const Matrix<Real> A_k(order, order, Av.begin() + (sdv*COORD_DIM + k)*nnode, false);
              Matrix<Real> X_k(order, order, Xsub_buf.begin() + ((2*sdu+sdv)*COORD_DIM + k)*nnode, false);
              Matrix<Real>::GEMM(X_k, St_u, A_k);
            }
          }
        }
      }

      ScratchBuf<Real> acc_buf((Long)C*nnode);
      Vector<Real> acc(acc_buf);
      const Vector<GradeRule<Real>>& tab = NearGradeTable<order,Real>(q_near);
      const auto integrate_piece = [&tab, &normal_trg, &ker, &proxy_off, &proxy_w, &acc, &Xsub_buf](const Integer sdu, const Integer sdv, const Integer iu, const Integer iv) {
        const GradeRule<Real>& gu = tab[iu];
        const GradeRule<Real>& gv = tab[iv];
        if (!(gu.b > gu.a) || !(gv.b > gv.a)) return;
        const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
        const Long nsub = COORD_DIM*(Long)order*order;
        const Vector<Real> Xsub(nsub, Xsub_buf.begin() + (2*sdu+sdv)*nsub, false);
        IntegrateTensorRule<order,Real>(acc, Xsub, gu, gv, normal_trg, ker, nsign, proxy_off, proxy_w);
      };
      const Real b_ellipse = BEllipse<Real>(digits);
      const auto refine = [&integrate_piece, dist, b_ellipse](const Integer sdu, const Integer sdv, Real hu, Real hv) {
        Integer ku = 0, kv = 0;
        const bool refine_to_max = !(dist > 0) || isinf<Real>(dist) || isnan<Real>(dist);
        constexpr Integer KMAX = MaxRefineLvl<Real>-1;
        while ((refine_to_max || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
          if (hu >= hv && ku < KMAX) {
            integrate_piece(sdu, sdv, ku, MaxRefineLvl<Real> + kv);
            ku++;
            hu *= (Real)0.5;
          } else if (kv < KMAX) {
            integrate_piece(sdu, sdv, MaxRefineLvl<Real> + ku, kv);
            kv++;
            hv *= (Real)0.5;
          } else if (ku < KMAX) {
            integrate_piece(sdu, sdv, ku, MaxRefineLvl<Real> + kv);
            ku++;
            hu *= (Real)0.5;
          } else break;
        }
        integrate_piece(sdu, sdv, MaxRefineLvl<Real> + ku, MaxRefineLvl<Real> + kv);
      };
      for (Integer sdu = 0; sdu < 2; sdu++) { // Integrate each sub-rectangle, refining toward the closest point
        if (!(slen[0][sdu] > 0)) continue;
        for (Integer sdv = 0; sdv < 2; sdv++) {
          if (!(slen[1][sdv] > 0)) continue;
          acc.SetZero();
          refine(sdu, sdv, slen[0][sdu]*spd_u, slen[1][sdv]*spd_v);
          { // Map the sub-rectangle's nodal values back to the element
            ScratchBuf<Real> accB((Long)C*nnode), accE(nnode);
            const Matrix<Real> St_v(order, order, St_buf.begin() + (2+sdv)*nnode, false);
            const Matrix<Real> Sf_u(order, order, Sf_buf.begin() + sdu*nnode, false);
            const Matrix<Real> A_all((Long)C*order, order, acc.begin(), false);
            Matrix<Real> B_all((Long)C*order, order, accB.begin(), false);
            Matrix<Real>::GEMM(B_all, A_all, St_v);
            for (Integer c = 0; c < C; c++) {
              const Matrix<Real> B_c(order, order, accB.begin() + (Long)c*nnode, false);
              Matrix<Real> E_c(order, order, accE.begin(), false);
              Matrix<Real>::GEMM(E_c, Sf_u, B_c);
              for (Long p = 0; p < nnode; p++) M_acc[p][c] += accE[p];
            }
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracDyadic(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      const auto near_interac_one_trg = [&qel, elem_idx, &ker, digits, &proxy_off, &proxy_w](Matrix<Real>& M_acc, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
        NearInteracBlockDyadic<order,Real>(M_acc, qel, elem_idx, Xtrg, ntrg, ker, digits, proxy_off, proxy_w);
      };
      NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel, near_interac_one_trg);
    }

  }

  namespace detail_tensorprod_near {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::MaxRefineLvl;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::DiffMat;
    using detail_quadelem::EvalPoint;
    using detail_quadelem::GetClosestPoint;
    using detail_quadelem::IntegratePanel;
    using detail_quadelem::NearInteracTargets;
    using detail_quadelem::QuadParamsForDigits;

    template <class Real> void QuadParams(const Real tol, Real& b_ellipse, Integer& quad_order) {
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real rho = (Real)2.5;
      b_ellipse = (rho + 1/rho) / 4;
      quad_order = std::max<Integer>(1, (Integer)ceil<Real>(-log<Real>(((15*(rho*rho-1))/64)*tol_)/log<Real>(rho)*(Real)0.5 + 1));
    }

    /** Returns the Gauss-Legendre order of each segment for each digits, from the tensor-product QuadParams. */
    template <class Real> inline Integer QuadOrder(const Integer digits) {
      return QuadParamsForDigits<Real, QuadParams<Real>>(digits).quad_order;
    }

    /** Returns b_ellipse for each digits (0.725 for all); it sets the segment grading ratio and the smallest segment. */
    template <class Real> inline Real BEllipse(const Integer digits) {
      return QuadParamsForDigits<Real, QuadParams<Real>>(digits).b_ellipse;
    }

    /** Returns QuadOrder(digits) Gauss-Legendre nodes and weights on [0, 1] for each digits. */
    template <class Real> const std::pair<Vector<Real>, Vector<Real>>& GLRule(const Integer digits) {
      static const std::array<std::pair<Vector<Real>, Vector<Real>>,MaxDigits<Real>> gl = []() {
        std::array<std::pair<Vector<Real>, Vector<Real>>,MaxDigits<Real>> t;
        for (Integer d = 0; d < MaxDigits<Real>; d++) LegQuadRule<Real>::ComputeNdsWts(&t[d].first, &t[d].second, QuadOrder<Real>(d));
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real>);
      return gl[digits];
    }

    template <class Real> void ExpandSegments(Vector<Real>& param, Vector<Real>& w, const Vector<Real>& seg, const Vector<Real>& qnds, const Vector<Real>& qwts) {
      const Integer quad_order = qnds.Dim();
      const Long nseg = seg.Dim()/2;
      const Long N = nseg * quad_order;
      if (param.Dim() != N) param.ReInit(N);
      if (w.Dim() != N) w.ReInit(N);
      Long idx = 0;
      for (Long si = 0; si < nseg; si++) {
        const Real a0 = seg[si*2+0], a1 = seg[si*2+1];
        const Real len = a1 - a0;
        for (Integer a = 0; a < quad_order; a++) {
          param[idx] = a0 + len*qnds[a];
          w[idx] = qwts[a]*len;
          idx++;
        }
      }
    }

    static constexpr Long MaxSegments = 4096;

    template <class Real> Long BuildGradedSegments1D(Iterator<Real> seg, const Real center, const Real b_ellipse, const Real w_min) {
      const Real r = std::min<Real>((Real)0.9, b_ellipse/(1 + b_ellipse) * (Real)1.05);
      const Real w_stop = std::max<Real>(w_min, (Real)1e-300);

      Long nseg = 0;
      const auto add = [&seg, &nseg](const Real e0, const Real e1) {
        const Real lo = std::min<Real>(e0, e1);
        const Real hi = std::max<Real>(e0, e1);
        if (!(hi - lo > 0)) return;
        SCTL_ASSERT(nseg < MaxSegments);
        seg[2*nseg+0] = lo;
        seg[2*nseg+1] = hi;
        nseg++;
      };
      for (Integer side = 0; side < 2; side++) {
        const Real sgn = (side ? (Real)1 : (Real)-1);
        const Real span = (side ? 1 - center : center);
        if (!(span > 0)) continue;

        Real w = span;
        while (w > w_stop) {
          const Real w_next = w*r;
          add(center + sgn*w, center + sgn*w_next);
          w = w_next;
        }
        add(center + sgn*w, center);
      }
      return nseg;
    }

    template <class Real> void BuildNearSegments(Iterator<Real> useg, Long& nseg_u, Iterator<Real> vseg, Long& nseg_v, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse) {
      Real ustar, vstar, h_param;
      { // Closest point, and its distance in parameter units
        const Real dist = GetClosestPoint(qel, ustar, vstar, elem_idx, Xtrg);
        Real Xc[COORD_DIM], dXdu[COORD_DIM], dXdv[COORD_DIM];
        EvalPoint<Real>(qel, Xc, dXdu, dXdv, ustar, vstar, elem_idx, nullptr);
        Real su2 = 0, sv2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          su2 += dXdu[k]*dXdu[k];
          sv2 += dXdv[k]*dXdv[k];
        }
        const Real L_phys = std::max<Real>(sqrt<Real>(su2), sqrt<Real>(sv2));
        const bool degenerate = !(dist > 0) || isinf<Real>(dist) || isnan<Real>(dist) || !(L_phys > 0);
        h_param = (degenerate ? 0 : dist/L_phys);
      }

      const Real w_floor = pow<Real>((Real)0.5, MaxRefineLvl<Real>);
      const Real w_min = std::max<Real>(h_param/b_ellipse, w_floor);
      nseg_u = BuildGradedSegments1D<Real>(useg, ustar, b_ellipse, w_min);
      nseg_v = BuildGradedSegments1D<Real>(vseg, vstar, b_ellipse, w_min);
    }

    template <Integer order, class Real, class Kernel> void NearInteracBlockTensorProduct(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();

      ScratchBuf<Real> useg(2*MaxSegments), vseg(2*MaxSegments);
      Long nseg_u, nseg_v;
      BuildNearSegments<Real>(useg.begin(), nseg_u, vseg.begin(), nseg_v, qel, elem_idx, Xtrg, BEllipse<Real>(digits));
      const std::pair<Vector<Real>, Vector<Real>>& gl = GLRule<Real>(digits);
      const Long Nu = nseg_u * gl.first.Dim();
      const Long Nv = nseg_v * gl.first.Dim();
      if (!Nu || !Nv) return;

      ScratchBuf<Real> rule_u(Nu*(1 + 4*order)), rule_v(Nv*(1 + 4*order));
      const auto rule_view = [](Iterator<Real> buf, const Long N) {
        return QuadRule1D<Real>{Vector<Real>(N, buf, false),
            Matrix<Real>(order, N, buf + N, false), Matrix<Real>(order, N, buf + N*(1 + order), false),
            Matrix<Real>(N, order, buf + N*(1 + 2*order), false), Matrix<Real>(N, order, buf + N*(1 + 3*order), false)};
      };
      QuadRule1D<Real> ru = rule_view(rule_u.begin(), Nu);
      QuadRule1D<Real> rv = rule_view(rule_v.begin(), Nv);
      { // Gauss-Legendre rule on each segment, and its interpolation matrices
        const auto build_rule = [&gl](QuadRule1D<Real>& r, Vector<Real>& param, Iterator<Real> seg, const Long nseg) {
          const Vector<Real> seg_v(2*nseg, seg, false);
          ExpandSegments<Real>(param, r.w, seg_v, gl.first, gl.second);
          const Long N = param.Dim();
          Vector<Real> M_v(order*N, r.M.begin(), false);
          LagrangeInterp<Real>::Interpolate(M_v, QuadElemList<Real>::ParamNodes(order), param);
          Matrix<Real>::GEMM(r.dM, DiffMat<Real>(order), r.M);
          for (Integer i = 0; i < order; i++) {
            for (Long a = 0; a < N; a++) {
              r.MT[a][i] = r.M[i][a];
              r.dMT[a][i] = r.dM[i][a];
            }
          }
        };
        ScratchBuf<Real> param_u(Nu), param_v(Nv);
        Vector<Real> u_param(param_u), v_param(param_v);
        build_rule(ru, u_param, useg.begin(), nseg_u);
        build_rule(rv, v_param, vseg.begin(), nseg_v);
      }
      IntegratePanel<order,Real>(M_acc, qel, elem_idx, Xtrg, normal_trg, ru, rv, ker);
    }

    template <Integer order, class Real, class Kernel> void NearInteracTensorProduct(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits) {
      const auto near_interac_one_trg = [&qel, elem_idx, &ker, digits](Matrix<Real>& M_acc, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
        NearInteracBlockTensorProduct<order,Real>(M_acc, qel, elem_idx, Xtrg, ntrg, ker, digits);
      };
      NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel, near_interac_one_trg);
    }

  }

  namespace detail_duffy {

    using detail_quadelem::COORD_DIM;

    using detail_quadelem::DiffMat;
    using detail_quadelem::ScatterTargetBlock;
    using detail_quadelem::SelfInteracElems;
    using detail_quadelem::ShiftedElemCoord;
    using detail_quadelem::WeightedKernel;
    using detail_dyadic_near::NearGradeTable;
    using detail_dyadic_near::QuadOrder;

    template <class Real> struct DuffyTri {
      bool swap_ab = false;
      Real nsign = 1;
      Real J0 = 0;
      Matrix<Real> beta_interp;
      Matrix<Real> beta_interp_T;
      Vector<Matrix<Real>> alpha_interp, alpha_interp_T;
    };
    template <class Real> struct DuffySelfTable {
      Integer ns = 0;
      Vector<Real> sn, sw;
      std::vector<DuffyTri<Real>> tri;
    };

    inline Integer DuffyTRuleOrder(const Integer digits, const Integer order, const Integer kdim0) {
      const Integer nt = (kdim0 > 1 ? 4*digits : (5*digits + 1)/2);
      return std::max<Integer>(order/2, nt);
    }

    /** Returns, for each order, an order-point radial rule and, for each of the 4*order^2 (node, triangle) pairs, its Jacobian, orientation and interpolation matrices along beta and alpha. */
    template <Integer order, class Real> const DuffySelfTable<Real>& DuffyTable() {
      static const DuffySelfTable<Real> table = []() {
        DuffySelfTable<Real> tbl;
        const Integer qs = order;
        tbl.ns = qs;
        LegQuadRule<Real>::ComputeNdsWts(&tbl.sn, &tbl.sw, qs);

        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Matrix<Real>& D = DiffMat<Real>(order);
        tbl.tri.resize((size_t)(4*order*order));
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
            { // Interpolation along beta, with its derivative, at the s-nodes
              Vector<Real> beta_vals(qs);
              for (Integer i = 0; i < qs; i++) beta_vals[i] = beta0 + tbl.sn[i]*a_beta;
              Matrix<Real> Mbeta(order, qs), dMbeta(order, qs);
              Vector<Real> Mbeta_v((Long)order*qs, Mbeta.begin(), false);
              LagrangeInterp<Real>::Interpolate(Mbeta_v, nds, beta_vals);
              Matrix<Real>::GEMM(dMbeta, D, Mbeta);
              T.beta_interp.ReInit(order, 2*qs);
              for (Integer r = 0; r < order; r++) for (Integer i = 0; i < qs; i++) {
                T.beta_interp[r][i] = Mbeta[r][i];
                T.beta_interp[r][qs+i] = dMbeta[r][i];
              }
              T.beta_interp_T = Mbeta.Transpose();
            }
            { // Interpolation along alpha, with its derivative, at each s-node
              T.alpha_interp.ReInit(qs);
              T.alpha_interp_T.ReInit(qs);
              Vector<Real> alpha_vals(order);
              Matrix<Real> Malpha(order, order), dMalpha(order, order);
              Vector<Real> Malpha_v((Long)order*order, Malpha.begin(), false);
              for (Integer i = 0; i < qs; i++) {
                for (Integer k = 0; k < order; k++) alpha_vals[k] = alpha0 + tbl.sn[i]*(a_alpha + nds[k]*e_alpha);
                LagrangeInterp<Real>::Interpolate(Malpha_v, nds, alpha_vals);
                Matrix<Real>::GEMM(dMalpha, D, Malpha);
                T.alpha_interp[i].ReInit(order, 2*order);
                for (Integer r = 0; r < order; r++) for (Integer k = 0; k < order; k++) {
                  T.alpha_interp[i][r][k] = Malpha[r][k];
                  T.alpha_interp[i][r][order+k] = dMalpha[r][k];
                }
                T.alpha_interp_T[i] = Malpha.Transpose();
              }
            }
          }
        }
        return tbl;
      }();
      return table;
    }

    template <Integer order, class Real, class Kernel> void SelfInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order*order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full/COORD_DIM : KDIM1full;
      const Integer C = KDIM0*KDIM1_out;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
      M_acc.SetZero();

      const DuffySelfTable<Real>& tbl = DuffyTable<order,Real>();
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      ScratchBuf<Real> cs_buf(COORD_DIM*nnode);
      Vector<Real> cs(cs_buf);
      ShiftedElemCoord(cs, qel, elem_idx, Xtrg);

      Real G[4];
      { // Metric of the element at the target node
        const Matrix<Real>& D = DiffMat<Real>(order);
        Real du[COORD_DIM], dv[COORD_DIM];
        for (Integer k = 0; k < COORD_DIM; k++) {
          Real su = 0, sv = 0;
          for (Integer i = 0; i < order; i++) su += cs[k*nnode + (Long)i*order + tj]*D[i][ti];
          for (Integer j = 0; j < order; j++) sv += cs[k*nnode + (Long)ti*order + j]*D[j][tj];
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

      const Long ns = tbl.ns;
      const Long nt = DuffyTRuleOrder(digits, order, KDIM0);
      const Long nq = ns*nt;
      for (Integer kt = 0; kt < 4; kt++) { // Triangles joining the target node to each edge
        const DuffyTri<Real>& T = tbl.tri[(size_t)((ti*order + tj)*4 + kt)];

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

        ScratchBuf<Real> tw(nt), Tt_buf((Long)order*nt), TtT_buf((Long)nt*order);
        { // Rule along t, graded toward the closest edge point
          static constexpr Integer MaxGLOrder = 128;
          const Vector<Real>& qn = LegQuadRule<Real>::template nds<MaxGLOrder>(nt);
          const Vector<Real>& qw = LegQuadRule<Real>::template wts<MaxGLOrder>(nt);
          const auto arcsinh = [](const Real x) { return log<Real>(x + sqrt<Real>(x*x + (Real)1)); };
          ScratchBuf<Real> tn_buf(nt);
          Vector<Real> tn(tn_buf);
          const Real x0 = -arcsinh(tstar/dOverL), x1 = arcsinh(((Real)1-tstar)/dOverL);
          for (Long i = 0; i < nt; i++) {
            const Real xi = x0 + (x1-x0)*qn[i];
            const Real ex = exp<Real>(xi), iex = (Real)1/ex;
            tn[i] = tstar + dOverL*(ex-iex)/(Real)2;
            tw[i] = dOverL*(ex+iex)/(Real)2*(x1-x0)*qw[i];
          }
          Vector<Real> Tt_v((Long)order*nt, Tt_buf.begin(), false);
          LagrangeInterp<Real>::Interpolate(Tt_v, nds, tn);
          for (Integer r = 0; r < order; r++) for (Long j = 0; j < nt; j++) TtT_buf[j*order + r] = Tt_buf[r*nt + j];
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
            ScratchBuf<Real> FS_buf(COORD_DIM*nnode), Gm_buf(2*COORD_DIM*(Long)order*ns);
            ScratchBuf<Real> As_buf((Long)NA*order), Tmp_buf(2*(Long)NA*order), HG_buf(ns*NR*(Long)order);
            Matrix<Real> FS(COORD_DIM*order, order, FS_buf.begin(), false);
            Matrix<Real> Gm(COORD_DIM*order, 2*ns, Gm_buf.begin(), false);
            Matrix<Real> As(NA, order, As_buf.begin(), false);
            Matrix<Real> Tmp(NA, 2*order, Tmp_buf.begin(), false);
            Matrix<Real> HG(ns*NR, order, HG_buf.begin(), false);
            for (Integer k = 0; k < COORD_DIM; k++)
              for (Integer i = 0; i < order; i++) for (Integer j = 0; j < order; j++)
                FS[k*order + (T.swap_ab ? j : i)][T.swap_ab ? i : j] = cs[k*nnode + (Long)i*order + j];
            Matrix<Real>::GEMM(Gm, FS, T.beta_interp);
            for (Long i = 0; i < ns; i++) {
              for (Integer k = 0; k < COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
                As[k][m] = Gm[k*order+m][i];
                As[COORD_DIM+k][m] = Gm[k*order+m][ns+i];
              }
              Matrix<Real>::GEMM(Tmp, As, T.alpha_interp[i]);
              for (Integer k = 0; k < COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
                HG[i*NR + k][m]               = Tmp[k][m];
                HG[i*NR + COORD_DIM + k][m]   = Tmp[k][order+m];
                HG[i*NR + 2*COORD_DIM + k][m] = Tmp[COORD_DIM+k][m];
              }
            }
            Matrix<Real>::GEMM(XdX, HG, Tt);
          }
          for (Long i = 0; i < ns; i++) {
            const Real jw = tbl.sn[i]*T.J0*tbl.sw[i];
            for (Long j = 0; j < nt; j++) {
              const Long q = i*nt + j;
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

        ScratchBuf<Real> KW_buf((Long)C*nq);
        { // Weighted kernel at the rule's points
          StaticArray<Real,COORD_DIM> Xt0{0,0,0};
          const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);
          WeightedKernel<Real>(KW_buf.begin(), Xt0_v.begin(), Xs.begin(), Xn.begin(), wq.begin(), ns*nt, nt, (Real)1, false, normal_trg, ker);
        }

        { // Project onto the element nodes and add into M_acc
          ScratchBuf<Real> Zall_buf(ns*(Long)C*order), Yi_buf((Long)C*order), Yall_buf((Long)C*order*ns), Pc_buf(nnode);
          const Matrix<Real> KW(ns*C, nt, KW_buf.begin(), false);
          Matrix<Real> Zall(ns*C, order, Zall_buf.begin(), false);
          Matrix<Real> Yi(C, order, Yi_buf.begin(), false);
          Matrix<Real> Yall(C*order, ns, Yall_buf.begin(), false);
          Matrix<Real> Pc(order, order, Pc_buf.begin(), false);
          Matrix<Real>::GEMM(Zall, KW, TtT);
          for (Long i = 0; i < ns; i++) {
            const Matrix<Real> Zi(C, order, (Iterator<Real>)Zall.begin() + i*(Long)C*order, false);
            Matrix<Real>::GEMM(Yi, Zi, T.alpha_interp_T[i]);
            for (Integer c = 0; c < C; c++) for (Integer m = 0; m < order; m++) Yall[c*order+m][i] = Yi[c][m];
          }
          for (Integer c = 0; c < C; c++) {
            const Matrix<Real> Yc(order, ns, (Iterator<Real>)Yall.begin() + (Long)c*order*ns, false);
            Matrix<Real>::GEMM(Pc, Yc, T.beta_interp_T);
            for (Integer m = 0; m < order; m++) for (Integer n = 0; n < order; n++)
              M_acc[T.swap_ab ? (Long)n*order+m : (Long)m*order+n][c] += Pc[m][n];
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void SelfInteracDuffy(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      DuffyTable<order,Real>(); // precomp cache
      NearGradeTable<order,Real>(QuadOrder<Real>(digits)); // precomp cache
      const auto self_interac_one_trg = [&qel, &ker, digits](Matrix<Real>& M, const Long elem_idx, const Long t, const Integer ti, const Integer tj,
          const Vector<Real>& Xtrg, const Vector<Real>& ntrg, const Vector<Real>&, const Vector<Real>&,
          const Vector<Real>&, const Vector<Real>&, const Integer KDIM1_out) {
        ScratchBuf<Real> M_acc_buf((Long)order*order*Kernel::SrcDim()*KDIM1_out);
        Matrix<Real> M_acc((Long)order*order, Kernel::SrcDim()*KDIM1_out, M_acc_buf.begin(), false);
        SelfInteracBlockDuffy<order,Real>(M_acc, qel, elem_idx, ti, tj, Xtrg, ntrg, ker, digits);
        ScatterTargetBlock(M, M_acc, t, KDIM1_out);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, false, self_interac_one_trg);
    }

  }

  namespace detail_hedgehog {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::PrecompReal;

    using detail_quadelem::ScatterTargetBlock;
    using detail_quadelem::SelfInteracElems;
    using detail_dyadic_near::NearGradeTable;
    using detail_dyadic_near::NearInteracDyadic;
    using detail_dyadic_near::QuadOrder;

    template <class Kernel, class = void> struct KernelSingularOrder {
      static constexpr Integer value = 2;
    };
    template <class Kernel> struct KernelSingularOrder<Kernel, std::void_t<decltype(Kernel::SingularOrder())>> {
      static constexpr Integer value = Kernel::SingularOrder();
    };

    /** Returns the 5 proxy distances 4^(j/4), j < 5, in units of rmin. */
    template <class Real> inline const Vector<Real>& HedgehogProxyOffsets() {
      static const Vector<Real> s = []() {
        Vector<Real> v;
        for (Integer j = 0; j < 5; j++) v.PushBack(pow<Real>((Real)4, (Real)j/(Real)4));
        return v;
      }();
      return s;
    }

    /** Returns the 5 weights that extrapolate values at the proxy distances to distance 0. */
    template <class Real> inline const Vector<Real>& HedgehogWeights() {
      static const Vector<Real> w = []() {
        using W = PrecompReal;
        const Vector<Real>& s = HedgehogProxyOffsets<Real>();
        const Long p = s.Dim();
        Vector<Real> wj(p);
        for (Long j = 0; j < p; j++) {
          W v = 1;
          for (Long k = 0; k < p; k++) if (k != j) v *= (0 - (W)s[k])/((W)s[j] - (W)s[k]);
          wj[j] = (Real)v;
        }
        return wj;
      }();
      return w;
    }

    /** Returns the proxy spacing coefficient 0.1*10^(-digits/6) for each digits, at most 3e-3 when sing_order > 1. */
    template <class Real> inline Real HedgehogRminCoeff(const Integer digits, const Integer sing_order) {
      static const std::array<Real,MaxDigits<Real>> c = []() {
        std::array<Real,MaxDigits<Real>> t{};
        for (Integer d = 0; d < MaxDigits<Real>; d++) t[d] = (Real)0.1 * pow<Real>(pow<Real,Long>((Real)0.1, (Long)d), (Real)1/(Real)6);
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real>);
      return (sing_order <= 1 ? c[digits] : std::min<Real>(c[digits], (Real)3e-3));
    }

    template <Integer order, class Real, class Kernel> void SelfInteracHedgehog(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      static constexpr Integer sing_order = KernelSingularOrder<Kernel>::value;
      const Integer near_digits = std::min<Integer>(MaxDigits<Real>-1, digits + (sing_order <= 1 ? 2 : 6));
      NearGradeTable<order,Real>(QuadOrder<Real>(near_digits)); // precomp cache
      NearGradeTable<order,Real>(QuadOrder<Real>(digits)); // precomp cache

      const Vector<Real>& hh_w = HedgehogWeights<Real>();
      const Real hh_c = HedgehogRminCoeff<Real>(digits, sing_order);
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      const auto self_interac_one_trg = [&nds, hh_c, &qel, &ker, near_digits, &hh_w](Matrix<Real>& M, const Long elem_idx, const Long t, const Integer ti, const Integer tj,
          const Vector<Real>&, const Vector<Real>& ntrg, const Vector<Real>& Xnodes, const Vector<Real>& Xnnodes,
          const Vector<Real>& dXu, const Vector<Real>& dXv, const Integer KDIM1_out) {
        const Vector<Real>& proxy_dist = HedgehogProxyOffsets<Real>();
        ScratchBuf<Real> hh_Xt1_buf(COORD_DIM), hh_off_buf(proxy_dist.Dim()*COORD_DIM);
        Vector<Real> hh_Xt1(hh_Xt1_buf), hh_off(hh_off_buf);
        { // Proxy points along the normal, sized by edge distance
          Real su2 = 0, sv2 = 0;
          for (Integer k = 0; k < COORD_DIM; k++) {
            su2 += dXu[t*COORD_DIM+k]*dXu[t*COORD_DIM+k];
            sv2 += dXv[t*COORD_DIM+k]*dXv[t*COORD_DIM+k];
          }
          const Real edge_u = std::min<Real>(nds[ti], 1-nds[ti]), edge_v = std::min<Real>(nds[tj], 1-nds[tj]);
          const Real rmin = hh_c * std::min<Real>(edge_u*sqrt<Real>(su2), edge_v*sqrt<Real>(sv2));
          for (Integer k = 0; k < COORD_DIM; k++) hh_Xt1[k] = Xnodes[t*COORD_DIM+k] + rmin*Xnnodes[t*COORD_DIM+k];
          for (Long j = 0; j < proxy_dist.Dim(); j++) {
            const Real rj = rmin*proxy_dist[j];
            for (Integer k = 0; k < COORD_DIM; k++) hh_off[j*COORD_DIM+k] = (rj-rmin)*Xnnodes[t*COORD_DIM+k];
          }
        }
        ScratchBuf<Real> M_hh_buf((Long)order*order*Kernel::SrcDim()*KDIM1_out);
        Matrix<Real> M_hh((Long)order*order*Kernel::SrcDim(), KDIM1_out, M_hh_buf.begin(), false);
        NearInteracDyadic<order,Real>(M_hh, hh_Xt1, ntrg, ker, elem_idx, qel, near_digits, hh_off, hh_w);
        ScatterTargetBlock(M, M_hh, t, KDIM1_out);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, true, self_interac_one_trg);
    }

  }

  namespace detail_tensorprod_singular {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::MaxRefineLvl;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::DiffMat;
    using detail_quadelem::IntegratePanel;
    using detail_quadelem::ScatterTargetBlock;
    using detail_quadelem::SelfInteracElems;
    using detail_tensorprod_near::GLRule;
    using detail_tensorprod_near::QuadOrder;

    template <Integer order, class Real> void LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& delta, const Integer ti) {
      const Long N = delta.Dim();
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
        for (Long a = 0; a < N; a++) {
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
      MT = M.Transpose();
      dMT = dM.Transpose();
    }

    template <class Real> void BuildCenteredGraded1D(Vector<Real>& delta, Vector<Real>& w, const Real u0, const Integer levels, const Vector<Real>& qnds, const Vector<Real>& qwts) {
      const Integer q = qnds.Dim();
      std::vector<Real> d_, w_;
      const auto side = [&d_, &w_, levels, q, &qnds, &qwts](const Real span, const Real sgn) {
        if (!(span > 0)) return;
        Real a = 0;
        for (Integer k = levels; k >= 0; k--) {
          const Real b = span * pow<Real>((Real)0.5, (Integer)k);
          const Real len = b - a;
          if (len > 0) {
            for (Integer i = 0; i < q; i++) {
              d_.push_back(sgn*(a + len*qnds[i]));
              w_.push_back(len*qwts[i]);
            }
          }
          a = b;
        }
      };
      side(1-u0, (Real)1);
      side(u0,   (Real)-1);
      const Long N = (Long)d_.size();
      delta.ReInit(N);
      w.ReInit(N);
      for (Long i = 0; i < N; i++) {
        delta[i] = d_[i];
        w[i] = w_[i];
      }
    }

    template <class Real> void BuildCenteredLogSingular1D(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer quad_order) {
      const Integer ord = 16;
      std::vector<Real> px, pw;
      const auto add_alpert = [&px, &pw, ord](const Real a, const Real b, const bool log_a, const bool log_b) {
        const auto L = (log_a ? QuadLogExtraPtNodes<Real>(ord) : QuadSmoothExtraPtNodes<Real>(ord));
        const auto R = (log_b ? QuadLogExtraPtNodes<Real>(ord) : QuadSmoothExtraPtNodes<Real>(ord));
        const Integer skipL = L.NodesToSkip, skipR = R.NodesToSkip;
        const Integer N = std::max<Integer>(skipL + skipR + 2, 2 * ord);
        const Integer N1 = N - 1;
        const Real h = (b - a) / (Real)N1;
        for (Integer i = skipL; i <= N1 - skipR; i++) {
          px.push_back(a + (Real)i*h);
          pw.push_back(h);
        }
        for (Integer i = 0; i < L.ExtraNodes.Dim(); i++) {
          px.push_back(a + L.ExtraNodes[i]*h);
          pw.push_back(L.ExtraWeights[i]*h);
        }
        for (Integer i = 0; i < R.ExtraNodes.Dim(); i++) {
          px.push_back(b - R.ExtraNodes[i]*h);
          pw.push_back(R.ExtraWeights[i]*h);
        }
      };
      Vector<Real> gnds, gwts;
      LegQuadRule<Real>::ComputeNdsWts(&gnds, &gwts, quad_order);
      const auto add_gl = [&px, &pw, quad_order, &gnds, &gwts](const Real a, const Real b) {
        const Real len = b - a;
        for (Integer i = 0; i < quad_order; i++) {
          px.push_back(a + len*gnds[i]);
          pw.push_back(len*gwts[i]);
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
      const Long N = (Long)px.size();
      delta.ReInit(N);
      w.ReInit(N);
      for (Long i = 0; i < N; i++) {
        delta[i] = px[i];
        w[i] = pw[i];
      }
    }

    template <class T, Integer N> class LazyTable {
      public:
        LazyTable() = default;
        LazyTable(const LazyTable&) = delete;
        LazyTable& operator=(const LazyTable&) = delete;
        ~LazyTable() {
          for (auto& p : slot) delete p.load(std::memory_order_relaxed);
        }

        template <class BuildFn> const T& Get(const Integer i, BuildFn build) {
          SCTL_ASSERT(i >= 0 && i < N);
          T* p = slot[i].load(std::memory_order_acquire);
          if (!p) {
            std::lock_guard<std::mutex> lk(mtx);
            p = slot[i].load(std::memory_order_relaxed);
            if (!p) {
              p = new T(build());
              slot[i].store(p, std::memory_order_release);
            }
          }
          return *p;
        }

      private:
        std::atomic<T*> slot[N] = {};
        std::mutex mtx;
    };

    /** Returns the rule toward node ti along u for each (order, digits): QuadOrder(digits)-point Gauss-Legendre on intervals halving min(MaxRefineLvl, 2*digits+6) times toward the node on each side, with order x N interpolation matrices. */
    template <Integer order, class Real> const QuadRule1D<Real>& CenteredURule(const Integer ti, const Integer digits) {
      static LazyTable<Vector<QuadRule1D<Real>>, MaxDigits<Real>> table;
      const auto build = [digits]() {
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Integer quad_order = QuadOrder<Real>(digits);
        const Integer Lvl = std::min<Integer>(MaxRefineLvl<Real>, 2*digits + 6);
        Vector<Real> qnds, qwts;
        LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, quad_order);
        Vector<QuadRule1D<Real>> rules(order);
        for (Integer i = 0; i < order; i++) {
          Vector<Real> delta;
          BuildCenteredGraded1D(delta, rules[i].w, nds[i], Lvl, qnds, qwts);
          LagrangeAtOffset<order,Real>(rules[i].M, rules[i].dM, rules[i].MT, rules[i].dMT, delta, i);
        }
        return rules;
      };
      return table.Get(digits, build)[ti];
    }

    /** Returns the rule toward node tj along v for each (order, digits): QuadOrder(digits)-point Gauss-Legendre on min(12, max(1, digits-5)) panels halving toward the node on each side, then one order-16 Alpert panel per side, log-corrected at the node, with order x N interpolation matrices. */
    template <Integer order, class Real> const QuadRule1D<Real>& CenteredVRule(const Integer tj, const Integer digits) {
      static LazyTable<Vector<QuadRule1D<Real>>, MaxDigits<Real>> table;
      const auto build = [digits]() {
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Integer Lvl = std::min<Integer>(12, std::max<Integer>(1, digits - 5));
        const Integer quad_order = QuadOrder<Real>(digits);
        Vector<QuadRule1D<Real>> rules(order);
        for (Integer j = 0; j < order; j++) {
          Vector<Real> delta;
          BuildCenteredLogSingular1D(delta, rules[j].w, nds[j], Lvl, quad_order);
          LagrangeAtOffset<order,Real>(rules[j].M, rules[j].dM, rules[j].MT, rules[j].dMT, delta, j);
        }
        return rules;
      };
      return table.Get(digits, build)[tj];
    }

    template <Integer order, class Real, class Kernel> void SelfInteracBlockTensorProduct(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();

      const QuadRule1D<Real>& ru = CenteredURule<order,Real>(ti, digits);
      const QuadRule1D<Real>& rv = CenteredVRule<order,Real>(tj, digits);
      IntegratePanel<order,Real>(M_acc, qel, elem_idx, Xtrg, normal_trg, ru, rv, ker);
    }

    template <Integer order, class Real, class Kernel> void SelfInteracTensorProduct(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      CenteredURule<order,Real>(0, digits); // precomp cache
      CenteredVRule<order,Real>(0, digits); // precomp cache
      GLRule<Real>(digits); // precomp cache
      const auto self_interac_one_trg = [&qel, &ker, digits](Matrix<Real>& M, const Long elem_idx, const Long t, const Integer ti, const Integer tj,
          const Vector<Real>& Xtrg, const Vector<Real>& ntrg, const Vector<Real>&, const Vector<Real>&,
          const Vector<Real>&, const Vector<Real>&, const Integer KDIM1_out) {
        ScratchBuf<Real> M_acc_buf((Long)order*order*Kernel::SrcDim()*KDIM1_out);
        Matrix<Real> M_acc((Long)order*order, Kernel::SrcDim()*KDIM1_out, M_acc_buf.begin(), false);
        SelfInteracBlockTensorProduct<order,Real>(M_acc, qel, elem_idx, ti, tj, Xtrg, ntrg, ker, digits);
        ScatterTargetBlock(M, M_acc, t, KDIM1_out);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, false, self_interac_one_trg);
    }

  }

  namespace detail_dispatch {

    using detail_quadelem::Access;

    using detail_dyadic_near::NearInteracDyadic;
    using detail_tensorprod_near::NearInteracTensorProduct;
    using detail_duffy::SelfInteracDuffy;
    using detail_hedgehog::SelfInteracHedgehog;
    using detail_tensorprod_singular::SelfInteracTensorProduct;

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

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const Real tol, const bool trg_dot_prod, const ElementListBase<Real>* self) {
    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
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
      const Long nnode_per_elem = (Long)order * order;
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * detail_quadelem::COORD_DIM * nnode_per_elem;
        for (Long p = 0; p < nnode_per_elem; p++) {
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
    Vector<Long> order_markers;
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

    StaticArray<Long,2> file_order{(order_markers.Dim() ? order_markers[0] : 0), 0};
    comm.Allreduce(file_order + 0, file_order + 1, 1, CommOp::MAX);
    SCTL_ASSERT(file_order[1] > 0);
    { // Check that every element starts with that order
      const Long nnode_per_elem = file_order[1] * file_order[1];
      SCTL_ASSERT(order_markers.Dim() % nnode_per_elem == 0);
      const Long Nelem_local = order_markers.Dim() / nnode_per_elem;
      for (Long elem = 0; elem < Nelem_local; elem++) {
        const Long offset = elem * nnode_per_elem;
        SCTL_ASSERT(order_markers[offset] == file_order[1]);
        for (Long j = 1; j < nnode_per_elem; j++) {
          SCTL_ASSERT(order_markers[offset + j] == file_order[1] || order_markers[offset + j] == -1);
        }
      }
    }
    Init<ValueType>((Integer)file_order[1], coord_);
  }

  template <class Real> void QuadElemList<Real>::GetVTUData(VTUData& vtu_data, const Vector<Real>& F, const Long elem_idx) const {
    if (elem_idx == -1) { // Every element, each with its part of F
      const Long nnode_per_elem = (Long)order * order;
      Long dof = 0;
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

    const Long Ng = order + 2;
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
      const Long nnode_per_elem = (Long)order * order;
      const Long dof = F.Dim() / nnode_per_elem;
      SCTL_ASSERT(F.Dim() == nnode_per_elem * dof);

      ScratchBuf<Real> F_soa_buf(dof * nnode_per_elem);
      Vector<Real> F_soa(F_soa_buf);
      for (Long p = 0; p < nnode_per_elem; p++) {
        for (Long k = 0; k < dof; k++) {
          F_soa[k * nnode_per_elem + p] = F[p * dof + k];
        }
      }

      ScratchBuf<Real> M_buf(order * Ng), MT_buf(Ng * order);
      Vector<Real> M_v(order * Ng, M_buf.begin(), false);
      LagrangeInterp<Real>::Interpolate(M_v, ParamNodes(order), grid);
      for (Integer i = 0; i < order; i++) {
        for (Long a = 0; a < Ng; a++) MT_buf[a * order + i] = M_buf[i * Ng + a];
      }
      const Matrix<Real> M(order, Ng, M_buf.begin(), false);
      const Matrix<Real> MT(Ng, order, MT_buf.begin(), false);

      ScratchBuf<Real> F_grid_buf(dof * Ng * Ng);
      Vector<Real> F_grid(F_grid_buf);
      detail_quadelem::EvalTensorProduct(F_grid, F_soa, MT, M);
      for (Long p = 0; p < Ng * Ng; p++) {
        for (Long k = 0; k < dof; k++) vtu_data.value.PushBack((VTUData::VTKReal)F_grid[k * (Ng * Ng) + p]);
      }
    }

    const Long point_offset = vtu_data.coord.Dim() / detail_quadelem::COORD_DIM;
    for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
    for (Long i = 0; i < Ng - 1; i++) { // Quadrilateral cells of the grid
      for (Long j = 0; j < Ng - 1; j++) {
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
    elem_lst.nelem = nelem;
    elem_lst.order = order;
    elem_lst.scheme_ = static_cast<typename QuadElemList<ValueType>::QuadScheme>(static_cast<int>(scheme_));

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
