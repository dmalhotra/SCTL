#ifndef _SCTL_QUAD_ELEMENT_CPP_
#define _SCTL_QUAD_ELEMENT_CPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
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

    template <class Real> struct Access {
      static const Vector<Real>& Coord(const QuadElemList<Real>& qel) { return qel.coord; }
      static typename QuadElemList<Real>::QuadScheme Scheme(const QuadElemList<Real>& qel) { return qel.scheme_; }
      static const Vector<Real>& XnNode(const QuadElemList<Real>& qel) { return qel.Xn_node; }
      static const Vector<Real>& DCoordDu(const QuadElemList<Real>& qel) { return qel.dcoord_du; }
      static const Vector<Real>& DCoordDv(const QuadElemList<Real>& qel) { return qel.dcoord_dv; }
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
      const Integer ncomp = in.Dim() / (R * S);
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

    template <class Real> void LagrangeDiffMat(Matrix<Real>& D, const Vector<Real>& nds) {
      const Integer n = nds.Dim();
      Vector<Real> f(n * n);
      f.SetZero();
      for (Integer i = 0; i < n; i++) f[i * n + i] = 1;
      D.ReInit(n, n);
      Vector<Real> df(n * n, D.begin(), false);
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
    template <class Real, void (*QuadParams)(Real, Real&, Integer&)> const QuadParamSet<Real>& CachedQuadParams(const Integer digits) {
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

    template <class Real> void EvalPoint(const Vector<Real>& coord, const Integer order, Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Vector<Real>* origin) {
      const Integer nnode = order * order;
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
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            const Integer p = i*order + j;
            const Real w = Lu[i]*Lv[j];
            for (Integer k = 0; k < COORD_DIM; k++) x[k] += coord[k*nnode + p]*w;
            if (want_d) {
              const Real w_du = dLu[i]*Lv[j];
              const Real w_dv = Lu[i]*dLv[j];
              for (Integer k = 0; k < COORD_DIM; k++) {
                xu[k] += coord[k*nnode + p]*w_du;
                xv[k] += coord[k*nnode + p]*w_dv;
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

    template <class Real> void ShiftedElemCoord(Vector<Real>& out, const Vector<Real>& coord, const Vector<Real>& Xtrg) {
      const Integer nnode = coord.Dim() / COORD_DIM;
      if (out.Dim() != COORD_DIM*nnode) out.ReInit(COORD_DIM*nnode);
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Integer p = 0; p < nnode; p++) out[k*nnode + p] = coord[k*nnode + p] - ok;
      }
    }

    template <class Real> Real GetClosestNode(const Vector<Real>& coord, const Integer order, Real& ustar, Real& vstar, const Vector<Real>& Xtrg) {
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

    template <class Real> Real GetClosestPoint(const Vector<Real>& coord, const Integer order, Real& ustar, Real& vstar, const Vector<Real>& Xtrg) {
      const auto dist2_at = [&coord, order, &Xtrg](const Real uu, const Real vv) -> Real {
        Real X[COORD_DIM];
        EvalPoint<Real>(coord, order, X, nullptr, nullptr, uu, vv, &Xtrg);
        Real r2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) r2 += X[k]*X[k];
        return r2;
      };

      Real u, v, f;
      { // Start from the nearest node
        const Real f_seed = GetClosestNode(coord, order, u, v, Xtrg);
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
          EvalPoint<Real>(coord, order, X, dXu, dXv, u, v, &Xtrg);
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
    static void WeightedKernelVec(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Integer nq, const Integer run, const Integer j0, const Integer j1, const Real wj, const bool accum, ConstIterator<Real> ntrg, const void* ctx) {
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
              const Integer id = blk*C*run + (a*KD1o+b)*run + j;
              if (accum) (VecType::Load(&out[id]) + val*vw).Store(&out[id]);
              else       (val*vw).Store(&out[id]);
            }
          }
        }
      }
    }

    template <class Real, class Kernel> void WeightedKernel(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Integer nq, const Integer run, const Real wj, const bool accum, const Vector<Real>& normal_trg, const Kernel& ker) {
      static constexpr bool HAS_N = UKerNeedsN<Kernel, Vec<Real,1>>::value;
      using WVec = Vec<Real, DefaultVecLen<Real>()>;
      const Integer jmain = (run/WVec::Size())*WVec::Size();
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
      const Integer nnode = order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      const Integer C = KDIM0 * KDIM1_out;
      const Integer Nu = ru.M.Dim(1);
      const Integer Nv = rv.M.Dim(1);
      SCTL_ASSERT(Nu > 0 && Nv > 0);

      ScratchBuf<Real> Cv(COORD_DIM*order*Nv), Cdv(COORD_DIM*order*Nv);
      { // Interpolate the coordinates and their v-derivative along v
        const Matrix<Real> cs_all(COORD_DIM*order, order, (Iterator<Real>)src_nodal.begin(), false);
        Matrix<Real> Cv_all (COORD_DIM*order, Nv, Cv.begin(),  false);
        Matrix<Real> Cdv_all(COORD_DIM*order, Nv, Cdv.begin(), false);
        Matrix<Real>::GEMM(Cv_all,  cs_all, rv.M);
        Matrix<Real>::GEMM(Cdv_all, cs_all, rv.dM);
      }

      ScratchBuf<Real> Tall(Nu*C*order);
      const Integer UBLK = std::max<Integer>(1, std::min<Integer>(Nu, MaxUnblockedPts / Nv));
      for (Integer a0 = 0; a0 < Nu; a0 += UBLK) { // Blocks of u-rows
        const Integer nu = std::min<Integer>(UBLK, Nu - a0);
        const Integer nqb = nu*Nv;

        ScratchBuf<Real> Xs(COORD_DIM*nqb), Xn(COORD_DIM*nqb), wq(nqb);
        { // Points, normals and weights of the block
          ScratchBuf<Real> dXu(COORD_DIM*nqb), dXv(COORD_DIM*nqb);
          { // Interpolate coordinates and tangents along u
            const Matrix<Real> MuT_b (nu, order, (Iterator<Real>)ru.MT.begin()  + a0*order, false);
            const Matrix<Real> dMuT_b(nu, order, (Iterator<Real>)ru.dMT.begin() + a0*order, false);
            for (Integer k = 0; k < COORD_DIM; k++) {
              const Matrix<Real> Cv_k (order, Nv, Cv.begin()  + k*order*Nv, false);
              const Matrix<Real> Cdv_k(order, Nv, Cdv.begin() + k*order*Nv, false);
              Matrix<Real> X_k(nu, Nv, Xs.begin() + k*nqb, false);
              Matrix<Real> dXu_k(nu, Nv, dXu.begin() + k*nqb, false);
              Matrix<Real> dXv_k(nu, Nv, dXv.begin() + k*nqb, false);
              Matrix<Real>::GEMM(X_k,   MuT_b,  Cv_k);
              Matrix<Real>::GEMM(dXu_k, dMuT_b, Cv_k);
              Matrix<Real>::GEMM(dXv_k, MuT_b,  Cdv_k);
            }
          }
          for (Integer a = 0; a < nu; a++) {
            for (Integer b = 0; b < Nv; b++) {
              const Integer q = a*Nv + b;
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
        const Integer np = std::max<Integer>(1, proxy_w.Dim());
        for (Integer j = 0; j < np; j++) { // Weighted kernel, summed over the proxy points
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
          ScratchBuf<Real> Tblk(C*nu*order);
          const Matrix<Real> KW_m(C*nu, Nv, KW.begin(), false);
          Matrix<Real> T_m(C*nu, order, Tblk.begin(), false);
          Matrix<Real>::GEMM(T_m, KW_m, rv.MT);
          for (Integer a = 0; a < nu; a++) {
            for (Integer c = 0; c < C; c++) {
              for (Integer j = 0; j < order; j++) Tall[((a0 + a)*C + c)*order + j] = Tblk[(c*nu + a)*order + j];
            }
          }
        }
      }

      { // Project onto the u-nodes and add into acc_cm
        ScratchBuf<Real> Aall(order*C*order);
        const Matrix<Real> T_m(Nu, C*order, Tall.begin(), false);
        Matrix<Real> A_m(order, C*order, Aall.begin(), false);
        Matrix<Real>::GEMM(A_m, ru.M, T_m);
        for (Integer c = 0; c < C; c++) {
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) acc_cm[c*nnode + i*order + j] += Aall[(i*C + c)*order + j];
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void IntegratePanel(Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const QuadRule1D<Real>& ru, const QuadRule1D<Real>& rv, const Kernel& ker) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(coord.Dim() == COORD_DIM*order*order);
      const Integer nnode = order * order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full / COORD_DIM : KDIM1full;
      const Integer C = KDIM0 * KDIM1_out;

      ScratchBuf<Real> coord_shift_buf(COORD_DIM*nnode), acc_buf(C*nnode);
      Vector<Real> coord_shift(coord_shift_buf), acc(acc_buf);
      ShiftedElemCoord(coord_shift, coord, Xtrg);
      acc.SetZero();
      IntegrateTensorRule<order,Real>(acc, coord_shift, ru, rv, normal_trg, ker);
      for (Integer p = 0; p < nnode; p++) for (Integer c = 0; c < C; c++) M_acc[p][c] = acc[c*nnode + p];
    }

    template <class Real> void ScatterTargetBlock(Matrix<Real>& M, const Matrix<Real>& src, const Long t, const Integer KDIM1_out) {
      const Integer nrow = M.Dim(0);
      SCTL_ASSERT(src.Dim(0)*src.Dim(1) == nrow*KDIM1_out);
      const ConstIterator<Real> src_ = src.begin();
      for (Integer r = 0; r < nrow; r++) {
        for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          M[r][t*KDIM1_out+k1] = src_[r*KDIM1_out+k1];
        }
      }
    }

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

      const Vector<Real> coord(COORD_DIM*nnode, (Iterator<Real>)Access<Real>::Coord(qel).begin() + elem_idx*COORD_DIM*nnode, false);
      ScratchBuf<Real> M_acc_buf(nnode*KDIM0*KDIM1_out);
      Matrix<Real> M_acc(nnode, KDIM0*KDIM1_out, M_acc_buf.begin(), false);
      for (Long t = 0; t < Ntrg; t++) {
        const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)normal_trg.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        near_interac_one_trg(M_acc, coord, Xtrg, ntrg);
        ScatterTargetBlock(M, M_acc, t, KDIM1_out);
      }
    }

    template <Integer order, class Real, class Kernel, class SelfInteracOneTrg>
    void SelfInteracElems(Vector<Matrix<Real>>& M_lst, const bool trg_dot_prod, const QuadElemList<Real>& qel, SelfInteracOneTrg self_interac_one_trg) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      SCTL_ASSERT((Long)M_lst.Dim() == qel.Size());
      const Integer nnode = order * order;
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

      #pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < qel.Size(); elem_idx++) {
        // Coordinates and tangents component by component, normals point by point
        const Long offset = elem_idx*nnode*COORD_DIM;
        const Vector<Real> coord(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::Coord(qel).begin() + offset, false);
        const Vector<Real> dXu(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::DCoordDu(qel).begin() + offset, false);
        const Vector<Real> dXv(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::DCoordDv(qel).begin() + offset, false);
        const Vector<Real> Xnnodes(nnode*COORD_DIM, (Iterator<Real>)Access<Real>::XnNode(qel).begin() + offset, false);

        Matrix<Real>& M = M_lst[elem_idx];
        if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nnode*KDIM0, nnode*KDIM1_out);
        ScratchBuf<Real> M_acc_buf(nnode*KDIM0*KDIM1_out);
        Matrix<Real> M_acc(nnode, KDIM0*KDIM1_out, M_acc_buf.begin(), false);
        for (Integer ti = 0; ti < order; ti++) {
          for (Integer tj = 0; tj < order; tj++) {
            self_interac_one_trg(M_acc, coord, Xnnodes, dXu, dXv, ti, tj);
            ScatterTargetBlock(M, M_acc, ti*order + tj, KDIM1_out);
          }
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

    using detail_quadelem::CachedQuadParams;
    using detail_quadelem::EvalPoint;
    using detail_quadelem::GetClosestPoint;
    using detail_quadelem::IntegrateTensorRule;
    using detail_quadelem::LagrangeDiffMat;
    using detail_quadelem::NearInteracTargets;
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

    template <class Real> struct GradeRule : QuadRule1D<Real> {
      Real a, b;
    };

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

    /** Returns 2*MaxRefineLvl rules for each (order, q): q-point Gauss-Legendre on the dyadic intervals [1-2^-k, 1-2^-(k+1)] and tails [1-2^-k, 1], each with order x q interpolation matrices. */
    template <Integer order, class Real> const Vector<GradeRule<Real>>& NearGradeTable(const Integer q) {
      const auto build = [](const Integer q, const Matrix<PrecompReal>& Dsub) {
        using W = PrecompReal;
        const Vector<W>& sig = NearSubOffsets<W>(order);
        Vector<W> qn, qw;
        LegQuadRule<W>::template ComputeNdsWts<W>(&qn, &qw, q);
        Vector<GradeRule<Real>> tab(2*MaxRefineLvl<Real>);
        Vector<W> tq(q), Twts(order*q);
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
        Matrix<PrecompReal> Dsub;
        { // Derivatives of the Lagrange basis on the Chebyshev extreme points
          using W = PrecompReal;
          Vector<W> sub_nds(order);
          for (Integer i = 0; i < order; i++) {
            const W sh = sin<W>(const_pi<W>()*i/(2*(order-1)));
            sub_nds[i] = sh*sh;
          }
          sub_nds[0] = 0;
          sub_nds[order-1] = 1;
          LagrangeDiffMat(Dsub, sub_nds);
        }
        std::vector<Vector<GradeRule<Real>>> t(NearMaxQuadOrder+1);
        for (Integer q = 4; q <= NearMaxQuadOrder; q += 4) t[q] = build(q, Dsub);
        for (Integer d = 0; d < MaxDigits<Real>; d++) {
          const Integer qi = CachedQuadParams<Real, QuadParams<Real>>(d).quad_order;
          if (qi > 0 && qi <= NearMaxQuadOrder && t[qi].Dim() == 0) t[qi] = build(qi, Dsub);
        }
        return t;
      }();
      SCTL_ASSERT(q > 0 && q <= NearMaxQuadOrder && all[q].Dim());
      return all[q];
    }

    template <Integer order, class Real, class Kernel> void NearInteracBlockDyadic(Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Integer nnode = order*order;
      const Integer KDIM1_out = (normal_trg.Dim() > 0) ? KDIM1full/COORD_DIM : KDIM1full;
      const Integer C = KDIM0*KDIM1_out;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
      M_acc.SetZero();

      Real ustar, vstar;
      const Real dist = GetClosestPoint(coord, order, ustar, vstar, Xtrg);
      const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};

      Real spd_u, spd_v;
      Integer q_near;
      { // Speeds, and the quadrature order raised for skewed tangents
        Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
        EvalPoint<Real>(coord, order, Xc, dXu, dXv, ustar, vstar, nullptr);
        Real guu = 0, gvv = 0, guv = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          guu += dXu[k]*dXu[k];
          gvv += dXv[k]*dXv[k];
          guv += dXu[k]*dXv[k];
        }
        spd_u = sqrt<Real>(guu);
        spd_v = sqrt<Real>(gvv);

        const Integer q_iso = CachedQuadParams<Real, QuadParams<Real>>(digits).quad_order;
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
        ShiftedElemCoord(cs, coord, Xtrg);
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

      ScratchBuf<Real> acc_buf(C*nnode);
      Vector<Real> acc(acc_buf);
      const Vector<GradeRule<Real>>& tab = NearGradeTable<order,Real>(q_near);
      const auto integrate_piece = [&tab, &normal_trg, &ker, &proxy_off, &proxy_w, &acc, &Xsub_buf](const Integer sdu, const Integer sdv, const Integer iu, const Integer iv) {
        const GradeRule<Real>& gu = tab[iu];
        const GradeRule<Real>& gv = tab[iv];
        if (!(gu.b > gu.a) || !(gv.b > gv.a)) return;
        const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
        const Integer nsub = COORD_DIM*order*order;
        const Vector<Real> Xsub(nsub, Xsub_buf.begin() + (2*sdu+sdv)*nsub, false);
        IntegrateTensorRule<order,Real>(acc, Xsub, gu, gv, normal_trg, ker, nsign, proxy_off, proxy_w);
      };
      const Real b_ellipse = CachedQuadParams<Real, QuadParams<Real>>(digits).b_ellipse;
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
            ScratchBuf<Real> accB(C*nnode), accE(nnode);
            const Matrix<Real> St_v(order, order, St_buf.begin() + (2+sdv)*nnode, false);
            const Matrix<Real> Sf_u(order, order, Sf_buf.begin() + sdu*nnode, false);
            const Matrix<Real> A_all(C*order, order, acc.begin(), false);
            Matrix<Real> B_all(C*order, order, accB.begin(), false);
            Matrix<Real>::GEMM(B_all, A_all, St_v);
            for (Integer c = 0; c < C; c++) {
              const Matrix<Real> B_c(order, order, accB.begin() + c*nnode, false);
              Matrix<Real> E_c(order, order, accE.begin(), false);
              Matrix<Real>::GEMM(E_c, Sf_u, B_c);
              for (Integer p = 0; p < nnode; p++) M_acc[p][c] += accE[p];
            }
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracDyadic(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits) {
      const auto near_interac_one_trg = [&ker, digits](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
        NearInteracBlockDyadic<order,Real>(M_acc, coord, Xtrg, ntrg, ker, digits);
      };
      NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel, elem_idx, near_interac_one_trg);
    }

  }

  namespace detail_tensorprod_near {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::MaxRefineLvl;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::CachedQuadParams;
    using detail_quadelem::DiffMat;
    using detail_quadelem::EvalPoint;
    using detail_quadelem::GetClosestPoint;
    using detail_quadelem::IntegratePanel;
    using detail_quadelem::NearInteracTargets;

    template <class Real> void QuadParams(const Real tol, Real& b_ellipse, Integer& quad_order) {
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real rho = (Real)2.5;
      b_ellipse = (rho + 1/rho) / 4;
      quad_order = std::max<Integer>(1, (Integer)ceil<Real>(-log<Real>(((15*(rho*rho-1))/64)*tol_)/log<Real>(rho)*(Real)0.5 + 1));
    }

    /** Returns quad_order-point Gauss-Legendre nodes and weights on [0, 1] for each digits. */
    template <class Real> const std::pair<Vector<Real>, Vector<Real>>& GLRule(const Integer digits) {
      static const std::array<std::pair<Vector<Real>, Vector<Real>>,MaxDigits<Real>> gl = []() {
        std::array<std::pair<Vector<Real>, Vector<Real>>,MaxDigits<Real>> t;
        for (Integer d = 0; d < MaxDigits<Real>; d++) {
          const Integer quad_order = CachedQuadParams<Real, QuadParams<Real>>(d).quad_order;
          LegQuadRule<Real>::ComputeNdsWts(&t[d].first, &t[d].second, quad_order);
        }
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real>);
      return gl[digits];
    }

    template <Integer order, class Real, class Kernel> void NearInteracTensorProduct(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const QuadElemList<Real>& qel, const Integer digits) {
      const auto near_interac_one_trg = [&ker, digits](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
        constexpr Integer MaxSegments = 4096;
        ScratchBuf<Real> useg(2*MaxSegments), vseg(2*MaxSegments);
        Integer nseg_u, nseg_v;
        { // Segments graded toward the closest point
          const Real b_ellipse = CachedQuadParams<Real, QuadParams<Real>>(digits).b_ellipse;
          Real ustar, vstar, h_param;
          { // Closest point, and its distance in parameter units
            const Real dist = GetClosestPoint(coord, order, ustar, vstar, Xtrg);
            Real Xc[COORD_DIM], dXdu[COORD_DIM], dXdv[COORD_DIM];
            EvalPoint<Real>(coord, order, Xc, dXdu, dXdv, ustar, vstar, nullptr);
            Real su2 = 0, sv2 = 0;
            for (Integer k = 0; k < COORD_DIM; k++) {
              su2 += dXdu[k]*dXdu[k];
              sv2 += dXdv[k]*dXdv[k];
            }
            const Real L_phys = std::max<Real>(sqrt<Real>(su2), sqrt<Real>(sv2));
            const bool degenerate = !(dist > 0) || isinf<Real>(dist) || isnan<Real>(dist) || !(L_phys > 0);
            h_param = (degenerate ? 0 : dist/L_phys);
          }

          const auto graded_segments = [b_ellipse](Iterator<Real> seg, const Real center, const Real w_min) {
            const Real r = std::min<Real>((Real)0.9, b_ellipse/(1 + b_ellipse) * (Real)1.05);
            const Real w_stop = std::max<Real>(w_min, (Real)1e-300);

            Integer nseg = 0;
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
          };
          const Real w_floor = pow<Real>((Real)0.5, MaxRefineLvl<Real>);
          const Real w_min = std::max<Real>(h_param/b_ellipse, w_floor);
          nseg_u = graded_segments(useg.begin(), ustar, w_min);
          nseg_v = graded_segments(vseg.begin(), vstar, w_min);
        }
        const std::pair<Vector<Real>, Vector<Real>>& gl = GLRule<Real>(digits);
        const Integer Nu = nseg_u * gl.first.Dim();
        const Integer Nv = nseg_v * gl.first.Dim();
        SCTL_ASSERT(Nu > 0 && Nv > 0);

        ScratchBuf<Real> rule_u(Nu*(1 + 4*order)), rule_v(Nv*(1 + 4*order));
        const auto rule_view = [](Iterator<Real> buf, const Integer N) {
          return QuadRule1D<Real>{Vector<Real>(N, buf, false),
              Matrix<Real>(order, N, buf + N, false), Matrix<Real>(order, N, buf + N*(1 + order), false),
              Matrix<Real>(N, order, buf + N*(1 + 2*order), false), Matrix<Real>(N, order, buf + N*(1 + 3*order), false)};
        };
        QuadRule1D<Real> ru = rule_view(rule_u.begin(), Nu);
        QuadRule1D<Real> rv = rule_view(rule_v.begin(), Nv);
        { // Gauss-Legendre rule on each segment, and its interpolation matrices
          const auto build_rule = [&gl](QuadRule1D<Real>& r, Vector<Real>& param, Iterator<Real> seg, const Integer nseg) {
            { // Nodes and weights of every segment
              const Integer quad_order = gl.first.Dim();
              Integer idx = 0;
              for (Integer si = 0; si < nseg; si++) {
                const Real a0 = seg[si*2+0], a1 = seg[si*2+1];
                const Real len = a1 - a0;
                for (Integer a = 0; a < quad_order; a++) {
                  param[idx] = a0 + len*gl.first[a];
                  r.w[idx] = gl.second[a]*len;
                  idx++;
                }
              }
            }
            const Integer N = param.Dim();
            Vector<Real> M_v(order*N, r.M.begin(), false);
            LagrangeInterp<Real>::Interpolate(M_v, QuadElemList<Real>::ParamNodes(order), param);
            Matrix<Real>::GEMM(r.dM, DiffMat<Real>(order), r.M);
            for (Integer i = 0; i < order; i++) {
              for (Integer a = 0; a < N; a++) {
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
        IntegratePanel<order,Real>(M_acc, coord, Xtrg, ntrg, ru, rv, ker);
      };
      NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel, elem_idx, near_interac_one_trg);
    }

  }

  namespace detail_duffy {

    using detail_quadelem::COORD_DIM;

    using detail_quadelem::CachedQuadParams;
    using detail_quadelem::DiffMat;
    using detail_quadelem::SelfInteracElems;
    using detail_quadelem::ShiftedElemCoord;
    using detail_quadelem::WeightedKernel;
    using detail_dyadic_near::NearGradeTable;
    using detail_dyadic_near::QuadParams;

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
              Vector<Real> Mbeta_v(order*qs, Mbeta.begin(), false);
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
              Vector<Real> Malpha_v(order*order, Malpha.begin(), false);
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
      const Integer nt = std::max<Integer>(order/2, (KDIM0 > 1 ? 4*digits : (5*digits + 1)/2));
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
          static constexpr Integer MaxGLOrder = 128;
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
          Vector<Real> Tt_v(order*nt, Tt_buf.begin(), false);
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
            Matrix<Real>::GEMM(Gm, FS, T.beta_interp);
            for (Integer i = 0; i < ns; i++) {
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
          WeightedKernel<Real>(KW_buf.begin(), Xt0_v.begin(), Xs.begin(), Xn.begin(), wq.begin(), ns*nt, nt, (Real)1, false, normal_trg, ker);
        }

        { // Project onto the element nodes and add into M_acc
          ScratchBuf<Real> Zall_buf(ns*C*order), Yi_buf(C*order), Yall_buf(C*order*ns), Pc_buf(nnode);
          const Matrix<Real> KW(ns*C, nt, KW_buf.begin(), false);
          Matrix<Real> Zall(ns*C, order, Zall_buf.begin(), false);
          Matrix<Real> Yi(C, order, Yi_buf.begin(), false);
          Matrix<Real> Yall(C*order, ns, Yall_buf.begin(), false);
          Matrix<Real> Pc(order, order, Pc_buf.begin(), false);
          Matrix<Real>::GEMM(Zall, KW, TtT);
          for (Integer i = 0; i < ns; i++) {
            const Matrix<Real> Zi(C, order, (Iterator<Real>)Zall.begin() + i*C*order, false);
            Matrix<Real>::GEMM(Yi, Zi, T.alpha_interp_T[i]);
            for (Integer c = 0; c < C; c++) for (Integer m = 0; m < order; m++) Yall[c*order+m][i] = Yi[c][m];
          }
          for (Integer c = 0; c < C; c++) {
            const Matrix<Real> Yc(order, ns, (Iterator<Real>)Yall.begin() + c*order*ns, false);
            Matrix<Real>::GEMM(Pc, Yc, T.beta_interp_T);
            for (Integer m = 0; m < order; m++) for (Integer n = 0; n < order; n++)
              M_acc[T.swap_ab ? n*order+m : m*order+n][c] += Pc[m][n];
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void SelfInteracDuffy(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      DuffyTable<order,Real>(); // precomp cache
      NearGradeTable<order,Real>(CachedQuadParams<Real, QuadParams<Real>>(digits).quad_order); // precomp cache
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

    using detail_quadelem::CachedQuadParams;
    using detail_quadelem::SelfInteracElems;
    using detail_dyadic_near::NearGradeTable;
    using detail_dyadic_near::NearInteracBlockDyadic;
    using detail_dyadic_near::QuadParams;

    template <class Kernel, class = void> struct KernelSingularOrder {
      static constexpr Integer value = 2;
    };
    template <class Kernel> struct KernelSingularOrder<Kernel, std::void_t<decltype(Kernel::SingularOrder())>> {
      static constexpr Integer value = Kernel::SingularOrder();
    };

    template <Integer order, class Real, class Kernel> void SelfInteracHedgehog(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      static constexpr Integer sing_order = KernelSingularOrder<Kernel>::value;
      const Integer near_digits = std::min<Integer>(MaxDigits<Real>-1, digits + (sing_order <= 1 ? 2 : 6));
      NearGradeTable<order,Real>(CachedQuadParams<Real, QuadParams<Real>>(near_digits).quad_order); // precomp cache
      NearGradeTable<order,Real>(CachedQuadParams<Real, QuadParams<Real>>(digits).quad_order); // precomp cache

      static const Vector<Real> proxy_dist = []() { // Proxy distances, in units of rmin
        Vector<Real> v;
        for (Integer j = 0; j < 5; j++) v.PushBack(pow<Real>((Real)4, (Real)j/(Real)4));
        return v;
      }();
      static const Vector<Real> hh_w = []() { // Weights extrapolating the proxy values to distance 0
        using W = PrecompReal;
        const Integer p = proxy_dist.Dim();
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
      const auto self_interac_one_trg = [&nds, rmin_coeff, &ker, near_digits, trg_dot_prod](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xnnodes, const Vector<Real>& dXu, const Vector<Real>& dXv, const Integer ti, const Integer tj) {
        const Integer nnode = order*order;
        const Integer t = ti*order + tj;
        ScratchBuf<Real> hh_Xt1_buf(COORD_DIM), hh_off_buf(proxy_dist.Dim()*COORD_DIM);
        Vector<Real> hh_Xt1(hh_Xt1_buf), hh_off(hh_off_buf);
        { // Proxy points along the normal, sized by edge distance
          Real su2 = 0, sv2 = 0;
          for (Integer k = 0; k < COORD_DIM; k++) {
            su2 += dXu[k*nnode+t]*dXu[k*nnode+t];
            sv2 += dXv[k*nnode+t]*dXv[k*nnode+t];
          }
          const Real edge_u = std::min<Real>(nds[ti], 1-nds[ti]), edge_v = std::min<Real>(nds[tj], 1-nds[tj]);
          const Real rmin = rmin_coeff * std::min<Real>(edge_u*sqrt<Real>(su2), edge_v*sqrt<Real>(sv2));
          for (Integer k = 0; k < COORD_DIM; k++) hh_Xt1[k] = coord[k*nnode + t] + rmin*Xnnodes[t*COORD_DIM+k];
          for (Integer j = 0; j < proxy_dist.Dim(); j++) {
            const Real rj = rmin*proxy_dist[j];
            for (Integer k = 0; k < COORD_DIM; k++) hh_off[j*COORD_DIM+k] = (rj-rmin)*Xnnodes[t*COORD_DIM+k];
          }
        }
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)Xnnodes.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        NearInteracBlockDyadic<order,Real>(M_acc, coord, hh_Xt1, ntrg, ker, near_digits, hh_off, hh_w);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, self_interac_one_trg);
    }

  }

  namespace detail_tensorprod_singular {

    using detail_quadelem::COORD_DIM;
    using detail_quadelem::MaxDigits;
    using detail_quadelem::MaxRefineLvl;
    using detail_quadelem::QuadRule1D;

    using detail_quadelem::CachedQuadParams;
    using detail_quadelem::DiffMat;
    using detail_quadelem::IntegratePanel;
    using detail_quadelem::SelfInteracElems;
    using detail_tensorprod_near::GLRule;
    using detail_tensorprod_near::QuadParams;

    template <Integer order, class Real> void LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& delta, const Integer ti) {
      const Integer N = delta.Dim();
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
      MT = M.Transpose();
      dMT = dM.Transpose();
    }

    template <class Real> void BuildCenteredLogSingular1D(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer quad_order) {
      const Integer ord = 16;
      delta.ReInit(0);
      w.ReInit(0);
      const auto add_alpert = [&delta, &w, ord](const Real a, const Real b, const bool log_a, const bool log_b) {
        const auto L = (log_a ? QuadLogExtraPtNodes<Real>(ord) : QuadSmoothExtraPtNodes<Real>(ord));
        const auto R = (log_b ? QuadLogExtraPtNodes<Real>(ord) : QuadSmoothExtraPtNodes<Real>(ord));
        const Integer skipL = L.NodesToSkip, skipR = R.NodesToSkip;
        const Integer N = std::max<Integer>(skipL + skipR + 2, 2 * ord);
        const Integer N1 = N - 1;
        const Real h = (b - a) / (Real)N1;
        for (Integer i = skipL; i <= N1 - skipR; i++) {
          delta.PushBack(a + (Real)i*h);
          w.PushBack(h);
        }
        for (Integer i = 0; i < L.ExtraNodes.Dim(); i++) {
          delta.PushBack(a + L.ExtraNodes[i]*h);
          w.PushBack(L.ExtraWeights[i]*h);
        }
        for (Integer i = 0; i < R.ExtraNodes.Dim(); i++) {
          delta.PushBack(b - R.ExtraNodes[i]*h);
          w.PushBack(R.ExtraWeights[i]*h);
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

    /** Returns the rule toward node ti along u for each (order, digits): quad_order-point Gauss-Legendre on intervals halving min(MaxRefineLvl, 2*digits+6) times toward the node on each side, with order x N interpolation matrices. */
    template <Integer order, class Real> const QuadRule1D<Real>& CenteredURule(const Integer ti, const Integer digits) {
      const auto build = [](const Integer digits) {
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Integer quad_order = CachedQuadParams<Real, QuadParams<Real>>(digits).quad_order;
        const Integer Lvl = std::min<Integer>(MaxRefineLvl<Real>, 2*digits + 6);
        Vector<Real> qnds, qwts;
        LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, quad_order);
        const auto graded_rule = [Lvl, &qnds, &qwts](Vector<Real>& delta, Vector<Real>& w, const Real u0) {
          const Integer q = qnds.Dim();
          const auto side = [&delta, &w, Lvl, q, &qnds, &qwts](const Real span, const Real sgn) {
            if (!(span > 0)) return;
            Real a = 0;
            for (Integer k = Lvl; k >= 0; k--) {
              const Real b = span * pow<Real>((Real)0.5, (Integer)k);
              const Real len = b - a;
              if (len > 0) {
                for (Integer i = 0; i < q; i++) {
                  delta.PushBack(sgn*(a + len*qnds[i]));
                  w.PushBack(len*qwts[i]);
                }
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
          LagrangeAtOffset<order,Real>(rules[i].M, rules[i].dM, rules[i].MT, rules[i].dMT, delta, i);
        }
        return rules;
      };
      static const std::array<Vector<QuadRule1D<Real>>, MaxDigits<Real>> table = [&build]() {
        std::array<Vector<QuadRule1D<Real>>, MaxDigits<Real>> t;
        for (Integer d = 0; d < MaxDigits<Real>; d++) t[d] = build(d);
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real>);
      return table[digits][ti];
    }

    /** Returns the rule toward node tj along v for each (order, digits): quad_order-point Gauss-Legendre on min(12, max(1, digits-5)) panels halving toward the node on each side, then one order-16 Alpert panel per side, log-corrected at the node, with order x N interpolation matrices. */
    template <Integer order, class Real> const QuadRule1D<Real>& CenteredVRule(const Integer tj, const Integer digits) {
      const auto build = [](const Integer digits) {
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Integer Lvl = std::min<Integer>(12, std::max<Integer>(1, digits - 5));
        const Integer quad_order = CachedQuadParams<Real, QuadParams<Real>>(digits).quad_order;
        Vector<QuadRule1D<Real>> rules(order);
        for (Integer j = 0; j < order; j++) {
          Vector<Real> delta;
          BuildCenteredLogSingular1D(delta, rules[j].w, nds[j], Lvl, quad_order);
          LagrangeAtOffset<order,Real>(rules[j].M, rules[j].dM, rules[j].MT, rules[j].dMT, delta, j);
        }
        return rules;
      };
      static const std::array<Vector<QuadRule1D<Real>>, MaxDigits<Real>> table = [&build]() {
        std::array<Vector<QuadRule1D<Real>>, MaxDigits<Real>> t;
        for (Integer d = 0; d < MaxDigits<Real>; d++) t[d] = build(d);
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < MaxDigits<Real>);
      return table[digits][tj];
    }

    template <Integer order, class Real, class Kernel> void SelfInteracTensorProduct(Vector<Matrix<Real>>& M_lst, const Kernel& ker, const bool trg_dot_prod, const QuadElemList<Real>& qel, const Integer digits) {
      CenteredURule<order,Real>(0, digits); // precomp cache
      CenteredVRule<order,Real>(0, digits); // precomp cache
      GLRule<Real>(digits); // precomp cache
      const auto self_interac_one_trg = [&ker, digits, trg_dot_prod](Matrix<Real>& M_acc, const Vector<Real>& coord, const Vector<Real>& Xnnodes, const Vector<Real>&, const Vector<Real>&, const Integer ti, const Integer tj) {
        const Integer t = ti*order + tj;
        StaticArray<Real,COORD_DIM> Xtrg_buf;
        for (Integer k = 0; k < COORD_DIM; k++) Xtrg_buf[k] = coord[k*order*order + t];
        const Vector<Real> Xtrg(COORD_DIM, Xtrg_buf, false);
        const Vector<Real> ntrg((trg_dot_prod ? COORD_DIM : 0), (trg_dot_prod ? (Iterator<Real>)Xnnodes.begin() + t*COORD_DIM : NullIterator<Real>()), false);
        const QuadRule1D<Real>& ru = CenteredURule<order,Real>(ti, digits);
        const QuadRule1D<Real>& rv = CenteredVRule<order,Real>(tj, digits);
        IntegratePanel<order,Real>(M_acc, coord, Xtrg, ntrg, ru, rv, ker);
      };
      SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, self_interac_one_trg);
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
      const Integer chunk = std::max<Long>(1, std::min<Long>(64, (nblk + nthreads - 1) / nthreads));
      const Long nchunk = (nblk + chunk - 1) / chunk;
      #pragma omp parallel for schedule(static)
      for (Long b = 0; b < nchunk; b++) {
        const Long offset = b * chunk * nnode_per_elem;
        const Integer n = (std::min(nblk, (b + 1) * chunk) - b * chunk) * nnode_per_elem;
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
    const Integer Nu = u_param.Dim();
    const Integer Nv = v_param.Dim();
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
      Vector<Real> Mu_(order * Nu, Mu_buf.begin(), false);
      Vector<Real> Mv_(order * Nv, Mv_buf.begin(), false);
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
      Vector<Real> M_v(order * Ng, M_buf.begin(), false);
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
