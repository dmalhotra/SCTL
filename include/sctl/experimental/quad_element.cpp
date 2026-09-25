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
    static constexpr Long MaxUnblockedPts = 16384;

    template <class Real> struct Access {
      static const Vector<Real>& Coord(const QuadElemList<Real>& qel) { return qel.coord; }
      static typename QuadElemList<Real>::QuadScheme Scheme(const QuadElemList<Real>& qel) { return qel.scheme_; }
    };

    template <class Real> static constexpr Integer MaxDigits = 1 + GetSigBits<Real>::value()*30103/100000;

    template <class Real> static constexpr Integer MaxRefineLvl = GetSigBits<Real>::value();

    template <class Real> void PartitionRange(Long Nelem_total, const Comm& comm, Long& i0, Long& i1) {
      const Long Np = comm.Size();
      const Long pid = comm.Rank();
      i0 = Nelem_total * (pid + 0) / Np;
      i1 = Nelem_total * (pid + 1) / Np;
    }

    template <class ValueType> void EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv) {
      const Integer Nu = MuT.Dim(0);
      const Integer R  = MuT.Dim(1);
      const Integer S  = Mv.Dim(0);
      const Integer Nv = Mv.Dim(1);
      const Long ncomp = in.Dim() / ((Long)R * S);
      SCTL_ASSERT(in.Dim() == ncomp * (Long)R * S);

      const Long Nout = (Long)Nu * Nv;
      if (out.Dim() != ncomp * Nout) out.ReInit(ncomp * Nout);

      constexpr Integer Nbuff = 1024;
      StaticArray<ValueType,Nbuff> tmp_buf;
      Matrix<ValueType> tmp(R, Nv, ((Long)R * Nv > Nbuff ? NullIterator<ValueType>() : tmp_buf), (Long)R * Nv > Nbuff);

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
      Vector<Real> line_in(order), line_out(order);
      for (Long k = 0; k < ncomp; k++) {
        const Long cb = k * nnode_per_elem;

        for (Integer j = 0; j < order; j++) {
          for (Integer i = 0; i < order; i++) line_in[i] = coord_slab[cb + i * order + j];
          LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
          for (Integer i = 0; i < order; i++) du_slab[cb + i * order + j] = line_out[i];
        }

        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) line_in[j] = coord_slab[cb + i * order + j];
          LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
          for (Integer j = 0; j < order; j++) dv_slab[cb + i * order + j] = line_out[j];
        }
      }
    }

    template <class Real> inline const Matrix<Real>& DiffMat(const Integer order) {
      constexpr Integer MAX_ORDER = 50;
      SCTL_ASSERT(0 < order && order <= MAX_ORDER);
      auto compute_all = []() {
        Vector<Matrix<Real>> D(MAX_ORDER + 1);
        for (Integer n = 2; n <= MAX_ORDER; n++) {
          const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(n);
          Vector<Real> f((Long)n * n);
          f.SetZero();
          for (Integer i = 0; i < n; i++) f[i * n + i] = 1;
          Vector<Real> df;
          LagrangeInterp<Real>::Derivative(df, f, nds);
          D[n].ReInit(n, n);
          for (Integer i = 0; i < n; i++)
            for (Integer a = 0; a < n; a++) D[n][i][a] = df[i * n + a];
        }
        return D;
      };
      static const Vector<Matrix<Real>> all = compute_all();
      return all[order];
    }

    template <class Real> inline Integer DigitsFromTol(const Real tol) {
      for (Integer d = MaxDigits<Real>-1; d > 0; d--) if (tol <= pow<Real,Long>((Real)0.1, (Long)d)) return d;
      return 0;
    }

    template <class Kernel, class = void> struct KernelSingularOrder {
      static constexpr Integer value = 2;
    };
    template <class Kernel> struct KernelSingularOrder<Kernel, std::void_t<decltype(Kernel::SingularOrder())>> {
      static constexpr Integer value = Kernel::SingularOrder();
    };

    #ifdef SCTL_QUAD_T
    using PrecompReal = QuadReal;
    #else
    using PrecompReal = long double;
    #endif

    template <class Real> void EvalPoint(const QuadElemList<Real>& qel, Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Long elem_idx, const Vector<Real>* origin) {
      constexpr Integer MaxOrder = 48;
      SCTL_ASSERT(qel.Order() <= MaxOrder);
      const Long nnode = (Long)qel.Order() * qel.Order();
      const Long base = elem_idx * nnode * COORD_DIM;

      StaticArray<Real,MaxOrder> Lu, Lv, dLu, dLv;
      {
        StaticArray<Real,1> up;
        up[0] = u;
        Vector<Real> p(1, up, false), o(qel.Order(), Lu, false);
        LagrangeInterp<Real>::Interpolate(o, QuadElemList<Real>::ParamNodes(qel.Order()), p);
      }
      {
        StaticArray<Real,1> vp;
        vp[0] = v;
        Vector<Real> p(1, vp, false), o(qel.Order(), Lv, false);
        LagrangeInterp<Real>::Interpolate(o, QuadElemList<Real>::ParamNodes(qel.Order()), p);
      }

      const bool want_d = (dXu || dXv);
      if (want_d) {
        const Matrix<Real>& D = detail_quadelem::DiffMat<Real>(qel.Order());
        for (Integer i = 0; i < qel.Order(); i++) {
          Real su = 0, sv = 0;
          for (Integer a = 0; a < qel.Order(); a++) {
            su += D[i][a]*Lu[a];
            sv += D[i][a]*Lv[a];
          }
          dLu[i] = su;
          dLv[i] = sv;
        }
      }

      Real x0 = 0, x1 = 0, x2 = 0, du0 = 0, du1 = 0, du2 = 0, dv0 = 0, dv1 = 0, dv2 = 0;
      for (Integer i = 0; i < qel.Order(); i++) {
        for (Integer j = 0; j < qel.Order(); j++) {
          const Long p = i*qel.Order() + j;
          const Real c0 = Access<Real>::Coord(qel)[base + 0*nnode + p], c1 = Access<Real>::Coord(qel)[base + 1*nnode + p], c2 = Access<Real>::Coord(qel)[base + 2*nnode + p];
          const Real wv = Lu[i]*Lv[j];
          x0 += c0*wv;
          x1 += c1*wv;
          x2 += c2*wv;
          if (want_d) {
            const Real wu_ = dLu[i]*Lv[j], wvv = Lu[i]*dLv[j];
            du0 += c0*wu_;
            du1 += c1*wu_;
            du2 += c2*wu_;
            dv0 += c0*wvv;
            dv1 += c1*wvv;
            dv2 += c2*wvv;
          }
        }
      }
      if (origin) {
        x0 -= (*origin)[0];
        x1 -= (*origin)[1];
        x2 -= (*origin)[2];
      }
      X[0] = x0;
      X[1] = x1;
      X[2] = x2;
      if (dXu) {
        dXu[0] = du0;
        dXu[1] = du1;
        dXu[2] = du2;
      }
      if (dXv) {
        dXv[0] = dv0;
        dXv[1] = dv1;
        dXv[2] = dv2;
      }
    }

    template <class Real> Real GetClosestNode(const QuadElemList<Real>& qel, Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) {
      const auto& nds = QuadElemList<Real>::ParamNodes(qel.Order());
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

      ustar = nds[seed/qel.Order()];
      vstar = nds[seed%qel.Order()];

      return sqrt<Real>(best);
    }

    template <class Real> Real GetClosestPoint(const QuadElemList<Real>& qel, Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter = nullptr, bool* used_fallback = nullptr) {

      auto dist2_at = [&](const Real uu, const Real vv) -> Real {
        Real X[COORD_DIM];
        EvalPoint<Real>(qel, X, nullptr, nullptr, uu, vv, elem_idx, &Xtrg);
        Real r2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) r2 += X[k]*X[k];
        return r2;
      };

      Real u, v;
      const Real f_seed = GetClosestNode(qel, u, v, elem_idx, Xtrg);
      Real f = f_seed * f_seed;

      constexpr Integer max_iter = 30;
      const Real utol = (Real)machine_eps<Real>() * 64;
      const Real gtol = sqrt<Real>(machine_eps<Real>()) * 16;
      const Real c_eps = machine_eps<Real>() * 8;
      const Real gtol_stall = sqrt<Real>(machine_eps<Real>()) * 256;
      bool converged = false;
      Integer iters = 0;
      for (Integer it = 0; it < max_iter; it++) {
        iters = it + 1;
        Real X[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
        EvalPoint<Real>(qel, X, dXu, dXv, u, v, elem_idx, &Xtrg);

        Real E = 0, F = 0, G = 0, gu = 0, gv = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          const Real r = X[k], a = dXu[k], b = dXv[k];
          E += a*a;
          F += a*b;
          G += b*b;
          gu += r*a;
          gv += r*b;
        }

        Real Pu = gu, Pv = gv;
        if      (u <= 0) Pu = std::min<Real>(gu, (Real)0);
        else if (u >= 1) Pu = std::max<Real>(gu, (Real)0);
        if      (v <= 0) Pv = std::min<Real>(gv, (Real)0);
        else if (v >= 1) Pv = std::max<Real>(gv, (Real)0);
        const bool opt_u = (fabs(Pu) <= gtol * sqrt<Real>(E*f));
        const bool opt_v = (fabs(Pv) <= gtol * sqrt<Real>(G*f));
        if (opt_u && opt_v) {
          converged = true;
          break;
        }

        const bool u_act = ((u <= 0 && gu >= 0) || (u >= 1 && gu <= 0));
        const bool v_act = ((v <= 0 && gv >= 0) || (v >= 1 && gv <= 0));
        Real du = 0, dv = 0;
        if (!u_act && !v_act) {
          const Real det = E*G - F*F;
          if (fabs(det) > (Real)1e-30 * (E*G + F*F + 1)) {
            du = ( G*gu - F*gv) / det;
            dv = (-F*gu + E*gv) / det;
          } else {
            du = gu / (E + (Real)1e-30);
            dv = gv / (G + (Real)1e-30);
          }
        } else if (u_act) {
          dv = gv / (G + (Real)1e-30);
        } else {
          du = gu / (E + (Real)1e-30);
        }

        Real lambda = 1;
        bool improved = false;
        Real un = u, vn = v, fn = f;
        for (Integer ls = 0; ls < 40; ls++) {
          un = std::min<Real>(1, std::max<Real>(0, u - lambda*du));
          vn = std::min<Real>(1, std::max<Real>(0, v - lambda*dv));
          fn = dist2_at(un, vn);
          if (fn <= f * (1 + c_eps)) {
            improved = true;
            break;
          }
          lambda *= (Real)0.5;
        }
        if (!improved) {
          const Real gu_s = Pu / (E + (Real)1e-30), gv_s = Pv / (G + (Real)1e-30);
          lambda = 1;
          for (Integer ls = 0; ls < 40; ls++) {
            un = std::min<Real>(1, std::max<Real>(0, u - lambda*gu_s));
            vn = std::min<Real>(1, std::max<Real>(0, v - lambda*gv_s));
            fn = dist2_at(un, vn);
            if (fn <= f * (1 + c_eps)) {
              improved = true;
              break;
            }
            lambda *= (Real)0.5;
          }
        }
        if (!improved) {
          const bool stat = (fabs(Pu) <= gtol_stall * sqrt<Real>(E*f)) && (fabs(Pv) <= gtol_stall * sqrt<Real>(G*f));
          const bool tiny = (fabs(du) < utol && fabs(dv) < utol);
          if (iters > 1 && (stat || tiny)) converged = true;
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

      if (!converged) {
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
      if (n_iter) *n_iter = iters;
      if (used_fallback) *used_fallback = !converged;
      return sqrt<Real>(f);
    }

    template <class K, class VT, class = void> struct UKerNeedsN : std::false_type {};
    template <class K, class VT> struct UKerNeedsN<K, VT, std::void_t<decltype(
        K::template uKerMatrix<0,VT>(std::declval<VT(&)[K::SrcDim()][K::TrgDim()]>(),
          std::declval<const VT(&)[3]>(),
          std::declval<const VT(&)[3]>(),
          (const void*)nullptr))>> : std::true_type {};

    template <class Real, class Kernel, class VecType, bool HAS_N, bool TRG_DOT>
    static void KerFoldSoA(Iterator<Real> out, ConstIterator<Real> Xt, ConstIterator<Real> Xs, ConstIterator<Real> Xn, ConstIterator<Real> wq, const Long nq, const Long run, const Long j0, const Long j1, const Real wj, const bool accum, ConstIterator<Real> ntrg) {
      static constexpr Integer CD = 3;
      static constexpr Integer KD0 = Kernel::SrcDim();
      static constexpr Integer KD1 = Kernel::TrgDim();
      static constexpr Integer KD1o = (TRG_DOT ? KD1/CD : KD1);
      static constexpr Integer C_ = KD0*KD1o;
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
            Kernel::template uKerMatrix<digits,VecType>(u, r, n, nullptr);
          } else {
            Kernel::template uKerMatrix<digits,VecType>(u, r, nullptr);
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
              const Long id = blk*C_*run + (Long)(a*KD1o+b)*run + j;
              if (accum) (VecType::Load(&out[id]) + val*vw).Store(&out[id]);
              else       (val*vw).Store(&out[id]);
            }
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel, class BlockFn>
    void NearInteracTargets(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const QuadElemList<Real>& qel, BlockFn block) {
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

      thread_local Matrix<Real> M_acc;
      for (Long t = 0; t < Ntrg; t++) {
        Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
        Vector<Real> ntrg;
        if (trg_dot_prod) ntrg.ReInit(COORD_DIM, (Iterator<Real>)normal_trg.begin() + t*COORD_DIM, false);
        block(M_acc, Xtrg, ntrg);
        for (Long pnode = 0; pnode < nnode; pnode++) {
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
              M[pnode*KDIM0+k0][t*KDIM1_out+k1] = M_acc[pnode][k0*KDIM1_out+k1];
            }
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel, class BlockFn>
    void SelfInteracElems(Vector<Matrix<Real>>& M_lst, bool trg_dot_prod, const QuadElemList<Real>& qel, const bool want_tangents, BlockFn block) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      SCTL_ASSERT((Long)M_lst.Dim() == qel.Size());
      const Long nnode = (Long)order * order;
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);

      #pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < qel.Size(); elem_idx++) {
        thread_local Vector<Real> Xnodes, Xnnodes, dXu, dXv;
        qel.GetGeom(&Xnodes, (want_tangents || trg_dot_prod ? &Xnnodes : nullptr), nullptr,
            (want_tangents ? &dXu : nullptr), (want_tangents ? &dXv : nullptr), nds, nds, elem_idx);

        Matrix<Real>& M = M_lst[elem_idx];
        if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nnode*KDIM0, nnode*KDIM1_out);
        M.SetZero();

        for (Integer ti = 0; ti < order; ti++) {
          for (Integer tj = 0; tj < order; tj++) {
            const Long t = ti*order + tj;
            Vector<Real> Xtrg(COORD_DIM, Xnodes.begin() + t*COORD_DIM, false);
            Vector<Real> ntrg;
            if (trg_dot_prod) ntrg.ReInit(COORD_DIM, Xnnodes.begin() + t*COORD_DIM, false);
            block(M, elem_idx, t, ti, tj, Xtrg, ntrg, Xnodes, Xnnodes, dXu, dXv, KDIM1_out);
          }
        }
      }
    }

    template <Integer order, class Real, class KDIM1_t, class MSrc>
    void ScatterSelfBlock(Matrix<Real>& M, const MSrc& src, const Long t, const Integer KDIM0, const KDIM1_t KDIM1_out, const bool from_near) {
      const Long nnode = (Long)order * order;
      for (Long pnode = 0; pnode < nnode; pnode++) {
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
            M[pnode*KDIM0+k0][t*KDIM1_out+k1] = (from_near ? src[pnode*KDIM0+k0][k1] : src[pnode][k0*KDIM1_out+k1]);
          }
        }
      }
    }

  }

  template <class Real> template <class ValueType> QuadElemList<Real>::QuadElemList(Integer order0, const Vector<ValueType>& coord0, const Comm& comm) {
    Init(order0, coord0, comm);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Init(Integer order0, const Vector<ValueType>& coord0, const Comm& comm) {
    order = order0;
    SCTL_ASSERT(order > 0);

    const Long nnode_per_elem = (Long)order * order;
    SCTL_ASSERT(coord0.Dim() % (nnode_per_elem * detail_quadelem::COORD_DIM) == 0);
    const Long nelem_total = coord0.Dim() / (nnode_per_elem * detail_quadelem::COORD_DIM);

    Long i0, i1;
    detail_quadelem::PartitionRange<Real>(nelem_total, comm, i0, i1);
    nelem = i1 - i0;

    coord.ReInit(nelem * detail_quadelem::COORD_DIM * nnode_per_elem);
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      const Long base = elem_idx * detail_quadelem::COORD_DIM * nnode_per_elem;
      const Long src_elem = i0 + elem_idx;
      for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
        for (Long p = 0; p < nnode_per_elem; p++) {
          coord[base + k * nnode_per_elem + p] = (Real)coord0[(src_elem * nnode_per_elem + p) * detail_quadelem::COORD_DIM + k];
        }
      }
    }

    {
      dcoord_du.ReInit(coord.Dim());
      dcoord_dv.ReInit(coord.Dim());
      const Long elem_stride = detail_quadelem::COORD_DIM * nnode_per_elem;
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * elem_stride;
        const Vector<Real> coord_(elem_stride, (Iterator<Real>)coord.begin() + base, false);
        Vector<Real> du_(elem_stride, dcoord_du.begin() + base, false);
        Vector<Real> dv_(elem_stride, dcoord_dv.begin() + base, false);
        detail_quadelem::NodalDerivs<Real>(coord_, order, du_, dv_);
      }
    }

    {
      const Long Nnode = nelem * nnode_per_elem;
      X_node.ReInit(Nnode * detail_quadelem::COORD_DIM);
      Xn_node.ReInit(Nnode * detail_quadelem::COORD_DIM);
      node_cnt.ReInit(nelem);
      node_cnt = nnode_per_elem;

      const auto& nodes = ParamNodes(order);
      const Long elem_stride = nnode_per_elem * detail_quadelem::COORD_DIM;
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

  template <class Real> void QuadElemList<Real>::GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx, const Vector<Real>* origin) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nu = u_param.Dim();
    const Long Nv = v_param.Dim();
    const Long N = Nu * Nv;

    if (X && X->Dim() != N * detail_quadelem::COORD_DIM) X->ReInit(N * detail_quadelem::COORD_DIM);
    if (Xn && Xn->Dim() != N * detail_quadelem::COORD_DIM) Xn->ReInit(N * detail_quadelem::COORD_DIM);
    if (Xa && Xa->Dim() != N) Xa->ReInit(N);
    if (dX_du && dX_du->Dim() != N * detail_quadelem::COORD_DIM) dX_du->ReInit(N * detail_quadelem::COORD_DIM);
    if (dX_dv && dX_dv->Dim() != N * detail_quadelem::COORD_DIM) dX_dv->ReInit(N * detail_quadelem::COORD_DIM);

    thread_local Matrix<Real> Mu, MuT, Mv;
    if (Mu.Dim(0) != order || Mu.Dim(1) != Nu) {
      Mu.ReInit(order, Nu);
      MuT.ReInit(Nu, order);
    }
    if (Mv.Dim(0) != order || Mv.Dim(1) != Nv) Mv.ReInit(order, Nv);
    { Vector<Real> Mu_(order * Nu, Mu.begin(), false);
      Vector<Real> Mv_(order * Nv, Mv.begin(), false);
      LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_param);
      LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_param); }
    for (Integer i = 0; i < order; i++) for (Long a = 0; a < Nu; a++) MuT[a][i] = Mu[i][a];

    SCTL_ASSERT(elem_idx >= 0 && elem_idx < nelem);
    const Long base = elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM;
    const Vector<Real> coord_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)coord.begin() + base, false);
    const Vector<Real> dcoord_du_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_du.begin() + base, false);
    const Vector<Real> dcoord_dv_(detail_quadelem::COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_dv.begin() + base, false);

    thread_local Vector<Real> coord_shift, du_shift, dv_shift;
    const Vector<Real>* pos_in = &coord_;
    const Vector<Real>* du_in = &dcoord_du_;
    const Vector<Real>* dv_in = &dcoord_dv_;
    if (origin) {
      if (coord_shift.Dim() != detail_quadelem::COORD_DIM * nnode_per_elem) coord_shift.ReInit(detail_quadelem::COORD_DIM * nnode_per_elem);
      for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
        const Real ok = (*origin)[k];
        for (Long p = 0; p < nnode_per_elem; p++) coord_shift[k * nnode_per_elem + p] = coord_[k * nnode_per_elem + p] - ok;
      }
      if (Xn || Xa || dX_du || dX_dv) detail_quadelem::NodalDerivs<Real>(coord_shift, order, du_shift, dv_shift);
      pos_in = &coord_shift;
      du_in = &du_shift;
      dv_in = &dv_shift;
    }

    if (X) {
      thread_local Vector<Real> X_soa;
      detail_quadelem::EvalTensorProduct(X_soa, *pos_in, MuT, Mv);
      for (Long i = 0; i < N; i++) {
        (*X)[i * detail_quadelem::COORD_DIM + 0] = X_soa[0 * N + i];
        (*X)[i * detail_quadelem::COORD_DIM + 1] = X_soa[1 * N + i];
        (*X)[i * detail_quadelem::COORD_DIM + 2] = X_soa[2 * N + i];
      }
    }
    if (Xn || Xa || dX_du || dX_dv) {
      thread_local Vector<Real> dXdu_soa, dXdv_soa;
      detail_quadelem::EvalTensorProduct(dXdu_soa, *du_in, MuT, Mv);
      detail_quadelem::EvalTensorProduct(dXdv_soa, *dv_in, MuT, Mv);
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

    if (X.Dim() != Nnode * detail_quadelem::COORD_DIM) X.ReInit(Nnode * detail_quadelem::COORD_DIM);
    if (Xn.Dim() != Nnode * detail_quadelem::COORD_DIM) Xn.ReInit(Nnode * detail_quadelem::COORD_DIM);
    if (wts.Dim() != Nnode) wts.ReInit(Nnode);
    if (dist_far.Dim() != Nnode) dist_far.ReInit(Nnode);
    if (element_wise_node_cnt.Dim() != nelem) element_wise_node_cnt.ReInit(nelem);
    element_wise_node_cnt = nnode_per_elem;

    const auto& nodes = ParamNodes(order);
    const auto& node_wts = LegQuadRule<Real>::wts(order);

    Vector<Real> dist_nodes(order);
    {
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

    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      Vector<Real> X_(nnode_per_elem * detail_quadelem::COORD_DIM, X.begin() + elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM, false);
      Vector<Real> Xn_(nnode_per_elem * detail_quadelem::COORD_DIM, Xn.begin() + elem_idx * nnode_per_elem * detail_quadelem::COORD_DIM, false);
      Vector<Real> wts_(nnode_per_elem, wts.begin() + elem_idx * nnode_per_elem, false);
      Vector<Real> dist_far_(nnode_per_elem, dist_far.begin() + elem_idx * nnode_per_elem, false);

      Vector<Real> Xa, dXdu, dXdv;
      GetGeom(&X_, &Xn_, &Xa, &dXdu, &dXdv, nodes, nodes, elem_idx);

      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          const Long p = i * order + j;
          const Real wu = node_wts[i];
          const Real wv = node_wts[j];
          wts_[p] = Xa[p] * wu * wv;

          const Real du = sqrt<Real>(dXdu[p * detail_quadelem::COORD_DIM + 0] * dXdu[p * detail_quadelem::COORD_DIM + 0] +
              dXdu[p * detail_quadelem::COORD_DIM + 1] * dXdu[p * detail_quadelem::COORD_DIM + 1] +
              dXdu[p * detail_quadelem::COORD_DIM + 2] * dXdu[p * detail_quadelem::COORD_DIM + 2]);
          const Real dv = sqrt<Real>(dXdv[p * detail_quadelem::COORD_DIM + 0] * dXdv[p * detail_quadelem::COORD_DIM + 0] +
              dXdv[p * detail_quadelem::COORD_DIM + 1] * dXdv[p * detail_quadelem::COORD_DIM + 1] +
              dXdv[p * detail_quadelem::COORD_DIM + 2] * dXdv[p * detail_quadelem::COORD_DIM + 2]);
          dist_far_[p] = std::max(dist_nodes[i] * du, dist_nodes[j] * dv);
        }
      }
    }
  }

  namespace detail_dyadic_near {

    template <class Real> void QuadParams(const Real tol, Real& b_ellipse, Integer& QuadOrder) {
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const double d = -std::log10((double)tol_);
      const double rho = std::min(3.0, std::max(2.0, 2.0 + 0.25*(d - 6)));
      const double C = std::max(1e-3, (15.0*(rho*rho - 1))/64.0);
      QuadOrder = std::max<Integer>(2, (Integer)std::ceil(-std::log(C*(double)tol_)/std::log(rho)*0.5 + 1));

      const Real rho_ = (Real)rho;
      const Real a = (rho_ + 1/rho_)/2, b = (rho_ - 1/rho_)/2;
      b_ellipse = b*b/(2*a);
    }

    template <class Real> inline Integer DigitsQuadOrder(const Integer digits) {
      static const std::array<Integer,detail_quadelem::MaxDigits<Real>> q = []() {
        std::array<Integer,detail_quadelem::MaxDigits<Real>> t{};
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) {
          Real b;
          Integer qq;
          QuadParams(pow<Real,Long>((Real)0.1, (Long)d), b, qq);
          t[d] = qq;
        }
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      return q[digits];
    }

    template <class Real> inline Real DigitsBEllipse(const Integer digits) {
      static const std::array<Real,detail_quadelem::MaxDigits<Real>> b = []() {
        std::array<Real,detail_quadelem::MaxDigits<Real>> t{};
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) {
          Real bb;
          Integer qq;
          QuadParams(pow<Real,Long>((Real)0.1, (Long)d), bb, qq);
          t[d] = bb;
        }
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      return b[digits];
    }

    template <class Real> struct GradeRule {
      Vector<Real> nds, w;
      Matrix<Real> T, dT, TT, TD;
      Real a, b;
    };

    static constexpr Integer NearMaxQuadOrder = 60;

    template <class Real> static const Vector<Real>& NearSubNodes(const Integer order) {
      constexpr Integer MAX_ORDER = 50;
      SCTL_ASSERT(1 < order && order <= MAX_ORDER);
      static const Vector<Vector<Real>> all = []() {
        Vector<Vector<Real>> v(MAX_ORDER + 1);
        for (Integer n = 2; n <= MAX_ORDER; n++) {
          v[n].ReInit(n);
          using W = detail_quadelem::PrecompReal;
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

    template <class Real> static const Vector<Real>& NearSubOffsets(const Integer order) {
      constexpr Integer MAX_ORDER = 50;
      SCTL_ASSERT(1 < order && order <= MAX_ORDER);
      static const Vector<Vector<Real>> all = []() {
        Vector<Vector<Real>> v(MAX_ORDER + 1);
        for (Integer n = 2; n <= MAX_ORDER; n++) {
          v[n].ReInit(n);
          using W = detail_quadelem::PrecompReal;
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

    template <class Real> static const Matrix<Real>& NearSubDiffMat(const Integer order) {
      constexpr Integer MAX_ORDER = 50;
      SCTL_ASSERT(1 < order && order <= MAX_ORDER);
      static const Vector<Matrix<Real>> all = []() {
        Vector<Matrix<Real>> D(MAX_ORDER + 1);
        for (Integer n = 2; n <= MAX_ORDER; n++) {
          const Vector<Real>& nds = NearSubNodes<Real>(n);
          Vector<Real> f((Long)n*n);
          f.SetZero();
          for (Integer i = 0; i < n; i++) f[i*n + i] = 1;
          Vector<Real> df;
          LagrangeInterp<Real>::Derivative(df, f, nds);
          D[n].ReInit(n, n);
          for (Integer i = 0; i < n; i++) for (Integer a = 0; a < n; a++) D[n][i][a] = df[i*n + a];
        }
        return D;
      }();
      return all[order];
    }

    template <Integer order, class Real> const Vector<GradeRule<Real>>& NearGradeTable(const Integer q) {
      auto build = [](const Integer q) {
        using W = detail_quadelem::PrecompReal;
        const Vector<W>& sig = NearSubOffsets<W>(order);
        const Matrix<W>& Dsub = NearSubDiffMat<W>(order);
        Vector<W> qn, qw;
        LegQuadRule<W>::template ComputeNdsWts<W>(&qn, &qw, q);
        Vector<GradeRule<Real>> tab(2*detail_quadelem::MaxRefineLvl<Real>);
        Vector<W> tq(q), Twts((Long)order*q);
        Matrix<W> dT(order, q);
        auto fill = [&](GradeRule<Real>& r, const Real a, const Real b) {
          r.a = a;
          r.b = b;
          const W aw = (W)a, w = (W)b - (W)a;
          r.nds.ReInit(q);
          r.w.ReInit(q);
          for (Integer i = 0; i < q; i++) {
            r.nds[i] = (Real)(aw + w*qn[i]);
            r.w[i] = (Real)(w*qw[i]);
          }
          const W t_hi = (W)1 - aw, t_w = t_hi - ((W)1 - (W)b);
          for (Integer j = 0; j < q; j++) tq[j] = t_hi - t_w*qn[j];
          LagrangeInterp<W>::Interpolate(Twts, sig, tq);
          const Matrix<W> T(order, q, Twts.begin(), false);
          Matrix<W>::GEMM(dT, Dsub, T);
          r.T.ReInit(order, q);
          r.dT.ReInit(order, q);
          r.TT.ReInit(q, order);
          r.TD.ReInit(2*q, order);
          for (Integer i = 0; i < order; i++) for (Integer j = 0; j < q; j++) {
            r.T[i][j] = (Real)T[i][j];
            r.dT[i][j] = (Real)dT[i][j];
            r.TT[j][i] = r.T[i][j];
            r.TD[j][i] = r.T[i][j];
            r.TD[q+j][i] = r.dT[i][j];
          }
        };
        for (Integer k = 0; k < detail_quadelem::MaxRefineLvl<Real>; k++) {
          const Real lo = 1 - pow<Real>((Real)0.5, k), hi = 1 - pow<Real>((Real)0.5, k+1);
          fill(tab[k], lo, hi);
          fill(tab[detail_quadelem::MaxRefineLvl<Real> + k], lo, (Real)1);
        }
        return tab;
      };
      static const std::vector<Vector<GradeRule<Real>>> all = [&build]() {
        std::vector<Vector<GradeRule<Real>>> t(NearMaxQuadOrder+1);
        for (Integer q = 4; q <= NearMaxQuadOrder; q += 4) t[q] = build(q);
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) {
          const Integer qi = DigitsQuadOrder<Real>(d);
          if (qi > 0 && qi <= NearMaxQuadOrder && t[qi].Dim() == 0) t[qi] = build(qi);
        }
        return t;
      }();
      SCTL_ASSERT(q > 0 && q <= NearMaxQuadOrder && all[q].Dim());
      return all[q];
    }

    template <Integer order, class Real, class Kernel> void IntegrateCell(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker, const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD, const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT, const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Long nnode = (Long)order * order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / detail_quadelem::COORD_DIM : KDIM1full;

      const Long Nu = Mu.Dim(1), Nv = Mv.Dim(1), nq = Nu * Nv;
      if (!nq) return;
      const Integer C = KDIM0 * KDIM1_out;

      thread_local Vector<Real> Cv, Cdv;
      if (Cv.Dim() != detail_quadelem::COORD_DIM*order*Nv) {
        Cv.ReInit(detail_quadelem::COORD_DIM*order*Nv);
        Cdv.ReInit(detail_quadelem::COORD_DIM*order*Nv);
      }
      {
        const Matrix<Real> cs_all(detail_quadelem::COORD_DIM*order, order, (Iterator<Real>)src_nodal.begin(), false);
        Matrix<Real> Cv_all (detail_quadelem::COORD_DIM*order, Nv, Cv.begin(),  false);
        Matrix<Real> Cdv_all(detail_quadelem::COORD_DIM*order, Nv, Cdv.begin(), false);
        Matrix<Real>::GEMM(Cv_all,  cs_all, Mv);
        Matrix<Real>::GEMM(Cdv_all, cs_all, dMv);
      }
      const Long ldc = detail_quadelem::COORD_DIM*Nv;
      thread_local Vector<Real> Cvc, Cdvc, XdU, dXdv_soa;
      if (Cvc.Dim() != (Long)order*ldc) {
        Cvc.ReInit((Long)order*ldc);
        Cdvc.ReInit((Long)order*ldc);
      }
      for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
        for (Integer i = 0; i < order; i++) {
          const Long src = ((Long)k*order + i)*Nv, dst = (Long)i*ldc + k*Nv;
          for (Long b = 0; b < Nv; b++) {
            Cvc[dst+b] = Cv[src+b];
            Cdvc[dst+b] = Cdv[src+b];
          }
        }
      }
      if (XdU.Dim() != 2*(Long)Nu*ldc) {
        XdU.ReInit(2*(Long)Nu*ldc);
        dXdv_soa.ReInit((Long)Nu*ldc);
      }
      {
        const Matrix<Real> Cvc_m(order, ldc, Cvc.begin(), false), Cdvc_m(order, ldc, Cdvc.begin(), false);
        Matrix<Real> dV_m(Nu, ldc, dXdv_soa.begin(), false);
        {
          Matrix<Real> XdU_m(2*Nu, ldc, XdU.begin(), false);
          Matrix<Real>::GEMM(XdU_m, MuD, Cvc_m);
        }
        Matrix<Real>::GEMM(dV_m, MuT, Cdvc_m);
      }

      thread_local Vector<Real> Xsrc, Xnsrc, wq;
      if (Xsrc.Dim() != nq*detail_quadelem::COORD_DIM) {
        Xsrc.ReInit(nq*detail_quadelem::COORD_DIM);
        Xnsrc.ReInit(nq*detail_quadelem::COORD_DIM);
        wq.ReInit(nq);
      }
      for (Long a = 0; a < Nu; a++) {
        for (Long b = 0; b < Nv; b++) {
          const Long q = a*Nv + b;
          const Long r = (Long)a*ldc + b, ru = ((Long)Nu + a)*ldc + b;
          const Real du0 = XdU[ru+0*Nv], du1 = XdU[ru+1*Nv], du2 = XdU[ru+2*Nv];
          const Real dv0 = dXdv_soa[r+0*Nv], dv1 = dXdv_soa[r+1*Nv], dv2 = dXdv_soa[r+2*Nv];
          const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
          const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
          const Real inv_area = (area > 0 ? nrm_sign/area : 0);
          Xsrc[0*nq+q] = XdU[r+0*Nv];
          Xsrc[1*nq+q] = XdU[r+1*Nv];
          Xsrc[2*nq+q] = XdU[r+2*Nv];
          Xnsrc[0*nq+q] = n0*inv_area;
          Xnsrc[1*nq+q] = n1*inv_area;
          Xnsrc[2*nq+q] = n2*inv_area;
          wq[q] = area*wu[a]*wv[b];
        }
      }

      thread_local Vector<Real> KWc;
      if (KWc.Dim() != C*nq) KWc.ReInit(C*nq);
      static constexpr bool HAS_N = detail_quadelem::UKerNeedsN<Kernel, Vec<Real,1>>::value;
      using WVec = Vec<Real, DefaultVecLen<Real>()>;
      const Long qmain = (nq/WVec::Size())*WVec::Size();
      const auto fold = [&](const auto has_proxy) {
        constexpr bool HP = decltype(has_proxy)::value;
        const Long np = (HP ? proxy_w.Dim() : 1);
        for (Long j = 0; j < np; j++) {
          StaticArray<Real,detail_quadelem::COORD_DIM> Xtj{0, 0, 0};
          if constexpr (HP) for (Integer l = 0; l < detail_quadelem::COORD_DIM; l++) Xtj[l] = proxy_off[j*detail_quadelem::COORD_DIM+l];
          const Vector<Real> Xtj_v(detail_quadelem::COORD_DIM, Xtj, false);
          const Real wj = (HP ? proxy_w[j] : (Real)1);
          const bool accum = (HP && j > 0);
          const ConstIterator<Real> xt = Xtj_v.begin(), xs = Xsrc.begin(), xn = Xnsrc.begin(), w = wq.begin();
          const ConstIterator<Real> nt = (trg_dot_prod ? normal_trg.begin() : ConstIterator<Real>(NullIterator<Real>()));
          if (trg_dot_prod) {
            detail_quadelem::KerFoldSoA<Real,Kernel,WVec,        HAS_N,true >(KWc.begin(), xt, xs, xn, w, nq, nq,     0, qmain, wj, accum, nt);
            detail_quadelem::KerFoldSoA<Real,Kernel,Vec<Real,1>, HAS_N,true >(KWc.begin(), xt, xs, xn, w, nq, nq, qmain,    nq, wj, accum, nt);
          } else {
            detail_quadelem::KerFoldSoA<Real,Kernel,WVec,        HAS_N,false>(KWc.begin(), xt, xs, xn, w, nq, nq,     0, qmain, wj, accum, nt);
            detail_quadelem::KerFoldSoA<Real,Kernel,Vec<Real,1>, HAS_N,false>(KWc.begin(), xt, xs, xn, w, nq, nq, qmain,    nq, wj, accum, nt);
          }
        }
      };
      if (proxy_w.Dim()) fold(std::true_type{}); else fold(std::false_type{});

      thread_local Vector<Real> Yv;
      if (Yv.Dim() != (Long)C*Nu*order) Yv.ReInit((Long)C*Nu*order);
      {
        const Matrix<Real> KW_all((Long)C*Nu, Nv, KWc.begin(), false);
        Matrix<Real> Y_all((Long)C*Nu, order, Yv.begin(), false);
        Matrix<Real>::GEMM(Y_all, KW_all, MvT);
      }
      for (Integer c = 0; c < C; c++) {
        const Matrix<Real> Y_c(Nu, order, Yv.begin() + (Long)c*Nu*order, false);
        Matrix<Real> A_c(order, order, acc_cm.begin() + (Long)c*nnode, false);
        Matrix<Real>::GEMM(A_c, Mu, Y_c, (Real)1);
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracBlockDyadic(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      const Long nnode = (Long)order*order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full/detail_quadelem::COORD_DIM : KDIM1full;
      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      const Integer C_ = KDIM0*KDIM1_out;
      thread_local Vector<Real> acc, accB, accE;
      if (acc.Dim() != (Long)C_*nnode) {
        acc.ReInit((Long)C_*nnode);
        accB.ReInit((Long)C_*nnode);
        accE.ReInit(nnode);
      }
      M_acc.SetZero();

      const Real b_ellipse = DigitsBEllipse<Real>(digits);

      Real ustar, vstar;
      const Real dist = detail_quadelem::GetClosestPoint(qel, ustar, vstar, elem_idx, Xtrg);

      const auto near_order = [](const Real* dXu, const Real* dXv, const Integer q_iso) {
        Real guu=0, gvv=0, guv=0;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          guu+=dXu[k]*dXu[k];
          gvv+=dXv[k]*dXv[k];
          guv+=dXu[k]*dXv[k];
        }
        const Real den = sqrt<Real>(guu*gvv);
        if (!(den > 0)) return q_iso;
        const double c = std::min(1.0, (double)(fabs<Real>(guv)/den));
        const double phi = std::acos(c)*180.0/const_pi<double>();
        constexpr double Ck = 400.0;
        const double f = std::max(1.0, Ck/(10.0*std::max(1e-3, phi)));
        if (f <= 1.0) return q_iso;
        Integer q = (Integer)std::ceil(f*(double)q_iso);
        q = ((q + 3)/4)*4;
        return std::min<Integer>(NearMaxQuadOrder, std::max<Integer>(q_iso, q));
      };
      Real spd_u, spd_v;
      Integer q_near;
      {
        Real Xc[detail_quadelem::COORD_DIM], dXu[detail_quadelem::COORD_DIM], dXv[detail_quadelem::COORD_DIM];
        detail_quadelem::EvalPoint<Real>(qel, Xc, dXu, dXv, ustar, vstar, elem_idx, nullptr);
        Real su2 = 0, sv2 = 0;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          su2 += dXu[k]*dXu[k];
          sv2 += dXv[k]*dXv[k];
        }
        spd_u = sqrt<Real>(su2);
        spd_v = sqrt<Real>(sv2);
        q_near = near_order(&dXu[0], &dXv[0], DigitsQuadOrder<Real>(digits));
      }
      const Vector<GradeRule<Real>>& tab = NearGradeTable<order,Real>(q_near);
      const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};

      thread_local Vector<Real> cs;
      {
        if (cs.Dim() != detail_quadelem::COORD_DIM*nnode) cs.ReInit(detail_quadelem::COORD_DIM*nnode);
        const Long base = elem_idx * nnode * detail_quadelem::COORD_DIM;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          const Real ok = Xtrg[k];
          for (Long p = 0; p < nnode; p++) cs[k*nnode + p] = detail_quadelem::Access<Real>::Coord(qel)[base + k*nnode + p] - ok;
        }
      }
      const auto build_interp = [](Matrix<Real> (&Sf)[2][2], Matrix<Real> (&St)[2][2], const Real (&slen)[2][2], const Real ustar, const Real vstar) {
        const Long nnode = (Long)order*order;
        const Vector<Real>& gnds = QuadElemList<Real>::ParamNodes(order);
        const Vector<Real>& soff = NearSubOffsets<Real>(order);
        thread_local Vector<Real> gsh, sub, Sbuf;
        if (sub.Dim() != order) {
          gsh.ReInit(order);
          sub.ReInit(order);
          Sbuf.ReInit(nnode);
        }
        for (Integer d = 0; d < 2; d++) {
          const Real xs = (d ? vstar : ustar);
          for (Integer i = 0; i < order; i++) gsh[i] = gnds[i] - xs;
          for (Integer sd = 0; sd < 2; sd++) {
            if (!(slen[d][sd] > 0)) continue;
            const Real sg = (sd ? slen[d][sd] : -slen[d][sd]);
            for (Integer i = 0; i < order; i++) sub[i] = sg*soff[i];
            {
              Vector<Real> v(nnode, Sbuf.begin(), false);
              LagrangeInterp<Real>::Interpolate(v, gsh, sub);
            }
            Sf[d][sd].ReInit(order, order);
            St[d][sd].ReInit(order, order);
            for (Integer i = 0; i < order; i++) for (Integer aa = 0; aa < order; aa++) {
              Sf[d][sd][i][aa] = Sbuf[i*order+aa];
              St[d][sd][aa][i] = Sbuf[i*order+aa];
            }
          }
        }
      };
      thread_local Matrix<Real> Sf[2][2], St[2][2];
      build_interp(Sf, St, slen, ustar, vstar);
      const auto build_geom = [](Vector<Real> (&Xsub)[2][2], Vector<Real>& cs, const Matrix<Real> (&Sf)[2][2], const Matrix<Real> (&St)[2][2], const Real (&slen)[2][2]) {
        const Long nnode = (Long)order*order;
        thread_local Vector<Real> Av[2];
        for (Integer sdv = 0; sdv < 2; sdv++) {
          if (!(slen[1][sdv] > 0)) continue;
          if (Av[sdv].Dim() != detail_quadelem::COORD_DIM*nnode) Av[sdv].ReInit(detail_quadelem::COORD_DIM*nnode);
          const Matrix<Real> cs_all(detail_quadelem::COORD_DIM*order, order, cs.begin(), false);
          Matrix<Real> A_all(detail_quadelem::COORD_DIM*order, order, Av[sdv].begin(), false);
          Matrix<Real>::GEMM(A_all, cs_all, Sf[1][sdv]);
        }
        for (Integer sdu = 0; sdu < 2; sdu++) {
          if (!(slen[0][sdu] > 0)) continue;
          for (Integer sdv = 0; sdv < 2; sdv++) {
            if (!(slen[1][sdv] > 0)) continue;
            if (Xsub[sdu][sdv].Dim() != detail_quadelem::COORD_DIM*nnode) Xsub[sdu][sdv].ReInit(detail_quadelem::COORD_DIM*nnode);
            for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
              const Matrix<Real> A_k(order, order, Av[sdv].begin() + k*nnode, false);
              Matrix<Real> X_k(order, order, Xsub[sdu][sdv].begin() + k*nnode, false);
              Matrix<Real>::GEMM(X_k, St[0][sdu], A_k);
            }
          }
        }
      };
      thread_local Vector<Real> Xsub[2][2];
      build_geom(Xsub, cs, Sf, St, slen);

      const auto emit = [&tab, &normal_trg, &ker, &proxy_off, &proxy_w](const Integer sdu, const Integer sdv, const Integer iu, const Integer iv) {
        const GradeRule<Real>& gu = tab[iu];
        const GradeRule<Real>& gv = tab[iv];
        if (!(gu.b > gu.a) || !(gv.b > gv.a)) return;
        const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
        IntegrateCell<order,Real>(normal_trg, gu.w, gv.w, ker,
            gu.T, gu.TT, gu.TD, gv.T, gv.dT, gv.TT,
            Xsub[sdu][sdv], nsign, acc, proxy_off, proxy_w);
      };
      const auto refine = [&emit, dist, b_ellipse](const Integer sdu, const Integer sdv, Real hu, Real hv) {
        Integer ku = 0, kv = 0;
        const bool cap = !(dist > 0) || isinf<Real>(dist) || isnan<Real>(dist);
        constexpr Integer KMAX = detail_quadelem::MaxRefineLvl<Real>-1;
        while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
          if (hu >= hv && ku < KMAX) {
            emit(sdu, sdv, ku, detail_quadelem::MaxRefineLvl<Real> + kv);
            ku++;
            hu *= (Real)0.5;
          } else if (kv < KMAX) {
            emit(sdu, sdv, detail_quadelem::MaxRefineLvl<Real> + ku, kv);
            kv++;
            hv *= (Real)0.5;
          } else if (ku < KMAX) {
            emit(sdu, sdv, ku, detail_quadelem::MaxRefineLvl<Real> + kv);
            ku++;
            hu *= (Real)0.5;
          } else break;
        }
        emit(sdu, sdv, detail_quadelem::MaxRefineLvl<Real> + ku, detail_quadelem::MaxRefineLvl<Real> + kv);
      };
      for (Integer sdu = 0; sdu < 2; sdu++) {
        if (!(slen[0][sdu] > 0)) continue;
        for (Integer sdv = 0; sdv < 2; sdv++) {
          if (!(slen[1][sdv] > 0)) continue;
          acc.SetZero();
          refine(sdu, sdv, slen[0][sdu]*spd_u, slen[1][sdv]*spd_v);

          {
            const Matrix<Real> A_all((Long)C_*order, order, acc.begin(), false);
            Matrix<Real> B_all((Long)C_*order, order, accB.begin(), false);
            Matrix<Real>::GEMM(B_all, A_all, St[1][sdv]);
            for (Integer c = 0; c < C_; c++) {
              const Matrix<Real> B_c(order, order, accB.begin() + (Long)c*nnode, false);
              Matrix<Real> E_c(order, order, accE.begin(), false);
              Matrix<Real>::GEMM(E_c, Sf[0][sdu], B_c);
              for (Long p = 0; p < nnode; p++) M_acc[p][c] += accE[p];
            }
          }
        }
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracDyadic(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits, const Vector<Real>& proxy_off = Vector<Real>(), const Vector<Real>& proxy_w = Vector<Real>()) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      detail_quadelem::NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel,
          [&](Matrix<Real>& M_acc, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
          NearInteracBlockDyadic<order,Real>(M_acc, qel, elem_idx, Xtrg, ntrg, ker, digits, proxy_off, proxy_w);
          });
    }

  }

  namespace detail_duffy {

    template <class Real> struct DuffyTri {
      bool swap_ab = false;
      Real nsign = 1;
      Real J0 = 0;
      Matrix<Real> WbC;
      Matrix<Real> WbT;
      Vector<Matrix<Real>> MiC, MiT;
    };
    template <class Real> struct DuffySelfTable {
      Integer ns = 0;
      Vector<Real> sn, sw;
      std::vector<DuffyTri<Real>> tri;
    };

    template <class Real> inline Integer DuffyTRuleOrder(const Integer digits, const Integer order, const Integer kdim0) {
      const double per_digit = (kdim0 > 1 ? 4.0 : 2.5);
      return std::max<Integer>(order/2, (Integer)std::ceil(per_digit*(double)digits));
    }

    template <Integer order, class Real> const DuffySelfTable<Real>& DuffyTable() {
      static const DuffySelfTable<Real> table = []() {
        DuffySelfTable<Real> tbl;
        const Integer qs = order;
        tbl.ns = qs;
        LegQuadRule<Real>::ComputeNdsWts(&tbl.sn, &tbl.sw, qs);

        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Matrix<Real>& D = detail_quadelem::DiffMat<Real>(order);
        tbl.tri.resize((size_t)(4*order*order));
        const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
        for (Integer ti = 0; ti < order; ti++) for (Integer tj = 0; tj < order; tj++) {
          const Real u0 = nds[ti], v0 = nds[tj];
          for (Integer kt = 0; kt < 4; kt++) {
            DuffyTri<Real>& T = tbl.tri[(size_t)((ti*order + tj)*4 + kt)];
            const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
            const Real b[2] = {cu[(kt+1)%4]-u0, cv[(kt+1)%4]-v0};
            const Real e[2] = {b[0]-a[0], b[1]-a[1]};
            T.J0 = a[0]*b[1] - a[1]*b[0];
            SCTL_ASSERT_MSG(T.J0 > 0, "Duffy triangle orientation");
            T.swap_ab = (fabs<Real>(e[0]) < fabs<Real>(e[1]));
            T.nsign = (T.swap_ab ? (Real)-1 : (Real)1);
            const Real al0 = (T.swap_ab ? v0 : u0), be0 = (T.swap_ab ? u0 : v0);
            const Real aal = (T.swap_ab ? a[1] : a[0]), abe = (T.swap_ab ? a[0] : a[1]);
            const Real eal = (T.swap_ab ? e[1] : e[0]);
            {
              Vector<Real> bv(qs);
              for (Integer i = 0; i < qs; i++) bv[i] = be0 + tbl.sn[i]*abe;
              Matrix<Real> Wb(order, qs), WbD(order, qs);
              {
                Vector<Real> t((Long)order*qs, Wb.begin(), false);
                LagrangeInterp<Real>::Interpolate(t, nds, bv);
              }
              Matrix<Real>::GEMM(WbD, D, Wb);
              T.WbC.ReInit(order, 2*qs);
              for (Integer r = 0; r < order; r++) for (Integer i = 0; i < qs; i++) {
                T.WbC[r][i] = Wb[r][i];
                T.WbC[r][qs+i] = WbD[r][i];
              }
              T.WbT = Wb.Transpose();
            }
            {
              T.MiC.ReInit(qs);
              T.MiT.ReInit(qs);
              Vector<Real> av(order);
              Matrix<Real> Mi(order, order), MiD(order, order);
              for (Integer i = 0; i < qs; i++) {
                for (Integer k = 0; k < order; k++) av[k] = al0 + tbl.sn[i]*(aal + nds[k]*eal);
                {
                  Vector<Real> t((Long)order*order, Mi.begin(), false);
                  LagrangeInterp<Real>::Interpolate(t, nds, av);
                }
                Matrix<Real>::GEMM(MiD, D, Mi);
                T.MiC[i].ReInit(order, 2*order);
                for (Integer r = 0; r < order; r++) for (Integer k = 0; k < order; k++) {
                  T.MiC[i][r][k] = Mi[r][k];
                  T.MiC[i][r][order+k] = MiD[r][k];
                }
                T.MiT[i] = Mi.Transpose();
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
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full/detail_quadelem::COORD_DIM : KDIM1full;
      const Integer C = KDIM0*KDIM1_out;
      constexpr Integer NR = 3*detail_quadelem::COORD_DIM;
      constexpr Integer NA = 2*detail_quadelem::COORD_DIM;
      M_acc.ReInit(nnode, C);
      M_acc.SetZero();

      const DuffySelfTable<Real>& tbl = DuffyTable<order,Real>();
      const Long ns = tbl.ns, nt = DuffyTRuleOrder<Real>(digits, order, KDIM0);
      static constexpr Integer MaxGLOrder = 128;
      const Vector<Real>& qn = LegQuadRule<Real>::template nds<MaxGLOrder>(nt);
      const Vector<Real>& qw = LegQuadRule<Real>::template wts<MaxGLOrder>(nt);
      const Long sblk = ns;
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      const Matrix<Real>& D = detail_quadelem::DiffMat<Real>(order);

      auto ash = [](const Real x) { return log<Real>(x + sqrt<Real>(x*x + (Real)1)); };

      thread_local Vector<Real> cs;
      if (cs.Dim() != detail_quadelem::COORD_DIM*nnode) cs.ReInit(detail_quadelem::COORD_DIM*nnode);
      const Long base = elem_idx*nnode*detail_quadelem::COORD_DIM;

      for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long q = 0; q < nnode; q++) cs[k*nnode + q] = detail_quadelem::Access<Real>::Coord(qel)[base + k*nnode + q] - ok;
      }

      Real G[4];
      {
        Real du[detail_quadelem::COORD_DIM], dv[detail_quadelem::COORD_DIM];
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          Real su = 0, sv = 0;
          for (Integer i = 0; i < order; i++) su += cs[k*nnode + (Long)i*order + tj]*D[i][ti];
          for (Integer j = 0; j < order; j++) sv += cs[k*nnode + (Long)ti*order + j]*D[j][tj];
          du[k] = su;
          dv[k] = sv;
        }
        Real guu = 0, guv = 0, gvv = 0;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          guu += du[k]*du[k];
          guv += du[k]*dv[k];
          gvv += dv[k]*dv[k];
        }
        G[0] = guu;
        G[1] = guv;
        G[2] = guv;
        G[3] = gvv;
      }

      StaticArray<Real,detail_quadelem::COORD_DIM> Xt0{0,0,0};
      const Vector<Real> Xt0_v(detail_quadelem::COORD_DIM, Xt0, false);
      const Vector<Real>& pnds = nds;

      for (Integer kt = 0; kt < 4; kt++) {
        const DuffyTri<Real>& T = tbl.tri[(size_t)((ti*order + tj)*4 + kt)];
        const Real u0 = nds[ti], v0 = nds[tj];
        const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
        const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
        const Real e[2] = {cu[(kt+1)%4]-cu[kt], cv[(kt+1)%4]-cv[kt]};

        Real tstar, dOverL;
        {
          const Real Me[2] = {G[0]*e[0]+G[1]*e[1], G[2]*e[0]+G[3]*e[1]};
          const Real am = e[0]*Me[0] + e[1]*Me[1];
          Real ts = -(a[0]*Me[0] + a[1]*Me[1])/am;
          ts = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
          const Real c[2] = {a[0]+ts*e[0], a[1]+ts*e[1]};
          const Real d2 = c[0]*(G[0]*c[0]+G[1]*c[1]) + c[1]*(G[2]*c[0]+G[3]*c[1]);
          tstar = ts;
          dOverL = sqrt<Real>(d2)/sqrt<Real>(am);
        }

        const Long szt = 2*nt + (Long)order*nt + (Long)nt*order;
        ScratchBuf<Real> sbt(szt);
        Long offt = 0;
        auto taket = [&](const Long n) {
          Iterator<Real> r = sbt.begin() + offt;
          offt += n;
          return r;
        };
        Vector<Real> tn(nt, taket(nt), false), tw(nt, taket(nt), false);
        {
          const Real dd = dOverL;
          const Real x0 = -ash(tstar/dd), x1 = ash(((Real)1-tstar)/dd);
          for (Long i = 0; i < nt; i++) {
            const Real xi = x0 + (x1-x0)*qn[i];
            const Real ex = exp<Real>(xi), iex = (Real)1/ex;
            tn[i] = tstar + dd*(ex-iex)/(Real)2;
            tw[i] = dd*(ex+iex)/(Real)2*(x1-x0)*qw[i];
          }
        }
        Matrix<Real> Tt(order, nt, taket((Long)order*nt), false), TtT(nt, order, taket((Long)nt*order), false);
        {
          Vector<Real> t((Long)order*nt, Tt.begin(), false);
          LagrangeInterp<Real>::Interpolate(t, pnds, tn);
        }
        for (Integer r = 0; r < order; r++) for (Long j = 0; j < nt; j++) TtT[j][r] = Tt[r][j];

        const Long nq = ns*nt;
        const Long sz = detail_quadelem::COORD_DIM*nnode + 2*detail_quadelem::COORD_DIM*(Long)order*ns + (Long)NA*order + 2*(Long)NA*order
          + sblk*NR*(Long)order + sblk*NR*nt + 2*detail_quadelem::COORD_DIM*nq + nq
          + (Long)C*nq + ns*(Long)C*order + (Long)C*order + (Long)C*order*ns + nnode;
        ScratchBuf<Real> sb(sz);
        Long off = 0;
        auto take = [&](const Long n) {
          Iterator<Real> r = sb.begin() + off;
          off += n;
          return r;
        };

        Matrix<Real> FS(detail_quadelem::COORD_DIM*order, order, take(detail_quadelem::COORD_DIM*nnode), false);
        Matrix<Real> Gm(detail_quadelem::COORD_DIM*order, 2*ns, take(2*detail_quadelem::COORD_DIM*(Long)order*ns), false);
        Matrix<Real> As(NA, order, take((Long)NA*order), false), Tmp(NA, 2*order, take(2*(Long)NA*order), false);
        Matrix<Real> HG(sblk*NR, order, take(sblk*NR*(Long)order), false);
        Matrix<Real> XdX(sblk*NR, nt, take(sblk*NR*nt), false);
        Vector<Real> Xs(detail_quadelem::COORD_DIM*nq, take(detail_quadelem::COORD_DIM*nq), false), Xn(detail_quadelem::COORD_DIM*nq, take(detail_quadelem::COORD_DIM*nq), false);
        Vector<Real> wq(nq, take(nq), false);
        Matrix<Real> KW(ns*C, nt, take((Long)C*nq), false);
        Matrix<Real> Zall(ns*C, order, take(ns*(Long)C*order), false);
        Matrix<Real> Yi(C, order, take((Long)C*order), false), Yall(C*order, ns, take((Long)C*order*ns), false);
        Matrix<Real> Pc(order, order, take(nnode), false);

        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++)
          for (Integer i = 0; i < order; i++) for (Integer j = 0; j < order; j++)
            FS[k*order + (T.swap_ab ? j : i)][T.swap_ab ? i : j] = cs[k*nnode + (Long)i*order + j];

        Matrix<Real>::GEMM(Gm, FS, T.WbC);

        for (Long i0 = 0; i0 < ns; i0 += sblk) {
          const Long nb = std::min<Long>(sblk, ns-i0);
          for (Long b = 0; b < nb; b++) {
            const Long i = i0 + b;
            for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
              As[k][m] = Gm[k*order+m][i];
              As[detail_quadelem::COORD_DIM+k][m] = Gm[k*order+m][ns+i];
            }
            Matrix<Real>::GEMM(Tmp, As, T.MiC[i]);
            for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
              HG[b*NR + k][m]               = Tmp[k][m];
              HG[b*NR + detail_quadelem::COORD_DIM + k][m]   = Tmp[k][order+m];
              HG[b*NR + 2*detail_quadelem::COORD_DIM + k][m] = Tmp[detail_quadelem::COORD_DIM+k][m];
            }
          }
          {
            const Matrix<Real> HGb(nb*NR, order, (Iterator<Real>)HG.begin(), false);
            Matrix<Real> XdXb(nb*NR, nt, (Iterator<Real>)XdX.begin(), false);
            Matrix<Real>::GEMM(XdXb, HGb, Tt);
          }

          for (Long b = 0; b < nb; b++) {
            const Long i = i0 + b;
            const Real jw = tbl.sn[i]*T.J0*tbl.sw[i];
            for (Long j = 0; j < nt; j++) {
              const Long q = i*nt + j;
              const Real a0 = XdX[b*NR+detail_quadelem::COORD_DIM+0][j], a1 = XdX[b*NR+detail_quadelem::COORD_DIM+1][j], a2 = XdX[b*NR+detail_quadelem::COORD_DIM+2][j];
              const Real b0 = XdX[b*NR+2*detail_quadelem::COORD_DIM+0][j], b1 = XdX[b*NR+2*detail_quadelem::COORD_DIM+1][j], b2 = XdX[b*NR+2*detail_quadelem::COORD_DIM+2][j];
              const Real n0 = T.nsign*(a1*b2-a2*b1), n1 = T.nsign*(a2*b0-a0*b2), n2 = T.nsign*(a0*b1-a1*b0);
              const Real ar = sqrt<Real>(n0*n0+n1*n1+n2*n2), ia = (ar > 0 ? (Real)1/ar : (Real)0);
              for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) Xs[k*nq+q] = XdX[b*NR+k][j];
              Xn[0*nq+q] = n0*ia;
              Xn[1*nq+q] = n1*ia;
              Xn[2*nq+q] = n2*ia;
              wq[q] = ar*jw*tw[j];
            }
          }
        }

        {
          static constexpr bool HAS_N = detail_quadelem::UKerNeedsN<Kernel, Vec<Real,1>>::value;
          using WVec = Vec<Real, DefaultVecLen<Real>()>;
          const Long jmain = (nt/WVec::Size())*WVec::Size();
          const ConstIterator<Real> xt = Xt0_v.begin(), xs = Xs.begin(), xn = Xn.begin(), w = wq.begin();
          const ConstIterator<Real> ntg = (trg_dot_prod ? normal_trg.begin() : ConstIterator<Real>(NullIterator<Real>()));
          if (trg_dot_prod) {
            detail_quadelem::KerFoldSoA<Real,Kernel,WVec,        HAS_N,true >(KW.begin(), xt, xs, xn, w, ns*nt, nt,     0, jmain, (Real)1, false, ntg);
            detail_quadelem::KerFoldSoA<Real,Kernel,Vec<Real,1>, HAS_N,true >(KW.begin(), xt, xs, xn, w, ns*nt, nt, jmain,    nt, (Real)1, false, ntg);
          } else {
            detail_quadelem::KerFoldSoA<Real,Kernel,WVec,        HAS_N,false>(KW.begin(), xt, xs, xn, w, ns*nt, nt,     0, jmain, (Real)1, false, ntg);
            detail_quadelem::KerFoldSoA<Real,Kernel,Vec<Real,1>, HAS_N,false>(KW.begin(), xt, xs, xn, w, ns*nt, nt, jmain,    nt, (Real)1, false, ntg);
          }
        }

        Matrix<Real>::GEMM(Zall, KW, TtT);
        for (Long i = 0; i < ns; i++) {
          const Matrix<Real> Zi(C, order, (Iterator<Real>)Zall.begin() + i*(Long)C*order, false);
          Matrix<Real>::GEMM(Yi, Zi, T.MiT[i]);
          for (Integer c = 0; c < C; c++) for (Integer m = 0; m < order; m++) Yall[c*order+m][i] = Yi[c][m];
        }
        for (Integer c = 0; c < C; c++) {
          const Matrix<Real> Yc(order, ns, (Iterator<Real>)Yall.begin() + (Long)c*order*ns, false);
          Matrix<Real>::GEMM(Pc, Yc, T.WbT);
          for (Integer m = 0; m < order; m++) for (Integer n = 0; n < order; n++)
            M_acc[T.swap_ab ? (Long)n*order+m : (Long)m*order+n][c] += Pc[m][n];
        }
      }
    }

    template <Integer order, class Real, class Kernel> void SelfInteracDuffy(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      DuffyTable<order,Real>();
      detail_dyadic_near::NearGradeTable<order,Real>(detail_dyadic_near::DigitsQuadOrder<Real>(digits));
      detail_dyadic_near::DigitsBEllipse<Real>(digits);
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      detail_quadelem::SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, false,
          [&](Matrix<Real>& M, const Long elem_idx, const Long t, const Integer ti, const Integer tj,
            const Vector<Real>& Xtrg, const Vector<Real>& ntrg, const Vector<Real>&, const Vector<Real>&,
            const Vector<Real>&, const Vector<Real>&, const Integer KDIM1_out) {
          thread_local Matrix<Real> M_acc;
          SelfInteracBlockDuffy<order,Real>(M_acc, qel, elem_idx, ti, tj, Xtrg, ntrg, ker, digits);
          detail_quadelem::ScatterSelfBlock<order,Real>(M, M_acc, t, KDIM0, KDIM1_out, false);
          });
    }

  }

  namespace detail_tensorprod {

    template <class Real> void QuadParams(const Real tol, Real& b_ellipse, Integer& QuadOrder) {
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const double rho = 2.5;
      const Real rho_ = (Real)rho;
      b_ellipse = (rho_ + 1/rho_) / 4;
      QuadOrder = std::max<Integer>(1, (Integer)std::ceil(-std::log(((15.0*(rho*rho-1))/64.0)*(double)tol_)/std::log(rho)*0.5 + 1));
    }

    template <class Real> inline Integer DigitsQuadOrder(const Integer digits) {
      static const std::array<Integer,detail_quadelem::MaxDigits<Real>> q = []() {
        std::array<Integer,detail_quadelem::MaxDigits<Real>> t{};
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) {
          Real b;
          Integer qq;
          QuadParams<Real>(pow<Real,Long>((Real)0.1, (Long)d), b, qq);
          t[d] = qq;
        }
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      return q[digits];
    }

    template <class Real> inline Real DigitsBEllipse(const Integer digits) {
      static const std::array<Real,detail_quadelem::MaxDigits<Real>> b = []() {
        std::array<Real,detail_quadelem::MaxDigits<Real>> t{};
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) {
          Real bb;
          Integer qq;
          QuadParams<Real>(pow<Real,Long>((Real)0.1, (Long)d), bb, qq);
          t[d] = bb;
        }
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      return b[digits];
    }

    template <class Real> const std::pair<Vector<Real>, Vector<Real>>& DigitsGLRule(const Integer digits) {
      static const std::array<std::pair<Vector<Real>, Vector<Real>>,detail_quadelem::MaxDigits<Real>> gl = []() {
        std::array<std::pair<Vector<Real>, Vector<Real>>,detail_quadelem::MaxDigits<Real>> t;
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) LegQuadRule<Real>::ComputeNdsWts(&t[d].first, &t[d].second, DigitsQuadOrder<Real>(d));
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      return gl[digits];
    }

    template <class Real> struct NodeRuleData {
      Vector<Real> param, w;
      Matrix<Real> M, dM, MT, dMT;
    };

    template <Integer order, class Real> void LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& delta, const Integer ti) {
      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      const Long N = delta.Dim();
      StaticArray<Real,order> inv_den, off;
      for (Integer i = 0; i < order; i++) {
        Real d = 1;
        for (Integer j = 0; j < order; j++) if (j != i) d *= (nds[i]-nds[j]);
        inv_den[i] = 1/d;
      }
      for (Integer j = 0; j < order; j++) off[j] = nds[ti]-nds[j];
      M.ReInit(order, N);
      StaticArray<Real,order> f;
      for (Long a = 0; a < N; a++) {
        for (Integer j = 0; j < order; j++) f[j] = delta[a] + off[j];
        for (Integer i = 0; i < order; i++) {
          Real p = inv_den[i];
          for (Integer j = 0; j < order; j++) if (j != i) p *= f[j];
          M[i][a] = p;
        }
      }
      dM.ReInit(order, N);
      Matrix<Real>::GEMM(dM, detail_quadelem::DiffMat<Real>(order), M);
      MT = M.Transpose();
      dMT = dM.Transpose();
    }

    template <class Real> void BuildCenteredGraded1D(Vector<Real>& delta, Vector<Real>& w, const Real u0, const Integer levels, const Vector<Real>& qnds, const Vector<Real>& qwts) {
      const Integer q = qnds.Dim();
      std::vector<Real> d_, w_;
      auto side = [&](const Real Len, const Real sgn) {
        if (!(Len > 0)) return;
        Real a = 0;
        for (Integer k = levels; k >= 0; k--) {
          const Real b = Len * pow<Real>((Real)0.5, (Integer)k);
          const Real len = b - a;
          if (len > 0) for (Integer i = 0; i < q; i++) {
            d_.push_back(sgn*(a + len*qnds[i]));
            w_.push_back(len*qwts[i]);
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

    template <class Real> void BuildCenteredLogSingular1D(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
      const Integer ord = 16;
      std::vector<Real> px, pw;
      auto add_alpert = [&](const Real a, const Real b, const Integer corra, const Integer corrb) {
        const auto L = (corra == 2 ? QuadLogExtraPtNodes<Real>(ord) : QuadSmoothExtraPtNodes<Real>(ord));
        const auto R = (corrb == 2 ? QuadLogExtraPtNodes<Real>(ord) : QuadSmoothExtraPtNodes<Real>(ord));
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
      LegQuadRule<Real>::ComputeNdsWts(&gnds, &gwts, QuadOrder);
      auto add_gl = [&](const Real a, const Real b) {
        const Real len = b - a;
        for (Integer i = 0; i < QuadOrder; i++) {
          px.push_back(a + len*gnds[i]);
          pw.push_back(len*gwts[i]);
        }
      };
      const Real Ll = v0, Lr = (Real)1 - v0;
      {
        Real prev = -Ll;
        for (Integer i = 1; i <= Lvl; i++) {
          const Real bnd = -Ll*pow<Real,Long>((Real)0.5, (Long)i);
          add_gl(prev, bnd);
          prev = bnd;
        }
        add_alpert(prev, (Real)0, 1, 2);
      }
      {
        Real prev = Lr;
        for (Integer i = 1; i <= Lvl; i++) {
          const Real bnd = Lr*pow<Real,Long>((Real)0.5, (Long)i);
          add_gl(bnd, prev);
          prev = bnd;
        }
        add_alpert((Real)0, prev, 2, 1);
      }
      const Long N = (Long)px.size();
      delta.ReInit(N);
      w.ReInit(N);
      for (Long i = 0; i < N; i++) {
        delta[i] = px[i];
        w[i] = pw[i];
      }
    }

    template <Integer order, class Real> const NodeRuleData<Real>& CenteredURule(const Integer ti, const Integer digits) {
      static std::atomic<Vector<NodeRuleData<Real>>*> slot[detail_quadelem::MaxDigits<Real>];
      static std::mutex mtx;
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      Vector<NodeRuleData<Real>>* p = slot[digits].load(std::memory_order_acquire);
      if (!p) {
        std::lock_guard<std::mutex> lk(mtx);
        p = slot[digits].load(std::memory_order_relaxed);
        if (!p) {
          const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
          const Integer QuadOrder = DigitsQuadOrder<Real>(digits);
          const Integer Lvl = std::min<Integer>(detail_quadelem::MaxRefineLvl<Real>, 2*digits + 6);
          Vector<Real> qnds, qwts;
          LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);
          auto* d = new Vector<NodeRuleData<Real>>(order);
          for (Integer i = 0; i < order; i++) {
            BuildCenteredGraded1D((*d)[i].param, (*d)[i].w, nds[i], Lvl, qnds, qwts);
            LagrangeAtOffset<order,Real>((*d)[i].M, (*d)[i].dM, (*d)[i].MT, (*d)[i].dMT, (*d)[i].param, i);
          }
          p = d;
          slot[digits].store(p, std::memory_order_release);
        }
      }
      return (*p)[ti];
    }

    template <Integer order, class Real> const NodeRuleData<Real>& CenteredVRule(const Integer tj, const Integer digits) {
      static std::atomic<Vector<NodeRuleData<Real>>*> slot[detail_quadelem::MaxDigits<Real>];
      static std::mutex mtx;
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      Vector<NodeRuleData<Real>>* p = slot[digits].load(std::memory_order_acquire);
      if (!p) {
        std::lock_guard<std::mutex> lk(mtx);
        p = slot[digits].load(std::memory_order_relaxed);
        if (!p) {
          const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
          const Integer Lvl = std::min<Integer>(12, std::max<Integer>(1, digits - 5));
          const Integer QuadOrder = DigitsQuadOrder<Real>(digits);
          auto* d = new Vector<NodeRuleData<Real>>(order);
          for (Integer j = 0; j < order; j++) {
            BuildCenteredLogSingular1D((*d)[j].param, (*d)[j].w, nds[j], Lvl, QuadOrder);
            LagrangeAtOffset<order,Real>((*d)[j].M, (*d)[j].dM, (*d)[j].MT, (*d)[j].dMT, (*d)[j].param, j);
          }
          p = d;
          slot[digits].store(p, std::memory_order_release);
        }
      }
      return (*p)[tj];
    }

    template <class Real> void ExpandSegments(Vector<Real>& param, Vector<Real>& w, const Vector<Real>& seg, const Vector<Real>& qnds, const Vector<Real>& qwts) {
      const Integer QuadOrder = qnds.Dim();
      const Long nseg = seg.Dim()/2;
      const Long N = nseg * QuadOrder;
      if (param.Dim() != N) param.ReInit(N);
      if (w.Dim() != N) w.ReInit(N);
      Long idx = 0;
      for (Long si = 0; si < nseg; si++) {
        const Real a0 = seg[si*2+0], a1 = seg[si*2+1];
        const Real len = a1 - a0;
        for (Integer a = 0; a < QuadOrder; a++) {
          param[idx] = a0 + len*qnds[a];
          w[idx] = qwts[a]*len;
          idx++;
        }
      }
    }

    template <class Real> void BuildFootGraded1DSegments(Vector<Real>& seg, Vector<Long>& seg_depth, const Real center, const Real b_ellipse, const Real w_min) {
      constexpr Long MaxLeaves = 4096;
      const Real r = std::min<Real>((Real)0.9, b_ellipse/(1 + b_ellipse) * (Real)1.05);
      const Real wmin = std::max<Real>(w_min, (Real)1e-300);

      std::vector<Real> a;
      std::vector<Long> d;
      for (Integer side = 0; side < 2; side++) {
        const Real sgn = (side ? (Real)1 : (Real)-1);
        const Real span = (side ? 1 - center : center);
        if (!(span > 0)) continue;

        Real w = span;
        Long lvl = 0;
        while (w > wmin) {
          const Real w_next = w*r;
          const Real e0 = center + sgn*w, e1 = center + sgn*w_next;
          const Real lo = std::min<Real>(e0, e1), hi = std::max<Real>(e0, e1);
          if (hi - lo > 0) {
            a.push_back(lo);
            a.push_back(hi);
            d.push_back(lvl);
          }
          w = w_next;
          lvl++;
          SCTL_ASSERT((Long)d.size() <= MaxLeaves);
        }
        const Real e0 = center + sgn*w;
        const Real lo = std::min<Real>(e0, center), hi = std::max<Real>(e0, center);
        if (hi - lo > 0) {
          a.push_back(lo);
          a.push_back(hi);
          d.push_back(lvl);
        }
        SCTL_ASSERT((Long)d.size() <= MaxLeaves);
      }

      const Long nseg = (Long)d.size();
      seg.ReInit(nseg*2);
      seg_depth.ReInit(nseg);
      for (Long i = 0; i < nseg; i++) {
        seg[i*2+0] = a[i*2+0];
        seg[i*2+1] = a[i*2+1];
        seg_depth[i] = d[i];
      }
    }

    template <Integer order, class Real, class Kernel> void IntegratePanel(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker, const Matrix<Real>* Mv_pre = nullptr, const Matrix<Real>* dMv_pre = nullptr, const Matrix<Real>* Mu_pre = nullptr, const Matrix<Real>* dMu_pre = nullptr, const Matrix<Real>* MvT_pre = nullptr, const Matrix<Real>* MuT_pre = nullptr, const Matrix<Real>* dMuT_pre = nullptr, const Vector<Real>* src_nodal = nullptr, const Matrix<Real>* MuD_pre = nullptr, const Real nrm_sign = 1, Vector<Real>* acc_cm = nullptr) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / detail_quadelem::COORD_DIM : KDIM1full;

      const Long Nu = (Mu_pre ? Mu_pre->Dim(1) : u_param.Dim());
      const Long Nv = (Mv_pre ? Mv_pre->Dim(1) : v_param.Dim());
      const Long nq = Nu * Nv;
      if (!nq) return;
      const Integer C = KDIM0 * KDIM1_out;

      const Vector<Real>& pnds = QuadElemList<Real>::ParamNodes(order);
      const Matrix<Real>& D = detail_quadelem::DiffMat<Real>(order);

      Matrix<Real> Mu_local, dMu_local, MuT_local, dMuT_local;
      Matrix<Real> Mv_local, dMv_local, MvT_local;
      {
        if (!Mu_pre || !MuT_pre || !dMuT_pre) {
          Mu_local.ReInit(order, Nu);
          {
            Vector<Real> v(order*Nu, Mu_local.begin(), false);
            LagrangeInterp<Real>::Interpolate(v, pnds, u_param);
          }
          dMu_local.ReInit(order, Nu);
          Matrix<Real>::GEMM(dMu_local, D, Mu_local);
          MuT_local = Mu_local.Transpose();
          dMuT_local = dMu_local.Transpose();
        }
        if (!Mv_pre) {
          Mv_local.ReInit(order, Nv);
          {
            Vector<Real> v(order*Nv, Mv_local.begin(), false);
            LagrangeInterp<Real>::Interpolate(v, pnds, v_param);
          }
          dMv_local.ReInit(order, Nv);
          Matrix<Real>::GEMM(dMv_local, D, Mv_local);
          MvT_local = Mv_local.Transpose();
        }
      }
      const Matrix<Real>& Mu  = (Mu_pre  ? *Mu_pre  : Mu_local);
      const Matrix<Real>& MuT  = (MuT_pre  ? *MuT_pre  : MuT_local);
      const Matrix<Real>& dMuT = (dMuT_pre ? *dMuT_pre : dMuT_local);
      const Matrix<Real>& Mv  = (Mv_pre  ? *Mv_pre  : Mv_local);
      const Matrix<Real>& dMv = (dMv_pre ? *dMv_pre : dMv_local);
      const Matrix<Real>& MvT  = (MvT_pre  ? *MvT_pre  : MvT_local);

      const Long base = elem_idx * nnode * detail_quadelem::COORD_DIM;
      thread_local Vector<Real> coord_shift;
      thread_local const QuadElemList<Real>* cs_qel = nullptr;
      thread_local Long cs_elem = -1;
      thread_local StaticArray<Real,detail_quadelem::COORD_DIM> cs_trg{0,0,0};
      if (coord_shift.Dim() != detail_quadelem::COORD_DIM*nnode) {
        coord_shift.ReInit(detail_quadelem::COORD_DIM*nnode);
        cs_qel = nullptr;
      }
      if (!src_nodal && (cs_qel != &qel || cs_elem != elem_idx || cs_trg[0] != Xtrg[0] || cs_trg[1] != Xtrg[1] || cs_trg[2] != Xtrg[2])) {
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          const Real ok = Xtrg[k];
          for (Long p = 0; p < nnode; p++) coord_shift[k*nnode + p] = detail_quadelem::Access<Real>::Coord(qel)[base + k*nnode + p] - ok;
        }
        cs_qel = &qel;
        cs_elem = elem_idx;
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) cs_trg[k] = Xtrg[k];
      }
      const Vector<Real>& cs_ref = (src_nodal ? *src_nodal : coord_shift);
      thread_local Vector<Real> Cv, Cdv;
      if (Cv.Dim() != detail_quadelem::COORD_DIM*order*Nv) {
        Cv.ReInit(detail_quadelem::COORD_DIM*order*Nv);
        Cdv.ReInit(detail_quadelem::COORD_DIM*order*Nv);
      }
      {
        const Matrix<Real> cs_all(detail_quadelem::COORD_DIM*order, order, (Iterator<Real>)cs_ref.begin(), false);
        Matrix<Real> Cv_all (detail_quadelem::COORD_DIM*order, Nv, Cv.begin(),  false);
        Matrix<Real> Cdv_all(detail_quadelem::COORD_DIM*order, Nv, Cdv.begin(), false);
        Matrix<Real>::GEMM(Cv_all,  cs_all, Mv);
        Matrix<Real>::GEMM(Cdv_all, cs_all, dMv);
      }
      if (Nu * Nv <= detail_quadelem::MaxUnblockedPts) {
        const Long ldc = detail_quadelem::COORD_DIM*Nv;
        thread_local Vector<Real> Cvc, Cdvc, XdU, dXdv_soa;
        if (Cvc.Dim() != (Long)order*ldc) {
          Cvc.ReInit((Long)order*ldc);
          Cdvc.ReInit((Long)order*ldc);
        }
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          for (Integer i = 0; i < order; i++) {
            const Long src = ((Long)k*order + i)*Nv, dst = (Long)i*ldc + k*Nv;
            for (Long b = 0; b < Nv; b++) {
              Cvc[dst+b] = Cv[src+b];
              Cdvc[dst+b] = Cdv[src+b];
            }
          }
        }
        if (XdU.Dim() != 2*(Long)Nu*ldc) {
          XdU.ReInit(2*(Long)Nu*ldc);
          dXdv_soa.ReInit((Long)Nu*ldc);
        }
        {
          const Matrix<Real> Cvc_m(order, ldc, Cvc.begin(), false), Cdvc_m(order, ldc, Cdvc.begin(), false);
          Matrix<Real> dV_m(Nu, ldc, dXdv_soa.begin(), false);
          if (MuD_pre) {
            Matrix<Real> XdU_m(2*Nu, ldc, XdU.begin(), false);
            Matrix<Real>::GEMM(XdU_m, *MuD_pre, Cvc_m);
          } else {
            Matrix<Real> X_m(Nu, ldc, XdU.begin(), false), dU_m(Nu, ldc, XdU.begin() + (Long)Nu*ldc, false);
            Matrix<Real>::GEMM(X_m,  MuT,  Cvc_m);
            Matrix<Real>::GEMM(dU_m, dMuT, Cvc_m);
          }
          Matrix<Real>::GEMM(dV_m, MuT, Cdvc_m);
        }

        StaticArray<Real,detail_quadelem::COORD_DIM> Xt0_{0, 0, 0};
        const Vector<Real> Xt0_v_(detail_quadelem::COORD_DIM, Xt0_, false);
        thread_local Vector<Real> Xsrc, Xnsrc, wq;
        if (Xsrc.Dim() != nq*detail_quadelem::COORD_DIM) {
          Xsrc.ReInit(nq*detail_quadelem::COORD_DIM);
          Xnsrc.ReInit(nq*detail_quadelem::COORD_DIM);
          wq.ReInit(nq);
        }
        for (Long a = 0; a < Nu; a++) {
          for (Long b = 0; b < Nv; b++) {
            const Long q = a*Nv + b;
            const Long r = (Long)a*ldc + b, ru = ((Long)Nu + a)*ldc + b;
            const Real du0 = XdU[ru+0*Nv], du1 = XdU[ru+1*Nv], du2 = XdU[ru+2*Nv];
            const Real dv0 = dXdv_soa[r+0*Nv], dv1 = dXdv_soa[r+1*Nv], dv2 = dXdv_soa[r+2*Nv];
            const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
            const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
            const Real inv_area = (area > 0 ? nrm_sign/area : 0);
            Xsrc[q*detail_quadelem::COORD_DIM+0] = XdU[r+0*Nv];
            Xsrc[q*detail_quadelem::COORD_DIM+1] = XdU[r+1*Nv];
            Xsrc[q*detail_quadelem::COORD_DIM+2] = XdU[r+2*Nv];
            Xnsrc[q*detail_quadelem::COORD_DIM+0] = n0*inv_area;
            Xnsrc[q*detail_quadelem::COORD_DIM+1] = n1*inv_area;
            Xnsrc[q*detail_quadelem::COORD_DIM+2] = n2*inv_area;
            wq[q] = area*wu[a]*wv[b];
          }
        }

        thread_local Matrix<Real> Mker;
        ker.template KernelMatrix<Real,false>(Mker, Xt0_v_, Xsrc, Xnsrc);

        thread_local Vector<Real> KWc;
        if (KWc.Dim() != C*nq) KWc.ReInit(C*nq);
        for (Long q = 0; q < nq; q++) {
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
              Real val;
              if (trg_dot_prod) {
                val = 0;
                for (Integer l = 0; l < detail_quadelem::COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*detail_quadelem::COORD_DIM+l] * normal_trg[l];
              } else {
                val = Mker[q*KDIM0+k0][k1];
              }
              KWc[(Long)(k0*KDIM1_out+k1)*nq + q] = val*wq[q];
            }
          }
        }

        thread_local Vector<Real> Yv, proj;
        if (Yv.Dim() != (Long)C*Nu*order) Yv.ReInit((Long)C*Nu*order);
        {
          const Matrix<Real> KW_all((Long)C*Nu, Nv, KWc.begin(), false);
          Matrix<Real> Y_all((Long)C*Nu, order, Yv.begin(), false);
          Matrix<Real>::GEMM(Y_all, KW_all, MvT);
        }
        if (acc_cm) {
          for (Integer c = 0; c < C; c++) {
            const Matrix<Real> Y_c(Nu, order, Yv.begin() + (Long)c*Nu*order, false);
            Matrix<Real> A_c(order, order, acc_cm->begin() + (Long)c*nnode, false);
            Matrix<Real>::GEMM(A_c, Mu, Y_c, (Real)1);
          }
        } else {
          if (proj.Dim() != (Long)C*nnode) proj.ReInit((Long)C*nnode);
          for (Integer c = 0; c < C; c++) {
            const Matrix<Real> Y_c(Nu, order, Yv.begin() + (Long)c*Nu*order, false);
            Matrix<Real> P_c(order, order, proj.begin() + (Long)c*nnode, false);
            Matrix<Real>::GEMM(P_c, Mu, Y_c);
          }
          if (acc_cm) for (Long i = 0; i < (Long)C*nnode; i++) (*acc_cm)[i] += proj[i];
          else for (Long p = 0; p < nnode; p++) for (Integer c = 0; c < C; c++) M_acc[p][c] += proj[(Long)c*nnode + p];
        }
        return;
      }

      StaticArray<Real,detail_quadelem::COORD_DIM> Xt0{0, 0, 0};
      const Vector<Real> Xt0_v(detail_quadelem::COORD_DIM, Xt0, false);

      const Long UBLK = std::max<Long>(1, std::min<Long>(Nu, detail_quadelem::MaxUnblockedPts / std::max<Long>(1, Nv)));
      const Long nqmax = UBLK*Nv, cs = nqmax;

      thread_local Vector<Real> Xb, dXub, dXvb, Xsrcb, Xnsrcb, wqb, KWcb, Mkerb, Tfull, projb;
      if (Xb.Dim() != detail_quadelem::COORD_DIM*nqmax) {
        Xb.ReInit(detail_quadelem::COORD_DIM*nqmax);
        dXub.ReInit(detail_quadelem::COORD_DIM*nqmax);
        dXvb.ReInit(detail_quadelem::COORD_DIM*nqmax);
        Xsrcb.ReInit(detail_quadelem::COORD_DIM*nqmax);
        Xnsrcb.ReInit(detail_quadelem::COORD_DIM*nqmax);
        wqb.ReInit(nqmax);
      }
      if (KWcb.Dim() != C*nqmax) {
        KWcb.ReInit(C*nqmax);
        Mkerb.ReInit(nqmax*KDIM0*KDIM1full);
      }
      if (Tfull.Dim() != C*Nu*(Long)order) Tfull.ReInit(C*Nu*(Long)order);
      if (projb.Dim() != C*nnode) projb.ReInit(C*nnode);

      for (Long a0 = 0; a0 < Nu; a0 += UBLK) {
        const Long nu = std::min<Long>(UBLK, Nu - a0), nqb = nu*Nv;
        const Matrix<Real> MuT_b (nu, order, (Iterator<Real>)MuT.begin()  + a0*(Long)order, false);
        const Matrix<Real> dMuT_b(nu, order, (Iterator<Real>)dMuT.begin() + a0*(Long)order, false);

        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          const Matrix<Real> Cv_k (order, Nv, Cv.begin()  + k*(Long)order*Nv, false);
          const Matrix<Real> Cdv_k(order, Nv, Cdv.begin() + k*(Long)order*Nv, false);
          Matrix<Real> Xk(nu, Nv, Xb.begin() + k*cs, false), dUk(nu, Nv, dXub.begin() + k*cs, false), dVk(nu, Nv, dXvb.begin() + k*cs, false);
          Matrix<Real>::GEMM(Xk,  MuT_b,  Cv_k);
          Matrix<Real>::GEMM(dUk, dMuT_b, Cv_k);
          Matrix<Real>::GEMM(dVk, MuT_b,  Cdv_k);
        }

        for (Long a = 0; a < nu; a++) {
          for (Long b = 0; b < Nv; b++) {
            const Long q = a*Nv + b;
            const Real du0 = dXub[0*cs+q], du1 = dXub[1*cs+q], du2 = dXub[2*cs+q];
            const Real dv0 = dXvb[0*cs+q], dv1 = dXvb[1*cs+q], dv2 = dXvb[2*cs+q];
            const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
            const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
            const Real inv_area = (area > 0 ? 1/area : 0);
            Xsrcb[q*detail_quadelem::COORD_DIM+0] = Xb[0*cs+q];
            Xsrcb[q*detail_quadelem::COORD_DIM+1] = Xb[1*cs+q];
            Xsrcb[q*detail_quadelem::COORD_DIM+2] = Xb[2*cs+q];
            Xnsrcb[q*detail_quadelem::COORD_DIM+0] = n0*inv_area;
            Xnsrcb[q*detail_quadelem::COORD_DIM+1] = n1*inv_area;
            Xnsrcb[q*detail_quadelem::COORD_DIM+2] = n2*inv_area;
            wqb[q] = area*wu[a0+a]*wv[b];
          }
        }

        Matrix<Real> Mker(nqb*KDIM0, KDIM1full, Mkerb.begin(), false);
        const Vector<Real> Xsrc_v(nqb*detail_quadelem::COORD_DIM, Xsrcb.begin(), false), Xnsrc_v(nqb*detail_quadelem::COORD_DIM, Xnsrcb.begin(), false);
        ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xsrc_v, Xnsrc_v);

        for (Long q = 0; q < nqb; q++) {
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
              Real val;
              if (trg_dot_prod) {
                val = 0;
                for (Integer l = 0; l < detail_quadelem::COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*detail_quadelem::COORD_DIM+l] * normal_trg[l];
              } else {
                val = Mker[q*KDIM0+k0][k1];
              }
              KWcb[(Long)(k0*KDIM1_out+k1)*cs + q] = val*wqb[q];
            }
          }
        }

        for (Integer c = 0; c < C; c++) {
          const Matrix<Real> KW_c(nu, Nv, KWcb.begin() + (Long)c*cs, false);
          Matrix<Real> T_c(nu, order, Tfull.begin() + (Long)c*Nu*order + a0*(Long)order, false);
          Matrix<Real>::GEMM(T_c, KW_c, MvT);
        }
      }

      for (Integer c = 0; c < C; c++) {
        const Matrix<Real> T_c(Nu, order, Tfull.begin() + (Long)c*Nu*order, false);
        Matrix<Real> P_c(order, order, projb.begin() + (Long)c*nnode, false);
        Matrix<Real>::GEMM(P_c, Mu, T_c);
      }
      for (Long p = 0; p < nnode; p++)
        for (Integer c = 0; c < C; c++) M_acc[p][c] += projb[(Long)c*nnode + p];
    }

    template <class Real> Integer ClosestPointAndRefineLevel(Real& ustar, Real& vstar, Real& dist, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse, Real* h_param = nullptr) {
      const Integer max_depth = detail_quadelem::MaxRefineLvl<Real>;
      dist = detail_quadelem::GetClosestPoint(qel, ustar, vstar, elem_idx, Xtrg);

      Real Xc[detail_quadelem::COORD_DIM], dXdu[detail_quadelem::COORD_DIM], dXdv[detail_quadelem::COORD_DIM];
      detail_quadelem::EvalPoint<Real>(qel, Xc, dXdu, dXdv, ustar, vstar, elem_idx, nullptr);
      Real su2 = 0, sv2 = 0;
      for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
        su2 += dXdu[k]*dXdu[k];
        sv2 += dXdv[k]*dXdv[k];
      }
      const Real L_phys = std::max<Real>(sqrt<Real>(su2), sqrt<Real>(sv2));

      if (!(dist > 0) || isinf<Real>(dist) || isnan<Real>(dist) || !(L_phys > 0)) {
        if (h_param) *h_param = 0;
        return max_depth;
      }
      if (h_param) *h_param = dist/L_phys;
      const double lvl = std::ceil(std::log2((double)(b_ellipse*L_phys/dist)));
      return (Integer)std::min<double>((double)max_depth, std::max<double>(0.0, lvl));
    }

    template <class Real> Integer BuildNearTensorRule(Vector<Real>& u_param, Vector<Real>& wu, Vector<Real>& v_param, Vector<Real>& wv,
        Vector<Real>* useg, Vector<Long>* useg_depth, Vector<Real>* vseg, Vector<Long>* vseg_depth,
        const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg,
        const Real b_ellipse, const Vector<Real>& qnds, const Vector<Real>& qwts) {
      const Integer max_depth = detail_quadelem::MaxRefineLvl<Real>;
      Real ustar, vstar, dist, h_param;
      const Integer L = ClosestPointAndRefineLevel<Real>(ustar, vstar, dist, qel, elem_idx, Xtrg, b_ellipse, &h_param);

      Vector<Real> useg_local, vseg_local;
      Vector<Long> udep_local, vdep_local;
      Vector<Real>& us = (useg ? *useg : useg_local);
      Vector<Real>& vs = (vseg ? *vseg : vseg_local);
      Vector<Long>& ud = (useg_depth ? *useg_depth : udep_local);
      Vector<Long>& vd = (vseg_depth ? *vseg_depth : vdep_local);

      const Real w_floor = pow<Real>((Real)0.5, max_depth);
      const Real w_min = std::max<Real>(h_param/b_ellipse, w_floor);
      BuildFootGraded1DSegments<Real>(us, ud, ustar, b_ellipse, w_min);
      BuildFootGraded1DSegments<Real>(vs, vd, vstar, b_ellipse, w_min);
      ExpandSegments<Real>(u_param, wu, us, qnds, qwts);
      ExpandSegments<Real>(v_param, wv, vs, qnds, qwts);
      return L;
    }

    template <Integer order, class Real, class Kernel> void SelfInteracBlockTensorProduct(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / detail_quadelem::COORD_DIM : KDIM1full;

      const NodeRuleData<Real>& ru = CenteredURule<order,Real>(ti, digits);
      const NodeRuleData<Real>& rv = CenteredVRule<order,Real>(tj, digits);

      M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();
      IntegratePanel<order,Real>(M_acc, qel, elem_idx, Xtrg, normal_trg, ru.param, ru.w, rv.param, rv.w, ker, &rv.M, &rv.dM, &ru.M, &ru.dM, &rv.MT, &ru.MT, &ru.dMT);
    }

    template <Integer order, class Real, class Kernel> void NearInteracBlockTensorProduct(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {

      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1full = Kernel::TrgDim();
      SCTL_ASSERT(qel.Order() == order);
      const Long nnode = (Long)order * order;
      const bool trg_dot_prod = (normal_trg.Dim() > 0);
      const Integer KDIM1_out = trg_dot_prod ? KDIM1full / detail_quadelem::COORD_DIM : KDIM1full;

      const Real b_ellipse = DigitsBEllipse<Real>(digits);
      const std::pair<Vector<Real>, Vector<Real>>& gl = DigitsGLRule<Real>(digits);

      thread_local Vector<Real> u_param, wu, v_param, wv;
      BuildNearTensorRule<Real>(u_param, wu, v_param, wv, nullptr, nullptr, nullptr, nullptr,
          qel, elem_idx, Xtrg, b_ellipse, gl.first, gl.second);
      const Long Nu = u_param.Dim(), Nv = v_param.Dim();

      if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();
      if (!Nu || !Nv) return;

      thread_local NodeRuleData<Real> ru, rv;
      const auto build_interp = [](NodeRuleData<Real>& r, const Vector<Real>& param) {
        const Long N = param.Dim();
        r.M.ReInit(order, N);
        {
          Vector<Real> v(order*N, r.M.begin(), false);
          LagrangeInterp<Real>::Interpolate(v, QuadElemList<Real>::ParamNodes(order), param);
        }
        r.dM.ReInit(order, N);
        Matrix<Real>::GEMM(r.dM, detail_quadelem::DiffMat<Real>(order), r.M);
        r.MT = r.M.Transpose();
        r.dMT = r.dM.Transpose();
      };
      build_interp(ru, u_param);
      build_interp(rv, v_param);

      IntegratePanel<order,Real>(M_acc, qel, elem_idx, Xtrg, normal_trg, u_param, wu, v_param, wv, ker,
          &rv.M, &rv.dM, &ru.M, &ru.dM, &rv.MT, &ru.MT, &ru.dMT);
    }

    template <Integer order, class Real, class Kernel> void NearInteracTensorProduct(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      detail_quadelem::NearInteracTargets<order,Real,Kernel>(M, Xt, normal_trg, qel,
          [&](Matrix<Real>& M_acc, const Vector<Real>& Xtrg, const Vector<Real>& ntrg) {
          NearInteracBlockTensorProduct<order,Real>(M_acc, qel, elem_idx, Xtrg, ntrg, ker, digits);
          });
    }

    template <Integer order, class Real, class Kernel> void SelfInteracTensorProduct(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      CenteredURule<order,Real>(0, digits);
      CenteredVRule<order,Real>(0, digits);
      DigitsGLRule<Real>(digits);
      DigitsBEllipse<Real>(digits);
      DigitsQuadOrder<Real>(digits);
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      detail_quadelem::SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, false,
          [&](Matrix<Real>& M, const Long elem_idx, const Long t, const Integer ti, const Integer tj,
            const Vector<Real>& Xtrg, const Vector<Real>& ntrg, const Vector<Real>&, const Vector<Real>&,
            const Vector<Real>&, const Vector<Real>&, const Integer KDIM1_out) {
          thread_local Matrix<Real> M_acc;
          SelfInteracBlockTensorProduct<order,Real>(M_acc, qel, elem_idx, ti, tj, Xtrg, ntrg, ker, digits);
          detail_quadelem::ScatterSelfBlock<order,Real>(M, M_acc, t, KDIM0, KDIM1_out, false);
          });
    }

  }

  namespace detail_hedgehog {

    template <class Real> inline const Vector<Real>& HedgehogProxyOffsets() {
      static const Vector<Real> s = []() {
        Vector<Real> v;
        for (Integer j = 0; j < 5; j++) v.PushBack(pow<Real>((Real)4, (Real)j/(Real)4));
        return v;
      }();
      return s;
    }

    template <class Real> inline Real HedgehogWeights(Vector<Real>& w) {
      static const std::pair<Vector<Real>,Real> tab = []() {
        using W = detail_quadelem::PrecompReal;
        const Vector<Real>& s = HedgehogProxyOffsets<Real>();
        const Long p = s.Dim();
        Vector<Real> wj(p);
        Real A = 0;
        for (Long j = 0; j < p; j++) {
          W v = 1;
          for (Long k = 0; k < p; k++) if (k != j) v *= (0 - (W)s[k])/((W)s[j] - (W)s[k]);
          wj[j] = (Real)v;
          A += fabs<Real>(wj[j]);
        }
        return std::make_pair(wj, A);
      }();
      if (w.Dim() != tab.first.Dim()) w.ReInit(tab.first.Dim());
      w = tab.first;
      return tab.second;
    }

    template <class Real> inline Real HedgehogRminCoeff(const Integer digits, const Integer sing_order) {
      static const std::array<Real,detail_quadelem::MaxDigits<Real>> c = []() {
        std::array<Real,detail_quadelem::MaxDigits<Real>> t{};
        for (Integer d = 0; d < detail_quadelem::MaxDigits<Real>; d++) t[d] = (Real)0.1 * pow<Real>(pow<Real,Long>((Real)0.1, (Long)d), (Real)1/(Real)6);
        return t;
      }();
      SCTL_ASSERT(digits >= 0 && digits < detail_quadelem::MaxDigits<Real>);
      return (sing_order <= 1 ? c[digits] : std::min<Real>(c[digits], (Real)3e-3));
    }

    template <Integer order, class Real, class Kernel> void SelfInteracHedgehog(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer sing_order = detail_quadelem::KernelSingularOrder<Kernel>::value;

      Vector<Real> hh_w;
      HedgehogWeights<Real>(hh_w);
      const Real hh_c = HedgehogRminCoeff<Real>(digits, sing_order);
      const Integer near_digits = std::min<Integer>(detail_quadelem::MaxDigits<Real>-1, digits + (sing_order <= 1 ? 2 : 6));
      detail_dyadic_near::NearGradeTable<order,Real>(detail_dyadic_near::DigitsQuadOrder<Real>(near_digits));
      detail_dyadic_near::DigitsBEllipse<Real>(near_digits);
      detail_dyadic_near::NearGradeTable<order,Real>(detail_dyadic_near::DigitsQuadOrder<Real>(digits));
      detail_dyadic_near::DigitsBEllipse<Real>(digits);

      const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
      detail_quadelem::SelfInteracElems<order,Real,Kernel>(M_lst, trg_dot_prod, qel, true,
          [&](Matrix<Real>& M, const Long elem_idx, const Long t, const Integer ti, const Integer tj,
            const Vector<Real>&, const Vector<Real>& ntrg, const Vector<Real>& Xnodes, const Vector<Real>& Xnnodes,
            const Vector<Real>& dXu, const Vector<Real>& dXv, const Integer KDIM1_out) {
          thread_local Vector<Real> hh_Xt1, hh_off;
          thread_local Matrix<Real> M_hh;
          const Vector<Real>& soff = HedgehogProxyOffsets<Real>();
          if (hh_Xt1.Dim() != detail_quadelem::COORD_DIM) {
          hh_Xt1.ReInit(detail_quadelem::COORD_DIM);
          hh_off.ReInit(soff.Dim()*detail_quadelem::COORD_DIM);
          }
          Real su2 = 0, sv2 = 0;
          for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          su2 += dXu[t*detail_quadelem::COORD_DIM+k]*dXu[t*detail_quadelem::COORD_DIM+k];
          sv2 += dXv[t*detail_quadelem::COORD_DIM+k]*dXv[t*detail_quadelem::COORD_DIM+k];
          }
          const Real du = std::min<Real>(nds[ti], 1-nds[ti]), dv = std::min<Real>(nds[tj], 1-nds[tj]);
          const Real rmin = hh_c * std::min<Real>(du*sqrt<Real>(su2), dv*sqrt<Real>(sv2));
          for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) hh_Xt1[k] = Xnodes[t*detail_quadelem::COORD_DIM+k] + rmin*Xnnodes[t*detail_quadelem::COORD_DIM+k];
          for (Long j = 0; j < soff.Dim(); j++) {
          const Real rj = rmin*soff[j];
          for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) hh_off[j*detail_quadelem::COORD_DIM+k] = (rj-rmin)*Xnnodes[t*detail_quadelem::COORD_DIM+k];
          }
          detail_dyadic_near::NearInteracDyadic<order,Real>(M_hh, hh_Xt1, ntrg, ker, elem_idx, &qel, near_digits, hh_off, hh_w);
          detail_quadelem::ScatterSelfBlock<order,Real>(M, M_hh, t, KDIM0, KDIM1_out, true);
          });
    }

  }

  namespace detail_dispatch {

    template <Integer order, class Real, class Kernel> void SelfInteracDispatch(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      switch (detail_quadelem::Access<Real>::Scheme(qel)) {
        case QuadElemList<Real>::QuadScheme::TensorProduct: detail_tensorprod::SelfInteracTensorProduct<order,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
        case QuadElemList<Real>::QuadScheme::Duffy:         detail_duffy::SelfInteracDuffy<order,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
        case QuadElemList<Real>::QuadScheme::Hedgehog:      detail_hedgehog::SelfInteracHedgehog<order,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
      }
    }

    template <Integer order, class Real, class Kernel> void NearInteracDispatch(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits) {
      const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
      if (detail_quadelem::Access<Real>::Scheme(qel) == QuadElemList<Real>::QuadScheme::TensorProduct) {
        detail_tensorprod::NearInteracTensorProduct<order,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits);
      } else {
        detail_dyadic_near::NearInteracDyadic<order,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits);
      }
    }

  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->Order();
    const Integer digits = detail_quadelem::DigitsFromTol<Real>(tol);
    switch (order) {
      case  4: detail_dispatch::SelfInteracDispatch<4,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
      case  8: detail_dispatch::SelfInteracDispatch<8,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
      case 12: detail_dispatch::SelfInteracDispatch<12,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
      case 16: detail_dispatch::SelfInteracDispatch<16,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
      case 20: detail_dispatch::SelfInteracDispatch<20,Real>(M_lst, ker, trg_dot_prod, self, digits); break;
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,12,16,20} for the templated near/self schemes.");
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->Order();
    const Integer digits = detail_quadelem::DigitsFromTol<Real>(tol);
    switch (order) {
      case  4: detail_dispatch::NearInteracDispatch<4,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case  8: detail_dispatch::NearInteracDispatch<8,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case 12: detail_dispatch::NearInteracDispatch<12,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case 16: detail_dispatch::NearInteracDispatch<16,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case 20: detail_dispatch::NearInteracDispatch<20,Real>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,12,16,20} for the templated near/self schemes.");
    }
  }

  template <class Real> const Vector<Real>& QuadElemList<Real>::ParamNodes(const Integer Order) {
    return LegQuadRule<Real>::nds(Order);
  }

  template <class Real> void QuadElemList<Real>::Write(const std::string& fname, const Comm& comm) const {
    auto allgather = [&comm](Vector<Real>& v_out, const Vector<Real>& v_in) {
      const Long Nproc = comm.Size();
      StaticArray<Long,1> len{v_in.Dim()};
      Vector<Long> cnt(Nproc), dsp(Nproc);
      comm.Allgather(len + 0, 1, cnt.begin(), 1);
      dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), Nproc);

      v_out.ReInit(dsp[Nproc-1] + cnt[Nproc-1]);
      comm.Allgatherv(v_in.begin(), v_in.Dim(), v_out.begin(), cnt.begin(), dsp.begin());
    };

    Vector<Real> coord_;
    allgather(coord_, coord);

    const Long nnode_per_elem = (Long)order * order;
    const Long Nelem_total = coord_.Dim() / (detail_quadelem::COORD_DIM * nnode_per_elem);
    SCTL_ASSERT(coord_.Dim() == Nelem_total * detail_quadelem::COORD_DIM * nnode_per_elem);

    if (comm.Rank()) return;

    const Integer precision = (Integer)std::ceil(-std::log((double)machine_eps<Real>()) / std::log(10.0));
    const Integer width = precision + 8;
    std::ofstream file(fname, std::ofstream::out | std::ofstream::trunc);
    SCTL_ASSERT_MSG(file.good(), std::string("Unable to open file for writing: ") + fname);

    file << "#";
    file << std::setw(width - 1) << "X";
    file << std::setw(width) << "Y";
    file << std::setw(width) << "Z";
    file << std::setw(width) << "ElemOrder";
    file << '\n';

    file << std::scientific << std::setprecision(precision);
    for (Long elem_idx = 0; elem_idx < Nelem_total; elem_idx++) {
      const Long base = elem_idx * detail_quadelem::COORD_DIM * nnode_per_elem;
      for (Long p = 0; p < nnode_per_elem; p++) {
        for (Integer k = 0; k < detail_quadelem::COORD_DIM; k++) {
          file << std::setw(width) << coord_[base + k * nnode_per_elem + p];
        }
        if (!p) file << std::setw(width) << order;
        file << '\n';
      }
    }
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Read(const std::string& fname, const Comm& comm) {
    std::ifstream file(fname, std::ifstream::in);
    SCTL_ASSERT_MSG(file.good(), std::string("Unable to open file for reading: ") + fname);

    std::string line;
    Vector<ValueType> coord_;
    Vector<Long> order_markers;
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
    file.close();

    SCTL_ASSERT(order_markers.Dim() > 0);
    const Integer file_order = order_markers[0];
    SCTL_ASSERT(file_order > 0);
    const Long nnode_per_elem = (Long)file_order * file_order;

    SCTL_ASSERT(order_markers.Dim() % nnode_per_elem == 0);
    const Long Nelem_total = order_markers.Dim() / nnode_per_elem;
    for (Long elem = 0; elem < Nelem_total; elem++) {
      const Long offset = elem * nnode_per_elem;
      SCTL_ASSERT(order_markers[offset] == file_order);
      for (Long j = 1; j < nnode_per_elem; j++) {
        SCTL_ASSERT(order_markers[offset + j] == file_order || order_markers[offset + j] == -1);
      }
    }

    {
      Long i0, i1;
      detail_quadelem::PartitionRange<Real>(Nelem_total, comm, i0, i1);

      const Long j0 = i0 * nnode_per_elem;
      const Long j1 = i1 * nnode_per_elem;

      Vector<ValueType> coord_local;
      coord_local.ReInit((j1 - j0) * detail_quadelem::COORD_DIM, coord_.begin() + j0 * detail_quadelem::COORD_DIM, false);
      Init<ValueType>(file_order, coord_local, Comm::Self());
    }
  }

  template <class Real> void QuadElemList<Real>::GetVTUData(VTUData& vtu_data, const Vector<Real>& F, const Long elem_idx) const {
    if (elem_idx == -1) {
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

    Vector<Real> u_nodes(order + 2), v_nodes(order + 2);
    u_nodes[0] = 0;
    v_nodes[0] = 0;
    u_nodes[order + 1] = 1;
    v_nodes[order + 1] = 1;
    Vector<Real>(order, u_nodes.begin() + 1, false) = ParamNodes(order);
    Vector<Real>(order, v_nodes.begin() + 1, false) = ParamNodes(order);

    Vector<Real> X;
    GetGeom(&X, nullptr, nullptr, nullptr, nullptr, u_nodes, v_nodes, elem_idx);

    const Long Nu = u_nodes.Dim();
    const Long Nv = v_nodes.Dim();
    Vector<Real> Fgrid;
    if (F.Dim()) {
      const Long nnode_per_elem = (Long)order * order;
      const Long dof = F.Dim() / nnode_per_elem;
      SCTL_ASSERT(F.Dim() == nnode_per_elem * dof);

      Vector<Real> F_soa(dof * nnode_per_elem);
      for (Long p = 0; p < nnode_per_elem; p++) {
        for (Long k = 0; k < dof; k++) {
          F_soa[k * nnode_per_elem + p] = F[p * dof + k];
        }
      }

      Matrix<Real> MuT(order, Nu), Mv(order, Nv);
      Vector<Real> Mu_(order * Nu, MuT.begin(), false);
      Vector<Real> Mv_(order * Nv, Mv.begin(), false);
      LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_nodes);
      LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_nodes);
      MuT = MuT.Transpose();

      Vector<Real> F_soa_eval;
      detail_quadelem::EvalTensorProduct(F_soa_eval, F_soa, MuT, Mv);

      Fgrid.ReInit(Nu * Nv * dof);
      for (Long p = 0; p < Nu * Nv; p++) {
        for (Long k = 0; k < dof; k++) {
          Fgrid[p * dof + k] = F_soa_eval[k * (Nu * Nv) + p];
        }
      }
    }

    const Long point_offset = vtu_data.coord.Dim() / detail_quadelem::COORD_DIM;
    for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
    for (const auto& f : Fgrid) vtu_data.value.PushBack((VTUData::VTKReal)f);

    for (Long i = 0; i < Nu - 1; i++) {
      for (Long j = 0; j < Nv - 1; j++) {
        const Long idx = point_offset + i * Nv + j;
        vtu_data.connect.PushBack(idx);
        vtu_data.connect.PushBack(idx + 1);
        vtu_data.connect.PushBack(idx + Nv + 1);
        vtu_data.connect.PushBack(idx + Nv);
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

    elem_lst.coord.ReInit(coord.Dim());
    elem_lst.dcoord_du.ReInit(dcoord_du.Dim());
    elem_lst.dcoord_dv.ReInit(dcoord_dv.Dim());
    for (Long i = 0; i < coord.Dim(); i++) elem_lst.coord[i] = (ValueType)coord[i];
    for (Long i = 0; i < dcoord_du.Dim(); i++) elem_lst.dcoord_du[i] = (ValueType)dcoord_du[i];
    for (Long i = 0; i < dcoord_dv.Dim(); i++) elem_lst.dcoord_dv[i] = (ValueType)dcoord_dv[i];

    elem_lst.X_node.ReInit(X_node.Dim());
    elem_lst.Xn_node.ReInit(Xn_node.Dim());
    for (Long i = 0; i < X_node.Dim(); i++) elem_lst.X_node[i] = (ValueType)X_node[i];
    for (Long i = 0; i < Xn_node.Dim(); i++) elem_lst.Xn_node[i] = (ValueType)Xn_node[i];
    elem_lst.node_cnt = node_cnt;
  }

}

#endif
