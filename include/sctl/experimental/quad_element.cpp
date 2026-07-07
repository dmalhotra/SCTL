#ifndef _SCTL_QUAD_ELEMENT_CPP_
#define _SCTL_QUAD_ELEMENT_CPP_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sctl.hpp>
#include "sctl/experimental/quad_element.hpp"
#include "sctl/experimental/alpert_quadr.cpp"
#include "sctl/experimental/bench_quad.hpp"

namespace sctl {

  template <class Real> void QuadElemList<Real>::PartitionRange(Long Nelem_total, const Comm& comm, Long& i0, Long& i1) {
    const Long Np = comm.Size();
    const Long pid = comm.Rank();
    i0 = Nelem_total * (pid + 0) / Np;
    i1 = Nelem_total * (pid + 1) / Np;
  }

  template <class Real> template <class ValueType> QuadElemList<Real>::QuadElemList(Integer order0, const Vector<ValueType>& coord0, const Comm& comm) {
    Init(order0, coord0, comm);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Init(Integer order0, const Vector<ValueType>& coord0, const Comm& comm) {
    order = order0;
    SCTL_ASSERT(order > 0);

    const Long nnode_per_elem = (Long)order * order;
    SCTL_ASSERT(coord0.Dim() % (nnode_per_elem * COORD_DIM) == 0);
    const Long nelem_total = coord0.Dim() / (nnode_per_elem * COORD_DIM);

    // When distributed, `coord0` holds the full (replicated) mesh; keep only this
    // rank's contiguous element slice [i0,i1).
    Long i0, i1;
    PartitionRange(nelem_total, comm, i0, i1);
    nelem = i1 - i0;

    coord.ReInit(nelem * COORD_DIM * nnode_per_elem);
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      const Long base = elem_idx * COORD_DIM * nnode_per_elem;
      const Long src_elem = i0 + elem_idx;
      for (Integer k = 0; k < COORD_DIM; k++) {
        for (Long p = 0; p < nnode_per_elem; p++) {
          coord[base + k * nnode_per_elem + p] = (Real)coord0[(src_elem * nnode_per_elem + p) * COORD_DIM + k];
        }
      }
    }

    BuildDerivativeCache();
  }

  template <class Real> void QuadElemList<Real>::NodalDerivs(const Vector<Real>& coord_slab, const Integer order, Vector<Real>& du_slab, Vector<Real>& dv_slab) {
    const Long nnode_per_elem = (Long)order * order;
    const Long ncomp = coord_slab.Dim() / nnode_per_elem;
    SCTL_ASSERT(coord_slab.Dim() == ncomp * nnode_per_elem);
    if (du_slab.Dim() != coord_slab.Dim()) du_slab.ReInit(coord_slab.Dim());
    if (dv_slab.Dim() != coord_slab.Dim()) dv_slab.ReInit(coord_slab.Dim());

    const auto& nodes = ParamNodes(order);
    Vector<Real> line_in(order), line_out(order);
    for (Long k = 0; k < ncomp; k++) {
      const Long cb = k * nnode_per_elem;

      for (Integer j = 0; j < order; j++) { // d/du: differentiate along i (u-slow), fixed j
        for (Integer i = 0; i < order; i++) line_in[i] = coord_slab[cb + i * order + j];
        LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
        for (Integer i = 0; i < order; i++) du_slab[cb + i * order + j] = line_out[i];
      }

      for (Integer i = 0; i < order; i++) { // d/dv: differentiate along j (v-fast), fixed i
        for (Integer j = 0; j < order; j++) line_in[j] = coord_slab[cb + i * order + j];
        LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
        for (Integer j = 0; j < order; j++) dv_slab[cb + i * order + j] = line_out[j];
      }
    }
  }

  template <class Real> void QuadElemList<Real>::BuildDerivativeCache() {
    dcoord_du.ReInit(coord.Dim());
    dcoord_dv.ReInit(coord.Dim());

    const Long nnode_per_elem = (Long)order * order;
    const Long elem_stride = COORD_DIM * nnode_per_elem;
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      const Long base = elem_idx * elem_stride;
      const Vector<Real> coord_(elem_stride, (Iterator<Real>)coord.begin() + base, false);
      Vector<Real> du_(elem_stride, dcoord_du.begin() + base, false);
      Vector<Real> dv_(elem_stride, dcoord_dv.begin() + base, false);
      NodalDerivs(coord_, order, du_, dv_);
    }
  }

  template <class Real> const Matrix<Real>& QuadElemList<Real>::DiffMat(const Integer order) {
    // D[i][a] = L_i'(node_a). Cached for all orders at first use to avoid an
    // O(order^3) per-self-target rebuild.
    constexpr Integer MAX_ORDER = 50;
    SCTL_ASSERT(0 < order && order <= MAX_ORDER);
    auto compute_all = []() {
      Vector<Matrix<Real>> D(MAX_ORDER + 1);
      for (Integer n = 1; n <= MAX_ORDER; n++) {
        const Vector<Real>& nds = ParamNodes(n);
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

  template <class Real> template <Integer order> void QuadElemList<Real>::BuildInterp1D(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& param) {
    const Long N = param.Dim();
    M.ReInit(order, N);
    { Vector<Real> v(order*N, M.begin(), false); LagrangeInterp<Real>::Interpolate(v, ParamNodes(order), param); }
    dM.ReInit(order, N);
    Matrix<Real>::GEMM(dM, DiffMat<order>(), M);
    MT = M.Transpose();
    dMT = dM.Transpose();
  }

  template <class Real> template <Integer order, Integer digits> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::SelfVRule(const Integer tj) {
    // Composite/graded Alpert log-singular v-rule + interpolation for v0 = nds[tj].
    // Geometry-independent; grading levels toward v0 grow with `digits` so v-accuracy
    // tracks tolerance. Cached once per (order, digits, tj).

    auto compute_all = []() {
      const Vector<Real>& nds = ParamNodes(order);
      const Integer Lvl = DigitsVLevels<digits>();
      const Integer QuadOrder = DigitsQuadOrder<digits>();
      Vector<NodeRuleData> data(order);
      for (Integer j = 0; j < order; j++) {
        LogSingularQuad1D(data[j].param, data[j].w, nds[j], Lvl, QuadOrder);
        BuildInterp1D<order>(data[j].M, data[j].dM, data[j].MT, data[j].dMT, data[j].param);
      }
      return data;
    };
    static const Vector<NodeRuleData> data = compute_all();
    return data[tj];
  }

  template <class Real> template <Integer order, Integer digits> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::SelfURule(const Integer ti) {
    // Graded u-rule + interpolation for u0 = nds[ti]. The subdivision is geometry-
    // independent (scale-invariance), so fixed by (order, ti, digits); cached once.
    auto compute_all = []() {
      const Vector<Real>& nds = ParamNodes(order);
      const Integer QuadOrder = DigitsQuadOrder<digits>();
      const Real b_ellipse = DigitsBEllipse<digits>();
      Vector<Real> qnds, qwts;
      LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);
      Vector<NodeRuleData> data(order);
      for (Integer i = 0; i < order; i++) {
        BuildGraded1D(data[i].param, data[i].w, nds[i], b_ellipse, qnds, qwts);
        BuildInterp1D<order>(data[i].M, data[i].dM, data[i].MT, data[i].dMT, data[i].param);
      }
      return data;
    };
    static const Vector<NodeRuleData> data = compute_all();
    return data[ti];
  }

  template <class Real> template <Integer Nbeta> const std::pair<Vector<Real>, Vector<Real>>& QuadElemList<Real>::GLRuleNbeta() {
    // GL rule on [0,1] for Nbeta points (exceeds LegQuadRule's compile-time cache).
    // Built once as a function-local static for lock-free reads.
    static const std::pair<Vector<Real>, Vector<Real>> gl = []() {
      std::pair<Vector<Real>, Vector<Real>> p;
      LegQuadRule<Real>::ComputeNdsWts(&p.first, &p.second, Nbeta);
      return p;
    }();
    return gl;
  }

  template <class Real> const std::pair<Vector<Real>, Vector<Real>>& QuadElemList<Real>::GLRuleNbetaDispatch(const Integer Nbeta) {
    if      (Nbeta == 32)  return GLRuleNbeta<32>();
    else if (Nbeta == 64)  return GLRuleNbeta<64>();
    else if (Nbeta == 96)  return GLRuleNbeta<96>();
    else if (Nbeta == 128) return GLRuleNbeta<128>();
    else if (Nbeta == 192) return GLRuleNbeta<192>();
    else if (Nbeta == 256) return GLRuleNbeta<256>();
    else if (Nbeta == 384) return GLRuleNbeta<384>();
    else if (Nbeta == 512) return GLRuleNbeta<512>();
    SCTL_ASSERT_MSG(false, "RectPolar Nbeta (cov_order) must be one of {32,64,96,128,192,256,384,512}.");
    return GLRuleNbeta<512>(); // unreachable
  }

  template <class Real> template <Integer order, Integer Nbeta, Integer q> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::RPSelfRule(const Integer k) {
    // Self-RP COV rule + interpolation for the singularity at nds[m] (same rule serves
    // u and v). Geometry-independent (fixed COV), so cached once per (order, Nbeta, q).
    static const Vector<NodeRuleData> data = []() {
      const Vector<Real>& nds = ParamNodes(order);
      const std::pair<Vector<Real>, Vector<Real>>& gl = GLRuleNbeta<Nbeta>();
      Vector<NodeRuleData> d(order);
      for (Integer m = 0; m < order; m++) {
        RectPolarNodes1D(d[m].param, d[m].w, 2*nds[m] - 1, q, gl.first, gl.second);
        BuildInterp1D<order>(d[m].M, d[m].dM, d[m].MT, d[m].dMT, d[m].param);
      }
      return d;
    }();
    return data[k];
  }

  template <class Real> template <Integer order> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::RPSelfRuleDispatch(const Integer k, const Integer q, const Integer Nbeta) {
    // Map the runtime (q, Nbeta) to the compile-time RPSelfRule instantiation.
    if (q == 6) {
      if      (Nbeta == 32)  return RPSelfRule<order,32,6>(k);
      else if (Nbeta == 64)  return RPSelfRule<order,64,6>(k);
      else if (Nbeta == 96)  return RPSelfRule<order,96,6>(k);
      else if (Nbeta == 128) return RPSelfRule<order,128,6>(k);
      else if (Nbeta == 192) return RPSelfRule<order,192,6>(k);
      else if (Nbeta == 256) return RPSelfRule<order,256,6>(k);
      else if (Nbeta == 384) return RPSelfRule<order,384,6>(k);
      else if (Nbeta == 512) return RPSelfRule<order,512,6>(k);
    } else if (q == 10) {
      if      (Nbeta == 32)  return RPSelfRule<order,32,10>(k);
      else if (Nbeta == 64)  return RPSelfRule<order,64,10>(k);
      else if (Nbeta == 96)  return RPSelfRule<order,96,10>(k);
      else if (Nbeta == 128) return RPSelfRule<order,128,10>(k);
      else if (Nbeta == 192) return RPSelfRule<order,192,10>(k);
      else if (Nbeta == 256) return RPSelfRule<order,256,10>(k);
      else if (Nbeta == 384) return RPSelfRule<order,384,10>(k);
      else if (Nbeta == 512) return RPSelfRule<order,512,10>(k);
    }
    SCTL_ASSERT_MSG(false, "RectPolar (cov_q, Nbeta) must have cov_q in {6,10} and Nbeta in {32,64,96,128,192,256,384,512}.");
    return RPSelfRule<order,512,6>(k); // unreachable
  }

  template <class Real> Long QuadElemList<Real>::Size() const {
    return nelem;
  }

  template <class Real> Integer QuadElemList<Real>::Order() const {
    return order;
  }

  template <class Real> typename QuadElemList<Real>::QuadScheme QuadElemList<Real>::Scheme() const {
    return scheme_;
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv) {
    // Per component, out = MuT . in . Mv with general (non-square) shapes:
    //   MuT: Nu x R, in: R x S, Mv: S x Nv -> out: Nu x Nv (R, S independent
    //   contraction dims; common case R = S = order, square `in`).
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

    // 2 multiply-adds per inner-product entry: (R x S).(S x Nv) and (Nu x R).(R x Nv).
    BENCH_FLOPS(ncomp * 2.0 * Nv * ((double)R * S + (double)Nu * R));
    for (Long k = 0; k < ncomp; k++) {
      const Matrix<ValueType> in_(R, S, (Iterator<ValueType>)in.begin() + k * (Long)R * S, false);
      Matrix<ValueType> out_(Nu, Nv, out.begin() + k * Nout, false);
      Matrix<ValueType>::GEMM(tmp, in_, Mv);   // (R x S) . (S x Nv) = (R x Nv)
      Matrix<ValueType>::GEMM(out_, MuT, tmp); // (Nu x R) . (R x Nv) = (Nu x Nv)
    }
  }

  template <class Real> void QuadElemList<Real>::GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx, const Vector<Real>* origin) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nu = u_param.Dim();
    const Long Nv = v_param.Dim();
    const Long N = Nu * Nv;

    if (X && X->Dim() != N * COORD_DIM) X->ReInit(N * COORD_DIM);
    if (Xn && Xn->Dim() != N * COORD_DIM) Xn->ReInit(N * COORD_DIM);
    if (Xa && Xa->Dim() != N) Xa->ReInit(N);
    if (dX_du && dX_du->Dim() != N * COORD_DIM) dX_du->ReInit(N * COORD_DIM);
    if (dX_dv && dX_dv->Dim() != N * COORD_DIM) dX_dv->ReInit(N * COORD_DIM);

    Matrix<Real> MuT(order, Nu), Mv(order, Nv);
    Vector<Real> Mu_(order * Nu, MuT.begin(), false);
    Vector<Real> Mv_(order * Nv, Mv.begin(), false);
    LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_param);
    LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_param);
    MuT = MuT.Transpose();

    SCTL_ASSERT(elem_idx >= 0 && elem_idx < nelem);
    const Long base = elem_idx * nnode_per_elem * COORD_DIM;
    const Vector<Real> coord_(COORD_DIM * nnode_per_elem, (Iterator<Real>)coord.begin() + base, false);
    const Vector<Real> dcoord_du_(COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_du.begin() + base, false);
    const Vector<Real> dcoord_dv_(COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_dv.begin() + base, false);

    // Target-centering: subtract `origin` from nodal positions before interpolation so
    // X is target-relative (accurate near the singularity); derivatives recomputed from
    // the shifted slab. origin == nullptr keeps the cached absolute-coordinate path.
    Vector<Real> coord_shift, du_shift, dv_shift;
    const Vector<Real>* pos_in = &coord_;
    const Vector<Real>* du_in = &dcoord_du_;
    const Vector<Real>* dv_in = &dcoord_dv_;
    if (origin) {
      coord_shift.ReInit(COORD_DIM * nnode_per_elem);
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = (*origin)[k];
        for (Long p = 0; p < nnode_per_elem; p++) coord_shift[k * nnode_per_elem + p] = coord_[k * nnode_per_elem + p] - ok;
      }
      if (Xn || Xa || dX_du || dX_dv) NodalDerivs(coord_shift, order, du_shift, dv_shift);
      pos_in = &coord_shift; du_in = &du_shift; dv_in = &dv_shift;
    }

    if (X) {
      Vector<Real> X_soa;
      EvalTensorProduct(X_soa, *pos_in, MuT, Mv);
      for (Long i = 0; i < N; i++) {
        (*X)[i * COORD_DIM + 0] = X_soa[0 * N + i];
        (*X)[i * COORD_DIM + 1] = X_soa[1 * N + i];
        (*X)[i * COORD_DIM + 2] = X_soa[2 * N + i];
      }
    }
    if (Xn || Xa || dX_du || dX_dv) {
      Vector<Real> dXdu_soa, dXdv_soa;
      EvalTensorProduct(dXdu_soa, *du_in, MuT, Mv);
      EvalTensorProduct(dXdv_soa, *dv_in, MuT, Mv);
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
          (*Xn)[i * COORD_DIM + 0] = n0 * inv_area;
          (*Xn)[i * COORD_DIM + 1] = n1 * inv_area;
          (*Xn)[i * COORD_DIM + 2] = n2 * inv_area;
        }
        if (Xa) {
          (*Xa)[i] = area;
        }
        if (dX_du) {
          (*dX_du)[i * COORD_DIM + 0] = du0;
          (*dX_du)[i * COORD_DIM + 1] = du1;
          (*dX_du)[i * COORD_DIM + 2] = du2;
        }
        if (dX_dv) {
          (*dX_dv)[i * COORD_DIM + 0] = dv0;
          (*dX_dv)[i * COORD_DIM + 1] = dv1;
          (*dX_dv)[i * COORD_DIM + 2] = dv2;
        }
      }
    }
  }

  template <class Real> void QuadElemList<Real>::GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nnode = nelem * nnode_per_elem;

    if (X && X->Dim() != Nnode * COORD_DIM) X->ReInit(Nnode * COORD_DIM);
    if (Xn && Xn->Dim() != Nnode * COORD_DIM) Xn->ReInit(Nnode * COORD_DIM);
    if (element_wise_node_cnt) {
      if (element_wise_node_cnt->Dim() != nelem) element_wise_node_cnt->ReInit(nelem);
      (*element_wise_node_cnt) = nnode_per_elem;
    }

    const auto& nodes = ParamNodes(order);
    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      Vector<Real> X_, Xn_;
      if (X) X_.ReInit(nnode_per_elem * COORD_DIM, X->begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      if (Xn) Xn_.ReInit(nnode_per_elem * COORD_DIM, Xn->begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      GetGeom((X ? &X_ : nullptr), (Xn ? &Xn_ : nullptr), nullptr, nullptr, nullptr, nodes, nodes, elem_idx);
    }
  }

  template <class Real> void QuadElemList<Real>::GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nnode = nelem * nnode_per_elem;

    if (X.Dim() != Nnode * COORD_DIM) X.ReInit(Nnode * COORD_DIM);
    if (Xn.Dim() != Nnode * COORD_DIM) Xn.ReInit(Nnode * COORD_DIM);
    if (wts.Dim() != Nnode) wts.ReInit(Nnode);
    if (dist_far.Dim() != Nnode) dist_far.ReInit(Nnode);
    if (element_wise_node_cnt.Dim() != nelem) element_wise_node_cnt.ReInit(nelem);
    element_wise_node_cnt = nnode_per_elem;

    const auto& nodes = ParamNodes(order);
    const auto& node_wts = LegQuadRule<Real>::wts(order);

    // dist_nodes[i]: param-space distance from node i to the Bernstein ellipse boundary
    // for [0,1]. rho is chosen so rho^{2n} = 64/(15*tol) bounds the far-field GL error
    // below tol (semi-axes a=(rho-1/rho)/4, b=(rho+1/rho)/4, centered at 0.5). Closest
    // point: on the curve when |cos_t|<=1, else the vertex b-|x-0.5|.
    Vector<Real> dist_nodes(order);
    {
      const Integer n = order;
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real rho = pow<Real>((64 / (15 * tol_)), 1 / (Real)(2 * n));
      const Real a = (rho - 1 / rho) / 4;
      const Real b = (rho + 1 / rho) / 4;
      for (Integer i = 0; i < n; i++) {
        dist_nodes[i] = b - fabs(nodes[i] - (Real)0.5);  // vertex fallback
        const Real cos_t = 4 * b * (nodes[i] - (Real)0.5);
        if (fabs(cos_t) <= 1) {
          dist_nodes[i] = a * sqrt<Real>(1 + ((a * a) / (b * b) - 1) * cos_t * cos_t);
        }
      }
    }

    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      Vector<Real> X_(nnode_per_elem * COORD_DIM, X.begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      Vector<Real> Xn_(nnode_per_elem * COORD_DIM, Xn.begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
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

          // Scale param-space distances to physical by element arc-length; max over u,v.
          const Real du = sqrt<Real>(dXdu[p * COORD_DIM + 0] * dXdu[p * COORD_DIM + 0] +
                                     dXdu[p * COORD_DIM + 1] * dXdu[p * COORD_DIM + 1] +
                                     dXdu[p * COORD_DIM + 2] * dXdu[p * COORD_DIM + 2]);
          const Real dv = sqrt<Real>(dXdv[p * COORD_DIM + 0] * dXdv[p * COORD_DIM + 0] +
                                     dXdv[p * COORD_DIM + 1] * dXdv[p * COORD_DIM + 1] +
                                     dXdv[p * COORD_DIM + 2] * dXdv[p * COORD_DIM + 2]);
          dist_far_[p] = std::max(dist_nodes[i] * du, dist_nodes[j] * dv);
        }
      }
    }
  }
  
  template <class Real> void QuadElemList<Real>::QuadParams(const Real tol, Real& b_ellipse, Integer& QuadOrder) {
    // Fix rho, derive per-panel GL order from rho and tol (cf.
    // SlenderElemList::NearInteracHelper). A panel of extent L is admissible when the
    // target is at distance >= b_ellipse*L.
    const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
    const double rho = 2.5;
    b_ellipse = (Real)((rho + 1/rho) / 4);
    QuadOrder = std::max<Integer>(1, (Integer)std::ceil(-std::log(((15.0*(rho*rho-1))/64.0)*(double)tol_)/std::log(rho)*0.5 + 1));
  }

  template <class Real> template <Integer digits> Integer QuadElemList<Real>::DigitsQuadOrder() {
    // Evaluated once per `digits` at tol = 10^-digits.
    static const Integer QuadOrder = []() { Real b; Integer q; QuadParams(pow<digits,Real>((Real)0.1), b, q); return q; }();
    return QuadOrder;
  }

  template <class Real> template <Integer digits> Real QuadElemList<Real>::DigitsBEllipse() {
    static const Real b_ellipse = []() { Real b; Integer q; QuadParams(pow<digits,Real>((Real)0.1), b, q); return b; }();
    return b_ellipse;
  }

  template <class Real> Integer QuadElemList<Real>::VLevelsForDigits(const Integer digits) {
    // Geometric grading levels per side toward v0 in the composite Alpert v-rule. 
    return std::min<Integer>(12, std::max<Integer>(1, digits - 5));
  }

  template <class Real> template <Integer digits> Integer QuadElemList<Real>::DigitsVLevels() {
    return VLevelsForDigits(digits);
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker, const Matrix<Real>* Mv_pre, const Matrix<Real>* dMv_pre, const Matrix<Real>* Mu_pre, const Matrix<Real>* dMu_pre, const Matrix<Real>* MvT_pre, const Matrix<Real>* MuT_pre, const Matrix<Real>* dMuT_pre) {
    // Accumulate the tensor-product quadrature (u_param x v_param, weights wu (x) wv)
    // against the single target Xtrg. Shared by the near (per-leaf) and self schemes.
    // Tensor grid is u-slow/v-fast: node (a,b) has flat index q = a*Nv + b.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const Long Nu = (Mu_pre ? Mu_pre->Dim(1) : u_param.Dim());
    const Long Nv = (Mv_pre ? Mv_pre->Dim(1) : v_param.Dim());
    const Long nq = Nu * Nv;
    if (!nq) return;
    const Integer C = KDIM0 * KDIM1_out;

    const Vector<Real>& pnds = ParamNodes(order);
    const Matrix<Real>& D = DiffMat<order>();

    // 1D value + derivative interpolation (patch nodes -> quad nodes), dMu = D.Mu.
    // Tangents come from the SAME target-shifted slab via the tensor interpolation
    // below -- no per-target NodalDerivs. Use preloaded M*_pre/dM*_pre when supplied
    // (self's fixed Alpert/COV rule), else build from u_param/v_param (adaptive rule).
    Matrix<Real> Mu_local, dMu_local, MuT_local, dMuT_local;
    Matrix<Real> Mv_local, dMv_local, MvT_local;
    if (!Mu_pre || !Mv_pre) {
      BENCH_TIC(InterpBuild);
      if (!Mu_pre) {
        Mu_local.ReInit(order, Nu);
        { Vector<Real> v(order*Nu, Mu_local.begin(), false); LagrangeInterp<Real>::Interpolate(v, pnds, u_param); }
        dMu_local.ReInit(order, Nu);
        Matrix<Real>::GEMM(dMu_local, D, Mu_local);
        MuT_local = Mu_local.Transpose();
        dMuT_local = dMu_local.Transpose();
      }
      if (!Mv_pre) {
        Mv_local.ReInit(order, Nv);
        { Vector<Real> v(order*Nv, Mv_local.begin(), false); LagrangeInterp<Real>::Interpolate(v, pnds, v_param); }
        dMv_local.ReInit(order, Nv);
        Matrix<Real>::GEMM(dMv_local, D, Mv_local);
        MvT_local = Mv_local.Transpose();
      }
      BENCH_TOC(InterpBuild);
    }
    const Matrix<Real>& Mu  = (Mu_pre  ? *Mu_pre  : Mu_local);
    // const Matrix<Real>& dMu = (dMu_pre ? *dMu_pre : dMu_local);
    const Matrix<Real>& MuT  = (MuT_pre  ? *MuT_pre  : MuT_local);
    const Matrix<Real>& dMuT = (dMuT_pre ? *dMuT_pre : dMuT_local);
    const Matrix<Real>& Mv  = (Mv_pre  ? *Mv_pre  : Mv_local);
    const Matrix<Real>& dMv = (dMv_pre ? *dMv_pre : dMv_local);
    const Matrix<Real>& MvT  = (MvT_pre  ? *MvT_pre  : MvT_local);
    // const Matrix<Real> MuT = Mu.Transpose();
    // const Matrix<Real> dMuT = dMu.Transpose();
    // const Matrix<Real> MvT = Mv.Transpose();

    // Target-centering: subtract Xtrg from nodal coords before interpolation so
    // positions are source-minus-target (accurate r near the singularity); tangents
    // come from the same shifted slab.
    BENCH_TIC(GeomTensor);
    const Long base = elem_idx * nnode * COORD_DIM; // TODO: assumes uniform per-element grid; consider omp scan of elem_cnt.
    // Per-call scratch reused across the many IntegrateBlock calls (order^2 per element
    // for self, per leaf for near). thread_local so each OMP thread has its own; every
    // buffer is fully overwritten before use, so reuse is safe. Avoids re-malloc churn.
    thread_local Vector<Real> coord_shift;
    if (coord_shift.Dim() != COORD_DIM*nnode) coord_shift.ReInit(COORD_DIM*nnode);
    for (Integer k = 0; k < COORD_DIM; k++) {
      const Real ok = Xtrg[k];
      for (Long p = 0; p < nnode; p++) coord_shift[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
    }
    thread_local Vector<Real> X_soa, dXdu_soa, dXdv_soa; // component-major SoA: [k*nq + q] (ReInit'd by EvalTensorProduct)
    EvalTensorProduct(X_soa,    coord_shift, MuT,  Mv);
    EvalTensorProduct(dXdu_soa, coord_shift, dMuT, Mv);
    EvalTensorProduct(dXdv_soa, coord_shift, MuT,  dMv);
    BENCH_TOC(GeomTensor);

    // Sources are target-relative -> kernel target at the origin Xt0 = 0.
    StaticArray<Real,COORD_DIM> Xt0{0, 0, 0};
    const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);
    BENCH_TIC(Assembly);
    thread_local Vector<Real> Xsrc, Xnsrc, wq;
    if (Xsrc.Dim() != nq*COORD_DIM) { Xsrc.ReInit(nq*COORD_DIM); Xnsrc.ReInit(nq*COORD_DIM); wq.ReInit(nq); }
    for (Long a = 0; a < Nu; a++) {
      for (Long b = 0; b < Nv; b++) {
        const Long q = a*Nv + b;
        const Real du0 = dXdu_soa[0*nq+q], du1 = dXdu_soa[1*nq+q], du2 = dXdu_soa[2*nq+q];
        const Real dv0 = dXdv_soa[0*nq+q], dv1 = dXdv_soa[1*nq+q], dv2 = dXdv_soa[2*nq+q];
        const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
        const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
        const Real inv_area = (area > 0 ? 1/area : 0);
        Xsrc[q*COORD_DIM+0] = X_soa[0*nq+q]; Xsrc[q*COORD_DIM+1] = X_soa[1*nq+q]; Xsrc[q*COORD_DIM+2] = X_soa[2*nq+q];
        Xnsrc[q*COORD_DIM+0] = n0*inv_area; Xnsrc[q*COORD_DIM+1] = n1*inv_area; Xnsrc[q*COORD_DIM+2] = n2*inv_area;
        wq[q] = area*wu[a]*wv[b];
      }
    }
    BENCH_TOC(Assembly);

    // Kernel matrix from sources to the shifted target (r = trg - src, source normal).
    BENCH_TIC(KernelEval);
    thread_local Matrix<Real> Mker;
    ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xsrc, Xnsrc); // (nq*KDIM0 x KDIM1full)
    BENCH_TOC(KernelEval);

    // Weighted kernel in component-major layout KWc[c*nq + q] so it feeds
    // EvalTensorProduct directly; dot-product case contracts the inner COORD_DIM with
    // the target normal. NOTE: q is kept OUTERMOST on purpose -- it streams Mker
    // (the dense, row-major (nq*KDIM0 x KDIM1full) read) contiguously. A c-outer/q-inner
    // reorder (to make the KWc store unit-stride) was tried and was ~25% SLOWER for
    // Stokes self because it turns the Mker read into KDIM0*KDIM1_out strided gather
    // passes; streaming Mker once wins.
    BENCH_TIC(KernelWeight);
    thread_local Vector<Real> KWc;
    if (KWc.Dim() != C*nq) KWc.ReInit(C*nq);
    for (Long q = 0; q < nq; q++) {
      for (Integer k0 = 0; k0 < KDIM0; k0++) {
        for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          Real val;
          if (trg_dot_prod) {
            val = 0;
            for (Integer l = 0; l < COORD_DIM; l++) {
              val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * normal_trg[l];
            }
          } else {
            val = Mker[q*KDIM0+k0][k1];
          }
          KWc[(Long)(k0*KDIM1_out+k1)*nq + q] = val*wq[q];
        }
      }
    }
    BENCH_TOC(KernelWeight);

    // Tensor-factored projection M_acc[i*order+j][c] += (Mu . KW_c . MvT)[i][j], per
    // channel c via EvalTensorProduct. Loop runs over C channels (vs ~order GEMMs),
    // so much cheaper when C < order (e.g. scalar kernels). proj is component-major.
    
    BENCH_TIC(Projection);
    thread_local Vector<Real> proj;
    EvalTensorProduct(proj, KWc, Mu, MvT);
    for (Long p = 0; p < nnode; p++)
      for (Integer c = 0; c < C; c++) M_acc[p][c] += proj[(Long)c*nnode + p];
    BENCH_TOC(Projection);
  }

  template <class Real> void QuadElemList<Real>::BuildGraded1D(Vector<Real>& param, Vector<Real>& w, const Real center, const Real b_ellipse, const Vector<Real>& qnds, const Vector<Real>& qwts) {
    // Dyadic subdivision graded toward `center` (geometry-independent), then a
    // QuadOrder GL rule on each leaf.
    const Integer QuadOrder = qnds.Dim();

    Vector<Real> seg; Vector<Long> seg_depth;
    BuildGraded1DSegments(seg, seg_depth, center, b_ellipse);
    const Long nseg = seg_depth.Dim();

    const Long N = nseg * QuadOrder;
    param.ReInit(N);
    w.ReInit(N);
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

  template <class Real> void QuadElemList<Real>::BuildGraded1DSegments(Vector<Real>& seg, Vector<Long>& seg_depth, const Real center, const Real b_ellipse) {
    // Dyadic subdivision of [0,1] graded toward `center`, leaf when admissible. The
    // physical test dist < b_ellipse*L reduces (surface speed cancels) to the
    // geometry-independent parameter test below, so the self subdivision is fixed by
    // (center, b_ellipse) and precomputable. Returns leaf segments + depths.
    constexpr Long MaxLeaves = 4096;

    struct Seg { Real a0, a1; Integer depth; };
    std::vector<Seg> stack, leaves;
    stack.push_back({0, 1, 0});

    while (!stack.empty()) {
      const Seg s = stack.back(); stack.pop_back();

      // Parameter distance from `center` to the segment (clamp into [a0,a1]) vs width.
      const Real pdist = std::fabs(std::min<Real>(s.a1, std::max<Real>(s.a0, center)) - center);
      if (pdist >= b_ellipse*(s.a1-s.a0) || s.depth >= MAX_ADAPTIVE_DEPTH) {
        leaves.push_back(s);
        SCTL_ASSERT((Long)leaves.size() <= MaxLeaves);
      } else {
        const Real am = (s.a0+s.a1)/2;
        stack.push_back({s.a0, am, s.depth+1});
        stack.push_back({am, s.a1, s.depth+1});
      }
    }

    const Long nseg = (Long)leaves.size();
    seg.ReInit(nseg*2);
    seg_depth.ReInit(nseg);
    for (Long i = 0; i < nseg; i++) {
      seg[i*2+0] = leaves[i].a0;
      seg[i*2+1] = leaves[i].a1;
      seg_depth[i] = leaves[i].depth;
    }
  }

  template <class Real> void QuadElemList<Real>::LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
    // Composite/graded rule on [0,1] for a log singularity at interior v0: split at v0, then
    // grade `Lvl` geometric panels per side toward v0 (panels halve in width approaching v0).
    // The panel touching v0 carries the Alpert log endpoint correction; the smooth panels
    // (away from v0) use a plain QuadOrder Gauss-Legendre rule. Grading adds v-resolution
    // near v0, which the sheared-panel v-integrand needs; a single fixed-N panel could not.

    // Alpert correction order fixed at 16 (the max supported by the QuadLog tables).
    const int ord = 16;

    std::vector<double> px, pw;

    // Alpert log/smooth-corrected trapezoidal rule on [a,b]; corr == 2 -> log endpoint, else
    // smooth. Used only on the panel touching v0 (where the log singularity sits).
    auto add_alpert = [&](double a, double b, int corra, int corrb) {
      const ExtraPtResult L = (corra == 2 ? QuadLogExtraPtNodes((double)ord) : QuadSmoothExtraPtNodes((double)ord));
      const ExtraPtResult R = (corrb == 2 ? QuadLogExtraPtNodes((double)ord) : QuadSmoothExtraPtNodes((double)ord));
      const int skipL = L.NodesToSkip, skipR = R.NodesToSkip;

      // Uniform grid with ~2*ord intervals, enough to host both corrections.
      const int N = std::max(skipL + skipR + 2, 2 * ord);
      const int N1 = N - 1;
      const double h = (b - a) / N1;

      // Trapezoidal nodes, dropping the skipped nodes nearest each endpoint.
      for (int i = skipL; i <= N1 - skipR; ++i) {
        px.push_back(a + i * h);
        pw.push_back(h);
      }
      // Left endpoint correction nodes (measured from a).
      for (size_t i = 0; i < L.ExtraNodes.size(); ++i) {
        px.push_back(a + L.ExtraNodes[i] * h);
        pw.push_back(L.ExtraWeights[i] * h);
      }
      // Right endpoint correction nodes (measured from b).
      for (size_t i = 0; i < R.ExtraNodes.size(); ++i) {
        px.push_back(b - R.ExtraNodes[i] * h);
        pw.push_back(R.ExtraWeights[i] * h);
      }
    };

    // Plain Gauss-Legendre rule on a smooth panel [a,b]; the integrand is analytic there.
    Vector<Real> gnds, gwts;
    LegQuadRule<Real>::ComputeNdsWts(&gnds, &gwts, QuadOrder);
    auto add_gl = [&](double a, double b) {
      const double len = b - a;
      for (Integer i = 0; i < QuadOrder; i++) {
        px.push_back(a + len * (double)gnds[i]);
        pw.push_back(len * (double)gwts[i]);
      }
    };

    const double v0d = (double)v0;
    // Left side [0, v0]: smooth GL panels [0, v0*(1-2^-1)], ..., then the Alpert panel
    // [v0*(1-2^-Lvl), v0] touching the singularity.
    {
      double prev = 0.0;
      for (int i = 1; i <= Lvl; i++) {
        const double bnd = v0d * (1.0 - std::ldexp(1.0, -i)); // v0*(1 - 2^-i)
        add_gl(prev, bnd);
        prev = bnd;
      }
      add_alpert(prev, v0d, /*smooth*/ 1, /*log*/ 2); // panel touching v0
    }
    // Right side [v0, 1]: mirror.
    {
      double prev = 1.0;
      for (int i = 1; i <= Lvl; i++) {
        const double bnd = v0d + (1.0 - v0d) * std::ldexp(1.0, -i);
        add_gl(bnd, prev);
        prev = bnd;
      }
      add_alpert(v0d, prev, /*log*/ 2, /*smooth*/ 1); // panel touching v0
    }

    const Long N = (Long)px.size();
    param.ReInit(N);
    w.ReInit(N);
    for (Long i = 0; i < N; ++i) { param[i] = (Real)px[i]; w[i] = (Real)pw[i]; }
  }

  namespace quad_rp { // Bruno-2018 rectangular-polar change-of-variable scalar helpers.

    // Integer power.
    template <class Real> static Real ipow(const Real b, const Integer e) {
      Real r = 1; for (Integer i = 0; i < e; i++) r *= b; return r;
    }

    // v(tau), v'(tau): cubic backbone of the COV (v >= 0 on [0,2*pi] for q>2).
    template <class Real> static Real cov_v(const Real tau, const Integer q) {
      const Real pi = const_pi<Real>();
      const Real a = (1/(Real)q - (Real)0.5);
      const Real t = (pi - tau)/pi;
      return a*t*t*t + (1/(Real)q)*((tau - pi)/pi) + (Real)0.5;
    }
    template <class Real> static Real cov_vp(const Real tau, const Integer q) {
      const Real pi = const_pi<Real>();
      const Real a = (1/(Real)q - (Real)0.5);
      const Real t = (pi - tau)/pi;
      return -(3/pi)*a*t*t + 1/(pi*(Real)q);
    }

    // w(tau), w'(tau): [0,2*pi]->[0,2*pi] map, derivatives 1..q-1 vanish at the
    // endpoints (cf. surface_quadr_schemes.tex).
    template <class Real> static Real cov_w(const Real tau, const Integer q) {
      const Real pi = const_pi<Real>();
      const Real aq = ipow(cov_v(tau, q), q);
      const Real bq = ipow(cov_v(2*pi - tau, q), q);
      return 2*pi*aq/(aq + bq);
    }
    template <class Real> static Real cov_wp(const Real tau, const Integer q) {
      const Real pi = const_pi<Real>();
      const Real va = cov_v(tau, q), vb = cov_v(2*pi - tau, q);
      const Real vpa = cov_vp(tau, q), vpb = cov_vp(2*pi - tau, q);
      const Real aq = ipow(va, q), bq = ipow(vb, q);
      const Real aqm = ipow(va, q-1), bqm = ipow(vb, q-1);
      const Real den = (aq + bq)*(aq + bq);
      return 2*pi*(Real)q*(aqm*bq*vpa + aq*bqm*vpb)/den;
    }

    // xi_alpha(tau), xi'_alpha(tau) on tau in [-1,1], singularity removed at alpha.
    // alpha=+-1 (edge singularity) needs separate branches; generic formula
    // degenerates there, detected with a small tolerance.
    template <class Real> static Real cov_xi(const Real alpha, const Real tau, const Integer q) {
      const Real pi = const_pi<Real>();
      const Real eps = machine_eps<Real>() * 64;
      const Real edge = (Real)1 - eps;
      if (alpha >  edge) return alpha - (1 + alpha)/pi * cov_w(pi*fabs((tau - 1)/2), q);
      if (alpha < -edge) return alpha + (1 - alpha)/pi * cov_w(pi*fabs((tau + 1)/2), q);
      const Real sgn = (tau > 0) ? (Real)1 : ((tau < 0) ? (Real)-1 : (Real)0);
      return alpha + (sgn - alpha)/pi * cov_w(pi*fabs(tau), q);
    }
    template <class Real> static Real cov_xip(const Real alpha, const Real tau, const Integer q) {
      const Real pi = const_pi<Real>();
      const Real eps = machine_eps<Real>() * 64;
      const Real edge = (Real)1 - eps;
      if (alpha >  edge) return (Real)0.5*(1 + alpha)*cov_wp(pi*(1 - tau)/2, q);
      if (alpha < -edge) return (Real)0.5*(1 - alpha)*cov_wp(pi*(1 + tau)/2, q);
      if (tau > 0) return (1 - alpha)*cov_wp( pi*tau, q);
      if (tau < 0) return (1 + alpha)*cov_wp(-pi*tau, q);
      return (Real)0; // tau == 0: derivative vanishes at the singularity.
    }

  } // namespace quad_rp

  template <class Real> void QuadElemList<Real>::RectPolarNodes1D(Vector<Real>& nodes, Vector<Real>& wts, const Real alpha, const Integer q, const Vector<Real>& gl_nds, const Vector<Real>& gl_wts) {
    // Map GL nodes/weights on [0,1] through eta_alpha(u) = (xi_alpha(2u-1)+1)/2. The
    // COV weight xi'_alpha is folded into the weights; it vanishes at the singularity
    // u* = (alpha+1)/2, so the (near-)singular kernel is never evaluated there.
    const Long N = gl_nds.Dim();
    nodes.ReInit(N);
    wts.ReInit(N);
    for (Long i = 0; i < N; i++) {
      const Real tau = 2*gl_nds[i] - 1;
      nodes[i] = (quad_rp::cov_xi(alpha, tau, q) + 1)/2;
      wts[i] = gl_wts[i]*quad_rp::cov_xip(alpha, tau, q);
    }
  }

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Adaptive 2D quadtree for an off-surface target: refine [0,1]^2 into graded leaf
    // panels, integrate each with a QuadOrder GL rule via IntegrateBlock.

    // TODO: only binary split right now, see CSBQ for variable step-size.

    if (qel.NearUsesRectPolar()) { NearInteracBlockRP<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ker); return; }

    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Per-panel GL order / Bernstein parameter fixed at compile time by `digits`;
    // node/weight tables preloaded.
    const Integer QuadOrder = DigitsQuadOrder<digits>();
    const Real b_ellipse = DigitsBEllipse<digits>();
    Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);

    // Leaf panels in parameter space (shared with WriteNearInteracVTK so picture
    // matches solve).
    Vector<Real> leaf_box; Vector<Long> leaf_depth;
    BENCH_TIC(QuadtreeBuild);
    BuildNearLeaves(leaf_box, leaf_depth, qel, elem_idx, Xtrg, b_ellipse);
    BENCH_TOC(QuadtreeBuild);
    const Long nleaf = leaf_depth.Dim();

    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) {
      M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();
    }

    // Integrate each leaf via the tensor-factored IntegrateBlock. NOTE: a "flat"
    // variant that gathers all leaves into one big GEMM was tried (IntegrateBlockFlat)
    // and was ~20x SLOWER: the geometry/projection GEMMs are skinny (contract against
    // COORD_DIM=3 rows / C=1 cols), so they stay memory-bound no matter how large nq
    // gets, while the dense 2D interpolation does ~order x more flops than this
    // separable tensor form. The per-leaf tensor path wins here.
    Vector<Real> u_param(QuadOrder), v_param(QuadOrder), wu(QuadOrder), wv(QuadOrder);
    for (Long li = 0; li < nleaf; li++) {
      const Real pu0 = leaf_box[li*4+0], pu1 = leaf_box[li*4+1];
      const Real pv0 = leaf_box[li*4+2], pv1 = leaf_box[li*4+3];
      const Real du = pu1-pu0, dv = pv1-pv0;
      u_param = pu0 + du*qnds; wu = qwts * du;
      v_param = pv0 + dv*qnds; wv = qwts * dv;
      IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, u_param, wu, v_param, wv, ker);
    }
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Rectangular-polar near-interaction: cluster a single tensor-product GL rule
    // toward the nearest point on the element via the COV, integrate once.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Nbeta GL points per direction for the (finitely smooth) post-COV integrand,
    // decoupled from the field order (Bruno 2018: one to a few hundred). Default 512.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : 512);
    const std::pair<Vector<Real>, Vector<Real>>& gl = GLRuleNbetaDispatch(Nbeta);

    // True closest point (u*,v*) sets the clustering center (alpha = 2*u*-1): bunch
    // nodes at the foot of the perpendicular, not merely the nearest node.
    Real ustar, vstar;
    BENCH_TIC(ClosestPoint);
    qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    BENCH_TOC(ClosestPoint);

    Vector<Real> u_param, wu, v_param, wv;
    RectPolarNodes1D(u_param, wu, 2*ustar - 1, qel.cov_q_, gl.first, gl.second);
    RectPolarNodes1D(v_param, wv, 2*vstar - 1, qel.cov_q_, gl.first, gl.second);

    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero();
    IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, u_param, wu, v_param, wv, ker);
  }

  template <class Real> void QuadElemList<Real>::BuildNearLeaves(Vector<Real>& leaf_box, Vector<Long>& leaf_depth, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse) {
    // Refine [0,1]^2 into graded leaf panels for Xtrg, grading toward the closest
    // point (u*,v*). Returns leaf rectangles ({u0,u1,v0,v1} per leaf) and depths.
    // Extracted from NearInteracBlock so solve and visualization share one refinement.
    constexpr Long MaxLeaves = 4096;

    // Closest point (u*,v*) on the element seeds the grading.
    Real ustar, vstar;
    qel.GetClosestNode(ustar, vstar, elem_idx, Xtrg);

    struct Panel { Real u0, u1, v0, v1; Integer depth; };
    std::vector<Panel> stack, leaves;
    stack.push_back({0, 1, 0, 1, 0});
    Vector<Real> us(1), vs(1), Xc, dXdu, dXdv;
    while (!stack.empty()) {
      const Panel p = stack.back(); stack.pop_back();

      // Distance from target to the panel: clamp (u*,v*) into the rectangle.
      us[0] = std::min<Real>(p.u1, std::max<Real>(p.u0, ustar));
      vs[0] = std::min<Real>(p.v1, std::max<Real>(p.v0, vstar));
      qel.GetGeom(&Xc, nullptr, nullptr, nullptr, nullptr, us, vs, elem_idx);
      Real dist2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { const Real d = Xc[k]-Xtrg[k]; dist2 += d*d; }
      const Real dist = sqrt<Real>(dist2);

      // Physical panel extents from the surface speeds at the panel center.
      us[0] = (p.u0+p.u1)/2; vs[0] = (p.v0+p.v1)/2;
      qel.GetGeom(nullptr, nullptr, nullptr, &dXdu, &dXdv, us, vs, elem_idx);
      Real su2 = 0, sv2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXdu[k]*dXdu[k]; sv2 += dXdv[k]*dXdv[k]; }
      const Real Lu = sqrt<Real>(su2)*(p.u1-p.u0);
      const Real Lv = sqrt<Real>(sv2)*(p.v1-p.v0);

      // Admissible (leaf) when the target is outside the Bernstein neighborhood; a
      // non-finite dist forces a leaf so a degenerate metric cannot loop to depth.
      if (!(dist < b_ellipse*std::max<Real>(Lu, Lv)) || p.depth >= MAX_ADAPTIVE_DEPTH) {
        leaves.push_back(p);
        SCTL_ASSERT((Long)leaves.size() <= MaxLeaves);
        if (p.depth >= MAX_ADAPTIVE_DEPTH) {
          std::cout << "Warning: reached MAX_ADAPTIVE_DEPTH of adaptive refinement. dist - b_ellipse = " << dist - b_ellipse*std::max<Real>(Lu,Lv) << std::endl;
        }
      } else {
        const Real um = (p.u0+p.u1)/2, vm = (p.v0+p.v1)/2;
        stack.push_back({p.u0, um, p.v0, vm, p.depth+1});
        stack.push_back({um, p.u1, p.v0, vm, p.depth+1});
        stack.push_back({p.u0, um, vm, p.v1, p.depth+1});
        stack.push_back({um, p.u1, vm, p.v1, p.depth+1});
      }
    }
    

    const Long nleaf = (Long)leaves.size();
    leaf_box.ReInit(nleaf*4);
    leaf_depth.ReInit(nleaf);
    for (Long i = 0; i < nleaf; i++) {
      leaf_box[i*4+0] = leaves[i].u0;
      leaf_box[i*4+1] = leaves[i].u1;
      leaf_box[i*4+2] = leaves[i].v0;
      leaf_box[i*4+3] = leaves[i].v1;
      leaf_depth[i] = leaves[i].depth;
    }
  }

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Singular self-interaction for on-surface node (ti,tj). 1D reduction: graded u-rule
    // toward u0 x Alpert log-singular v-rule toward v0; both rules + interpolation are
    // preloaded (geometry-independent, fixed by order/ti/tj/digits), integrated by
    // IntegrateBlock. IntegrateBlock still does the target-centered geometry per target.
    if (qel.SelfUsesRectPolar()) { SelfInteracBlockRP<order>(M_acc, qel, elem_idx, ti, tj, Xtrg, normal_trg, ker); return; }

    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const NodeRuleData& ru = SelfURule<order, digits>(ti);  // u: graded rule (per order,ti,digits)
    const NodeRuleData& rv = SelfVRule<order, digits>(tj);  // v: composite Alpert rule (per order,digits,tj)

    M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero();
    IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ru.param, ru.w, rv.param, rv.w, ker, &rv.M, &rv.dM, &ru.M, &ru.dM, &rv.MT, &ru.MT, &ru.dMT);
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::SelfInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Rectangular-polar singular self-interaction for on-surface node (ti,tj). A single
    // tensor-product GL rule clustered toward (u0,v0) in both directions; the COV weight
    // vanishes at the singularity, so no log-singular split is needed. RP is non-adaptive,
    // so both directions are preloaded from RPSelfRule (cached per order,cov_q,Nbeta).
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Nbeta GL points per direction for the (finitely smooth) post-COV integrand,
    // decoupled from the field order (Bruno 2018: one to a few hundred). Default 512.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : 512);
    const NodeRuleData& ru = RPSelfRuleDispatch<order>(ti, qel.cov_q_, Nbeta); // u-direction
    const NodeRuleData& rv = RPSelfRuleDispatch<order>(tj, qel.cov_q_, Nbeta); // v-direction

    M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero();
    IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ru.param, ru.w, rv.param, rv.w, ker, &rv.M, &rv.dM, &ru.M, &ru.dM, &rv.MT, &ru.MT, &ru.dMT);
  }

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self) {
    // On-surface singular self-interaction: every node is an on-element target, built
    // by the singular block. M_lst[e] is (nnode*KDIM0) x (nnode*KDIM1_out), applied as
    // U = F * M_lst[e].
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();

    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;
    const Vector<Real>& nds = ParamNodes(order);

    SCTL_ASSERT((Long)M_lst.Dim() == qel.nelem);

    // Pre-warm the (thread-safe) static rule caches serially: the init lambdas fill all
    // `order` indices in one shot, so the OpenMP loop below never serializes on first-touch
    // static initialization. Warm the rules of the active scheme only.
    if (qel.SelfUsesRectPolar()) {
      const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : 512);
      RPSelfRuleDispatch<order>(0, qel.cov_q_, Nbeta);
    } else {
      SelfURule<order, digits>(0);
      SelfVRule<order, digits>(0);
    }

    // Per-element singular blocks are independent: each writes its own M_lst[elem_idx],
    // all temporaries are loop-local, and GetGeom/rule reads are const. Not nested (SetupSelf
    // is not itself inside an OMP parallel region, unlike SetupNear).
    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < qel.nelem; elem_idx++) {
      // Surface nodes (targets) and their normals on this element.
      Vector<Real> Xnodes, Xnnodes;
      qel.GetGeom(&Xnodes, (trg_dot_prod ? &Xnnodes : nullptr), nullptr, nullptr, nullptr, nds, nds, elem_idx);

      Matrix<Real>& M = M_lst[elem_idx];
      if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nnode*KDIM0, nnode*KDIM1_out);
      M.SetZero();

      for (Integer ti = 0; ti < order; ti++) {
        for (Integer tj = 0; tj < order; tj++) {
          const Long t = ti*order + tj; // target node index = column block

          Vector<Real> Xtrg(COORD_DIM, Xnodes.begin() + t*COORD_DIM, false);
          Vector<Real> ntrg;
          if (trg_dot_prod) ntrg.ReInit(COORD_DIM, Xnnodes.begin() + t*COORD_DIM, false);

          Matrix<Real> M_acc;
          SelfInteracBlock<digits, order>(M_acc, qel, elem_idx, ti, tj, Xtrg, ntrg, ker);

          // Scatter into column block t of M: M[(i*order+j)*KDIM0+k0][t*KDIM1_out+k1].
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) {
              const Long pnode = i*order + j;
              for (Integer k0 = 0; k0 < KDIM0; k0++) {
                for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
                  M[pnode*KDIM0+k0][t*KDIM1_out+k1] = M_acc[pnode][k0*KDIM1_out+k1];
                }
              }
            }
          }
        }
      }
    }
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::SelfInteracDispatchDigits(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    // Map runtime tol to compile-time `digits` (CSBQ-style) so the per-panel quad order
    // and preloaded tables are fixed at compile time per accuracy level.
    if      (tol <= pow<15,Real>((Real)0.1)) SelfInteracHelper<15,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow<14,Real>((Real)0.1)) SelfInteracHelper<14,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow<13,Real>((Real)0.1)) SelfInteracHelper<13,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow<12,Real>((Real)0.1)) SelfInteracHelper<12,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow<11,Real>((Real)0.1)) SelfInteracHelper<11,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow<10,Real>((Real)0.1)) SelfInteracHelper<10,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 9,Real>((Real)0.1)) SelfInteracHelper< 9,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 8,Real>((Real)0.1)) SelfInteracHelper< 8,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 7,Real>((Real)0.1)) SelfInteracHelper< 7,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 6,Real>((Real)0.1)) SelfInteracHelper< 6,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 5,Real>((Real)0.1)) SelfInteracHelper< 5,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 4,Real>((Real)0.1)) SelfInteracHelper< 4,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 3,Real>((Real)0.1)) SelfInteracHelper< 3,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 2,Real>((Real)0.1)) SelfInteracHelper< 2,order>(M_lst, ker, trg_dot_prod, self);
    else if (tol <= pow< 1,Real>((Real)0.1)) SelfInteracHelper< 1,order>(M_lst, ker, trg_dot_prod, self);
    else                                     SelfInteracHelper< 0,order>(M_lst, ker, trg_dot_prod, self);
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    // Dispatch the runtime element order to a compile-time `order` in {4,8,...,48}.
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->order;
    switch (order) {
      case  4: SelfInteracDispatchDigits< 4>(M_lst, ker, tol, trg_dot_prod, self); break;
      case  8: SelfInteracDispatchDigits< 8>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 12: SelfInteracDispatchDigits<12>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 16: SelfInteracDispatchDigits<16>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 20: SelfInteracDispatchDigits<20>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 24: SelfInteracDispatchDigits<24>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 28: SelfInteracDispatchDigits<28>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 32: SelfInteracDispatchDigits<32>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 36: SelfInteracDispatchDigits<36>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 40: SelfInteracDispatchDigits<40>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 44: SelfInteracDispatchDigits<44>(M_lst, ker, tol, trg_dot_prod, self); break;
      case 48: SelfInteracDispatchDigits<48>(M_lst, ker, tol, trg_dot_prod, self); break;
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,...,48} for the templated near/self schemes.");
    }
  }

  template <class Real> void QuadElemList<Real>::EvalPoint(Real* X, Real* dXu, Real* dXv, const Real u, const Real v, const Long elem_idx, const Vector<Real>* origin) const {
    // Single-point evaluation of position (and optional tangents) without any heap
    // allocation. coord is component-major: coord[base + k*nnode + (i*order+j)] with
    // i the u-index, j the v-index (matches GetClosestNode's seed/order, seed%order).
    constexpr Integer MaxOrder = 48; // largest templated element order
    SCTL_ASSERT(order <= MaxOrder);
    const Long nnode = (Long)order * order;
    const Long base = elem_idx * nnode * COORD_DIM;

    // 1D Lagrange value bases Lu_i(u), Lv_j(v) over the GL nodes (stack buffers).
    StaticArray<Real,MaxOrder> Lu, Lv, dLu, dLv;
    { StaticArray<Real,1> up; up[0] = u; Vector<Real> p(1, up, false), o(order, Lu, false); LagrangeInterp<Real>::Interpolate(o, ParamNodes(order), p); }
    { StaticArray<Real,1> vp; vp[0] = v; Vector<Real> p(1, vp, false), o(order, Lv, false); LagrangeInterp<Real>::Interpolate(o, ParamNodes(order), p); }

    // Derivative bases via the cached differentiation matrix: L_i'(u) = sum_a D[i][a] L_a(u)
    // (exact since deg L_i' <= order-1). Only needed when tangents are requested.
    const bool want_d = (dXu || dXv);
    if (want_d) {
      const Matrix<Real>& D = DiffMat(order);
      for (Integer i = 0; i < order; i++) {
        Real su = 0, sv = 0;
        for (Integer a = 0; a < order; a++) { su += D[i][a]*Lu[a]; sv += D[i][a]*Lv[a]; }
        dLu[i] = su; dLv[i] = sv;
      }
    }

    Real x0 = 0, x1 = 0, x2 = 0, du0 = 0, du1 = 0, du2 = 0, dv0 = 0, dv1 = 0, dv2 = 0;
    for (Integer i = 0; i < order; i++) {
      for (Integer j = 0; j < order; j++) {
        const Long p = i*order + j;
        const Real c0 = coord[base + 0*nnode + p], c1 = coord[base + 1*nnode + p], c2 = coord[base + 2*nnode + p];
        const Real wv = Lu[i]*Lv[j];
        x0 += c0*wv; x1 += c1*wv; x2 += c2*wv;
        if (want_d) {
          const Real wu_ = dLu[i]*Lv[j], wvv = Lu[i]*dLv[j];
          du0 += c0*wu_; du1 += c1*wu_; du2 += c2*wu_;
          dv0 += c0*wvv; dv1 += c1*wvv; dv2 += c2*wvv;
        }
      }
    }
    if (origin) { x0 -= (*origin)[0]; x1 -= (*origin)[1]; x2 -= (*origin)[2]; }
    X[0] = x0; X[1] = x1; X[2] = x2;
    if (dXu) { dXu[0] = du0; dXu[1] = du1; dXu[2] = du2; }
    if (dXv) { dXv[0] = dv0; dXv[1] = dv1; dXv[2] = dv2; }
  }

  template <class Real> Real QuadElemList<Real>::GetClosestNode(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) const {
    const auto& nds = ParamNodes(order);
    const Long nnode = (Long)order * order;

    // Brute-force seed over the order x order nodal grid. The param nodes ARE the
    // element's stored nodes, so read coord directly (no interpolation / normals).
    const Long base = elem_idx * nnode * COORD_DIM;
    Long seed = 0;
    Real best = -1;
    for (Long p = 0; p < nnode; p++) {
      Real r2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real d = coord[base + k*nnode + p] - Xtrg[k];
        r2 += d*d;
      }
      if (best < 0 || r2 < best) { best = r2; seed = p; }
    }

    ustar = nds[seed/order];
    vstar = nds[seed%order];

    return sqrt<Real>(best);
  }

  template <class Real> Real QuadElemList<Real>::GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) const {
    // Closest point on patch to Xtrg over (u,v) in [0,1]^2. Minimize 1/2|y-x|^2 by
    // Gauss-Newton (first fundamental form), seeded by the nearest node, clamped with
    // backtracking; shrinking-box grid search is the fallback if Newton stalls.

    // r^2 at (u,v). Target-centering (origin = Xtrg) keeps the residual accurate near
    // the surface, locating the foot sharply for near-touching targets.
    auto dist2_at = [&](const Real uu, const Real vv) -> Real {
      Real X[COORD_DIM];
      EvalPoint(X, nullptr, nullptr, uu, vv, elem_idx, &Xtrg);
      Real r2 = 0; for (Integer k = 0; k < COORD_DIM; k++) r2 += X[k]*X[k];
      return r2;
    };

    // Seed: nearest node.
    Real u, v;
    Real f = GetClosestNode(u, v, elem_idx, Xtrg);

    // Gauss-Newton with clamping and backtracking line search.
    constexpr Integer max_iter = 30;
    const Real utol = (Real)machine_eps<Real>() * 64;
    bool converged = false;
    for (Integer it = 0; it < max_iter; it++) {
      Real X[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
      EvalPoint(X, dXu, dXv, u, v, elem_idx, &Xtrg); // X = y(u,v) - Xtrg

      // gradient g = [r.y_u, r.y_v], metric (first fundamental form) [[E,F],[F,G]].
      Real E = 0, F = 0, G = 0, gu = 0, gv = 0;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real r = X[k], a = dXu[k], b = dXv[k];
        E += a*a; F += a*b; G += b*b;
        gu += r*a; gv += r*b;
      }

      // Gauss-Newton step d = metric^{-1} g (fall back to scaled gradient if the
      // metric is degenerate, e.g. at a patch corner).
      const Real det = E*G - F*F;
      Real du, dv;
      if (fabs(det) > (Real)1e-30 * (E*G + F*F + 1)) {
        du = ( G*gu - F*gv) / det;
        dv = (-F*gu + E*gv) / det;
      } else {
        du = gu / (E + (Real)1e-30);
        dv = gv / (G + (Real)1e-30);
      }

      // Backtrack on the clamped step until f decreases.
      Real lambda = 1;
      bool improved = false;
      Real un = u, vn = v, fn = f;
      for (Integer ls = 0; ls < 40; ls++) {
        un = std::min<Real>(1, std::max<Real>(0, u - lambda*du));
        vn = std::min<Real>(1, std::max<Real>(0, v - lambda*dv));
        fn = dist2_at(un, vn);
        if (fn < f) { improved = true; break; }
        lambda *= (Real)0.5;
      }
      if (!improved) break; // stalled: leave converged=false to trigger the fallback
      const bool small_step = (fabs(un-u) < utol && fabs(vn-v) < utol);
      u = un; v = vn; f = fn;
      if (small_step) { converged = true; break; }
    }

    // Fallback: shrinking-box grid search over the whole patch. Robust to a poor
    // Newton seed / non-convex patch; keeps whichever point is closer. The shrink
    // factor (~2/K per level) hits utol well before the level cap, so a modest cap
    // suffices. Uses the allocation-free point evaluator.
    if (!converged) {
      constexpr Integer K = 8, levels = 25;
      Real u0 = 0, u1 = 1, v0 = 0, v1 = 1;
      for (Integer L = 0; L < levels; L++) {
        for (Integer i = 0; i <= K; i++) {
          const Real ui = u0 + (u1-u0)*i/(Real)K;
          for (Integer j = 0; j <= K; j++) {
            const Real vj = v0 + (v1-v0)*j/(Real)K;
            const Real r2 = dist2_at(ui, vj);
            if (r2 < f) { f = r2; u = ui; v = vj; }
          }
        }
        const Real hu = (u1-u0)/K, hv = (v1-v0)/K;
        u0 = std::max<Real>(0, u-hu); u1 = std::min<Real>(1, u+hu);
        v0 = std::max<Real>(0, v-hv); v1 = std::min<Real>(1, v+hv);
        if ((u1-u0) < utol && (v1-v0) < utol) break;
      }
    }

    ustar = u; vstar = v;
    return sqrt<Real>(f);
  }

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self) {
    // Per-target near-singular interaction: off-surface targets are integrated by the
    // adaptive 2D quadtree (NearInteracBlock). On-surface singular self interactions
    // proper are built by SelfInterac.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();

    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const Long Ntrg = Xt.Dim() / COORD_DIM;
    if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != Ntrg*KDIM1_out) {
      M.ReInit(nnode*KDIM0, Ntrg*KDIM1_out);
    }
    M.SetZero();
    if (!Ntrg) return;

    for (Long t = 0; t < Ntrg; t++) {
      Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
      Vector<Real> ntrg;
      if (trg_dot_prod) ntrg.ReInit(COORD_DIM, (Iterator<Real>)normal_trg.begin() + t*COORD_DIM, false);

      Matrix<Real> M_acc;
      NearInteracBlock<digits, order>(M_acc, qel, elem_idx, Xtrg, ntrg, ker);

      // Scatter into M for target t: M[(i*order+j)*KDIM0+k0][t*KDIM1_out+k1].
      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          const Long pnode = i*order + j;
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
              M[pnode*KDIM0+k0][t*KDIM1_out+k1] = M_acc[pnode][k0*KDIM1_out+k1];
            }
          }
        }
      }
    }
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracDispatchDigits(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    // Map runtime tol to compile-time `digits` (CSBQ-style) so the per-panel quad order
    // and preloaded tables are fixed at compile time per accuracy level.
    if      (tol <= pow<15,Real>((Real)0.1)) NearInteracHelper<15,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow<14,Real>((Real)0.1)) NearInteracHelper<14,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow<13,Real>((Real)0.1)) NearInteracHelper<13,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow<12,Real>((Real)0.1)) NearInteracHelper<12,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow<11,Real>((Real)0.1)) NearInteracHelper<11,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow<10,Real>((Real)0.1)) NearInteracHelper<10,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 9,Real>((Real)0.1)) NearInteracHelper< 9,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 8,Real>((Real)0.1)) NearInteracHelper< 8,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 7,Real>((Real)0.1)) NearInteracHelper< 7,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 6,Real>((Real)0.1)) NearInteracHelper< 6,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 5,Real>((Real)0.1)) NearInteracHelper< 5,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 4,Real>((Real)0.1)) NearInteracHelper< 4,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 3,Real>((Real)0.1)) NearInteracHelper< 3,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 2,Real>((Real)0.1)) NearInteracHelper< 2,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else if (tol <= pow< 1,Real>((Real)0.1)) NearInteracHelper< 1,order>(M, Xt, normal_trg, ker, elem_idx, self);
    else                                     NearInteracHelper< 0,order>(M, Xt, normal_trg, ker, elem_idx, self);
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    // Dispatch the runtime element order to a compile-time `order` in {4,8,...,48}.
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->order;
    switch (order) {
      case  4: NearInteracDispatchDigits< 4>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case  8: NearInteracDispatchDigits< 8>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 12: NearInteracDispatchDigits<12>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 16: NearInteracDispatchDigits<16>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 20: NearInteracDispatchDigits<20>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 24: NearInteracDispatchDigits<24>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 28: NearInteracDispatchDigits<28>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 32: NearInteracDispatchDigits<32>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 36: NearInteracDispatchDigits<36>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 40: NearInteracDispatchDigits<40>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 44: NearInteracDispatchDigits<44>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      case 48: NearInteracDispatchDigits<48>(M, Xt, normal_trg, ker, tol, elem_idx, self); break;
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,...,48} for the templated near/self schemes.");
    }
  }

  template <class Real> const Vector<Real>& QuadElemList<Real>::ParamNodes(const Integer Order) {
    return LegQuadRule<Real>::nds(Order);
  }

  template <class Real> const Vector<Real>& QuadElemList<Real>::ParamGrid(const Integer Order, const Integer Nelem_perside) {
    const Vector<Real> nodes = ParamNodes(Order);

    Vector<Real> x_param(Order * Nelem_perside);
    for (int pind=0; pind < Nelem_perside; pind ++) {
        for (int nind=0; nind < Order; nind ++) {
            x_param[pind * Order + nind] = (nodes[nind] + pind) / Nelem_perside; // TODO check
        }
    }
    static Vector<Real> coord0;
    coord0.ReInit(x_param.Dim() * x_param.Dim() * COORD_DIM); // resize every call (the static ctor runs only once)
    for (int xind=0; xind < x_param.Dim(); xind ++) {
        for (int yind=0; yind < x_param.Dim(); yind ++) {
            const Long idx = xind * x_param.Dim() * COORD_DIM + yind * COORD_DIM;
            coord0[idx + 0] = x_param[xind];
            coord0[idx + 1] = x_param[yind];
            coord0[idx + 2] = 0.; 
        }
    }
    return coord0;
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
    const Long Nelem_total = coord_.Dim() / (COORD_DIM * nnode_per_elem);
    SCTL_ASSERT(coord_.Dim() == Nelem_total * COORD_DIM * nnode_per_elem);

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
      const Long base = elem_idx * COORD_DIM * nnode_per_elem;
      for (Long p = 0; p < nnode_per_elem; p++) {
        for (Integer k = 0; k < COORD_DIM; k++) {
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
      for (Integer k = 0; k < COORD_DIM; k++) {
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

    // Determine order from the first element marker and verify uniformity.
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
      PartitionRange(Nelem_total, comm, i0, i1);

      const Long j0 = i0 * nnode_per_elem;
      const Long j1 = i1 * nnode_per_elem;

      Vector<ValueType> coord_local;
      coord_local.ReInit((j1 - j0) * COORD_DIM, coord_.begin() + j0 * COORD_DIM, false);
      // Slice already local to this rank; pass Comm::Self() so Init does not re-partition.
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
      EvalTensorProduct(F_soa_eval, F_soa, MuT, Mv);

      Fgrid.ReInit(Nu * Nv * dof);
      for (Long p = 0; p < Nu * Nv; p++) {
        for (Long k = 0; k < dof; k++) {
          Fgrid[p * dof + k] = F_soa_eval[k * (Nu * Nv) + p];
        }
      }
    }

    const Long point_offset = vtu_data.coord.Dim() / COORD_DIM;
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

  template <class Real> void QuadElemList<Real>::WriteNearInteracVTK(const std::string& fname, const Long elem_idx, const Vector<Real>& Xtrg, const Real tol, const Comm& comm) const {
    // Reconstruct the adaptive near quadtree for Xtrg and dump the per-leaf GL nodes
    // as a VTK_QUAD mesh (colored by depth). Target written to a separate file.
    Real b_ellipse; Integer QuadOrder;
    QuadParams(tol, b_ellipse, QuadOrder);

    Vector<Real> leaf_box; Vector<Long> leaf_depth;
    BuildNearLeaves(leaf_box, leaf_depth, *this, elem_idx, Xtrg, b_ellipse);
    const Long nleaf = leaf_depth.Dim();

    VTUData vtu;
    {
      const Vector<Real>& qnds = ParamNodes(QuadOrder);
      Vector<Real> u_param(QuadOrder), v_param(QuadOrder), Xg;
      for (Long li = 0; li < nleaf; li++) {
        const Real u0 = leaf_box[li*4+0], u1 = leaf_box[li*4+1];
        const Real v0 = leaf_box[li*4+2], v1 = leaf_box[li*4+3];
        const Real du = u1-u0, dv = v1-v0;

        // Per-leaf GL node grid -> VTK_QUAD cells; point_offset resets per leaf so
        // cells never bridge two leaves.
        for (Integer a = 0; a < QuadOrder; a++) u_param[a] = u0 + du*qnds[a];
        for (Integer b = 0; b < QuadOrder; b++) v_param[b] = v0 + dv*qnds[b];
        GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, u_param, v_param, elem_idx);

        const Long point_offset = vtu.coord.Dim() / COORD_DIM;
        for (const auto& x : Xg) vtu.coord.PushBack((VTUData::VTKReal)x);

        for (Long i = 0; i < QuadOrder - 1; i++) {
          for (Long j = 0; j < QuadOrder - 1; j++) {
            const Long idx = point_offset + i * QuadOrder + j;
            vtu.connect.PushBack(idx);
            vtu.connect.PushBack(idx + 1);
            vtu.connect.PushBack(idx + QuadOrder + 1);
            vtu.connect.PushBack(idx + QuadOrder);
            vtu.offset.PushBack(vtu.connect.Dim());
            vtu.types.PushBack(9);
          }
        }
      }
    }
    vtu.WriteVTK(fname, comm);

    // Target: a single VTK_VERTEX in its own file.
    VTUData target;
    for (Integer k = 0; k < COORD_DIM; k++) target.coord.PushBack((VTUData::VTKReal)Xtrg[k]);
    target.value.PushBack(0);
    target.connect.PushBack(0);
    target.offset.PushBack(target.connect.Dim());
    target.types.PushBack(1);
    target.WriteVTK(fname + "-target", comm);
  }

  template <class Real> void QuadElemList<Real>::WriteSelfInteracVTK(const std::string& fname, const Long elem_idx, const Real u0, const Real v0, const Real tol, const Comm& comm) const {
    // Reconstruct the on-surface self-interaction structure at (u0,v0): graded
    // u-refinement x 1D Alpert log-singular v-rule. Dumps the tensor nodes as a
    // VTK_VERTEX cloud (nodes not monotonically ordered, so no index mesh). Singular
    // point in a separate file.
    Real b_ellipse; Integer QuadOrder;
    QuadParams(tol, b_ellipse, QuadOrder);

    // On-surface target at (u0,v0) seeds the graded u-refinement.
    Vector<Real> us0(1), vs0(1), Xtrg;
    us0[0] = u0; vs0[0] = v0;
    GetGeom(&Xtrg, nullptr, nullptr, nullptr, nullptr, us0, vs0, elem_idx);

    VTUData vtu;
    {
      // Tensor product of graded-GL u nodes and Alpert v nodes (the rule
      // SelfInteracBlock integrates) -> VTK_VERTEX.
      Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);
      // Composite-graded v-rule levels + matched u-depth from tol (matches SelfVRule/SelfURule).
      const Integer digits = std::max<Integer>(0, (Integer)std::lround(-std::log10((double)std::max<Real>(tol, machine_eps<Real>()))));
      const Integer Lvl = VLevelsForDigits(digits);
      Vector<Real> u_param, wu, v_param, wv, Xg;
      BuildGraded1D(u_param, wu, u0, b_ellipse, qnds, qwts);
      LogSingularQuad1D(v_param, wv, v0, Lvl, QuadOrder);
      GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, u_param, v_param, elem_idx);
      const Long nq = u_param.Dim()*v_param.Dim();
      for (Long q = 0; q < nq; q++) {
        const Long idx = vtu.coord.Dim()/COORD_DIM;
        for (Integer k = 0; k < COORD_DIM; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[q*COORD_DIM+k]);
        vtu.connect.PushBack((int32_t)idx);
        vtu.offset.PushBack(vtu.connect.Dim());
        vtu.types.PushBack(1); // VTK_VERTEX
      }
    }
    vtu.WriteVTK(fname, comm);

    // Singular point: a single VTK_VERTEX at the on-surface target (u0,v0).
    VTUData singpt;
    for (Integer k = 0; k < COORD_DIM; k++) singpt.coord.PushBack((VTUData::VTKReal)Xtrg[k]);
    singpt.value.PushBack(0);
    singpt.connect.PushBack(0);
    singpt.offset.PushBack(singpt.connect.Dim());
    singpt.types.PushBack(1);
    singpt.WriteVTK(fname + "-singpt", comm);
  }

  template <class Real> void QuadElemList<Real>::WriteRectPolarGridVTK(const std::string& fname, const Long elem_idx, const Real ustar, const Real vstar, const Integer Nbeta) const {
    // Shared RP visualizer core: push an Nbeta x Nbeta GL grid through the COV
    // (clustering toward (u*,v*)) and dump as a VTK_QUAD mesh. COV is monotone per
    // direction, so the tensor grid meshes cleanly.
    Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, Nbeta);
    Vector<Real> u_param, wu, v_param, wv, Xg;
    RectPolarNodes1D(u_param, wu, 2*ustar - 1, cov_q_, qnds, qwts);
    RectPolarNodes1D(v_param, wv, 2*vstar - 1, cov_q_, qnds, qwts);
    GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, u_param, v_param, elem_idx);

    VTUData vtu;
    for (const auto& x : Xg) vtu.coord.PushBack((VTUData::VTKReal)x);
    for (Long i = 0; i < Nbeta - 1; i++) {
      for (Long j = 0; j < Nbeta - 1; j++) {
        const Long idx = i*Nbeta + j;
        vtu.connect.PushBack(idx);
        vtu.connect.PushBack(idx + 1);
        vtu.connect.PushBack(idx + Nbeta + 1);
        vtu.connect.PushBack(idx + Nbeta);
        vtu.offset.PushBack(vtu.connect.Dim());
        vtu.types.PushBack(9); // VTK_QUAD
      }
    }
    vtu.WriteVTK(fname, Comm::Self());
  }

  template <class Real> void QuadElemList<Real>::WriteNearInteracRPVTK(const std::string& fname, const Long elem_idx, const Vector<Real>& Xtrg, const Integer Nbeta, const Comm& comm) const {
    // RP near-interaction grid: cluster toward the closest point (same (u*,v*) as
    // NearInteracBlockRP). Grid in `<fname>`, target in `<fname>-target`.
    Real ustar, vstar;
    GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    WriteRectPolarGridVTK(fname, elem_idx, ustar, vstar, Nbeta);

    VTUData target;
    for (Integer k = 0; k < COORD_DIM; k++) target.coord.PushBack((VTUData::VTKReal)Xtrg[k]);
    target.value.PushBack(0);
    target.connect.PushBack(0);
    target.offset.PushBack(target.connect.Dim());
    target.types.PushBack(1); // VTK_VERTEX
    target.WriteVTK(fname + "-target", comm);
  }

  template <class Real> void QuadElemList<Real>::WriteSelfInteracRPVTK(const std::string& fname, const Long elem_idx, const Real u0, const Real v0, const Integer Nbeta, const Comm& comm) const {
    // RP self-interaction grid: cluster toward the on-surface target (u0,v0).
    // Grid in `<fname>`, singular point in `<fname>-singpt`.
    WriteRectPolarGridVTK(fname, elem_idx, u0, v0, Nbeta);

    Vector<Real> us0(1), vs0(1), Xtrg;
    us0[0] = u0; vs0[0] = v0;
    GetGeom(&Xtrg, nullptr, nullptr, nullptr, nullptr, us0, vs0, elem_idx);
    VTUData singpt;
    for (Integer k = 0; k < COORD_DIM; k++) singpt.coord.PushBack((VTUData::VTKReal)Xtrg[k]);
    singpt.value.PushBack(0);
    singpt.connect.PushBack(0);
    singpt.offset.PushBack(singpt.connect.Dim());
    singpt.types.PushBack(1); // VTK_VERTEX
    singpt.WriteVTK(fname + "-singpt", comm);
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
  }

}

#endif // _SCTL_QUAD_ELEMENT_CPP_
