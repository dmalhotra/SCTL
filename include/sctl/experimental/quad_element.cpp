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
      for (Integer n = 2; n <= MAX_ORDER; n++) {
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
    if      (Nbeta == 48)  return GLRuleNbeta<48>();
    else if (Nbeta == 100)  return GLRuleNbeta<100>();
    else if (Nbeta == 200)  return GLRuleNbeta<200>();
    else if (Nbeta == 300)  return GLRuleNbeta<300>();
    else if (Nbeta == 400)  return GLRuleNbeta<400>();
    else if (Nbeta == 512) return GLRuleNbeta<512>();
    SCTL_ASSERT_MSG(false, "RectPolar Nbeta (cov_order) must be one of {48, 100, 200, 300, 400, 512}.");
  }

  template <class Real> template <Integer order, Integer Nbeta, Integer q> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::RPSelfRule(const Integer k) {
    // Self-RP COV rule + interpolation for the singularity at nds[m] (same rule serves
    // u and v). Geometry-independent (fixed COV), so cached once per (order, Nbeta, q).
    static const Vector<NodeRuleData> data = []() {
      const Vector<Real>& nds = ParamNodes(order);
      const std::pair<Vector<Real>, Vector<Real>>& gl = GLRuleNbetaDispatch(Nbeta);
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
      if      (Nbeta == 48)  return RPSelfRule<order,48,6>(k);
      else if (Nbeta == 100)  return RPSelfRule<order,100,6>(k);
      else if (Nbeta == 200)  return RPSelfRule<order,200,6>(k);
      else if (Nbeta == 300)  return RPSelfRule<order,300,6>(k);
      else if (Nbeta == 400)  return RPSelfRule<order,400,6>(k);
      else if (Nbeta == 512) return RPSelfRule<order,512,6>(k);
    } else if (q == 10) {
      if      (Nbeta == 48)  return RPSelfRule<order,48,10>(k);
      else if (Nbeta == 100)  return RPSelfRule<order,100,10>(k);
      else if (Nbeta == 200)  return RPSelfRule<order,200,10>(k);
      else if (Nbeta == 300)  return RPSelfRule<order,300,10>(k);
      else if (Nbeta == 400)  return RPSelfRule<order,400,10>(k);
      else if (Nbeta == 512) return RPSelfRule<order,512,10>(k);
    }
    SCTL_ASSERT_MSG(false, "RectPolar (cov_q, Nbeta) must have cov_q in {6,10} and Nbeta in {48,100,200,300,400,512}.");
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

  template <class Real> template <Integer digits> const std::pair<Vector<Real>, Vector<Real>>& QuadElemList<Real>::DigitsGLRule() {
    // Per-panel GL rule for `digits`, built once. LegQuadRule::ComputeNdsWts is an uncached
    // O(N^2) Newton solve, so the adaptive near path must not call it per (element,target).
    static const std::pair<Vector<Real>, Vector<Real>> gl = []() {
      std::pair<Vector<Real>, Vector<Real>> p;
      LegQuadRule<Real>::ComputeNdsWts(&p.first, &p.second, DigitsQuadOrder<digits>());
      return p;
    }();
    return gl;
  }

  template <class Real> Integer QuadElemList<Real>::VLevelsForDigits(const Integer digits) {
    // Geometric grading levels per side toward v0 in the composite Alpert v-rule.
    return std::min<Integer>(12, std::max<Integer>(1, digits - 5));
  }

  template <class Real> template <Integer digits> Integer QuadElemList<Real>::DigitsVLevels() {
    return VLevelsForDigits(digits);
  }

  template <class Real> Integer QuadElemList<Real>::NbetaForDigits(const Integer digits) {
    // Worst-case tol->Nbeta ladder for the RectPolar COV (the cov_order_==0 fallback), rounded to
    // a supported dispatch Nbeta {48,100,200,300,400,512}. Calibrated on the maximally twisted
    // sphere (theta=pi, PatchPerFace=5, Nbeta_sweep.txt); RectPolar converges much faster in Nbeta
    // on near-flat geometry, so this is conservative there. A user-set cov_order overrides it.
    if      (digits <= 2) return 100; // 1e-1..1e-2 (below the old 128 calibration point; re-verify if tight 1e-2 on strong twist matters)
    else if (digits == 3) return 300; // 1e-3
    else if (digits <= 5) return 400; // 1e-4,1e-5
    else                  return 512; // <=1e-6 (ladder max)
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker, const Matrix<Real>* Mv_pre, const Matrix<Real>* dMv_pre, const Matrix<Real>* Mu_pre, const Matrix<Real>* dMu_pre, const Matrix<Real>* MvT_pre, const Matrix<Real>* MuT_pre, const Matrix<Real>* dMuT_pre, const Vector<Real>* src_nodal, const Matrix<Real>* MuD_pre, const Real nrm_sign, Vector<Real>* acc_cm) {
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
    }
    const Matrix<Real>& Mu  = (Mu_pre  ? *Mu_pre  : Mu_local);
    const Matrix<Real>& MuT  = (MuT_pre  ? *MuT_pre  : MuT_local);
    const Matrix<Real>& dMuT = (dMuT_pre ? *dMuT_pre : dMuT_local);
    const Matrix<Real>& Mv  = (Mv_pre  ? *Mv_pre  : Mv_local);
    const Matrix<Real>& dMv = (dMv_pre ? *dMv_pre : dMv_local);
    const Matrix<Real>& MvT  = (MvT_pre  ? *MvT_pre  : MvT_local);

    // Target-centering: subtract Xtrg from nodal coords before interpolation so
    // positions are source-minus-target (accurate r near the singularity); tangents
    // come from the same shifted slab.
    BENCH_TIC(GeomTensor);
    const Long base = elem_idx * nnode * COORD_DIM; // TODO: assumes uniform per-element grid; consider omp scan of elem_cnt.
    // thread_local scratch reused across the many IntegrateBlock calls (fully overwritten
    // before use, so reuse is safe; avoids re-malloc churn). The shifted slab is memoized on
    // (qel, elem_idx, Xtrg) so repeated calls sharing a target (e.g. near cells) skip the shift.
    thread_local Vector<Real> coord_shift;
    thread_local const QuadElemList<Real>* cs_qel = nullptr;
    thread_local Long cs_elem = -1;
    thread_local StaticArray<Real,COORD_DIM> cs_trg{0,0,0};
    if (coord_shift.Dim() != COORD_DIM*nnode) { coord_shift.ReInit(COORD_DIM*nnode); cs_qel = nullptr; }
    // src_nodal (when non-null): caller already produced the target-shifted slab, so bind a view.
    if (!src_nodal && (cs_qel != &qel || cs_elem != elem_idx || cs_trg[0] != Xtrg[0] || cs_trg[1] != Xtrg[1] || cs_trg[2] != Xtrg[2])) {
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) coord_shift[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
      }
      cs_qel = &qel; cs_elem = elem_idx;
      for (Integer k = 0; k < COORD_DIM; k++) cs_trg[k] = Xtrg[k];
    }
    const Vector<Real>& cs_ref = (src_nodal ? *src_nodal : coord_shift);
    // The v-side contraction is u-independent, so hoist it (X and dXdu both use Mv). All
    // COORD_DIM components share Mv and coord_shift is component-major contiguous, so the three
    // (order x order).(order x Nv) products fuse into one (COORD_DIM*order x order) GEMM.
    thread_local Vector<Real> Cv, Cdv;
    if (Cv.Dim() != COORD_DIM*order*Nv) { Cv.ReInit(COORD_DIM*order*Nv); Cdv.ReInit(COORD_DIM*order*Nv); }
    {
      const Matrix<Real> cs_all(COORD_DIM*order, order, (Iterator<Real>)cs_ref.begin(), false); // component-major: one GEMM for all 3
      Matrix<Real> Cv_all (COORD_DIM*order, Nv, Cv.begin(),  false);
      Matrix<Real> Cdv_all(COORD_DIM*order, Nv, Cdv.begin(), false);
      Matrix<Real>::GEMM(Cv_all,  cs_all, Mv);
      Matrix<Real>::GEMM(Cdv_all, cs_all, dMv);
      BENCH_FLOPS(2.0 * 2 * (COORD_DIM*(double)order) * order * Nv);
    }
    static const Long ublk_pts_ = []() { const char* v = std::getenv("SCTL_UBLK_PTS"); return v ? std::max<Long>(64, atol(v)) : 16384; }();
    if (Nu * Nv <= ublk_pts_) { // Sweep already fits: original single-shot path.
      // Near integrates tiny per-leaf blocks (Nu = Nv = QuadOrder), where blocking buys
      // nothing -- and the batched/per-leaf near gate requires this path bit-for-bit, so
      // it must keep using the same buffers and GEMM calls.
      // Column-stage Cv/Cdv (component index moved into the COLUMNS) so stage 2 batches over
      // components as well as over outputs: the nine original (Nu x order).(order x Nv) products
      // collapse to two GEMMs against an (order x COORD_DIM*Nv) operand. The restage is an
      // L1-resident copy -- Matrix::GEMM has no strided-output form, and padding Mv block-diagonal
      // instead would triple the stage-1 flops.
      const Long ldc = COORD_DIM*Nv;
      thread_local Vector<Real> Cvc, Cdvc, XdU, dXdv_soa;
      if (Cvc.Dim() != (Long)order*ldc) { Cvc.ReInit((Long)order*ldc); Cdvc.ReInit((Long)order*ldc); }
      for (Integer k = 0; k < COORD_DIM; k++) {
        for (Integer i = 0; i < order; i++) {
          const Long src = ((Long)k*order + i)*Nv, dst = (Long)i*ldc + k*Nv;
          for (Long b = 0; b < Nv; b++) { Cvc[dst+b] = Cv[src+b]; Cdvc[dst+b] = Cdv[src+b]; }
        }
      }
      if (XdU.Dim() != 2*(Long)Nu*ldc) { XdU.ReInit(2*(Long)Nu*ldc); dXdv_soa.ReInit((Long)Nu*ldc); }
      {
        const Matrix<Real> Cvc_m(order, ldc, Cvc.begin(), false), Cdvc_m(order, ldc, Cdvc.begin(), false);
        Matrix<Real> dV_m(Nu, ldc, dXdv_soa.begin(), false);
        if (MuD_pre) { // [T^T; dT^T] -> X and dXdu in one GEMM
          Matrix<Real> XdU_m(2*Nu, ldc, XdU.begin(), false);
          Matrix<Real>::GEMM(XdU_m, *MuD_pre, Cvc_m);
        } else {
          Matrix<Real> X_m(Nu, ldc, XdU.begin(), false), dU_m(Nu, ldc, XdU.begin() + (Long)Nu*ldc, false);
          Matrix<Real>::GEMM(X_m,  MuT,  Cvc_m);
          Matrix<Real>::GEMM(dU_m, dMuT, Cvc_m);
        }
        Matrix<Real>::GEMM(dV_m, MuT, Cdvc_m);
        BENCH_FLOPS(2.0 * (3.0*Nu) * order * ldc);
      }
      BENCH_TOC(GeomTensor);

      StaticArray<Real,COORD_DIM> Xt0_{0, 0, 0};
      const Vector<Real> Xt0_v_(COORD_DIM, Xt0_, false);
      BENCH_TIC(Assembly);
      thread_local Vector<Real> Xsrc, Xnsrc, wq;
      if (Xsrc.Dim() != nq*COORD_DIM) { Xsrc.ReInit(nq*COORD_DIM); Xnsrc.ReInit(nq*COORD_DIM); wq.ReInit(nq); }
      for (Long a = 0; a < Nu; a++) {
        for (Long b = 0; b < Nv; b++) {
          const Long q = a*Nv + b;
          const Long r = (Long)a*ldc + b, ru = ((Long)Nu + a)*ldc + b;
          const Real du0 = XdU[ru+0*Nv], du1 = XdU[ru+1*Nv], du2 = XdU[ru+2*Nv];
          const Real dv0 = dXdv_soa[r+0*Nv], dv1 = dXdv_soa[r+1*Nv], dv2 = dXdv_soa[r+2*Nv];
          const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
          const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
          // nrm_sign flips the normal when exactly one direction is mirrored: the tangents are
          // then d/dx (sub-element coords), whose cross product is anti-parallel to dXu x dXv.
          const Real inv_area = (area > 0 ? nrm_sign/area : 0);
          Xsrc[q*COORD_DIM+0] = XdU[r+0*Nv]; Xsrc[q*COORD_DIM+1] = XdU[r+1*Nv]; Xsrc[q*COORD_DIM+2] = XdU[r+2*Nv];
          Xnsrc[q*COORD_DIM+0] = n0*inv_area; Xnsrc[q*COORD_DIM+1] = n1*inv_area; Xnsrc[q*COORD_DIM+2] = n2*inv_area;
          wq[q] = area*wu[a]*wv[b];
        }
      }
      BENCH_TOC(Assembly);

      BENCH_TIC(KernelEval);
      thread_local Matrix<Real> Mker;
      ker.template KernelMatrix<Real,false>(Mker, Xt0_v_, Xsrc, Xnsrc); // (nq*KDIM0 x KDIM1full)
      BENCH_TOC(KernelEval);

      BENCH_TIC(KernelWeight);
      thread_local Vector<Real> KWc;
      if (KWc.Dim() != C*nq) KWc.ReInit(C*nq);
      for (Long q = 0; q < nq; q++) {
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
            Real val;
            if (trg_dot_prod) {
              val = 0;
              for (Integer l = 0; l < COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * normal_trg[l];
            } else {
              val = Mker[q*KDIM0+k0][k1];
            }
            KWc[(Long)(k0*KDIM1_out+k1)*nq + q] = val*wq[q];
          }
        }
      }
      BENCH_TOC(KernelWeight);

      BENCH_TIC(Projection);
      // Projection is the adjoint of the geometry interpolation: quadrature -> nodal. The
      // v-contraction batches over ALL C channels for free -- KWc is channel-major with each
      // block (Nu x Nv), so it is already one (C*Nu x Nv) operand. The u-contraction then writes
      // one (order x order) block per channel, which is exactly a channel-major accumulator's
      // layout, so with acc_cm it accumulates in place via beta = 1 (no temporary, no += sweep;
      // the caller transposes to M_acc's node-major layout once per target instead of per cell).
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
      BENCH_FLOPS(2.0 * C * order * ((double)Nu*Nv + (double)order*Nu));
      BENCH_TOC(Projection);
      return;
    }

    BENCH_TOC(GeomTensor);

    // Sources are target-relative -> kernel target at the origin Xt0 = 0.
    StaticArray<Real,COORD_DIM> Xt0{0, 0, 0};
    const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);

    // u-BLOCKING. Unblocked, the per-target scratch is 18*nq doubles (~9 MB at nq=62k),
    // which exceeds the L3 each thread gets once several threads share a CCX. Sweeping u
    // in blocks keeps the live set at 18*UBLK*Nv and leaves the flop count unchanged:
    // the u-rows of the geometry are independent, and the projection is a sum over u, so
    // only its (Nu x order) intermediate spans blocks (42 KB at order 8).
    const Long UBLK = std::max<Long>(1, std::min<Long>(Nu, ublk_pts_ / std::max<Long>(1, Nv)));
    const Long nqmax = UBLK*Nv, cs = nqmax; // cs: per-component stride in the block buffers

    thread_local Vector<Real> Xb, dXub, dXvb, Xsrcb, Xnsrcb, wqb, KWcb, Mkerb, Tfull, projb;
    if (Xb.Dim() != COORD_DIM*nqmax) {
      Xb.ReInit(COORD_DIM*nqmax); dXub.ReInit(COORD_DIM*nqmax); dXvb.ReInit(COORD_DIM*nqmax);
      Xsrcb.ReInit(COORD_DIM*nqmax); Xnsrcb.ReInit(COORD_DIM*nqmax); wqb.ReInit(nqmax);
    }
    if (KWcb.Dim() != C*nqmax) { KWcb.ReInit(C*nqmax); Mkerb.ReInit(nqmax*KDIM0*KDIM1full); }
    if (Tfull.Dim() != C*Nu*(Long)order) Tfull.ReInit(C*Nu*(Long)order);
    if (projb.Dim() != C*nnode) projb.ReInit(C*nnode);

    for (Long a0 = 0; a0 < Nu; a0 += UBLK) {
      const Long nu = std::min<Long>(UBLK, Nu - a0), nqb = nu*Nv;
      // MuT/dMuT are (Nu x order) row-major, so a u-block is a contiguous row slice.
      const Matrix<Real> MuT_b (nu, order, (Iterator<Real>)MuT.begin()  + a0*(Long)order, false);
      const Matrix<Real> dMuT_b(nu, order, (Iterator<Real>)dMuT.begin() + a0*(Long)order, false);

      BENCH_TIC(GeomTensor);
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Matrix<Real> Cv_k (order, Nv, Cv.begin()  + k*(Long)order*Nv, false);
        const Matrix<Real> Cdv_k(order, Nv, Cdv.begin() + k*(Long)order*Nv, false);
        Matrix<Real> Xk(nu, Nv, Xb.begin() + k*cs, false), dUk(nu, Nv, dXub.begin() + k*cs, false), dVk(nu, Nv, dXvb.begin() + k*cs, false);
        Matrix<Real>::GEMM(Xk,  MuT_b,  Cv_k);
        Matrix<Real>::GEMM(dUk, dMuT_b, Cv_k);
        Matrix<Real>::GEMM(dVk, MuT_b,  Cdv_k);
        BENCH_FLOPS(3.0 * 2 * (double)nu * order * Nv);
      }
      BENCH_TOC(GeomTensor);

      BENCH_TIC(Assembly);
      for (Long a = 0; a < nu; a++) {
        for (Long b = 0; b < Nv; b++) {
          const Long q = a*Nv + b;
          const Real du0 = dXub[0*cs+q], du1 = dXub[1*cs+q], du2 = dXub[2*cs+q];
          const Real dv0 = dXvb[0*cs+q], dv1 = dXvb[1*cs+q], dv2 = dXvb[2*cs+q];
          const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
          const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
          const Real inv_area = (area > 0 ? 1/area : 0);
          Xsrcb[q*COORD_DIM+0] = Xb[0*cs+q]; Xsrcb[q*COORD_DIM+1] = Xb[1*cs+q]; Xsrcb[q*COORD_DIM+2] = Xb[2*cs+q];
          Xnsrcb[q*COORD_DIM+0] = n0*inv_area; Xnsrcb[q*COORD_DIM+1] = n1*inv_area; Xnsrcb[q*COORD_DIM+2] = n2*inv_area;
          wqb[q] = area*wu[a0+a]*wv[b];
        }
      }
      BENCH_TOC(Assembly);

      // Sized views, so KernelMatrix never re-allocates (it only ReInits on dim mismatch).
      BENCH_TIC(KernelEval);
      Matrix<Real> Mker(nqb*KDIM0, KDIM1full, Mkerb.begin(), false);
      const Vector<Real> Xsrc_v(nqb*COORD_DIM, Xsrcb.begin(), false), Xnsrc_v(nqb*COORD_DIM, Xnsrcb.begin(), false);
      ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xsrc_v, Xnsrc_v);
      BENCH_TOC(KernelEval);

      // q stays OUTERMOST on purpose: it streams Mker (dense, row-major) contiguously. A
      // c-outer/q-inner reorder was ~25% SLOWER for Stokes self (strided Mker gathers).
      BENCH_TIC(KernelWeight);
      for (Long q = 0; q < nqb; q++) {
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
            Real val;
            if (trg_dot_prod) {
              val = 0;
              for (Integer l = 0; l < COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * normal_trg[l];
            } else {
              val = Mker[q*KDIM0+k0][k1];
            }
            KWcb[(Long)(k0*KDIM1_out+k1)*cs + q] = val*wqb[q];
          }
        }
      }
      BENCH_TOC(KernelWeight);

      // Contract v now; the u-contraction is deferred so Mu is never sliced (its columns
      // would be strided). Tfull rows [a0,a0+nu) are written contiguously per channel.
      BENCH_TIC(Projection);
      for (Integer c = 0; c < C; c++) {
        const Matrix<Real> KW_c(nu, Nv, KWcb.begin() + (Long)c*cs, false);
        Matrix<Real> T_c(nu, order, Tfull.begin() + (Long)c*Nu*order + a0*(Long)order, false);
        Matrix<Real>::GEMM(T_c, KW_c, MvT);
      }
      BENCH_TOC(Projection);
    }

    // Final u-contraction: M_acc[i*order+j][c] += (Mu . Tfull_c)[i][j].
    BENCH_TIC(Projection);
    for (Integer c = 0; c < C; c++) {
      const Matrix<Real> T_c(Nu, order, Tfull.begin() + (Long)c*Nu*order, false);
      Matrix<Real> P_c(order, order, projb.begin() + (Long)c*nnode, false);
      Matrix<Real>::GEMM(P_c, Mu, T_c);
    }
    for (Long p = 0; p < nnode; p++)
      for (Integer c = 0; c < C; c++) M_acc[p][c] += projb[(Long)c*nnode + p];
    BENCH_TOC(Projection);
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

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer nbeta_default) {
    // Rectangular-polar near-interaction: cluster a single tensor-product GL rule
    // toward the nearest point on the element via the COV, integrate once.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Nbeta GL points per direction for the (finitely smooth) post-COV integrand, decoupled from
    // the field order (Bruno 2018). cov_order_ if the user set it, else the tol-derived default.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : nbeta_default);
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

  // ============================ ported from upstream 0f12ddf ============================
  // Centered self rules + split-at-foot near scheme. Grafted onto the RP work below.

  template <class Real> template <Integer order> void QuadElemList<Real>::LagrangeAtOffset(Matrix<Real>& M, Matrix<Real>& dM, Matrix<Real>& MT, Matrix<Real>& dMT, const Vector<Real>& delta, const Integer ti) {
    // L_i(nds[ti]+d) = prod_{j!=i} (d + (nds[ti]-nds[j])) / (nds[i]-nds[j]).
    // The j==ti factor is exactly `d`, so the term that vanishes at the singularity carries
    // full relative precision instead of cancelling two O(1) coordinates.
    const Vector<Real>& nds = ParamNodes(order);
    const Long N = delta.Dim();
    StaticArray<Real,order> inv_den, off;
    for (Integer i = 0; i < order; i++) { Real d = 1; for (Integer j = 0; j < order; j++) if (j != i) d *= (nds[i]-nds[j]); inv_den[i] = 1/d; }
    for (Integer j = 0; j < order; j++) off[j] = nds[ti]-nds[j]; // off[ti] == 0 exactly
    M.ReInit(order, N);
    StaticArray<Real,order> f;
    for (Long a = 0; a < N; a++) {
      for (Integer j = 0; j < order; j++) f[j] = delta[a] + off[j];
      for (Integer i = 0; i < order; i++) { Real p = inv_den[i]; for (Integer j = 0; j < order; j++) if (j != i) p *= f[j]; M[i][a] = p; }
    }
    dM.ReInit(order, N);
    Matrix<Real>::GEMM(dM, DiffMat<order>(), M);
    MT = M.Transpose();
    dMT = dM.Transpose();
  }

  template <class Real> void QuadElemList<Real>::BuildCenteredGraded1D(Vector<Real>& delta, Vector<Real>& w, const Real u0, const Integer levels, const Vector<Real>& qnds, const Vector<Real>& qwts) {
    const Integer q = qnds.Dim();
    std::vector<Real> d_, w_;
    // Panels march outward: [0,L*2^-levels], [L*2^-levels, L*2^-(levels-1)], ... , [L/2, L].
    auto side = [&](const Real Len, const Real sgn) {
      if (!(Len > 0)) return;
      Real a = 0;
      for (Integer k = levels; k >= 0; k--) {
        const Real b = Len * pow<Real>((Real)0.5, (Integer)k);
        const Real len = b - a;
        if (len > 0) for (Integer i = 0; i < q; i++) { d_.push_back(sgn*(a + len*qnds[i])); w_.push_back(len*qwts[i]); }
        a = b;
      }
    };
    side(1-u0, (Real)1);
    side(u0,   (Real)-1);
    const Long N = (Long)d_.size();
    delta.ReInit(N); w.ReInit(N);
    for (Long i = 0; i < N; i++) { delta[i] = d_[i]; w[i] = w_[i]; }
  }

  template <class Real> void QuadElemList<Real>::LogSingularQuad1DCentered(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
    // Outward-graded log-singular panel layout, emitted as offsets from v0 (the
    // singular node sits at offset exactly 0, preserving relative precision).
    const int ord = 16;
    std::vector<double> px, pw;
    auto add_alpert = [&](double a, double b, int corra, int corrb) {
      const ExtraPtResult L = (corra == 2 ? QuadLogExtraPtNodes((double)ord) : QuadSmoothExtraPtNodes((double)ord));
      const ExtraPtResult R = (corrb == 2 ? QuadLogExtraPtNodes((double)ord) : QuadSmoothExtraPtNodes((double)ord));
      const int skipL = L.NodesToSkip, skipR = R.NodesToSkip;
      const int N = std::max(skipL + skipR + 2, 2 * ord);
      const int N1 = N - 1;
      const double h = (b - a) / N1;
      for (int i = skipL; i <= N1 - skipR; ++i) { px.push_back(a + i*h); pw.push_back(h); }
      for (size_t i = 0; i < L.ExtraNodes.size(); ++i) { px.push_back(a + L.ExtraNodes[i]*h); pw.push_back(L.ExtraWeights[i]*h); }
      for (size_t i = 0; i < R.ExtraNodes.size(); ++i) { px.push_back(b - R.ExtraNodes[i]*h); pw.push_back(R.ExtraWeights[i]*h); }
    };
    Vector<Real> gnds, gwts;
    LegQuadRule<Real>::ComputeNdsWts(&gnds, &gwts, QuadOrder);
    auto add_gl = [&](double a, double b) {
      const double len = b - a;
      for (Integer i = 0; i < QuadOrder; i++) { px.push_back(a + len*(double)gnds[i]); pw.push_back(len*(double)gwts[i]); }
    };
    const double Ll = (double)v0, Lr = 1.0 - (double)v0;  // offsets: left side negative
    { double prev = -Ll;
      for (int i = 1; i <= Lvl; i++) { const double bnd = -Ll*std::ldexp(1.0,-i); add_gl(prev, bnd); prev = bnd; }
      add_alpert(prev, 0.0, 1, 2); }
    { double prev = Lr;
      for (int i = 1; i <= Lvl; i++) { const double bnd = Lr*std::ldexp(1.0,-i); add_gl(bnd, prev); prev = bnd; }
      add_alpert(0.0, prev, 2, 1); }
    const Long N = (Long)px.size();
    delta.ReInit(N); w.ReInit(N);
    for (Long i = 0; i < N; ++i) { delta[i] = (Real)px[i]; w[i] = (Real)pw[i]; }
  }

  template <class Real> template <Integer order, Integer digits> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::CenteredURule(const Integer ti, const Integer levels) {
    // `levels` is runtime, so slots are indexed by it; atomic so post-init reads are lock-free
    // (this is read once per target inside the parallel self loop).
    static constexpr Integer MaxLvl = 41;
    static std::atomic<Vector<NodeRuleData>*> slot[MaxLvl];
    static std::mutex mtx;
    SCTL_ASSERT(levels >= 0 && levels < MaxLvl);
    Vector<NodeRuleData>* p = slot[levels].load(std::memory_order_acquire);
    if (!p) {
      std::lock_guard<std::mutex> lk(mtx);
      p = slot[levels].load(std::memory_order_relaxed);
      if (!p) {
        const Vector<Real>& nds = ParamNodes(order);
        const Integer QuadOrder = DigitsQuadOrder<digits>();
        Vector<Real> qnds, qwts;
        LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);
        auto* d = new Vector<NodeRuleData>(order);
        for (Integer i = 0; i < order; i++) {
          BuildCenteredGraded1D((*d)[i].param, (*d)[i].w, nds[i], levels, qnds, qwts);
          LagrangeAtOffset<order>((*d)[i].M, (*d)[i].dM, (*d)[i].MT, (*d)[i].dMT, (*d)[i].param, i);
        }
        p = d;
        slot[levels].store(p, std::memory_order_release);
      }
    }
    return (*p)[ti];
  }

  template <class Real> template <Integer order, Integer digits> const typename QuadElemList<Real>::NodeRuleData& QuadElemList<Real>::CenteredVRule(const Integer tj) {
    auto compute_all = []() {
      const Vector<Real>& nds = ParamNodes(order);
      const Integer Lvl = DigitsVLevels<digits>();
      const Integer QuadOrder = DigitsQuadOrder<digits>();
      Vector<NodeRuleData> data(order);
      for (Integer j = 0; j < order; j++) {
        LogSingularQuad1DCentered(data[j].param, data[j].w, nds[j], Lvl, QuadOrder);
        LagrangeAtOffset<order>(data[j].M, data[j].dM, data[j].MT, data[j].dMT, data[j].param, j);
      }
      return data;
    };
    static const Vector<NodeRuleData> data = compute_all();
    return data[tj];
  }

  template <class Real> template <Integer digits> Integer QuadElemList<Real>::NearQuadOrder() {
    static const Integer q = []() { const char* v = std::getenv("SCTL_NEAR_QORDER");
      const Integer x = (v ? (Integer)atoi(v) : 0); if (x > 0) return x;
      Real b; Integer qq; NearRhoRule(pow<digits,Real>((Real)0.1), b, qq); return qq; }();
    return q;
  }

  template <class Real> void QuadElemList<Real>::NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder) {
    // Measured cost-optimal rho vs requested digits (order 12, 4x4 panels/face, Laplace SL+DL):
    // 1e-4 -> 2.0, 1e-6 -> 2.0, 1e-8 -> 2.5, 1e-10 -> 3.0, 1e-12 -> 2.5. Accuracy breaks down
    // above rho ~ 3.2 at every tolerance: the attained rate saturates near rho_eff ~ 3, so a
    // larger design rho only buys refinement levels without improving the per-cell rate.
    const double d = -std::log10((double)std::max<Real>(tol, (Real)1e-16));
    const double rho = std::min(3.0, std::max(2.0, 2.0 + 0.25*(d - 6)));
    const double C = std::max(1e-3, (15.0*(rho*rho - 1))/64.0);
    QuadOrder = std::max<Integer>(2, (Integer)std::ceil(-std::log(C*(double)std::max<Real>(tol, (Real)1e-16))/std::log(rho)*0.5 + 1));
    // End-foot reach, not the semi-major axis. E_rho has semi-axes a,b with a^2-b^2 = 1; a
    // singularity at parameter s with perpendicular offset d~ = 2d/L lies outside E_rho when
    // s^2/a^2 + d~^2/b^2 > 1. Splitting at (u0,v0) puts the foot at the corner cell's endpoint
    // (s = +-1), giving d~ > b^2/a -- weaker by a^2/b^2 than the (rho+1/rho)/4 semi-major reach,
    // and weaker than the true worst case d~ > b (foot at the panel centre, which cannot occur here).
    const double a = (rho + 1/rho)/2, b = (rho - 1/rho)/2;
    b_ellipse = (Real)(b*b/(2*a));
  }

  template <class Real> Integer QuadElemList<Real>::NearMaxLvlOverride() {
    static const Integer L = []() { const char* v = std::getenv("SCTL_NEAR_MAXLVL");
      const Integer x = (v ? (Integer)atoi(v) : 0); return x > 0 ? x : 0; }();
    return L;
  }

  template <class Real> template <Integer digits> Real QuadElemList<Real>::NearBEllipse() {
    static const Real b = []() { const char* v = std::getenv("SCTL_NEAR_BELLIPSE");
      const double x = (v ? atof(v) : 0); if (x > 0) return (Real)x;
      Real bb; Integer qq; NearRhoRule(pow<digits,Real>((Real)0.1), bb, qq); return bb; }();
    return b;
  }

  template <class Real> template <Integer order, Integer digits> const Vector<typename QuadElemList<Real>::GradeRule>& QuadElemList<Real>::NearGradeTable() {
    // Built once per (order,digits). Every entry is in NORMALIZED sub-element coordinates and
    // carries no positional index -- that is the point of splitting at the foot.
    auto build = []() {
      const Integer q = NearQuadOrder<digits>();
      const Vector<Real>& gnds = ParamNodes(order);   // sub-element's own nodes, normalized
      Vector<Real> qn, qw; LegQuadRule<Real>::ComputeNdsWts(&qn, &qw, q);
      Vector<GradeRule> tab(2*MaxNearLvl);
      auto fill = [&](GradeRule& r, const Real a, const Real b) {
        r.a = a; r.b = b;
        const Real w = b - a;
        r.nds.ReInit(q); r.w.ReInit(q);
        for (Integer i = 0; i < q; i++) { r.nds[i] = a + w*qn[i]; r.w[i] = w*qw[i]; }
        // T[i][j] = Lhat_i(nds[j]): sub-element nodes -> this interval's quadrature nodes.
        r.T.ReInit(order, q);
        { Vector<Real> v(order*q, r.T.begin(), false); LagrangeInterp<Real>::Interpolate(v, gnds, r.nds); }
        r.dT.ReInit(order, q);
        Matrix<Real>::GEMM(r.dT, DiffMat(order), r.T);
        r.TT.ReInit(q, order); r.TD.ReInit(2*q, order);
        for (Integer i = 0; i < order; i++) for (Integer a = 0; a < q; a++) {
          r.TT[a][i] = r.T[i][a]; r.TD[a][i] = r.T[i][a]; r.TD[q+a][i] = r.dT[i][a];
        }
      };
      for (Integer k = 0; k < MaxNearLvl; k++) {
        const Real lo = 1 - pow<Real>((Real)0.5, k), hi = 1 - pow<Real>((Real)0.5, k+1);
        fill(tab[k], lo, hi);                                   // shell_k
        fill(tab[MaxNearLvl + k], lo, (Real)1);                  // core_k = [1-2^-k, 1]
      }
      return tab;
    };
    static const Vector<GradeRule> tab = build();
    return tab;
  }

  template <class Real> void QuadElemList<Real>::ExpandSegments(Vector<Real>& param, Vector<Real>& w, const Vector<Real>& seg, const Vector<Real>& qnds, const Vector<Real>& qwts) {
    // One QuadOrder GL rule per segment, concatenated in segment order (so a contiguous run of
    // segments maps to a contiguous slice of `param`/`w` -- relied on by the near chunking).
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

  template <class Real> void QuadElemList<Real>::BuildFootGraded1DSegments(Vector<Real>& seg, Vector<Long>& seg_depth, const Real center, const Real b_ellipse, const Real w_min) {
    // Split [0,1] at `center`, then grade geometrically outward on each side.
    //
    // Ratio: a segment [c+r^{k+1}s, c+r^k s] has width r^k s(1-r) and parameter distance r^{k+1} s
    // to the center, so the admissibility test pdist >= b*width holds iff r >= b/(1+b) -- and since
    // a SMALLER r decays faster (fewer segments), r = b/(1+b) is the optimal choice. A small safety
    // factor keeps the marginal segments strictly inside the test under rounding.
    //
    // The innermost segment TOUCHES `center` (pdist = 0) and so fails the pure parameter-space
    // test at any width; it is admissible instead under the off-surface effective distance
    // sqrt(pdist^2 + h^2) = h with h = b_ellipse*w_min, which needs width <= w_min. That is exactly
    // what the caller's w_min = dist/(b_ellipse*L_phys) encodes, so near targets (dist > 0) are
    // covered; this partition is NOT valid for an on-surface singularity.
    constexpr Long MaxLeaves = 4096;
    const Real r = std::min<Real>((Real)0.9, b_ellipse/(1 + b_ellipse) * (Real)1.05);
    const Real wmin = std::max<Real>(w_min, (Real)1e-300);

    std::vector<Real> a; std::vector<Long> d;
    for (Integer side = 0; side < 2; side++) {
      const Real sgn = (side ? (Real)1 : (Real)-1);
      const Real span = (side ? 1 - center : center);
      if (!(span > 0)) continue;

      // Edges marching inward from the far end: c+sgn*span, c+sgn*span*r, ... until width <= wmin.
      Real w = span;
      Long lvl = 0;
      while (w > wmin) {
        const Real w_next = w*r;
        const Real e0 = center + sgn*w, e1 = center + sgn*w_next;
        const Real lo = std::min<Real>(e0, e1), hi = std::max<Real>(e0, e1);
        if (hi - lo > 0) { a.push_back(lo); a.push_back(hi); d.push_back(lvl); }
        w = w_next;
        lvl++;
        SCTL_ASSERT((Long)d.size() <= MaxLeaves);
      }
      // Innermost segment, touching the center (width w <= wmin).
      const Real e0 = center + sgn*w;
      const Real lo = std::min<Real>(e0, center), hi = std::max<Real>(e0, center);
      if (hi - lo > 0) { a.push_back(lo); a.push_back(hi); d.push_back(lvl); }
      SCTL_ASSERT((Long)d.size() <= MaxLeaves);
    }

    const Long nseg = (Long)d.size();
    seg.ReInit(nseg*2);
    seg_depth.ReInit(nseg);
    for (Long i = 0; i < nseg; i++) { seg[i*2+0] = a[i*2+0]; seg[i*2+1] = a[i*2+1]; seg_depth[i] = d[i]; }
  }

  template <class Real> Integer QuadElemList<Real>::NearFootAndDepth(Real& ustar, Real& vstar, Real& dist, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Real b_ellipse, const Integer max_depth, Real* h_param) {
    // The refinement CENTER and the depth cap both come from GetClosestPoint (the FOOT), not the
    // closest NODE: for an off-surface target whose foot lies BETWEEN nodes (panel-interior near
    // target) the nearest-node distance overestimates the near distance, so a node-based cap
    // under-refines and leaves the foot mid-cell where a smooth GL rule cannot resolve it
    // (~0.1-0.5 error). Shared by both near partitionings so they agree on center and depth.
    BENCH_TIC(ClosestNode);
    dist = qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    BENCH_TOC(ClosestNode);

    // Panel scale from the surface speeds at the foot (full parameter width is 1).
    Real Xc[COORD_DIM], dXdu[COORD_DIM], dXdv[COORD_DIM];
    qel.EvalPoint(Xc, dXdu, dXdv, ustar, vstar, elem_idx, nullptr);
    Real su2 = 0, sv2 = 0;
    for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXdu[k]*dXdu[k]; sv2 += dXdv[k]*dXdv[k]; }
    const Real L_phys = std::max<Real>(sqrt<Real>(su2), sqrt<Real>(sv2));

    // Depth cap L from the foot distance: the innermost cell at depth L has physical size
    // ~ b_ellipse*L_phys*2^-L, admissible once that drops below dist, i.e. L ~ log2(b_ellipse*
    // L_phys/dist). A non-finite / non-positive dist (near-touching / degenerate) forces the full cap.
    if (!(dist > 0) || !std::isfinite((double)dist) || !(L_phys > 0)) {
      if (h_param) *h_param = 0; // degenerate/near-touching: force the full cap
      return max_depth;
    }
    if (h_param) *h_param = dist/L_phys; // off-surface distance in PARAMETER units
    const double lvl = std::ceil(std::log2((double)(b_ellipse*L_phys) / (double)dist));
    return (Integer)std::min<double>((double)max_depth, std::max<double>(0.0, lvl));
  }

  template <class Real> Integer QuadElemList<Real>::BuildNearTensorRule(Vector<Real>& u_param, Vector<Real>& wu, Vector<Real>& v_param, Vector<Real>& wv,
                                                                       Vector<Real>* useg, Vector<Long>* useg_depth, Vector<Real>* vseg, Vector<Long>* vseg_depth,
                                                                       const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg,
                                                                       const Real b_ellipse, const Vector<Real>& qnds, const Vector<Real>& qwts, const Integer max_depth) {
    // Grade [0,1] toward u* and toward v* INDEPENDENTLY -- splitting each side AT the foot and
    // grading geometrically outward (BuildFootGraded1DSegments) -- then take the full tensor
    // product. Separability is what turns the interpolation into one large tensor multiply per
    // side instead of a small GEMM set per cell of a 2D tree.
    //
    // Anchoring the innermost segment ON the foot (rather than letting the foot fall mid-cell, as
    // midpoint bisection would) is what makes this robust: the near-singularity then sits on a cell
    // BOUNDARY, outside every integration domain. Measured consequences vs a mid-cell foot: ~40-48%
    // fewer cells, ~10^4x lower error at a truncated depth cap (max_depth=4), and 1-2 orders of
    // magnitude lower error for targets sitting just off a panel SEAM. The price is that the
    // foot-touching segment is admissible only under the off-surface effective distance
    // sqrt(pdist^2 + h^2), so this rule requires dist > 0 and must never serve the self path.
    //
    // Cost note: a tensor product of two O(L) partitions is O(L^2) cells vs a quadtree's O(L), so
    // this buys larger BLAS-efficient GEMMs at the price of more quadrature points; the two roughly
    // cancel in end-to-end solve time.
    Real ustar, vstar, dist, h_param;
    const Integer L = NearFootAndDepth(ustar, vstar, dist, qel, elem_idx, Xtrg, b_ellipse, max_depth, &h_param);

    Vector<Real> useg_local, vseg_local; Vector<Long> udep_local, vdep_local;
    Vector<Real>& us = (useg ? *useg : useg_local);
    Vector<Real>& vs = (vseg ? *vseg : vseg_local);
    Vector<Long>& ud = (useg_depth ? *useg_depth : udep_local);
    Vector<Long>& vd = (vseg_depth ? *vseg_depth : vdep_local);

    // Innermost width w_min = h_param/b_ellipse (== the continuous 2^-L, before L's ceil -- not
    // rounding up to a whole dyadic level is worth up to a full level of refinement), floored at
    // 2^-max_depth so a near-touching / degenerate foot cannot blow up the segment count.
    const Real w_floor = pow<Real>((Real)0.5, max_depth);
    const Real w_min = std::max<Real>(h_param/b_ellipse, w_floor);
    BuildFootGraded1DSegments(us, ud, ustar, b_ellipse, w_min);
    BuildFootGraded1DSegments(vs, vd, vstar, b_ellipse, w_min);
    (void)L; // L now only bounds w_min via w_floor; returned for the caller's BENCH_NEAR stat
    ExpandSegments(u_param, wu, us, qnds, qwts);
    ExpandSegments(v_param, wv, vs, qnds, qwts);
    return L;
  }

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockGraded(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // THE adaptive near block. ONE foot-graded tensor grid over the whole panel -> ONE
    // IntegrateBlock: no quadtree, no per-leaf loop, no distinct-interval dedup, and the
    // interpolation is a single large tensor multiply per quantity instead of a small GEMM set per
    // tree cell. See BuildNearTensorRule for the construction and its accuracy rationale.
    if (qel.NearUsesRectPolar()) { NearInteracBlockRP<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ker, NbetaForDigits(digits)); return; }

    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Per-panel GL order / Bernstein parameter fixed at compile time by `digits`; the GL rule
    // itself is a build-once static (NOT recomputed per target -- ComputeNdsWts is an O(N^2)
    // uncached Newton solve).
    const Integer QuadOrder = DigitsQuadOrder<digits>();
    const Real b_ellipse = DigitsBEllipse<digits>();
    const std::pair<Vector<Real>, Vector<Real>>& gl = DigitsGLRule<digits>();

    thread_local Vector<Real> u_param, wu, v_param, wv;
    BENCH_TIC(QuadtreeBuild); // same phase label as the quadtree build it replaces
    const Integer L = BuildNearTensorRule(u_param, wu, v_param, wv, nullptr, nullptr, nullptr, nullptr,
                                          qel, elem_idx, Xtrg, b_ellipse, gl.first, gl.second, qel.max_depth_);
    BENCH_TOC(QuadtreeBuild);
    const Long Nu = u_param.Dim(), Nv = v_param.Dim();
    BENCH_NEAR((Nu/QuadOrder) * (Nv/QuadOrder), L);
    (void)L; (void)QuadOrder; // only read by BENCH_NEAR, which compiles away without -DBENCH_QUAD

    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero(); // IntegrateBlock accumulates (+=), so zero regardless of whether ReInit fired
    if (!Nu || !Nv) return;

    // ONE interpolation matrix per side over ALL concatenated nodes -- not one per segment or per
    // refinement level. The Lagrange basis is global over the panel, so the (order x N) matrix
    // built from the concatenated node list IS the horizontal concatenation of the per-segment
    // blocks: same numbers, one formation, and one large GEMM downstream instead of many tiny ones.
    // Passed via IntegrateBlock's *_pre args so the operators live in thread_local storage rather
    // than being re-allocated per target inside it.
    thread_local NodeRuleData ru, rv;
    BENCH_TIC(InterpBuild);
    BuildInterp1D<order>(ru.M, ru.dM, ru.MT, ru.dMT, u_param);
    BuildInterp1D<order>(rv.M, rv.dM, rv.MT, rv.dMT, v_param);
    BENCH_TOC(InterpBuild);

    IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, u_param, wu, v_param, wv, ker,
                          &rv.M, &rv.dM, &ru.M, &ru.dM, &rv.MT, &ru.MT, &ru.dMT);
  }

  // SUPERSEDED by NearInteracBlockGraded (foot-graded separable tensor) in the Adaptive/Hybrid near
  // path -- this isotropic graded-quadtree split-at-foot rule lost ~2-3 orders vs RectPolar even on
  // smooth geometry (Hybrid DL_stk 3.1e-6 here vs 1.3e-8 with the graded tensor). Retained only as a
  // reference for the split-at-foot cell layout (WriteNearInteracVTK mirrors it).
  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    if (qel.NearUsesRectPolar()) { NearInteracBlockRP<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ker, NbetaForDigits(digits)); return; }
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Long nnode = (Long)order*order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full/COORD_DIM : KDIM1full;
    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    // Leaves accumulate into a CHANNEL-major buffer, in the ELEMENT-node basis (IntegrateNearCM's
    // src is the full element slab), so a single channel-major -> node-major transpose at the end
    // replaces the old per-quadrant S_u/S_v map-back entirely.
    const Integer C_ = KDIM0*KDIM1_out;
    thread_local Vector<Real> acc;
    if (acc.Dim() != (Long)C_*nnode) acc.ReInit((Long)C_*nnode);
    M_acc.SetZero();
    acc.SetZero();

    const Real b_ellipse = NearBEllipse<digits>();
    const Vector<GradeRule>& tab = NearGradeTable<order,digits>();   // precomputed shell_k/core_k intervals: xi-nodes & weights per level

    // Foot + per-direction surface speeds (same criterion as the Duffy near).
    Real ustar, vstar;
    BENCH_TIC(ClosestNode);
    const Real dist = qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    BENCH_TOC(ClosestNode);
    Real Xc[COORD_DIM], dXu_[COORD_DIM], dXv_[COORD_DIM];
    qel.EvalPoint(Xc, dXu_, dXv_, ustar, vstar, elem_idx, nullptr);
    Real su2 = 0, sv2 = 0;
    for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXu_[k]*dXu_[k]; sv2 += dXv_[k]*dXv_[k]; }
    const Real spd_u = sqrt<Real>(su2), spd_v = sqrt<Real>(sv2);

    const Vector<Real>& gnds = ParamNodes(order);
    const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};   // [dir][side] quadrant length (= Jacobian of the xi->element map)

    BENCH_TIC(ClosestPoint);
    // Target-shifted element nodal coords, component-major: one contiguous (COORD_DIM*order x order).
    // Passed straight to IntegrateNearCM as src_nodal -- NO S_u/S_v remap to a per-quadrant grid.
    thread_local Vector<Real> cs;
    if (cs.Dim() != COORD_DIM*nnode) cs.ReInit(COORD_DIM*nnode);
    {
      const Long base = elem_idx * nnode * COORD_DIM;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) cs[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
      }
    }

    // Per-quadrant composite interpolation operators, built DIRECTLY from the element nodes to a
    // leaf's quadrature points in element (u,v) coordinates -- the "small tensor product at each
    // leaf". A shell_k/core_k interval is shared by up to two leaves per level, so each composite
    // is cached once per quadrant (keyed on the table index) and reused. The xi-nodes/weights come
    // from the precomputed level table; only the target-dependent foot map (affine) is applied.
    thread_local std::vector<GradeRule> Ucomp, Vcomp;
    thread_local std::vector<char> Ubuilt, Vbuilt;
    if ((Integer)Ucomp.size() != 2*MaxNearLvl) { Ucomp.resize(2*MaxNearLvl); Vcomp.resize(2*MaxNearLvl); Ubuilt.resize(2*MaxNearLvl); Vbuilt.resize(2*MaxNearLvl); }
    auto build_comp = [&](GradeRule& out, const GradeRule& base, const Integer side, const Real xs, const Real jac) {
      const Integer q = (Integer)base.nds.Dim();
      thread_local Vector<Real> pts;
      if (pts.Dim() != q) pts.ReInit(q);
      for (Integer i = 0; i < q; i++) pts[i] = (side ? (1 - (1-xs)*base.nds[i]) : (xs*base.nds[i]));  // xi -> element coord
      out.w.ReInit(q);
      for (Integer i = 0; i < q; i++) out.w[i] = jac*base.w[i];                                       // element-space GL weights (affine Jacobian |d elem/d xi| = jac)
      out.T.ReInit(order, q);
      { Vector<Real> v(order*q, out.T.begin(), false); LagrangeInterp<Real>::Interpolate(v, gnds, pts); }  // T[i][j] = L_i(pts[j]): element nodes -> leaf quad pts
      out.dT.ReInit(order, q);
      Matrix<Real>::GEMM(out.dT, DiffMat(order), out.T);                                              // d/d(element coord)
      out.TT.ReInit(q, order); out.TD.ReInit(2*q, order);
      for (Integer i = 0; i < order; i++) for (Integer a = 0; a < q; a++) {
        out.TT[a][i] = out.T[i][a]; out.TD[a][i] = out.T[i][a]; out.TD[q+a][i] = out.dT[i][a];
      }
    };
    BENCH_TOC(ClosestPoint);

    const Integer ovr = NearMaxLvlOverride();
    const Integer KMAX = std::min<Integer>(ovr ? ovr : qel.max_depth_, MaxNearLvl-1); // table bound

    for (Integer sdu = 0; sdu < 2; sdu++) {
      if (!(slen[0][sdu] > 0)) continue;
      for (Integer sdv = 0; sdv < 2; sdv++) {
        if (!(slen[1][sdv] > 0)) continue;

        // Per-quadrant composite caches (the foot map differs per quadrant).
        for (Integer i = 0; i < 2*MaxNearLvl; i++) { Ubuilt[i] = 0; Vbuilt[i] = 0; }
        auto getU = [&](const Integer idx) -> const GradeRule& {
          if (!Ubuilt[idx]) { build_comp(Ucomp[idx], tab[idx], sdu, ustar, slen[0][sdu]); Ubuilt[idx] = 1; }
          return Ucomp[idx];
        };
        auto getV = [&](const Integer idx) -> const GradeRule& {
          if (!Vbuilt[idx]) { build_comp(Vcomp[idx], tab[idx], sdv, vstar, slen[1][sdv]); Vbuilt[idx] = 1; }
          return Vcomp[idx];
        };
        // The element's natural (u,v) orientation is used throughout (no xi-mirroring), so a leaf
        // normal is always the element normal -- no per-quadrant sign correction (nrm_sign = +1).
        auto emit = [&](const Integer iu, const Integer iv) {
          if (!(tab[iu].b > tab[iu].a) || !(tab[iv].b > tab[iv].a)) return;
          const GradeRule& gu = getU(iu);
          const GradeRule& gv = getV(iv);
          IntegrateNearCM<order>(normal_trg, gu.w, gv.w, ker,
                                 gu.T, gu.TT, gu.TD, gv.T, gv.dT, gv.TT,
                                 cs, (Real)1, acc);
        };

        // ISOTROPIC graded-quadtree refinement toward the foot corner. At each level quadrisect the
        // corner (core_L x core_L) cell, emit its 3 non-corner children as leaves, and recurse into
        // the corner child; u and v advance in lockstep (isotropic dyadic). Depth stops when the
        // corner cell is admissible against the target distance (end-foot Bernstein reach).
        Real hu = slen[0][sdu]*spd_u, hv = slen[1][sdv]*spd_v;
        const bool cap = !(dist > 0) || !std::isfinite((double)dist);
        Integer L = 0;
        while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && L < KMAX) {
          emit(L, L);                              // shell_L x shell_L
          emit(L, MaxNearLvl + L + 1);             // shell_L x core_{L+1}
          emit(MaxNearLvl + L + 1, L);             // core_{L+1} x shell_L
          L++; hu *= (Real)0.5; hv *= (Real)0.5;
        }
        emit(MaxNearLvl + L, MaxNearLvl + L);      // terminal corner cell core_L x core_L
      }
    }

    // acc accumulated directly in the ELEMENT-node basis, so no S_u/S_v map-back is needed: one
    // channel-major -> node-major transpose into M_acc finishes the block.
    for (Integer c = 0; c < C_; c++)
      for (Long p = 0; p < nnode; p++)
        M_acc[p][c] = acc[(Long)c*nnode + p];
  }


  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Singular self-interaction for on-surface node (ti,tj). 1D reduction: graded u-rule
    // toward u0 x Alpert log-singular v-rule toward v0; both rules + interpolation are
    // preloaded (geometry-independent, fixed by order/ti/tj/digits), integrated by
    // IntegrateBlock. IntegrateBlock still does the target-centered geometry per target.
    if (qel.SelfUsesRectPolar()) { SelfInteracBlockRP<order>(M_acc, qel, elem_idx, ti, tj, Xtrg, normal_trg, ker, NbetaForDigits(digits)); return; }
    if (qel.SelfUsesDuffy()) { SelfInteracBlockDuffy<order>(M_acc, qel, elem_idx, ti, tj, Xtrg, normal_trg, ker, digits); return; }

    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Centered rules: graded u-rule + Alpert v-rule built OUTWARD from the singular node
    // (offset-stored, endpoint-anchored), so the singularity lands at parameter offset zero exactly.
    const NodeRuleData& ru = CenteredURule<order, digits>(ti, qel.max_depth_);  // u: graded rule
    const NodeRuleData& rv = CenteredVRule<order, digits>(tj);                   // v: composite Alpert rule

    M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero();
    IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ru.param, ru.w, rv.param, rv.w, ker, &rv.M, &rv.dM, &ru.M, &ru.dM, &rv.MT, &ru.MT, &ru.dMT);
  }

  // ============================ Duffy edge-collapsed self scheme (ported from upstream) ============================

  template <class Real> inline Integer QuadElemList<Real>::DuffyTOrder(const Integer digits, const Integer order, const Integer kdim0) {
    // t-points per digit, with margin: the error falls only ~0.35 decades per node, so a thin
    // margin is not safe. Vector kernels need ~1.5x the t-nodes of a scalar one at the same
    // tolerance. Calibrated end-to-end on the Green's identity with a varying density. Twist pi/6
    // binds: pi/2's discretization floor masks the self error and twist 0 is benign, so the
    // ends of the twist range alone understate nt by 2x.
    // CAVEAT: the vector constant is calibrated over twists {0, pi/6} only -- treat it as
    // provisional until it gets the four-twist check the scalar one had.
    const double per_digit = (kdim0 > 1 ? 4.0 : 2.5);
    // order/2 floor: the t-integrand carries degree order-1. The measured minima dip below it
    // at loose tolerance only because a resolved geometry has eps-small top coefficients.
    return std::max<Integer>(order/2, (Integer)std::ceil(per_digit*(double)digits));
  }

  template <class Real> template <Integer order> const typename QuadElemList<Real>::DuffySelfTable& QuadElemList<Real>::DuffyTable() {
    // Fixed by `order` alone: q_s = order and the t-rule -- the only accuracy- and
    // metric-dependent part -- is built per target. Function-local static, so it
    // self-initializes on first use from any thread.
    static const DuffySelfTable table = []() {
      DuffySelfTable tbl;
      const Integer qs = order;   // radial GL order; see the DuffyTOrder note on the t-rule
      tbl.ns = qs;
      LegQuadRule<Real>::ComputeNdsWts(&tbl.sn, &tbl.sw, qs);

      const Vector<Real>& nds = ParamNodes(order);
      const Matrix<Real>& D = DiffMat<order>();
      tbl.tri.resize(4*(size_t)order*order);
      const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
      for (Integer ti = 0; ti < order; ti++) for (Integer tj = 0; tj < order; tj++) {
        const Real u0 = nds[ti], v0 = nds[tj];
        for (Integer kt = 0; kt < 4; kt++) {
          DuffyTri& T = tbl.tri[((size_t)ti*order + tj)*4 + kt];
          const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
          const Real b[2] = {cu[(kt+1)%4]-u0, cv[(kt+1)%4]-v0};
          const Real e[2] = {b[0]-a[0], b[1]-a[1]};
          T.J0 = a[0]*b[1] - a[1]*b[0];
          SCTL_ASSERT_MSG(T.J0 > 0, "Duffy triangle orientation");
          T.swap_ab = (fabs<Real>(e[0]) < fabs<Real>(e[1]));  // e is axis aligned
          T.nsign = (T.swap_ab ? (Real)-1 : (Real)1);
          const Real al0 = (T.swap_ab ? v0 : u0), be0 = (T.swap_ab ? u0 : v0);
          const Real aal = (T.swap_ab ? a[1] : a[0]), abe = (T.swap_ab ? a[0] : a[1]);
          const Real eal = (T.swap_ab ? e[1] : e[0]);
          { // collapsed direction beta(s_i): value and derivative side by side
            Vector<Real> bv(qs);
            for (Integer i = 0; i < qs; i++) bv[i] = be0 + tbl.sn[i]*abe;
            Matrix<Real> Wb(order, qs), WbD(order, qs);
            { Vector<Real> t((Long)order*qs, Wb.begin(), false); LagrangeInterp<Real>::Interpolate(t, nds, bv); }
            Matrix<Real>::GEMM(WbD, D, Wb);
            T.WbC.ReInit(order, 2*qs);
            for (Integer r = 0; r < order; r++) for (Integer i = 0; i < qs; i++) { T.WbC[r][i] = Wb[r][i]; T.WbC[r][qs+i] = WbD[r][i]; }
            T.WbT = Wb.Transpose();
          }
          { // alpha(s_i,.) is affine in t, so its Lagrange values at `order` reference nodes
            // reproduce it exactly; the t-rule then enters only through Tt.
            T.MiC.ReInit(qs); T.MiT.ReInit(qs);
            Vector<Real> av(order);
            Matrix<Real> Mi(order, order), MiD(order, order);
            for (Integer i = 0; i < qs; i++) {
              for (Integer k = 0; k < order; k++) av[k] = al0 + tbl.sn[i]*(aal + nds[k]*eal);
              { Vector<Real> t((Long)order*order, Mi.begin(), false); LagrangeInterp<Real>::Interpolate(t, nds, av); }
              Matrix<Real>::GEMM(MiD, D, Mi);
              T.MiC[i].ReInit(order, 2*order);
              for (Integer r = 0; r < order; r++) for (Integer k = 0; k < order; k++) { T.MiC[i][r][k] = Mi[r][k]; T.MiC[i][r][order+k] = MiD[r][k]; }
              T.MiT[i] = Mi.Transpose();
            }
          }
        }
      }
      return tbl;
    }();
    return table;
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::SelfInteracBlockDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order*order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full/COORD_DIM : KDIM1full;
    const Integer C = KDIM0*KDIM1_out;
    constexpr Integer NR = 3*COORD_DIM;   // value, d/d_alpha, d/d_beta rows per s-node
    constexpr Integer NA = 2*COORD_DIM;   // Ai, Adi rows fed to [Mi | Mi']
    M_acc.ReInit(nnode, C); M_acc.SetZero();

    const DuffySelfTable& tbl = DuffyTable<order>();
    const Long ns = tbl.ns, nt = DuffyTOrder(digits, order, KDIM0);
    // nt is fixed for the whole call, so the GL rule is shared by every triangle of every
    // target. MaxGLOrder covers DuffyTOrder's largest value (4 t-points per digit).
    static constexpr Integer MaxGLOrder = 128;
    const Vector<Real>& qn = LegQuadRule<Real>::template nds<MaxGLOrder>(nt);
    const Vector<Real>& qw = LegQuadRule<Real>::template wts<MaxGLOrder>(nt);
    // Tt does not depend on the s-node, so stage 2b contracts the whole s-range in one
    // (ns*NR x order)(order x nt) GEMM.
    const Long sblk = ns;
    const Vector<Real>& nds = ParamNodes(order);
    const Matrix<Real>& D = DiffMat<order>();

    auto ash = [](const Real x) { return log<Real>(x + sqrt<Real>(x*x + (Real)1)); };

    // Target-shifted nodal slab: positions are source-minus-target, so the kernel target
    // sits at the origin and r stays accurate at the singularity.
    thread_local Vector<Real> cs;
    if (cs.Dim() != COORD_DIM*nnode) cs.ReInit(COORD_DIM*nnode);
    const Long base = elem_idx*nnode*COORD_DIM;

    for (Integer k = 0; k < COORD_DIM; k++) {
      const Real ok = Xtrg[k];
      for (Long q = 0; q < nnode; q++) cs[k*nnode + q] = qel.coord[base + k*nnode + q] - ok;
    }

    // Surface metric at (u0,v0). t* and the peak width are set by distance ON THE SURFACE:
    // placing them in parameter space instead misplaces the peak by |cot(theta)| widths.
    Real G[4];
    {
      Real du[COORD_DIM], dv[COORD_DIM];
      for (Integer k = 0; k < COORD_DIM; k++) {
        Real su = 0, sv = 0;
        for (Integer i = 0; i < order; i++) su += cs[k*nnode + (Long)i*order + tj]*D[i][ti];
        for (Integer j = 0; j < order; j++) sv += cs[k*nnode + (Long)ti*order + j]*D[j][tj];
        du[k] = su; dv[k] = sv;
      }
      Real guu = 0, guv = 0, gvv = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { guu += du[k]*du[k]; guv += du[k]*dv[k]; gvv += dv[k]*dv[k]; }
      G[0] = guu; G[1] = guv; G[2] = guv; G[3] = gvv;
    }

    StaticArray<Real,COORD_DIM> Xt0{0,0,0};
    const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);
    const Vector<Real>& pnds = nds;

    for (Integer kt = 0; kt < 4; kt++) {
      const DuffyTri& T = tbl.tri[((size_t)ti*order + tj)*4 + kt];
      const Real u0 = nds[ti], v0 = nds[tj];
      const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
      const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
      const Real e[2] = {cu[(kt+1)%4]-cu[kt], cv[(kt+1)%4]-cv[kt]};

      Real tstar, dOverL;
      { // metric-aware foot and width
        const Real Me[2] = {G[0]*e[0]+G[1]*e[1], G[2]*e[0]+G[3]*e[1]};
        const Real am = e[0]*Me[0] + e[1]*Me[1];
        Real ts = -(a[0]*Me[0] + a[1]*Me[1])/am;
        ts = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
        const Real c[2] = {a[0]+ts*e[0], a[1]+ts*e[1]};
        const Real d2 = c[0]*(G[0]*c[0]+G[1]*c[1]) + c[1]*(G[2]*c[0]+G[3]*c[1]);
        tstar = ts; dOverL = sqrt<Real>(d2)/sqrt<Real>(am);
      }

      // sinh substitution t = t* + (d/L)*sinh(xi): one GL rule, and cheaper than dyadic
      // grading toward t* at equal accuracy.
      const Long szt = 2*nt + (Long)order*nt + (Long)nt*order;
      ScratchBuf<Real> sbt(szt);
      Long offt = 0;
      auto taket = [&](const Long n) { Iterator<Real> r = sbt.begin() + offt; offt += n; return r; };
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
      { Vector<Real> t((Long)order*nt, Tt.begin(), false); LagrangeInterp<Real>::Interpolate(t, pnds, tn); }
      for (Integer r = 0; r < order; r++) for (Long j = 0; j < nt; j++) TtT[j][r] = Tt[r][j];

      const Long nq = ns*nt;
      const Long sz = COORD_DIM*nnode + 2*COORD_DIM*(Long)order*ns + (Long)NA*order + 2*(Long)NA*order
                    + sblk*NR*(Long)order + sblk*NR*nt + 2*COORD_DIM*nq + nq
                    + nq*KDIM0*KDIM1full + (Long)C*nq + ns*(Long)C*order + (Long)C*order + (Long)C*order*ns + nnode;
      ScratchBuf<Real> sb(sz);
      Long off = 0;
      auto take = [&](const Long n) { Iterator<Real> r = sb.begin() + off; off += n; return r; };

      Matrix<Real> FS(COORD_DIM*order, order, take(COORD_DIM*nnode), false);
      Matrix<Real> Gm(COORD_DIM*order, 2*ns, take(2*COORD_DIM*(Long)order*ns), false);
      Matrix<Real> As(NA, order, take((Long)NA*order), false), Tmp(NA, 2*order, take(2*(Long)NA*order), false);
      Matrix<Real> HG(sblk*NR, order, take(sblk*NR*(Long)order), false);
      Matrix<Real> XdX(sblk*NR, nt, take(sblk*NR*nt), false);
      Vector<Real> Xs(COORD_DIM*nq, take(COORD_DIM*nq), false), Xn(COORD_DIM*nq, take(COORD_DIM*nq), false);
      Vector<Real> wq(nq, take(nq), false);
      Matrix<Real> Mker(nq*KDIM0, KDIM1full, take(nq*KDIM0*KDIM1full), false);
      Matrix<Real> KW(ns*C, nt, take((Long)C*nq), false);
      Matrix<Real> Zall(ns*C, order, take(ns*(Long)C*order), false);
      Matrix<Real> Yi(C, order, take((Long)C*order), false), Yall(C*order, ns, take((Long)C*order*ns), false);
      Matrix<Real> Pc(order, order, take(nnode), false);

      for (Integer k = 0; k < COORD_DIM; k++)
        for (Integer i = 0; i < order; i++) for (Integer j = 0; j < order; j++)
          FS[k*order + (T.swap_ab ? j : i)][T.swap_ab ? i : j] = cs[k*nnode + (Long)i*order + j];

      Matrix<Real>::GEMM(Gm, FS, T.WbC);            // stage 1: collapsed index, value+derivative

      for (Long i0 = 0; i0 < ns; i0 += sblk) {
        const Long nb = std::min<Long>(sblk, ns-i0);
        // Stage 2a: [Ai; Adi] . [Mi | Mi'] gives value, d/d_alpha and d/d_beta in one GEMM
        // (the fourth quadrant is unused). Mi differs per s-node, so 2a stays per-node.
        for (Long b = 0; b < nb; b++) {
          const Long i = i0 + b;
          for (Integer k = 0; k < COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
            As[k][m] = Gm[k*order+m][i]; As[COORD_DIM+k][m] = Gm[k*order+m][ns+i];
          }
          Matrix<Real>::GEMM(Tmp, As, T.MiC[i]);
          for (Integer k = 0; k < COORD_DIM; k++) for (Integer m = 0; m < order; m++) {
            HG[b*NR + k][m]               = Tmp[k][m];
            HG[b*NR + COORD_DIM + k][m]   = Tmp[k][order+m];
            HG[b*NR + 2*COORD_DIM + k][m] = Tmp[COORD_DIM+k][m];
          }
        }
        { // Stage 2b: Tt is shared across s-nodes, so the whole block is a single GEMM.
          const Matrix<Real> HGb(nb*NR, order, (Iterator<Real>)HG.begin(), false);
          Matrix<Real> XdXb(nb*NR, nt, (Iterator<Real>)XdX.begin(), false);
          Matrix<Real>::GEMM(XdXb, HGb, Tt);
        }

        for (Long b = 0; b < nb; b++) {
          const Long i = i0 + b;
          const Real jw = tbl.sn[i]*T.J0*tbl.sw[i];
          for (Long j = 0; j < nt; j++) {
            const Long q = i*nt + j;
            const Real a0 = XdX[b*NR+COORD_DIM+0][j], a1 = XdX[b*NR+COORD_DIM+1][j], a2 = XdX[b*NR+COORD_DIM+2][j];
            const Real b0 = XdX[b*NR+2*COORD_DIM+0][j], b1 = XdX[b*NR+2*COORD_DIM+1][j], b2 = XdX[b*NR+2*COORD_DIM+2][j];
            const Real n0 = T.nsign*(a1*b2-a2*b1), n1 = T.nsign*(a2*b0-a0*b2), n2 = T.nsign*(a0*b1-a1*b0);
            const Real ar = sqrt<Real>(n0*n0+n1*n1+n2*n2), ia = (ar > 0 ? (Real)1/ar : (Real)0);
            for (Integer k = 0; k < COORD_DIM; k++) Xs[q*COORD_DIM+k] = XdX[b*NR+k][j];
            Xn[q*COORD_DIM+0] = n0*ia; Xn[q*COORD_DIM+1] = n1*ia; Xn[q*COORD_DIM+2] = n2*ia;
            wq[q] = ar*jw*tw[j];
          }
        }
      }

      ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xs, Xn);
      for (Long i = 0; i < ns; i++) for (Long j = 0; j < nt; j++) {
        const Long q = i*nt + j;
        for (Integer k0 = 0; k0 < KDIM0; k0++) for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          Real val;
          if (trg_dot_prod) {
            val = 0;
            for (Integer l = 0; l < COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*COORD_DIM+l]*normal_trg[l];
          } else val = Mker[q*KDIM0+k0][k1];
          KW[i*C + k0*KDIM1_out+k1][j] = val*wq[q];
        }
      }

      // Projection is the exact adjoint of stages 1-2b: same operators, reversed order.
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

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::SelfInteracBlockRP(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer nbeta_default) {
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

    // Nbeta GL points per direction for the (finitely smooth) post-COV integrand, decoupled from
    // the field order (Bruno 2018). cov_order_ if the user set it, else the tol-derived default.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : nbeta_default);
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
    // static initialization. SetupSelf (this serial SelfInterac) always precedes SetupNear,
    // whose NearInterac runs inside an OMP parallel region -- so warm BOTH the self- and the
    // near-scheme caches here to keep first-touch off the concurrent near path. ParamNodes /
    // DiffMat (used by IntegrateBlock on both paths) are warmed transitively by the rule
    // builds below. The scheme branches can differ (Hybrid = RP self + adaptive near), so the
    // near warm-up is not redundant with the self one.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : NbetaForDigits(digits)); // match the RP blocks' fallback
    if (qel.SelfUsesRectPolar()) {
      RPSelfRuleDispatch<order>(0, qel.cov_q_, Nbeta);
    } else if (qel.SelfUsesDuffy()) {
      DuffyTable<order>();                              // Duffy self: per-(order) triangle operators (ParamNodes/DiffMat come along)
    } else {
      CenteredURule<order, digits>(0, qel.max_depth_);  // centered self: graded u-rule (mutex-cached)
      CenteredVRule<order, digits>(0);                  // centered Alpert v-rule
    }
    if (qel.NearUsesRectPolar()) {
      GLRuleNbetaDispatch(Nbeta);  // near-RP: RectPolarNodes1D GL rule (also DiffMat/ParamNodes, already warm)
    } else if (qel.SelfUsesDuffy()) {
      NearGradeTableQ<order>(NearQuadOrderRt(digits));  // upstream near: full rung ladder built on first call
      NearBEllipseRt(digits); NearQuadOrderRt(digits);
    } else {
      NearGradeTable<order, digits>();  // split-near: normalized shell/core interval table + operators, keyed on (order,digits).
      NearBEllipse<digits>();           // near admissibility constant (end-foot reach). Not covered by the RP self warm-up in Hybrid.
      NearQuadOrder<digits>();          // near per-cell GL order.
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

  // ================ Upstream-ported near path (QuadScheme::Duffy only) ================
  // Runtime-digits split-at-foot near with a corner-angle GL-order bump and a deeper refinement
  // ladder. Isolated from the compile-time-digits NearInteracBlockSplit above so the Adaptive /
  // Hybrid / RectPolar paths are untouched.

  template <class Real> Integer QuadElemList<Real>::NearQuadOrderRt(const Integer digits) {
    static const Vector<Integer> q = []() {
      Vector<Integer> t(MaxDigitsCM);
      for (Integer d = 0; d < MaxDigitsCM; d++) { Real b; Integer qq; NearRhoRule(pow<Real,Long>((Real)0.1, (Long)d), b, qq); t[d] = qq; }
      return t;
    }();
    SCTL_ASSERT(digits >= 0 && digits < MaxDigitsCM);
    return q[digits];
  }

  template <class Real> Real QuadElemList<Real>::NearBEllipseRt(const Integer digits) {
    static const Vector<Real> b = []() {
      Vector<Real> t(MaxDigitsCM);
      for (Integer d = 0; d < MaxDigitsCM; d++) { Real bb; Integer qq; NearRhoRule(pow<Real,Long>((Real)0.1, (Long)d), bb, qq); t[d] = bb; }
      return t;
    }();
    SCTL_ASSERT(digits >= 0 && digits < MaxDigitsCM);
    return b[digits];
  }

  template <class Real> template <Integer order> const Vector<typename QuadElemList<Real>::GradeRule>& QuadElemList<Real>::NearGradeTableQ(const Integer q) {
    // Built once per `order`. Every entry is in NORMALIZED sub-element coordinates and carries no
    // positional index -- that is the point of splitting at the foot.
    auto build = [](const Integer q) {
      const Vector<Real>& gnds = ParamNodes(order);   // sub-element's own nodes, normalized
      Vector<Real> qn, qw; LegQuadRule<Real>::ComputeNdsWts(&qn, &qw, q);
      Vector<GradeRule> tab(2*MaxNearLvlCM);
      auto fill = [&](GradeRule& r, const Real a, const Real b) {
        r.a = a; r.b = b;
        const Real w = b - a;
        r.nds.ReInit(q); r.w.ReInit(q);
        for (Integer i = 0; i < q; i++) { r.nds[i] = a + w*qn[i]; r.w[i] = w*qw[i]; }
        // T[i][j] = Lhat_i(nds[j]): sub-element nodes -> this interval's quadrature nodes.
        r.T.ReInit(order, q);
        { Vector<Real> v(order*q, r.T.begin(), false); LagrangeInterp<Real>::Interpolate(v, gnds, r.nds); }
        r.dT.ReInit(order, q);
        Matrix<Real>::GEMM(r.dT, DiffMat(order), r.T);
        r.TT.ReInit(q, order); r.TD.ReInit(2*q, order);
        for (Integer i = 0; i < order; i++) for (Integer a = 0; a < q; a++) {
          r.TT[a][i] = r.T[i][a]; r.TD[a][i] = r.T[i][a]; r.TD[q+a][i] = r.dT[i][a];
        }
      };
      for (Integer k = 0; k < MaxNearLvlCM; k++) {
        const Real lo = 1 - pow<Real>((Real)0.5, k), hi = 1 - pow<Real>((Real)0.5, k+1);
        fill(tab[k], lo, hi);                                   // shell_k
        fill(tab[MaxNearLvlCM + k], lo, (Real)1);               // core_k = [1-2^-k, 1]
      }
      return tab;
    };
    // One static init builds every rung the corner-angle correction can select: each multiple of 4
    // up to NearMaxQuadOrderCM, plus each accuracy level's isotropic order (which need not be a
    // multiple of 4). The per-target lookup has to be O(1) and allocation-free.
    static const std::vector<Vector<GradeRule>> all = [&build]() {
      std::vector<Vector<GradeRule>> t(NearMaxQuadOrderCM+1);
      for (Integer qq = 4; qq <= NearMaxQuadOrderCM; qq += 4) t[qq] = build(qq);
      for (Integer d = 0; d < MaxDigitsCM; d++) {
        const Integer qi = NearQuadOrderRt(d);
        if (qi > 0 && qi <= NearMaxQuadOrderCM && t[qi].Dim() == 0) t[qi] = build(qi);
      }
      return t;
    }();
    SCTL_ASSERT(q > 0 && q <= NearMaxQuadOrderCM && all[q].Dim());
    return all[q];
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::IntegrateNearCM(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker, const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD, const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT, const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm) {
    // One near leaf cell: accumulate its tensor-product quadrature (weights wu (x) wv) against the
    // target into acc_cm. src_nodal is the caller's target-shifted nodal slab, so the kernel target
    // sits at the origin. Tensor grid is u-slow/v-fast: node (a,b) has flat index a*Nv+b.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const Long Nu = Mu.Dim(1), Nv = Mv.Dim(1), nq = Nu * Nv;
    if (!nq) return;
    const Integer C = KDIM0 * KDIM1_out;

    thread_local Vector<Real> Cv, Cdv;
    if (Cv.Dim() != COORD_DIM*order*Nv) { Cv.ReInit(COORD_DIM*order*Nv); Cdv.ReInit(COORD_DIM*order*Nv); }
    {
      const Matrix<Real> cs_all(COORD_DIM*order, order, (Iterator<Real>)src_nodal.begin(), false);
      Matrix<Real> Cv_all (COORD_DIM*order, Nv, Cv.begin(),  false);
      Matrix<Real> Cdv_all(COORD_DIM*order, Nv, Cdv.begin(), false);
      Matrix<Real>::GEMM(Cv_all,  cs_all, Mv);
      Matrix<Real>::GEMM(Cdv_all, cs_all, dMv);
    }
    const Long ldc = COORD_DIM*Nv;
    thread_local Vector<Real> Cvc, Cdvc, XdU, dXdv_soa;
    if (Cvc.Dim() != (Long)order*ldc) { Cvc.ReInit((Long)order*ldc); Cdvc.ReInit((Long)order*ldc); }
    for (Integer k = 0; k < COORD_DIM; k++) {
      for (Integer i = 0; i < order; i++) {
        const Long src = ((Long)k*order + i)*Nv, dst = (Long)i*ldc + k*Nv;
        for (Long b = 0; b < Nv; b++) { Cvc[dst+b] = Cv[src+b]; Cdvc[dst+b] = Cdv[src+b]; }
      }
    }
    if (XdU.Dim() != 2*(Long)Nu*ldc) { XdU.ReInit(2*(Long)Nu*ldc); dXdv_soa.ReInit((Long)Nu*ldc); }
    {
      const Matrix<Real> Cvc_m(order, ldc, Cvc.begin(), false), Cdvc_m(order, ldc, Cdvc.begin(), false);
      Matrix<Real> dV_m(Nu, ldc, dXdv_soa.begin(), false);
      { Matrix<Real> XdU_m(2*Nu, ldc, XdU.begin(), false); Matrix<Real>::GEMM(XdU_m, MuD, Cvc_m); }
      Matrix<Real>::GEMM(dV_m, MuT, Cdvc_m);
    }

    StaticArray<Real,COORD_DIM> Xt0_{0, 0, 0};
    const Vector<Real> Xt0_v_(COORD_DIM, Xt0_, false);
    thread_local Vector<Real> Xsrc, Xnsrc, wq;
    if (Xsrc.Dim() != nq*COORD_DIM) { Xsrc.ReInit(nq*COORD_DIM); Xnsrc.ReInit(nq*COORD_DIM); wq.ReInit(nq); }
    for (Long a = 0; a < Nu; a++) {
      for (Long b = 0; b < Nv; b++) {
        const Long q = a*Nv + b;
        const Long r = (Long)a*ldc + b, ru = ((Long)Nu + a)*ldc + b;
        const Real du0 = XdU[ru+0*Nv], du1 = XdU[ru+1*Nv], du2 = XdU[ru+2*Nv];
        const Real dv0 = dXdv_soa[r+0*Nv], dv1 = dXdv_soa[r+1*Nv], dv2 = dXdv_soa[r+2*Nv];
        const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
        const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
        const Real inv_area = (area > 0 ? nrm_sign/area : 0);
        Xsrc[q*COORD_DIM+0] = XdU[r+0*Nv]; Xsrc[q*COORD_DIM+1] = XdU[r+1*Nv]; Xsrc[q*COORD_DIM+2] = XdU[r+2*Nv];
        Xnsrc[q*COORD_DIM+0] = n0*inv_area; Xnsrc[q*COORD_DIM+1] = n1*inv_area; Xnsrc[q*COORD_DIM+2] = n2*inv_area;
        wq[q] = area*wu[a]*wv[b];
      }
    }

    thread_local Matrix<Real> Mker;
    ker.template KernelMatrix<Real,false>(Mker, Xt0_v_, Xsrc, Xnsrc); // (nq*KDIM0 x KDIM1full)

    thread_local Vector<Real> KWc;
    if (KWc.Dim() != C*nq) KWc.ReInit(C*nq);
    for (Long q = 0; q < nq; q++) {
      for (Integer k0 = 0; k0 < KDIM0; k0++) {
        for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          Real val;
          if (trg_dot_prod) { val = 0; for (Integer l = 0; l < COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * normal_trg[l]; }
          else { val = Mker[q*KDIM0+k0][k1]; }
          KWc[(Long)(k0*KDIM1_out+k1)*nq + q] = val*wq[q];
        }
      }
    }

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
      Matrix<Real>::GEMM(A_c, Mu, Y_c, (Real)1);   // beta = 1: accumulate in place
    }
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockSplitDuffy(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits) {
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Long nnode = (Long)order*order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full/COORD_DIM : KDIM1full;
    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    const Integer C_ = KDIM0*KDIM1_out;
    thread_local Vector<Real> acc, accB, accE;
    if (acc.Dim() != (Long)C_*nnode) { acc.ReInit((Long)C_*nnode); accB.ReInit((Long)C_*nnode); accE.ReInit(nnode); }
    M_acc.SetZero();

    const Real b_ellipse = NearBEllipseRt(digits);

    Real ustar, vstar;
    const Real dist = qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    Real Xc[COORD_DIM], dXu_[COORD_DIM], dXv_[COORD_DIM];
    qel.EvalPoint(Xc, dXu_, dXv_, ustar, vstar, elem_idx, nullptr);
    // Corner-angle correction to the near GL order. The required order is flat to ~120 deg, then
    // grows like 1/(180-phi) as the corner flattens and the element wraps around the target; the
    // parameter-space admissibility test cannot see this. phi is the acute angle between the
    // surface tangents at the foot -- the corner the target actually sees.
    const auto near_order = [](const Real* dXu, const Real* dXv, const Integer q_iso) {
      Real guu=0, gvv=0, guv=0;
      for (Integer k = 0; k < COORD_DIM; k++) { guu+=dXu[k]*dXu[k]; gvv+=dXv[k]*dXv[k]; guv+=dXu[k]*dXv[k]; }
      const double den = std::sqrt((double)guu*(double)gvv);
      if (!(den > 0)) return q_iso;
      const double c = std::min(1.0, std::fabs((double)guv)/den);
      const double phi = std::acos(c)*180.0/const_pi<double>();
      constexpr double Ck = 400.0;   // fitted on Laplace SL/DL, flat elements, one target offset
      const double f = std::max(1.0, Ck/(10.0*std::max(1e-3, phi)));
      if (f <= 1.0) return q_iso;
      Integer q = (Integer)std::ceil(f*(double)q_iso);
      q = ((q + 3)/4)*4;                                  // snap to the precomputed ladder
      return std::min<Integer>(NearMaxQuadOrderCM, std::max<Integer>(q_iso, q));
    };
    const Vector<GradeRule>& tab = NearGradeTableQ<order>(near_order(&dXu_[0], &dXv_[0], NearQuadOrderRt(digits)));
    Real su2 = 0, sv2 = 0;
    for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXu_[k]*dXu_[k]; sv2 += dXv_[k]*dXv_[k]; }
    const Real spd_u = sqrt<Real>(su2), spd_v = sqrt<Real>(sv2);

    const Vector<Real>& gnds = ParamNodes(order);
    const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};

    thread_local Vector<Real> cs;
    if (cs.Dim() != COORD_DIM*nnode) cs.ReInit(COORD_DIM*nnode);
    {
      const Long base = elem_idx * nnode * COORD_DIM;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) cs[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
      }
    }
    thread_local Matrix<Real> Sf[2][2], St[2][2];
    thread_local Vector<Real> sub, Sbuf;
    if (sub.Dim() != order) { sub.ReInit(order); Sbuf.ReInit(nnode); }
    for (Integer d = 0; d < 2; d++) {
      const Real xs = (d ? vstar : ustar);
      for (Integer sd = 0; sd < 2; sd++) {
        if (!(slen[d][sd] > 0)) continue;
        for (Integer i = 0; i < order; i++) sub[i] = sd ? (1 - (1-xs)*gnds[i]) : (xs*gnds[i]);
        { Vector<Real> v(nnode, Sbuf.begin(), false); LagrangeInterp<Real>::Interpolate(v, gnds, sub); }
        Sf[d][sd].ReInit(order, order); St[d][sd].ReInit(order, order);
        for (Integer i = 0; i < order; i++) for (Integer aa = 0; aa < order; aa++) {
          Sf[d][sd][i][aa] = Sbuf[i*order+aa];   // S
          St[d][sd][aa][i] = Sbuf[i*order+aa];   // S^T
        }
      }
    }
    thread_local Vector<Real> Av[2], Xsub[2][2];
    for (Integer sdv = 0; sdv < 2; sdv++) {
      if (!(slen[1][sdv] > 0)) continue;
      if (Av[sdv].Dim() != COORD_DIM*nnode) Av[sdv].ReInit(COORD_DIM*nnode);
      const Matrix<Real> cs_all(COORD_DIM*order, order, cs.begin(), false);
      Matrix<Real> A_all(COORD_DIM*order, order, Av[sdv].begin(), false);
      Matrix<Real>::GEMM(A_all, cs_all, Sf[1][sdv]);
    }
    for (Integer sdu = 0; sdu < 2; sdu++) {
      if (!(slen[0][sdu] > 0)) continue;
      for (Integer sdv = 0; sdv < 2; sdv++) {
        if (!(slen[1][sdv] > 0)) continue;
        if (Xsub[sdu][sdv].Dim() != COORD_DIM*nnode) Xsub[sdu][sdv].ReInit(COORD_DIM*nnode);
        for (Integer k = 0; k < COORD_DIM; k++) {
          const Matrix<Real> A_k(order, order, Av[sdv].begin() + k*nnode, false);
          Matrix<Real> X_k(order, order, Xsub[sdu][sdv].begin() + k*nnode, false);
          Matrix<Real>::GEMM(X_k, St[0][sdu], A_k);
        }
      }
    }

    auto emit = [&](Integer sdu, Integer sdv, Integer iu, Integer iv) {
      const GradeRule& gu = tab[iu];
      const GradeRule& gv = tab[iv];
      if (!(gu.b > gu.a) || !(gv.b > gv.a)) return;
      const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
      IntegrateNearCM<order>(normal_trg, gu.w, gv.w, ker,
                             gu.T, gu.TT, gu.TD, gv.T, gv.dT, gv.TT,
                             Xsub[sdu][sdv], nsign, acc);
    };
    for (Integer sdu = 0; sdu < 2; sdu++) {
      if (!(slen[0][sdu] > 0)) continue;
      for (Integer sdv = 0; sdv < 2; sdv++) {
        if (!(slen[1][sdv] > 0)) continue;
        acc.SetZero();
        Integer ku = 0, kv = 0;
        Real hu = slen[0][sdu]*spd_u, hv = slen[1][sdv]*spd_v;
        const bool cap = !(dist > 0) || !std::isfinite((double)dist);
        constexpr Integer KMAX = MaxNearLvlCM-1;
        while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
          if (hu >= hv && ku < KMAX) { emit(sdu, sdv, ku, MaxNearLvlCM + kv); ku++; hu *= (Real)0.5; }
          else if (kv < KMAX) { emit(sdu, sdv, MaxNearLvlCM + ku, kv); kv++; hv *= (Real)0.5; }
          else if (ku < KMAX) { emit(sdu, sdv, ku, MaxNearLvlCM + kv); ku++; hu *= (Real)0.5; }
          else break;
        }
        emit(sdu, sdv, MaxNearLvlCM + ku, MaxNearLvlCM + kv);      // terminal corner cell

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

  template <class Real> Real QuadElemList<Real>::GetClosestPoint(Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg, Integer* n_iter, bool* used_fallback) const {
    // Closest point on patch to Xtrg over (u,v) in [0,1]^2. Minimize 1/2|y-x|^2 by an
    // ACTIVE-SET Gauss-Newton (first fundamental form), seeded by the nearest node, clamped with
    // backtracking; shrinking-box grid search is the fallback if Newton stalls. The step is
    // computed AFTER the KKT test and in the FREE subspace: a coordinate pinned at a bound by an
    // outward gradient is held fixed, so metric coupling F cannot contaminate the surviving
    // component with the constrained gradient (this is what makes edge/corner feet converge
    // cleanly instead of bailing to the grid-search fallback).
    //
    // r^2 at (u,v). Target-centering (origin = Xtrg) keeps the residual accurate near
    // the surface, locating the foot sharply for near-touching targets.
    auto dist2_at = [&](const Real uu, const Real vv) -> Real {
      Real X[COORD_DIM];
      EvalPoint(X, nullptr, nullptr, uu, vv, elem_idx, &Xtrg);
      Real r2 = 0; for (Integer k = 0; k < COORD_DIM; k++) r2 += X[k]*X[k];
      return r2;
    };

    Real u, v;
    const Real d0 = GetClosestNode(u, v, elem_idx, Xtrg);
    Real f = d0 * d0;

    constexpr Integer max_iter = 30;
    const Real utol = (Real)machine_eps<Real>() * 64;
    const Real gtol = sqrt<Real>(machine_eps<Real>()) * 16;
    // Roundoff-tolerant line-search accept: a near-touching foot sits at the f=|y-x|^2 rounding
    // floor, where no clamped step gives STRICT decrease -- so a strict `fn < f` test false-stalls
    // and drops to the (expensive) grid-search fallback. Accept a step within rounding of f instead
    // (CSBQ does exactly this in its slender near solve: slender_element.cpp, d2 < d2_*(1+sqrt(eps))).
    const Real c_eps = machine_eps<Real>() * 8;
    // Relaxed first-order optimality scale for the stall branch: a point that failed the strict KKT
    // test above may still be stationary to rounding; accept it as converged rather than falling back.
    const Real gtol_stall = sqrt<Real>(machine_eps<Real>()) * 256;
    bool converged = false;
    Integer iters = 0;
    for (Integer it = 0; it < max_iter; it++) {
      iters = it + 1;
      Real X[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
      EvalPoint(X, dXu, dXv, u, v, elem_idx, &Xtrg); // X = y(u,v) - Xtrg

      Real E = 0, F = 0, G = 0, gu = 0, gv = 0;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real r = X[k], a = dXu[k], b = dXv[k];
        E += a*a; F += a*b; G += b*b;
        gu += r*a; gv += r*b;
      }

      // First-order optimality (KKT) via the PROJECTED gradient.
      Real Pu = gu, Pv = gv;
      if      (u <= 0) Pu = std::min<Real>(gu, (Real)0);
      else if (u >= 1) Pu = std::max<Real>(gu, (Real)0);
      if      (v <= 0) Pv = std::min<Real>(gv, (Real)0);
      else if (v >= 1) Pv = std::max<Real>(gv, (Real)0);
      const bool opt_u = (fabs(Pu) <= gtol * sqrt<Real>(E*f));
      const bool opt_v = (fabs(Pv) <= gtol * sqrt<Real>(G*f));
      if (opt_u && opt_v) { converged = true; break; }

      // ACTIVE-SET reduced Gauss-Newton step: a coordinate pinned at a bound by an outward gradient
      // is held FIXED and the step is solved in the free subspace only. u_act implies opt_u (Pu is
      // then exactly 0), so both-active is already converged above.
      const bool u_act = ((u <= 0 && gu >= 0) || (u >= 1 && gu <= 0));
      const bool v_act = ((v <= 0 && gv >= 0) || (v >= 1 && gv <= 0));
      Real du = 0, dv = 0;
      if (!u_act && !v_act) {
        const Real det = E*G - F*F;
        if (fabs(det) > (Real)1e-30 * (E*G + F*F + 1)) {
          du = ( G*gu - F*gv) / det;   // interior: full 2D Gauss-Newton
          dv = (-F*gu + E*gv) / det;
        } else {                       // degenerate metric (patch corner): scaled gradient
          du = gu / (E + (Real)1e-30);
          dv = gv / (G + (Real)1e-30);
        }
      } else if (u_act) {
        dv = gv / (G + (Real)1e-30);   // 1D Newton along the v-edge, u held at its bound
      } else {
        du = gu / (E + (Real)1e-30);   // 1D Newton along the u-edge, v held at its bound
      }

      // Backtrack on the clamped Newton step until f decreases (or is within its rounding floor).
      Real lambda = 1;
      bool improved = false;
      Real un = u, vn = v, fn = f;
      for (Integer ls = 0; ls < 40; ls++) {
        un = std::min<Real>(1, std::max<Real>(0, u - lambda*du));
        vn = std::min<Real>(1, std::max<Real>(0, v - lambda*dv));
        fn = dist2_at(un, vn);
        if (fn <= f * (1 + c_eps)) { improved = true; break; }   // decrease OR within f's rounding floor
        lambda *= (Real)0.5;
      }
      // If the clamped Newton step stalls, retry along the projected (metric-scaled) gradient.
      if (!improved) {
        const Real gu_s = Pu / (E + (Real)1e-30), gv_s = Pv / (G + (Real)1e-30);
        lambda = 1;
        for (Integer ls = 0; ls < 40; ls++) {
          un = std::min<Real>(1, std::max<Real>(0, u - lambda*gu_s));
          vn = std::min<Real>(1, std::max<Real>(0, v - lambda*gv_s));
          fn = dist2_at(un, vn);
          if (fn <= f * (1 + c_eps)) { improved = true; break; }   // decrease OR within f's rounding floor
          lambda *= (Real)0.5;
        }
      }
      if (!improved) {
        // No accepted step from either line search. Treat an already-stationary or step-floored
        // point (after >=1 successful step) as converged instead of dropping to the grid search --
        // the false stalls that dominate the near-setup cost live here. Genuine non-convergence
        // (first-iteration bad basin, ill-conditioned metric) still falls back.
        const bool stat = (fabs(Pu) <= gtol_stall * sqrt<Real>(E*f)) && (fabs(Pv) <= gtol_stall * sqrt<Real>(G*f));
        const bool tiny = (fabs(du) < utol && fabs(dv) < utol);
        if (iters > 1 && (stat || tiny)) converged = true;
        break;
      }
      const bool small_step = (fabs(un-u) < utol && fabs(vn-v) < utol);
      u = un; v = vn; f = fn;
      if (small_step) { converged = true; break; }
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
    if (n_iter) *n_iter = iters;
    if (used_fallback) *used_fallback = !converged;
    return sqrt<Real>(f);
  }

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self) {
    // Per-target near-singular interaction (off-surface targets). Dispatches to the
    // scheme's near block: foot-graded separable tensor (Adaptive/Hybrid), RectPolar,
    // or Duffy. On-surface self interactions are built by SelfInterac.
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
      // Duffy scheme: upstream-ported near (corner-angle order + deeper ladder). Others:
      // Adaptive/Hybrid foot-graded separable-tensor near (RP handled by the guard inside the
      // block). NearInteracBlockGraded matches RP near accuracy under parametric shear, where
      // the isotropic-quadtree NearInteracBlockSplit lost ~2-3 orders even on smooth geometry.
      if (qel.SelfUsesDuffy()) NearInteracBlockSplitDuffy<order>(M_acc, qel, elem_idx, Xtrg, ntrg, ker, digits);
      else NearInteracBlockGraded<digits, order>(M_acc, qel, elem_idx, Xtrg, ntrg, ker);

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
    // Reconstruct the split-at-foot near quadrature for Xtrg and dump its per-cell GL nodes as a
    // VTK_QUAD mesh (mirrors NearInteracBlockSplit's cell layout). Target in a separate file.
    Real b_ellipse; Integer QuadOrder;
    NearRhoRule(tol, b_ellipse, QuadOrder);

    VTUData vtu;
    const Vector<Real>& qnds = ParamNodes(QuadOrder);
    Vector<Real> u_param(QuadOrder), v_param(QuadOrder), Xg;

    // One quadrature cell -> its own QuadOrder x QuadOrder GL node patch of VTK_QUAD cells;
    // point_offset resets per cell so cells never bridge two quadrature cells.
    auto emit_cell = [&](const Real u0, const Real u1, const Real v0, const Real v1) {
      const Real du = u1-u0, dv = v1-v0;
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
    };

    {
      // Replicate NearInteracBlockSplit's cell layout: split at the foot (u*,v*), then refine each
      // quadrant with an ISOTROPIC graded quadtree -- at each level quadrisect the corner cell,
      // emit its 3 non-corner children, recurse into the corner. Flat index i -> normalized
      // interval [a,b]: shell_k=[1-2^-k,1-2^-(k+1)] for i=k<L, core_k=[1-2^-k,1] for i=L+k.
      // Mapped back to actual param through the sub-element affine map.
      Real ustar, vstar;
      const Real dist = GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
      Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
      EvalPoint(Xc, dXu, dXv, ustar, vstar, elem_idx, nullptr);
      Real su2 = 0, sv2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXu[k]*dXu[k]; sv2 += dXv[k]*dXv[k]; }
      const Real spd_u = sqrt<Real>(su2), spd_v = sqrt<Real>(sv2);
      const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};
      const Integer ovr = NearMaxLvlOverride();
      const Integer KMAX = std::min<Integer>(ovr ? ovr : max_depth_, MaxNearLvl-1);
      const bool cap = !(dist > 0) || !std::isfinite((double)dist);
      auto ivl = [](Integer i, Real& a, Real& b) {                 // flat index -> normalized [a,b]
        const Integer k = (i < MaxNearLvl ? i : i - MaxNearLvl);
        a = 1 - pow<Real>((Real)0.5, k);
        b = (i < MaxNearLvl) ? 1 - pow<Real>((Real)0.5, k+1) : (Real)1;
      };
      // normalized x in [0,1] (x=1 at the foot) -> actual param on this sub-element side.
      auto mp = [](Real a, Real b, Integer sd, Real xs, Real& lo, Real& hi) {
        if (sd == 0) { lo = xs*a; hi = xs*b; } else { lo = 1-(1-xs)*b; hi = 1-(1-xs)*a; }
      };
      auto cell = [&](Integer sdu, Integer sdv, Integer iu, Integer iv) {
        Real au,bu,av,bv; ivl(iu,au,bu); ivl(iv,av,bv);
        if (!(bu > au) || !(bv > av)) return;
        Real u0,u1,v0,v1; mp(au,bu,sdu,ustar,u0,u1); mp(av,bv,sdv,vstar,v0,v1);
        emit_cell(u0,u1,v0,v1);
      };
      for (Integer sdu = 0; sdu < 2; sdu++) {
        if (!(slen[0][sdu] > 0)) continue;
        for (Integer sdv = 0; sdv < 2; sdv++) {
          if (!(slen[1][sdv] > 0)) continue;
          Real hu = slen[0][sdu]*spd_u, hv = slen[1][sdv]*spd_v;
          Integer L = 0;
          while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && L < KMAX) {
            cell(sdu,sdv,L,L);                     // shell_L x shell_L
            cell(sdu,sdv,L,MaxNearLvl+L+1);        // shell_L x core_{L+1}
            cell(sdu,sdv,MaxNearLvl+L+1,L);        // core_{L+1} x shell_L
            L++; hu *= (Real)0.5; hv *= (Real)0.5;
          }
          cell(sdu,sdv,MaxNearLvl+L,MaxNearLvl+L);   // terminal corner cell
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

  template <class Real> void QuadElemList<Real>::WriteNearInteracGradedVTK(const std::string& fname, const Long elem_idx, const Vector<Real>& Xtrg, const Real tol, const Comm& comm) const {
    // Reconstruct THE production adaptive near rule (BuildNearTensorRule): the foot-graded
    // separable tensor grid over the whole panel, and dump each (u-seg x v-seg) cell as its own
    // QuadOrder x QuadOrder GL-node patch of VTK_QUAD cells (mirrors NearInteracBlockGraded's rule,
    // NOT the superseded isotropic quadtree of WriteNearInteracVTK). Cells cluster toward the foot
    // (u*,v*) and read as four quadrants meeting there. Target in a separate file.
    Real b_ellipse; Integer QuadOrder;
    NearRhoRule(tol, b_ellipse, QuadOrder);
    Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);

    // Two 1D foot-graded partitions (in u and v) + their full tensor product, with the per-side
    // segment boundaries/depths so cells can be delineated and colored.
    Vector<Real> u_param, wu, v_param, wv, useg, vseg;
    Vector<Long> useg_depth, vseg_depth;
    BuildNearTensorRule(u_param, wu, v_param, wv, &useg, &useg_depth, &vseg, &vseg_depth,
                        *this, elem_idx, Xtrg, b_ellipse, qnds, qwts, max_depth_);
    const Long nu_seg = useg.Dim()/2, nv_seg = vseg.Dim()/2;

    VTUData vtu;
    Vector<Real> u_cell(QuadOrder), v_cell(QuadOrder), Xg;
    // One (u-seg si) x (v-seg sj) cell -> its own QuadOrder x QuadOrder GL node patch of VTK_QUAD
    // cells; point_offset resets per cell so cells never bridge two quadrature cells. Colored by
    // the cell's grading depth = max(u-side depth, v-side depth).
    for (Long si = 0; si < nu_seg; si++) {
      for (Integer a = 0; a < QuadOrder; a++) u_cell[a] = u_param[si*QuadOrder + a];
      for (Long sj = 0; sj < nv_seg; sj++) {
        for (Integer b = 0; b < QuadOrder; b++) v_cell[b] = v_param[sj*QuadOrder + b];
        GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, u_cell, v_cell, elem_idx);

        const Long point_offset = vtu.coord.Dim() / COORD_DIM;
        const Real depth = (Real)std::max<Long>(useg_depth[si], vseg_depth[sj]);
        for (Long p = 0; p < (Long)QuadOrder*QuadOrder; p++) {
          for (Integer k = 0; k < COORD_DIM; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[p*COORD_DIM+k]);
          vtu.value.PushBack((VTUData::VTKReal)depth);
        }
        for (Long i = 0; i < QuadOrder - 1; i++) {
          for (Long j = 0; j < QuadOrder - 1; j++) {
            const Long idx = point_offset + i * QuadOrder + j;
            vtu.connect.PushBack(idx);
            vtu.connect.PushBack(idx + 1);
            vtu.connect.PushBack(idx + QuadOrder + 1);
            vtu.connect.PushBack(idx + QuadOrder);
            vtu.offset.PushBack(vtu.connect.Dim());
            vtu.types.PushBack(9); // VTK_QUAD
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
      // Tensor product of the centered graded-GL u nodes and the centered Alpert v nodes (the rule
      // SelfInteracBlock integrates) -> VTK_VERTEX. The centered builders return OFFSETS from
      // (u0,v0); add them back to get actual parameters.
      Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);
      // Composite-graded v-rule levels + matched u-depth from tol (matches CenteredVRule/CenteredURule).
      const Integer digits = std::max<Integer>(0, (Integer)std::lround(-std::log10((double)std::max<Real>(tol, machine_eps<Real>()))));
      const Integer Lvl = VLevelsForDigits(digits);
      Vector<Real> du, wu, dv, wv, u_param, v_param, Xg;
      BuildCenteredGraded1D(du, wu, u0, max_depth_, qnds, qwts);
      LogSingularQuad1DCentered(dv, wv, v0, Lvl, QuadOrder);
      u_param.ReInit(du.Dim()); for (Long i = 0; i < du.Dim(); i++) u_param[i] = u0 + du[i];
      v_param.ReInit(dv.Dim()); for (Long i = 0; i < dv.Dim(); i++) v_param[i] = v0 + dv[i];
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

  template <class Real> void QuadElemList<Real>::WriteSelfInteracGradedVTK(const std::string& fname, const Long elem_idx, const Real u0, const Real v0, const Real tol, const Comm& comm) const {
    // Reconstruct the on-surface self rule as PANELS, matching SelfInteracBlock's tensor of
    // centered graded-GL u-panels (BuildCenteredGraded1D) x centered composite-v panels
    // (LogSingularQuad1DCentered). Every panel is a GL x GL patch EXCEPT the innermost v-row
    // touching v0, whose v-nodes are the Alpert log-singular rule (still ordered within the panel).
    // Each (u-panel, v-panel) cell -> its own VTK_QUAD patch; the point scalar flags the Alpert row.
    Real b_ellipse; Integer QuadOrder;
    QuadParams(tol, b_ellipse, QuadOrder);
    const Integer digits = std::max<Integer>(0, (Integer)std::lround(-std::log10((double)std::max<Real>(tol, machine_eps<Real>()))));
    const Integer Lvl = VLevelsForDigits(digits);

    // Full node sets (offsets from (u0,v0)) -- the exact nodes SelfInteracBlock integrates.
    Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, QuadOrder);
    Vector<Real> du, wu, dv, wv;
    BuildCenteredGraded1D(du, wu, u0, max_depth_, qnds, qwts);
    LogSingularQuad1DCentered(dv, wv, v0, Lvl, QuadOrder);

    // Panel intervals in param space (mirror the two builders' panel layouts exactly).
    std::vector<std::pair<Real,Real>> upan, vpan;
    std::vector<int> valert; // per v-panel: 1 iff Alpert singular row
    auto push_iv = [](std::vector<std::pair<Real,Real>>& P, const Real p0, const Real p1) {
      const Real lo = std::min(p0,p1), hi = std::max(p0,p1);
      if (hi > lo) P.push_back({lo,hi});
    };
    // u: geometric-graded GL panels marching outward on each side of u0 (BuildCenteredGraded1D).
    auto uside = [&](const Real Len, const Real sgn) {
      if (!(Len > 0)) return;
      Real a = 0;
      for (Integer k = max_depth_; k >= 0; k--) { const Real b = Len*pow<Real>((Real)0.5,(Integer)k); push_iv(upan, u0+sgn*a, u0+sgn*b); a = b; }
    };
    uside(1-u0, (Real)+1); uside(u0, (Real)-1);
    // v: Lvl geometric-graded GL panels + 1 Alpert panel touching v0, per side (LogSingularQuad1DCentered).
    auto vside = [&](const Real Len, const Real sgn) {
      if (!(Len > 0)) return;
      Real prev = Len;
      for (Integer i = 1; i <= Lvl; i++) { const Real bnd = Len*pow<Real>((Real)0.5,(Integer)i); push_iv(vpan, v0+sgn*bnd, v0+sgn*prev); valert.push_back(0); prev = bnd; }
      push_iv(vpan, v0, v0+sgn*prev); valert.push_back(1); // innermost: Alpert, touches v0
    };
    vside(1-v0, (Real)+1); vside(v0, (Real)-1);

    // Bucket every node into its panel (nearest-containing; no node sits exactly on a boundary),
    // then sort within the panel so consecutive indices form a structured grid (Alpert row included).
    auto bucket = [](const Vector<Real>& delta, const Real center, const std::vector<std::pair<Real,Real>>& P, std::vector<std::vector<Real>>& out) {
      out.assign(P.size(), {});
      for (Long i = 0; i < delta.Dim(); i++) {
        const Real x = center + delta[i];
        Long best = 0; Real bestd = -1;
        for (Long k = 0; k < (Long)P.size(); k++) {
          const Real d = (x < P[k].first ? P[k].first - x : (x > P[k].second ? x - P[k].second : (Real)0));
          if (bestd < 0 || d < bestd) { bestd = d; best = k; }
        }
        out[best].push_back(x);
      }
      for (auto& v : out) std::sort(v.begin(), v.end());
    };
    std::vector<std::vector<Real>> unodes, vnodes;
    bucket(du, u0, upan, unodes);
    bucket(dv, v0, vpan, vnodes);

    VTUData vtu;
    Vector<Real> u_cell, v_cell, Xg;
    for (Long si = 0; si < (Long)unodes.size(); si++) {
      const Long nu = (Long)unodes[si].size();
      if (nu < 2) continue;
      u_cell.ReInit(nu); for (Long a = 0; a < nu; a++) u_cell[a] = unodes[si][a];
      for (Long sj = 0; sj < (Long)vnodes.size(); sj++) {
        const Long nv = (Long)vnodes[sj].size();
        if (nv < 2) continue;
        v_cell.ReInit(nv); for (Long b = 0; b < nv; b++) v_cell[b] = vnodes[sj][b];
        GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, u_cell, v_cell, elem_idx); // AoS, u slow / v fast

        const Long point_offset = vtu.coord.Dim() / COORD_DIM;
        const Real flag = (Real)valert[sj];
        for (Long p = 0; p < nu*nv; p++) {
          for (Integer k = 0; k < COORD_DIM; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[p*COORD_DIM+k]);
          vtu.value.PushBack((VTUData::VTKReal)flag);
        }
        for (Long i = 0; i < nu - 1; i++) {
          for (Long j = 0; j < nv - 1; j++) {
            const Long idx = point_offset + i*nv + j;
            vtu.connect.PushBack(idx);
            vtu.connect.PushBack(idx + 1);
            vtu.connect.PushBack(idx + nv + 1);
            vtu.connect.PushBack(idx + nv);
            vtu.offset.PushBack(vtu.connect.Dim());
            vtu.types.PushBack(9); // VTK_QUAD
          }
        }
      }
    }
    vtu.WriteVTK(fname, comm);

    // Singular point: a single VTK_VERTEX at the on-surface target (u0,v0).
    Vector<Real> us0(1), vs0(1), Xs; us0[0] = u0; vs0[0] = v0;
    GetGeom(&Xs, nullptr, nullptr, nullptr, nullptr, us0, vs0, elem_idx);
    VTUData singpt;
    for (Integer k = 0; k < COORD_DIM; k++) singpt.coord.PushBack((VTUData::VTKReal)Xs[k]);
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

  template <class Real> void QuadElemList<Real>::WriteSelfInteracDuffyVTK(const std::string& fname, const Long elem_idx, const Real u0, const Real v0, const Real tol, const Comm& comm) const {
    // Reconstruct the Duffy edge-collapsed self rule at (u0,v0): four target-anchored triangles
    // (base = one panel edge, apex = the singular point). Each triangle is parametrized by a radial
    // GL variable s in [0,1] (order points, s=0 at the apex) and an along-edge variable t in [0,1]
    // whose nodes are the sinh substitution clustered at the metric foot t*. Mirrors
    // SelfInteracBlockDuffy exactly. Each triangle -> its own (ns x nt) VTK_QUAD patch; the s=0 row
    // collapses onto the apex, so the cells fan out from the singular point. Point scalar = triangle
    // index 0..3. Singular point in a separate file.
    const Integer digits = std::max<Integer>(0, (Integer)std::lround(-std::log10((double)std::max<Real>(tol, machine_eps<Real>()))));
    const Integer ns = order;
    const Integer nt = DuffyTOrder(digits, order, /*kdim0 (scalar Laplace SL)*/ 1);

    Vector<Real> sn, sw, tn0, tw0;
    LegQuadRule<Real>::ComputeNdsWts(&sn, &sw, ns);   // s-nodes on [0,1] (radial, s=0 at apex)
    LegQuadRule<Real>::ComputeNdsWts(&tn0, &tw0, nt); // reference nodes the sinh map warps

    // Surface metric at (u0,v0): the foot t* and the sinh width are set by ON-surface distance.
    Real G[4];
    {
      Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
      EvalPoint(Xc, dXu, dXv, u0, v0, elem_idx, nullptr);
      Real guu = 0, guv = 0, gvv = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { guu += dXu[k]*dXu[k]; guv += dXu[k]*dXv[k]; gvv += dXv[k]*dXv[k]; }
      G[0] = guu; G[1] = guv; G[2] = guv; G[3] = gvv;
    }
    auto ash = [](const Real x) { return log<Real>(x + sqrt<Real>(x*x + (Real)1)); };
    const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};

    VTUData vtu;
    for (Integer kt = 0; kt < 4; kt++) {
      const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
      const Real b[2] = {cu[(kt+1)%4]-u0, cv[(kt+1)%4]-v0};
      const Real e[2] = {b[0]-a[0], b[1]-a[1]};
      const bool swap_ab = (fabs<Real>(e[0]) < fabs<Real>(e[1]));   // e axis-aligned; collapse beta
      const Real al0 = (swap_ab ? v0 : u0), be0 = (swap_ab ? u0 : v0);
      const Real aal = (swap_ab ? a[1] : a[0]), abe = (swap_ab ? a[0] : a[1]);
      const Real eal = (swap_ab ? e[1] : e[0]);

      Real tstar, dOverL;
      { // metric-aware foot and width (identical to SelfInteracBlockDuffy)
        const Real Me[2] = {G[0]*e[0]+G[1]*e[1], G[2]*e[0]+G[3]*e[1]};
        const Real am = e[0]*Me[0] + e[1]*Me[1];
        Real ts = -(a[0]*Me[0] + a[1]*Me[1])/am;
        ts = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
        const Real c[2] = {a[0]+ts*e[0], a[1]+ts*e[1]};
        const Real d2 = c[0]*(G[0]*c[0]+G[1]*c[1]) + c[1]*(G[2]*c[0]+G[3]*c[1]);
        tstar = ts; dOverL = sqrt<Real>(d2)/sqrt<Real>(am);
      }
      Vector<Real> tn(nt);
      { // t = t* + (d/L)*sinh(xi), xi linear on the reference nodes
        const Real dd = dOverL;
        const Real x0 = -ash(tstar/dd), x1 = ash(((Real)1-tstar)/dd);
        for (Integer j = 0; j < nt; j++) {
          const Real xi = x0 + (x1-x0)*tn0[j];
          const Real ex = exp<Real>(xi), iex = (Real)1/ex;
          tn[j] = tstar + dd*(ex-iex)/(Real)2;
        }
      }

      // (s,t) grid -> (u,v) -> physical, one point at a time (the map is non-separable).
      const Long point_offset = vtu.coord.Dim() / COORD_DIM;
      for (Integer i = 0; i < ns; i++) {
        const Real s = sn[i];
        for (Integer j = 0; j < nt; j++) {
          const Real t = tn[j];
          const Real alpha = al0 + s*(aal + t*eal);
          const Real beta  = be0 + s*abe;
          const Real u = (swap_ab ? beta : alpha), v = (swap_ab ? alpha : beta);
          Real Xc[COORD_DIM];
          EvalPoint(Xc, nullptr, nullptr, u, v, elem_idx, nullptr);
          for (Integer k = 0; k < COORD_DIM; k++) vtu.coord.PushBack((VTUData::VTKReal)Xc[k]);
          vtu.value.PushBack((VTUData::VTKReal)kt);
        }
      }
      for (Integer i = 0; i < ns-1; i++) {
        for (Integer j = 0; j < nt-1; j++) {
          const Long idx = point_offset + (Long)i*nt + j;
          vtu.connect.PushBack(idx);
          vtu.connect.PushBack(idx + 1);
          vtu.connect.PushBack(idx + nt + 1);
          vtu.connect.PushBack(idx + nt);
          vtu.offset.PushBack(vtu.connect.Dim());
          vtu.types.PushBack(9); // VTK_QUAD
        }
      }
    }
    vtu.WriteVTK(fname, comm);

    // Singular point: a single VTK_VERTEX at the on-surface apex (u0,v0).
    Real Xs[COORD_DIM];
    EvalPoint(Xs, nullptr, nullptr, u0, v0, elem_idx, nullptr);
    VTUData singpt;
    for (Integer k = 0; k < COORD_DIM; k++) singpt.coord.PushBack((VTUData::VTKReal)Xs[k]);
    singpt.value.PushBack(0);
    singpt.connect.PushBack(0);
    singpt.offset.PushBack(singpt.connect.Dim());
    singpt.types.PushBack(1);
    singpt.WriteVTK(fname + "-singpt", comm);
  }

  template <class Real> void QuadElemList<Real>::WriteNearInteracDuffyVTK(const std::string& fname, const Long elem_idx, const Vector<Real>& Xtrg, const Real tol, const Comm& comm) const {
    // Reconstruct the Duffy near rule for Xtrg and dump its per-cell GL nodes as a VTK_QUAD mesh
    // (mirrors NearInteracBlockSplitDuffy's split-at-foot layout with the ANISOTROPIC u/v ladder --
    // refine whichever direction is coarser on the surface -- unlike the isotropic quadtree of
    // WriteNearInteracVTK). QuadOrder and b_ellipse come from tol just as the solver does, including
    // the corner-angle order bump. Point scalar = cell refinement step. Target in a separate file.
    const Integer digits = std::max<Integer>(0, (Integer)std::lround(-std::log10((double)std::max<Real>(tol, machine_eps<Real>()))));
    const Real b_ellipse = NearBEllipseRt(digits);

    Real ustar, vstar;
    const Real dist = GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
    EvalPoint(Xc, dXu, dXv, ustar, vstar, elem_idx, nullptr);

    // Corner-angle order bump (identical to NearInteracBlockSplitDuffy). Flat/orthogonal panels
    // leave QuadOrder unchanged; strongly sheared feet raise it.
    Integer QuadOrder;
    {
      const Integer q_iso = NearQuadOrderRt(digits);
      Real guu=0, gvv=0, guv=0;
      for (Integer k = 0; k < COORD_DIM; k++) { guu+=dXu[k]*dXu[k]; gvv+=dXv[k]*dXv[k]; guv+=dXu[k]*dXv[k]; }
      const double den = std::sqrt((double)guu*(double)gvv);
      QuadOrder = q_iso;
      if (den > 0) {
        const double c = std::min(1.0, std::fabs((double)guv)/den);
        const double phi = std::acos(c)*180.0/const_pi<double>();
        const double f = std::max(1.0, 400.0/(10.0*std::max(1e-3, phi)));
        if (f > 1.0) { Integer q = ((Integer)std::ceil(f*(double)q_iso) + 3)/4*4; QuadOrder = std::min<Integer>(NearMaxQuadOrderCM, std::max<Integer>(q_iso, q)); }
      }
    }

    Real su2 = 0, sv2 = 0;
    for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXu[k]*dXu[k]; sv2 += dXv[k]*dXv[k]; }
    const Real spd_u = sqrt<Real>(su2), spd_v = sqrt<Real>(sv2);
    const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};

    VTUData vtu;
    const Vector<Real>& qnds = ParamNodes(QuadOrder);
    Vector<Real> u_param(QuadOrder), v_param(QuadOrder), Xg;

    // Flat cell index -> normalized [a,b] (x=1 at the foot): shell_k=[1-2^-k,1-2^-(k+1)] for
    // i=k<MaxNearLvlCM, core_k=[1-2^-k,1] for i=MaxNearLvlCM+k. Then the sub-element affine map.
    auto ivl = [](Integer i, Real& a, Real& b) {
      const Integer k = (i < MaxNearLvlCM ? i : i - MaxNearLvlCM);
      a = 1 - pow<Real>((Real)0.5, k);
      b = (i < MaxNearLvlCM) ? 1 - pow<Real>((Real)0.5, k+1) : (Real)1;
    };
    auto mp = [](Real a, Real b, Integer sd, Real xs, Real& lo, Real& hi) {
      if (sd == 0) { lo = xs*a; hi = xs*b; } else { lo = 1-(1-xs)*b; hi = 1-(1-xs)*a; }
    };
    auto emit_cell = [&](const Real u0, const Real u1, const Real v0, const Real v1, const Real level) {
      const Real du = u1-u0, dv = v1-v0;
      if (!(du > 0) || !(dv > 0)) return;
      for (Integer a = 0; a < QuadOrder; a++) u_param[a] = u0 + du*qnds[a];
      for (Integer b = 0; b < QuadOrder; b++) v_param[b] = v0 + dv*qnds[b];
      GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, u_param, v_param, elem_idx);
      const Long point_offset = vtu.coord.Dim() / COORD_DIM;
      for (Long p = 0; p < (Long)QuadOrder*QuadOrder; p++) {
        for (Integer k = 0; k < COORD_DIM; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[p*COORD_DIM+k]);
        vtu.value.PushBack((VTUData::VTKReal)level);
      }
      for (Long i = 0; i < QuadOrder-1; i++) {
        for (Long j = 0; j < QuadOrder-1; j++) {
          const Long idx = point_offset + i*QuadOrder + j;
          vtu.connect.PushBack(idx);
          vtu.connect.PushBack(idx + 1);
          vtu.connect.PushBack(idx + QuadOrder + 1);
          vtu.connect.PushBack(idx + QuadOrder);
          vtu.offset.PushBack(vtu.connect.Dim());
          vtu.types.PushBack(9); // VTK_QUAD
        }
      }
    };
    auto cell = [&](Integer sdu, Integer sdv, Integer iu, Integer iv, Real level) {
      Real au,bu,av,bv; ivl(iu,au,bu); ivl(iv,av,bv);
      Real u0,u1,v0,v1; mp(au,bu,sdu,ustar,u0,u1); mp(av,bv,sdv,vstar,v0,v1);
      emit_cell(u0,u1,v0,v1,level);
    };

    const bool cap = !(dist > 0) || !std::isfinite((double)dist);
    constexpr Integer KMAX = MaxNearLvlCM-1;
    for (Integer sdu = 0; sdu < 2; sdu++) {
      if (!(slen[0][sdu] > 0)) continue;
      for (Integer sdv = 0; sdv < 2; sdv++) {
        if (!(slen[1][sdv] > 0)) continue;
        Integer ku = 0, kv = 0;
        Real hu = slen[0][sdu]*spd_u, hv = slen[1][sdv]*spd_v;
        Real level = 0;
        while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
          if (hu >= hv && ku < KMAX) { cell(sdu, sdv, ku, MaxNearLvlCM + kv, level); ku++; hu *= (Real)0.5; }
          else if (kv < KMAX) { cell(sdu, sdv, MaxNearLvlCM + ku, kv, level); kv++; hv *= (Real)0.5; }
          else if (ku < KMAX) { cell(sdu, sdv, ku, MaxNearLvlCM + kv, level); ku++; hu *= (Real)0.5; }
          else break;
          level += 1;
        }
        cell(sdu, sdv, MaxNearLvlCM + ku, MaxNearLvlCM + kv, level);   // terminal corner cell
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
