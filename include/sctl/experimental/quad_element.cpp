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

  template <class Real> Integer QuadElemList<Real>::VLevelsForDigits(const Integer digits) {
    // Geometric grading levels per side toward v0 in the composite Alpert v-rule. 
    return std::min<Integer>(12, std::max<Integer>(1, digits - 5));
  }

  template <class Real> template <Integer digits> Integer QuadElemList<Real>::DigitsVLevels() {
    return VLevelsForDigits(digits);
  }

  template <class Real> Integer QuadElemList<Real>::NbetaForDigits(const Integer digits) {
    // Worst-case tol->Nbeta ladder for the RectPolar COV, calibrated on the maximally
    // twisted sphere (theta=pi, PatchPerFace=5, near R=1.001 column of Nbeta_sweep.txt):
    // smallest ladder Nbeta reaching 10^-digits with margin. RectPolar converges much
    // faster in Nbeta on near-flat geometry, so this is conservative there.
    if      (digits <= 2) return 128; // 1e-1..1e-2: 128 -> 7.6e-3
    else if (digits == 3) return 256; // 1e-3     : 256 -> 1.3e-4
    else if (digits <= 5) return 384; // 1e-4,1e-5: 384 -> 2.4e-6
    else                  return 512; // <=1e-6   : 512 -> 5.6e-8 (ladder max)
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
    thread_local const QuadElemList<Real>* cs_qel = nullptr;
    thread_local Long cs_elem = -1;
    thread_local StaticArray<Real,COORD_DIM> cs_trg{0,0,0};
    if (coord_shift.Dim() != COORD_DIM*nnode) { coord_shift.ReInit(COORD_DIM*nnode); cs_qel = nullptr; }
    // Near emits ~10 cells per target, all sharing (elem_idx, Xtrg), so the shift is hoisted
    // out of the per-cell loop by memoizing on that key rather than plumbing it through.
    // src_nodal: caller already produced the target-shifted nodal slab for this sub-element
    // (near split does it once per target), so bind a view rather than rebuilding or copying.
    if (!src_nodal && (cs_qel != &qel || cs_elem != elem_idx || cs_trg[0] != Xtrg[0] || cs_trg[1] != Xtrg[1] || cs_trg[2] != Xtrg[2])) {
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) coord_shift[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
      }
      cs_qel = &qel; cs_elem = elem_idx;
      for (Integer k = 0; k < COORD_DIM; k++) cs_trg[k] = Xtrg[k];
    }
    const Vector<Real>& cs_ref = (src_nodal ? *src_nodal : coord_shift);
    // The v-side contraction does not depend on the u-block/u-sweep, so hoist it for BOTH paths:
    // X and dXdu share it (both use Mv), which removed a duplicate GEMM set from the single-shot
    // path. All COORD_DIM components share Mv and coord_shift is component-major contiguous, so
    // the three (order x order).(order x Nv) products are one (COORD_DIM*order x order) GEMM.
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

    // Nbeta GL points per direction for the (finitely smooth) post-COV integrand,
    // decoupled from the field order (Bruno 2018: one to a few hundred). Default 200.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : 200);
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

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockQBX(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Line-QBX / "hedgehog" near-interaction (Lu 2019 sec.3.1): evaluate THIS source patch's layer
    // potential at a line of check points off the surface with a single upsampled smooth GL rule,
    // then 1D-polynomial-extrapolate back to the near target.
    //
    // ACCURACY CAVEAT: this per-pair form is accurate only when the target is FAR FROM PANEL SEAMS
    // (edges/corners). For a foot on/near a shared edge the check-point line cannot resolve the
    // adjacent panel's own edge singularity, so error floors near seams (~5e-3 vs ~5e-7 interior).
    // Use for panel-INTERIOR near targets; near seams prefer Adaptive (closest-point) or RectPolar.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    // Foot point (u*,v*) + local geometry: y (foot), n (unit normal), h (local element size).
    Real ustar, vstar;
    BENCH_TIC(ClosestPoint);
    qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    BENCH_TOC(ClosestPoint);
    Vector<Real> up{ustar}, vp{vstar}, Xfoot, Nfoot, Afoot;
    qel.GetGeom(&Xfoot, &Nfoot, &Afoot, nullptr, nullptr, up, vp, elem_idx);
    const Real h = sqrt<Real>(Afoot[0]); // sqrt(area element) ~ local edge length on [0,1]^2

    // Check-point line through the TARGET along the patch normal d = s*n at the foot (s = target's
    // side): check point i sits at height R+i*r above the patch; extrapolate from {R+i*r} to the
    // target height Hx = |(x-y).n|. Marching along n (not (x-y)/|x-y|) keeps the check points off
    // the source patch (no grazing) -- correct for off-surface near targets.
    Real d[COORD_DIM], Hx;
    {
      Real xy_dot_n = 0;
      for (Integer k = 0; k < COORD_DIM; k++) xy_dot_n += (Xtrg[k] - Xfoot[k]) * Nfoot[k];
      const Real s = (xy_dot_n >= 0 ? (Real)1 : (Real)-1);
      for (Integer k = 0; k < COORD_DIM; k++) d[k] = s * Nfoot[k];
      Hx = s * xy_dot_n;
    }

    // Upsampled smooth rule over the source patch: 4^eta = (2^eta)^2 uniform square subpatches, each
    // an `up_order` GL rule (eta=0 => single panel). Interp operators built once per 1D subinterval
    // (uniform => shared by u and v) and reused across check points via IntegrateBlock's *_pre.
    const Integer up_order = (qel.qbx_up_order_ > 0 ? qel.qbx_up_order_ : 2*order);
    Vector<Real> qnds, qwts;
    LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, up_order);
    const Integer nsub = (Integer)1 << std::max<Integer>(0, qel.qbx_eta_); // 2^eta subpanels/dir
    const Real hsub = (Real)1 / nsub;
    Vector<NodeRuleData> sub_rule(nsub);
    for (Integer s = 0; s < nsub; s++) {
      sub_rule[s].param = s*hsub + hsub*qnds;
      sub_rule[s].w = qwts * hsub;
      BuildInterp1D<order>(sub_rule[s].M, sub_rule[s].dM, sub_rule[s].MT, sub_rule[s].dMT, sub_rule[s].param);
    }

    // Check-point heights t_i = R + i*r (above the patch) and the 1D extrapolation weights e_i
    // from {t_i} to the target height Hx (Lagrange weights, (p+1) x 1 row-major).
    const Real R = qel.qbx_R_ * h, r = qel.qbx_r_ * h;
    const Integer p = qel.qbx_p_;
    Vector<Real> src_nds(p+1), trg_nds{Hx}, evec;
    for (Integer i = 0; i <= p; i++) src_nds[i] = R + i*r;
    LagrangeInterp<Real>::Interpolate(evec, src_nds, trg_nds);

    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero();

    // Accumulate sum_i e_i * sum_{subpatch} (nodal->c_i operator). IntegrateBlock is linear in
    // the quadrature weights and accumulates (+=) into M_acc, so folding e_i into the u-weights
    // adds e_i*M_i directly with no temp matrices.
    Vector<Real> wu(up_order);
    for (Integer i = 0; i <= p; i++) {
      // Check point on the line through the target, at height R+i*r above the patch:
      // c_i = x + (R + i*r - Hx)*d  (shift the target along the normal to the desired height).
      Vector<Real> ci(COORD_DIM);
      for (Integer k = 0; k < COORD_DIM; k++) ci[k] = Xtrg[k] + (R + i*r - Hx) * d[k];
      for (Integer su = 0; su < nsub; su++) {
        for (Integer a = 0; a < up_order; a++) wu[a] = sub_rule[su].w[a] * evec[i];
        for (Integer sv = 0; sv < nsub; sv++) {
          IntegrateBlock<order>(M_acc, qel, elem_idx, ci, normal_trg,
                                sub_rule[su].param, wu, sub_rule[sv].param, sub_rule[sv].w, ker,
                                &sub_rule[sv].M, &sub_rule[sv].dM, &sub_rule[su].M, &sub_rule[su].dM,
                                &sub_rule[sv].MT, &sub_rule[su].MT, &sub_rule[su].dMT);
        }
      }
    }
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracBatchedQBX(Matrix<Real>& M, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Production Line-QBX / hedgehog near-interaction (Lu 2019 sec.3.1) for ALL near targets of one
    // element. Algebraically equivalent to calling NearInteracBlockQBX per target, but with the
    // target-independent work hoisted out of the hot loops:
    //   * the upsampled source-patch geometry (absolute positions, unit normals, quadrature weights)
    //     depends only on (element, sub-panel rule), so it is built ONCE and reused across every
    //     target AND every check point -- IntegrateBlock rebuilt it (p+1)*nsub^2 times per target;
    //   * the sub-panel interp rules depend only on (order,up_order,nsub), reused across elements;
    //   * projection is linear, so the (p+1) check-point contributions are accumulated per sub-panel
    //     and projected back to the nodes ONCE per sub-panel (was one projection per check point).
    // Only the kernel matrix + weighting remain per (target, check point) -- irreducible for hedgehog.
    // Accurate for panel-INTERIOR targets only (see the SetLineQBXParams seam caveat).
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    SCTL_ASSERT(qel.order == order);
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;
    const Integer C = KDIM0 * KDIM1_out;

    const Long Ntrg = Xt.Dim() / COORD_DIM;
    if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != Ntrg*KDIM1_out) M.ReInit(nnode*KDIM0, Ntrg*KDIM1_out);
    M.SetZero();
    if (!Ntrg) return;

    const Integer up_order = (qel.qbx_up_order_ > 0 ? qel.qbx_up_order_ : 2*order);
    const Integer nsub = (Integer)1 << std::max<Integer>(0, qel.qbx_eta_); // 2^eta subpanels/dir
    const Long nq = (Long)up_order * up_order;
    const Integer p = qel.qbx_p_;

    // sub_rule: 4^eta uniform square subpatches, each an `up_order` GL rule + its (nodal->quad) interp
    // operators. Depends only on (order,up_order,nsub); thread_local last-key cache -> built once and
    // reused across every element on this thread (rebuilt only if the params change).
    thread_local Integer sr_up = -1, sr_nsub = -1;
    thread_local Vector<NodeRuleData> sub_rule;
    if (sr_up != up_order || sr_nsub != nsub) {
      Vector<Real> qnds, qwts;
      LegQuadRule<Real>::ComputeNdsWts(&qnds, &qwts, up_order);
      const Real hsub = (Real)1 / nsub;
      sub_rule.ReInit(nsub);
      for (Integer s = 0; s < nsub; s++) {
        sub_rule[s].param = s*hsub + hsub*qnds;
        sub_rule[s].w = qwts * hsub;
        BuildInterp1D<order>(sub_rule[s].M, sub_rule[s].dM, sub_rule[s].MT, sub_rule[s].dMT, sub_rule[s].param);
      }
      sr_up = up_order; sr_nsub = nsub;
    }

    // Upsampled ABSOLUTE source geometry per sub-panel (target-independent), built ONCE per call and
    // reused across all this element's targets AND all check points. Rebuilt every call rather than
    // cached across calls: the geometry depends on the exact (qel,elem_idx) coords, and a thread_local
    // cross-call cache keyed on &qel is unsafe -- short-lived QuadElemList objects reuse addresses, so
    // a stale key silently returns a previous panel's geometry (curved read as flat). The per-call
    // build is ~2% of the per-target cost (cold~=warm in the bench), so the safe choice is cheap.
    // Layout: Xabs/Xnsrc AoS [(sp*nq+q)*COORD_DIM+k]; wq_base [sp*nq+q]; sp=su*nsub+sv; q=a*up_order+b.
    thread_local Vector<Real> Xabs, Xnsrc, wq_base;
    {
      BENCH_TIC(GeomTensor);
      const Long nsp = (Long)nsub*nsub;
      if (Xabs.Dim() != nsp*nq*COORD_DIM) { Xabs.ReInit(nsp*nq*COORD_DIM); Xnsrc.ReInit(nsp*nq*COORD_DIM); wq_base.ReInit(nsp*nq); }
      const Long base = elem_idx * nnode * COORD_DIM;
      const Vector<Real> coord_abs(COORD_DIM*nnode, (Iterator<Real>)qel.coord.begin()+base, false); // CS3-shaped: (COORD_DIM*order x order)
      thread_local Vector<Real> X_soa, dXu_soa, dXv_soa; // component-major [k*nq+q]
      for (Integer su = 0; su < nsub; su++) {
        for (Integer sv = 0; sv < nsub; sv++) {
          const Long sp = (Long)su*nsub + sv;
          EvalTensorProduct(X_soa,   coord_abs, sub_rule[su].MT,  sub_rule[sv].M);  // absolute positions
          EvalTensorProduct(dXu_soa, coord_abs, sub_rule[su].dMT, sub_rule[sv].M);  // du tangents
          EvalTensorProduct(dXv_soa, coord_abs, sub_rule[su].MT,  sub_rule[sv].dM); // dv tangents
          for (Long a = 0; a < up_order; a++) {
            for (Long b = 0; b < up_order; b++) {
              const Long q = a*up_order + b;
              const Real du0 = dXu_soa[0*nq+q], du1 = dXu_soa[1*nq+q], du2 = dXu_soa[2*nq+q];
              const Real dv0 = dXv_soa[0*nq+q], dv1 = dXv_soa[1*nq+q], dv2 = dXv_soa[2*nq+q];
              const Real n0 = du1*dv2 - du2*dv1, n1 = du2*dv0 - du0*dv2, n2 = du0*dv1 - du1*dv0;
              const Real area = sqrt<Real>(n0*n0 + n1*n1 + n2*n2);
              const Real inv_area = (area > 0 ? 1/area : 0);
              const Long o = (sp*nq + q)*COORD_DIM;
              Xabs[o+0] = X_soa[0*nq+q]; Xabs[o+1] = X_soa[1*nq+q]; Xabs[o+2] = X_soa[2*nq+q];
              Xnsrc[o+0] = n0*inv_area;  Xnsrc[o+1] = n1*inv_area;  Xnsrc[o+2] = n2*inv_area;
              wq_base[sp*nq + q] = area * sub_rule[su].w[a] * sub_rule[sv].w[b]; // NO evec (target-independent)
            }
          }
        }
      }
      BENCH_TOC(GeomTensor);
    }

    // Kernel eval: sources relative to the check point (Xsrc = Xabs - ci), target at the origin, one
    // KernelMatrix call PER check point. (Stage-2 batching -- stacking all (p+1) check points into one
    // call -- was tried and is 17-30% SLOWER: the nq=5184 source vectorization already saturates per
    // call, so batching saves no arithmetic while replicating normals adds memory traffic. Same
    // memory-bound loss as the Adaptive leaf-batching. Kept per-check-point.)
    thread_local Vector<Real> Xsrc, KW_total, proj;
    thread_local Matrix<Real> Mker, M_acc;
    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != C) M_acc.ReInit(nnode, C);
    if (Xsrc.Dim() != nq*COORD_DIM) Xsrc.ReInit(nq*COORD_DIM);
    if (KW_total.Dim() != C*nq) KW_total.ReInit(C*nq);

    for (Long t = 0; t < Ntrg; t++) {
      const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
      Vector<Real> ntrg;
      if (trg_dot_prod) ntrg.ReInit(COORD_DIM, (Iterator<Real>)normal_trg.begin() + t*COORD_DIM, false);

      // Foot point + local size h; check-point ray d = s*n through the target; target height Hx.
      Real ustar, vstar;
      BENCH_TIC(ClosestPoint);
      qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
      BENCH_TOC(ClosestPoint);
      Vector<Real> up{ustar}, vp{vstar}, Xfoot, Nfoot, Afoot;
      qel.GetGeom(&Xfoot, &Nfoot, &Afoot, nullptr, nullptr, up, vp, elem_idx);
      const Real h = sqrt<Real>(Afoot[0]);
      Real d[COORD_DIM], Hx;
      {
        Real xy_dot_n = 0;
        for (Integer k = 0; k < COORD_DIM; k++) xy_dot_n += (Xtrg[k] - Xfoot[k]) * Nfoot[k];
        const Real s = (xy_dot_n >= 0 ? (Real)1 : (Real)-1);
        for (Integer k = 0; k < COORD_DIM; k++) d[k] = s * Nfoot[k];
        Hx = s * xy_dot_n;
      }

      // Check-point heights t_i = R + i*r and 1D Lagrange extrapolation weights from {t_i} to Hx.
      const Real R = qel.qbx_R_ * h, r = qel.qbx_r_ * h;
      Vector<Real> src_nds(p+1), trg_nds{Hx}, evec;
      for (Integer i = 0; i <= p; i++) src_nds[i] = R + i*r;
      LagrangeInterp<Real>::Interpolate(evec, src_nds, trg_nds);

      M_acc.SetZero();
      for (Integer su = 0; su < nsub; su++) {
        for (Integer sv = 0; sv < nsub; sv++) {
          const Long sp = (Long)su*nsub + sv;
          const Vector<Real> Xnsrc_sp(nq*COORD_DIM, (Iterator<Real>)Xnsrc.begin() + sp*nq*COORD_DIM, false);
          for (Long e = 0; e < C*nq; e++) KW_total[e] = 0;

          for (Integer i = 0; i <= p; i++) {
            Real ci[COORD_DIM];
            for (Integer k = 0; k < COORD_DIM; k++) ci[k] = Xtrg[k] + (R + i*r - Hx) * d[k];
            for (Long q = 0; q < nq; q++)
              for (Integer k = 0; k < COORD_DIM; k++)
                Xsrc[q*COORD_DIM+k] = Xabs[(sp*nq+q)*COORD_DIM+k] - ci[k];

            StaticArray<Real,COORD_DIM> Xt0{0, 0, 0};
            const Vector<Real> Xt0_v(COORD_DIM, Xt0, false); // sources already relative to ci
            BENCH_TIC(KernelEval);
            ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xsrc, Xnsrc_sp); // (nq*KDIM0 x KDIM1full)
            BENCH_TOC(KernelEval);

            // Weighted kernel, fold in the extrapolation weight evec[i], accumulate (linear projection
            // deferred): KW_total[c*nq+q] += evec[i] * (trg-dot-prod?) * wq_base. Component-major.
            BENCH_TIC(KernelWeight);
            const Real ei = evec[i];
            for (Long q = 0; q < nq; q++) {
              const Real w = ei * wq_base[sp*nq+q];
              for (Integer k0 = 0; k0 < KDIM0; k0++) {
                for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
                  Real val;
                  if (trg_dot_prod) {
                    val = 0;
                    for (Integer l = 0; l < COORD_DIM; l++) val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * ntrg[l];
                  } else {
                    val = Mker[q*KDIM0+k0][k1];
                  }
                  KW_total[(Long)(k0*KDIM1_out+k1)*nq + q] += val * w;
                }
              }
            }
            BENCH_TOC(KernelWeight);
          }

          // ONE tensor-factored projection per sub-panel: M_acc += (Mu . KW_total . MvT).
          BENCH_TIC(Projection);
          EvalTensorProduct(proj, KW_total, sub_rule[su].M, sub_rule[sv].MT);
          for (Long pn = 0; pn < nnode; pn++)
            for (Integer c = 0; c < C; c++) M_acc[pn][c] += proj[(Long)c*nnode + pn];
          BENCH_TOC(Projection);
        }
      }

      // Scatter M_acc into M for target t: M[(i*order+j)*KDIM0+k0][t*KDIM1_out+k1].
      for (Integer ii = 0; ii < order; ii++) {
        for (Integer jj = 0; jj < order; jj++) {
          const Long pnode = ii*order + jj;
          for (Integer k0 = 0; k0 < KDIM0; k0++)
            for (Integer k1 = 0; k1 < KDIM1_out; k1++)
              M[pnode*KDIM0+k0][t*KDIM1_out+k1] = M_acc[pnode][k0*KDIM1_out+k1];
        }
      }
    }
  }

  // ============================ ported from upstream 0f12ddf ============================
  // Centered self rules + split-at-foot near scheme. Grafted onto the QBX/RP work below.

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
    // Same panel layout as LogSingularQuad1D, emitted as offsets from v0.
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

  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    if (qel.NearUsesRectPolar()) { NearInteracBlockRP<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ker, NbetaForDigits(digits)); return; }
    if (qel.NearUsesLineQBX()) { NearInteracBlockQBX<order>(M_acc, qel, elem_idx, Xtrg, normal_trg, ker); return; }
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Long nnode = (Long)order*order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full/COORD_DIM : KDIM1full;
    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    // Cells accumulate into a CHANNEL-major buffer so the projection's last GEMM can add in place
    // (beta = 1); one transpose into M_acc's node-major layout at the end replaces a per-cell sweep.
    const Integer C_ = KDIM0*KDIM1_out;
    thread_local Vector<Real> acc, accB, accE;
    if (acc.Dim() != (Long)C_*nnode) { acc.ReInit((Long)C_*nnode); accB.ReInit((Long)C_*nnode); accE.ReInit(nnode); }
    M_acc.SetZero();

    const Real b_ellipse = NearBEllipse<digits>();
    const Vector<GradeRule>& tab = NearGradeTable<order,digits>();

    // Foot + level count (same criterion as BuildNearLeaves).
    Real ustar, vstar;
    BENCH_TIC(ClosestNode);
    const Real dist = qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    BENCH_TOC(ClosestNode);
    Real Xc[COORD_DIM], dXu_[COORD_DIM], dXv_[COORD_DIM];
    qel.EvalPoint(Xc, dXu_, dXv_, ustar, vstar, elem_idx, nullptr);
    Real su2 = 0, sv2 = 0;
    for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXu_[k]*dXu_[k]; sv2 += dXv_[k]*dXv_[k]; }
    // Refinement stops per sub-element via the admissibility test in the loop below, so no
    // global depth is needed; only the per-direction surface speeds are.
    const Real spd_u = sqrt<Real>(su2), spd_v = sqrt<Real>(sv2);

    const Vector<Real>& gnds = ParamNodes(order);
    const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};   // [dir][side] sub-element length

    BENCH_TIC(ClosestPoint);
    // Target-shifted element nodal coords, component-major: one contiguous (COORD_DIM*order x order).
    thread_local Vector<Real> cs;
    if (cs.Dim() != COORD_DIM*nnode) cs.ReInit(COORD_DIM*nnode);
    {
      const Long base = elem_idx * nnode * COORD_DIM;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) cs[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
      }
    }
    // S[i][a] = L_i(sub[a]), element nodes -> sub-element nodes; side 1 mirrored so BOTH sides
    // grade toward the foot at normalized x = 1. The v-contraction needs S, the u-contraction
    // needs S^T, so build only the one each direction uses.
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
    // Per-quadrant sub-element nodal slabs, Xsub = S_u^T . cs . S_v, built ONCE per target. This
    // is what makes every per-cell operator a precomputed table entry (T/dT/TT/TD) instead of a
    // per-target S.T product: nothing depending on (u*,v*) survives into the per-cell loop.
    // The v-contraction batches all COORD_DIM components into a single GEMM.
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
    BENCH_TOC(ClosestPoint);

    // Every cell operator is a table entry now. nrm_sign corrects the normal on quadrants with
    // exactly one mirrored direction, where d/dx_u x d/dx_v is anti-parallel to dXu x dXv; the
    // area element carries |du/dx . dv/dx| = slen_u.slen_v, so weights stay the normalized g.w.
    const Vector<Real> empty_param;
    auto emit = [&](Integer sdu, Integer sdv, Integer iu, Integer iv) {
      const GradeRule& gu = tab[iu];
      const GradeRule& gv = tab[iv];
      if (!(gu.b > gu.a) || !(gv.b > gv.a)) return;
      const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
      IntegrateBlock<order>(M_acc, qel, elem_idx, Xtrg, normal_trg,
                            empty_param, gu.w, empty_param, gv.w, ker,
                            &gv.T, &gv.dT, &gu.T, nullptr, &gv.TT, &gu.TT, nullptr,
                            &Xsub[sdu][sdv], &gu.TD, nsign, &acc);
    };
    for (Integer sdu = 0; sdu < 2; sdu++) {
      if (!(slen[0][sdu] > 0)) continue;
      for (Integer sdv = 0; sdv < 2; sdv++) {
        if (!(slen[1][sdv] > 0)) continue;
        acc.SetZero();
        // ANISOTROPIC refinement. Splitting the element at (u*,v*) makes the sub-elements
        // anisotropic, and quadrisection would hand that aspect ratio to every descendant --
        // leaving cells that are large in one direction while sitting close to the target in the
        // other, which is inadmissible. So split the corner cell along its longer PHYSICAL
        // dimension only (parameter extent x surface speed), one bisection at a time, until the
        // longer side is admissible against the target distance. Each split emits exactly one
        // leaf (the half not touching the corner); the u- and v-levels advance independently, so
        // every interval is still shell_k / core_k at some level and comes from the table.
        Integer ku = 0, kv = 0;
        Real hu = slen[0][sdu]*spd_u, hv = slen[1][sdv]*spd_v;
        const bool cap = !(dist > 0) || !std::isfinite((double)dist);
        const Integer ovr = NearMaxLvlOverride();
        const Integer KMAX = std::min<Integer>(ovr ? ovr : qel.max_depth_, MaxNearLvl-1); // table bound
        while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
          if (hu >= hv && ku < KMAX) {
            emit(sdu, sdv, ku, MaxNearLvl + kv);               // shell_ku x core_kv
            ku++; hu *= (Real)0.5;
          } else if (kv < KMAX) {
            emit(sdu, sdv, MaxNearLvl + ku, kv);               // core_ku x shell_kv
            kv++; hv *= (Real)0.5;
          } else if (ku < KMAX) {
            emit(sdu, sdv, ku, MaxNearLvl + kv);
            ku++; hu *= (Real)0.5;
          } else break;
        }
        emit(sdu, sdv, MaxNearLvl + ku, MaxNearLvl + kv);      // terminal corner cell

        // The cells projected onto the SUB-ELEMENT basis, but the density lives on the ELEMENT
        // nodes, so map back: L_p^elem restricted to the sub-element is exactly sum_a S[p][a]
        // L_a^sub (affine map, degree order-1), giving M_elem += S_u . A_sub . S_v^T -- the adjoint
        // of the Xsub = S_u^T . cs . S_v used for the geometry. Once per quadrant, not per cell.
        // Constant density hides this error entirely (both bases are partitions of unity, so the
        // row sum is unchanged); it shows up only for varying density, e.g. Green's identity.
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


  template <class Real> template <Integer digits, Integer order, class Kernel> void QuadElemList<Real>::SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Integer ti, const Integer tj, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
    // Singular self-interaction for on-surface node (ti,tj). 1D reduction: graded u-rule
    // toward u0 x Alpert log-singular v-rule toward v0; both rules + interpolation are
    // preloaded (geometry-independent, fixed by order/ti/tj/digits), integrated by
    // IntegrateBlock. IntegrateBlock still does the target-centered geometry per target.
    if (qel.SelfUsesRectPolar()) { SelfInteracBlockRP<order>(M_acc, qel, elem_idx, ti, tj, Xtrg, normal_trg, ker, NbetaForDigits(digits)); return; }

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

    // Nbeta GL points per direction for the (finitely smooth) post-COV integrand,
    // decoupled from the field order (Bruno 2018: one to a few hundred). Default 512.
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : 200);
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
    const Integer Nbeta = (qel.cov_order_ > 0 ? qel.cov_order_ : 200);
    if (qel.SelfUsesRectPolar()) {
      RPSelfRuleDispatch<order>(0, qel.cov_q_, Nbeta);
    } else {
      CenteredURule<order, digits>(0, qel.max_depth_);  // centered self: graded u-rule (mutex-cached)
      CenteredVRule<order, digits>(0);                  // centered Alpert v-rule
    }
    if (qel.NearUsesRectPolar()) {
      GLRuleNbetaDispatch(Nbeta);  // near-RP: RectPolarNodes1D GL rule (also DiffMat/ParamNodes, already warm)
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
    const Real utol = (Real)machine_eps<Real>() * 64;      // step tolerance (boundary optima)
    // Relative first-order optimality tolerance. Because f = |r|^2 is a squared residual,
    // the gradient can only be driven to ~sqrt(eps) (relative) before f flattens at its
    // rounding floor -- pushing further just stalls the line search. Test at that scale.
    const Real gtol = sqrt<Real>(machine_eps<Real>()) * 16;
    bool converged = false;
    Integer iters = 0;
    for (Integer it = 0; it < max_iter; it++) {
      iters = it + 1;
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

      // First-order optimality (KKT for the box [0,1]^2) via the PROJECTED gradient: at an active
      // bound only the feasible-direction gradient component counts (f = |r|^2, so d f/du = 2 gu; a
      // lower bound u=0 is stationary when gu >= 0, an upper bound u=1 when gu <= 0), interior when
      // gu is negligible vs sqrt(E*f). Testing the projected GRADIENT -- not the sign of the coupled
      // Newton step du/dv -- is what makes EDGE/CORNER optima converge: at an edge the constrained
      // gu is NOT small (it balances the constraint) and metric coupling (F != 0) can flip du's
      // sign, so the old step-sign test misclassified boundary optima, the clamped line search then
      // stalled, and it bailed to the grid-search fallback (~40% of near-pair targets, whose foot
      // lies on a shared patch edge). Only interior feet ever converged cleanly before this.
      Real Pu = gu, Pv = gv;
      if      (u <= 0) Pu = std::min<Real>(gu, (Real)0);
      else if (u >= 1) Pu = std::max<Real>(gu, (Real)0);
      if      (v <= 0) Pv = std::min<Real>(gv, (Real)0);
      else if (v >= 1) Pv = std::max<Real>(gv, (Real)0);
      const bool opt_u = (fabs(Pu) <= gtol * sqrt<Real>(E*f));
      const bool opt_v = (fabs(Pv) <= gtol * sqrt<Real>(G*f));
      if (opt_u && opt_v) { converged = true; break; }

      // Backtrack on the clamped Newton step until f decreases.
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
      // If the clamped Newton step stalls, retry along the PROJECTED (metric-scaled) GRADIENT.
      // At an active bound the coupled Newton step can point outward, so clamping freezes it at a
      // non-optimal point; the projected gradient (Pu,Pv) is a feasible descent direction whenever
      // the point is not KKT-optimal (already returned above), so some backtracked step lowers f.
      // This is what actually eliminates the edge-foot grid-search fallbacks (the KKT test alone
      // did not: the loop was exiting here, not at the optimality check).
      if (!improved) {
        const Real gu_s = Pu / (E + (Real)1e-30), gv_s = Pv / (G + (Real)1e-30);
        lambda = 1;
        for (Integer ls = 0; ls < 40; ls++) {
          un = std::min<Real>(1, std::max<Real>(0, u - lambda*gu_s));
          vn = std::min<Real>(1, std::max<Real>(0, v - lambda*gv_s));
          fn = dist2_at(un, vn);
          if (fn < f) { improved = true; break; }
          lambda *= (Real)0.5;
        }
      }
      if (!improved) break; // genuine stall -> grid-search fallback (now rare)
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
    if (n_iter) *n_iter = iters;
    if (used_fallback) *used_fallback = !converged;
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

    // LineQBX/hedgehog: batch all of this element's near targets so the target-independent
    // upsampled source geometry + sub-panel rules are built once per element (not per target).
    if (qel.NearUsesLineQBX()) { NearInteracBatchedQBX<order>(M, qel, elem_idx, Xt, normal_trg, ker); return; }

    for (Long t = 0; t < Ntrg; t++) {
      Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
      Vector<Real> ntrg;
      if (trg_dot_prod) ntrg.ReInit(COORD_DIM, (Iterator<Real>)normal_trg.begin() + t*COORD_DIM, false);

      Matrix<Real> M_acc;
      // Adaptive/Hybrid near: split-at-foot (RP/QBX handled by the guards inside the block).
      NearInteracBlockSplit<digits, order>(M_acc, qel, elem_idx, Xtrg, ntrg, ker);

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
      // Replicate NearInteracBlockSplit's cell layout: split at the foot (u*,v*), grade each
      // sub-element toward the foot with shell_k/core_k intervals, bisecting the longer physical
      // side. Flat index i -> normalized interval [a,b]: shell_k=[1-2^-k,1-2^-(k+1)] for i=k<L,
      // core_k=[1-2^-k,1] for i=L+k. Mapped back to actual param through the sub-element affine map.
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
          Integer ku = 0, kv = 0;
          Real hu = slen[0][sdu]*spd_u, hv = slen[1][sdv]*spd_v;
          while ((cap || b_ellipse*std::max<Real>(hu,hv) > dist) && (ku < KMAX || kv < KMAX)) {
            if (hu >= hv && ku < KMAX) { cell(sdu,sdv,ku,MaxNearLvl+kv); ku++; hu *= (Real)0.5; }
            else if (kv < KMAX)        { cell(sdu,sdv,MaxNearLvl+ku,kv); kv++; hv *= (Real)0.5; }
            else if (ku < KMAX)        { cell(sdu,sdv,ku,MaxNearLvl+kv); ku++; hu *= (Real)0.5; }
            else break;
          }
          cell(sdu,sdv,MaxNearLvl+ku,MaxNearLvl+kv);   // terminal corner cell
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
