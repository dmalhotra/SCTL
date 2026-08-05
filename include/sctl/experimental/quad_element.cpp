#ifndef _SCTL_QUAD_ELEMENT_CPP_
#define _SCTL_QUAD_ELEMENT_CPP_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sctl.hpp>
#include "sctl/experimental/quad_element.hpp"

#include <array>
#include <map>
#include <mutex>

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

    { // nodal dX/du, dX/dv cache
      dcoord_du.ReInit(coord.Dim());
      dcoord_dv.ReInit(coord.Dim());
      const Long elem_stride = COORD_DIM * nnode_per_elem;
      for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
        const Long base = elem_idx * elem_stride;
        const Vector<Real> coord_(elem_stride, (Iterator<Real>)coord.begin() + base, false);
        Vector<Real> du_(elem_stride, dcoord_du.begin() + base, false);
        Vector<Real> dv_(elem_stride, dcoord_dv.begin() + base, false);
        NodalDerivs(coord_, order, du_, dv_);
      }
    }
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

  template <class Real> inline const Matrix<Real>& QuadElemList<Real>::DiffMat(const Integer order) {
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

  template <class Real> Long QuadElemList<Real>::Size() const {
    return nelem;
  }

  template <class Real> Integer QuadElemList<Real>::Order() const {
    return order;
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

    // Right-first: flop-optimal when the transform EXPANDS (geometry: Nu>>R), ~7% off when it
    // CONTRACTS (projection: Nu<<R).
    constexpr Integer Nbuff = 1024;
    StaticArray<ValueType,Nbuff> tmp_buf;
    Matrix<ValueType> tmp(R, Nv, ((Long)R * Nv > Nbuff ? NullIterator<ValueType>() : tmp_buf), (Long)R * Nv > Nbuff);

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

    thread_local Matrix<Real> Mu, MuT, Mv;
    if (Mu.Dim(0) != order || Mu.Dim(1) != Nu) { Mu.ReInit(order, Nu); MuT.ReInit(Nu, order); }
    if (Mv.Dim(0) != order || Mv.Dim(1) != Nv) Mv.ReInit(order, Nv);
    { Vector<Real> Mu_(order * Nu, Mu.begin(), false);
      Vector<Real> Mv_(order * Nv, Mv.begin(), false);
      LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_param);
      LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_param); }
    for (Integer i = 0; i < order; i++) for (Long a = 0; a < Nu; a++) MuT[a][i] = Mu[i][a];

    SCTL_ASSERT(elem_idx >= 0 && elem_idx < nelem);
    const Long base = elem_idx * nnode_per_elem * COORD_DIM;
    const Vector<Real> coord_(COORD_DIM * nnode_per_elem, (Iterator<Real>)coord.begin() + base, false);
    const Vector<Real> dcoord_du_(COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_du.begin() + base, false);
    const Vector<Real> dcoord_dv_(COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_dv.begin() + base, false);

    // Target-centering: subtract `origin` from nodal positions before interpolation so
    // X is target-relative (accurate near the singularity); derivatives recomputed from
    // the shifted slab. origin == nullptr keeps the cached absolute-coordinate path.
    thread_local Vector<Real> coord_shift, du_shift, dv_shift;
    const Vector<Real>* pos_in = &coord_;
    const Vector<Real>* du_in = &dcoord_du_;
    const Vector<Real>* dv_in = &dcoord_dv_;
    if (origin) {
      if (coord_shift.Dim() != COORD_DIM * nnode_per_elem) coord_shift.ReInit(COORD_DIM * nnode_per_elem);
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = (*origin)[k];
        for (Long p = 0; p < nnode_per_elem; p++) coord_shift[k * nnode_per_elem + p] = coord_[k * nnode_per_elem + p] - ok;
      }
      if (Xn || Xa || dX_du || dX_dv) NodalDerivs(coord_shift, order, du_shift, dv_shift);
      pos_in = &coord_shift; du_in = &du_shift; dv_in = &dv_shift;
    }

    if (X) {
      thread_local Vector<Real> X_soa;
      EvalTensorProduct(X_soa, *pos_in, MuT, Mv);
      for (Long i = 0; i < N; i++) {
        (*X)[i * COORD_DIM + 0] = X_soa[0 * N + i];
        (*X)[i * COORD_DIM + 1] = X_soa[1 * N + i];
        (*X)[i * COORD_DIM + 2] = X_soa[2 * N + i];
      }
    }
    if (Xn || Xa || dX_du || dX_dv) {
      thread_local Vector<Real> dXdu_soa, dXdv_soa;
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

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::IntegrateBlock(const Vector<Real>& normal_trg, const Vector<Real>& wu, const Vector<Real>& wv, const Kernel& ker, const Matrix<Real>& Mu, const Matrix<Real>& MuT, const Matrix<Real>& MuD, const Matrix<Real>& Mv, const Matrix<Real>& dMv, const Matrix<Real>& MvT, const Vector<Real>& src_nodal, const Real nrm_sign, Vector<Real>& acc_cm, const Vector<Real>& proxy_off, const Vector<Real>& proxy_w) {
    // One near leaf cell: accumulate its tensor-product quadrature (weights wu (x) wv) against
    // the target into acc_cm. src_nodal is the caller's target-shifted nodal slab, so the kernel
    // target sits at the origin. Tensor grid is u-slow/v-fast: node (a,b) has flat index a*Nv+b.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const Long Nu = Mu.Dim(1), Nv = Mv.Dim(1), nq = Nu * Nv;
    if (!nq) return;
    const Integer C = KDIM0 * KDIM1_out;

    // The v-side contraction is shared by X and dXdu (both use Mv). All COORD_DIM components
    // share it and src_nodal is component-major contiguous, so the three (order x order).
    // (order x Nv) products are one (COORD_DIM*order x order) GEMM.
    thread_local Vector<Real> Cv, Cdv;
    if (Cv.Dim() != COORD_DIM*order*Nv) { Cv.ReInit(COORD_DIM*order*Nv); Cdv.ReInit(COORD_DIM*order*Nv); }
    {
      const Matrix<Real> cs_all(COORD_DIM*order, order, (Iterator<Real>)src_nodal.begin(), false);
      Matrix<Real> Cv_all (COORD_DIM*order, Nv, Cv.begin(),  false);
      Matrix<Real> Cdv_all(COORD_DIM*order, Nv, Cdv.begin(), false);
      Matrix<Real>::GEMM(Cv_all,  cs_all, Mv);
      Matrix<Real>::GEMM(Cdv_all, cs_all, dMv);
    }
    // Column-stage Cv/Cdv (component index moved into the COLUMNS) so stage 2 batches over
    // components as well as over outputs: the nine original (Nu x order).(order x Nv) products
    // collapse to two GEMMs against an (order x COORD_DIM*Nv) operand. The restage is an
    // L1-resident copy; Matrix::GEMM has no strided-output form.
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
      { // MuD = [T^T; dT^T] gives X and dXdu in one GEMM
        Matrix<Real> XdU_m(2*Nu, ldc, XdU.begin(), false);
        Matrix<Real>::GEMM(XdU_m, MuD, Cvc_m);
      }
      Matrix<Real>::GEMM(dV_m, MuT, Cdvc_m);
    }

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

    thread_local Matrix<Real> Mker;
    thread_local Vector<Real> KWc;
    if (KWc.Dim() != C*nq) KWc.ReInit(C*nq);
    // The proxy targets of one hedgehog line share this cell's geometry, and the extrapolation
    // weights are applied HERE, so only the kernel evaluation scales with the number of proxies:
    // the fold below and the projection GEMMs run once. Valid because both are linear in the
    // kernel values. proxy_off holds the offsets from the target that positioned the cell.
    //
    // Dispatched on whether there are any proxies, so the ordinary near path -- which has none --
    // folds with no loop over proxies, no length tests, no weight multiply and no first-iteration
    // test in the innermost sweep. Carrying those unconditionally cost ~11% on the cubed sphere.
    const auto fold = [&](const auto has_proxy) {
      constexpr bool HP = decltype(has_proxy)::value;
      const Long np = (HP ? proxy_w.Dim() : 1);
      for (Long j = 0; j < np; j++) {
        StaticArray<Real,COORD_DIM> Xtj{0, 0, 0};
        if constexpr (HP) for (Integer l = 0; l < COORD_DIM; l++) Xtj[l] = proxy_off[j*COORD_DIM+l];
        const Vector<Real> Xtj_v(COORD_DIM, Xtj, false);
        ker.template KernelMatrix<Real,false>(Mker, Xtj_v, Xsrc, Xnsrc); // (nq*KDIM0 x KDIM1full)
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
              const Long id = (Long)(k0*KDIM1_out+k1)*nq + q;
              if constexpr (HP) {
                const Real wj = proxy_w[j];
                if (j == 0) KWc[id]  = wj*val*wq[q];
                else        KWc[id] += wj*val*wq[q];
              } else {
                KWc[id] = val*wq[q];
              }
            }
          }
        }
      }
    };
    if (proxy_w.Dim()) fold(std::true_type{}); else fold(std::false_type{});

    // Adjoint of the geometry interpolation: quadrature -> nodal. KWc is channel-major with
    // (Nu x Nv) blocks, so the v-contraction is already one (C*Nu x Nv) operand and batches
    // over all C channels for free. The u-contraction then writes one (order x order) block
    // per channel -- a channel-major accumulator's layout -- so with acc_cm it accumulates in
    // place via beta = 1, and the caller transposes to M_acc's node-major layout once per
    // target rather than per cell.
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



  template <class Real> inline Integer QuadElemList<Real>::DigitsFromTol(const Real tol) {
    // Reproduces the old if-else dispatch exactly: the largest d < MaxDigits with
    // tol <= 10^-d. pow(0.1, d) is the same repeated multiplication `pow<d,Real>` used,
    // so the branch points are bit-identical to the templated version.
    for (Integer d = MaxDigits-1; d > 0; d--) if (tol <= pow<Real,Long>((Real)0.1, (Long)d)) return d;
    return 0;
  }

  template <class Real> inline Integer QuadElemList<Real>::NearQuadOrder(const Integer digits) {
    static const std::array<Integer,MaxDigits> q = []() {
      std::array<Integer,MaxDigits> t{};
      for (Integer d = 0; d < MaxDigits; d++) {
        Real b; Integer qq; NearRhoRule(pow<Real,Long>((Real)0.1, (Long)d), b, qq); t[d] = qq;
      }
      return t;
    }();
    SCTL_ASSERT(digits >= 0 && digits < MaxDigits);
    return q[digits];
  }

  template <class Real> void QuadElemList<Real>::NearRhoRule(const Real tol, Real& b_ellipse, Integer& QuadOrder) {
    // Measured cost-optimal rho vs requested digits (order 12, 4x4 panels/face, Laplace SL+DL):
    // 1e-4 -> 2.0, 1e-6 -> 2.0, 1e-8 -> 2.5, 1e-10 -> 3.0, 1e-12 -> 2.5. Accuracy breaks down
    // above rho ~ 3.2 at every tolerance: the attained rate saturates near rho_eff ~ 3, so a
    // larger design rho only buys refinement levels without improving the per-cell rate.
    const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
    const double d = -std::log10((double)tol_);
    const double rho = std::min(3.0, std::max(2.0, 2.0 + 0.25*(d - 6)));
    const double C = std::max(1e-3, (15.0*(rho*rho - 1))/64.0);
    QuadOrder = std::max<Integer>(2, (Integer)std::ceil(-std::log(C*(double)tol_)/std::log(rho)*0.5 + 1));

    // End-foot reach, not the semi-major axis. E_rho has semi-axes a,b with a^2-b^2 = 1, and a
    // singularity at parameter s with perpendicular offset d~ = 2d/L lies outside it when
    // s^2/a^2 + d~^2/b^2 > 1. The split puts the foot at a cell endpoint (s = +-1), giving
    // d~ > b^2/a -- weaker than the semi-major reach by a^2/b^2.
    const double a = (rho + 1/rho)/2, b = (rho - 1/rho)/2;
    b_ellipse = (Real)(b*b/(2*a));
  }

  template <class Real> inline Real QuadElemList<Real>::NearBEllipse(const Integer digits) {
    static const std::array<Real,MaxDigits> b = []() {
      std::array<Real,MaxDigits> t{};
      for (Integer d = 0; d < MaxDigits; d++) {
        Real bb; Integer qq; NearRhoRule(pow<Real,Long>((Real)0.1, (Long)d), bb, qq); t[d] = bb;
      }
      return t;
    }();
    SCTL_ASSERT(digits >= 0 && digits < MaxDigits);
    return b[digits];
  }


  // Work type for PRECOMPUTED near-scheme tables. These are built once per order and cached, so
  // building them a precision step up and rounding to Real costs nothing at run time and leaves
  // the nodes and interpolation weights correctly rounded instead of carrying their own build
  // error. Only the tables use this; everything per-target stays in Real.
#ifdef SCTL_QUAD_T
  template <class Real> struct NearTabWork { using type = QuadReal; };
  template <> struct NearTabWork<QuadReal> { using type = QuadReal; };
#else
  template <class Real> struct NearTabWork { using type = long double; };
#endif

  template <class Real> inline const Vector<Real>& QuadElemList<Real>::HedgehogProxyOffsets() {
    // Five points over a span of 4. Wider sets ({1,4,..,36}) cut sum|w| from 61 to 3.4 but measured
    // worse: amplification is not what binds, and the wide span spoils the fit.
    static const Vector<Real> s = []() {
      Vector<Real> v;
      for (Integer j = 0; j < 5; j++) v.PushBack(pow<Real>((Real)4, (Real)j/(Real)4));
      return v;
    }();
    return s;
  }

  template <class Real> inline Real QuadElemList<Real>::HedgehogWeights(Vector<Real>& w) {
    static const std::pair<Vector<Real>,Real> tab = []() {
      using W = typename NearTabWork<Real>::type;
      const Vector<Real>& s = HedgehogProxyOffsets();
      const Long p = s.Dim();
      Vector<Real> wj(p);
      Real A = 0;
      for (Long j = 0; j < p; j++) {
        W v = 1;
        for (Long k = 0; k < p; k++) if (k != j) v *= (0 - (W)s[k])/((W)s[j] - (W)s[k]);
        wj[j] = (Real)v; A += fabs<Real>(wj[j]);
      }
      return std::make_pair(wj, A);
    }();
    if (w.Dim() != tab.first.Dim()) w.ReInit(tab.first.Dim());
    w = tab.first;
    return tab.second;
  }

  template <class Real> inline Real QuadElemList<Real>::HedgehogRminCoeff(const Integer digits, const Integer sing_order) {
    // Largest offset still reaching Duffy's error on the sphere; larger is cheaper (less refinement).
    // 1/r wants 0.1*tol^(1/6) exactly. Steeper wants ~3e-3 flat -- below that its floor is the
    // amplified quadrature error, not the fit, so shrinking further only costs.
    static const std::array<Real,MaxDigits> c = []() {
      std::array<Real,MaxDigits> t{};
      for (Integer d = 0; d < MaxDigits; d++) t[d] = (Real)0.1 * pow<Real>(pow<Real,Long>((Real)0.1, (Long)d), (Real)1/(Real)6);
      return t;
    }();
    SCTL_ASSERT(digits >= 0 && digits < MaxDigits);
    return (sing_order <= 1 ? c[digits] : std::min<Real>(c[digits], (Real)3e-3));
  }

  template <class Real> inline Integer QuadElemList<Real>::HedgehogNearDigits(const Integer digits, const Integer sing_order) {
    // Extrapolation multiplies the proxy quadrature error by sum|w| = 61. Two extra digits covers
    // that for 1/r; steeper kernels measured needing six, which was the whole double-layer gap.
    return std::min<Integer>(MaxDigits-1, digits + (sing_order <= 1 ? 2 : 6));
  }

  // Sub-element interpolation basis for the near scheme: Chebyshev-Lobatto on [0,1]. Unlike the
  // element's Gauss nodes it INCLUDES the endpoints, so the foot -- which the split places at
  // s = 1 on every side -- is itself a node. Interpolating toward a node makes every other
  // weight vanish linearly in the offset t, so the products L_i(t)*X_i are each O(t) and the
  // small result is no longer a cancelling sum of O(1) terms: with Gauss nodes the largest term
  // saturates at ~0.17 however small t is, giving absolute error eps and relative error eps/t.
  template <class Real> static const Vector<Real>& NearSubNodes(const Integer order) {
    constexpr Integer MAX_ORDER = 50;
    SCTL_ASSERT(1 < order && order <= MAX_ORDER);
    static const Vector<Vector<Real>> all = []() {
      Vector<Vector<Real>> v(MAX_ORDER + 1);
      for (Integer n = 2; n <= MAX_ORDER; n++) {
        v[n].ReInit(n);
        // half-angle form: sin^2 is cancellation-free near 0, so the nodes closest to the
        // element edge keep full relative accuracy.
        using W = typename NearTabWork<Real>::type;
        for (Integer i = 0; i < n; i++) {
          const W sh = sin<W>(const_pi<W>()*i/(2*(n-1)));
          v[n][i] = (Real)(sh*sh);
        }
        v[n][0] = 0; v[n][n-1] = 1;               // endpoints exact
      }
      return v;
    }();
    return all[order];
  }
  // Offsets of the sub-element nodes from the SPLIT POINT (the node at reference 1), built
  // directly in the half-angle form. Forming these as one-minus-node instead would subtract two
  // order-one numbers, leaving the nodes nearest the split point with only absolute accuracy --
  // the same loss this whole path exists to avoid, just moved into the node table.
  template <class Real> static const Vector<Real>& NearSubOffs(const Integer order) {
    constexpr Integer MAX_ORDER = 50;
    SCTL_ASSERT(1 < order && order <= MAX_ORDER);
    static const Vector<Vector<Real>> all = []() {
      Vector<Vector<Real>> v(MAX_ORDER + 1);
      for (Integer n = 2; n <= MAX_ORDER; n++) {
        v[n].ReInit(n);
        using W = typename NearTabWork<Real>::type;
        for (Integer i = 0; i < n; i++) {
          const W ch = cos<W>(const_pi<W>()*i/(2*(n-1)));
          v[n][i] = (Real)(ch*ch);
        }
        v[n][0] = 1; v[n][n-1] = 0;               // endpoints exact
      }
      return v;
    }();
    return all[order];
  }

  // d/ds of the sub-element basis at its own nodes, D[i][a] = L_i'(node_a).
  template <class Real> static const Matrix<Real>& NearSubDiffMat(const Integer order) {
    constexpr Integer MAX_ORDER = 50;
    SCTL_ASSERT(1 < order && order <= MAX_ORDER);
    static const Vector<Matrix<Real>> all = []() {
      Vector<Matrix<Real>> D(MAX_ORDER + 1);
      for (Integer n = 2; n <= MAX_ORDER; n++) {
        const Vector<Real>& nds = NearSubNodes<Real>(n);
        Vector<Real> f((Long)n*n); f.SetZero();
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

  template <class Real> template <Integer order> const Vector<typename QuadElemList<Real>::GradeRule>& QuadElemList<Real>::NearGradeTable(const Integer q) {
    // Built once per `order`. Every entry is in NORMALIZED sub-element coordinates and carries
    // no positional index -- that is the point of splitting at the foot.
    // Every entry is formed in the work type and rounded to Real only on the way out, so no
    // intermediate step inherits Real rounding from the one before it. That means the Gauss rule,
    // the node offsets and the sub-element derivative operator all have to arrive at work
    // precision too -- a table built from already-rounded inputs gains nothing from working wide.
    auto build = [](const Integer q) {
      using W = typename NearTabWork<Real>::type;
      const Vector<W>& sig = NearSubOffs<W>(order);       // sub-element nodes, offset from the foot
      const Matrix<W>& Dsub = NearSubDiffMat<W>(order);   // d/ds of the sub-element basis at its nodes
      Vector<W> qn, qw; LegQuadRule<W>::template ComputeNdsWts<W>(&qn, &qw, q);
      Vector<GradeRule> tab(2*MaxNearLvl);
      Vector<W> tq(q), Twts((Long)order*q);
      Matrix<W> dT(order, q);
      auto fill = [&](GradeRule& r, const Real a, const Real b) {
        r.a = a; r.b = b;
        const W aw = (W)a, w = (W)b - (W)a;
        r.nds.ReInit(q); r.w.ReInit(q);
        for (Integer i = 0; i < q; i++) { r.nds[i] = (Real)(aw + w*qn[i]); r.w[i] = (Real)(w*qw[i]); }
        // T[i][j] = Lhat_i(nds[j]): sub-element nodes -> this interval's quadrature nodes.
        //
        // Interpolated in t, the offset from the foot, NOT in nds -- both the nodes (sig) and the
        // quadrature points are measured from there. The interval ends are 1 - 2^-k so t_hi = 1-a
        // and t_lo = 1-b are exact and t = t_hi - (t_hi-t_lo)*qn keeps full relative accuracy,
        // whereas nds = a + w*qn adds ~2^-k to ~1 and rounds at ABSOLUTE eps. The foot is itself a
        // node, at sig = 0, so its weight is 1 - O(t) and EVERY other weight carries a factor of t:
        // an interpolated position is a sum of terms that are each O(t) rather than a cancelling
        // sum of O(1) ones, which is what makes its error relative instead of absolute.
        const W t_hi = (W)1 - aw, t_w = t_hi - ((W)1 - (W)b);
        for (Integer j = 0; j < q; j++) tq[j] = t_hi - t_w*qn[j];
        LagrangeInterp<W>::Interpolate(Twts, sig, tq);   // pre-sized, so the view below stays valid
        const Matrix<W> T(order, q, Twts.begin(), false);
        Matrix<W>::GEMM(dT, Dsub, T);
        r.T.ReInit(order, q); r.dT.ReInit(order, q);
        r.TT.ReInit(q, order); r.TD.ReInit(2*q, order);
        for (Integer i = 0; i < order; i++) for (Integer j = 0; j < q; j++) {
          r.T[i][j] = (Real)T[i][j]; r.dT[i][j] = (Real)dT[i][j];
          r.TT[j][i] = r.T[i][j]; r.TD[j][i] = r.T[i][j]; r.TD[q+j][i] = r.dT[i][j];
        }
      };
      for (Integer k = 0; k < MaxNearLvl; k++) {
        const Real lo = 1 - pow<Real>((Real)0.5, k), hi = 1 - pow<Real>((Real)0.5, k+1);
        fill(tab[k], lo, hi);                                   // shell_k
        fill(tab[MaxNearLvl + k], lo, (Real)1);                  // core_k = [1-2^-k, 1]
      }
      return tab;
    };
    // One static init builds every rung the corner-angle correction can select: each multiple
    // of 4 up to NearMaxQuadOrder, plus each accuracy level's isotropic order (which need not
    // be a multiple of 4). The per-target lookup has to be O(1) and allocation-free.
    static const std::vector<Vector<GradeRule>> all = [&build]() {
      std::vector<Vector<GradeRule>> t(NearMaxQuadOrder+1);
      for (Integer q = 4; q <= NearMaxQuadOrder; q += 4) t[q] = build(q);
      for (Integer d = 0; d < MaxDigits; d++) {
        const Integer qi = NearQuadOrder(d);
        if (qi > 0 && qi <= NearMaxQuadOrder && t[qi].Dim() == 0) t[qi] = build(qi);
      }
      return t;
    }();
    SCTL_ASSERT(q > 0 && q <= NearMaxQuadOrder && all[q].Dim());
    return all[q];
  }

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracBlockSplit(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Integer digits, const Vector<Real>& proxy_off, const Vector<Real>& proxy_w) {
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

    const Real b_ellipse = NearBEllipse(digits);


    // Foot of the target on the element.
    Real ustar, vstar;
    const Real dist = qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);

    // Corner-angle correction to the near GL order. The required order is flat to ~120 deg,
    // then grows like 1/(180-phi) as the corner flattens and the element wraps around the
    // target; the parameter-space admissibility test cannot see this. phi is the acute angle
    // between the surface tangents at the foot -- the corner the target actually sees -- so an
    // orthogonal parametrisation gives phi=90, a factor of 1, and costs well-shaped meshes
    // nothing. Ck is fitted on Laplace SL/DL, flat elements, one target offset.
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
      return std::min<Integer>(NearMaxQuadOrder, std::max<Integer>(q_iso, q));
    };
    Real spd_u, spd_v;
    Integer q_near;
    { // The tangents at the foot give both the surface speeds and the GL order, so the order costs
      // nothing extra. Refinement stops per sub-element via the admissibility test in the loop
      // below, so no global depth is needed; only the per-direction surface speeds are.
      Real Xc[COORD_DIM], dXu[COORD_DIM], dXv[COORD_DIM];
      qel.EvalPoint(Xc, dXu, dXv, ustar, vstar, elem_idx, nullptr);
      Real su2 = 0, sv2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXu[k]*dXu[k]; sv2 += dXv[k]*dXv[k]; }
      spd_u = sqrt<Real>(su2); spd_v = sqrt<Real>(sv2);
      q_near = near_order(&dXu[0], &dXv[0], NearQuadOrder(digits));
    }
    const Vector<GradeRule>& tab = NearGradeTable<order>(q_near);
    const Real slen[2][2] = {{ustar, 1-ustar}, {vstar, 1-vstar}};   // [dir][side] sub-element length

    thread_local Vector<Real> cs;
    { // Target-shifted element nodal coords, component-major: one contiguous (COORD_DIM*order x order).
      if (cs.Dim() != COORD_DIM*nnode) cs.ReInit(COORD_DIM*nnode);
      const Long base = elem_idx * nnode * COORD_DIM;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real ok = Xtrg[k];
        for (Long p = 0; p < nnode; p++) cs[k*nnode + p] = qel.coord[base + k*nnode + p] - ok;
      }
    }
    // S[i][a] = L_i(sub[a]), element nodes -> sub-element nodes; side 1 mirrored so BOTH sides
    // grade toward the foot at normalized x = 1. The v-contraction needs S, the u-contraction
    // needs S^T, so build only the one each direction uses.
    //
    // BOTH node sets are measured from the foot, and the sub-element nodes come straight from the
    // half-angle offsets rather than from one-minus-node, so the node AT the foot is exactly 0 and
    // its neighbours keep full relative accuracy. This is what makes the weights right near the
    // foot: when the foot is itself an element node -- the on-surface case -- that node's shifted
    // coordinate is exactly 0, so every other weight carries an exact factor of the offset instead
    // of a cancelling difference of two order-one parameters, and the barycentric form takes its
    // exact-hit branch at the foot. Absolute parameters lose that to rounding.
    const auto build_interp = [](Matrix<Real> (&Sf)[2][2], Matrix<Real> (&St)[2][2], const Real (&slen)[2][2], const Real ustar, const Real vstar) {
      const Long nnode = (Long)order*order;
      const Vector<Real>& gnds = ParamNodes(order);          // element basis (density)
      const Vector<Real>& soff = NearSubOffs<Real>(order);   // sub-element nodes, offset from the foot
      thread_local Vector<Real> gsh, sub, Sbuf;
      if (sub.Dim() != order) { gsh.ReInit(order); sub.ReInit(order); Sbuf.ReInit(nnode); }
      for (Integer d = 0; d < 2; d++) {
        const Real xs = (d ? vstar : ustar);
        for (Integer i = 0; i < order; i++) gsh[i] = gnds[i] - xs;   // element nodes, foot at the origin
        for (Integer sd = 0; sd < 2; sd++) {
          if (!(slen[d][sd] > 0)) continue;
          const Real sg = (sd ? slen[d][sd] : -slen[d][sd]);   // side 0 runs toward decreasing parameter
          for (Integer i = 0; i < order; i++) sub[i] = sg*soff[i];
          { Vector<Real> v(nnode, Sbuf.begin(), false); LagrangeInterp<Real>::Interpolate(v, gsh, sub); }
          Sf[d][sd].ReInit(order, order); St[d][sd].ReInit(order, order);
          for (Integer i = 0; i < order; i++) for (Integer aa = 0; aa < order; aa++) {
            Sf[d][sd][i][aa] = Sbuf[i*order+aa];   // S
            St[d][sd][aa][i] = Sbuf[i*order+aa];   // S^T
          }
        }
      }
    };
    thread_local Matrix<Real> Sf[2][2], St[2][2];
    build_interp(Sf, St, slen, ustar, vstar);
    // Per-quadrant nodal geometry slab Xsub = S_u^T . cs . S_v, built ONCE per target, so nothing
    // depending on (u*,v*) survives into the per-cell loop and every per-cell operator stays a
    // precomputed table entry. The v-contraction batches all COORD_DIM components.
    const auto build_geom = [](Vector<Real> (&Xsub)[2][2], Vector<Real>& cs, const Matrix<Real> (&Sf)[2][2], const Matrix<Real> (&St)[2][2], const Real (&slen)[2][2]) {
      const Long nnode = (Long)order*order;
      thread_local Vector<Real> Av[2];
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
    };
    thread_local Vector<Real> Xsub[2][2];
    build_geom(Xsub, cs, Sf, St, slen);

    // Every cell operator is a table entry now. nrm_sign corrects the normal on quadrants with
    // exactly one mirrored direction, where d/dx_u x d/dx_v is anti-parallel to dXu x dXv; the
    // area element carries |du/dx . dv/dx| = slen_u.slen_v, so weights stay the normalized g.w.
    // Xsub and acc are thread_local, so they cannot be captured; they are referenced directly.
    const auto emit = [&tab, &normal_trg, &ker, &proxy_off, &proxy_w](const Integer sdu, const Integer sdv, const Integer iu, const Integer iv) {
      const GradeRule& gu = tab[iu];
      const GradeRule& gv = tab[iv];
      if (!(gu.b > gu.a) || !(gv.b > gv.a)) return;
      const Real nsign = ((sdu == 1) != (sdv == 1)) ? (Real)-1 : (Real)1;
      IntegrateBlock<order>(normal_trg, gu.w, gv.w, ker,
                            gu.T, gu.TT, gu.TD, gv.T, gv.dT, gv.TT,
                            Xsub[sdu][sdv], nsign, acc, proxy_off, proxy_w);
    };
    // Bisect the corner cell along its longer physical dimension (hu, hv = parameter extent x
    // surface speed) until that side is admissible, emitting one leaf per split.
    const auto refine = [&emit, dist, b_ellipse](const Integer sdu, const Integer sdv, Real hu, Real hv) {
      Integer ku = 0, kv = 0;
      const bool cap = !(dist > 0) || !std::isfinite((double)dist);
      // Near-touching targets (a neighbouring patch's node, foot distance ~0) refine to the
      // cap regardless of the admissibility constant, so the cap -- not b_ellipse -- is what
      // controls their error.
      constexpr Integer KMAX = MaxNearLvl-1;   // table bound
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
    };
    for (Integer sdu = 0; sdu < 2; sdu++) {
      if (!(slen[0][sdu] > 0)) continue;
      for (Integer sdv = 0; sdv < 2; sdv++) {
        if (!(slen[1][sdv] > 0)) continue;
        acc.SetZero();
        refine(sdu, sdv, slen[0][sdu]*spd_u, slen[1][sdv]*spd_v);

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


  // ---- Duffy edge-collapsed self scheme ----


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

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::SelfInteracHelper(Vector<Matrix<Real>>& M_lst, const Kernel& ker, bool trg_dot_prod, const ElementListBase<Real>* self, const Integer digits) {
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

    static constexpr Integer sing_order = KernelSingularOrder<Kernel>::value;
    // hh_w must NOT be thread_local: it is filled outside the parallel region, so only the master
    // would have it and the workers would see it empty -- which silently drops IntegrateBlock into
    // its no-proxy path. Invisible on a single element, since the master takes it.
    Vector<Real> hh_w;
    if (UseHedgehogSelf) HedgehogWeights(hh_w);
    const Real hh_c = (UseHedgehogSelf ? HedgehogRminCoeff(digits, sing_order) : (Real)0);
    const Integer near_digits = (UseHedgehogSelf ? HedgehogNearDigits(digits, sing_order) : digits);

    // Build this order's tables before the parallel loop so the first iteration does not
    // serialize the rest on first-touch static init. ParamNodes / DiffMat come along with it.
    if (UseHedgehogSelf) NearGradeTable<order>(NearQuadOrder(near_digits));
    else                 DuffyTable<order>();

    // Per-element blocks are independent: each writes its own M_lst[elem_idx], temporaries are
    // loop-local, and GetGeom/table reads are const.
    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < qel.nelem; elem_idx++) {
      // Surface nodes (targets) and their normals on this element.
      thread_local Vector<Real> Xnodes, Xnnodes, dXu, dXv;
      thread_local Matrix<Real> M_acc;
      // Hedgehog needs the normal to lay the proxy line along, and the tangents to turn the
      // parameter-space edge distance into a physical one.
      qel.GetGeom(&Xnodes, (UseHedgehogSelf || trg_dot_prod ? &Xnnodes : nullptr), nullptr,
                  (UseHedgehogSelf ? &dXu : nullptr), (UseHedgehogSelf ? &dXv : nullptr), nds, nds, elem_idx);

      // Proxy line along the outward normal. Returns the closest proxy in Xt1 and the rest as
      // offsets from it, which is what the near scheme takes.
      const auto proxy_line = [&nds, hh_c](Vector<Real>& Xt1, Vector<Real>& off, const Long t, const Integer ti, const Integer tj) {
        Real su2 = 0, sv2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          su2 += dXu[t*COORD_DIM+k]*dXu[t*COORD_DIM+k];
          sv2 += dXv[t*COORD_DIM+k]*dXv[t*COORD_DIM+k];
        }
        const Real du = std::min<Real>(nds[ti], 1-nds[ti]), dv = std::min<Real>(nds[tj], 1-nds[tj]);
        const Real edge_dist = std::min<Real>(du*sqrt<Real>(su2), dv*sqrt<Real>(sv2));
        const Real rmin = hh_c * edge_dist;
        for (Integer k = 0; k < COORD_DIM; k++) Xt1[k] = Xnodes[t*COORD_DIM+k] + rmin*Xnnodes[t*COORD_DIM+k];
        const Vector<Real>& soff = HedgehogProxyOffsets();
        for (Long j = 0; j < soff.Dim(); j++) {
          const Real rj = rmin*soff[j];
          for (Integer k = 0; k < COORD_DIM; k++) off[j*COORD_DIM+k] = (rj-rmin)*Xnnodes[t*COORD_DIM+k];
        }
      };
      thread_local Vector<Real> hh_Xt1, hh_off;
      thread_local Matrix<Real> M_hh;
      if (UseHedgehogSelf && hh_Xt1.Dim() != COORD_DIM) { hh_Xt1.ReInit(COORD_DIM); hh_off.ReInit(HedgehogProxyOffsets().Dim()*COORD_DIM); }

      Matrix<Real>& M = M_lst[elem_idx];
      if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nnode*KDIM0, nnode*KDIM1_out);
      M.SetZero();

      for (Integer ti = 0; ti < order; ti++) {
        for (Integer tj = 0; tj < order; tj++) {
          const Long t = ti*order + tj; // target node index = column block

          Vector<Real> Xtrg(COORD_DIM, Xnodes.begin() + t*COORD_DIM, false);
          Vector<Real> ntrg;
          if (trg_dot_prod) ntrg.ReInit(COORD_DIM, Xnnodes.begin() + t*COORD_DIM, false);

          if (UseHedgehogSelf) {
            proxy_line(hh_Xt1, hh_off, t, ti, tj);
            NearInteracHelper<order>(M_hh, hh_Xt1, ntrg, ker, elem_idx, self, near_digits, hh_off, hh_w);
          } else {
            SelfInteracBlockDuffy<order>(M_acc, qel, elem_idx, ti, tj, Xtrg, ntrg, ker, digits);
          }

          // Scatter into column block t of M: M[(i*order+j)*KDIM0+k0][t*KDIM1_out+k1]. The two
          // schemes hand back the same numbers in different layouts.
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) {
              const Long pnode = i*order + j;
              for (Integer k0 = 0; k0 < KDIM0; k0++) {
                for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
                  M[pnode*KDIM0+k0][t*KDIM1_out+k1] = (UseHedgehogSelf ? M_hh[pnode*KDIM0+k0][k1]
                                                                       : M_acc[pnode][k0*KDIM1_out+k1]);
                }
              }
            }
          }
        }
      }
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    // Dispatch the runtime element order to a compile-time `order` in {4,8,12,16,20}; the
    // tolerance stays runtime (see MaxDigits note in the header).
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->order;
    const Integer digits = DigitsFromTol(tol);
    switch (order) {
      case  4: SelfInteracHelper<4>(M_lst, ker, trg_dot_prod, self, digits); break;
      case  8: SelfInteracHelper<8>(M_lst, ker, trg_dot_prod, self, digits); break;
      case 12: SelfInteracHelper<12>(M_lst, ker, trg_dot_prod, self, digits); break;
      case 16: SelfInteracHelper<16>(M_lst, ker, trg_dot_prod, self, digits); break;
      case 20: SelfInteracHelper<20>(M_lst, ker, trg_dot_prod, self, digits); break;
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,12,16,20} for the templated near/self schemes.");
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

    Real u, v;
    // GetClosestNode returns a DISTANCE; f is the squared residual |r|^2 that the optimality
    // test, the line search and the return value all assume. Without squaring here, a seed that
    // is already optimal -- the target sitting on the normal through a node -- breaks out before
    // f is ever reassigned, and the routine returns sqrt(dist) instead of dist.
    const Real f_seed = GetClosestNode(u, v, elem_idx, Xtrg);
    Real f = f_seed * f_seed;

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

      // First-order optimality (KKT for the box [0,1]^2) via the PROJECTED gradient: at an active
      // bound only the feasible-direction component counts (f = |r|^2, so d f/du = 2 gu; a lower
      // bound u=0 is stationary when gu >= 0, an upper bound u=1 when gu <= 0), interior when gu
      // is negligible vs sqrt(E*f). The projected gradient is the right test at a bound because
      // the constrained gu is large there -- it balances the constraint -- and metric coupling
      // (F != 0) can flip the sign of the coupled Newton step. Near-pair feet usually lie on a
      // shared patch edge, so this is the common case, not a corner case.
      Real Pu = gu, Pv = gv;
      if      (u <= 0) Pu = std::min<Real>(gu, (Real)0);
      else if (u >= 1) Pu = std::max<Real>(gu, (Real)0);
      if      (v <= 0) Pv = std::min<Real>(gv, (Real)0);
      else if (v >= 1) Pv = std::max<Real>(gv, (Real)0);
      const bool opt_u = (fabs(Pu) <= gtol * sqrt<Real>(E*f));
      const bool opt_v = (fabs(Pv) <= gtol * sqrt<Real>(G*f));
      if (opt_u && opt_v) { converged = true; break; }

      // ACTIVE-SET reduced Gauss-Newton step: a coordinate pinned at a bound by an outward
      // gradient is held FIXED and the step is solved in the free subspace only, so the metric
      // coupling F cannot contaminate the surviving component with the constrained gradient.
      // u_act implies opt_u (Pu is then exactly 0), so both-active is already converged.
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
      // If the clamped Newton step stalls, retry along the projected (metric-scaled) gradient:
      // it is a feasible descent direction whenever the point is not KKT-optimal, which the test
      // above has already established, so some backtracked step lowers f.
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
      if (!improved) break; // genuine stall -> grid-search fallback
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

  template <class Real> template <Integer order, class Kernel> void QuadElemList<Real>::NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx, const ElementListBase<Real>* self, const Integer digits, const Vector<Real>& proxy_off, const Vector<Real>& proxy_w) {
    // Off-surface near-singular targets, one block per target. On-surface (singular) self
    // interactions are built by SelfInterac instead.
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

    thread_local Matrix<Real> M_acc;   // persists across calls so ReInit reuses its capacity
    for (Long t = 0; t < Ntrg; t++) {
      Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
      Vector<Real> ntrg;
      if (trg_dot_prod) ntrg.ReInit(COORD_DIM, (Iterator<Real>)normal_trg.begin() + t*COORD_DIM, false);

      NearInteracBlockSplit<order>(M_acc, qel, elem_idx, Xtrg, ntrg, ker, digits, proxy_off, proxy_w);

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

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    // Dispatch the runtime element order to a compile-time `order` in {4,8,12,16,20}; the
    // tolerance stays runtime (see MaxDigits note in the header).
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->order;
    const Integer digits = DigitsFromTol(tol);
    switch (order) {
      case  4: NearInteracHelper<4>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case  8: NearInteracHelper<8>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case 12: NearInteracHelper<12>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case 16: NearInteracHelper<16>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      case 20: NearInteracHelper<20>(M, Xt, normal_trg, ker, elem_idx, self, digits); break;
      default: SCTL_ASSERT_MSG(false, "QuadElemList element order must be one of {4,8,12,16,20} for the templated near/self schemes.");
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInteracHedgehog(Matrix<Real>& M, const Vector<Real>& Xt_proxy, const Vector<Real>& wts, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    const Long p_ = wts.Dim();
    SCTL_ASSERT(p_ > 0 && Xt_proxy.Dim() == p_*COORD_DIM);
    // Foot, distance and refinement are driven by the CLOSEST proxy (first in Xt_proxy), so the
    // one hierarchy built here is fine enough for every proxy on the line; the rest enter as
    // offsets from it.
    StaticArray<Real,COORD_DIM> Xt_;
    ScratchBuf<Real> off_buf(p_*COORD_DIM);
    Vector<Real> Xt(COORD_DIM, Xt_, false), off(off_buf);
    for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xt_proxy[k];
    for (Long j = 0; j < p_; j++)
      for (Integer k = 0; k < COORD_DIM; k++) off[j*COORD_DIM+k] = Xt_proxy[j*COORD_DIM+k] - Xt[k];
    const Integer order = static_cast<const QuadElemList<Real>*>(self)->order;
    const Integer digits = DigitsFromTol(tol);
    switch (order) {
      case  4: NearInteracHelper<4>(M, Xt, normal_trg, ker, elem_idx, self, digits, off, wts); break;
      case  8: NearInteracHelper<8>(M, Xt, normal_trg, ker, elem_idx, self, digits, off, wts); break;
      case 12: NearInteracHelper<12>(M, Xt, normal_trg, ker, elem_idx, self, digits, off, wts); break;
      case 16: NearInteracHelper<16>(M, Xt, normal_trg, ker, elem_idx, self, digits, off, wts); break;
      case 20: NearInteracHelper<20>(M, Xt, normal_trg, ker, elem_idx, self, digits, off, wts); break;
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
