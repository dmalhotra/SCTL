#ifndef _SCTL_LAGRANGE_INTERP_TXX_
#define _SCTL_LAGRANGE_INTERP_TXX_

#include <iostream>                  // for cout

#include "sctl/common.hpp"           // for Long, Integer, SCTL_ASSERT, SCTL...
#include "sctl/lagrange-interp.hpp"  // for LagrangeInterp
#include "sctl/iterator.txx"         // for NullIterator
#include "sctl/matrix.hpp"           // for Matrix
#include "sctl/static-array.hpp"     // for StaticArray
#include "sctl/vec.hpp"              // for Vec
#include "sctl/vec.txx"              // for DefaultVecLen
#include "sctl/vector.hpp"           // for Vector

namespace sctl {

  template <class Real> void LagrangeInterp<Real>::test() {
    Vector<Real> src, trg; // source and target interpolation nodes
    for (Long i = 0; i < 3; i++) src.PushBack(i);
    for (Long i = 0; i < 11; i++) trg.PushBack(i*0.2);

    Matrix<Real> f(1,3); // values at source nodes
    f[0][0] = 0; f[0][1] = 1; f[0][2] = 0.5;

    // Interpolate
    Vector<Real> wts;
    Interpolate(wts,src,trg);
    Matrix<Real> Mwts(src.Dim(), trg.Dim(), wts.begin(), false);
    Matrix<Real> ff = f * Mwts;
    std::cout<<ff<<'\n';

    // Compute derivative
    Vector<Real> df;
    Derivative(df, Vector<Real>(f.Dim(0)*f.Dim(1),f.begin()), src);
    std::cout<<df<<'\n';
  }

  template <class Real> void LagrangeInterp<Real>::Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds) {
    static constexpr Integer VecLen = DefaultVecLen<Real>();
    using VecType = Vec<Real, VecLen>;
    VecType vec_one((Real)1);

    const Long Nsrc = src_nds.Dim();
    const Long Ntrg = trg_nds.Dim();
    const Long Ntrg_ = (Ntrg/VecLen)*VecLen;
    if (wts.Dim() != Nsrc*Ntrg) wts.ReInit(Nsrc*Ntrg);
    Matrix<Real> M(Nsrc, Ntrg, wts.begin(), false);

    StaticArray<Real,200> w_buff;
    Vector<Real> w(Nsrc, (Nsrc>=200?NullIterator<Real>():w_buff), (Nsrc>=200));
    const Real normal_factor = [src_nds]() { // normalize
      if (src_nds.Dim() < 2) return (Real)1;
      Real max_src = src_nds[0], min_src = src_nds[0];
      for (const auto x : src_nds) {
        max_src = std::max<Real>(max_src, x);
        min_src = std::min<Real>(min_src, x);
      }
      return 4/(max_src - min_src);
    }();
    for (Integer j = 0; j < Nsrc; j++) {
      Real w_inv = 1;
      Real src_nds_j(src_nds[j]);
      for (Integer k =   0; k <    j; k++) w_inv *= (src_nds[k] - src_nds_j)*normal_factor;
      for (Integer k = j+1; k < Nsrc; k++) w_inv *= (src_nds[k] - src_nds_j)*normal_factor;
      w[j] = 1/w_inv;
    }

    if (0) {
      for (Long i1 = 0; i1 < Ntrg_; i1+=VecLen) {
        VecType x = VecType::Load(&trg_nds[i1]);
        for (Integer j = 0; j < Nsrc; j++) {
          VecType y0(vec_one);
          for (Integer k =   0; k <    j; k++) y0 *= VecType(src_nds[k]) - x;
          for (Integer k = j+1; k < Nsrc; k++) y0 *= VecType(src_nds[k]) - x;
          VecType y = y0 * w[j];
          y.Store(&M[j][i1]);
        }
      }
      for (Long i1 = Ntrg_; i1 < Ntrg; i1++) {
        Real x = trg_nds[i1];
        for (Integer j = 0; j < Nsrc; j++) {
          Real y0 = 1;
          for (Integer k =   0; k <    j; k++) y0 *= src_nds[k] - x;
          for (Integer k = j+1; k < Nsrc; k++) y0 *= src_nds[k] - x;
          M[j][i1] = y0 * w[j];
        }
      }
    }
    if (1) { // Barycentric // TODO: vectorize
      //static constexpr Integer digits = (Integer)(TypeTraits<Real>::SigBits*0.3010299957);
      for (Long t = 0; t< Ntrg; t++) {
        Long s_ = -1;
        Real scal = 0;
        for (Long s = 0; s < Nsrc; s++) {
          if (trg_nds[t] == src_nds[s]) {
            s_ = s;
            break;
          }
          M[s][t] = w[s] / (trg_nds[t] - src_nds[s]);
          scal += M[s][t];
        }
        if (s_ == -1) {
          scal = 1/scal;
          for (Long s = 0; s < Nsrc; s++) M[s][t] *= scal;
        } else {
          for (Long s = 0; s < Nsrc; s++) M[s][t] = 0;
          M[s_][t] = 1;
        }
      }
    }
  }

  template <class Real> void LagrangeInterp<Real>::Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds) {
    Long N = nds.Dim();
    Long dof = f.Dim() / N;
    SCTL_ASSERT(f.Dim() == N * dof);
    if (df.Dim() != N * dof) df.ReInit(N * dof);
    if (N*dof == 0) return;

    const Real normal_factor = [nds]() { // normalize
      if (!nds.Dim()) return (Real)1;
      Real max_src = nds[0], min_src = nds[0];
      for (const auto x : nds) {
        max_src = std::max<Real>(max_src, x);
        min_src = std::min<Real>(min_src, x);
      }
      return 4/(max_src - min_src);
    }();

    auto dp = [&nds,&N,&normal_factor](Real x, Long i) {
      Real scal = 1;
      for (Long j = 0; j < N; j++) {
        if (i!=j) scal *= (nds[i] - nds[j]);
        scal *= normal_factor;
      }
      scal = 1/scal;
      Real wt = 0;
      for (Long k = 0; k < N; k++) {
        Real wt_ = 1;
        if (k!=i) {
          for (Long j = 0; j < N; j++) {
            if (j!=k && j!=i) wt_ *= (x - nds[j]);
            wt_ *= normal_factor;
          }
          wt += wt_;
        }
      }
      return wt * scal;
    };
    for (Long k = 0; k < dof; k++) {
      for (Long i = 0; i < N; i++) {
        Real df_ = 0;
        for (Long j = 0; j < N; j++) {
          df_ += (f[k*N+j]-f[k*N+i]) * dp(nds[i],j);
        }
        df[k*N+i] = df_;
      }
    }
  }

}

#endif // _SCTL_LAGRANGE_INTERP_TXX_
