#ifndef _SCTL_QUADRULE_TXX_
#define _SCTL_QUADRULE_TXX_

#include <algorithm>                 // for max, min, lower_bound, sort
#include <functional>                // for greater
#include <iomanip>                   // for operator<<, setw, setprecision
#include <iostream>                  // for basic_ostream, cout, operator<<
#include <tuple>                     // for tie, make_tuple
#include <utility>                   // for pair

#include "sctl/common.hpp"           // for Long, Integer, SCTL_ASSERT, SCTL...
#include "sctl/quadrule.hpp"         // for LegQuadRule, ChebQuadRule, Inter...
#include "sctl/iterator.hpp"         // for Iterator
#include "sctl/iterator.txx"         // for Iterator::Iterator<ValueType>
#include "sctl/lagrange-interp.hpp"  // for LagrangeInterp
#include "sctl/math_utils.hpp"       // for fabs, sqrt, const_pi, cos, log
#include "sctl/math_utils.txx"       // for machine_eps, pow
#include "sctl/matrix.hpp"           // for Matrix
#include "sctl/static-array.hpp"     // for StaticArray
#include "sctl/vector.hpp"           // for Vector
#include "sctl/vector.txx"           // for Vector::operator[], Vector::Dim

namespace sctl {

  template <class Real> template <Integer MAX_ORDER, class ValueType> const Vector<Real>& ChebQuadRule<Real>::nds(Integer N) {
    SCTL_ASSERT(0 <= N && N <= MAX_ORDER);
    auto compute_all = [](){
      Vector<Vector<Real>> nds(MAX_ORDER+1);
      for (Long i = 0; i <= MAX_ORDER; i++) {
        ComputeNdsWts<ValueType>(&nds[i], nullptr, i);
      }
      return nds;
    };
    static const Vector<Vector<Real>> all_nds = compute_all();
    return all_nds[N];
  }
  template <class Real> template <Integer MAX_ORDER, class ValueType> const Vector<Real>& ChebQuadRule<Real>::wts(Integer N) {
    SCTL_ASSERT(0 <= N && N <= MAX_ORDER);
    auto compute_all = [](){
      Vector<Vector<Real>> wts(MAX_ORDER+1);
      for (Long i = 0; i <= MAX_ORDER; i++) {
        ComputeNdsWts<ValueType>(nullptr, &wts[i], i);
      }
      return wts;
    };
    static const Vector<Vector<Real>> all_wts = compute_all();
    return all_wts[N];
  }

  template <class Real> template <Integer N, class ValueType> const Vector<Real>& ChebQuadRule<Real>::nds() {
    auto compute = [](){
      Vector<Real> nds;
      ComputeNdsWts<ValueType>(&nds, nullptr, N);
      return nds;
    };
    static Vector<Real> nds = compute;
    return nds;
  }
  template <class Real> template <Integer N, class ValueType> const Vector<Real>& ChebQuadRule<Real>::wts() {
    const auto compute = [](){
      Vector<Real> wts;
      ComputeNdsWts<ValueType>(nullptr, &wts, N);
      return wts;
    };
    static Vector<Real> wts = compute();
    return wts;
  }

  template <class Real> template <class ValueType> void ChebQuadRule<Real>::ComputeNdsWts(Vector<Real>* nds, Vector<Real>* wts, Integer N){
    if (nds) {
      if (nds->Dim() != N) nds->ReInit(N);
      for (Long i = 0; i < N; i++) {
        (*nds)[i] = (Real)(1 - cos<ValueType>((2*i+1) * const_pi<ValueType>() / (2*N))) / 2;
      }
    }
    if (wts) {
      Matrix<ValueType> M_cheb_inv(N, N);
      { // Set M_cheb_inv
        const ValueType scal = 1/(ValueType)N;
        for (Long i = 0; i < N; i++) {
          ValueType theta = (2*i+1)*const_pi<ValueType>()/(2*N);
          for (Long j = 0; j < N; j++) {
            M_cheb_inv[i][j] = cos<ValueType>(j*theta) * scal * (j==0 ? 1 : 2);
          }
        }
      }

      Vector<ValueType> b(N); // integral of each Chebyshev polynomial
      for (Integer i = 0; i < N; i++) {
        b[i] = (i % 2 ? 0 : -(1/(ValueType)(i*i-1)));
      }

      if (wts->Dim() != N) wts->ReInit(N);
      for (Integer j = 0; j < N; j++) { // Solve: M_cheb * wts = b
        ValueType wts_ = 0;
        for (Integer i = 0; i < N; i++) {
          wts_ += M_cheb_inv[j][i] * b[i];
        }
        (*wts)[j] = (Real)wts_;
      }
    }
  }



  template <class Real> template <Integer MAX_ORDER, class ValueType> const Vector<Real>& LegQuadRule<Real>::nds(Integer N) {
    SCTL_ASSERT(0 <= N && N <= MAX_ORDER);
    auto compute_all = [](){
      Vector<Vector<Real>> nds(MAX_ORDER+1);
      for (Long i = 1; i <= MAX_ORDER; i++) {
        ComputeNdsWts<ValueType>(&nds[i], nullptr, i);
      }
      return nds;
    };
    static const Vector<Vector<Real>> all_nds = compute_all();
    return all_nds[N];
  }
  template <class Real> template <Integer MAX_ORDER, class ValueType> const Vector<Real>& LegQuadRule<Real>::wts(Integer N) {
    SCTL_ASSERT(0 <= N && N <= MAX_ORDER);
    auto compute_all = [](){
      Vector<Vector<Real>> wts(MAX_ORDER+1);
      for (Long i = 1; i <= MAX_ORDER; i++) {
        ComputeNdsWts<ValueType>(nullptr, &wts[i], i);
      }
      return wts;
    };
    static const Vector<Vector<Real>> all_wts = compute_all();
    return all_wts[N];
  }

  template <class Real> template <Integer N, class ValueType> const Vector<Real>& LegQuadRule<Real>::nds() {
    const auto compute = [](){
      Vector<Real> nds;
      ComputeNdsWts<ValueType>(&nds, nullptr, N);
      return nds;
    };
    static Vector<Real> nds = compute();
    return nds;
  }
  template <class Real> template <Integer N, class ValueType> const Vector<Real>& LegQuadRule<Real>::wts() {
    const auto compute = [](){
      Vector<Real> wts;
      ComputeNdsWts<ValueType>(nullptr, &wts, N);
      return wts;
    };
    static Vector<Real> wts = compute();
    return wts;
  }

  template <class Real> template <class ValueType> void LegQuadRule<Real>::ComputeNdsWts(Vector<Real>* nds, Vector<Real>* wts, Integer N) {
    if (!nds && !wts) return;
    Vector<ValueType> nds0, Pn, dPn;
    constexpr Long BUFF_SIZE = 128;
    StaticArray<ValueType,BUFF_SIZE*3> buff;
    nds0.ReInit(N, N>=BUFF_SIZE ? NullIterator<ValueType>() : buff+0*BUFF_SIZE, N>=BUFF_SIZE);
    Pn  .ReInit(N, N>=BUFF_SIZE ? NullIterator<ValueType>() : buff+1*BUFF_SIZE, N>=BUFF_SIZE);
    dPn .ReInit(N, N>=BUFF_SIZE ? NullIterator<ValueType>() : buff+2*BUFF_SIZE, N>=BUFF_SIZE);

    constexpr Integer MAX_ITER = 5;
    for (Long i = 0; i < N; i++) { // Inital guess for roots of Pn
      nds0[i] = -(1 - 1/(ValueType)(8*N*N) + 1/(ValueType)(8*N*N*N)) * cos<ValueType>(const_pi<ValueType>() * (4*i+3) / (4*N+2));
    }
    for (Integer iter = 0; iter < MAX_ITER; iter++) { // Newton iterations
      LegPoly(&Pn, &dPn, nds0, N);
      ValueType max_dx = 0;
      for (Long j = 0; j < N; j++) {
        const ValueType dx = Pn[j] / dPn[j];
        max_dx = std::max<ValueType>(max_dx, fabs(dx));
        nds0[j] -= dx;
      }
      if (max_dx < machine_eps<ValueType>()) break;
    }

    if (nds) {
      if (nds->Dim() != N) nds->ReInit(N);
      for (Integer i = 0; i < N; i++) {
        (*nds)[i] = (Real)((nds0[i] + 1) / 2);
      }
    }
    if (wts) {
      if (wts->Dim() != N) wts->ReInit(N);
      LegPoly<ValueType>(nullptr, &dPn, nds0, N);
      for (Long i = 0; i < N; i++) {
        (*wts)[i] = (Real)(1 / ((1 - nds0[i] * nds0[i]) * (dPn[i] * dPn[i])));
      }
    }
  }

  template <class Real> template <class ValueType> void LegQuadRule<Real>::LegPoly(Vector<ValueType>* P, Vector<ValueType>* dP, const Vector<ValueType>& X, Long degree){
    if (!P && !dP) return;
    const Long N = X.Dim();
    if (P && P->Dim() != N) P->ReInit(N);
    if (dP && dP->Dim() != N) dP->ReInit(N);
    if (!degree) {
      if (P) (*P) = 1;
      if (dP) (*dP) = 0;
      return;
    }

    Vector<ValueType> dP1, P0, P1;
    constexpr Long BUFF_SIZE = 128;
    StaticArray<ValueType,BUFF_SIZE*3> buff;
    dP1.ReInit(N, N>=BUFF_SIZE ? NullIterator<ValueType>() : buff+0*BUFF_SIZE, N>=BUFF_SIZE);
    P0 .ReInit(N, N>=BUFF_SIZE ? NullIterator<ValueType>() : buff+1*BUFF_SIZE, N>=BUFF_SIZE);
    P1 .ReInit(N, N>=BUFF_SIZE ? NullIterator<ValueType>() : buff+2*BUFF_SIZE, N>=BUFF_SIZE);
    dP1 = 1; P0 = 1; P1 = X;
    for (Long n = 2; n <= degree; n++) {
      const ValueType scal0 = -(n - 1) / (ValueType)n;
      const ValueType scal1 = (2 * n - 1) / (ValueType)n;
      for (Long i = 0; i < N; i++) {
        const ValueType Ptmp = X[i] * P1[i] * scal1 + P0[i] * scal0;
        P0[i] = P1[i];
        P1[i] = Ptmp;
        dP1[i] = P0[i] * n + dP1[i] * X[i];
      }
    }
    if (P) (*P) = P1;
    if (dP) (*dP) = dP1;
  }



  template <class Real> template <class BasisObj> Real InterpQuadRule<Real>::Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, const BasisObj& integrands, const Real interval_start, const Real interval_end, const Real eps, const Long ORDER, const Real nds_interval_start, const Real nds_interval_end, const bool UseSVD) {
    Vector<Real> nds, wts;
    adap_quad_rule(nds, wts, integrands, interval_start, interval_end, eps);
    Matrix<Real> M = integrands(nds);
    return Build(quad_nds, quad_wts, M, nds, wts, eps, ORDER, nds_interval_start, nds_interval_end, UseSVD);
  }

  template <class Real> Real InterpQuadRule<Real>::Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, const Matrix<Real> M, const Vector<Real>& nds, const Vector<Real>& wts, const Real eps, const Long ORDER, const Real nds_interval_start, const Real nds_interval_end, const bool UseSVD) {
    Vector<Real> eps_vec(1);
    Vector<Long> ORDER_vec(1);
    ORDER_vec[0] = ORDER;
    eps_vec[0] = eps;

    Vector<Vector<Real>> quad_nds_;
    Vector<Vector<Real>> quad_wts_;
    Vector<Real> cond_num_vec = Build(quad_nds_, quad_wts_, M, nds, wts, eps_vec, ORDER_vec, nds_interval_start, nds_interval_end, UseSVD);
    if (quad_nds_.Dim() &&  quad_wts_.Dim()) {
      quad_nds = quad_nds_[0];
      quad_wts = quad_wts_[0];
      return cond_num_vec[0];
    }
    return -1;
  }

  template <class Real> Vector<Real> InterpQuadRule<Real>::Build(Vector<Vector<Real>>& quad_nds, Vector<Vector<Real>>& quad_wts, const Matrix<Real>& M0, const Vector<Real>& nds, const Vector<Real>& wts, const Vector<Real>& eps_vec_, const Vector<Long>& ORDER_vec_, const Real nds_interval_start, const Real nds_interval_end, const bool UseSVD) {
    const Long N_rules = std::max(eps_vec_.Dim(), ORDER_vec_.Dim());
    if (M0.Dim(0) * M0.Dim(1) == 0 || N_rules == 0) return Vector<Real>();
    Matrix<Real> M = M0;

    Real min_eps;
    Long max_ORDER;
    Vector<Real> eps_vec;
    Vector<Long> ORDER_vec;
    { // Set eps_vec, ORDER_vec, min_eps, max_ORDER
      eps_vec = eps_vec_;
      ORDER_vec = ORDER_vec_;
      SCTL_ASSERT(eps_vec.Dim() == N_rules || eps_vec.Dim() == 0);
      SCTL_ASSERT(ORDER_vec.Dim() == N_rules || ORDER_vec.Dim() == 0);

      if (!eps_vec.Dim()) {
        eps_vec.ReInit(N_rules);
        eps_vec = machine_eps<Real>();
      }
      if (!ORDER_vec.Dim()) {
        ORDER_vec.ReInit(N_rules);
        ORDER_vec = M.Dim(1);
      }
      for (auto& x : eps_vec) x = std::max(x, machine_eps<Real>());
      for (auto& x : ORDER_vec) x = ( x<=0 ? M.Dim(1) : std::min(x, M.Dim(1)) );

      min_eps = eps_vec[0];
      max_ORDER = ORDER_vec[0];
      for (const auto& e : eps_vec) min_eps = std::min(min_eps, fabs(e));
      for (const auto k : ORDER_vec) max_ORDER = std::max(max_ORDER, k);
    }

    Vector<Real> sqrt_wts(wts.Dim());
    for (Long i = 0; i < sqrt_wts.Dim(); i++) { // Set sqrt_wts
      SCTL_ASSERT(wts[i] > 0);
      sqrt_wts[i] = sqrt<Real>(wts[i]);
    }
    for (Long i = 0; i < M.Dim(0); i++) { // M <-- diag(sqrt_wts) * M
      Real sqrt_wts_ = sqrt_wts[i];
      for (Long j = 0; j < M.Dim(1); j++) {
        M[i][j] *= sqrt_wts_;
      }
    }

    Vector<Real> S_vec;
    auto modified_gram_schmidt = [](Matrix<Real>& Q, Vector<Real>& S, Vector<Long>& pivot, const Matrix<Real>& M_, const Real tol_, const Long max_rows_, const bool verbose) { // orthogonalize rows
      const Long max_rows = std::min(max_rows_, std::min(M_.Dim(0), M_.Dim(1)));
      const Real tol = std::max(tol_, machine_eps<Real>());
      const Long N0 = M_.Dim(0), N1 = M_.Dim(1);
      if (N0*N1 == 0) return;

      Matrix<Real> M = M_;
      Vector<Real> row_norm(N0);
      S.ReInit(max_rows); S.SetZero();
      pivot.ReInit(max_rows); pivot = -1;
      Q.ReInit(max_rows, N1); Q.SetZero();
      for (Long i = 0; i < max_rows; i++) {
        #pragma omp parallel for schedule(static)
        for (Long j = 0; j < N0; j++) { // compute row_norm
          Real row_norm2 = 0;
          for (Long k = 0; k < N1; k++) {
            row_norm2 += M[j][k]*M[j][k];
          }
          row_norm[j] = sqrt<Real>(row_norm2);
        }

        Long pivot_idx = 0;
        Real pivot_norm = 0;
        for (Long j = 0; j < N0; j++) { // determine pivot
          if (row_norm[j] > pivot_norm) {
            pivot_norm = row_norm[j];
            pivot_idx = j;
          }
        }
        pivot[i] = pivot_idx;
        S[i] = pivot_norm;

        #pragma omp parallel for schedule(static)
        for (Long k = 0; k < N1; k++) Q[i][k] = M[pivot_idx][k] / pivot_norm;

        #pragma omp parallel for schedule(static)
        for (Long j = 0; j < N0; j++) { // orthonormalize
          Real dot_prod = 0;
          for (Long k = 0; k < N1; k++) dot_prod += M[j][k] * Q[i][k];
          for (Long k = 0; k < N1; k++) M[j][k] -= Q[i][k] * dot_prod;
        }

        if (verbose) std::cout<<pivot_norm/S[0]<<'\n';
        if (pivot_norm/S[0] < tol) {
          pivot[i] = -1;
          S[i] = 0;
          break;
        }
      }
    };
    auto approx_SVD = [&modified_gram_schmidt](Matrix<Real>& U, Matrix<Real>& S, Matrix<Real>& Vt, const Matrix<Real>& M, const Real tol, const Long N){
      Vector<Real> S_;
      Matrix<Real> Q_;
      Vector<Long> pivot;
      modified_gram_schmidt(Q_, S_, pivot, M, tol*0.1, (Long)(N*1.1), false);

      Long k = 0;
      while (k < S_.Dim() && S_[k] > 0) k++;
      modified_gram_schmidt(Q_, S_, pivot, Matrix<Real>(k, Q_.Dim(1), Q_.begin()), 0, k, false);
      Matrix<Real> Q(k, Q_.Dim(1), Q_.begin(), false);

      Matrix<Real> R(M.Dim(0), k);
      Matrix<Real>::GEMM(R, M, Q.Transpose());

      R.SVD(U, S, Vt);
      Vt = Vt * Q;
    };
    if (UseSVD) { // orthonormalize M and get truncation errors S_vec (using SVD)
      // TODO: try M = W * M where W is a random matrix to reduce number of rows in M
      Matrix<Real> U, S, Vt;
      approx_SVD(U, S, Vt, M, min_eps, max_ORDER); // faster than full SVD
      //M.SVD(U,S,Vt);

      Long N = S.Dim(0);
      S_vec.ReInit(N);
      Vector<std::pair<Real,Long>> S_idx_lst(N);
      for (Long i = 0; i < N; i++) {
        S_idx_lst[i] = std::pair<Real,Long>(S[i][i],i);
      }
      std::sort(S_idx_lst.begin(), S_idx_lst.end(), std::greater<std::pair<Real,Long>>());
      for (Long i = 0; i < N; i++) {
        S_vec[i] = S_idx_lst[i].first;
      }

      Matrix<Real> UU(nds.Dim(),N);
      for (Long i = 0; i < nds.Dim(); i++) {
        for (Long j = 0; j < N; j++) {
          UU[i][j] = U[i][S_idx_lst[j].second];
        }
      }
      M = UU;
    } else { // orthonormalize M and get truncation errors S_vec (using modified Gram-Schmidt)
      Matrix<Real> Q;
      Vector<Long> pivot;
      modified_gram_schmidt(Q, S_vec, pivot, M.Transpose(), min_eps, max_ORDER, false);

      if (1) {
        M = Q.Transpose();
      } else {
        M.ReInit(Q.Dim(1), Q.Dim(0));
        for (Long i = 0; i < Q.Dim(1); i++) {
          for (Long j = 0; j < Q.Dim(0); j++) {
            M[i][j] = Q[j][i] * S_vec[j];
          }
        }
      }
    }
    for (Long i = 0; i < N_rules; i++) {
      Long ORDER = std::lower_bound(S_vec.begin(), S_vec.end(), eps_vec[i]*S_vec[0], std::greater<Real>()) - S_vec.begin();
      ORDER = std::min(std::max<Long>(ORDER, 1), S_vec.Dim());
      ORDER_vec[i] = std::min(ORDER_vec[i], ORDER);
    }

    Vector<Real> cond_num_vec;
    quad_nds.ReInit(ORDER_vec.Dim());
    quad_wts.ReInit(ORDER_vec.Dim());
    auto build_quad_rule = [&nds_interval_start,&nds_interval_end, &nds, &modified_gram_schmidt](Vector<Real>& quad_nds, Vector<Real>& quad_wts, const Matrix<Real>& M, const Vector<Real>& sqrt_wts) {
      Long idx0 = 0, idx1 = nds.Dim();
      if (nds_interval_start != nds_interval_end) {
        idx0 = std::lower_bound(nds.begin(), nds.end(), nds_interval_start) - nds.begin();
        idx1 = std::lower_bound(nds.begin(), nds.end(), nds_interval_end  ) - nds.begin();
      }
      const Long N = M.Dim(0), ORDER = M.Dim(1);

      { // Set quad_nds
        Matrix<Real> M_(N, ORDER);
        for (Long i = 0; i < idx0*ORDER; i++) M_[0][i] = 0;
        for (Long i = idx1*ORDER; i < N*ORDER; i++) M_[0][i] = 0;
        for (Long i = idx0; i < idx1; i++) {
          for (Long j = 0; j < ORDER; j++) {
            M_[i][j] = M[i][j] / sqrt_wts[i];
          }
        }

        Matrix<Real> Q;
        Vector<Real> S;
        Vector<Long> pivot_rows;
        modified_gram_schmidt(Q, S, pivot_rows, M_, machine_eps<Real>(), ORDER, false);

        quad_nds.ReInit(ORDER);
        for (Long i = 0; i < ORDER; i++) {
          SCTL_ASSERT(0<=pivot_rows[i] && pivot_rows[i]<N);
          quad_nds[i] = nds[pivot_rows[i]];
        }
        std::sort(quad_nds.begin(), quad_nds.end());

        if (0) { // print spectrum of the sub-matrix
          Matrix<Real> MM(ORDER,ORDER);
          for (Long i = 0; i < ORDER; i++) {
            for (Long j = 0; j < ORDER; j++) {
              MM[i][j] = M[pivot_rows[i]][j];
            }
          }
          Matrix<Real> U, S, Vt;
          MM.SVD(U,S,Vt);
          std::cout<<S<<'\n';
        }
      }

      Real cond_num, smallest_wt = 1;
      { // Set quad_wts, cond_num
        const Matrix<Real> b = Matrix<Real>(1, sqrt_wts.Dim(), (Iterator<Real>)sqrt_wts.begin()) * M;

        Matrix<Real> MM(ORDER,ORDER);
        { // Set MM <-- M[quad_nds][:] / sqrt_wts
          Vector<std::pair<Real,Long>> sorted_nds(nds.Dim());
          for (Long i = 0; i < nds.Dim(); i++) {
            sorted_nds[i].first = nds[i];
            sorted_nds[i].second = i;
          }
          std::sort(sorted_nds.begin(), sorted_nds.end());
          for (Long i = 0; i < ORDER; i++) { // Set MM <-- M[quad_nds][:] / sqrt_wts
            Long row_id = std::lower_bound(sorted_nds.begin(), sorted_nds.end(), std::pair<Real,Long>(quad_nds[i],0))->second;
            Real inv_sqrt_wts = 1/sqrt_wts[row_id];
            for (Long j = 0; j < ORDER; j++) {
              MM[i][j] = M[row_id][j] * inv_sqrt_wts;
            }
          }
        }

        { // set quad_wts <-- b * MM.pinv()
          Matrix<Real> U, S, Vt;
          MM.SVD(U,S,Vt);
          Real Smax = S[0][0], Smin = S[0][0];
          for (Long i = 0; i < ORDER; i++) {
            Smin = std::min<Real>(Smin, fabs<Real>(S[i][i]));
            Smax = std::max<Real>(Smax, fabs<Real>(S[i][i]));
          }
          cond_num = Smax / Smin;
          auto quad_wts_ = (b * Vt.Transpose()) * S.pinv(machine_eps<Real>()) * U.Transpose();
          quad_wts = Vector<Real>(ORDER, quad_wts_.begin(), false);
          for (const auto& a : quad_wts) smallest_wt = std::min<Real>(smallest_wt, a);
        }
        //std::cout<<(Matrix<Real>(1,ORDER,quad_wts.begin())*(Matrix<Real>(ORDER,1)*0+1))[0][0]-1<<'\n';
      }
      std::cout<<"condition number = "<<cond_num<<"   nodes = "<<ORDER<<"   smallest_wt = "<<smallest_wt<<'\n';
      return cond_num;
    };
    for (Long i = 0; i < ORDER_vec.Dim(); i++) {
      const Long N0 = M.Dim(0);
      const Long N1_ = std::min(ORDER_vec[i],M.Dim(1));
      Matrix<Real> MM(N0, N1_);
      for (Long j0 = 0; j0 < N0; j0++) {
        for (Long j1 =   0; j1 < N1_; j1++) MM[j0][j1] = M[j0][j1];
      }
      Real cond_num = build_quad_rule(quad_nds[i], quad_wts[i], MM, sqrt_wts);
      cond_num_vec.PushBack(cond_num);
    }
    return cond_num_vec;
  }

  template <class Real> template <class FnObj> void InterpQuadRule<Real>::adap_quad_rule(Vector<Real>& nds, Vector<Real>& wts, const FnObj& fn, const Real a, const Real b, const Real tol) {
    static constexpr Integer LegOrder = 25;
    static const Real eps = machine_eps<Real>();
    static const Real sqrt_eps = sqrt<Real>(machine_eps<Real>());
    static const auto& nds0 = LegQuadRule<Real>::template nds<2*LegOrder>();
    static const auto& wts0 = LegQuadRule<Real>::template wts<2*LegOrder>();
    static const auto& nds1 = LegQuadRule<Real>::template nds<1*LegOrder>();
    static const auto Minterp = []() {
      Matrix<Real> M(nds1.Dim(), nds0.Dim());
      Vector<Real> wts(M.Dim(0)*M.Dim(1), M.begin(), false);
      LagrangeInterp<Real>::Interpolate(wts, nds1, nds0);
      return M;
    }();

    auto concat_vec = [](const Vector<Real>& v0, const Vector<Real>& v1) {
      Long N0 = v0.Dim();
      Long N1 = v1.Dim();
      Vector<Real> v(N0 + N1);
      for (Long i = 0; i < N0; i++) v[   i] = v0[i];
      for (Long i = 0; i < N1; i++) v[N0+i] = v1[i];
      return v;
    };
    auto interp_error = [&fn](Real a, Real b) {
      const Matrix<Real> M0 = fn(nds0*(b-a)+a);
      const Matrix<Real> M1 = fn(nds1*(b-a)+a);
      const auto err = M1.Transpose() * Minterp - M0.Transpose();

      Real max_err = 0, max_val = 0;
      for (const auto x : M0 ) max_val = std::max<Real>(max_val, fabs(x));
      for (const auto x : err) max_err = std::max<Real>(max_err, fabs(x));
      return std::make_tuple(max_err, max_val);
    };

    Real max_err, max_val;
    std::tie(max_err , max_val ) = interp_error(a, b);

    if (max_err/max_val < std::max(eps,tol)               /* relative interpolation-error of smooth part < tol */
        || max_val*fabs(b-a) < tol                        /* estimated magnitude of singular part < tol */
        || fabs(b-a)/std::max(fabs(a),fabs(b)) <= eps ) { /* not enough spatial resolution */
      nds = nds0 * (b-a) + a;
      wts = wts0 * (b-a);
      //std::cout<<"Converged; "<<a<<"      "<<b<<"      "<<max_err/max_val<<'\n';
    } else {
      Real max_err1, max_val1;
      Real max_err2, max_val2;
      std::tie(max_err1, max_val1) = interp_error(a, (a+b)/2);
      std::tie(max_err2, max_val2) = interp_error((a+b)/2, b);
      const Real sqrt_tol = std::max(sqrt_eps, sqrt<Real>(tol));
      if (std::min(max_err1,max_err2) < 0.5*max_err   /* still converging */
          || max_err/max_val > sqrt_tol) {            /* not in asymptototic part  */
        Vector<Real>  nds0_, wts0_, nds1_, wts1_;
        adap_quad_rule(nds0_, wts0_, fn, a, (a+b)/2, tol);
        adap_quad_rule(nds1_, wts1_, fn, (a+b)/2, b, tol);
        nds = concat_vec(nds0_, nds1_);
        wts = concat_vec(wts0_, wts1_);
      } else { /* not converging - stop */
        nds = nds0 * (b-a) + a;
        wts = wts0 * (b-a);
        //std::cout<<"Stagnated: "<<a<<"      "<<b<<"      "<<max_err/max_val<<'\n';
      }
    }
  }

  template <class Real> void InterpQuadRule<Real>::test() {
    Integer order = 16;
    auto integrands = [order](const Vector<Real>& nds) { // p(x) + q(x) log(x)
      const Long N = nds.Dim();
      Matrix<Real> M(N, order);
      for (Long j = 0; j < N; j++) {
        for (Long i = 0; i < order/2; i++) { // p(x)
          M[j][i] = pow<Real>(nds[j],i);
        }
        for (Long i = order/2; i < order; i++) { // q(x) log(x)
          M[j][i] = pow<Real>(nds[j],i-order/2) * log<Real>(nds[j]);
        }
      }
      return M;
    };

    Vector<Real> nds, wts;
    InterpQuadRule::Build(nds, wts, integrands, 0.0, 1.0, 1e-16, 0, 1e-4, 1, false);
    for (Integer i = 0; i < nds.Dim(); i++) {
      std::cout<<std::scientific<<std::setprecision(20);
      std::cout<<std::setw(27)<<nds[i]<<' '<<std::setw(27)<<wts[i]<<'\n';
    }
    std::cout<<"\n";
  }
}

#endif // _SCTL_QUADRULE_TXX_
