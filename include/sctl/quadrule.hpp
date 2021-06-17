#ifndef _SCTL_QUADRULE_HPP_
#define _SCTL_QUADRULE_HPP_

#include SCTL_INCLUDE(legendre_rule.hpp)
#include SCTL_INCLUDE(common.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(matrix.hpp)

namespace SCTL_NAMESPACE {

  template <class Real> class ChebQuadRule { // p(x)
    public:
      template <Integer MAX_ORDER=50> static const Vector<Real>& nds(Integer ChebOrder) {
        SCTL_ASSERT(ChebOrder < MAX_ORDER);
        auto compute_all = [](){
          Vector<Vector<Real>> nds(MAX_ORDER);
          for (Long i = 0; i < MAX_ORDER; i++) {
            nds[i] = ComputeNds(i);
          }
          return nds;
        };
        static const Vector<Vector<Real>> all_nds = compute_all();
        return all_nds[ChebOrder];
      }
      template <Integer MAX_ORDER=50> static const Vector<Real>& wts(Integer ChebOrder) {
        SCTL_ASSERT(ChebOrder < MAX_ORDER);
        auto compute_all = [](){
          Vector<Vector<Real>> wts(MAX_ORDER);
          for (Long i = 0; i < MAX_ORDER; i++) {
            wts[i] = ComputeWts(i);
          }
          return wts;
        };
        static const Vector<Vector<Real>> all_wts = compute_all();
        return all_wts[ChebOrder];
      }

      template <Integer ChebOrder> static const Vector<Real>& nds() {
        static Vector<Real> nds = ComputeNds(ChebOrder);
        return nds;
      }
      template <Integer ChebOrder> static const Vector<Real>& wts() {
        static const Vector<Real> wts = ComputeWts(ChebOrder);
        return wts;
      }

      static Vector<Real> ComputeNds(Integer ChebOrder){
        Vector<Real> nds(ChebOrder);
        for (Long i = 0; i < ChebOrder; i++) {
          nds[i] = 0.5 - cos<Real>((2*i+1)*const_pi<Real>()/(2*ChebOrder)) * 0.5;
        }
        return nds;
      }
      static Vector<Real> ComputeWts(Integer ChebOrder){
        Matrix<Real> M_cheb(ChebOrder, ChebOrder);
        { // Set M_cheb
          for (Long i = 0; i < ChebOrder; i++) {
            Real theta = (2*i+1)*const_pi<Real>()/(2*ChebOrder);
            for (Long j = 0; j < ChebOrder; j++) {
              M_cheb[j][i] = cos<Real>(j*theta);
            }
          }
          M_cheb = M_cheb.pinv(machine_eps<Real>());
        }

        Vector<Real> w_sample(ChebOrder);
        for (Integer i = 0; i < ChebOrder; i++) {
          w_sample[i] = (i % 2 ? 0 : -(ChebOrder/(Real)(i*i-1)));
        }

        Vector<Real> wts(ChebOrder);
        for (Integer j = 0; j < ChebOrder; j++) {
          wts[j] = 0;
          for (Integer i = 0; i < ChebOrder; i++) {
            wts[j] += M_cheb[j][i] * w_sample[i] / ChebOrder;
          }
        }
        return wts;
      }
  };

  template <class Real> class LegQuadRule {
    public:

      template <Integer MAX_ORDER=25> static const Vector<Real>& nds(Integer Order) {
        SCTL_ASSERT(Order < MAX_ORDER);
        auto compute_all = [](){
          Vector<Vector<Real>> nds(MAX_ORDER);
          for (Long i = 1; i < MAX_ORDER; i++) {
            nds[i] = ComputeNds<MAX_ORDER>(i);
          }
          return nds;
        };
        static const Vector<Vector<Real>> all_nds = compute_all();
        return all_nds[Order];
      }
      template <Integer MAX_ORDER=25> static const Vector<Real>& wts(Integer Order) {
        SCTL_ASSERT(Order < MAX_ORDER);
        auto compute_all = [](){
          Vector<Vector<Real>> wts(MAX_ORDER);
          for (Long i = 1; i < MAX_ORDER; i++) {
            wts[i] = ComputeWts<MAX_ORDER>(nds(i));
          }
          return wts;
        };
        static const Vector<Vector<Real>> all_wts = compute_all();
        return all_wts[Order];
      }

      template <Integer Order> static const Vector<Real>& nds() {
        static Vector<Real> nds = ComputeNds(Order);
        return nds;
      }
      template <Integer Order> static const Vector<Real>& wts() {
        static const Vector<Real> wts = ComputeWts(nds<Order>());
        return wts;
      }

      static Vector<Real> ComputeNds(Integer order) {
        Vector<Real> nds, wts;
        gauss_legendre_approx(nds, wts, order);
        nds = nds*2-1;

        auto EvalLegPoly = [](Vector<Real>& Pn, Vector<Real>& dPn, const Vector<Real>& nds, Long n) {
          Vector<Real> P, dP;
          LegPoly(P, nds, n);
          LegPolyDeriv(dP, nds, n);

          const Long M = nds.Dim();
          if (Pn.Dim() != M) Pn.ReInit(M);
          if (dPn.Dim() != M) dPn.ReInit(M);
          Pn = Vector<Real>(M, P.begin() + n*M, false);
          dPn = Vector<Real>(M, dP.begin() + n*M, false);
          for (Long i = 0; i < M; i++) dPn[i] = -dPn[i] / sqrt<Real>(1-nds[i]*nds[i]);
        };
        Vector<Real> Pn, dPn; // Newton iterations
        EvalLegPoly(Pn, dPn, nds, order); nds -= Pn/dPn;
        EvalLegPoly(Pn, dPn, nds, order); nds -= Pn/dPn;
        EvalLegPoly(Pn, dPn, nds, order); nds -= Pn/dPn;
        return (nds+1)/2;
      }
      static Vector<Real> ComputeWts(const Vector<Real>& nds) {
        const Long order = nds.Dim();
        Vector<Real> cheb_nds = ChebQuadRule<Real>::ComputeNds(2*order-1)*2-1;
        Vector<Real> cheb_wts = ChebQuadRule<Real>::ComputeWts(2*order-1)*2;

        auto EvalLegPoly = [](const Vector<Real>& nds, Long n) {
          Vector<Real> P;
          LegPoly(P, nds, n);
          const Long M = nds.Dim();
          return Matrix<Real>(n, M, P.begin());
        };
        Matrix<Real> b =  EvalLegPoly(cheb_nds, 2*order-1) * Matrix<Real>(cheb_wts.Dim(), 1, cheb_wts.begin(), false);
        Matrix<Real> M = EvalLegPoly(nds*2-1, 2*order-1);
        Matrix<Real> wts = Matrix<Real>(M).pinv() * b;
        return Vector<Real>(wts.Dim(0), wts.begin())/2;
      }

    private:

      static void LegPoly(Vector<Real>& poly_val, const Vector<Real>& X, Long degree){
        Vector<Real> theta(X.Dim());
        for (Long i = 0; i < X.Dim(); i++) theta[i] = acos<Real>(X[i]);
        LegPoly_(poly_val, theta, degree);
      }
      static void LegPoly_(Vector<Real>& poly_val, const Vector<Real>& theta, Long degree){
        Long N = theta.Dim();
        Long Npoly = (degree + 1) * (degree + 2) / 2;
        if (poly_val.Dim() != Npoly * N) poly_val.ReInit(Npoly * N);

        Real fact = 1 / sqrt<Real>(4 * const_pi<Real>());
        Vector<Real> cos_theta(N), sin_theta(N);
        for (Long n = 0; n < N; n++) {
          cos_theta[n] = cos(theta[n]);
          sin_theta[n] = sin(theta[n]);
          poly_val[n] = fact;
        }

        Long idx = 0;
        Long idx_nxt = 0;
        for (Long i = 1; i <= degree; i++) {
          idx_nxt += N*(degree-i+2);
          Real c = sqrt<Real>((2*i+1)/(Real)(2*i));
          for (Long n = 0; n < N; n++) {
            poly_val[idx_nxt+n] = -poly_val[idx+n] * sin_theta[n] * c;
          }
          idx = idx_nxt;
        }

        idx = 0;
        for (Long m = 0; m < degree; m++) {
          for (Long n = 0; n < N; n++) {
            Real pmm = 0;
            Real pmmp1 = poly_val[idx+n];
            for (Long ll = m + 1; ll <= degree; ll++) {
              Real a = sqrt<Real>(((2*ll-1)*(2*ll+1)         ) / (Real)((ll-m)*(ll+m)         ));
              Real b = sqrt<Real>(((2*ll+1)*(ll+m-1)*(ll-m-1)) / (Real)((ll-m)*(ll+m)*(2*ll-3)));
              Real pll = cos_theta[n]*a*pmmp1 - b*pmm;
              pmm = pmmp1;
              pmmp1 = pll;
              poly_val[idx + N*(ll-m) + n] = pll;
            }
          }
          idx += N * (degree - m + 1);
        }
      }
      static void LegPolyDeriv(Vector<Real>& poly_val, const Vector<Real>& X, Long degree){
        Vector<Real> theta(X.Dim());
        for (Long i = 0; i < X.Dim(); i++) theta[i] = acos<Real>(X[i]);
        LegPolyDeriv_(poly_val, theta, degree);
      }
      static void LegPolyDeriv_(Vector<Real>& poly_val, const Vector<Real>& theta, Long degree){
        Long N = theta.Dim();
        Long Npoly = (degree + 1) * (degree + 2) / 2;
        if (poly_val.Dim() != N * Npoly) poly_val.ReInit(N * Npoly);

        Vector<Real> cos_theta(N), sin_theta(N);
        for (Long i = 0; i < N; i++) {
          cos_theta[i] = cos(theta[i]);
          sin_theta[i] = sin(theta[i]);
        }

        Vector<Real> leg_poly(Npoly * N);
        LegPoly_(leg_poly, theta, degree);

        for (Long m = 0; m <= degree; m++) {
          for (Long n = m; n <= degree; n++) {
            ConstIterator<Real> Pn  = leg_poly.begin() + N * ((degree * 2 - m + 1) * (m + 0) / 2 + n);
            ConstIterator<Real> Pn_ = leg_poly.begin() + N * ((degree * 2 - m + 0) * (m + 1) / 2 + n) * (m < n);
            Iterator     <Real> Hn  = poly_val.begin() + N * ((degree * 2 - m + 1) * (m + 0) / 2 + n);

            Real c2 = sqrt<Real>(m<n ? (n+m+1)*(n-m) : 0);
            for (Long i = 0; i < N; i++) {
              Real c1 = (sin_theta[i]>0 ? m/sin_theta[i] : 0);
              Hn[i] = c1*cos_theta[i]*Pn[i] + c2*Pn_[i];
            }
          }
        }
      }
      static void gauss_legendre_approx(Vector<Real>& nds, Vector<Real>& wts, Integer order) {
        Vector<double> xd(order), wd(order);
        int kind = 1;
        double alpha = 0.0, beta = 0.0, a = 0.0, b = 1.0;
        cgqf(order, kind, (double)alpha, (double)beta, (double)a, (double)b, &xd[0], &wd[0]);

        if (nds.Dim()!=order) nds.ReInit(order);
        if (wts.Dim()!=order) wts.ReInit(order);
        for (Long i = 0; i < order; i++) {
          nds[i] = (Real)xd[i];
          wts[i] = (Real)wd[i];
        }
      }
  };

  template <class Real> class InterpQuadRule {
    public:
      template <class BasisObj> static Real Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, const BasisObj& integrands, Real eps = 1e-16, Long ORDER = 0, Real nds_start = 0, Real nds_end = 1) {
        Vector<Real> nds, wts;
        adap_quad_rule(nds, wts, integrands, 0, 1, eps);
        Matrix<Real> M = integrands(nds);
        return Build(quad_nds, quad_wts, M, nds, wts, eps, ORDER, nds_start, nds_end);
      }

      static Real Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, Matrix<Real> M, const Vector<Real>& nds, const Vector<Real>& wts, Real eps = 1e-16, Long ORDER = 0, Real nds_start = 0, Real nds_end = 1) {
        Vector<Real> eps_vec;
        Vector<Long> ORDER_vec;
        if (ORDER) {
          ORDER_vec.PushBack(ORDER);
        } else {
          eps_vec.PushBack(eps);
        }

        Vector<Vector<Real>> quad_nds_;
        Vector<Vector<Real>> quad_wts_;
        Vector<Real> cond_num_vec = Build(quad_nds_, quad_wts_, M, nds, wts, eps_vec, ORDER_vec, nds_start, nds_end);
        if (quad_nds_.Dim() &&  quad_wts_.Dim()) {
          quad_nds = quad_nds_[0];
          quad_wts = quad_wts_[0];
          return cond_num_vec[0];
        }
        return -1;
      }

      static Vector<Real> Build(Vector<Vector<Real>>& quad_nds, Vector<Vector<Real>>& quad_wts, Matrix<Real> M, const Vector<Real>& nds, const Vector<Real>& wts, Vector<Real> eps_vec = Vector<Real>(), Vector<Long> ORDER_vec = Vector<Long>(), Real nds_start = 0, Real nds_end = 1) {
        if (M.Dim(0) * M.Dim(1) == 0) return Vector<Real>();

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
        { // orthonormalize M and get singular values S_vec
          Matrix<Real> U, S, Vt;
          M.SVD(U,S,Vt);

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
        }
        if (eps_vec.Dim()) { //  Set ORDER_vec
          SCTL_ASSERT(!ORDER_vec.Dim());
          ORDER_vec.ReInit(eps_vec.Dim());
          for (Long i = 0; i < eps_vec.Dim(); i++) {
            ORDER_vec[i] = std::lower_bound(S_vec.begin(), S_vec.end(), eps_vec[i]*S_vec[0], std::greater<Real>()) - S_vec.begin();
            ORDER_vec[i] = std::min(std::max<Long>(ORDER_vec[i],1), S_vec.Dim());
          }
        }

        Vector<Real> cond_num_vec;
        quad_nds.ReInit(ORDER_vec.Dim());
        quad_wts.ReInit(ORDER_vec.Dim());
        auto build_quad_rule = [&nds_start, &nds_end, &nds](Vector<Real>& quad_nds, Vector<Real>& quad_wts, Matrix<Real> M, const Vector<Real>& sqrt_wts) {
          const Long ORDER = M.Dim(1);
          { // Set quad_nds
            auto find_largest_row = [&nds_start, &nds_end, &nds](const Matrix<Real>& M) {
              Long max_row = 0;
              Real max_norm = 0;
              for (Long i = 0; i < M.Dim(0); i++) {
                if (nds[i] < nds_start || nds[i] > nds_end) continue;
                Real norm = 0;
                for (Long j = 0; j < M.Dim(1); j++) {
                  norm += M[i][j]*M[i][j];
                }
                if (norm > max_norm) {
                  max_norm = norm;
                  max_row = i;
                }
              }
              return max_row;
            };
            auto orthogonalize_rows = [](Matrix<Real>& M, Vector<Long> pivot_rows) { // TODO: optimize
              if (!pivot_rows.Dim()) return M;
              Matrix<Real> MM(pivot_rows.Dim(), M.Dim(1));
              for (Long i = 0; i < MM.Dim(0); i++) {
                for (Long j = 0; j < MM.Dim(1); j++) {
                  MM[i][j] = M[pivot_rows[i]][j];
                }
              }
              Matrix<Real> U, S, Vt;
              MM.SVD(U,S,Vt);
              Matrix<Real> P = Vt.Transpose() * Vt;
              return M - M * P;
            };
            Vector<Long> pivot_rows;
            if (quad_nds.Dim() != ORDER) quad_nds.ReInit(ORDER);
            for (Long i = 0; i < ORDER; i++) {
              pivot_rows.PushBack(find_largest_row(orthogonalize_rows(M, pivot_rows)));
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

          Real cond_num;
          { // Set quad_wts, cond_num
            Vector<std::pair<Real,Long>> sorted_nds(nds.Dim());
            for (Long i = 0; i < nds.Dim(); i++) {
              sorted_nds[i].first = nds[i];
              sorted_nds[i].second = i;
            }
            std::sort(sorted_nds.begin(), sorted_nds.end());

            Matrix<Real> MM(ORDER,ORDER);
            for (Long i = 0; i < ORDER; i++) { // Set MM <-- M[quad_nds][:]
              Long row_id = std::lower_bound(sorted_nds.begin(), sorted_nds.end(), std::pair<Real,Long>(quad_nds[i],0))->second;
              Real inv_sqrt_wts = 1/sqrt_wts[row_id];
              for (Long j = 0; j < ORDER; j++) {
                MM[i][j] = M[row_id][j] * inv_sqrt_wts;
              }
            }
            Matrix<Real> b = Matrix<Real>(1, sqrt_wts.Dim(), (Iterator<Real>)sqrt_wts.begin()) * M;
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
            }
            //std::cout<<(Matrix<Real>(1,ORDER,quad_wts.begin())*(Matrix<Real>(ORDER,1)*0+1))[0][0]-1<<'\n';
          }
          std::cout<<"condition number = "<<cond_num<<"   nodes = "<<ORDER<<'\n';
          return cond_num;
        };
        for (Long i = 0; i < ORDER_vec.Dim(); i++) {
          Matrix<Real> MM(M.Dim(0), ORDER_vec[i]);
          for (Long j0 = 0; j0 < MM.Dim(0); j0++) {
            for (Long j1 = 0; j1 < MM.Dim(1); j1++) {
              MM[j0][j1] = M[j0][j1];
            }
          }
          Real cond_num = build_quad_rule(quad_nds[i], quad_wts[i], MM, sqrt_wts);
          cond_num_vec.PushBack(cond_num);
        }
        return cond_num_vec;
      }

      static void test() {
        const Integer ORDER = 28;
        auto integrands = [ORDER](const Vector<Real>& nds) {
          Integer K = ORDER;
          Long N = nds.Dim();
          Matrix<Real> M(N,K);
          for (Long j = 0; j < N; j++) {
            //for (Long i = 0; i < K; i++) {
            //  M[j][i] = pow<Real>(nds[j],i);
            //}
            for (Long i = 0; i < K/2; i++) {
              M[j][i] = pow<Real>(nds[j],i);
            }
            for (Long i = K/2; i < K; i++) {
              M[j][i] = pow<Real>(nds[j],K-i-1) * log<Real>(nds[j]);
            }
          }
          return M;
        };

        Vector<Real> nds, wts;
        Real cond_num = InterpQuadRule::Build(nds, wts, integrands);
        std::cout<<cond_num<<'\n';
      }

    private:
      template <class FnObj> static void adap_quad_rule(Vector<Real>& nds, Vector<Real>& wts, const FnObj& fn, Real a, Real b, Real tol) {
        const auto& nds0 = ChebQuadRule<Real>::template nds<40>();
        const auto& wts0 = ChebQuadRule<Real>::template wts<40>();
        const auto& nds1 = ChebQuadRule<Real>::template nds<20>();
        const auto& wts1 = ChebQuadRule<Real>::template wts<20>();

        auto concat_vec = [](const Vector<Real>& v0, const Vector<Real>& v1) {
          Long N0 = v0.Dim();
          Long N1 = v1.Dim();
          Vector<Real> v(N0 + N1);
          for (Long i = 0; i < N0; i++) v[   i] = v0[i];
          for (Long i = 0; i < N1; i++) v[N0+i] = v1[i];
          return v;
        };
        auto integration_error = [&fn,&nds0,&wts0,&nds1,&wts1](Real a, Real b) {
          const Matrix<Real> M0 = fn(nds0*(b-a)+a);
          const Matrix<Real> M1 = fn(nds1*(b-a)+a);
          const Long dof = M0.Dim(1);
          SCTL_ASSERT(M0.Dim(0) == nds0.Dim());
          SCTL_ASSERT(M1.Dim(0) == nds1.Dim());
          SCTL_ASSERT(M1.Dim(1) == dof);
          Real max_err = 0;
          for (Long i = 0; i < dof; i++) {
            Real I0 = 0, I1 = 0;
            for (Long j = 0; j < nds0.Dim(); j++) {
              I0 += M0[j][i] * wts0[j] * (b-a);
            }
            for (Long j = 0; j < nds1.Dim(); j++) {
              I1 += M1[j][i] * wts1[j] * (b-a);
            }
            max_err = std::max(max_err, fabs(I1-I0));
          }
          return max_err;
        };
        Real err = integration_error(a, b);
        if (err < tol) {
          //std::cout<<a<<"      "<<b<<"      "<<err<<'\n';
          nds = nds0 * (b-a) + a;
          wts = wts0 * (b-a);
        } else {
          Vector<Real>  nds0_, wts0_, nds1_, wts1_;
          adap_quad_rule(nds0_, wts0_, fn, a, (a+b)/2, tol);
          adap_quad_rule(nds1_, wts1_, fn, (a+b)/2, b, tol);
          nds = concat_vec(nds0_, nds1_);
          wts = concat_vec(wts0_, wts1_);
        }
      };
  };
}

#endif // _SCTL_QUADRULE_HPP_
