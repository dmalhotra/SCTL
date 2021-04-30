#ifndef _SCTL_QUADRULE_HPP_
#define _SCTL_QUADRULE_HPP_

#include SCTL_INCLUDE(common.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(matrix.hpp)

namespace SCTL_NAMESPACE {

  template <class Real> class ChebQuadRule { // p(x)
    public:
      static const Vector<Real>& nds(Integer ChebOrder) {
        if (ChebOrder > 50) SCTL_ASSERT(false);
        if (ChebOrder ==  1) return nds< 1>();
        if (ChebOrder ==  2) return nds< 2>();
        if (ChebOrder ==  3) return nds< 3>();
        if (ChebOrder ==  4) return nds< 4>();
        if (ChebOrder ==  5) return nds< 5>();
        if (ChebOrder ==  6) return nds< 6>();
        if (ChebOrder ==  7) return nds< 7>();
        if (ChebOrder ==  8) return nds< 8>();
        if (ChebOrder ==  9) return nds< 9>();
        if (ChebOrder == 10) return nds<10>();
        if (ChebOrder == 11) return nds<11>();
        if (ChebOrder == 12) return nds<12>();
        if (ChebOrder == 13) return nds<13>();
        if (ChebOrder == 14) return nds<14>();
        if (ChebOrder == 15) return nds<15>();
        if (ChebOrder == 16) return nds<16>();
        if (ChebOrder == 17) return nds<17>();
        if (ChebOrder == 18) return nds<18>();
        if (ChebOrder == 19) return nds<19>();
        if (ChebOrder == 20) return nds<20>();
        if (ChebOrder == 21) return nds<21>();
        if (ChebOrder == 22) return nds<22>();
        if (ChebOrder == 23) return nds<23>();
        if (ChebOrder == 24) return nds<24>();
        if (ChebOrder == 25) return nds<25>();
        if (ChebOrder == 26) return nds<26>();
        if (ChebOrder == 27) return nds<27>();
        if (ChebOrder == 28) return nds<28>();
        if (ChebOrder == 29) return nds<29>();
        if (ChebOrder == 30) return nds<30>();
        if (ChebOrder == 31) return nds<31>();
        if (ChebOrder == 32) return nds<32>();
        if (ChebOrder == 33) return nds<33>();
        if (ChebOrder == 34) return nds<34>();
        if (ChebOrder == 35) return nds<35>();
        if (ChebOrder == 36) return nds<36>();
        if (ChebOrder == 37) return nds<37>();
        if (ChebOrder == 38) return nds<38>();
        if (ChebOrder == 39) return nds<39>();
        if (ChebOrder == 40) return nds<40>();
        if (ChebOrder == 41) return nds<41>();
        if (ChebOrder == 42) return nds<42>();
        if (ChebOrder == 43) return nds<43>();
        if (ChebOrder == 44) return nds<44>();
        if (ChebOrder == 45) return nds<45>();
        if (ChebOrder == 46) return nds<46>();
        if (ChebOrder == 47) return nds<47>();
        if (ChebOrder == 48) return nds<48>();
        if (ChebOrder == 49) return nds<49>();
        return nds<50>();
      }
      static const Vector<Real>& wts(Integer ChebOrder) {
        if (ChebOrder > 50) SCTL_ASSERT(false);
        if (ChebOrder ==  1) return wts< 1>();
        if (ChebOrder ==  2) return wts< 2>();
        if (ChebOrder ==  3) return wts< 3>();
        if (ChebOrder ==  4) return wts< 4>();
        if (ChebOrder ==  5) return wts< 5>();
        if (ChebOrder ==  6) return wts< 6>();
        if (ChebOrder ==  7) return wts< 7>();
        if (ChebOrder ==  8) return wts< 8>();
        if (ChebOrder ==  9) return wts< 9>();
        if (ChebOrder == 10) return wts<10>();
        if (ChebOrder == 11) return wts<11>();
        if (ChebOrder == 12) return wts<12>();
        if (ChebOrder == 13) return wts<13>();
        if (ChebOrder == 14) return wts<14>();
        if (ChebOrder == 15) return wts<15>();
        if (ChebOrder == 16) return wts<16>();
        if (ChebOrder == 17) return wts<17>();
        if (ChebOrder == 18) return wts<18>();
        if (ChebOrder == 19) return wts<19>();
        if (ChebOrder == 20) return wts<20>();
        if (ChebOrder == 21) return wts<21>();
        if (ChebOrder == 22) return wts<22>();
        if (ChebOrder == 23) return wts<23>();
        if (ChebOrder == 24) return wts<24>();
        if (ChebOrder == 25) return wts<25>();
        if (ChebOrder == 26) return wts<26>();
        if (ChebOrder == 27) return wts<27>();
        if (ChebOrder == 28) return wts<28>();
        if (ChebOrder == 29) return wts<29>();
        if (ChebOrder == 30) return wts<30>();
        if (ChebOrder == 31) return wts<31>();
        if (ChebOrder == 32) return wts<32>();
        if (ChebOrder == 33) return wts<33>();
        if (ChebOrder == 34) return wts<34>();
        if (ChebOrder == 35) return wts<35>();
        if (ChebOrder == 36) return wts<36>();
        if (ChebOrder == 37) return wts<37>();
        if (ChebOrder == 38) return wts<38>();
        if (ChebOrder == 39) return wts<39>();
        if (ChebOrder == 40) return wts<40>();
        if (ChebOrder == 41) return wts<41>();
        if (ChebOrder == 42) return wts<42>();
        if (ChebOrder == 43) return wts<43>();
        if (ChebOrder == 44) return wts<44>();
        if (ChebOrder == 45) return wts<45>();
        if (ChebOrder == 46) return wts<46>();
        if (ChebOrder == 47) return wts<47>();
        if (ChebOrder == 48) return wts<48>();
        if (ChebOrder == 49) return wts<49>();
        return wts<50>();
      }
      template <Integer ChebOrder> static const Vector<Real>& nds() {
        static Vector<Real> nds = get_cheb_nds(ChebOrder);
        return nds;
      }
      template <Integer ChebOrder> static const Vector<Real>& wts() {
        static const Vector<Real> wts = get_cheb_wts(ChebOrder);
        return wts;
      }
    private:
      static Vector<Real> get_cheb_nds(Integer ChebOrder){
        Vector<Real> nds(ChebOrder);
        for (Long i = 0; i < ChebOrder; i++) {
          nds[i] = 0.5 - cos<Real>((2*i+1)*const_pi<Real>()/(2*ChebOrder)) * 0.5;
        }
        return nds;
      }
      static Vector<Real> get_cheb_wts(Integer ChebOrder){
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
