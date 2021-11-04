#ifndef _SCTL_ODE_SOLVER_
#define _SCTL_ODE_SOLVER_

#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(common.hpp)

#include <functional>

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;
template <class ValueType> class Matrix;

template <class Real, Integer ORDER> class SDC {
  public:

    using Fn = std::function<void(Vector<Real>*, const Vector<Real>&)>;

    SDC() {
      Vector<Real> x_cheb(ORDER);
      for (Long i = 0; i < ORDER; i++) {
        x_cheb[i] = 0.5 - 0.5 * cos(const_pi<Real>() * i / (ORDER - 1));
      }

      Matrix<Real> Mp(ORDER, ORDER);
      Matrix<Real> Mi(ORDER, ORDER);
      for (Long i = 0; i < ORDER; i++) {
        for (Long j = 0; j < ORDER; j++) {
          Mp[j][i] = pow<Real>(x_cheb[i],j);
          Mi[j][i] = pow<Real>(x_cheb[i],j+1) / (j+1);
        }
      }
      M_time_step = (Mp.pinv() * Mi).Transpose(); // TODO: replace Mp.pinv()

      Mp.ReInit(ORDER,ORDER); Mp = 0;
      Mi.ReInit(ORDER,ORDER); Mi = 0;
      Integer TRUNC_ORDER = ORDER;
      if (ORDER >= 2) TRUNC_ORDER = ORDER - 1;
      if (ORDER >= 6) TRUNC_ORDER = ORDER - 1;
      if (ORDER >= 9) TRUNC_ORDER = ORDER - 1;
      for (Long j = 0; j < TRUNC_ORDER; j++) {
        for (Long i = 0; i < ORDER; i++) {
          Mp[j][i] = pow<Real>(x_cheb[i],j);
          Mi[j][i] = pow<Real>(x_cheb[i],j);
        }
      }
      M_error = (Mp.pinv() * Mi).Transpose(); // TODO: replace Mp.pinv()
      for (Long i = 0; i < ORDER; i++) M_error[i][i] -= 1;
    }

    // solve u = \int_0^{dt} F(u)
    void operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0_, const Fn& F, Integer N_picard = ORDER, Real tol_picard = 0, Real* error_interp = nullptr, Real* error_picard = nullptr) {
      auto max_norm = [] (const Matrix<Real>& M) {
        Real max_val = 0;
        for (Long i = 0; i < M.Dim(0); i++) {
          for (Long j = 0; j < M.Dim(1); j++) {
            max_val = std::max<Real>(max_val, fabs(M[i][j]));
          }
        }
        return max_val;
      };

      const Long DOF = u0_.Dim();
      Matrix<Real> Mu0(ORDER, DOF);
      Matrix<Real> Mu1(ORDER, DOF);
      for (Long j = 0; j < ORDER; j++) { // Set u0
        for (Long k = 0; k < DOF; k++) {
          Mu0[j][k] = u0_[k];
        }
      }

      Matrix<Real> M_dudt(ORDER, DOF);
      { // Set M_dudt
        Vector<Real> dudt_(DOF, M_dudt[0], false);
        F(&dudt_, Vector<Real>(DOF, Mu0[0], false));
        for (Long i = 1; i < ORDER; i++) {
          for (Long j = 0; j < DOF; j++) {
            M_dudt[i][j] = M_dudt[0][j];
          }
        }
      }
      Mu1 = Mu0 + (M_time_step * M_dudt) * dt;

      Matrix<Real> Merr(ORDER, DOF);
      for (Long k = 0; k < N_picard; k++) { // Picard iteration
        auto Mu_previous = Mu1;
        for (Long i = 1; i < ORDER; i++) { // Set M_dudt
          Vector<Real> dudt_(DOF, M_dudt[i], false);
          F(&dudt_, Vector<Real>(DOF, Mu1[i], false));
        }
        Mu1 = Mu0 + (M_time_step * M_dudt) * dt;
        Merr = Mu1 - Mu_previous;
        if (max_norm(Merr) < tol_picard) break;
      }

      if (u->Dim() != DOF) u->ReInit(DOF);
      for (Long k = 0; k < DOF; k++) { // Set u
        u[0][k] = Mu1[ORDER - 1][k];
      }
      if (error_picard != nullptr) {
        error_picard[0] = max_norm(Merr);
      }
      if (error_interp != nullptr) {
        Merr = M_error * Mu1;
        error_interp[0] = max_norm(Merr);
      }
    }

    static void test() {
      auto ref_sol = [](Real t) { return exp<Real>(-t); };
      auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
        dudt[0] = u * -1.0;
      };
      std::function<void(Vector<Real>*, const Vector<Real>&)> F(fn);

      SDC<Real, ORDER> ode_solver;
      Real t = 0.0, dt = 1.0e-1;
      Vector<Real> u, u0(1);
      u0 = 1.0;
      while (t < 10.0) {
        Real error_interp, error_picard;
        ode_solver(&u, dt, u0, F, ORDER, 0.0, &error_interp, &error_picard);
        { // Accept solution
          u0 = u;
          t = t + dt;
        }

        printf("t = %e;  ", t);
        printf("u1 = %e;  ", u0[0]);
        printf("error = %e;  ", ref_sol(t) - u0[0]);
        printf("time_step_error_estimate = %e;  \n", std::max(error_interp, error_picard));
      }
    }

  private:
    Matrix<Real> M_time_step, M_error;
};

}

#endif  //_SCTL_ODE_SOLVER_
