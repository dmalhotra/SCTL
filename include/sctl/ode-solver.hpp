#ifndef _SCTL_ODE_SOLVER_
#define _SCTL_ODE_SOLVER_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(matrix.hpp)

#include <functional>

namespace SCTL_NAMESPACE {

// template <class ValueType> class Vector;
// template <class ValueType> class Matrix;

template <class Real> class SDC {
  public:

    // using Fn = std::function<void(Vector<Real>* dudt, const Vector<Real>& u)>;
    using MonitorFn = std::function<void(Real t, Real dt, const Vector<Real>& u)>;

    /**
     * Constructor
     *
     * @param[in] Order the order of the method.
     */
    explicit SDC(const Integer Order, const Comm& comm = Comm::Self());

    /**
     * Apply one step of spectral deferred correction (SDC).
     * Compute: u = u0 + \int_0^{dt} F(u)
     *
     * @param[out] u the solution
     * @param[in] dt the step size
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] N_picard the maximum number of picard iterations
     * @param[in] tol_picard the tolerance for stopping picard iterations
     * @param[out] error_interp an estimate of the truncation error of the solution interpolant
     * @param[out] error_picard the picard iteration error
     * @param[out] norm_dudt maximum norm of du/dt
     */
    template <class Fn> void operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, Fn&& F, Integer N_picard = -1, const Real tol_picard = 0, Real* error_interp = nullptr, Real* error_picard = nullptr, Real* norm_dudt = nullptr) const;

    /**
     * Solve ODE adaptively to required tolerance.
     * Compute: u = u0 + \int_0^{T} F(u)
     *
     * @param[out] u the final solution
     * @param[in] dt the initial step size guess
     * @param[in] T the final time
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] tol the required solution tolerance
     * @param[in] monitor_callback a callback function called after each accepted time-step
     * @param[in] continue_with_errors tries to compute the best solution even if the required tolerance cannot be satisfied.
     * @param[out] error estimate of the final output error
     *
     * @return the final time (should equal T if no errors)
     */
    template <class Fn> Real AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, Fn&& F, Real tol, const MonitorFn* monitor_callback = nullptr, bool continue_with_errors = false, Real* error = nullptr) const;

    static void test_one_step(const Integer Order = 5) {
      auto ref_sol = [](Real t) { return cos<Real>(-t); };
      auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
        (*dudt)[0] = -u[1];
        (*dudt)[1] = u[0];
      };
      std::function<void(Vector<Real>*, const Vector<Real>&)> F(fn);

      const SDC<Real> ode_solver(Order);
      Real t = 0.0, dt = 1.0e-1;
      Vector<Real> u, u0(2);
      u0[0] = 1.0;
      u0[1] = 0.0;
      while (t < 10.0) {
        Real error_interp, error_picard;
        ode_solver(&u, dt, u0, F, -1, 0.0, &error_interp, &error_picard);
        { // Accept solution
          u0 = u;
          t = t + dt;
        }

        printf("t = %e;  ", t);
        printf("u = %e;  ", u0[0]);
        printf("error = %e;  ", ref_sol(t) - u0[0]);
        printf("time_step_error_estimate = %e;  \n", std::max(error_interp, error_picard));
      }
    }

    static void test_adaptive_solve(const Integer Order = 5, const Real tol = 1e-5) {
      auto ref_sol = [](Real t) { return cos(-t); };
      auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
        (*dudt)[0] = -u[1];
        (*dudt)[1] = u[0];
      };
      std::function<void(Vector<Real>*, const Vector<Real>&)> F(fn);

      Vector<Real> u, u0(2);
      u0[0] = 1.0; u0[1] = 0.0;
      Real T = 10.0, dt = 1.0e-1;

      SDC<Real> ode_solver(Order);
      Real t = ode_solver.AdaptiveSolve(&u, dt, T, u0, F, tol);

      if (t == T) {
        printf("u = %e;  ", u[0]);
        printf("error = %e;  \n", ref_sol(T) - u[0]);
      }
    }

  private:
    Matrix<Real> M_time_step, M_error;
    Vector<Real> nds;
    Integer Order;
    Comm comm;
};

}

#endif  //_SCTL_ODE_SOLVER_
