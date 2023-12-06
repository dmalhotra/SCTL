#include "sctl/ode-solver.hpp"

#include SCTL_INCLUDE(lagrange-interp.hpp)

namespace SCTL_NAMESPACE {

  template <class Real> SDC<Real>::SDC(const Integer Order_, const Comm& comm_) : Order(Order_), comm(comm_) {
    SCTL_ASSERT(Order >= 2); // TODO: use explicit Euler if Order == 1

    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif

    auto second_kind_cheb_nds = [](const Integer Order) {
      Vector<ValueType> x_cheb(Order);
      for (Long i = 0; i < Order; i++) {
        x_cheb[i] = 0.5 - 0.5 * cos(const_pi<ValueType>() * i / (Order - 1));
      }
      return x_cheb;
    };
    const auto nds0 = second_kind_cheb_nds(Order); // TODO: use Gauss-Lobatto nodes
    SCTL_ASSERT(nds0.Dim() == Order);

    { // Set M_error
      Integer TRUNC_Order = Order;
      if (Order >= 2) TRUNC_Order = Order - 1;
      if (Order >= 6) TRUNC_Order = Order - 1;
      if (Order >= 9) TRUNC_Order = Order - 1;

      const auto nds1 = second_kind_cheb_nds(TRUNC_Order);
      SCTL_ASSERT(nds1.Dim() == TRUNC_Order);

      Matrix<ValueType> Minterp0(Order, TRUNC_Order);
      Matrix<ValueType> Minterp1(TRUNC_Order, Order);
      Vector<ValueType> interp0(Order*TRUNC_Order, Minterp0.begin(), false);
      Vector<ValueType> interp1(TRUNC_Order*Order, Minterp1.begin(), false);
      LagrangeInterp<ValueType>::Interpolate(interp0, nds0, nds1);
      LagrangeInterp<ValueType>::Interpolate(interp1, nds1, nds0);
      Matrix<ValueType> M_error_ = (Minterp0 * Minterp1).Transpose();
      for (Long i = 0; i < Order; i++) M_error_[i][i] -= 1;

      M_error.ReInit(Order, Order);
      for (Long i = 0; i < Order*Order; i++) M_error[0][i] = (Real)M_error_[0][i];
    }
    { // Set M_time_step
      const auto qx = ChebQuadRule<ValueType>::ComputeNds(Order);
      const auto qw = ChebQuadRule<ValueType>::ComputeWts(Order);
      const Matrix<ValueType> Mw(Order, 1, (Iterator<ValueType>)qw.begin(), false);
      SCTL_ASSERT(qw.Dim() == Order);
      SCTL_ASSERT(qx.Dim() == Order);

      Matrix<ValueType> Minterp(Order, Order), M_time_step_(Order, Order);
      Vector<ValueType> interp(Order*Order, Minterp.begin(), false);
      for (Integer i = 0; i < Order; i++) {
        LagrangeInterp<ValueType>::Interpolate(interp, nds0, qx*nds0[i]);
        Matrix<ValueType> M_time_step_i(Order,1, M_time_step_[i], false);
        M_time_step_i = Minterp * Mw * nds0[i];
      }

      M_time_step.ReInit(Order, Order);
      for (Long i = 0; i < Order*Order; i++) M_time_step[0][i] = (Real)M_time_step_[0][i];
    }
    { // Set nds
      nds.ReInit(Order);
      for (Long i = 0; i < Order; i++) {
        nds[i] = (Real)nds0[i];
      }
    }
  }

  // solve u = u0 + \int_0^{dt} F(u)
  template <class Real> template <class Fn> void SDC<Real>::operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, Fn&& F, Integer N_picard, const Real tol_picard, Real* error_interp, Real* error_picard, Real* norm_dudt) const {
    auto max_norm = [] (const Matrix<Real>& M, const Comm& comm) {
      StaticArray<Real,2> max_val{0,0};
      for (Long i = 0; i < M.Dim(0); i++) {
        for (Long j = 0; j < M.Dim(1); j++) {
          max_val[0] = std::max<Real>(max_val[0], fabs(M[i][j]));
        }
      }
      comm.Allreduce((ConstIterator<Real>)max_val, (Iterator<Real>)max_val+1, 1, Comm::CommOp::MAX);
      return max_val[1];
    };
    if (N_picard < 0) N_picard = Order;
    const Long DOF = u0.Dim();

    const Integer Nbuff = 1000;
    StaticArray<Real,Nbuff> buff;
    Matrix<Real> Mu;
    Matrix<Real> Mf0, Mf1;
    Matrix<Real> Mv, Mv_change;
    if (Nbuff < 5*Order*DOF) {
      Mu.ReInit(Order, DOF);
      Mf0.ReInit(Order, DOF);
      Mf1.ReInit(Order, DOF);
      Mv.ReInit(Order, DOF);
      Mv_change.ReInit(Order, DOF);
    } else {
      Mu.ReInit(Order, DOF, buff + 0*Order*DOF, false);
      Mf0.ReInit(Order, DOF, buff + 1*Order*DOF, false);
      Mf1.ReInit(Order, DOF, buff + 2*Order*DOF, false);
      Mv.ReInit(Order, DOF, buff + 3*Order*DOF, false);
      Mv_change.ReInit(Order, DOF, buff + 4*Order*DOF, false);
    }

    for (Long j = 0; j < Order; j++) { // Set Mu
      for (Long k = 0; k < DOF; k++) {
        Mu[j][k] = u0[k];
      }
    }
    { // Set Mf0
      Vector<Real> f_(DOF, Mf1[0], false);
      F(&f_, Vector<Real>(DOF, Mu[0], false));
      for (Long i = 0; i < Order; i++) {
        for (Long j = 0; j < DOF; j++) {
          Mf0[i][j] = Mf1[0][j];
        }
      }
    }

    Mv = 0;
    Real picard_err_curr = 0;
    for (Long k = 0; k < N_picard; k++) { // Picard iteration
      Mv_change = Mv;
      Matrix<Real>::GEMM(Mv, M_time_step, Mf0);
      Mv_change -= Mv;
      picard_err_curr = max_norm(Mv_change, comm) * dt;
      if (picard_err_curr < tol_picard) {
        for (Long i = 1; i < Order; i++) {
          const Vector<Real> v_1(DOF, Mv[i], false);
          Vector<Real> u_1(DOF, Mu[i], false);
          for (Long j = 0; j < DOF; j++) {
            u_1[j] = u0[j] + v_1[j] * dt;
          }
        }
        break;
      }

      for (Long i = 1; i < Order; i++) {
        const Vector<Real> f0_0(DOF, Mf0[i-1], false);
        const Vector<Real> f1_0(DOF, Mf1[i-1], false);
        Vector<Real> v_1(DOF, Mv[i], false);
        Vector<Real> u_1(DOF, Mu[i], false);
        Vector<Real> f1_1(DOF, Mf1[i], false);

        for (Long j = 0; j < DOF; j++) {
          v_1[j] += (f1_0[j] - f0_0[j]) * (nds[i]-nds[i-1]); // residual time-stepping
          u_1[j] = u0[j] + v_1[j] * dt;
        }
        F(&f1_1, u_1);
      }
      Mf0 = Mf1;
    }

    if (u->Dim() != DOF) u->ReInit(DOF);
    for (Long k = 0; k < DOF; k++) { // Set u
      (*u)[k] = Mu[Order - 1][k];
    }
    if (error_picard != nullptr) {
      (*error_picard) = picard_err_curr;
    }
    if (error_interp != nullptr) {
      Matrix<Real>& err = Mv_change; // reuse memory
      Matrix<Real>::GEMM(err, M_error, Mv);
      (*error_interp) = max_norm(err, comm) * dt;
    }
    if (norm_dudt != nullptr) {
      (*norm_dudt) = max_norm(Mv, comm) * dt;
    }
  }

  template <class Real> template <class Fn> Real SDC<Real>::AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, Fn&& F, const Real tol, const MonitorFn* monitor_callback, bool continue_with_errors, Real* error) const {
    const Real eps = machine_eps<Real>();
    Vector<Real> u_, u0_ = u0;

    Real t = 0;
    Real error_ = 0;
    while (t < T && dt > eps*T) {
      Real error_interp, error_picard, norm_dudt;
      (*this)(&u_, dt, u0_, std::forward<Fn>(F), 2*Order, tol*dt*pow<Real>(0.9,Order), &error_interp, &error_picard, &norm_dudt);

      Real tol_ = std::max<Real>(tol/T, (tol-error_)/(T-t));
      Real max_err = std::max<Real>(error_interp, error_picard);
      //std::cout<<t<<' '<<dt<<' '<<error_interp/dt<<' '<<error_picard/dt<<' '<<max_err/norm_dudt/eps<<'\n';
      if (max_err < tol_*dt || (continue_with_errors && max_err/norm_dudt < 2*eps)) { // Accept solution
        u0_.Swap(u_);
        t = t + dt;
        error_ += max_err;
        if (monitor_callback) (*monitor_callback)(t, dt, u0_);
      }

      if (continue_with_errors && max_err/norm_dudt < 2*eps) {
        dt = std::min<Real>(T-t, 1.1*dt);
      } else {
        // Adjust time-step size (Quaife, Biros - JCP 2016)
        dt = std::min<Real>(T-t, std::max<Real>(0.5*dt, 0.9*dt*pow<Real>((tol_*dt)/max_err, 1/(Real)(Order))));
      }
    }
    if (t < T || error_ > tol) SCTL_WARN("Could not solve ODE to the requested tolerance.");
    if (error != nullptr) (*error) = error_;

    (*u) = u0_;
    return t;
  }

}

