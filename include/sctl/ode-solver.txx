#include SCTL_INCLUDE(lagrange-interp.hpp)

namespace SCTL_NAMESPACE {

  template <class Real> SDC<Real>::SDC(const Integer Order_, const Comm& comm_) : Order(Order_), comm(comm_) {
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
    const auto nds0 = second_kind_cheb_nds(Order);
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
  }

  // solve u = u0 + \int_0^{dt} F(u)
  template <class Real> void SDC<Real>::operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, const Fn& F, Integer N_picard, const Real tol_picard, Real* error_interp, Real* error_picard, Real* norm_dudt) const {
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
    Matrix<Real> Mu0(Order, DOF);
    Matrix<Real> Mu1(Order, DOF);
    for (Long j = 0; j < Order; j++) { // Set Mu0
      for (Long k = 0; k < DOF; k++) {
        Mu0[j][k] = u0[k];
      }
    }

    Matrix<Real> M_dudt(Order, DOF);
    { // Set M_dudt
      Vector<Real> dudt_(DOF, M_dudt[0], false);
      F(&dudt_, Vector<Real>(DOF, Mu0[0], false));
      for (Long i = 1; i < Order; i++) {
        for (Long j = 0; j < DOF; j++) {
          M_dudt[i][j] = M_dudt[0][j];
        }
      }
    }
    Matrix<Real> Mv = (M_time_step * M_dudt) * dt;
    Mu1 = Mu0 + Mv;

    Real picard_err_curr = 0;
    for (Long k = 0; k < N_picard; k++) { // Picard iteration
      auto Mv_previous = Mv;
      for (Long i = 1; i < Order; i++) { // Set M_dudt
        Vector<Real> dudt_(DOF, M_dudt[i], false);
        F(&dudt_, Vector<Real>(DOF, Mu1[i], false));
      }
      Mv = (M_time_step * M_dudt) * dt;
      Mu1 = Mu0 + Mv;

      picard_err_curr = max_norm(Mv - Mv_previous, comm);
      if (picard_err_curr < tol_picard) break;
    }

    if (u->Dim() != DOF) u->ReInit(DOF);
    for (Long k = 0; k < DOF; k++) { // Set u
      (*u)[k] = Mu1[Order - 1][k];
    }
    if (error_picard != nullptr) {
      (*error_picard) = picard_err_curr;
    }
    if (error_interp != nullptr) {
      (*error_interp) = max_norm(M_error * Mv, comm);
    }
    if (norm_dudt != nullptr) {
      (*norm_dudt) = max_norm(Mv, comm);
    }
  }

  template <class Real> Real SDC<Real>::AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn& F, const Real tol, const MonitorFn* monitor_callback, bool continue_with_errors, Real* error) const {
    const Real eps = machine_eps<Real>();
    Vector<Real> u_, u0_ = u0;

    Real t = 0;
    Real error_ = 0;
    while (t < T && dt > eps*T) {
      Real error_interp, error_picard, norm_dudt;
      (*this)(&u_, dt, u0_, F, 2*Order, tol*dt*pow<Real>(0.9,Order), &error_interp, &error_picard, &norm_dudt);

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

