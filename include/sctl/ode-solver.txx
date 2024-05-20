#ifndef _SCTL_ODE_SOLVER_TXX_
#define _SCTL_ODE_SOLVER_TXX_

#include <stdio.h>                   // for printf
#include <algorithm>                 // for max, min
#include <utility>                   // for forward

#include "sctl/common.hpp"           // for Long, Integer, SCTL_ASSERT, SCTL...
#include SCTL_INCLUDE(ode-solver.hpp)       // for SDC
#include SCTL_INCLUDE(comm.hpp)             // for Comm (ptr only), CommOp
#include SCTL_INCLUDE(comm.txx)             // for Comm::Allreduce, Comm::Comm
#include SCTL_INCLUDE(iterator.hpp)         // for Iterator, ConstIterator
#include SCTL_INCLUDE(iterator.txx)         // for Iterator::operator[]
#include SCTL_INCLUDE(lagrange-interp.hpp)  // for LagrangeInterp
#include SCTL_INCLUDE(lagrange-interp.txx)  // for LagrangeInterp::Interpolate
#include SCTL_INCLUDE(math_utils.hpp)       // for QuadReal, cos, operator*, operator-
#include SCTL_INCLUDE(math_utils.txx)       // for const_pi, cos, pow, machine_eps
#include SCTL_INCLUDE(matrix.hpp)           // for Matrix
#include SCTL_INCLUDE(matrix.txx)           // for Matrix::operator[], Matrix::Matr...
#include SCTL_INCLUDE(quadrule.hpp)         // for ChebQuadRule
#include SCTL_INCLUDE(quadrule.txx)         // for ChebQuadRule::ComputeNdsWts
#include SCTL_INCLUDE(static-array.hpp)     // for StaticArray
#include SCTL_INCLUDE(vector.hpp)           // for Vector
#include SCTL_INCLUDE(vector.txx)           // for Vector::Vector<ValueType>, Vecto...

namespace SCTL_NAMESPACE {

  template <class Real> void SDC<Real>::test_one_step(const Integer Order) {
    auto ref_sol = [](Real t) { return cos<Real>(-t); };
    auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
      (*dudt)[0] = -u[1];
      (*dudt)[1] = u[0];
    };

    const SDC<Real> ode_solver(Order);
    Real t = 0.0, dt = 1.0e-1;
    Vector<Real> u, u0(2);
    u0[0] = 1.0;
    u0[1] = 0.0;
    while (t < 10.0) {
      Real error_interp, error_picard;
      ode_solver(&u, dt, u0, fn, -1, 0.0, &error_interp, &error_picard);
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

  template <class Real> void SDC<Real>::test_adaptive_solve(const Integer Order, const Real tol) {
    auto ref_sol = [](Real t) { return cos(-t); };
    auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
      (*dudt)[0] = -u[1];
      (*dudt)[1] = u[0];
    };

    Vector<Real> u, u0(2);
    u0[0] = 1.0; u0[1] = 0.0;
    Real T = 10.0, dt = 1.0e-1;

    SDC<Real> ode_solver(Order);
    Real t = ode_solver.AdaptiveSolve(&u, dt, T, u0, fn, tol);

    if (t == T) {
      printf("u = %e;  ", u[0]);
      printf("error = %e;  \n", ref_sol(T) - u[0]);
    }
  }

  template <class Real> SDC<Real>::SDC(const Integer Order_, const Comm& comm_) : order(Order_), comm(comm_) {
    SCTL_ASSERT(order >= 2); // TODO: use explicit Euler if order == 1

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
    const auto nds0 = second_kind_cheb_nds(order); // TODO: use Gauss-Lobatto nodes
    SCTL_ASSERT(nds0.Dim() == order);

    { // Set M_error
      Integer TRUNC_Order = order;
      if (order >= 2) TRUNC_Order = order - 1;
      if (order >= 6) TRUNC_Order = order - 1;
      if (order >= 9) TRUNC_Order = order - 1;

      const auto nds1 = second_kind_cheb_nds(TRUNC_Order);
      SCTL_ASSERT(nds1.Dim() == TRUNC_Order);

      Matrix<ValueType> Minterp0(order, TRUNC_Order);
      Matrix<ValueType> Minterp1(TRUNC_Order, order);
      Vector<ValueType> interp0(order*TRUNC_Order, Minterp0.begin(), false);
      Vector<ValueType> interp1(TRUNC_Order*order, Minterp1.begin(), false);
      LagrangeInterp<ValueType>::Interpolate(interp0, nds0, nds1);
      LagrangeInterp<ValueType>::Interpolate(interp1, nds1, nds0);
      Matrix<ValueType> M_error_ = (Minterp0 * Minterp1).Transpose();
      for (Long i = 0; i < order; i++) M_error_[i][i] -= 1;

      M_error.ReInit(order, order);
      for (Long i = 0; i < order*order; i++) M_error[0][i] = (Real)M_error_[0][i];
    }
    { // Set M_time_step
      Vector<ValueType> qx, qw;
      ChebQuadRule<ValueType>::ComputeNdsWts(&qx, &qw, order);
      const Matrix<ValueType> Mw(order, 1, (Iterator<ValueType>)qw.begin(), false);
      SCTL_ASSERT(qw.Dim() == order);
      SCTL_ASSERT(qx.Dim() == order);

      Matrix<ValueType> Minterp(order, order), M_time_step_(order, order);
      Vector<ValueType> interp(order*order, Minterp.begin(), false);
      for (Integer i = 0; i < order; i++) {
        LagrangeInterp<ValueType>::Interpolate(interp, nds0, qx*nds0[i]);
        Matrix<ValueType> M_time_step_i(order,1, M_time_step_[i], false);
        M_time_step_i = Minterp * Mw * nds0[i];
      }

      M_time_step.ReInit(order, order);
      for (Long i = 0; i < order*order; i++) M_time_step[0][i] = (Real)M_time_step_[0][i];
    }
    { // Set nds
      nds.ReInit(order);
      for (Long i = 0; i < order; i++) {
        nds[i] = (Real)nds0[i];
      }
    }
  }

  template <class Real> Integer SDC<Real>::Order() const { return order; }

  // solve u = u0 + \int_0^{dt} F(u)
  template <class Real> void SDC<Real>::operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, Fn0&& F, Integer N_picard, const Real tol_picard, Real* error_interp, Real* error_picard, Real* norm_dudt) const {
    auto max_norm = [] (const Matrix<Real>& M, const Comm& comm) {
      StaticArray<Real,2> max_val{0,0};
      for (Long i = 0; i < M.Dim(0); i++) {
        for (Long j = 0; j < M.Dim(1); j++) {
          max_val[0] = std::max<Real>(max_val[0], fabs(M[i][j]));
        }
      }
      comm.Allreduce((ConstIterator<Real>)max_val, (Iterator<Real>)max_val+1, 1, CommOp::MAX);
      return max_val[1];
    };
    if (N_picard < 0) N_picard = order;
    const Long DOF = u0.Dim();

    const Integer Nbuff = 1000;
    StaticArray<Real,Nbuff> buff;
    Matrix<Real> Mu;
    Matrix<Real> Mf0, Mf1;
    Matrix<Real> Mv, Mv_change;
    Vector<Real> picard_err;
    if (Nbuff < 1*order*DOF) Mu.ReInit(order, DOF);
    else Mu.ReInit(order, DOF, buff + 0*order*DOF, false);
    if (Nbuff < 2*order*DOF) Mf0.ReInit(order, DOF);
    else Mf0.ReInit(order, DOF, buff + 1*order*DOF, false);
    if (Nbuff < 3*order*DOF) Mf1.ReInit(order, DOF);
    else Mf1.ReInit(order, DOF, buff + 2*order*DOF, false);
    if (Nbuff < 4*order*DOF) Mv.ReInit(order, DOF);
    else Mv.ReInit(order, DOF, buff + 3*order*DOF, false);
    if (Nbuff < 5*order*DOF) Mv_change.ReInit(order, DOF);
    else Mv_change.ReInit(order, DOF, buff + 4*order*DOF, false);
    if (Nbuff < 5*order*DOF+N_picard) picard_err.ReInit(N_picard);
    else picard_err.ReInit(N_picard, buff + 5*order*DOF, false);

    for (Long j = 0; j < order; j++) { // Set Mu
      for (Long k = 0; k < DOF; k++) {
        Mu[j][k] = u0[k];
      }
    }
    { // Set Mf0
      Vector<Real> f_(DOF, Mf1[0], false);
      F(&f_, Vector<Real>(DOF, Mu[0], false), 0, 0);
      if (!f_.Dim()) { // abort
        u->ReInit(0);
        if (error_interp) (*error_interp) = 1;
        if (error_picard) (*error_picard) = 1;
        if (norm_dudt) (*norm_dudt) = 1;
        return;
      }
      for (Long i = 0; i < order; i++) {
        for (Long j = 0; j < DOF; j++) {
          Mf0[i][j] = Mf1[0][j];
        }
      }
    }

    Mv = 0;
    Long picard_iter = 0;
    for (picard_iter = 0; picard_iter < N_picard; picard_iter++) { // Picard iteration
      Mv_change = Mv;
      Matrix<Real>::GEMM(Mv, M_time_step, Mf0);
      Mv_change -= Mv;
      picard_err[picard_iter] = max_norm(Mv_change, comm) * dt;
      if (picard_err[picard_iter] < tol_picard || (picard_iter>1 && picard_err[picard_iter] > picard_err[picard_iter-2])) {
        for (Long i = 1; i < order; i++) {
          const Vector<Real> v_1(DOF, Mv[i], false);
          Vector<Real> u_1(DOF, Mu[i], false);
          for (Long j = 0; j < DOF; j++) {
            u_1[j] = u0[j] + v_1[j] * dt;
          }
        }
        break;
      }

      for (Long i = 1; i < order; i++) {
        const Vector<Real> f0_0(DOF, Mf0[i-1], false);
        const Vector<Real> f1_0(DOF, Mf1[i-1], false);
        Vector<Real> v_1(DOF, Mv[i], false);
        Vector<Real> u_1(DOF, Mu[i], false);
        Vector<Real> f1_1(DOF, Mf1[i], false);

        for (Long j = 0; j < DOF; j++) {
          v_1[j] += (f1_0[j] - f0_0[j]) * (nds[i]-nds[i-1]); // residual time-stepping
          u_1[j] = u0[j] + v_1[j] * dt;
        }

        F(&f1_1, u_1, picard_iter, i);
        if (!f1_1.Dim()) { // abort
          u->ReInit(0);
          if (error_interp) (*error_interp) = 1;
          if (error_picard) (*error_picard) = 1;
          if (norm_dudt) (*norm_dudt) = 1;
          return;
        }
      }
      Mf0 = Mf1;
    }

    if (u->Dim() != DOF) u->ReInit(DOF);
    for (Long k = 0; k < DOF; k++) { // Set u
      (*u)[k] = Mu[order - 1][k];
    }
    if (error_picard != nullptr) {
      (*error_picard) = picard_err[std::min<Long>(picard_iter, N_picard-1)];
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

  template <class Real> Real SDC<Real>::AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, Fn0&& F, const Real tol, const MonitorFn* monitor_callback, bool continue_with_errors, Real* error) const {
    const Real eps = machine_eps<Real>();
    Vector<Real> u_, u0_ = u0;

    Real t = 0;
    Real error_ = 0;
    while (t < T && dt > eps*T) {
      Real error_interp, error_picard, norm_dudt;
      Real tol_ = std::max<Real>(tol/T, (tol-error_)/(T-t));
      (*this)(&u_, dt, u0_, std::forward<Fn0>(F), 2*order, tol_*dt*pow<Real>(0.8,order), &error_interp, &error_picard, &norm_dudt);
      // The factor pow<Real>(0.8,order) ensures that we try to solve to
      // slightly higher accuracy. This allows us to determine if the next
      // time-step size can be increased by d 1/0.8.

      Real max_err = std::max<Real>(error_interp, error_picard);
      //std::cout<<"Adaptive time-step: "<<t<<' '<<dt<<' '<<error_interp/dt<<' '<<error_picard/dt<<' '<<max_err/norm_dudt/eps<<'\n';
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
        dt = std::min<Real>(T-t, std::max<Real>(0.5*dt, 0.9*dt*pow<Real>((tol_*dt)/max_err, 1/(Real)(order))));
      }
    }
    if (t < T || error_ > tol) SCTL_WARN("Could not solve ODE to the requested tolerance.");
    if (error != nullptr) (*error) = error_;

    (*u) = u0_;
    return t;
  }

}

#endif // _SCTL_ODE_SOLVER_TXX_
