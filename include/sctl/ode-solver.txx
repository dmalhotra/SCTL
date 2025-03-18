#ifndef _SCTL_ODE_SOLVER_TXX_
#define _SCTL_ODE_SOLVER_TXX_

#include <stdio.h>                   // for printf
#include <algorithm>                 // for max, min

#include "sctl/common.hpp"           // for Long, Integer, SCTL_ASSERT, SCTL...
#include "sctl/ode-solver.hpp"       // for SDC
#include "sctl/comm.hpp"             // for Comm (ptr only), CommOp
#include "sctl/comm.txx"             // for Comm::Allreduce, Comm::Comm
#include "sctl/iterator.hpp"         // for Iterator, ConstIterator
#include "sctl/iterator.txx"         // for Iterator::operator[]
#include "sctl/lagrange-interp.hpp"  // for LagrangeInterp
#include "sctl/lagrange-interp.txx"  // for LagrangeInterp::Interpolate
#include "sctl/math_utils.hpp"       // for QuadReal, cos, operator*, operator-
#include "sctl/math_utils.txx"       // for const_pi, cos, pow, machine_eps
#include "sctl/matrix.hpp"           // for Matrix
#include "sctl/matrix.txx"           // for Matrix::operator[], Matrix::Matr...
#include "sctl/quadrule.hpp"         // for ChebQuadRule
#include "sctl/quadrule.txx"         // for ChebQuadRule::ComputeNdsWts
#include "sctl/static-array.hpp"     // for StaticArray
#include "sctl/vector.hpp"           // for Vector
#include "sctl/vector.txx"           // for Vector::Vector<ValueType>, Vecto...

namespace sctl {

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
    Real t = ode_solver.AdaptiveSolve(&u, dt, T, u0, fn, tol, nullptr, true);

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

    const auto build_trunc_matrix = [](const Vector<ValueType>& nds0, const Vector<ValueType>& nds1) {
      const Integer order = nds0.Dim();
      const Integer TRUNC_Order = nds1.Dim();

      Matrix<ValueType> Minterp0(order, TRUNC_Order);
      Matrix<ValueType> Minterp1(TRUNC_Order, order);
      Vector<ValueType> interp0(order*TRUNC_Order, Minterp0.begin(), false);
      Vector<ValueType> interp1(TRUNC_Order*order, Minterp1.begin(), false);
      LagrangeInterp<ValueType>::Interpolate(interp0, nds0, nds1);
      LagrangeInterp<ValueType>::Interpolate(interp1, nds1, nds0);
      Matrix<ValueType> M_error_ = (Minterp0 * Minterp1).Transpose();
      for (Long i = 0; i < order; i++) M_error_[i][i] -= 1;

      Matrix<Real> M_error(order, order);
      for (Long i = 0; i < order*order; i++) M_error[0][i] = (Real)M_error_[0][i];
      return M_error;
    };
    { // Set M_error
      Integer TRUNC_Order = order;
      if (order >= 2) TRUNC_Order = order - 1;
      if (order >= 6) TRUNC_Order = order - 1;
      if (order >= 9) TRUNC_Order = order - 1;
      M_error = build_trunc_matrix(nds0, second_kind_cheb_nds(TRUNC_Order));
      M_error_half = build_trunc_matrix(nds0, second_kind_cheb_nds(order/2));
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
  template <class Real> void SDC<Real>::operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, const Fn0& F, Integer N_picard, const Real tol_picard, Real* error_interp, Real* error_picard, Integer* iter_count, Matrix<Real>* u_substep) const {
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
        if (error_interp) (*error_interp) = -1;
        if (error_picard) (*error_picard) = -1;
        if (iter_count) (*iter_count) = -1;
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
      picard_err[picard_iter] = max_norm(Mv_change) * dt;
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
          if (error_interp) (*error_interp) = -1;
          if (error_picard) (*error_picard) = -1;
          if (iter_count) (*iter_count) = -1;
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
      Matrix<Real>::GEMM(err, M_error, Mu);
      (*error_interp) = max_norm(err);
    }
    if (iter_count != nullptr) {
      (*iter_count) = picard_iter;
    }
    if (u_substep != nullptr) {
      if (u_substep->Dim(0) != order || u_substep->Dim(1) != DOF) u_substep->ReInit(order, DOF);
      (*u_substep) = Mu;
    }
  }

  template <class Real> void SDC<Real>::operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, const Fn1& F, Integer N_picard, const Real tol_picard, Real* error_interp, Real* error_picard, Integer* iter_count, Matrix<Real>* u_substep) const {
    const auto fn = [&F](Vector<Real>* dudt, const Vector<Real>& u, const Integer correction_idx, const Integer substep_idx) {
      F(dudt, u);
    };
    this->operator()(u, dt, u0, fn, N_picard, tol_picard, error_interp, error_picard, iter_count, u_substep);
  }

  template <class Real> Real SDC<Real>::AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn0& F, const Real tol, const MonitorFn* monitor_callback, bool continue_with_errors, Real* error) const {
    const Integer max_picard_iter = 2*order;
    const Real eps = machine_eps<Real>();
    Vector<Real> u_, u0_ = u0;

    Real t = 0;
    Real error_ = 0;
    Matrix<Real> u_substep;
    while (t < T && dt > eps*T) {
      Integer picard_iter;
      Real error_interp, error_picard;
      Real tol_ = std::max<Real>(tol/T, (tol-error_)/(T-t));
      (*this)(&u_, dt, u0_, F, max_picard_iter, tol_*dt, &error_interp, &error_picard, &picard_iter, &u_substep);

      const Real max_norm_u = max_norm(u_);
      const Real error_interp_half = pow<Real>(max_norm(M_error_half*u_substep)/max_norm_u, (Real)1.8) * max_norm_u;

      std::cout<<"Adaptive time-step: " << std::scientific << std::setw(10) <<t<<' '<<dt<<' '<<picard_iter<<" "<<error_interp_half/dt<<' '<<error_picard/dt<<' '<<error_/tol<<'\n';
      if (picard_iter>=0 && (error_interp < tol_*dt || error_interp_half < error_interp) && (error_picard < tol_*dt || picard_iter < max_picard_iter)) { // Accept solution
        // picard_iter>=0                     // SDC time-step succeeded
        // && (
        //   error_interp < tol_*dt           // interpolant error tolerance reached
        //   ||
        //   error_interp_half < error_interp // interpolant coefficients stagnated
        // ) && (
        //   error_picard < tol_*dt           // picard-iteration error tolerance reached
        //   ||
        //   picard_iter < max_picard_iter    // picard-iteration error stagnated
        // )

        u0_.Swap(u_);
        t = t + dt;
        error_ += std::max<Real>(error_interp, error_picard);
        SCTL_ASSERT_MSG(continue_with_errors || error_ < tol, "Could not solve ODE to the requested tolerance.");
        if (monitor_callback) (*monitor_callback)(t, dt, u0_);
      }

      const Real dt_picard = (picard_iter < max_picard_iter ? dt*(max_picard_iter-1)/picard_iter : log(error_picard)/log(tol_*dt)*dt );
      const Real dt_interp = (error_interp_half < error_interp ?
                              std::min<Real>(2.0*dt, 0.9*dt*pow<Real>(error_interp/error_interp_half, 1/(Real)(order))) : // adjust time-step size to match stagnation error
                              std::max<Real>(0.5*dt, 0.9*dt*pow<Real>((tol_*dt)   /error_interp_half, 1/(Real)(order)))); // Adjust time-step size (Quaife, Biros - JCP 2016)
      dt = std::min<Real>(T-t, std::min(dt_interp, dt_picard));
    }
    if (t < T || error_ > tol) SCTL_WARN("Could not solve ODE to the requested tolerance.");
    if (error != nullptr) (*error) = error_;

    (*u) = u0_;
    return t;
  }

  template <class Real> Real SDC<Real>::AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn1& F, Real tol, const MonitorFn* monitor_callback, bool continue_with_errors, Real* error) const {
    const auto fn = [&F](Vector<Real>* dudt, const Vector<Real>& u, const Integer correction_idx, const Integer substep_idx) {
      F(dudt, u);
    };
    return AdaptiveSolve(u, dt, T, u0, fn, tol, monitor_callback, continue_with_errors, error);
  }

  template <class Real> template <class Container> Real SDC<Real>::max_norm(const Container& M) const {
    StaticArray<Real,2> max_val{0,0};
    for (const auto x : M) max_val[0] = std::max<Real>(max_val[0], fabs((Real)x));
    comm.Allreduce((ConstIterator<Real>)max_val, (Iterator<Real>)max_val+1, 1, CommOp::MAX);
    return max_val[1];
  }
}

#endif // _SCTL_ODE_SOLVER_TXX_
