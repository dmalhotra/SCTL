#include "sctl.hpp"

template <class Real> void test_adaptive_solve(const int Order, const Real tol) {
  auto ref_sol = [](Real t) { return sctl::cos(-t); };
  auto fn = [](sctl::Vector<Real>* dudt, const sctl::Vector<Real>& u) {
    (*dudt)[0] = -u[1];
    (*dudt)[1] = u[0];
  };

  sctl::Vector<Real> u, u0(2);
  u0[0] = 1.0; u0[1] = 0.0;
  Real T = 10.0, dt = 1.0e-1;

  sctl::SDC<Real> ode_solver(Order);
  Real t = ode_solver.AdaptiveSolve(&u, dt, T, u0, fn, tol);

  if (t == T) {
    printf("u = %e;  ", (double)u[0]);
    printf("error = %e;  \n", (double)(ref_sol(T) - u[0]));
  }
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  test_adaptive_solve<double>(5, 1e-5); // 5-th order scheme
  test_adaptive_solve<double>(12, 1e-12); // 12-th order scheme

  //test_adaptive_solve<double>(12, 1e-18);

  sctl::Comm::MPI_Finalize();
  return 0;
}

