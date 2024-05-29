.. _tutorial-gmres:

Using GMRES and KrylovPrecond classes
=====================================

This tutorial demonstrates how to use the GMRES solver with a Krylov preconditioner to solve linear systems efficiently.
We'll walk through constructing a linear operator, creating instances of GMRES and KrylovPrecond, and solving a linear system.
For more advanced usage and additional features, please refer to the API in :ref:`lin-solve.hpp <lin-solve_hpp>`.

Step 1: Constructing a Linear Operator
---------------------------------------

First, we need to define our linear operator. This operator represents the matrix-vector multiplication in the linear system Ax = b. We'll use a lambda function to define this operator.

.. code-block:: cpp

    auto LinOp = [&A](Vector<double>* Ax, const Vector<double>& x) {
        // Define the action of A on x and store the result in Ax
    };

Step 2: Creating Instances of GMRES and KrylovPrecond
----------------------------------------------------

Next, we'll create instances of the GMRES solver and the Krylov preconditioner.

.. code-block:: cpp

    GMRES<double> solver;
    KrylovPrecond<double> krylov_precond;

Step 3: Solving the Linear System
----------------------------------

With our linear operator and solver instances ready, we can now solve the linear system Ax = b. We'll call the GMRES solver with the necessary arguments.

.. code-block:: cpp

    Vector<double> x, b;
    Real tol = 1e-10;
    solver(&x, LinOp, b, tol, -1, false, nullptr, &krylov_precond);

The above call to the GMRES solver will initialize the Krylov preconditioner and solve the linear system. Subsequent solves with the same linear operator will benefit from the preconditioner and may require fewer iterations.

.. note::

    Ensure that your linear operator and right-hand side vector `b` are properly defined before calling the solver.

Example Code
------------

The complete example code is shown below:

.. code-block:: cpp

    #include "sctl.hpp"
    using namespace sctl;

    Matrix<double> LowRankMatrix(const Long N, const Long rank) {
      Matrix<double> M(N, N);
      M.SetZero();

      for (Long r = 0; r < rank; r++) {
        Matrix<double> U(N,1), Vt(1, N);
        for (auto& a : U ) a = drand48();
        for (auto& a : Vt) a = drand48();
        M += U * Vt * exp(log(machine_eps<double>())*r/rank);
      }
      return M;
    }

    double max_norm(const Vector<double>& v) {
      double max_val = 0;
      for (const auto a : v) max_val = std::max<double>(max_val, fabs(a));
      return max_val;
    }

    int main(int argc, char** argv) {
      Comm::MPI_Init(&argc, &argv);

      { // Example usage for GMRES and KrylovPrecond
        Long N = 200, rank = 200;

        // Build A = I + <low-rank>
        Matrix<double> A = LowRankMatrix(N, rank);
        for (Long i = 0; i < N; i++) A[i][i] += 1;

        // Build linear operator
        auto LinOp = [&A](Vector<double>* Ax, const Vector<double>& x) {
          const Long N = x.Dim();
          Ax->ReInit(N);
          Matrix<double> Ax_(N, 1, Ax->begin(), false);
          Ax_ = A * Matrix<double>(N, 1, (Iterator<double>)x.begin(), false);
        };

        // Set exact solution x0 and the RHS b := A * x
        Vector<double> x0(N), b(N), x;
        for (auto& a : x0) a = drand48();
        LinOp(&b, x0);

        // Solve using GMRES
        GMRES<double> solver;
        KrylovPrecond<double> krylov_precond;
        solver(&x, LinOp, b, 1e-10, -1, false, nullptr, &krylov_precond);
        std::cout<<"Solution error = "<<max_norm(x-x0)<<"\n\n";

        // Solve a new problem, reusing the Krylov preconditioner
        for (auto& a : x0) a = drand48();
        LinOp(&b, x0);
        solver(&x, LinOp, b, 1e-10, -1, false, nullptr, &krylov_precond);
        std::cout<<"Solution error = "<<max_norm(x-x0)<<"\n\n";
      }

      Comm::MPI_Finalize();
      return 0;
    }

