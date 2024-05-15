.. _tutorial-sdc:

Using the SDC class
===================

The `SDC` (Spectral Deferred Correction) class is a template class designed to solve ordinary differential equations (ODEs) using the Spectral Deferred Correction method.
It provides methods for applying one step of the SDC method and for solving ODEs adaptively to a required tolerance.
For more advanced usage and additional features, please refer to the :ref:`SDC class documentation <sdc-dox>`.

To use the `SDC` class, follow these steps:

1. **Instantiate the Class**: Instantiate an object of the `SDC` class with the desired template parameter (`double`).

   .. code-block:: cpp

      SDC<double> ode_solver(Order);

2. **Define the Function `F`**:
   Define a function `F` that represents the derivative `du/dt` of the ODE to be solved. You can define `F` as a lambda function, a functor, or a regular function. The function `F` must have the following signature:

   .. code-block:: cpp

      void F(Vector<double>* dudt, const Vector<double>& u, const Integer correction_idx, const Integer substep_idx);

   Here, `dudt` is a pointer to a vector representing the derivative `du/dt`, `u` is the current solution vector, `correction_idx` is the index of the deferred correction step, and `substep_idx` is the index of the substep within the deferred correction step.

   As an example, `F` can be defined as,

   .. code-block:: cpp

      auto F = [](Vector<double>* dudt, const Vector<double>& u, const Integer correction_idx, const Integer substep_idx) {
          (*dudt)[0] = -u[1];
          (*dudt)[1] = u[0];
      };

3. **Apply One Step of SDC**:
   You can apply one step of the SDC method using the `operator()` method. This method computes the solution `u` at the next time step based on the current solution `u0` and the function `F` representing the derivative `du/dt`.

   .. code-block:: cpp

      Vector<double> u, u0(2);
      u0[0] = 1.0; u0[1] = 0.0;
      double dt = 0.1;
      ode_solver(&u, dt, u0, F);

4. **Solve ODE Adaptively**:
   To solve the ODE adaptively to a required tolerance, use the `AdaptiveSolve` method. This method computes the solution `u` over the interval [0, T] with an initial step size guess `dt`.

   .. code-block:: cpp

      Vector<double> u, u0(2);
      u0[0] = 1.0; u0[1] = 0.0;
      double T = 10.0, dt = 0.1, tol = 1e-5;
      ode_solver.AdaptiveSolve(&u, dt, T, u0, F, tol);

