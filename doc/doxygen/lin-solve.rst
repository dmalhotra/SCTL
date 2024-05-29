.. _lin-solve_hpp:

lin-solve.hpp
=============

This header file provides classes for solving linear systems using iterative methods, including a GMRES solver and a Krylov-subspace preconditioner.

Classes and Types
-----------------

.. doxygenclass:: SCTL_NAMESPACE::KrylovPrecond
..   :members:
..

    **Constructor**:

    - ``KrylovPrecond()``: Constructor.

    **Methods**:

    - ``Size() const``: Get the size of the input vector to the operator.

    - ``Rank() const``: Get the cumulative size of the Krylov-subspaces.

    - ``Append(Qt, U)``: Append a Krylov-subspace to the operator.

    - ``Apply(x) const``: Apply the preconditioner.


    **Usage guide**: :ref:`Using GMRES and KrylovPrecond classes <tutorial-gmres>`

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>


.. doxygenclass:: SCTL_NAMESPACE::GMRES
..   :members:
..

    **Constructor**:

    - ``GMRES(comm=Comm::Self(), verbose=true)``: Constructor.

    **Member Functions**:

    - ``operator()(x, A, b, tol, max_iter=-1, use_abs_tol=false, solve_iter=nullptr, krylov_precond=nullptr) const``: Solve the linear system A(x) = b.

    **Types**:

    - ``ParallelOp``: Function type for linear operator.


    **Usage guide**: :ref:`Using GMRES and KrylovPrecond classes <tutorial-gmres>`


.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/lin-solve.hpp
   :language: c++

