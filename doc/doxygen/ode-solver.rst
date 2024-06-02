.. _ode-solver_hpp:

ode-solver.hpp
==============

This header file provides the Spectral Deferred Correction (SDC) solver for ordinary differential equations (ODEs).

Classes and Types
-----------------

.. doxygenclass:: sctl::SDC
..   :members:
..

    **Constructor**:

    - ``SDC(order, comm=Comm::Self())``: Constructor.

    **Methods**:

    - ``Order() const``: Returns the order of the method.

    - ``operator()``: Applies one step of the SDC method.

    - ``AdaptiveSolve``: Solves the ODE adaptively to a required tolerance.

    **Types**:

    - ``Fn0``, ``Fn1``: Function types for specifying the RHS of the ODE.

    - ``MonitorFn``: Callback function type for monitoring the solution during time-stepping.

    **Usage guide**: :ref:`Using SDC class <tutorial-sdc>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/ode-solver.hpp
   :language: c++

