.. _quadrule_hpp:

quadrule.hpp
============

This header file provides various classes and methods for different types of quadrature rules,
including Clenshaw-Curtis, Gauss-Legendre, and generalized Chebyshev quadrature rules.

Classes and Types
-----------------

.. doxygenclass:: sctl::ChebQuadRule
..   :members:
..

    **Methods**:

    - ``nds<MAX_ORDER=50>(N)``, ``wts<MAX_ORDER=50>(N)``: Precompute (up to MAX_ORDER) and return the quadrature nodes and weights of order N.
    - ``nds<N>()``, ``wts<N>()``: Precompute and return the quadrature nodes and weights of order N (known at compile time).
    - ``ComputeNdsWts(nds, wts, N)``: Compute (on-the-fly) the quadrature nodes and weights for order N.

|

.. doxygenclass:: sctl::LegQuadRule
..   :members:
..

    **Methods**:

    - ``nds<MAX_ORDER=50>(N)``, ``wts<MAX_ORDER=50>(N)``: Precompute (up to MAX_ORDER) and return the quadrature nodes and weights of order N.
    - ``nds<N>()``, ``wts<N>()``: Precompute and return the quadrature nodes and weights of order N (known at compile time).
    - ``ComputeNdsWts(nds, wts, N)``: Compute (on-the-fly) the quadrature nodes and weights for order N.
    - ``LegPoly(P, dP, X, degree)``: Computes the Legendre polynomial and/or its first derivative.

|

.. doxygenclass:: sctl::InterpQuadRule
..   :members:
..

    **Methods**:

    - ``Build(Vector<Real>& quad_nds, quad_wts, integrands, interval_start, interval_end, eps = 1e-16, ...)``: Build a quadrature rule from a function pointer to the integrands.
    - ``Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, M, nds, wts, eps = 1e-16, ...)``: Build a quadrature rule from a given discretization of the integrand functions.
    - ``Build(Vector<Vector<Real>>& quad_nds, Vector<Vector<Real>>& quad_wts, M, nds, wts, ...)``: Build a set of quadrature rules for different accuracies from a given discretization of the integrand functions.

    **Usage guide**: :ref:`Using InterpQuadRule class <tutorial-interp-quadrule>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/quadrule.hpp
   :language: c++

