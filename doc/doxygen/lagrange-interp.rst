.. _lagrange-interp_hpp:

lagrange-interp.hpp
===================

This header file provides functionality for Lagrange interpolation, including computing interpolation weights and derivatives.

Classes and Types
-----------------

.. doxygenclass:: SCTL_NAMESPACE::LagrangeInterp
..   :members:
..

    **Methods**:

    - ``Interpolate(wts, src_nds, trg_nds)``: Computes the interpolation weights `wts` to interpolate values from the source nodes to the target nodes.

    - ``Derivative(df, f, nds)``: Computes the derivative `df` of the polynomial interpolant from function values given at the interpolation nodes.

    **Usage guide**: :ref:`Using the LagrangeInterp Class <tutorial-lagrange-interp>`

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/lagrange-interp.hpp
   :language: c++

