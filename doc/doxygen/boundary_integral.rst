.. _boundary_integral_hpp:

boundary_integral.hpp
======================

This header file provides classes and methods for computing boundary integrals.

Classes and Types
-----------------

.. doxygenclass:: sctl::ElementListBase
..   :members:
..

    **Constructor**:

    - ``ElementListBase()``: Constructor.

    **Methods**:

    - ``Size() const``: Returns the number of elements in the list.

    - ``GetNodeCoord``: Returns the position and normals of the surface nodal points for each element.

    - ``GetFarFieldNodes``: Returns the quadrature node positions, normals, weights, and cut-off distance for computing the far-field potential.

    - ``GetFarFieldDensity``: Interpolates the density from surface node points to far-field quadrature node points.

    - ``FarFieldDensityOperatorTranspose``: Applies the transpose of the density interpolation operator.

    - ``SelfInterac``: Computes the self-interaction operator for each element.

    - ``NearInterac``: Computes the near-interaction operator for a given element and each target.

|

.. doxygenclass:: sctl::BoundaryIntegralOp
..   :members:
..

    **Constructor**:

    - ``BoundaryIntegralOp(ker, trg_normal_dot_prod=False, comm=Comm::Self())``: Constructor.

    **Methods**:

    - ``SetAccuracy(tol)``: Sets quadrature accuracy tolerance.

    - ``SetFMMKer``: Sets kernel functions for FMM translation operators.

    - ``AddElemList``: Adds an element-list.

    - ``GetElemList``: Gets a reference to an element-list.

    - ``DeleteElemList``: Deletes an element-list.

    - ``SetTargetCoord``: Sets target point coordinates.

    - ``SetTargetNormal``: Sets target point normals.

    - ``Dim``: Gets the local dimension of the boundary integral operator.

    - ``Setup``: Sets up the boundary integral operator.

    - ``ClearSetup``: Clears setup data.

    - ``ComputePotential``: Evaluates the boundary integral operator.

    - ``SqrtScaling``: Scales input vector by sqrt of the area of the element.

    - ``InvSqrtScaling``: Scales input vector by inv-sqrt of the area of the element.

    **Usage guide**: :ref:`Using BoundaryIntegralOp class <tutorial-boundaryintegralop>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/boundary_integral.hpp
   :language: c++
