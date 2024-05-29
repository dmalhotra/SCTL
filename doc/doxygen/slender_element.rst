.. _slender_element_hpp:

slender_element.hpp
====================

This header file provides classes and methods for managing lists of slender boundary elements with circular cross-section.

Classes and Types
-----------------

.. doxygenclass:: SCTL_NAMESPACE::SlenderElemList
..   :members:
..

    **Constructor**:

    - ``SlenderElemList()``: Constructor.

    - ``SlenderElemList(cheb_order, fourier_order, coord, radius, orientation)``: Construct the element list from centerline coordinates and cross-sectional radius evaluated at the panel discretization nodes.

    **Methods**:

    - ``CenterlineNodes``: Returns the Chebyshev node points for a given order.

    - ``Init(cheb_order, fourier_order, coord, radius, orientation)``: Initialize list of elements from centerline coordinates and cross-sectional radius evaluated at the panel discretization nodes.

    - ``Size() const``: Returns the number of elements in the list.

    - ``GetNodeCoord``: Returns the position and normals of the surface nodal points for each element.

    - ``GetGeom``: Get geometry data for an element on a tensor-product grid of parameter values :math:`s` and :math:`{\theta}`.

    - ``Write(fname, comm=Comm::Self())``: Write elements to file.

    - ``Read(fname, comm=Comm::Self())``: Read elements from file.

    - ``GetVTUData``: Get the VTU data for one or all elements.

    - ``WriteVTK(fname, F, comm=Comm::Self())``: Write VTU data to file.

    - ``Copy(elem_lst)``: Create a copy of the element-list possibly from a different precision.

..    - ``GetFarFieldNodes``: Returns the quadrature node positions, normals, weights, and cut-off distance for computing the far-field potential.
..
..    - ``GetFarFieldDensity``: Interpolates the density from surface node points to far-field quadrature node points.
..
..    - ``FarFieldDensityOperatorTranspose``: Applies the transpose of the density interpolation operator.
..
..    - ``SelfInterac``: Computes the self-interaction operator for each element.
..
..    - ``NearInterac``: Computes the near-interaction operator for a given element and each target.
..

    **Usage guide**: :ref:`Using SlenderElemList class <tutorial-slenderelemlist>`

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/slender_element.hpp
   :language: c++
