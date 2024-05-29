.. _tutorial-slenderelemlist:

Using SlenderElemList Class
===========================

The `SlenderElemList` class is designed to represent slender-body geometries and use them in boundary integral equation (BIE) methods.
This tutorial will guide you through the basic usage of the `SlenderElemList` class, covering construction, file I/O, accessing surface geometry, and visualization.
To see how to compute boundary integrals see :ref:`Using BoundaryIntegralOp class <tutorial-boundaryintegralop>`.
The full API documentation please refer to :ref:`slender_element.hpp <slender_element_hpp>`.

Construction of Slender-Body Geometry
--------------------------------------

To construct a slender-body geometry, follow these steps:

1. Define parameters such as element order (`ElemOrder`), and number of elements (`Nelem`).
   Construct the centerline coordinates (`Xc`) and cross-sectional radii (`eps`). For example, constructing a circular centerline:

   .. code-block:: cpp

      // Loop to construct the centerline for a circle
      Vector<Real> Xc, eps;
      for (Long i = 0; i < Nelem; i++) {
          const Vector<Real>& elem_nodes = SlenderElemList<Real>::CenterlineNodes(ElemOrder); // discretization nodes within an element in [0,1] interval
          for (Long j = 0; j < ElemOrder; j++) {
              const Real phi = 2*const_pi<Real>() * (i + elem_nodes[j]) / Nelem; // circle parameterization phi in [0,2pi]
              Xc.PushBack(cos(phi)); // X-coord
              Xc.PushBack(sin(phi)); // Y-coord
              Xc.PushBack(0.0);      // Z-coord
              eps.PushBack(0.1);     // cross-sectional radius
          }
      }

2. Initialize the `SlenderElemList` object:

   .. code-block:: cpp

      SlenderElemList<Real> elem_lst(ElemOrderVec, FourierOrderVec, Xc, eps);

File I/O
--------

The `SlenderElemList` class provides methods to read from and write to files in a human-readable format. To perform file I/O:

1. Write the geometry data to a file:

   .. code-block:: cpp

      elem_lst.Write("path/to/file.geom", comm);

2. Read geometry data from a file:

   .. code-block:: cpp

      elem_lst.Read<Real>("path/to/file.geom", comm);


The geometry file contains the coordinates, the cross-sectional radius, and the orientation vector at the Chebyshev nodes of each panel along the centerline.
An example of a geometry file is shown below:

   .. code-block:: sh

      #          X           Y           Z           r    orient-x    orient-y    orient-z   ChebOrder FourierOrder
        1.2812e-01  1.5108e-04  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00          10           88
        1.2812e-01  1.3375e-03  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2812e-01  3.5943e-03  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2811e-01  6.7005e-03  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2809e-01  1.0352e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2807e-01  1.4191e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2804e-01  1.7842e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2801e-01  2.0948e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2799e-01  2.3205e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2797e-01  2.4392e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2797e-01  2.4694e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00          10           88
        1.2795e-01  2.5880e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2792e-01  2.8137e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        1.2788e-01  3.1242e-02  0.0000e+00  1.2500e-01  0.0000e+00  0.0000e+00  1.0000e+00
        ....
 

Accessing Surface Geometry
--------------------------

You can retrieve the surface discretization nodes and normals using the `GetNodeCoord` method:

.. code-block:: cpp

   Vector<Real> X, Xn;
   Vector<Long> element_wise_node_cnt;
   elem_lst.GetNodeCoord(&X, &Xn, &element_wise_node_cnt);

Visualization
-------------

Finally, you can visualize the geometry and surface normals using VTK. Write VTK visualization files using the `WriteVTK` method:

.. code-block:: cpp

   elem_lst.WriteVTK("path/to/output", Xn, comm);
