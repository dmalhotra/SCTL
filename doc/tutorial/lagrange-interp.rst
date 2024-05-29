.. _tutorial-lagrange-interp:

Using the LagrangeInterp Class
==============================

This tutorial will guide you through the process of using the `LagrangeInterp` class for performing Lagrange interpolation and computing derivatives. We will cover the following steps:

1. Initializing source and target nodes
2. Computing interpolation weights
3. Performing interpolation
4. Computing derivatives

For the API documentation for `LagrangeInterp` class please refer to :ref:`lagrange-interp.hpp <lagrange-interp_hpp>`.

Initializing Source and Target Nodes
------------------------------------

First, you need to initialize the source and target nodes. Source nodes are the points where the function values are known, and target nodes are the points where you want to interpolate the function values.

.. code-block:: cpp

   sctl::Vector<double> src_nodes, trg_nodes;

   // Initialize source nodes
   for (Long i = 0; i < 3; i++) src_nodes.PushBack(i);

   // Initialize target nodes
   for (Long i = 0; i < 11; i++) trg_nodes.PushBack(i * 0.2);

Computing Interpolation Weights
-------------------------------

Next, compute the interpolation weights using the `Interpolate` method. The weights are stored in a vector and will be used to interpolate the function values from the source nodes to the target nodes.

.. code-block:: cpp

   sctl::Vector<double> weights;
   sctl::LagrangeInterp<double>::Interpolate(weights, src_nodes, trg_nodes);

Performing Interpolation
------------------------

With the interpolation weights computed, you can now perform the interpolation. Define the function values at the source nodes and use the weights to calculate the interpolated values at the target nodes.

.. code-block:: cpp

   // Define function values at source nodes
   sctl::Matrix<double> f(1, 3);
   f[0][0] = 0;
   f[0][1] = 1;
   f[0][2] = 0.5;

   // Reshape the weights vector into a matrix for multiplication
   sctl::Matrix<double> Mwts(src_nodes.Dim(), trg_nodes.Dim(), weights.begin(), false);
   sctl::Matrix<double> interpolated_values = f * Mwts;

   // Output the interpolated values
   std::cout << interpolated_values << '\n';

Computing Derivatives
---------------------

To compute the derivatives of the interpolated values, use the `Derivative` method. This will give you the derivative values at the source nodes.

.. code-block:: cpp

   sctl::Vector<double> derivatives;
   sctl::LagrangeInterp<double>::Derivative(derivatives, sctl::Vector<double>(f.Dim(0) * f.Dim(1), f.begin(), false), src_nodes);

   // Output the derivatives
   std::cout << derivatives << '\n';

Putting It All Together
-----------------------

Here is the complete example, combining all the steps described above:

.. code-block:: cpp

   #include "sctl.hpp"
   #include <iostream>

   int main() {
       sctl::Vector<double> src_nodes, trg_nodes, weights, derivatives;

       // Initialize source nodes
       for (Long i = 0; i < 3; i++) {
           src_nodes.PushBack(i);
       }

       // Initialize target nodes
       for (Long i = 0; i < 11; i++) {
           trg_nodes.PushBack(i * 0.2);
       }

       // Compute interpolation weights
       sctl::LagrangeInterp<double>::Interpolate(weights, src_nodes, trg_nodes);

       // Define function values at source nodes
       sctl::Matrix<double> f(1, 3);
       f[0][0] = 0;
       f[0][1] = 1;
       f[0][2] = 0.5;

       // Perform interpolation
       sctl::Matrix<double> Mwts(src_nodes.Dim(), trg_nodes.Dim(), weights.begin(), false);
       sctl::Matrix<double> interpolated_values = f * Mwts;
       std::cout << interpolated_values << '\n';

       // Compute derivatives
       sctl::LagrangeInterp<double>::Derivative(derivatives, sctl::Vector<double>(f.Dim(0) * f.Dim(1), f.begin(), false), src_nodes);
       std::cout << derivatives << '\n';

       return 0;
   }


