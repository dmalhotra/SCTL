.. _tutorial-boundaryintegralop:

Using BoundaryIntegralOp class
==============================

The ``BoundaryIntegralOp`` class is designed to construct and evaluate boundary integral operators for solving problems in potential theory, such as the Laplace equation or the Stokes flow problem.

The following is a brief tutorial on using the ``BoundaryIntegralOp`` class. For more advanced usage and additional features, please refer to the class API in :ref:`boundary_integral.hpp <boundary_integral_hpp>`.

Building a Boundary Integral Operator
-------------------------------------

To build a boundary integral operator, follow these steps:

1. Initialize the Boundary Integral Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize the boundary integral operator with the desired kernel and communication context.
In this example, we use the Stokes double-layer kernel (``Stokes3D_DxU``).

.. code-block:: cpp

   const Stokes3D_DxU ker;
   BoundaryIntegralOp<double,Stokes3D_DxU> BIOp(ker, false, comm);

2. Set Quadrature Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the quadrature accuracy for numerical integration.

.. code-block:: cpp

   BIOp.SetAccuracy(1e-10);

3. Add Geometry to the Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the geometry of the surface to the boundary integral operator. This includes specifying the discretization of the surface.
In this example the boundary is given by an instance of `SlenderElemList class <https://csbq.readthedocs.io/en/latest/doxygen/slender_element.html>`_ (in `CSBQ <https://csbq.readthedocs.io>`_).

.. code-block:: cpp

   SlenderElemList<double> elem_lst;
   elem_lst.Read<double>("data/loop.geom", comm);
   BIOp.AddElemList(elem_lst);

4. Set Evaluation Points
~~~~~~~~~~~~~~~~~~~~~~~~

The target points can be specified as follows.
If not set or Xt is empty, then the default target points are the surface discretization nodes.

.. code-block:: cpp

   BIOp.SetTargetCoord(Xt);

Evaluating Potentials
----------------------

Once the boundary integral operator is constructed, you can evaluate potentials at on- and off-surface target points.

1. Compute the Potential
~~~~~~~~~~~~~~~~~~~~~~~~

Compute the potential using the boundary integral operator and a density function defined on the surface.

.. code-block:: cpp

   Vector<double> sigma(Ninput);
   sigma = 1; // Set the density function sigma at each surface discretization node
   Vector<double> U;
   BIOp.ComputePotential(U, sigma); // Compute the potential U

2. Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize the geometry and the computed potential.

.. code-block:: cpp

   elem_lst.WriteVTK("vis/U", U, comm); // Write visualization data to VTK file

Example Code
------------

Below is an example code demonstrating the usage of the ``BoundaryIntegralOp`` class:

.. code-block:: cpp

   #include <sctl.hpp>
   using namespace sctl;

   int main(int argc, char** argv) {
     Comm::MPI_Init(&argc, &argv);

     {
       const Comm comm = Comm::World();

       const Stokes3D_DxU ker;
       BoundaryIntegralOp<double,Stokes3D_DxU> BIOp(ker, false, comm);
       BIOp.SetAccuracy(1e-10);

       SlenderElemList<double> elem_lst;
       elem_lst.Read<double>("data/loop.geom", comm); // load geometry
       BIOp.AddElemList(elem_lst); // add element list to boundary integral operator

       //BIOp.SetTargetCoord(Xt); // set target points (default is discretization nodes)

       const Long Ninput = BIOp.Dim(0); // (local) input dimension of the operator

       Vector<double> sigma(Ninput);
       sigma = 1;

       Vector<double> U;
       BIOp.ComputePotential(U, sigma); // compute potential

       elem_lst.WriteVTK("vis/U", U, comm); // write visualization
     }

     Comm::MPI_Finalize();
     return 0;
   }


