.. _kernel_functions_hpp:

kernel_functions.hpp
====================

This header file defines various kernel functions used for computing potentials and gradients in Laplace and Stokes problems in 3D.
The kernel objects inherit from the ``GenericKernel`` class defined in :ref:`generic-kernel.hpp <generic-kernel_hpp>`.
These kernel implementations can be used as templates for other user defined kernels.

     - ``Laplace3D_FxU``: Laplace single-layer kernel.
     
     - ``Laplace3D_DxU``: Laplace double-layer kernel.
     
     - ``Laplace3D_FxdU``: Laplace single-layer gradient kernel.
     
     - ``Stokes3D_FxU``: Stokes single-layer velocity kernel.
     
     - ``Stokes3D_DxU``: Stokes double-layer velocity kernel.
     
     - ``Stokes3D_FxT``: Stokes traction kernel.
     
     - ``Stokes3D_FSxU``: Stokes single-layer + source-term kernel (required for multipole-to-local translations in FMM when double-layer sources are involved).
     
     - ``Stokes3D_FxUP``: Stokes single-layer velocity and pressure kernel.

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/kernel_functions.hpp
   :language: c++

