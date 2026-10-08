.. _kernel_functions_hpp:

kernel_functions.hpp
====================

This header file defines various kernel functions used for computing potentials and gradients in Laplace and Stokes problems in 3D.
The kernel objects inherit from the ``GenericKernel`` class defined in :ref:`generic-kernel.hpp <generic-kernel_hpp>`.
These kernel implementations can be used as templates for writing new user defined kernels.
This is explained further in :ref:`Writing Custom Kernel Objects <tutorial-kernels>`.

     - ``Laplace3D_FxU``: Laplace single-layer kernel.
     
     - ``Laplace3D_DxU``: Laplace double-layer kernel.
     
     - ``Laplace3D_FxdU``: Laplace single-layer gradient kernel.
     
     - ``Stokes3D_FxU``: Stokes single-layer velocity kernel.
     
     - ``Stokes3D_DxU``: Stokes double-layer velocity kernel.
     
     - ``Stokes3D_FxT``: Stokes traction kernel.
     
     - ``Stokes3D_FSxU``: Stokes single-layer + source-term kernel (required for multipole-to-local translations in FMM when double-layer sources are involved). The source term :math:`s\,r/(8\pi|r|^3)` is half the velocity of a point source of volume flux :math:`s`, a scale that the FMM translations absorb.
     
     - ``Stokes3D_FxUP``: Stokes single-layer velocity and pressure kernel. The pressure :math:`(f\cdot r)/(8\pi|r|^3)` is half that of the Stokeslet, :math:`(f\cdot r)/(4\pi|r|^3)`, a scale that the FMM translations absorb.

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/kernel_functions.hpp
   :language: c++

