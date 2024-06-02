.. _generic-kernel_hpp:

generic-kernel.hpp
===================

This header file defines the `GenericKernel` template class, which simplifies building new custom kernel objects.
Kernels for Laplace and Stokes in 3D are defined in :ref:`kernel_functions.hpp <kernel_functions_hpp>` and can be used as a template.

Classes and Types
-----------------

.. doxygenclass:: sctl::GenericKernel
..   :members:
..

    **Static Member Functions**:

    - ``CoordDim()``: Returns the coordinate dimension.
    - ``NormalDim()``: Returns the normal dimension.
    - ``SrcDim()``: Returns the source dimension.
    - ``TrgDim()``: Returns the target dimension.
    - ``Eval(v_trg, r_trg, r_src, n_src, v_src, digits, self)``: Evaluates the kernel and stores the result in `v_trg`.

    **Member Functions**:

    - ``GetCtxPtr() const``: Returns a constant pointer to the context.
    - ``Eval(v_trg, r_trg, r_src, n_src, v_src) const``: Evaluates the kernel with optional template parameters for OpenMP and digits.
    - ``KernelMatrix(M, Xt, Xs, Xn) const``: Computes the kernel matrix and stores it in `M`.

    **Usage guide**: :ref:`Writing Custom Kernel Objects <tutorial-kernels>`, :ref:`kernel_functions.hpp <kernel_functions_hpp>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/generic-kernel.hpp
   :language: c++
