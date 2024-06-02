.. _fft_wrapper_hpp:

fft_wrapper.hpp
===============

This header file provides a wrapper class for Fast Fourier Transform (FFT) operations using FFTW library.
If the FFTW library is not available, it computes the Fourier Transform directly (by building the Fourier matrix).

Classes and Types
-----------------

.. doxygenclass:: sctl::FFT
..   :members:
..

    **Constructor**:

    - ``FFT()``: Constructor.

    **Methods**:

    - ``Dim(Integer i) const``: Returns the dimension of the FFT operator for input (i=0) and output (i=1) arrays.
    - ``Setup(fft_type, howmany, dim_vec, Nthreads = 1)``: Sets up the FFT operator.
    - ``Execute(in, out) const``: Executes the FFT transform.

    **Usage guide**: :ref:`Using the FFT class <tutorial-fft>`

|

.. doxygenenum:: sctl::FFT_Type
..

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/fft_wrapper.hpp
   :language: c++
