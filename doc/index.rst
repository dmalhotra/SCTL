.. _sctl_documentation:

.. .. contents::
..    :local: 

.. toctree::
   :hidden:
   :maxdepth: 1

   Introduction <self>

.. toctree::
   :hidden:
   :maxdepth: 3

   tutorial/index
 
.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 1

   doxygen/index

SCTL: Scientific Computing Template Library
===========================================

.. image:: https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml/badge.svg
   :target: https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml
   :alt: Build and test status

.. image:: https://codecov.io/gh/dmalhotra/SCTL/branch/master/graph/badge.svg?token=UIM2RYMF6D
   :target: https://codecov.io/gh/dmalhotra/SCTL
   :alt: Code coverage

.. image:: https://badgen.net/github/tag/dmalhotra/SCTL
   :target: https://github.com/dmalhotra/SCTL/tags
   :alt: Stable version

.. image:: https://img.shields.io/github/v/release/dmalhotra/SCTL?color=%233D9970
   :target: https://github.com/dmalhotra/SCTL/releases
   :alt: Latest release

`SCTL <https://github.com/dmalhotra/SCTL>`_ is a header-only C++ library that provides several functionalities useful in scientific computing.
This documentation outlines these functionalities.

Getting Started
---------------

.. note::

    SCTL requires a C++11 compliant compiler with OpenMP 4.0 support.

To get started, download the latest version of SCTL from the `SCTL GitHub <https://github.com/dmalhotra/SCTL>`_.

.. code-block:: bash

   git clone https://github.com/dmalhotra/SCTL.git

Since SCTL is a header-only library, it does not require compilation or installation.
Simply include the ``sctl.hpp`` header file in your C++ project and start using the provided classes and functions.

.. code-block:: cpp

   #include <sctl.hpp>

Make sure to provide the path to ``SCTL_ROOT/include`` (where the header ``sctl.hpp`` is located) to the compiler using the flag ``-I ${SCTL_ROOT}/include``.

Dependencies
------------

The only requirement to use SCTL is a working C++11 compliant compiler with OpenMP 4.0 support. It has been tested with GCC-9 and newer.

SCTL can optionally use the following libraries if they are available:

- **MPI**: enable by defining ``SCTL_HAVE_MPI``.

- **BLAS**: enable by defining ``SCTL_HAVE_BLAS``.

- **LAPACK**: enable by defining ``SCTL_HAVE_LAPACK``.

- `FFTW <https://www.fftw.org>`_: enable for double precision by defining ``SCTL_HAVE_FFTW``, single precision by defining ``SCTL_HAVE_FFTWF``, or long double precision by defining ``SCTL_HAVE_FFTWL``.

- **libmvec**: enable by defining ``SCTL_HAVE_LIBMVEC``.

- **Intel SVML**: enable by defining ``SCTL_HAVE_SVML``.

- `PVFMM <http://pvfmm.org>`_: enable by defining ``SCTL_HAVE_PVFMM`` (requires MPI).

To enable support for any of these libraries, define the corresponding flag during compilation. For example, to enable MPI support, use ``-DSCTL_HAVE_MPI``.

..  SCTL_MEMDEBUG
..  SCTL_GLOBAL_MEM_BUFF
..  SCTL_PROFILE
..  SCTL_VERBOSE
..  SCTL_SIG_HANDLER
..  SCTL_QUAD_T

Containers
----------

1. :ref:`Vector <tutorial-vector>`
2. :ref:`Matrix <tutorial-matrix>`
3. :ref:`Permutation <tutorial-permutation>`
4. :ref:`Tensor <tutorial-tensor>`

Numerical Methods
-----------------

1. :ref:`SDC (Spectral Deferred Correction ODE solver) <tutorial-sdc>`
2. :ref:`GMRES solver, Krylov preconditioner <tutorial-gmres>`: distributed memory GMRES
3. :ref:`LagrangeInterp <tutorial-lagrange-interp>`
4. :ref:`InterpQuadRule, ChebQuadRule, LegQuadRule <quadrule_hpp>`: generalized Chebyshev quadrature, Clenshaw-Curtis quadrature, Gauss-Legendre quadrature
5. :ref:`SphericalHarmonics <sph_harm_hpp>`
6. :ref:`Tree, PtTree <tutorial-tree>`, :ref:`Morton <morton_hpp>`: Morton ordering based n-dimensional tree
7. :ref:`FFT <fft_wrapper_hpp>`: wrapper to `FFTW <https://www.fftw.org>`_
8. :ref:`FMM <fmm-wrapper_hpp>`: wrapper to `PVFMM <http://pvfmm.org>`_
9. :ref:`Kernel functions <kernel_functions_hpp>`

Boundary Integral Methods
-------------------------

1. :ref:`BoundaryIntegralOp <tutorial-boundaryintegralop>`: generic boundary integral method
2. :ref:`SlenderElemList <tutorial-slenderelemlist>`

High Performance Computing (HPC)
--------------------------------

1. :ref:`Comm <tutorial-comm>`: wrapper for MPI
2. :ref:`Vec <tutorial-vec>`: SIMD vectorization class
3. :ref:`OpenMP utilities <ompUtils_hpp>`: merge-sort, scan
4. :ref:`Profile <tutorial-profile>`

Miscellaneous
-------------

1. :ref:`Iterator, ConstIterator <iterator_hpp>`, :ref:`StaticArray <static-array_hpp>`
2. :ref:`MemoryManager <mem_mgr_hpp>`
3. :ref:`Stacktrace utility <stacktrace_h>`
4. :ref:`VTUData <tutorial-vtudata>`: write unstructured VTK files
5. :ref:`QuadReal, basic math functions, constants <math_utils_hpp>`

.. 6. :ref:`GEMM, SVD (unoptimized) <mat_utils_hpp>`

Legacy (Unmaintained)
---------------------

1. :ref:`Boundary quadrature <boundary_quadrature_hpp>`: generic boundary integral method based on quad-patches and hedgehog quadrature
2. :ref:`ChebBasis <cheb_utils_hpp>`: general-dimension tensor product Chebyshev basis
