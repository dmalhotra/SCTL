.. _sctl_documentation:


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

`SCTL <https://github.com/dmalhotra/SCTL>`_ is a header-only C++ library providing various functionalities for scientific computing. This documentation outlines these functionalities and provides a guide to getting started.

Requirements
------------

The only requirement to use SCTL is a working C++11 compliant compiler with OpenMP 4.0 support. It has been tested with GCC-9 and newer.

Getting Started
---------------

.. note::

    SCTL requires a C++11 compliant compiler with OpenMP 4.0 support.

To get started, download the latest version of SCTL from the `SCTL GitHub <https://github.com/dmalhotra/SCTL>`_.

.. code-block:: bash

   git clone https://github.com/dmalhotra/SCTL.git

Since SCTL is a header-only library, it does not require compilation or installation. Simply include the ``sctl.hpp`` header file in your C++ project and start using the provided classes and functions.

.. code-block:: cpp

   #include <sctl.hpp>

Ensure the compiler can locate the header file by providing the path to ``SCTL_ROOT/include`` using the flag ``-I ${SCTL_ROOT}/include``.

Optional Dependencies
---------------------

The following libraries can be optionally used when available. If not available, SCTL uses its own implementation which may be slower.

- **BLAS**: Enable by defining ``SCTL_HAVE_BLAS``.
- **LAPACK**: Enable by defining ``SCTL_HAVE_LAPACK``.
- **libmvec**: Enable by defining ``SCTL_HAVE_LIBMVEC``.
- **Intel SVML**: Enable by defining ``SCTL_HAVE_SVML``.
- **MPI**: Enable by defining ``SCTL_HAVE_MPI`` (see :ref:`Comm <comm_hpp>`).
- `FFTW <https://www.fftw.org>`_: Enable double precision by defining ``SCTL_HAVE_FFTW``, single precision by defining ``SCTL_HAVE_FFTWF``, or long double precision by defining ``SCTL_HAVE_FFTWL`` (see :ref:`FFT <fft_wrapper_hpp>`).
- `PVFMM <http://pvfmm.org>`_: Enable by defining ``SCTL_HAVE_PVFMM`` (requires MPI, see :ref:`ParticleFMM <fmm-wrapper_hpp>`).

To enable support for any of these libraries, define the corresponding flag during compilation. For example, to enable MPI support, use ``-DSCTL_HAVE_MPI``.

Optional Compiler Flags
-----------------------

The following compiler flags can be used to enable or disable specific features in SCTL:

- ``-DSCTL_MEMDEBUG``: Enable memory debugging (:ref:`iterator.hpp <iterator_hpp>`, :ref:`static-array.hpp <static-array_hpp>`).
- ``-DSCTL_GLOBAL_MEM_BUFF=<size in MB>``: Use a :ref:`global memory buffer <mem_mgr_hpp>` for allocations.
- ``-DSCTL_PROFILE=<level>``: Enable :ref:`profiling <profile_hpp>`.
- ``-DSCTL_VERBOSE``: Enable verbose :ref:`profiling <profile_hpp>` output.
- ``-DSCTL_SIG_HANDLER``: Enable :ref:`stack trace <stacktrace_h>`.
- ``-DSCTL_QUAD_T``: Enable support for :ref:`quad-precision type <math_utils_hpp>`.

Features and Capabilities
-------------------------

The following list outlines the primary features and capabilities provided by the library, along with references to detailed tutorials and documentation for each component:

- **Basic Data Structures**:
  Fundamental classes for storing and manipulating data.

  - :ref:`Vector <tutorial-vector>`: Dynamically allocated linear array.
  - :ref:`Matrix <tutorial-matrix>`, :ref:`Permutation <tutorial-permutation>`: Dynamically allocated 2D array for matrix operations.
  - :ref:`Tensor <tutorial-tensor>`: Statically allocated multi-dimensional array.

..

- **Numerical Solvers and Algorithms**:
  Methods for solving equations, performing interpolations, numerical integration, and partitioning data.

  - :ref:`SDC (Spectral Deferred Correction) <tutorial-sdc>`: High-order solver for ordinary differential equations.
  - :ref:`GMRES solver, Krylov preconditioner <tutorial-gmres>`: Distributed memory GMRES solver.
  - :ref:`LagrangeInterp <tutorial-lagrange-interp>`: Polynomial interpolation and differentiation.
  - :ref:`ChebQuadRule, LegQuadRule <quadrule_hpp>`: Clenshaw-Curtis and Gauss-Legendre quadrature rules.
  - :ref:`InterpQuadRule <tutorial-interp-quadrule>`: Generating special quadrature rules.
  - :ref:`Tree, PtTree <tutorial-tree>`, :ref:`Morton <morton_hpp>`: Morton order based N-dimensional parallel tree structure.

..

- **Spectral Methods**:
  Methods for spectral representations and transformations.

  - :ref:`FFT <tutorial-fft>`: Wrapper for FFTW to perform fast Fourier transforms.
  - :ref:`SphericalHarmonics <sph_harm_hpp>`: Computing spherical harmonics.

..

- **Boundary Integral Methods**:
  Techniques for solving partial differential equations using boundary integral representations.

  - :ref:`BoundaryIntegralOp <tutorial-boundaryintegralop>`: Generic class for instantiating layer-potential operators.
  - :ref:`Kernel functions <kernel_functions_hpp>`: Contains a variety of kernel functions for integral equations.
  - :ref:`ParticleFMM <tutorial-fmm>`: Integration with PVFMM for particle N-body calculations.

..

- **High Performance Computing (HPC)**:
  Tools for parallel and distributed computing.

  - :ref:`Comm <tutorial-comm>`: Wrapper for MPI to facilitate parallel computing.
  - :ref:`Vec <tutorial-vec>`: SIMD vectorization class for optimized computations.
  - :ref:`OpenMP utilities <ompUtils_hpp>`: Parallel algorithms such as merge-sort and scan using OpenMP.
  - :ref:`Profile <tutorial-profile>`: Tools for profiling and performance analysis.

..

- **Utilities**:
  Miscellaneous utilities for data handling, debugging, and visualization.

  - :ref:`VTUData <tutorial-vtudata>`: Writes unstructured VTK files for data visualization.
  - :ref:`QuadReal, basic math functions <math_utils_hpp>`: Quad-precision type and essential mathematical functions.
  - :ref:`Iterator, ConstIterator <iterator_hpp>`, :ref:`StaticArray <static-array_hpp>`: Iterator and static array utilities.
  - :ref:`MemoryManager <mem_mgr_hpp>`: Aligned memory allocation and deallocation.
  - :ref:`Stacktrace utility <stacktrace_h>`: Prints stack traces for debugging.

..  - :ref:`GEMM, SVD (unoptimized) <mat_utils_hpp>`: Provides basic implementations of GEMM and SVD operations.

..

- **Legacy (Unmaintained)**:
  Older functionalities that are no longer actively maintained.

  - :ref:`Boundary quadrature <boundary_quadrature_hpp>`: Boundary integrals on quad-patches using hedgehog quadrature.
  - :ref:`ChebBasis <cheb_utils_hpp>`: Tensor product Chebyshev basis for general-dimension computations.



.. toctree::
   :hidden:
   :maxdepth: 1

   Introduction <self>
   tutorial/index

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 1

   doxygen/index

