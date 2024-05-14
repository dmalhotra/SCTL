


..  .. autodoxygenindex::
..     :project: SCTL

..   .. autodoxygenindex::
..      :project: SCTL
..      :outline:
..      :no-link:
..      :allow-dot-graphs





..  .. toctree::
..     :maxdepth: 2
..     :caption: Contents:

.. :ref:`genindex`

.. ##.. doxygenclass:: SCTL_NAMESPACE::Profile
.. ##   :members:



.. The header files contain the class declaration and doxygen style documentation of the interface. An example/test code is also provided for most classes.

..  SCTL_MEMDEBUG
..  SCTL_GLOBAL_MEM_BUFF
..  SCTL_PROFILE
..  SCTL_VERBOSE
..  SCTL_SIG_HANDLER
..  SCTL_QUAD_T





.. _sctl_documentation:

SCTL: Scientific Computing Template Library
============================================

.. image:: https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml/badge.svg
   :target: https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml
   :alt: tests

.. image:: https://codecov.io/gh/dmalhotra/SCTL/branch/master/graph/badge.svg?token=UIM2RYMF6D
   :target: https://codecov.io/gh/dmalhotra/SCTL
   :alt: codecov

.. image:: https://badgen.net/github/tag/dmalhotra/SCTL
   :target: https://github.com/dmalhotra/SCTL/tags
   :alt: Stable Version

.. image:: https://img.shields.io/github/v/release/dmalhotra/SCTL?color=%233D9970
   :target: https://github.com/dmalhotra/SCTL/releases
   :alt: Latest Release

`SCTL <https://github.com/dmalhotra/SCTL>`_ is a header-only C++ library that provides several functionalities useful in scientific computing.
These functionalities are outlined in this document.


Getting Started
---------------

.. note::

    SCTL requires a C++11 compliant compiler with OpenMP 4.0 support.

You can download the latest version of SCTL from `SCTL GitHub <https://github.com/dmalhotra/SCTL>`_.

.. code-block:: bash

   git clone https://github.com/dmalhotra/SCTL.git



As a header-only library, SCTL does not require compilation or installation.
Simply include the ``sctl.hpp`` header file in your C++ project and start using the provided classes and functions.

.. code-block:: cpp

    #include <sctl.hpp>

The path to ``SCTL_ROOT/include`` (where the header ``sctl.hpp`` is located) must be provided to the compiler using the flag ``-I ${SCTL_ROOT}/include``.


Dependencies
------------

The only requirement to use SCTL is a working C++11 compliant compiler with OpenMP 4.0 support. It has been tested with GCC-9 and newer.

SCTL can optionally use the following libraries if they are available:

- **MPI**: enable by defining ``SCTL_HAVE_MPI``.

- **BLAS**: enable by defining ``SCTL_HAVE_BLAS``.

- **LAPACK**: enable by defining ``SCTL_HAVE_LAPACK``.

- **FFTW**: enable for double precision by defining ``SCTL_HAVE_FFTW``, single precision by defining ``SCTL_HAVE_FFTWF``, or long double precision by defining ``SCTL_HAVE_FFTWL``.

- **libmvec**: enable by defining ``SCTL_HAVE_LIBMVEC``.

- **Intel SVML**: enable by defining ``SCTL_HAVE_SVML``.

- `PVFMM <http://pvfmm.org>`_: enable by defining ``SCTL_HAVE_PVFMM`` (requires MPI).

To enable support for any of these libraries, define the corresponding flag during compilation. For example, to enable MPI support, use ``-DSCTL_HAVE_MPI``.





Containers
----------

1. :ref:`Vector <tutorial-vector>`
2. :ref:`Matrix <tutorial-matrix>`, :ref:`Permutation <permutation-doc>`
3. :ref:`Tensor <tensor-doc>`

Numerical Methods
------------------

1. :ref:`SDC (Spectral Deferred Correction ODE solver) <tutorial-sdc>`
2. `LinearSolver <include/sctl/lin-solve.hpp>`_: distributed memory GMRES (wrapper to PETSc when available)
3. `LagrangeInterp <include/sctl/lagrange-interp.hpp>`_
4. `InterpQuadRule, ChebQuadRule, LegQuadRule <include/sctl/quadrule.hpp>`_: generalized Chebyshev quadrature, Clenshaw-Curtis quadrature, Gauss-Legendre quadrature
5. `SphericalHarmonics <include/sctl/sph_harm.hpp>`_
6. `Tree, PtTree <include/sctl/tree.hpp>`_, `Morton <include/sctl/morton.hpp>`_: Morton ordering based n-dimensional tree
7. `FFT <include/sctl/fft_wrapper.hpp>`_: wrapper to FFT
8. `FMM <include/sctl/fmm-wrapper.hpp>`_: wrapper to `PVFMM <http://pvfmm.org>`_
9. `ChebBasis <include/sctl/cheb_utils.hpp>`_: general-dimension tensor product Chebyshev basis (unmaintained)

Boundary integral methods
--------------------------

1. `BoundaryIntegralOp <include/sctl/boundary_integral.hpp>`_: generic boundary integral method
2. `SlenderElemList <include/sctl/slender_element.hpp>`_
3. `Kernel functions <include/sctl/kernel_functions.hpp>`_
4. `Boundary quadrature <include/sctl/boundary_quadrature.hpp>`_: generic boundary integral method based on quad-patches and hedgehog quadrature (unmaintained)

HPC
---

1. `Comm <include/sctl/comm.hpp>`_: wrapper for MPI
2. `Vec <include/sctl/vec.hpp>`_: SIMD vectorization class
3. `OpenMP utilities <include/sctl/ompUtils.hpp>`_: merge-sort, scan
4. `Profile <include/sctl/profile.hpp>`_

Miscellaneous
-------------

1. `MemoryManager, Iterator, ConstIterator <include/sctl/mem_mgr.hpp>`_
2. `Stacktrace utility <include/sctl/stacktrace.h>`_
3. :ref:`VTUData <tutorial-vtudata>`: write unstructured VTK files
4. `QuadReal, basic math functions, constants <include/sctl/math_utils.hpp>`_
5. `GEMM, SVD (unoptimized) <include/sctl/mat_utils.hpp>`_

