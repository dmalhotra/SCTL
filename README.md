# SCTL: Scientific Computing Template Library

[![tests](https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml/badge.svg)](https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml)
[![codecov](https://codecov.io/gh/dmalhotra/SCTL/branch/master/graph/badge.svg?token=UIM2RYMF6D)](https://codecov.io/gh/dmalhotra/SCTL)
[![Stable Version](https://badgen.net/github/tag/dmalhotra/SCTL)](https://github.com/dmalhotra/SCTL/tags)
[![Latest Release](https://img.shields.io/github/v/release/dmalhotra/SCTL?color=%233D9970)](https://github.com/dmalhotra/SCTL/releases)

[SCTL](https://github.com/dmalhotra/SCTL) is a header-only C++ library providing various functionalities for scientific computing. This documentation outlines these functionalities and provides a guide to getting started.

## Requirements

The only requirement to use SCTL is a working C++11 compliant compiler with OpenMP 4.0 support. It has been tested with GCC-9 and newer.

## Getting Started

To get started, download the latest version of SCTL from the [SCTL GitHub](https://github.com/dmalhotra/SCTL).

```bash
git clone https://github.com/dmalhotra/SCTL.git
```

Since SCTL is a header-only library, it does not require compilation or installation. Simply include the `sctl.hpp` header file in your C++ project and start using the provided classes and functions.

```cpp
#include <sctl.hpp>
```

Ensure the compiler can locate the header file by providing the path to `SCTL_ROOT/include` using the flag `-I ${SCTL_ROOT}/include`.

## Optional Dependencies

The following libraries can be optionally used when available. If not available, SCTL uses its own implementation which may be slower.

- **BLAS**: Enable by defining `SCTL_HAVE_BLAS`.
- **LAPACK**: Enable by defining `SCTL_HAVE_LAPACK`.
- **libmvec**: Enable by defining `SCTL_HAVE_LIBMVEC`.
- **Intel SVML**: Enable by defining `SCTL_HAVE_SVML`.
- **MPI**: Enable by defining `SCTL_HAVE_MPI` (see [Comm](include/sctl/comm.hpp)).
- [FFTW](https://www.fftw.org): Enable double precision by defining `SCTL_HAVE_FFTW`, single precision by defining `SCTL_HAVE_FFTWF`, or long double precision by defining `SCTL_HAVE_FFTWL` (see [FFT](include/sctl/fft_wrapper_hpp)).
- [PVFMM](http://pvfmm.org): Enable by defining `SCTL_HAVE_PVFMM` (requires MPI, see [ParticleFMM](include/sctl/fmm-wrapper.hpp)).

To enable support for any of these libraries, define the corresponding flag during compilation. For example, to enable MPI support, use `-DSCTL_HAVE_MPI`.

## Optional Compiler Flags

The following compiler flags can be used to enable or disable specific features in SCTL:

- `-DSCTL_MEMDEBUG`: Enable memory debugging ([iterator.hpp](include/sctl/iterator.hpp), [static-array.hpp](include/sctl/static-array.hpp)).
- `-DSCTL_GLOBAL_MEM_BUFF=<size in MB>`: Use a [global memory buffer](include/sctl/mem_mgr.hpp) for allocations.
- `-DSCTL_PROFILE`: Enable [profiling](include/sctl/profile.hpp).
- `-DSCTL_VERBOSE=<level>`: Enable verbose [profiling](include/sctl/profile.hpp) output.
- `-DSCTL_SIG_HANDLER`: Enable [stack trace](include/sctl/stacktrace.h).
- `-DSCTL_QUAD_T`: Enable support for [quad-precision type](include/sctl/math_utils.hpp).

## Features and Capabilities

The following list outlines the primary features and capabilities provided by the library, along with references to detailed tutorials and documentation for each component:

- **Basic Data Structures**:
  Fundamental classes for storing and manipulating data.

  - [Vector](include/sctl/vector.hpp): Dynamically allocated linear array.
  - [Matrix](include/sctl/matrix.hpp), [Permutation](include/sctl/permutation.hpp): Dynamically allocated 2D array for matrix operations.
  - [Tensor](include/sctl/tensor.hpp): Statically allocated multi-dimensional array.

- **Numerical Solvers and Algorithms**:
  Methods for solving equations, performing interpolations, numerical integration, and partitioning data.

  - [SDC (Spectral Deferred Correction)](include/sctl/ode-solver.hpp): High-order solver for ordinary differential equations.
  - [GMRES solver, Krylov preconditioner](include/sctl/lin-solve.hpp): Distributed memory GMRES solver.
  - [LagrangeInterp](include/sctl/lagrange-interp.hpp): Polynomial interpolation and differentiation.
  - [ChebQuadRule, LegQuadRule](include/sctl/quadrule.hpp): Clenshaw-Curtis and Gauss-Legendre quadrature rules.
  - [InterpQuadRule](include/sctl/quadrule.hpp): Generating special quadrature rules.
  - [Tree, PtTree](include/sctl/tree.hpp), [Morton](include/sctl/morton.hpp): Morton order based N-dimensional parallel tree structure.

- **Spectral Methods**:
  Methods for spectral representations and transformations.

  - [FFT](include/sctl/fft_wrapper.hpp): Wrapper for FFTW to perform fast Fourier transforms.
  - [SphericalHarmonics](include/sctl/sph_harm.hpp): Computing spherical harmonics.

- **Boundary Integral Methods**:
  Techniques for solving partial differential equations using boundary integral representations.

  - [BoundaryIntegralOp](include/sctl/boundary_integral.hpp): Generic class for instantiating layer-potential operators.
  - [Kernel functions](include/sctl/kernel_functions.hpp): Contains a variety of kernel functions for integral equations.
  - [ParticleFMM](include/sctl/fmm-wrapper.hpp): Integration with PVFMM for particle N-body calculations.

- **High Performance Computing (HPC)**:
  Tools for parallel and distributed computing.

  - [Comm](include/sctl/comm.hpp): Wrapper for MPI to facilitate parallel computing.
  - [Vec](include/sctl/vec.hpp): SIMD vectorization class for optimized computations.
  - [OpenMP utilities](include/sctl/ompUtils.hpp): Parallel algorithms such as merge-sort and scan using OpenMP.
  - [Profile](include/sctl/profile.hpp): Tools for profiling and performance analysis.

- **Utilities**:
  Miscellaneous utilities for data handling, debugging, and visualization.

  - [VTUData](include/sctl/vtudata.hpp): Writes unstructured VTK files for data visualization.
  - [QuadReal, basic math functions](include/sctl/math_utils.hpp): Quad-precision type and essential mathematical functions.
  - [Iterator, ConstIterator](include/sctl/iterator.hpp), [StaticArray](include/sctl/static-array.hpp): Iterator and static array utilities.
  - [MemoryManager](include/sctl/mem_mgr.hpp): Aligned memory allocation and deallocation.
  - [Stacktrace utility](include/sctl/stacktrace.h): Prints stack traces for debugging.
  - [GEMM, SVD (unoptimized)](include/sctl/mat_utils.hpp): Provides basic implementations of GEMM and SVD operations.

- **Legacy (Unmaintained)**:
  Older functionalities that are no longer actively maintained.

  - [Boundary quadrature](include/sctl/boundary_quadrature.hpp): Boundary integrals on quad-patches using hedgehog quadrature.
  - [ChebBasis](include/sctl/cheb_utils.hpp): Tensor product Chebyshev basis for general-dimension computations.
