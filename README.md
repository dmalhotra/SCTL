# SCTL: Scientific Computing Template Library

[![tests](https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml/badge.svg)](https://github.com/dmalhotra/SCTL/actions/workflows/build-test.yml)
[![codecov](https://codecov.io/gh/dmalhotra/SCTL/branch/master/graph/badge.svg?token=UIM2RYMF6D)](https://codecov.io/gh/dmalhotra/SCTL)
[![Stable Version](https://badgen.net/github/tag/dmalhotra/SCTL)](https://github.com/dmalhotra/SCTL/tags)
[![Latest Release](https://img.shields.io/github/v/release/dmalhotra/SCTL?color=%233D9970)](https://github.com/dmalhotra/SCTL/releases)

This is a header-only C++ library that provides several functionalities useful in scientific computing.
These functionalities a outlined below.
The header files contain the class declaration and doxygen style documentation of the interface.
An example/test code is also provided for most classes.


#### Containers:
1) [Vector](include/sctl/vector.hpp)
2) [Matrix, Permutation](include/sctl/matrix.hpp)
3) [Tensor](include/sctl/tensor.hpp)


#### Numerical Methods:
1) [SDC (Spectral Deferred Correction ODE solver)](include/sctl/ode-solver.hpp)
2) [ParallelSolver](include/sctl/parallel_solver.hpp): distributed memory GMRES (wrapper to PETSc when available)
3) [LagrangeInterp](include/sctl/lagrange-interp.hpp)
4) [InterpQuadRule, ChebQuadRule, LegQuadRule](include/sctl/quadrule.hpp): generalized Chebyshev quadrature, Clenshaw-Curtis quadrature, Gauss-Legendre quadrature
5) [SphericalHarmonics](include/sctl/sph_harm.hpp)
6) [Tree, PtTree](include/sctl/tree.hpp), [Morton](include/sctl/morton.hpp): Morton ordering based n-dimensional tree
7) [FFT](include/sctl/fft_wrapper.hpp): wrapper to FFT
8) [FMM](include/sctl/fmm-wrapper.hpp): wrapper to [PVFMM](http://pvfmm.org)
9) [ChebBasis](include/sctl/cheb_utils.hpp): general-dimension tensor product Chebyshev basis (unmaintained)


#### Boundary integral methods:
1) [BoundaryIntegralOp, BoundaryIntegralOp](include/sctl/boundary_integral.hpp): generic boundary integral method
2) [SlenderElemList](include/sctl/slender_element.hpp)
3) [n-body kernel functions](include/sctl/kernel_functions.hpp)
4) [Boundary quadrature](include/sctl/boundary_quadrature.hpp): generic boundary integral method based on quad-patches and hedgehog quadrature (unmaintained)


#### HPC:
1) [Comm](include/sctl/comm.hpp): wrapper for MPI
2) [Vec](include/sctl/vec.hpp): SIMD vectorization class
3) [OpenMP utilities](include/sctl/ompUtils.hpp): merge-sort, scan
4) [Profile](include/sctl/profile.hpp)


#### Misceleneous
1) [MemoryManager, Iterator, ConstIterator](include/sctl/mem_mgr.hpp)
2) [Stacktrace utility](include/sctl/stacktrace.h)
3) [VTUData](include/sctl/vtudata.hpp): write unstructured VTK files
4) [QuadReal, basic math functions, constants](include/sctl/math_utils.hpp)
5) [GEMM, SVD (unoptimized)](include/sctl/mat_utils.hpp)
