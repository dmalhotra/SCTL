// Scientific Computing Template Library

#ifndef _SCTL_HPP_
#define _SCTL_HPP_

#include <sctl/common.hpp>

// Import PVFMM preprocessor macro definitions
#ifdef SCTL_HAVE_PVFMM
#  ifndef SCTL_HAVE_MPI
#    define SCTL_HAVE_MPI
#  endif
#  include "pvfmm_config.h"
#  if defined(PVFMM_QUAD_T) && !defined(SCTL_QUAD_T)
#    define SCTL_QUAD_T PVFMM_QUAD_T
#  endif
#endif

// Math utilities
#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(math_utils.txx)

// Boundary Integrals
#include SCTL_INCLUDE(boundary_integral.hpp)
#include SCTL_INCLUDE(boundary_integral.txx)
#include SCTL_INCLUDE(slender_element.hpp)
#include SCTL_INCLUDE(slender_element.txx)
#include SCTL_INCLUDE(quadrule.hpp)
#include SCTL_INCLUDE(quadrule.txx)
#include SCTL_INCLUDE(lagrange-interp.hpp)
#include SCTL_INCLUDE(lagrange-interp.txx)

// ODE solver
#include SCTL_INCLUDE(ode-solver.hpp)
#include SCTL_INCLUDE(ode-solver.txx)

// Tensor
#include SCTL_INCLUDE(tensor.hpp)
#include SCTL_INCLUDE(tensor.txx)

// Tree
#include SCTL_INCLUDE(tree.hpp)
#include SCTL_INCLUDE(tree.txx)
#include SCTL_INCLUDE(vtudata.hpp)
#include SCTL_INCLUDE(vtudata.txx)

// MPI Wrapper
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(comm.txx)

// Memory Manager, Iterators
#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(mem_mgr.txx)
#include SCTL_INCLUDE(iterator.hpp)
#include SCTL_INCLUDE(iterator.txx)
#include SCTL_INCLUDE(static-array.hpp)
#include SCTL_INCLUDE(static-array.txx)

// Vector
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(vector.txx)

// Matrix, Permutation operators
#include SCTL_INCLUDE(permutation.hpp)
#include SCTL_INCLUDE(permutation.txx)
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(matrix.txx)
#include SCTL_INCLUDE(mat_utils.hpp)
#include SCTL_INCLUDE(mat_utils.txx)
#include SCTL_INCLUDE(blas.h)
#include SCTL_INCLUDE(lapack.h)

// Template vector intrinsics (new)
#include SCTL_INCLUDE(vec.hpp)
#include SCTL_INCLUDE(vec.txx)
#include SCTL_INCLUDE(vec-test.hpp)
#include SCTL_INCLUDE(vec-test.hpp)
#include SCTL_INCLUDE(intrin-wrapper.hpp)

// OpenMP merge-sort and scan
#include SCTL_INCLUDE(ompUtils.hpp)
#include SCTL_INCLUDE(ompUtils.txx)

// Linear solver
#include SCTL_INCLUDE(lin-solve.hpp)
#include SCTL_INCLUDE(lin-solve.txx)

// Chebyshev basis
#include SCTL_INCLUDE(cheb_utils.hpp)

// Morton
#include SCTL_INCLUDE(morton.hpp)
#include SCTL_INCLUDE(morton.txx)

// Spherical Harmonics
#include SCTL_INCLUDE(sph_harm.hpp)
#include SCTL_INCLUDE(sph_harm.txx)

#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(fft_wrapper.txx)
#include SCTL_INCLUDE(complex.hpp)
#include SCTL_INCLUDE(complex.txx)

// Profiler
#include SCTL_INCLUDE(profile.hpp)
#include SCTL_INCLUDE(profile.txx)

// Print stack trace
#include SCTL_INCLUDE(stacktrace.h)

// Boundary quadrature, Kernel functions
#include SCTL_INCLUDE(generic-kernel.hpp)
#include SCTL_INCLUDE(generic-kernel.txx)
#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(boundary_quadrature.hpp)

// FMM wrapper
#include SCTL_INCLUDE(fmm-wrapper.hpp)
#include SCTL_INCLUDE(fmm-wrapper.txx)

#endif //_SCTL_HPP_
