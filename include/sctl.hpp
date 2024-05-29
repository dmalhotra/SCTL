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
#include "sctl/math_utils.hpp"
#include "sctl/math_utils.txx"

// Boundary Integrals
#include "sctl/boundary_integral.hpp"
#include "sctl/boundary_integral.txx"
#include "sctl/slender_element.hpp"
#include "sctl/slender_element.txx"
#include "sctl/quadrule.hpp"
#include "sctl/quadrule.txx"
#include "sctl/lagrange-interp.hpp"
#include "sctl/lagrange-interp.txx"

// ODE solver
#include "sctl/ode-solver.hpp"
#include "sctl/ode-solver.txx"

// Tensor
#include "sctl/tensor.hpp"
#include "sctl/tensor.txx"

// Tree
#include "sctl/tree.hpp"
#include "sctl/tree.txx"
#include "sctl/vtudata.hpp"
#include "sctl/vtudata.txx"

// MPI Wrapper
#include "sctl/comm.hpp"
#include "sctl/comm.txx"

// Memory Manager, Iterators
#include "sctl/mem_mgr.hpp"
#include "sctl/mem_mgr.txx"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/static-array.hpp"
#include "sctl/static-array.txx"

// Vector
#include "sctl/vector.hpp"
#include "sctl/vector.txx"

// Matrix, Permutation operators
#include "sctl/permutation.hpp"
#include "sctl/permutation.txx"
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"
#include "sctl/mat_utils.hpp"
#include "sctl/mat_utils.txx"
#include "sctl/blas.h"
#include "sctl/lapack.h"

// Template vector intrinsics (new)
#include "sctl/vec.hpp"
#include "sctl/vec.txx"
#include "sctl/vec-test.hpp"
#include "sctl/vec-test.hpp"
#include "sctl/intrin-wrapper.hpp"

// OpenMP merge-sort and scan
#include "sctl/ompUtils.hpp"
#include "sctl/ompUtils.txx"

// Linear solver
#include "sctl/lin-solve.hpp"
#include "sctl/lin-solve.txx"

// Chebyshev basis
#include "sctl/cheb_utils.hpp"

// Morton
#include "sctl/morton.hpp"
#include "sctl/morton.txx"

// Spherical Harmonics
#include "sctl/sph_harm.hpp"
#include "sctl/sph_harm.txx"

#include "sctl/fft_wrapper.hpp"
#include "sctl/fft_wrapper.txx"
#include "sctl/complex.hpp"
#include "sctl/complex.txx"

// Profiler
#include "sctl/profile.hpp"
#include "sctl/profile.txx"

// Print stack trace
#include "sctl/stacktrace.h"

// Boundary quadrature, Kernel functions
#include "sctl/generic-kernel.hpp"
#include "sctl/generic-kernel.txx"
#include "sctl/kernel_functions.hpp"
#include "sctl/boundary_quadrature.hpp"

// FMM wrapper
#include "sctl/fmm-wrapper.hpp"
#include "sctl/fmm-wrapper.txx"

#endif //_SCTL_HPP_
