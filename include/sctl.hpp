// Scientific Computing Template Library

#ifndef _SCTL_HPP_
#define _SCTL_HPP_

#define SCTL_NAMESPACE sctl

// Profiling parameters
#ifndef SCTL_PROFILE
#define SCTL_PROFILE -1 // Granularity level
#endif

// Parameters for memory manager
#define SCTL_MEM_ALIGN 64
#ifndef SCTL_GLOBAL_MEM_BUFF
#define SCTL_GLOBAL_MEM_BUFF 1024LL * 0LL  // in MB
#endif

#define SCTL_QUOTEME(x) SCTL_QUOTEME_1(x)
#define SCTL_QUOTEME_1(x) #x
#define SCTL_INCLUDE(x) SCTL_QUOTEME(SCTL_NAMESPACE/x)

// Import PVFMM preprocessor macro definitions
#ifdef SCTL_HAVE_PVFMM
#ifndef SCTL_HAVE_MPI
#define SCTL_HAVE_MPI
#endif
#include "pvfmm_config.h"
#ifndef SCTL_QUAD_T
#define SCTL_QUAD_T PVFMM_QUAD_T
#endif
#endif

// FMM wrapper
#include SCTL_INCLUDE(fmm-wrapper.hpp)

// Boundary Integrals
#include SCTL_INCLUDE(boundary_integral.hpp)
#include SCTL_INCLUDE(slender_element.hpp)
#include SCTL_INCLUDE(quadrule.hpp)

// ODE solver
#include SCTL_INCLUDE(ode-solver.hpp)

// Tensor
#include SCTL_INCLUDE(tensor.hpp)

// Tree
#include SCTL_INCLUDE(tree.hpp)
#include SCTL_INCLUDE(vtudata.hpp)

// MPI Wrapper
#include SCTL_INCLUDE(comm.hpp)

// Memory Manager, Iterators
#include SCTL_INCLUDE(mem_mgr.hpp)

// Vector
#include SCTL_INCLUDE(vector.hpp)

// Matrix, Permutation operators
#include SCTL_INCLUDE(matrix.hpp)

// Template vector intrinsics (new)
#include SCTL_INCLUDE(vec.hpp)
#include SCTL_INCLUDE(vec-test.hpp)

// OpenMP merge-sort and scan
#include SCTL_INCLUDE(ompUtils.hpp)

// Parallel solver
#include SCTL_INCLUDE(parallel_solver.hpp)

// Chebyshev basis
#include SCTL_INCLUDE(cheb_utils.hpp)

// Morton
#include SCTL_INCLUDE(morton.hpp)

// Spherical Harmonics
#include SCTL_INCLUDE(sph_harm.hpp)

#include SCTL_INCLUDE(fft_wrapper.hpp)

#include SCTL_INCLUDE(legendre_rule.hpp)

// Profiler
#include SCTL_INCLUDE(profile.hpp)

// Print stack trace
#include SCTL_INCLUDE(stacktrace.h)
const int sgh = SCTL_NAMESPACE::SetSigHandler(); // Set signal handler

// Boundary quadrature, Kernel functions
#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(boundary_quadrature.hpp)

// Math utilities
#include SCTL_INCLUDE(math_utils.hpp)

#endif //_SCTL_HPP_
