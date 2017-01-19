#ifndef _PVFMM_HPP_
#define _PVFMM_HPP_

// Have MPI
//#define PVFMM_HAVE_MPI

// Disable assert checks.
//#ifndef NDEBUG
//#define NDEBUG
//#endif

// Parameters for memory manager
#define PVFMM_MEM_ALIGN 64
#define PVFMM_GLOBAL_MEM_BUFF 1024LL * 20LL  // in MB
#ifndef NDEBUG
#define PVFMM_MEMDEBUG // Enable memory checks.
#endif

// Profiling parameters
#define PVFMM_PROFILE 5 // Granularity level
#define PVFMM_VERBOSE



// MPI Wrapper
#include <pvfmm/comm.hpp>

// Memory Manager, Iterators
#include <pvfmm/mem_mgr.hpp>

// Vector
#include <pvfmm/vector.hpp>

// Matrix, Permutation operators
#include <pvfmm/matrix.hpp>

// Template vector intrinsics
#include <pvfmm/intrin_wrapper.hpp>

// OpenMP merge-sort and scan
#include <pvfmm/ompUtils.hpp>

// Profiler
#include <pvfmm/profile.hpp>

// Print stack trace
#include <pvfmm/stacktrace.h>
const int sgh = pvfmm::SetSigHandler(); // Set signal handler

#endif //_PVFMM_HPP_
