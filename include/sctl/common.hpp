#ifndef _SCTL_COMMON_HPP_
#define _SCTL_COMMON_HPP_

#include <stdio.h>   // for NULL, stderr
#include <stdlib.h>  // for abort
#include <cstdint>   // for int64_t
#include <iostream>  // for char_traits, basic_ostream, operator<<, cerr

#ifndef SCTL_DATA_PATH
#  define SCTL_DATA_PATH ./data/
#endif

//#ifndef SCTL_NAMESPACE
//#define SCTL_NAMESPACE sctl
//#endif
#define SCTL_QUOTEME(x) SCTL_QUOTEME_1(x)
#define SCTL_QUOTEME_1(x) #x
//#define SCTL_INCLUDE(x) SCTL_QUOTEME(SCTL_NAMESPACE/x)

// Profiling parameters
#ifndef SCTL_PROFILE
#define SCTL_PROFILE -1 // Granularity level
#endif

#if defined(__AVX512__) || defined(__AVX512F__)
  #define SCTL_ALIGN_BYTES 64
#elif defined(__AVX__)
  #define SCTL_ALIGN_BYTES 32
#elif defined(__SSE__) || defined(__ARM_NEON)
  #define SCTL_ALIGN_BYTES 16
#else
  #define SCTL_ALIGN_BYTES 8
#endif

// Parameters for memory manager
#ifndef SCTL_MEM_ALIGN
#define SCTL_MEM_ALIGN (64 > SCTL_ALIGN_BYTES ? 64 : SCTL_ALIGN_BYTES)
#endif
#ifndef SCTL_GLOBAL_MEM_BUFF
#define SCTL_GLOBAL_MEM_BUFF 1024LL * 0LL  // in MB
#endif

namespace sctl {
typedef long Integer;  // bounded numbers < 32k
typedef int64_t Long;  // problem size
}

#define SCTL_WARN(msg)                                         \
  do {                                                          \
    std::cerr << "\n\033[1;31mWarning:\033[0m " << msg << '\n'; \
  } while (0)

#define SCTL_ERROR(msg)                                      \
  do {                                                        \
    std::cerr << "\n\033[1;31mError:\033[0m " << msg << '\n'; \
    abort();                                                  \
  } while (0)

#define SCTL_ASSERT_MSG(cond, msg) \
  do {                              \
    if (!(cond)) SCTL_ERROR(msg);  \
  } while (0)

#define SCTL_ASSERT(cond)                                                                                      \
  do {                                                                                                          \
    if (!(cond)) {                                                                                              \
      fprintf(stderr, "\n%s:%d: %s: Assertion `%s' failed.\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond); \
      abort();                                                                                                  \
    }                                                                                                           \
  } while (0)

#define SCTL_UNUSED(x) (void)(x)  // to ignore unused variable warning.

namespace sctl {
#ifdef SCTL_MEMDEBUG
template <class ValueType> class ConstIterator;
template <class ValueType> class Iterator;
template <class ValueType, Long DIM> class StaticArray;
#else
template <typename ValueType> using Iterator = ValueType*;
template <typename ValueType> using ConstIterator = const ValueType*;
template <typename ValueType, Long DIM> using StaticArray = ValueType[DIM];
#endif
}

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

#endif // _SCTL_COMMON_HPP_
