#ifndef _SCTL_COMMON_HPP_
#define _SCTL_COMMON_HPP_

#include <stdio.h>      // for NULL, stderr
#include <stdlib.h>     // for abort
#include <cstdint>      // for int64_t
#include <iostream>     // for char_traits, basic_ostream, operator<<, cerr
#include <type_traits>  // for underlying_type_t (Periodicity helpers)

#ifndef SCTL_DATA_PATH
#  define SCTL_DATA_PATH ./data/
#endif

#ifdef _OPENMP
#include <omp.h>
#define SCTL_GET_WTIME()       (omp_get_wtime())
#define SCTL_GET_NUM_THREADS() (omp_get_num_threads())
#define SCTL_GET_MAX_THREADS() (omp_get_max_threads())
#define SCTL_GET_THREAD_NUM()  (omp_get_thread_num())
#define SCTL_IN_PARALLEL()     (omp_in_parallel())
#else
#include <chrono>
#define SCTL_GET_WTIME() \
    (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count())
#define SCTL_GET_NUM_THREADS() (1)
#define SCTL_GET_MAX_THREADS() (1)
#define SCTL_GET_THREAD_NUM()  (0)
#define SCTL_IN_PARALLEL()     (0)
#endif

//#ifndef SCTL_NAMESPACE
//#define SCTL_NAMESPACE sctl
//#endif
#define SCTL_QUOTEME(x) SCTL_QUOTEME_1(x)
#define SCTL_QUOTEME_1(x) #x
//#define SCTL_INCLUDE(x) SCTL_QUOTEME(SCTL_NAMESPACE/x)

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

// Per-axis periodicity bitmask (bit d = axis d). Widen the underlying type for more than 8 axes.
enum class Periodicity : uint8_t {
  NONE = 0,
  X = 1u << 0,
  Y = 1u << 1,
  Z = 1u << 2,
  XY = X | Y,
  XYZ = X | Y | Z
};

using PeriodicityT = std::underlying_type_t<Periodicity>;
constexpr Integer PERIODICITY_MAX_DIM = static_cast<Integer>(sizeof(PeriodicityT) * 8);

constexpr Periodicity operator|(Periodicity a, Periodicity b) {
  return static_cast<Periodicity>(static_cast<PeriodicityT>(a) | static_cast<PeriodicityT>(b));
}

// True iff axis dim is periodic in p.
constexpr bool is_periodic(Periodicity p, Integer dim) {
  return ((static_cast<PeriodicityT>(p) >> dim) & PeriodicityT(1)) != 0;
}

// Mask with axes 0..dim-1 periodic.
constexpr Periodicity all_periodic(Integer dim) {
  return static_cast<Periodicity>(dim >= PERIODICITY_MAX_DIM ? static_cast<PeriodicityT>(~PeriodicityT(0))
                                                             : static_cast<PeriodicityT>((PeriodicityT(1) << dim) - 1));
}
}

#define SCTL_WARN(msg)                                         \
  do {                                                          \
    std::cerr << "\n\033[1;31mSCTL Warning:\033[0m " << msg << '\n'; \
  } while (0)

#define SCTL_ERROR(msg)                                      \
  do {                                                        \
    std::cerr << "\n\033[1;31mSCTL Error:\033[0m " << msg << '\n'; \
    abort();                                                  \
  } while (0)

#define SCTL_ASSERT_MSG(cond, msg) \
  do {                              \
    if (!(cond)) SCTL_ERROR(msg);  \
  } while (0)

#define SCTL_ASSERT(cond)                                                                                      \
  do {                                                                                                          \
    if (!(cond)) {                                                                                              \
      std::cerr << '\n' << __FILE__ << ':' << __LINE__ << ": " << __PRETTY_FUNCTION__                          \
                << ": SCTL Assertion `" << #cond << "' failed.\n";                                              \
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
