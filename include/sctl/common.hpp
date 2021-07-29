#ifndef _SCTL_COMMON_HPP_
#define _SCTL_COMMON_HPP_

// Define NULL
#ifndef NULL
#define NULL 0
#endif

#include <cstddef>
#include <cstdint>
namespace SCTL_NAMESPACE {
typedef long Integer;  // bounded numbers < 32k
typedef int64_t Long;  // problem size
}

#include <iostream>

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
#if defined(__ARM_FEATURE_SVE)
  #ifndef SCTL_SVE_SIZE
  #error "Must define SCTL_SVE_SIZE (bit length of SVE register for target hardware) when using Arm SVE instructions."
  #endif
  #define SCTL_ALIGN_BYTES ((SCTL_SVE_SIZE) / 8)
#elif defined(__AVX512__) || defined(__AVX512F__)
  #define SCTL_ALIGN_BYTES 64
#elif defined(__AVX__)
  #define SCTL_ALIGN_BYTES 32
#elif defined(__SSE__)
  #define SCTL_ALIGN_BYTES 16
#else
  #define SCTL_ALIGN_BYTES 8
#endif

namespace SCTL_NAMESPACE {
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

#endif  //_SCTL_COMMON_HPP_
