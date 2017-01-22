#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

// Define NULL
#ifndef NULL
#define NULL 0
#endif

#include <stdint.h>
namespace pvfmm {
typedef long Integer;  // bounded numbers < 32k
typedef int64_t Long;  // problem size
}

#include <iostream>

#define PVFMM_WARN(msg)                                         \
  do {                                                          \
    std::cerr << "\n\033[1;31mWarning:\033[0m " << msg << '\n'; \
  } while (0)

#define PVFMM_ERROR(msg)                                      \
  do {                                                        \
    std::cerr << "\n\033[1;31mError:\033[0m " << msg << '\n'; \
    abort();                                                  \
  } while (0)

#define PVFMM_ASSERT_MSG(cond, msg) \
  do {                              \
    if (!(cond)) PVFMM_ERROR(msg);  \
  } while (0)

#define PVFMM_ASSERT(cond)                                                                                      \
  do {                                                                                                          \
    if (!(cond)) {                                                                                              \
      fprintf(stderr, "\n%s:%d: %s: Assertion `%s' failed.\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond); \
      abort();                                                                                                  \
    }                                                                                                           \
  } while (0)

#define UNUSED(x) (void)(x)  // to ignore unused variable warning.

namespace pvfmm {
#ifdef PVFMM_MEMDEBUG
template <class ValueType> class ConstIterator;
template <class ValueType> class Iterator;
template <class ValueType, Integer DIM> class StaticArray;
#else
template <typename ValueType> using Iterator = ValueType*;
template <typename ValueType> using ConstIterator = const ValueType*;
template <typename ValueType, Integer DIM> using StaticArray = ValueType[DIM];
#endif
}

#include <pvfmm/math_utils.hpp>

#endif  //_PVFMM_COMMON_HPP_
