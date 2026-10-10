#ifndef _SCTL_INTRIN_WRAPPER_HPP_
#define _SCTL_INTRIN_WRAPPER_HPP_

#include <stdint.h>             // for int8_t, int16_t, int32_t, int64_t, uint8_t, ...
#include <cmath>                // for signbit
#include <limits>               // for numeric_limits
#include <tuple>                // for tie
#include <type_traits>          // for is_same
#include <utility>              // for pair

#include "sctl/common.hpp"      // for Integer, sctl, SCTL_ALIGN_B...
#include "sctl/math_utils.hpp"  // for const_pi, QuadReal, cos, exp, sin, sqrt
#include "sctl/math_utils.txx"  // for pow, significant_bits

#if defined(__ARM_NEON)
#  include "sctl/sse2neon.h"
#  define _MM_SHUFFLE2(fp1, fp0) (((fp1) << 1) | (fp0))
#elif defined(__MMX__) || defined(__SSE__) || defined(__SSE2__) || defined(__SSE4_2__) || defined(__AVX__) || defined(__AVX512F__)
#  ifdef _MSC_VER
#    include <intrin.h>
#  else
#    include <x86intrin.h>
#  endif
#endif
#if defined(SCTL_HAVE_LIBMVEC)
  #if defined(SCTL_HAVE_SVML)
    #error "SCTL_HAVE_LIBMVEC defined with mutually exclusive SCTL_HAVE_SVML"
  #endif
  // https://sourceware.org/glibc/wiki/libmvec?action=AttachFile&do=view&target=VectorABI.txt
  extern "C" {
    #ifdef __SSE4_2__
    __m128  _ZGVbN4v_logf(__m128);
    __m128d _ZGVbN2v_log(__m128d);
    __m128  _ZGVbN4vv_powf(__m128, __m128);
    __m128d _ZGVbN2vv_pow(__m128d, __m128d);
    #endif
    #ifdef __AVX__
    __m256 _ZGVcN8v_logf(__m256);
    __m256d _ZGVcN4v_log(__m256d);
    __m256 _ZGVdN8v_logf(__m256);
    __m256d _ZGVdN4v_log(__m256d);
    __m256 _ZGVcN8vv_powf(__m256, __m256);
    __m256d _ZGVcN4vv_pow(__m256d, __m256d);
    __m256 _ZGVdN8vv_powf(__m256, __m256);
    __m256d _ZGVdN4vv_pow(__m256d, __m256d);
    #endif
    #if defined(__AVX512F__)
    __m512 _ZGVeN16v_logf(__m512);
    __m512d _ZGVeN8v_log(__m512d);
    __m512 _ZGVeN16vv_powf(__m512, __m512);
    __m512d _ZGVeN8vv_pow(__m512d, __m512d);
    #endif
  }
#endif

// Code for AMD Zen, where vmaskmovps and vmaskmovpd take about 6 cycles per load and store (1.4-1.8 on Intel);
// defined by -march or -mtune znver1 to znver5, or by the user
#if !defined(SCTL_TUNE_ZEN) && (defined(__tune_znver1__) || defined(__tune_znver2__) || defined(__tune_znver3__) || \
    defined(__tune_znver4__) || defined(__tune_znver5__) || defined(__znver1__) || defined(__znver2__) || \
    defined(__znver3__) || defined(__znver4__) || defined(__znver5__))
#define SCTL_TUNE_ZEN
#endif

// TODO: Replace pointers with iterators


namespace sctl { // Traits

  enum class DataType {
    Integer,
    Real
  };

  template <class ValueType> class TypeTraits {
  };
  template <> class TypeTraits<int8_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int8_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
  template <> class TypeTraits<int16_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int16_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
  template <> class TypeTraits<int32_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int32_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
  template <> class TypeTraits<int64_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int64_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
#ifdef __SIZEOF_INT128__
  template <> class TypeTraits<__int128> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(__int128);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
#endif

  template <> class TypeTraits<float> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(float);
      static constexpr Integer SigBits = 23;
      static constexpr Integer ExpBits = 8;
  };
  template <> class TypeTraits<double> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(double);
      static constexpr Integer SigBits = 52;
      static constexpr Integer ExpBits = 11;
  };
  template <> class TypeTraits<long double> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(long double);
      static constexpr Integer SigBits = significant_bits<long double>();
      static constexpr Integer ExpBits = [] { // 15 for x87, which also stores the leading bit and pads to 128 bits
        Integer b = 0;
        for (long m = 2L * std::numeric_limits<long double>::max_exponent - 1; m > 0; m >>= 1) b++;
        return b;
      }();
  };
#ifdef SCTL_QUAD_T
  template <> class TypeTraits<QuadReal> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(QuadReal);
      static constexpr Integer SigBits = 112;
      static constexpr Integer ExpBits = 15;
  };
#endif

  template <class Real> inline constexpr bool ieee_layout() { // the bits are sign, exponent and fraction, filling the type; false for x87 long double (leading bit stored, padded) and double-double
    return TypeTraits<Real>::SigBits + TypeTraits<Real>::ExpBits + 1 == TypeTraits<Real>::Size * 8;
  }

  template <Integer N> struct IntegerType {};
  template <> struct IntegerType<sizeof( int8_t)> {
    using value =  int8_t;
    using unsigned_value =  uint8_t;
  };
  template <> struct IntegerType<sizeof(int16_t)> {
    using value = int16_t;
    using unsigned_value = uint16_t;
  };
  template <> struct IntegerType<sizeof(int32_t)> {
    using value = int32_t;
    using unsigned_value = uint32_t;
  };
  template <> struct IntegerType<sizeof(int64_t)> {
    using value = int64_t;
    using unsigned_value = uint64_t;
  };
#ifdef __SIZEOF_INT128__
  template <> struct IntegerType<sizeof(__int128)> {
    using value = __int128;
    using unsigned_value = unsigned __int128;
  };
#endif
}

namespace sctl { // Generic

  template <class ValueType, Integer N> struct alignas(sizeof(ValueType)*N>SCTL_ALIGN_BYTES?SCTL_ALIGN_BYTES:sizeof(ValueType)*N) VecData {
    using ScalarType = ValueType;
    static constexpr Integer Size = N;
    static constexpr bool ArrayLanes = true; // the lanes in an array, not in a vector register
    ScalarType v[N];
  };


  template <class VData> inline VData zero_intrin() {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } a_;
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = (typename VData::ScalarType)0;
    return a_.v;
  }
  template <class VData> inline VData set1_intrin(typename VData::ScalarType a) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } a_;
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = a;
    return a_.v;
  }

  template <Integer k, Integer Size, class Data, class T> inline void SetHelper(Data& vec, T x) {
    vec.x[Size-k-1] = x;
  }
  template <Integer k, Integer Size, class Data, class T, class... T2> inline void SetHelper(Data& vec, T x, T2... rest) {
    vec.x[Size-k-1] = x;
    SetHelper<k-1,Size>(vec, rest...);
  }
  template <class VData, class T, class ...T2> inline VData set_intrin(T x, T2 ...args) {
    static_assert(sizeof...(T2) + 1 == VData::Size, "set_intrin requires exactly one value per element");
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    SetHelper<VData::Size-1,VData::Size>(vec, x, args...);
    return vec.v;
  }

  template <class VData> inline VData load1_intrin(typename VData::ScalarType const* p) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    for (Integer i = 0; i < VData::Size; i++) vec.x[i] = p[0];
    return vec.v;
  }
  template <class VData> inline VData loadu_intrin(typename VData::ScalarType const* p) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    for (Integer i = 0; i < VData::Size; i++) vec.x[i] = p[i];
    return vec.v;
  }
  template <class VData> inline VData load_intrin(typename VData::ScalarType const* p) {
    return loadu_intrin<VData>(p);
  }

  template <class VData> inline void storeu_intrin(typename VData::ScalarType* p, VData vec) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i++) p[i] = vec_.x[i];
  }
  template <class VData> inline void store_intrin(typename VData::ScalarType* p, VData vec) {
    storeu_intrin(p,vec);
  }

  template <class VData> inline typename VData::ScalarType extract_intrin(VData vec, Integer i) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    return vec_.x[i];
  }
  template <class VData> inline void insert_intrin(VData& vec, Integer i, typename VData::ScalarType value) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    vec_.x[i] = value;
    vec = vec_.v;
  }

  // Arithmetic operators
  template <class VData> inline VData unary_minus_intrin(const VData& vec) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i++) vec_.x[i] = -vec_.x[i];
    return vec_.v;
  }
  template <class VData> inline VData mul_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] *= b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData div_intrin(const VData& a, const VData& b) {
    using Real = typename VData::ScalarType;
    union U {
      VData v;
      Real x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) {
      if constexpr (!std::is_integral<Real>::value) { // the IEEE 754 quotient by zero, as the vector instructions; undefined in C++
        if (b_.x[i] == 0) {
          const Real q = (a_.x[i] == 0 || !(a_.x[i] == a_.x[i]) ? (Real)NAN : (Real)INFINITY);
          a_.x[i] = (std::signbit((double)a_.x[i]) != std::signbit((double)b_.x[i]) ? -q : q);
          continue;
        }
      }
      a_.x[i] /= b_.x[i];
    }
    return a_.v;
  }
  template <class VData> inline VData add_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] += b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData sub_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] -= b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData rem_intrin(const VData& a, const VData& b) { // remainder of the integer division, a - (a/b)*b
    return sub_intrin(a, mul_intrin(div_intrin(a, b), b));
  }
  template <class VData> inline VData fma_intrin(const VData& a, const VData& b, const VData& c) {
    return add_intrin(mul_intrin(a,b), c);
  }

  // Bitwise operators
  template <class VData> inline VData not_intrin(const VData& vec) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union {
      VData v;
      int8_t x[N];
    } vec_ = {vec};
    for (Integer i = 0; i < N; i++) vec_.x[i] = ~vec_.x[i];
    return vec_.v;
  }
  template <class VData> inline VData and_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] & b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData xor_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] ^ b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData or_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] | b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData andnot_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] & (~b_.x[i]);
    return a_.v;
  }

  // Bitshift
  template <class VData> inline VData bitshiftleft_intrin(const VData& a, const Integer& rhs) {
    static constexpr Integer N = VData::Size;
    using UInt = typename IntegerType<sizeof(typename VData::ScalarType)>::unsigned_value;
    union {
      VData v;
      UInt x[N]; // unsigned: defined for negative values, unlike a signed bit shift in C++17
    } a_ = {a};
    for (Integer i = 0; i < N; i++) a_.x[i] <<= rhs;
    return a_.v;
  }
  template <class VData> inline VData bitshiftright_intrin(const VData& a, const Integer& rhs) {
    static constexpr Integer N = VData::Size;
    using ScalarType = typename VData::ScalarType;
    if constexpr (TypeTraits<ScalarType>::Type == DataType::Integer) { // fills with the sign bit
      union {
        VData v;
        ScalarType x[N];
      } a_ = {a};
      for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] >> rhs;
      return a_.v;
    } else { // the raw bits, filled with zeros, as in the SSE, AVX and AVX-512 paths
      union {
        VData v;
        typename IntegerType<sizeof(ScalarType)>::unsigned_value x[N];
      } a_ = {a};
      for (Integer i = 0; i < N; i++) a_.x[i] >>= rhs;
      return a_.v;
    }
  }
  template <class VData> inline VData bitshiftleft_intrin(const VData& a, const VData& rhs) { // lane i bit shifted by lane i of rhs
    static constexpr Integer N = VData::Size;
    using UInt = typename IntegerType<sizeof(typename VData::ScalarType)>::unsigned_value;
    union {
      VData v;
      UInt x[N];
    } a_ = {a};
    union {
      VData v;
      typename VData::ScalarType x[N];
    } k_ = {rhs};
    for (Integer i = 0; i < N; i++) a_.x[i] <<= k_.x[i];
    return a_.v;
  }
  template <class VData> inline VData bitshiftright_intrin(const VData& a, const VData& rhs) { // lane i bit shifted by lane i of rhs
    static constexpr Integer N = VData::Size;
    union U {
      VData v;
      typename VData::ScalarType x[N];
    };
    U a_ = {a};
    U k_ = {rhs};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] >> k_.x[i];
    return a_.v;
  }

  // Other functions
  template <class VData> inline VData max_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = (b_.x[i] < a_.x[i] ? a_.x[i] : b_.x[i]); // b if either is NaN, as maxps/maxpd
    return a_.v;
  }
  template <class VData> inline VData min_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = (a_.x[i] < b_.x[i] ? a_.x[i] : b_.x[i]);
    return a_.v;
  }

  template <class VData> inline void transpose_intrin(VData (&v)[VData::Size]) {
    static constexpr Integer N = VData::Size;
    typename VData::ScalarType buf[N][N];
    for (Integer i = 0; i < N; i++) storeu_intrin(buf[i], v[i]);
    for (Integer i = 0; i < N; i++) {
      for (Integer j = 0; j < i; j++) {
        const typename VData::ScalarType t = buf[i][j];
        buf[i][j] = buf[j][i];
        buf[j][i] = t;
      }
    }
    for (Integer i = 0; i < N; i++) v[i] = loadu_intrin<VData>(buf[i]);
  }

  template <class VData> inline VData swap_pairs_intrin(const VData& vec) {
    static_assert(VData::Size % 2 == 0, "swap_pairs requires an even number of lanes.");
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i += 2) {
      const typename VData::ScalarType t = vec_.x[i];
      vec_.x[i] = vec_.x[i + 1];
      vec_.x[i + 1] = t;
    }
    return vec_.v;
  }

  // Element k of the result is a[I_k] for 0 <= I_k < N, b[I_k - N] for N <= I_k < 2N, and 0 for I_k = -1. With GCC 12+
  // and clang by __builtin_shufflevector on a vector of the scalar type, mapped by the compiler to the shuffle, permute
  // and blend instructions of the target; else lane by lane.
  template <Integer... I> struct BlendIntrin {
    template <class VData> static inline VData apply(const VData& a, const VData& b) {
      using T = typename VData::ScalarType;
      static constexpr Integer N = VData::Size;
      static_assert(sizeof...(I) == N, "blend and permute take one index per lane.");
      static_assert(((-1 <= I && I < 2 * N) && ...), "a blend index outside [-1, 2N).");
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)
      if constexpr (std::is_arithmetic<T>::value && sizeof(T) <= 8 && (N & (N - 1)) == 0 && sizeof(VData) == N * sizeof(T)) {
        return lanes(a, b, std::make_integer_sequence<Integer, N>());
      }
#endif
      union U {
        VData v;
        T x[N];
      };
      const U a_ = {a};
      const U b_ = {b};
      U r_ = {a};
      static constexpr Integer idx[N] = {I...};
      for (Integer k = 0; k < N; k++) r_.x[k] = (idx[k] < 0 ? (T)0 : (idx[k] < N ? a_.x[idx[k]] : b_.x[idx[k] - N]));
      return r_.v;
    }
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)
    template <class VData, Integer... K> static inline VData lanes(const VData& a, const VData& b, std::integer_sequence<Integer, K...>) {
      using T = typename VData::ScalarType;
      static constexpr Integer N = VData::Size;
      typedef T V __attribute__((vector_size(N * sizeof(T))));
      union U {
        VData v;
        V w;
      };
      const U a_ = {a};
      const U b_ = {b};
      U r_ = {a};
      r_.w = __builtin_shufflevector(a_.w, b_.w, (I < 0 ? -1 : I)...);
      if constexpr (((I < 0) || ...)) r_.w = __builtin_shufflevector(r_.w, V{}, (I < 0 ? N + K : K)...); // the zeros, from a second operand
      return r_.v;
    }
#endif
  };
  template <Integer... I, class VData> inline VData blend_intrin(const VData& a, const VData& b) {
    return BlendIntrin<I...>::apply(a, b);
  }
  template <Integer... I, class VData> inline VData permute_intrin(const VData& a) {
    static_assert(((I < VData::Size) && ...), "a permute index outside [-1, N).");
    return BlendIntrin<I...>::apply(a, a);
  }

  // Conversion operators
  template <class RetType, class ValueType, Integer N> inline RetType reinterpret_intrin(const VecData<ValueType,N>& v) {
    static_assert(sizeof(RetType) == sizeof(VecData<ValueType,N>), "Illegal type cast -- size of types does not match.");
    union {
      VecData<ValueType,N> v;
      RetType r;
    } u = {v};
    return u.r;
  }
  // Lane movement is bit-level, so a type can borrow the transpose of any
  // same-layout type -- integer widths reuse the float/double specializations.
  template <class VData0, class VData> inline void transpose_reinterpret_intrin(VData (&v)[VData::Size]) {
    static constexpr Integer N = VData::Size;
    static_assert(VData0::Size == N && sizeof(VData0) == sizeof(VData), "vector layout must match.");
    VData0 w[N];
    for (Integer i = 0; i < N; i++) w[i] = reinterpret_intrin<VData0>(v[i]);
    transpose_intrin(w);
    for (Integer i = 0; i < N; i++) v[i] = reinterpret_intrin<VData>(w[i]);
  }
  template <class RealVec, class IntVec> inline RealVec convert_int2real_intrin(const IntVec& x) {
    using Real = typename RealVec::ScalarType;
    using Int = typename IntVec::ScalarType;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");
    static_assert(TypeTraits<Int>::Type == DataType::Integer, "Expected integer type!");
    static_assert(sizeof(RealVec) == sizeof(IntVec) && sizeof(Real) == sizeof(Int), "Real and integer types must have same size!");

    if constexpr (!ieee_layout<Real>()) { // one element at a time
      union {
        IntVec v;
        Int x[IntVec::Size];
      } n = {x};
      union {
        RealVec v;
        Real x[RealVec::Size];
      } r;
      for (Integer i = 0; i < RealVec::Size; i++) r.x[i] = (Real)n.x[i];
      return r.v;
    } else {
      static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
      union {
        Int Cint = (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
        Real Creal;
      };
      IntVec l(add_intrin(x, set1_intrin<IntVec>(Cint)));
      return sub_intrin(reinterpret_intrin<RealVec>(l), set1_intrin<RealVec>(Creal));
    }
  }
  template <class IntVec, class RealVec> inline IntVec lrint_intrin(const RealVec& x) { // as rint_intrin, to Int, for |x| < 2^(SigBits-1)
    using Int = typename IntVec::ScalarType;
    using Real = typename RealVec::ScalarType;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");
    static_assert(TypeTraits<Int>::Type == DataType::Integer, "Expected integer type!");
    static_assert(sizeof(RealVec) == sizeof(IntVec) && sizeof(Real) == sizeof(Int), "Real and integer types must have same size!");

    if constexpr (!ieee_layout<Real>()) { // one element at a time
      union {
        RealVec v;
        Real x[RealVec::Size];
      } r = {x};
      union {
        IntVec v;
        Int x[IntVec::Size];
      } n;
      for (Integer i = 0; i < RealVec::Size; i++) n.x[i] = (Int)std::rint(r.x[i]);
      return n.v;
    } else {
      static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
      union {
        Int Cint = (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
        Real Creal;
      };
      RealVec d(add_intrin(x, set1_intrin<RealVec>(Creal)));
      return sub_intrin(reinterpret_intrin<IntVec>(d), set1_intrin<IntVec>(Cint));
    }
  }
  template <class VData> inline VData rint_intrin(const VData& x) { // nearest integer, halves to even, as std::rint; generic: |x| < 2^(SigBits-1), zero results are +0
    using Real = typename VData::ScalarType;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");

    static constexpr Real Creal = (Real)1.5 * pow<TypeTraits<Real>::SigBits,Real>((Real)2);
    VData Vreal(set1_intrin<VData>(Creal));
    return sub_intrin(add_intrin(x, Vreal), Vreal);
  }
  template <class VDataTo, class VData> inline VDataTo convert_intrin(const VData& a) { // static_cast of each lane
    static_assert(VDataTo::Size == VData::Size, "Conversion requires the same number of lanes.");
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } a_ = {a};
    union {
      VDataTo v;
      typename VDataTo::ScalarType x[VData::Size];
    } b_;
    for (Integer i = 0; i < VData::Size; i++) b_.x[i] = (typename VDataTo::ScalarType)a_.x[i];
    return b_.v;
  }

  template <class VData> inline VecData<typename VData::ScalarType,VData::Size/2> get_low_intrin(const VData& a) {
    union {
      VData v;
      VecData<typename VData::ScalarType,VData::Size/2> h[2];
    } u = {a};
    return u.h[0];
  }
  template <class VData> inline VecData<typename VData::ScalarType,VData::Size/2> get_high_intrin(const VData& a) {
    union {
      VData v;
      VecData<typename VData::ScalarType,VData::Size/2> h[2];
    } u = {a};
    return u.h[1];
  }
  template <class VData> inline VecData<typename VData::ScalarType,VData::Size*2> concat_intrin(const VData& lo, const VData& hi) {
    union {
      VData h[2];
      VecData<typename VData::ScalarType,VData::Size*2> v;
    } u = {{lo, hi}};
    return u.v;
  }

  // Reductions: the two halves are combined lane by lane until one lane is left,
  // so the order of the operations is the same on every instruction set.
  template <class VData> inline typename VData::ScalarType reduce_add_intrin(const VData& a) {
    if constexpr (VData::Size == 1) {
      return extract_intrin(a, 0);
    } else {
      return reduce_add_intrin(add_intrin(get_low_intrin(a), get_high_intrin(a)));
    }
  }
  template <class VData> inline typename VData::ScalarType reduce_min_intrin(const VData& a) {
    if constexpr (VData::Size == 1) {
      return extract_intrin(a, 0);
    } else {
      return reduce_min_intrin(min_intrin(get_low_intrin(a), get_high_intrin(a)));
    }
  }
  template <class VData> inline typename VData::ScalarType reduce_max_intrin(const VData& a) {
    if constexpr (VData::Size == 1) {
      return extract_intrin(a, 0);
    } else {
      return reduce_max_intrin(max_intrin(get_low_intrin(a), get_high_intrin(a)));
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Storage of the lanes of a Mask<VData>, each all ones or all zeros: VData itself where it is a vector
   * register, and integers of the lane size where VData holds an array, whose element copies need not keep
   * all bytes (a copy of an x87 long double keeps 10 of its 16).
   */
  template <class VData, class = void> struct MaskLanesType {
    using type = VData;
  };
  template <class VData> struct MaskLanesType<VData, std::enable_if_t<VData::ArrayLanes>> {
    using type = VecData<typename IntegerType<sizeof(typename VData::ScalarType)>::value, VData::Size>;
  };
  template <class VData> using MaskLanes = typename MaskLanesType<VData>::type;

  template <class VData> struct Mask : public MaskLanes<VData> {
    using VDataType = VData;
    using ScalarType = typename VData::ScalarType;
    static constexpr Integer Size = VData::Size;

    static inline Mask Zero() {
      return Mask<VData>(zero_intrin<VDataType>());
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const VData& v) : MaskLanes<VData>(lanes(v)) {}

   private:
    static inline MaskLanes<VData> lanes(const VData& v) { // integer lanes from the first bytes of each lane of v, which every copy keeps
      if constexpr (std::is_same<MaskLanes<VData>, VData>::value) {
        return v;
      } else {
        using Int = typename MaskLanes<VData>::ScalarType;
        constexpr Integer K = (sizeof(Int) < 8 ? sizeof(Int) : 8); // bytes read per lane
        union {
          VData v;
          typename IntegerType<K>::value w[VData::Size * (sizeof(Int) / K)];
        } v_ = {v};
        MaskLanes<VData> m;
        for (Integer i = 0; i < VData::Size; i++) m.v[i] = (v_.w[i * (sizeof(Int) / K)] ? ~(Int)0 : (Int)0);
        return m;
      }
    }
  };

  template <class RetType, class VData> inline RetType reinterpret_mask(const Mask<VData>& v) {
    static_assert(sizeof(RetType) == sizeof(Mask<VData>), "Illegal type cast -- size of types does not match.");
    union {
      Mask<VData> v;
      RetType r;
    } u = {v};
    return u.r;
  }

  // Bitwise operators
  template <class VData> inline Mask<VData> operator~(const Mask<VData>& vec) {
    return Mask<VData>(not_intrin(vec));
  }
  template <class VData> inline Mask<VData> operator&(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(and_intrin(a,b));
  }
  template <class VData> inline Mask<VData> operator^(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(xor_intrin(a,b));
  }
  template <class VData> inline Mask<VData> operator|(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(or_intrin(a,b));
  }
  template <class VData> inline Mask<VData> AndNot(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(andnot_intrin(a,b));
  }

  template <class VData> inline VData convert_mask2vec_intrin(const Mask<VData>& v) {
    if constexpr (std::is_same<MaskLanes<VData>, VData>::value) {
      return v;
    } else {
      return reinterpret_intrin<VData>(static_cast<const MaskLanes<VData>&>(v));
    }
  }
  template <class VData> inline Mask<VData> convert_vec2mask_intrin(const VData& v) {
    return Mask<VData>(v);
  }
  template <class VData> using MaskIntVec = VecData<typename IntegerType<sizeof(typename VData::ScalarType)>::value, VData::Size>; // a mask as integers of all ones or all zeros
  template <class VData> inline MaskIntVec<VData> mask2int_intrin(const Mask<VData>& m) {
    if constexpr (sizeof(Mask<VData>) == sizeof(MaskIntVec<VData>)) { // lanes of the size of the data
      return reinterpret_mask<MaskIntVec<VData>>(m);
    } else { // bit masks of AVX-512
      return reinterpret_intrin<MaskIntVec<VData>>(convert_mask2vec_intrin(m));
    }
  }
  template <class VData> inline Mask<VData> int2mask_intrin(const MaskIntVec<VData>& q) {
    if constexpr (sizeof(Mask<VData>) == sizeof(MaskIntVec<VData>)) {
      union {
        MaskIntVec<VData> q;
        Mask<VData> m;
      } u = {q};
      return u.m;
    } else {
      return convert_vec2mask_intrin(reinterpret_intrin<VData>(q));
    }
  }
  template <class VDataTo, class VData> inline Mask<VDataTo> convert_mask_intrin(const Mask<VData>& m) { // same lanes, another element type
    if constexpr (sizeof(typename VDataTo::ScalarType) == sizeof(typename VData::ScalarType)) { // same mask layout
      return reinterpret_mask<Mask<VDataTo>>(m);
    } else {
      return int2mask_intrin<VDataTo>(convert_intrin<MaskIntVec<VDataTo>>(mask2int_intrin(m))); // lanes of -1 and 0 keep their value
    }
  }
  template <class VData> inline Integer mask_count_intrin(const Mask<VData>& m) { // number of selected lanes
    union {
      MaskIntVec<VData> v;
      typename MaskIntVec<VData>::ScalarType q[VData::Size];
    } m_ = {mask2int_intrin(m)};
    Integer count = 0;
    for (Integer i = 0; i < VData::Size; i++) count += (m_.q[i] ? 1 : 0);
    return count;
  }

  template <class VData> inline Integer mask_popcnt_intrin(const Mask<VData>& v) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "reads one lane per element; register masks need a specialization");
    union {
        Mask<VData> m;
        typename IntegerType<sizeof(typename VData::ScalarType)>::value q[VData::Size];
    } v_ = {v};

    Integer cnt = 0;
    for (Integer i = 0; i < VData::Size; i++) cnt += (v_.q[i]!=0);

    return cnt;
  }

  template <class VData> inline bool mask_any(const Mask<VData>& v) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "reads one lane per element; register masks need a specialization");
    union {
        Mask<VData> m;
        typename IntegerType<sizeof(typename VData::ScalarType)>::value q[VData::Size];
    } v_ = {v};

    for (Integer i = 0; i < VData::Size; i++) if (v_.q[i]) return true;
    return false;
  }

  template <class VData> inline void mask_compress_store(const Mask<VData>& mask, const VData& v, typename VData::ScalarType* ptr) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "reads one lane per element; register masks need a specialization");
    union {
        Mask<VData> m;
        typename IntegerType<sizeof(typename VData::ScalarType)>::value q[VData::Size];
    } mask_ = {mask};

    union {
        VData vec;
        typename VData::ScalarType s[VData::Size];
    } v_ = {v};

    Integer idx = 0;
    for (Integer i = 0; i < VData::Size; i++) {
        if (mask_.q[i]) {
            ptr[idx++] = v_.s[i];
        }
    }
  }

  template <class VData> inline Integer mask_compress_iota_store(const Mask<VData>& mask, int32_t base, int32_t* ptr) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "reads one lane per element; register masks need a specialization");
    union {
        Mask<VData> m;
        typename IntegerType<sizeof(typename VData::ScalarType)>::value q[VData::Size];
    } mask_ = {mask};

    Integer idx = 0;
    for (Integer i = 0; i < VData::Size; i++) {
        if (mask_.q[i]) {
            ptr[idx++] = base + (int32_t)i;
        }
    }
    return idx;
  }

  template <class VData> inline Integer mask_compress_iota_store2(const Mask<VData>& mask_lo, const Mask<VData>& mask_hi, int32_t base, int32_t* ptr) {
    const Integer c = mask_compress_iota_store(mask_lo, base, ptr);
    return c + mask_compress_iota_store(mask_hi, base + (int32_t)VData::Size, ptr + c);
  }

  template <class VData> inline VData mask_expand_load(const Mask<VData>& mask, const VData& zero, const typename VData::ScalarType* ptr) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "reads one lane per element; register masks need a specialization");
    union {
        Mask<VData> m;
        typename IntegerType<sizeof(typename VData::ScalarType)>::value q[VData::Size];
    } mask_ = {mask};

    union {
        VData vec;
        typename VData::ScalarType s[VData::Size];
    } z_ = {zero};

    Integer idx = 0;
    for (Integer i = 0; i < VData::Size; i++) {
        if (mask_.q[i]) {
            z_.s[i] = ptr[idx++];
        }
    }
    return z_.vec;
  }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  enum class ComparisonType { lt, le, gt, ge, eq, ne};
  template <ComparisonType TYPE, class VData> inline Mask<VData> comp_intrin(const VData& a, const VData& b) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "Invalid operation on Mask");
    using ScalarType = typename VData::ScalarType;
    using IntType = typename IntegerType<sizeof(ScalarType)>::value;

    union U {
      VData v;
      Mask<VData> m;
      ScalarType x[VData::Size];
      IntType q[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    U c_;

    static constexpr IntType zero_const = (IntType)0;
    static constexpr IntType  one_const =~(IntType)0;
    if (TYPE == ComparisonType::lt) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] <  b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::le) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] <= b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::gt) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] >  b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::ge) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] >= b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::eq) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] == b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::ne) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] != b_.x[i] ? one_const : zero_const);
    return c_.m;
  }

  template <class VData> inline VData select_intrin(const Mask<VData>& s, const VData& a, const VData& b) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "Invalid operation on Mask");
    if constexpr (std::is_same<MaskLanes<VData>, VData>::value) {
      union U {
        Mask<VData> m;
        VData v;
      } s_ = {s};
      return or_intrin(and_intrin(a,s_.v), andnot_intrin(b,s_.v));
    } else { // integer lanes: a or b in each lane
      VData r;
      for (Integer i = 0; i < VData::Size; i++) r.v[i] = (s.v[i] ? a.v[i] : b.v[i]);
      return r;
    }
  }

  // Masked load and store
  template <class VData> inline Mask<VData> mask_first_intrin(Integer n) { // true in lanes i < n, for n >= 0
    using ScalarType = typename VData::ScalarType;
    union {
      VData v;
      ScalarType x[VData::Size];
    } iota;
    for (Integer i = 0; i < VData::Size; i++) iota.x[i] = (ScalarType)i;
    return comp_intrin<ComparisonType::lt>(iota.v, set1_intrin<VData>((ScalarType)(n < VData::Size ? n : VData::Size)));
  }
  template <class VData> inline VData loadu_mask_intrin(typename VData::ScalarType const* p, const Mask<VData>& m) { // lanes not in m are zero and not read
    using ScalarType = typename VData::ScalarType;
    using IntType = typename IntegerType<sizeof(ScalarType)>::value;
    union {
      MaskIntVec<VData> v;
      IntType q[VData::Size];
    } m_ = {mask2int_intrin(m)};
    union {
      VData v;
      ScalarType x[VData::Size];
    } vec;
    for (Integer i = 0; i < VData::Size; i++) vec.x[i] = (m_.q[i] ? p[i] : (ScalarType)0);
    return vec.v;
  }
  template <class VData> inline void storeu_mask_intrin(typename VData::ScalarType* p, VData vec, const Mask<VData>& m) { // lanes not in m are not written
    using ScalarType = typename VData::ScalarType;
    using IntType = typename IntegerType<sizeof(ScalarType)>::value;
    union {
      MaskIntVec<VData> v;
      IntType q[VData::Size];
    } m_ = {mask2int_intrin(m)};
    union {
      VData v;
      ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i++) {
      if (m_.q[i]) p[i] = vec_.x[i];
    }
  }
  template <class VData> inline VData loadu_first_intrin(typename VData::ScalarType const* p, Integer n) { // lanes i < n, for n >= 0; the others are zero and not read
    return loadu_mask_intrin<VData>(p, mask_first_intrin<VData>(n));
  }
  template <class VData> inline void storeu_first_intrin(typename VData::ScalarType* p, VData vec, Integer n) { // lanes i < n, for n >= 0; the others are not written
    storeu_mask_intrin(p, vec, mask_first_intrin<VData>(n));
  }

  // Gather and scatter
  template <class VData, class IdxVData> inline VData gather_intrin(typename VData::ScalarType const* p, const IdxVData& idx) { // lane i is p[idx lane i]
    static_assert(IdxVData::Size == VData::Size, "Gather requires one index per lane.");
    union {
      IdxVData v;
      typename IdxVData::ScalarType x[VData::Size];
    } idx_ = {idx};
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    for (Integer i = 0; i < VData::Size; i++) vec.x[i] = p[idx_.x[i]];
    return vec.v;
  }
  template <class VData, class IdxVData> inline void scatter_intrin(typename VData::ScalarType* p, const VData& vec, const IdxVData& idx) { // in lane order: of equal indices, the last lane is stored
    static_assert(IdxVData::Size == VData::Size, "Scatter requires one index per lane.");
    union {
      IdxVData v;
      typename IdxVData::ScalarType x[VData::Size];
    } idx_ = {idx};
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i++) p[idx_.x[i]] = vec_.x[i];
  }

  // Special functions
  template <Integer MAX_ITER, Integer ITER, class VData> struct rsqrt_newton_iter {
    static inline VData eval(const VData& y, const VData& x) {
      using ValueType = typename VData::ScalarType;
      constexpr ValueType c1 = -3 * pow<pow<MAX_ITER-ITER>(3)-1,ValueType>(2);
      return rsqrt_newton_iter<MAX_ITER,ITER-1,VData>::eval(mul_intrin(y, fma_intrin(x,mul_intrin(y,y),set1_intrin<VData>(c1))), x);
    }
  };
  template <Integer MAX_ITER, class VData> struct rsqrt_newton_iter<MAX_ITER,1,VData> {
    static inline VData eval(const VData& y, const VData& x) {
      using ValueType = typename VData::ScalarType;
      constexpr ValueType c1 = -3 * pow<pow<MAX_ITER-1>(3)-1,ValueType>(2);
      constexpr ValueType c2 = pow<(pow<MAX_ITER-1>(3)-1)*3/2+1,ValueType>(-0.5);
      return mul_intrin(mul_intrin(y, fma_intrin(x,mul_intrin(y,y),set1_intrin<VData>(c1))), set1_intrin<VData>(c2));
    }
  };
  template <class VData> struct rsqrt_newton_iter<0,0,VData> {
    static inline VData eval(const VData& y, const VData& x) {
      return y;
    }
  };
  static inline constexpr Integer mylog2(Integer x) {
    return ((x<1) ? 0 : 1+mylog2(x/2));
  }

  // 1/sqrt(x) for double lanes from the estimate with bits R - (bits(x) >> 1), correct for
  // 1e-300 <= x <= 1e300. A first step y0 a (b - x y0^2) with tuned constants leaves a relative
  // error below 6.5e-4; each Newton step then squares the error and multiplies it by 1.5. The steps
  // leave out their constant factors (-a, then -1/2), so the iterate is s y; 1/s is applied at the end.
  template <Integer digits, class VData> struct rsqrt_bittrick_intrin {
    static_assert(std::is_same<typename VData::ScalarType, double>::value, "rsqrt_bittrick_intrin requires double lanes.");
    using IntVData = VecData<int64_t,VData::Size>;
    static constexpr double a = 0.703952253;
    static constexpr double b = 2.38924456;

    static constexpr Integer steps() {
      Integer n = 1;
      for (double d = 3.18; d < digits; d = 2*d - 0.18) n++;
      return n;
    }
    static constexpr double scale(Integer k) { // s after k steps
      double s = -1/a;
      for (Integer i = 1; i < k; i++) s = -2*s*s*s;
      return s;
    }
    template <Integer K> static inline VData newton(const VData& z, const VData& x) { // steps K+1, ..., steps() on z = s y
      if constexpr (K == steps()) {
        constexpr double c = 1/scale(K);
        return mul_intrin(z, set1_intrin<VData>(c));
      } else {
        constexpr double c = -3*scale(K)*scale(K);
        return newton<K+1>(mul_intrin(z, fma_intrin(x, mul_intrin(z, z), set1_intrin<VData>(c))), x);
      }
    }
    static inline VData estimate(const VData& x) {
      return reinterpret_intrin<VData>(sub_intrin(set1_intrin<IntVData>(0x5FE3FFFF20000000ll), reinterpret_intrin<IntVData>(bitshiftright_intrin(x, 1))));
    }
    static inline VData refine(const VData& y0, const VData& x) {
      if constexpr (steps() == 1) { // -a x, apart from y0, keeps the factor a off the path from y0
        return mul_intrin(y0, fma_intrin(mul_intrin(x, set1_intrin<VData>(-a)), mul_intrin(y0, y0), set1_intrin<VData>(a*b)));
      } else {
        return newton<1>(mul_intrin(y0, fma_intrin(x, mul_intrin(y0, y0), set1_intrin<VData>(-b))), x);
      }
    }

    static inline VData eval(const VData& x) {
      return refine(estimate(x), x);
    }
    static inline VData eval(const VData& x, const Mask<VData>& m) { // zero in the lanes not in m
      return refine(and_intrin(estimate(x), convert_mask2vec_intrin(m)), x);
    }
  };
  template <Integer digits, class VData> struct rsqrt_approx_intrin {
    static inline VData eval(const VData& a) {
      union {
        VData v;
        typename VData::ScalarType x[VData::Size];
      } a_ = {a}, b_;
      for (Integer i = 0; i < VData::Size; i++) b_.x[i] = (a_.x[i]==0 ? 0 : 1/sqrt<typename VData::ScalarType>((typename VData::ScalarType)a_.x[i]));
      return b_.v;

      //for (Integer i = 0; i < VData::Size; i++) b_.x[i] = (a_.x[i]==0 ? 0 : (typename VData::ScalarType)1/sqrt<float>((float)a_.x[i])); // converting to float results in overflow / underflow
      //constexpr Integer newton_iter = mylog2((Integer)(digits/7.2247198959));
      //return rsqrt_newton_iter<newton_iter,newton_iter,VData>::eval(b_.v, a);
    }
    static inline VData eval(const VData& a, const Mask<VData>& m) {
      return and_intrin(rsqrt_approx_intrin<digits,VData>::eval(a), convert_mask2vec_intrin(m));
    }
  };

  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0) {
    return set1_intrin<VData>(c0);
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1) {
    return fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2) {
    VData x2(mul_intrin<VData>(x1,x1));
    return fma_intrin(x2, set1_intrin<VData>(c2), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3) {
    VData x2(mul_intrin<VData>(x1,x1));
    return fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, set1_intrin<VData>(c4), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4)), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, fma_intrin(x2, set1_intrin<VData>(c6), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, set1_intrin<VData>(c8), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x2, set1_intrin<VData>(c10), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, set1_intrin<VData>(c12) , fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12)) , fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13, const CType& c14) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, fma_intrin(x2, set1_intrin<VData>(c14), fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13, const CType& c14, const CType& c15) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c15), set1_intrin<VData>(c14)), fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }

  // the lowest degree n >= first with d[n - first] >= digits, at most full, for the correct digits d of polynomials of
  // degrees first, first + 1, ...; full for digits = -1
  template <Integer K> inline constexpr Integer poly_degree(const double (&d)[K], const Integer first, const Integer full, const Integer digits) {
    Integer n = first;
    while (n < full && (digits < 0 || d[n - first] < digits)) n++;
    return n;
  }
  // sin(r) = r + r^3 Q(r^2), |r| <= pi/4: minimax Q of degree n, lowest degree first
  template <Integer n> struct SinPolyCoeffs;
  template <> struct SinPolyCoeffs<0> { static constexpr double c[] = {-0.16242791542888496}; };
  template <> struct SinPolyCoeffs<1> { static constexpr double c[] = {-0.16663390377250456, 0.008163281920009162}; };
  template <> struct SinPolyCoeffs<2> { static constexpr double c[] = {-0.16666654609548295, 0.008332160761835617, -0.00019515283188763054}; };
  template <> struct SinPolyCoeffs<3> { static constexpr double c[] = {-0.16666666640797043, 0.008333329304842221, -0.00019839312269356467, 2.718121626693011e-06}; };
  template <> struct SinPolyCoeffs<4> { static constexpr double c[] = {-0.1666666666663035, 0.008333333325077774, -0.0001984126372863407, 2.7555339656435837e-06, -2.4760454564777297e-08}; };
  template <> struct SinPolyCoeffs<5> { static constexpr double c[] = {-0.1666666666666663, 0.008333333333322118, -0.00019841269829589542, 2.755731362138634e-06, -2.5050747762944872e-08, 1.5896230162198227e-10}; };
  // cos(r) = 1 - r^2/2 + r^4 P(r^2), |r| <= pi/4: minimax P of degree n, lowest degree first
  template <Integer n> struct CosPolyCoeffs;
  template <> struct CosPolyCoeffs<0> { static constexpr double c[] = {0.040899305420584016}; };
  template <> struct CosPolyCoeffs<1> { static constexpr double c[] = {0.04166107130733634, -0.0013648714373470025}; };
  template <> struct CosPolyCoeffs<2> { static constexpr double c[] = {0.0416666456829738, -0.0013887316254283283, 2.4433157050415554e-05}; };
  template <> struct CosPolyCoeffs<3> { static constexpr double c[] = {0.04166666661949214, -0.001388888350013995, 2.4799460170717853e-05, -2.720575549271016e-07}; };
  template <> struct CosPolyCoeffs<4> { static constexpr double c[] = {0.04166666666659654, -0.0013888888877611816, 2.4801580707332747e-05, -2.7555523112319636e-07, 2.0645119045192055e-09}; };
  inline constexpr double sin_poly_digits[] = {3.24, 5.73, 8.42, 11.29, 14.3, 16.97}; // correct digits of degrees 0 to 5
  inline constexpr double cos_poly_digits[] = {4.36, 7.08, 9.93, 12.92, 16.03}; // degrees 0 to 4

  template <class Real> struct PiOver2Split { // pi/2 = hi + mid + lo from the integers of const_pi (A 2^-62 + B 2^-125); for |x| < lim, n hi and n mid are exact, with n = round(x 2/pi)
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Integer H = std::min<Integer>((SigBits + 1) * 3 / 8, 32);
    static constexpr uint64_t A = 14488038916154245684ull;
    static constexpr uint64_t B = 7089564414062235240ull;
    static constexpr Real hi = (Real)(A >> (64 - H)) / (Real)(1ull << (H - 1));
    static constexpr Real mid = (Real)((A >> (64 - 2*H)) & ((1ull << H) - 1)) / (Real)(1ull << (H - 1)) / (Real)(1ull << H);
    static constexpr Real lo = (Real)(A & ((1ull << (64 - 2*H)) - 1)) / (Real)(1ull << 63) + (Real)B / (Real)(1ull << 63) / (Real)(1ull << 63);
    static constexpr Real lim = (Real)(((uint64_t)1) << std::min<Integer>(SigBits - H, 62));
    template <class VData> static inline VData sub_n(const VData& x, const VData& n) { // x - n pi/2
      VData x1 = fma_intrin(n, set1_intrin<VData>(-hi), x);
      x1 = fma_intrin(n, set1_intrin<VData>(-mid), x1);
      return fma_intrin(n, set1_intrin<VData>(-lo), x1);
    }
  };

  // x = (n + w)/2 exactly, n = round(2x) (halfway cases to even), |w| <= 1/2; t = n + 1.5 2^SigBits holds the low bits of n.
  // FullRange = false: only for |x| < 2^(SigBits-2). w is NaN for inf and NaN
  template <bool FullRange = true, class VData> inline void sincos_pi_reduce_intrin(VData& w, VData& t, const VData& x) {
    using Real = typename VData::ScalarType;
    static constexpr Real c = (Real)1.5 * pow<TypeTraits<Real>::SigBits,Real>((Real)2); // v + c: round(v) in the low bits
    VData y = x;
    if constexpr (FullRange) { // x minus the nearest multiple of 4, exact: |y| <= 2 for |x| < 2^(SigBits+1), else y is even
      if (mask_count_intrin(comp_intrin<ComparisonType::lt>(fabs_intrin(x), set1_intrin<VData>(c / 6))) < VData::Size) {
        const VData c4 = set1_intrin<VData>(4 * c);
        y = sub_intrin(x, sub_intrin(add_intrin(x, c4), c4));
      }
    }
    const VData z = add_intrin(y, y);
    t = add_intrin(z, set1_intrin<VData>(c));
    w = sub_intrin(z, sub_intrin(t, set1_intrin<VData>(c)));
  }
  // x1 = x - n pi/2 and the low bits of x_int = n + 1.5 2^SigBits by a reduction in double, in the lanes of float x not
  // in in_range: the rare path of approx_sincos_intrin, not inlined; arguments and results by value
  template <class VData> [[gnu::noinline, gnu::cold]] std::pair<VData, VData> sincos_reduce_double_intrin(VData x1, VData x_int, const VData x, const Mask<VData> in_range) {
    using HalfVec = VecData<float, VData::Size/2>;
    using DoubleVec = VecData<double, VData::Size/2>;
    const auto reduce = [](HalfVec& r, HalfVec& q, const HalfVec& h) { // r = h - n pi/2, q = n mod 4
      static constexpr double offset = 1.5 * pow<TypeTraits<double>::SigBits,double>(2.0);
      const DoubleVec xd = convert_intrin<DoubleVec>(h);
      const DoubleVec n = sub_intrin(fma_intrin(xd, set1_intrin<DoubleVec>(2 / const_pi<double>()), set1_intrin<DoubleVec>(offset)), set1_intrin<DoubleVec>(offset));
      r = convert_intrin<HalfVec>(PiOver2Split<double>::sub_n(xd, n));
      q = convert_intrin<HalfVec>(fma_intrin(floor_intrin(mul_intrin(n, set1_intrin<DoubleVec>(0.25))), set1_intrin<DoubleVec>(-4.0), n));
    };
    HalfVec r_lo, q_lo, r_hi, q_hi;
    reduce(r_lo, q_lo, get_low_intrin(x));
    reduce(r_hi, q_hi, get_high_intrin(x));
    x1 = select_intrin(in_range, x1, concat_intrin(r_lo, r_hi));
    x_int = select_intrin(in_range, x_int, add_intrin(concat_intrin(q_lo, q_hi), set1_intrin<VData>((float)1.5 * pow<TypeTraits<float>::SigBits,float>((float)2))));
    return {x1, x_int};
  }
  // sin and cos one element at a time in the lanes with |x| >= lim, inf or NaN: the rare path of approx_sincos_intrin,
  // not inlined; arguments and results by value
  template <class VData> [[gnu::noinline, gnu::cold]] std::pair<VData, VData> sincos_beyond_intrin(const VData sinx, const VData cosx, const VData x, const typename VData::ScalarType lim) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U x_u = {x};
    U s_u = {sinx};
    U c_u = {cosx};
    for (Integer i = 0; i < VData::Size; i++) {
      if (!(fabs(x_u.x[i]) < lim)) {
        s_u.x[i] = sin(x_u.x[i]);
        c_u.x[i] = cos(x_u.x[i]);
      }
    }
    return {s_u.v, c_u.v};
  }
  // sin(x), cos(x) to DIGITS digits (-1: full): x = n pi/2 + r, by PiOver2Split with FullRange, else by one product, and
  // minimax sin and cos polynomials of r chosen by n mod 4
  template <Integer DIGITS, bool FullRange = true, class VData> inline void approx_sincos_intrin(VData& sinx, VData& cosx, const VData& x) {
    using Real = typename VData::ScalarType;
    static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value, "Expected float or double!");
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int, VData::Size>;
    static constexpr bool F = std::is_same<Real,float>::value;
    static constexpr Integer Bits = sizeof(Real) * 8;
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Real pi_over_2 = const_pi<Real>()/2;
    static constexpr Real neg_pi_over_2 = -const_pi<Real>()/2;
    static constexpr Real inv_pi_over_2 = 1 / pi_over_2;

    static constexpr Real Creal = (Real)1.5 * pow<SigBits,Real>((Real)2); // x + Creal: the integer in the low bits of the significand
    VData real_offset(set1_intrin<VData>(Creal));

    VData x_int(fma_intrin(x, set1_intrin<VData>(inv_pi_over_2), real_offset));
    VData x_(sub_intrin(x_int, real_offset)); // x_ <-- round(x*inv_pi_over_2)
    VData x1;
    if constexpr (FullRange) {
      x1 = PiOver2Split<Real>::sub_n(x, x_);
    } else {
      x1 = fma_intrin(x_, set1_intrin<VData>(neg_pi_over_2), x);
    }
    static constexpr bool reduce_in_double = FullRange && std::is_same<Real,float>::value && VData::Size % 2 == 0;
    bool beyond_lim = true; // a lane can be at or beyond PiOver2Split<Real>::lim
    if constexpr (reduce_in_double) { // these lanes: x1 and the low bits of x_int from the reduction in double
      const Mask<VData> in_range = comp_intrin<ComparisonType::lt>(fabs_intrin(x), set1_intrin<VData>(PiOver2Split<float>::lim));
      beyond_lim = (mask_count_intrin(in_range) < VData::Size);
      if (beyond_lim) std::tie(x1, x_int) = sincos_reduce_double_intrin(x1, x_int, x, in_range);
    }

    const VData r2 = mul_intrin(x1, x1);
    using SinC = SinPolyCoeffs<poly_degree(sin_poly_digits, 0, (F ? 2 : 5), DIGITS)>;
    using CosC = CosPolyCoeffs<poly_degree(cos_poly_digits, 0, (F ? 2 : 4), DIGITS)>;
    const VData s = fma_intrin(mul_intrin(x1, r2), eval_poly_horner_intrin(r2, SinC::c), x1);
    const VData c = fma_intrin(r2, fma_intrin(r2, eval_poly_horner_intrin(r2, CosC::c), set1_intrin<VData>((Real)-0.5)), set1_intrin<VData>((Real)1)); // 1 + r^2 (-1/2 + r^2 P)
    const IntVec ti = reinterpret_intrin<IntVec>(x_int);
    const Mask<VData> odd = reinterpret_mask<Mask<VData>>(comp_intrin<ComparisonType::ne>(and_intrin(ti, set1_intrin<IntVec>(1)), zero_intrin<IntVec>()));
    const IntVec sign = set1_intrin<IntVec>(((Int)1) << (Bits - 1));
    const IntVec t1 = bitshiftleft_intrin(ti, Bits - 2); // bit 1 of n at the sign
    sinx = xor_intrin(select_intrin(odd, c, s), reinterpret_intrin<VData>(and_intrin(t1, sign))); // n mod 4 = 0: s, 1: c, 2: -s, 3: -c
    cosx = xor_intrin(select_intrin(odd, s, c), reinterpret_intrin<VData>(and_intrin(xor_intrin(t1, bitshiftleft_intrin(ti, Bits - 1)), sign))); // c, -s, -c, s

    if constexpr (FullRange) { // |x| beyond the exact range of the reduction, inf and NaN: one element at a time
      static constexpr Real lim = (reduce_in_double ? (Real)PiOver2Split<double>::lim : PiOver2Split<Real>::lim);
      if (beyond_lim && mask_count_intrin(comp_intrin<ComparisonType::lt>(fabs_intrin(x), set1_intrin<VData>(lim))) < VData::Size) std::tie(sinx, cosx) = sincos_beyond_intrin(sinx, cosx, x, lim);
    }
  }
  template <class VData> inline void sincos_intrin(VData& sinx, VData& cosx, const VData& x) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U sinx_, cosx_, x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) {
      sinx_.x[i] = sin(x_.x[i]);
      cosx_.x[i] = cos(x_.x[i]);
    }
    sinx = sinx_.v;
    cosx = cosx_.v;
  }

  template <class Real, uint64_t A, uint64_t B> struct ConstSplit { // c = A 2^-63 + B 2^-126 = hi + lo; n hi is exact for |n| < 2^ExpBits
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Integer ExpBits = TypeTraits<Real>::ExpBits;
    static constexpr Integer HiBits = std::max<Integer>(0, std::min<Integer>(63, SigBits + 1 - ExpBits));
    static constexpr Real hi = (Real)(A >> (63 - HiBits)) / (Real)(1ull << HiBits);
    static constexpr Real lo = (Real)(A & ((1ull << (63 - HiBits)) - 1)) / (Real)(1ull << 63) + (Real)B / (Real)(1ull << 63) / (Real)(1ull << 63);
  };
  template <class Real> using Ln2Split = ConstSplit<Real, 6393154322601327829ull, 8248603190132260267ull>; // the integers of const_ln2
  template <class Real> using Log10Of2Split = ConstSplit<Real, 2776511644261678566ull, 1292862076836876847ull>; // log10(2)
  template <class Real> inline constexpr Real exp_normal_lim() { // for |x| below, e^x and 2^round(x/ln2) are normal and finite
    return (Real)((((Long)1) << (TypeTraits<Real>::ExpBits - 1)) - 2) * const_ln2<Real>();
  }
  enum class ExpArg { Plain, Sum, Base10 }; // approx_exp_intrin of x: e^x, e^(x + x_lo), 10^x
  // e^x; with ExpArg::Sum, e^(x + x_lo) for |x_lo| <= ulp(x)/2, x_lo added to the reduced argument; with ExpArg::Base10,
  // 10^x, x reduced by n log10(2)
  template <Integer ORDER, bool RangeCheck = true, ExpArg Arg = ExpArg::Plain, class VData> inline VData approx_exp_intrin(const VData& x, const VData& x_lo) {
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int, VData::Size>;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");

    static constexpr Int SigBits = TypeTraits<Real>::SigBits;
    static constexpr Real coeff2  = 1/(((Real)2));
    static constexpr Real coeff3  = 1/(((Real)2)*3);
    static constexpr Real coeff4  = 1/(((Real)2)*3*4);
    static constexpr Real coeff5  = 1/(((Real)2)*3*4*5);
    static constexpr Real coeff6  = 1/(((Real)2)*3*4*5*6);
    static constexpr Real coeff7  = 1/(((Real)2)*3*4*5*6*7);
    static constexpr Real coeff8  = 1/(((Real)2)*3*4*5*6*7*8);
    static constexpr Real coeff9  = 1/(((Real)2)*3*4*5*6*7*8*9);
    static constexpr Real coeff10 = 1/(((Real)2)*3*4*5*6*7*8*9*10);
    static constexpr Real coeff11 = 1/(((Real)2)*3*4*5*6*7*8*9*10*11);
    static constexpr Real coeff12 = 1/(((Real)2)*3*4*5*6*7*8*9*10*11*12);
    static constexpr Real coeff13 = 1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13); // err = 2^-57.2759
    static constexpr bool B10 = (Arg == ExpArg::Base10);
    static constexpr Real U = (B10 ? (Real)0.30102999566398119521373889472449302677L : const_ln2<Real>()); // the unit of x: n U = n ln2 in base e
    static constexpr Real x0 = -U;
    static constexpr Real invx0 = -1 / x0; // 1/ln(2), or log2(10)
    static constexpr Integer ExpBits = TypeTraits<Real>::ExpBits;
    static constexpr bool split_ln2 = [] { // Taylor error (ln2/2)^(ORDER+1)/(ORDER+1)! below the error 2^(ExpBits-1) eps of x1 with one-part ln2
      double taylor_err = 1;
      for (Integer k = 1; k <= std::min<Integer>(ORDER, 13) + 1; k++) taylor_err *= 0.34657359027997264 / k;
      double x1_err = 1;
      for (Integer k = ExpBits - 1; k < SigBits + 1; k++) x1_err *= 0.5;
      return taylor_err < x1_err;
    }();

    static constexpr Int Bias = (((Int)1) << (ExpBits - 1)) - 1;
    static constexpr Real magic = (Real)1.5 * pow<SigBits,Real>((Real)2) + (Real)Bias; // y + magic: y rounded to an integer n, with n + Bias in the low bits

    const auto reduce = [](VData& t, VData& e1, const VData& xx, [[maybe_unused]] const VData& xx_lo) { // e^(xx + xx_lo) = e1 2^n, t = n + magic
      t = fma_intrin(xx, set1_intrin<VData>(invx0), set1_intrin<VData>(magic));
      const VData n = sub_intrin(t, set1_intrin<VData>(magic));
      VData x1;
      if constexpr (B10) { // (x - n log10(2)) ln10
        x1 = fma_intrin(n, set1_intrin<VData>(-Log10Of2Split<Real>::hi), xx);
        x1 = mul_intrin(fma_intrin(n, set1_intrin<VData>(-Log10Of2Split<Real>::lo), x1), set1_intrin<VData>((Real)2.302585092994045684017991454684364208L));
      } else if constexpr (split_ln2) { // n * Ln2Split::hi is exact
        x1 = fma_intrin(n, set1_intrin<VData>(-Ln2Split<Real>::hi), xx);
        x1 = fma_intrin(n, set1_intrin<VData>(-Ln2Split<Real>::lo), x1);
      } else {
        x1 = fma_intrin(n, set1_intrin<VData>(x0), xx);
      }
      if constexpr (Arg == ExpArg::Sum) x1 = add_intrin(x1, xx_lo);
      if      (ORDER >= 13) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10, coeff11, coeff12, coeff13);
      else if (ORDER >= 12) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10, coeff11, coeff12);
      else if (ORDER >= 11) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10, coeff11);
      else if (ORDER >= 10) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10);
      else if (ORDER >=  9) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9);
      else if (ORDER >=  8) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8);
      else if (ORDER >=  7) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7);
      else if (ORDER >=  6) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6);
      else if (ORDER >=  5) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5);
      else if (ORDER >=  4) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4);
      else if (ORDER >=  3) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3);
      else if (ORDER >=  2) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2);
      else if (ORDER >=  1) e1 = EvalPolynomial(x1, (Real)1, (Real)1);
      else if (ORDER >=  0) e1 = set1_intrin<VData>(1);
    };
    const auto pow2 = [](const VData& t) { // 2^n for t = n + magic, -Bias < n <= Bias
      return reinterpret_intrin<VData>(bitshiftleft_intrin(reinterpret_intrin<IntVec>(t), SigBits));
    };

    VData t, e1;
    reduce(t, e1, x, x_lo);
    if constexpr (!ieee_layout<Real>()) { // 2^n one element at a time
      union {
        VData v;
        Real x[VData::Size];
      } u = {sub_intrin(t, set1_intrin<VData>(magic))};
      for (Integer i = 0; i < VData::Size; i++) u.x[i] = std::ldexp((Real)1, (int)std::max<Real>(-20000, std::min<Real>(20000, u.x[i])));
      const VData e = mul_intrin(e1, u.v);
      if constexpr (!RangeCheck) return e;
      static constexpr Real max_x = ((Real)(Bias + 1) + (Real)0.25) * U; // x = -inf gives e1 = NaN
      return select_intrin(comp_intrin<ComparisonType::lt>(x, set1_intrin<VData>(-2*max_x)), zero_intrin<VData>(), select_intrin(comp_intrin<ComparisonType::gt>(x, set1_intrin<VData>(max_x)), set1_intrin<VData>((Real)INFINITY), e));
    } else {
      if constexpr (!RangeCheck) return mul_intrin(e1, pow2(t));
      if (mask_count_intrin(comp_intrin<ComparisonType::lt>(fabs_intrin(x), set1_intrin<VData>(exp_normal_lim<Real>() / const_ln2<Real>() * U))) == VData::Size) return mul_intrin(e1, pow2(t));

      // some x near or beyond the range of the result, inf or NaN: x clamped to where e^x is 0 or inf, and 2^n = 2^hi 2^lo,
      // hi >= lo; e1 2^hi is exact and normal, so the result is rounded once, also where it is subnormal
      static constexpr Real xmax = (Real)(2 * (Bias - 2)) * U;
      reduce(t, e1, min_intrin(set1_intrin<VData>(xmax), max_intrin(set1_intrin<VData>(-xmax), x)), x_lo); // keeps NaN
      const VData n = sub_intrin(t, set1_intrin<VData>(magic));
      const VData th = fma_intrin(n, set1_intrin<VData>((Real)0.5), set1_intrin<VData>(magic));
      const VData h = sub_intrin(th, set1_intrin<VData>(magic));
      const VData k = sub_intrin(n, h);
      const VData hi = max_intrin(h, k);
      const VData lo = min_intrin(h, k);
      return mul_intrin(mul_intrin(e1, pow2(add_intrin(hi, set1_intrin<VData>(magic)))), pow2(add_intrin(lo, set1_intrin<VData>(magic))));
    }
  }
  template <Integer ORDER, bool RangeCheck = true, class VData> inline VData approx_exp_intrin(const VData& x) {
    return approx_exp_intrin<ORDER, RangeCheck, ExpArg::Plain>(x, x);
  }
  template <class VData> inline Mask<VData> positive_normal_mask_intrin(const VData& x) { // x normal, positive and finite
    using Real = typename VData::ScalarType;
    // x <= max, not x < inf: clang turns the pair of compares into a class test of integer instructions on x86
    return comp_intrin<ComparisonType::ge>(x, set1_intrin<VData>(std::numeric_limits<Real>::min())) & comp_intrin<ComparisonType::le>(x, set1_intrin<VData>(std::numeric_limits<Real>::max()));
  }
  template <class VData> inline void log_mant_intrin(VData& e, VData& m, const VData& x) { // x = 2^e m, m in [sqrt(1/2), sqrt(2)), for normal x > 0
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Int Bias = (((Int)1) << (sizeof(Real)*8 - SigBits - 2)) - 1;
    union U {
      Int i;
      Real r;
    };
    static const U frac_mask = {(((Int)1) << SigBits) - 1};
    static const U two_pow_sig = {(Bias + SigBits) << SigBits};
    const VData one = set1_intrin<VData>((Real)1);
    const VData m1 = or_intrin(and_intrin(x, set1_intrin<VData>(frac_mask.r)), one); // in [1, 2); bit operations on the real lanes, also without 256-bit integer instructions
    const VData e1 = sub_intrin(or_intrin(bitshiftright_intrin(x, SigBits), set1_intrin<VData>(two_pow_sig.r)), set1_intrin<VData>(two_pow_sig.r + (Real)Bias)); // the exponent field as the low bits of 2^SigBits
    const Mask<VData> big = comp_intrin<ComparisonType::gt>(m1, set1_intrin<VData>((Real)1.41421356237309504880));
    m = select_intrin(big, mul_intrin(m1, set1_intrin<VData>((Real)0.5)), m1);
    e = add_intrin(e1, select_intrin(big, one, zero_intrin<VData>()));
  }
  template <bool Subnormal = true, class VData> inline void log_split_intrin(VData& e, VData& f, const VData& x) { // x = 2^e (1+f), 1+f in [sqrt(1/2), sqrt(2)), for x > 0; without Subnormal, for normal x
    using Real = typename VData::ScalarType;
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    const VData zero = zero_intrin<VData>();
    const VData one = set1_intrin<VData>((Real)1);

    VData xs = x;
    Mask<VData> subnormal;
    if constexpr (Subnormal) { // scaled by 2^(SigBits+1) first
      subnormal = comp_intrin<ComparisonType::lt>(x, set1_intrin<VData>(std::numeric_limits<Real>::min()));
      xs = select_intrin(subnormal, mul_intrin(x, set1_intrin<VData>((Real)(((uint64_t)1) << (SigBits + 1)))), x);
    }
    VData m;
    log_mant_intrin(e, m, xs);
    if constexpr (Subnormal) e = sub_intrin(e, select_intrin(subnormal, set1_intrin<VData>((Real)(SigBits + 1)), zero));
    f = sub_intrin(m, one);
  }
  // e ln2 + log(1+f), log(1+f) = f - f^2/2 + f^3 P(f)/Q(f) (float: f^3 P(f)) with Cephes's coefficients, added as
  // (e ln2_lo + f^3 P/Q) + (f - f^2/2), then e ln2_hi
  template <class VData> inline VData log_split_poly_intrin(const VData& e, const VData& f) {
    using Real = typename VData::ScalarType;
    const VData f2 = mul_intrin(f, f);
    VData p;
    if constexpr (std::is_same<Real,float>::value) {
      p = mul_intrin(EvalPolynomial(f, 3.3333331174E-1f, -2.4999993993E-1f, 2.0000714765E-1f, -1.6668057665E-1f, 1.4249322787E-1f, -1.2420140846E-1f, 1.1676998740E-1f, -1.1514610310E-1f, 7.0376836292E-2f), mul_intrin(f, f2));
    } else {
      const VData P = EvalPolynomial(f, 7.70838733755885391666E0, 1.79368678507819816313E1, 1.44989225341610930846E1, 4.70579119878881725854E0, 4.97494994976747001425E-1, 1.01875663804580931796E-4);
      const VData Q = EvalPolynomial(f, 2.31251620126765340583E1, 7.11544750618563894466E1, 8.29875266912776603211E1, 4.52279145837532221105E1, 1.12873587189167450590E1, 1.0);
      p = div_intrin(mul_intrin(P, mul_intrin(f, f2)), Q);
    }
    const VData small = add_intrin(fma_intrin(e, set1_intrin<VData>(Ln2Split<Real>::lo), p), fma_intrin(f2, set1_intrin<VData>((Real)-0.5), f));
    return fma_intrin(e, set1_intrin<VData>(Ln2Split<Real>::hi), small);
  }
  // f + c 2^-e for 2^e (1+f) from log_split_intrin and a small correction c: log(2^e (1+f) + c) = e ln2 + log(1 + f + c 2^-e).
  // c = 0 where e >= Bias - 1
  template <class VData> inline VData log_split_add_intrin(const VData& e, const VData& f, const VData& c) {
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int,VData::Size>;
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Int Bias = (((Int)1) << (sizeof(Real)*8 - SigBits - 2)) - 1;
    static constexpr Real magic = (Real)1.5 * pow<SigBits,Real>((Real)2) + (Real)Bias; // magic - e: Bias - e in the low bits
    const VData t = sub_intrin(set1_intrin<VData>(magic), min_intrin(e, set1_intrin<VData>((Real)(Bias - 1))));
    const VData s = reinterpret_intrin<VData>(bitshiftleft_intrin(reinterpret_intrin<IntVec>(t), SigBits)); // 2^-e
    return fma_intrin(c, s, f);
  }
  template <class VData> inline VData log_poly_intrin(const VData& x) { // log_split_poly_intrin of log_split_intrin
    using Real = typename VData::ScalarType;
    static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value, "Expected float or double!");
    const VData zero = zero_intrin<VData>();
    const VData inf = set1_intrin<VData>((Real)INFINITY);
    VData e, f;
    log_split_intrin<false>(e, f, x);
    const VData r = log_split_poly_intrin(e, f);
    if (mask_count_intrin(positive_normal_mask_intrin(x)) == VData::Size) return r;

    // some x zero, subnormal, inf, negative or NaN
    log_split_intrin<true>(e, f, x);
    VData rs = log_split_poly_intrin(e, f);
    rs = select_intrin(comp_intrin<ComparisonType::eq>(x, zero), unary_minus_intrin(inf), rs);
    rs = select_intrin(comp_intrin<ComparisonType::eq>(x, inf), x, rs);
    return select_intrin(comp_intrin<ComparisonType::ge>(x, zero), rs, set1_intrin<VData>((Real)NAN)); // negative x and NaN
  }
  template <class VData> inline void split_half_intrin(VData& hi, VData& lo, const VData& x) { // x = hi + lo exactly, hi with the low half of the significand cleared
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    union {
      Int i;
      Real r;
    } static const high = {~((((Int)1) << ((TypeTraits<Real>::SigBits + 2) / 2)) - 1)};
    hi = and_intrin(x, set1_intrin<VData>(high.r));
    lo = sub_intrin(x, hi);
  }
  template <class VData, class = void> inline constexpr bool array_lanes = false; // VData holds its lanes in an array
  template <class VData> inline constexpr bool array_lanes<VData, std::enable_if_t<VData::ArrayLanes>> = true;
#if defined(__FMA__) || defined(__FMA4__)
  template <class VData> inline constexpr bool fused_fma = !array_lanes<VData>; // fma_intrin rounds once
#else
  template <class VData> inline constexpr bool fused_fma = false;
#endif
  template <class VData> inline VData mul_sub_exact_intrin(const VData& a, const VData& b, const VData& c) { // a b - c, the product to about twice the precision, for c near a b: by FMA, or else with a and b split in two halves
    if constexpr (fused_fma<VData>) {
      return fma_intrin(a, b, unary_minus_intrin(c));
    } else {
      VData a_hi, a_lo, b_hi, b_lo;
      split_half_intrin(a_hi, a_lo, a);
      split_half_intrin(b_hi, b_lo, b);
      const VData r = sub_intrin(mul_intrin(a_hi, b_hi), c); // a_hi b_hi is exact; for double, a_lo b_lo rounds, at about 2^-107 of a b
      return add_intrin(add_intrin(r, add_intrin(mul_intrin(a_hi, b_lo), mul_intrin(a_lo, b_hi))), mul_intrin(a_lo, b_lo));
    }
  }
  // log(2^e (1+f)) = L_hi + L_lo to about 1.5 times the precision of Real (e, f from log_split_intrin): 2s + s R,
  // s = f/(2+f)
  template <class VData> inline void log_hi_lo_intrin(VData& L_hi, VData& L_lo, const VData& e, const VData& f) {
    using Real = typename VData::ScalarType;
    const VData two = set1_intrin<VData>((Real)2);
    const VData t = add_intrin(f, two);
    const VData t_lo = add_intrin(sub_intrin(two, t), f); // 2 + f = t + t_lo
    const VData s = div_intrin(f, t);
    const VData inv = div_intrin(set1_intrin<VData>((Real)1), t); // in parallel with s; s_lo = rem inv
    const VData p = mul_intrin(s, t);
    const VData rem = sub_intrin(sub_intrin(sub_intrin(f, p), mul_sub_exact_intrin(s, t, p)), mul_intrin(s, t_lo)); // f - s (t + t_lo)
    const VData s_lo = mul_intrin(rem, inv);
    const VData z = mul_intrin(s, s);
    VData R; // sum_k 2/(2k+1) z^(k-1), k = 1 .. 12 (float: 6), to about 2^-72 (float: 2^-41); fdlibm's R reaches only 2^-58
    if constexpr (std::is_same<Real,float>::value) R = EvalPolynomial(z, 2.f/7, 2.f/9, 2.f/11, 2.f/13);
    else R = EvalPolynomial(z, 2./7, 2./9, 2./11, 2./13, 2./15, 2./17, 2./19, 2./21, 2./23, 2./25);
    R = fma_intrin(z, fma_intrin(z, R, set1_intrin<VData>((Real)2/5)), set1_intrin<VData>((Real)2/3)); // the two largest terms last, so that the rounding errors of the rest are scaled by z^2
    const VData Rz = mul_intrin(R, z);
    const VData sR = mul_intrin(s, Rz);
    const VData s2 = add_intrin(s, s);
    const VData u_hi = add_intrin(s2, sR); // log(1+f) = u_hi + u_lo; s_lo also changes s R by about 3 s_lo R z
    const VData u_lo = add_intrin(add_intrin(sub_intrin(s2, u_hi), sR), fma_intrin(mul_intrin(set1_intrin<VData>((Real)3), s_lo), Rz, add_intrin(s_lo, s_lo)));
    const VData eh = mul_intrin(e, set1_intrin<VData>(Ln2Split<Real>::hi)); // exact
    L_hi = add_intrin(eh, u_hi);
    L_lo = add_intrin(add_intrin(sub_intrin(eh, L_hi), u_hi), fma_intrin(e, set1_intrin<VData>(Ln2Split<Real>::lo), u_lo)); // |eh| >= |u_hi| unless e = 0
  }
  template <class VData> inline VData pow_poly_intrin(const VData& x, const VData& y) { // exp(y log|x|), log|x| = L_hi + L_lo to about 1.5x the precision of Real; std::pow at signs, zeros, inf, NaN
    using Real = typename VData::ScalarType;
    static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value, "Expected float or double!");
    const VData zero = zero_intrin<VData>();
    const VData one = set1_intrin<VData>((Real)1);
    const VData inf = set1_intrin<VData>((Real)INFINITY);
    const VData ax = fabs_intrin(x);

    // y log|x| = T_hi + T_lo; e^(T_hi + T_lo) = e^T_hi (1 + T_lo)
    VData e, f, L_hi, L_lo;
    log_split_intrin<false>(e, f, ax);
    log_hi_lo_intrin(L_hi, L_lo, e, f);
    VData T_hi = mul_intrin(y, L_hi);
    VData T_lo = add_intrin(mul_sub_exact_intrin(y, L_hi, T_hi), mul_intrin(y, L_lo));
    if (mask_count_intrin(positive_normal_mask_intrin(x) & comp_intrin<ComparisonType::lt>(fabs_intrin(T_hi), set1_intrin<VData>(exp_normal_lim<Real>()))) == VData::Size) {
      const VData r = exp_intrin(T_hi);
      return fma_intrin(r, T_lo, r);
    }

    // some x not positive and normal, or |y log x| large, or inf or NaN
    log_split_intrin<true>(e, f, ax);
    log_hi_lo_intrin(L_hi, L_lo, e, f);
    L_hi = select_intrin(comp_intrin<ComparisonType::eq>(ax, zero), unary_minus_intrin(inf), select_intrin(comp_intrin<ComparisonType::eq>(ax, inf), inf, L_hi));
    T_hi = mul_intrin(y, L_hi);
    T_lo = add_intrin(mul_sub_exact_intrin(y, L_hi, T_hi), mul_intrin(y, L_lo));
    T_lo = select_intrin(comp_intrin<ComparisonType::lt>(fabs_intrin(T_hi), set1_intrin<VData>((Real)4096)), T_lo, zero); // beyond, e^T_hi is inf or 0; also inf and NaN
    VData r = mul_intrin(exp_intrin(T_hi), add_intrin(one, T_lo)); // keeps inf

    // signs and special values, as std::pow
    const VData yr = rint_intrin(y);
    const VData yh = mul_intrin(y, set1_intrin<VData>((Real)0.5));
    const Mask<VData> y_odd = comp_intrin<ComparisonType::eq>(yr, y) & comp_intrin<ComparisonType::ne>(rint_intrin(yh), yh);
    const Mask<VData> x_neg = comp_intrin<ComparisonType::lt>(copysign_intrin(one, x), zero); // sign bit, also of -0
    r = select_intrin(y_odd & x_neg, unary_minus_intrin(r), r);
    r = select_intrin(comp_intrin<ComparisonType::lt>(x, zero) & comp_intrin<ComparisonType::ge>(x, set1_intrin<VData>(-std::numeric_limits<Real>::max())) & comp_intrin<ComparisonType::ne>(yr, y), set1_intrin<VData>((Real)NAN), r); // finite x < 0, y not an integer; -max as in positive_normal_mask_intrin
    r = select_intrin(comp_intrin<ComparisonType::eq>(ax, one) & comp_intrin<ComparisonType::eq>(fabs_intrin(y), inf), one, r);
    r = select_intrin(comp_intrin<ComparisonType::ne>(x, x), x, r);
    return select_intrin(comp_intrin<ComparisonType::eq>(y, zero) | comp_intrin<ComparisonType::eq>(x, one), one, r);
  }
  template <class VData> inline VData cbrt_poly_intrin(const VData& x) { // cbrt(2^e m) = 2^q cbrt(2^r m), e = 3q + r; a cubic first guess, then two Halley steps
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int, VData::Size>;
    static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value, "Expected float or double!");
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Int Bias = (((Int)1) << (sizeof(Real)*8 - SigBits - 2)) - 1;
    const VData zero = zero_intrin<VData>();
    const VData one = set1_intrin<VData>((Real)1);
    const VData two = set1_intrin<VData>((Real)2);
    const VData ax = fabs_intrin(x);

    VData e, f;
    log_split_intrin(e, f, ax);
    const VData q = floor_intrin(mul_intrin(add_intrin(e, set1_intrin<VData>((Real)0.5)), set1_intrin<VData>((Real)1/3)));
    const VData r = fma_intrin(q, set1_intrin<VData>((Real)-3), e); // 0, 1 or 2
    const VData a = mul_intrin(add_intrin(f, one), select_intrin(comp_intrin<ComparisonType::eq>(r, one), two, select_intrin(comp_intrin<ComparisonType::eq>(r, two), set1_intrin<VData>((Real)4), one))); // in [sqrt(1/2), 4 sqrt(2))
    VData y = fma_intrin(fma_intrin(fma_intrin(set1_intrin<VData>((Real)0.00572303598344627), a, set1_intrin<VData>((Real)-0.07497777771481334)), a, set1_intrin<VData>((Real)0.4499148623215124)), a, set1_intrin<VData>((Real)0.616489421488996)); // relative error 0.0093
    { // Halley: y - y (y^3 - a) / (2 y^3 + a)
      const VData t = mul_intrin(mul_intrin(y, y), y);
      y = sub_intrin(y, div_intrin(mul_intrin(y, sub_intrin(t, a)), fma_intrin(t, two, a)));
    }
    { // again, with the rounding errors of y^3 added to y^3 - a by FMA, so that cubes come out exact
      const VData y2 = mul_intrin(y, y);
      const VData t = mul_intrin(y2, y);
      const VData d = add_intrin(sub_intrin(t, a), fma_intrin(fma_intrin(y, y, unary_minus_intrin(y2)), y, fma_intrin(y2, y, unary_minus_intrin(t))));
      y = sub_intrin(y, div_intrin(mul_intrin(y, d), fma_intrin(t, two, a)));
    }
    const VData p2 = reinterpret_intrin<VData>(bitshiftleft_intrin(add_intrin(lrint_intrin<IntVec>(q), set1_intrin<IntVec>(Bias)), SigBits)); // 2^q
    const VData c = copysign_intrin(mul_intrin(y, p2), x);
    return select_intrin(comp_intrin<ComparisonType::gt>(ax, zero) & comp_intrin<ComparisonType::le>(ax, set1_intrin<VData>(std::numeric_limits<Real>::max())), c, x); // zeros, inf, NaN as they are; max as in positive_normal_mask_intrin
  }
  template <class VData> inline VData fmod_poly_intrin(const VData& x, const VData& y) { // |x| - q |y|, q = trunc(fl(|x|/|y|)), the quotient or one more; exact
    using Real = typename VData::ScalarType;
    static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value, "Expected float or double!");
    static constexpr Real lim = (Real)(((uint64_t)1) << (TypeTraits<Real>::SigBits - 1));
    const VData zero = zero_intrin<VData>();
    const VData ax = fabs_intrin(x);
    const VData ay = fabs_intrin(y);
    const VData qf = div_intrin(ax, ay);
    const VData q = trunc_intrin(qf);
    Mask<VData> vector_lanes = comp_intrin<ComparisonType::lt>(qf, set1_intrin<VData>(lim)); // the others: |x/y| >= lim, y = 0, inf x, NaN
#if defined(__FMA__) || defined(__FMA4__)
    VData r = fma_intrin(unary_minus_intrin(q), ay, ax);
#else // p + e = q |y| exactly (Dekker's product): with q below lim, the halves of q have SigBits - 1 bits together, so that each partial product is exact
    VData q_hi, q_lo, y_hi, y_lo;
    split_half_intrin(q_hi, q_lo, q);
    split_half_intrin(y_hi, y_lo, ay);
    const VData p = mul_intrin(q, ay);
    const VData e = add_intrin(add_intrin(add_intrin(sub_intrin(mul_intrin(q_hi, y_hi), p), mul_intrin(q_hi, y_lo)), mul_intrin(q_lo, y_hi)), mul_intrin(q_lo, y_lo));
    VData r = sub_intrin(sub_intrin(ax, p), e);
    vector_lanes = vector_lanes & comp_intrin<ComparisonType::lt>(ax, set1_intrin<VData>(std::numeric_limits<Real>::max() / 2)); // p is finite
#endif
    r = add_intrin(r, select_intrin(comp_intrin<ComparisonType::lt>(r, zero), ay, zero));
    r = copysign_intrin(select_intrin(comp_intrin<ComparisonType::lt>(qf, set1_intrin<VData>((Real)1)), ax, r), x); // |x| < |y|: x, also for inf y

    if (mask_count_intrin(vector_lanes) < VData::Size) { // the other lanes one element at a time
      union U {
        VData v;
        Real x[VData::Size];
      };
      union {
        MaskIntVec<VData> v;
        typename MaskIntVec<VData>::ScalarType q[VData::Size];
      } m_ = {mask2int_intrin(vector_lanes)};
      U x_u = {x};
      U y_u = {y};
      U r_u = {r};
      for (Integer i = 0; i < VData::Size; i++) {
        if (!m_.q[i]) r_u.x[i] = fmod(x_u.x[i], y_u.x[i]);
      }
      r = r_u.v;
    }
    return r;
  }
  template <class IntVec> inline IntVec div_int64_intrin(const IntVec& a, const IntVec& b) { // through double, exact for |a|, |b| < 2^51; other lanes one element at a time
    static_assert(std::is_same<typename IntVec::ScalarType, int64_t>::value, "Expected int64_t!");
    using RealVec = VecData<double, IntVec::Size>;
    union {
      double d;
      int64_t i;
    } C = {0x1.8p52}; // x + 1.5 2^52 has the integer x in its low bits, for |x| < 2^51
    const auto to_real = [&C](const IntVec& x) { return sub_intrin(reinterpret_intrin<RealVec>(add_intrin(x, set1_intrin<IntVec>(C.i))), set1_intrin<RealVec>(C.d)); };
    const auto to_int = [&C](const RealVec& x) { return sub_intrin(reinterpret_intrin<IntVec>(add_intrin(x, set1_intrin<RealVec>(C.d))), set1_intrin<IntVec>(C.i)); };
    IntVec q = to_int(trunc_intrin(div_intrin(to_real(a), to_real(b))));

    static constexpr int64_t lim = ((int64_t)1) << 51;
    const IntVec plim = set1_intrin<IntVec>(lim);
    const IntVec nlim = set1_intrin<IntVec>(-lim);
    const Mask<IntVec> ok = comp_intrin<ComparisonType::lt>(a, plim) & comp_intrin<ComparisonType::gt>(a, nlim) & comp_intrin<ComparisonType::lt>(b, plim) & comp_intrin<ComparisonType::gt>(b, nlim);
    if (mask_count_intrin(ok) < IntVec::Size) {
      union U {
        IntVec v;
        int64_t x[IntVec::Size];
      };
      U a_u = {a};
      U b_u = {b};
      U q_u = {q};
      for (Integer i = 0; i < IntVec::Size; i++) {
        if (!(a_u.x[i] < lim && a_u.x[i] > -lim && b_u.x[i] < lim && b_u.x[i] > -lim)) q_u.x[i] = a_u.x[i] / b_u.x[i];
      }
      q = q_u.v;
    }
    return q;
  }
  template <class IntVec> inline IntVec div_int32_intrin(const IntVec& a, const IntVec& b) { // through float, exact: |a| / |b| by two estimates from a reciprocal scaled down so that they are at most the quotient, then the sign
    static_assert(std::is_same<typename IntVec::ScalarType, int32_t>::value, "Expected int32_t!");
    using RealVec = VecData<float, IntVec::Size>;
    const IntVec ua = fabs_intrin(a); // as unsigned: 2^31 for INT32_MIN
    const IntVec ub = fabs_intrin(b);
    const RealVec rb = div_intrin(set1_intrin<RealVec>(1.0f - 0x1p-21f), fabs_intrin(convert_int2real_intrin<RealVec>(b)));
    const auto quot = [&rb](const IntVec& r) { return convert_intrin<IntVec>(mul_intrin(fabs_intrin(convert_int2real_intrin<RealVec>(r)), rb)); }; // from floor(|r| / |b|) - 1537 to floor(|r| / |b|), |r| as unsigned
    const IntVec q0 = quot(a);
    const IntVec r0 = sub_intrin(ua, mul_intrin(q0, ub)); // in [0, |a|]
    const IntVec q1 = quot(r0); // floor(r0 / |b|) - 1 or floor(r0 / |b|)
    const IntVec r1 = sub_intrin(r0, mul_intrin(q1, ub)); // in [0, 2 |b|)
    const IntVec q = add_intrin(add_intrin(q0, q1), add_intrin(set1_intrin<IntVec>(1), bitshiftright_intrin(sub_intrin(r1, ub), 31))); // + 1 where r1 >= |b|; r1 - |b| is in [-|b|, |b|) as signed
    const IntVec s = bitshiftright_intrin(xor_intrin(a, b), 31); // -1 where the quotient is negative
    return sub_intrin(xor_intrin(q, s), s);
  }
  template <class VData> inline VData exp_intrin(const VData& x) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U expx_, x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) expx_.x[i] = exp(x_.x[i]);
    return expx_.v;
  }
  template <class VData> inline VData log_intrin(const VData& x) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U logx_, x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) logx_.x[i] = log(x_.x[i]);
    return logx_.v;
  }
  template <class VData> inline VData pow_intrin(const VData& x, const VData& y) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U x_ = {x};
    U y_ = {y};
    for (Integer i = 0; i < VData::Size; i++) x_.x[i] = pow(x_.x[i], y_.x[i]);
    return x_.v;
  }
  template <Integer n> struct LogPolyCoeffs; // log(1+f) ~ f P(f) for f in [sqrt(1/2)-1, sqrt(2)-1], P of degree n, lowest degree first: Chebyshev fits; correct digits in log_poly_digits
  template <> struct LogPolyCoeffs<1> { static constexpr double c[] = {1.0185736191503331, -4.7559237223000092e-1}; };
  template <> struct LogPolyCoeffs<2> { static constexpr double c[] = {1.001256323625206, -5.2002197457901172e-1, 3.075125773580589e-1}; };
  template <> struct LogPolyCoeffs<3> { static constexpr double c[] = {9.9973106789736198e-1, -5.0232365594760311e-1, 3.536320724530045e-1, -2.2362563233616446e-1}; };
  template <> struct LogPolyCoeffs<4> { static constexpr double c[] = {9.9996217037960094e-1, -4.9950210920242519e-1, 3.3668781598037975e-1, -2.7010228270533408e-1, 1.7348631544940763e-1}; };
  template <> struct LogPolyCoeffs<5> { static constexpr double c[] = {1.0000037423879465, -4.998948024036182e-1, 3.3265905812990515e-1, -2.5433356361401844e-1, 2.196570849519215e-1, -1.40216232811136e-1}; };
  template <> struct LogPolyCoeffs<6> { static constexpr double c[] = {1.0000010273533488, -5.000087739481974e-1, 3.3313563286165821e-1, -2.4920336264665573e-1, 2.0525443619698836e-1, -1.8573840134002429e-1, 1.1657806143415021e-1}; };
  template <> struct LogPolyCoeffs<7> { static constexpr double c[] = {9.9999996811805992e-1, -5.0000375056289727e-1, 3.3334606023452299e-1, -2.4968906958952668e-1, 1.9913347882373827e-1, -1.7278206066426675e-1, 1.6126247905987103e-1, -9.895350736906514e-2}; };
  template <> struct LogPolyCoeffs<8> { static constexpr double c[] = {9.9999997413106708e-1, -4.9999996382123888e-1, 3.3334193286135078e-1, -2.5001352185496062e-1, 1.9955933358383247e-1, -1.6577992896798863e-1, 1.4977401280127375e-1, -1.4269258206218832e-1, 8.5333130829961519e-2}; };
  template <> struct LogPolyCoeffs<9> { static constexpr double c[] = {9.9999999938325496e-1, -4.9999988496142867e-1, 3.3333345349807466e-1, -2.500157909915946e-1, 2.0000939440373094e-1, -1.6608369414088408e-1, 1.4199650097916023e-1, -1.3266031360917081e-1, 1.2806610435006951e-1, -7.451186226757509e-2}; };
  template <> struct LogPolyCoeffs<10> { static constexpr double c[] = {1.0000000006010194, -4.9999999414017125e-1, 3.333330266101698e-1, -2.5000062374435879e-1, 2.0002535640193383e-1, -1.6666555575670009e-1, 1.4212291250421291e-1, -1.2420821262732774e-1, 1.194587822735682e-1, -1.1620648400270368e-1, 6.5723271585220237e-2}; };
  template <> struct LogPolyCoeffs<11> { static constexpr double c[] = {1.0000000000480121, -5.0000000308455383e-1, 3.3333330846280519e-1, -2.4999936585814818e-1, 2.0000169637942539e-1, -1.6670383652377297e-1, 1.4283798395270254e-1, -1.2410885665961422e-1, 1.1042736039719009e-1, -1.0898112577877182e-1, 1.0636542375579953e-1, -5.8456713350896371e-2}; };
  template <> struct LogPolyCoeffs<12> { static constexpr double c[] = {9.9999999998746515e-1, -5.0000000035951705e-1, 3.3333334249206593e-1, -2.4999992728449305e-1, 1.9999888139973632e-1, -1.666702457200908e-1, 1.429081173942899e-1, -1.2495435324992145e-1, 1.1006038270379548e-1, -9.9459961867196613e-2, 1.004720511888933e-1, -9.8044200677862028e-2, 5.2358885169237832e-2}; };
  template <> struct LogPolyCoeffs<13> { static constexpr double c[] = {9.9999999999808685e-1, -4.9999999992971808e-1, 3.3333333479574099e-1, -2.5000002048164391e-1, 1.999998295912731e-1, -1.6666490142581625e-1, 1.4286366102434627e-1, -1.2506640703431108e-1, 1.1102988279786334e-1, -9.8789698109920537e-2, 9.0545055509250049e-2, -9.3428814519668432e-2, 9.0897545094535187e-2, -4.7177590533593856e-2}; };
  template <> struct LogPolyCoeffs<14> { static constexpr double c[] = {1.0000000000002172, -4.9999999998512559e-1, 3.3333333311417252e-1, -2.5000000433128149e-1, 2.0000003805512754e-1, -1.6666632233794296e-1, 1.4285458260944881e-1, -1.2501075399501756e-1, 1.1119413052789981e-1, -9.9873661691216533e-2, 8.9541627892867124e-2, -8.3174352109032835e-2, 8.7504052326932136e-2, -8.4678764114581239e-2, 4.2728007304351816e-2}; };
  template <> struct LogPolyCoeffs<15> { static constexpr double c[] = {1.0000000000000617, -5.0000000000119965e-1, 3.3333333326964261e-1, -2.4999999950761742e-1, 2.000000104854377e-1, -1.666667282322684e-1, 1.4285651767534414e-1, -1.2499653070523752e-1, 1.1112762228270177e-1, -1.0010029904388152e-1, 9.0727875380103866e-2, -8.1813256819716009e-2, 7.6995122829726492e-2, -8.245025650742841e-2, 7.9206842719246771e-2, -3.8871612816231212e-2}; };
  inline constexpr double log_poly_digits[] = {1.65, 2.48, 3.31, 4.13, 4.95, 5.77, 6.58, 7.39, 8.19, 8.99, 9.79, 10.59, 11.38, 12.17, 12.96};
  inline constexpr Integer log_poly_degree(const Integer digits) { // the lowest degree of LogPolyCoeffs for the digits; 0 if none
    for (Integer n = 1; n <= 15; n++) {
      if (log_poly_digits[n - 1] >= digits) return n;
    }
    return 0;
  }
  inline constexpr Integer exp_taylor_order(const Integer digits) { // Taylor order k of e^r, |r| <= ln2/2: (ln2/2)^(k+1)/(k+1)! < 10^-digits
    double err = 0.34657359027997264;
    double lim = 1;
    for (Integer d = 0; d < digits; d++) lim *= 0.1;
    Integer k = 1;
    for (; err * 0.34657359027997264 / (k + 1) >= lim && k < 30; k++) err *= 0.34657359027997264 / (k + 1);
    return k;
  }
  inline constexpr Integer log_atanh_terms(const Integer digits) { // terms K of 2 atanh(s), s^2 <= 0.0295: 0.0295^K/(2K+1) < 10^-digits
    double zk = 0.0295;
    double lim = 1;
    for (Integer d = 0; d < digits; d++) lim *= 0.1;
    Integer K = 1;
    for (; zk / (2*K + 1) >= lim && K < 30; K++) zk *= 0.0295;
    return K;
  }
  template <class VData, class Coeffs, Integer... k> inline VData eval_coeffs_intrin(const VData& x, std::integer_sequence<Integer, k...>) { // sum Coeffs::c[k] x^k
    return EvalPolynomial(x, ((typename VData::ScalarType)Coeffs::c[k])...);
  }
  // sum c[Lo+k] x^k, k < Len, xp[j] = x^(2^j): split at the largest power of two below Len
  template <Integer Lo, Integer Len, class VData, Integer K, class CType, Integer N> inline VData eval_poly_estrin_intrin(const VData (&xp)[K], const CType (&c)[N]) {
    if constexpr (Len == 1) {
      return set1_intrin<VData>((typename VData::ScalarType)c[Lo]);
    } else {
      static constexpr Integer j = [] {
        Integer j_ = 0;
        while ((((Integer)2) << j_) < Len) j_++;
        return j_;
      }();
      static constexpr Integer H = ((Integer)1) << j;
      return fma_intrin(xp[j], eval_poly_estrin_intrin<Lo + H, Len - H>(xp, c), eval_poly_estrin_intrin<Lo, H>(xp, c));
    }
  }
  // sum c[k] x^k by Estrin's scheme, for any N
  template <class VData, class CType, Integer N> inline VData eval_poly_intrin(const VData& x, const CType (&c)[N]) {
    static constexpr Integer K = [] {
      Integer k = 1;
      while ((((Integer)1) << k) < N) k++;
      return k;
    }();
    VData xp[K];
    xp[0] = x;
    for (Integer j = 1; j < K; j++) xp[j] = mul_intrin(xp[j - 1], xp[j - 1]);
    return eval_poly_estrin_intrin<0, N>(xp, c);
  }
  // sum c[k] scale x^k by Horner's scheme: fewer operations than Estrin's, a longer chain of dependent operations
  template <class VData, class CType, Integer N> inline VData eval_poly_horner_intrin(const VData& x, const CType (&c)[N], const double scale = 1) {
    using Real = typename VData::ScalarType;
    VData p = set1_intrin<VData>((Real)(c[N - 1] * scale));
    for (Integer k = N - 2; k >= 0; k--) p = fma_intrin(p, x, set1_intrin<VData>((Real)(c[k] * scale)));
    return p;
  }
  // sum p[l][k] t_l^k, k < B, in each lane l: a polynomial per lane, with its coefficients in a row at p[l]
  template <Integer B, class VData> inline VData eval_poly_rows_intrin(const typename VData::ScalarType* const (&p)[VData::Size], const VData& t) {
    using Real = typename VData::ScalarType;
    union U {
      VData v;
      Real x[VData::Size];
    };
    const U t_ = {t};
    U r_ = {t};
    for (Integer l = 0; l < VData::Size; l++) {
      Real r = p[l][B - 1];
      for (Integer k = B - 2; k >= 0; k--) r = r * t_.x[l] + p[l][k];
      r_.x[l] = r;
    }
    return r_.v;
  }
  // P(x)/Q(x), one division, none for a constant Q
  template <class VData, class CType, Integer N, Integer M> inline VData eval_rational_intrin(const VData& x, const CType (&p)[N], const CType (&q)[M]) {
    if constexpr (M == 1) return mul_intrin(eval_poly_intrin(x, p), set1_intrin<VData>((typename VData::ScalarType)(1 / q[0])));
    else return div_intrin(eval_poly_intrin(x, p), eval_poly_intrin(x, q));
  }
  template <class VData, Integer... k> inline VData atanh_series_intrin(const VData& z, std::integer_sequence<Integer, k...>) { // sum 1/(2k+3) z^k
    return EvalPolynomial(z, ((typename VData::ScalarType)1 / (2*k + 3))...);
  }
  // e ln2 + log(1 + f) for f from log_split_intrin, to DIGITS digits: f P(f) without division up to 12.96 digits, else
  // 2 atanh(s), s = f/(2+f); DIGITS = -1: log_split_poly_intrin
  template <Integer DIGITS, class VData> inline VData approx_log_split_poly_intrin(const VData& e, const VData& f) {
    using Real = typename VData::ScalarType;
    if constexpr (DIGITS < 0) {
      return log_split_poly_intrin(e, f);
    } else if constexpr (log_poly_degree(DIGITS) > 0) {
      static constexpr Integer n = log_poly_degree(DIGITS);
      return fma_intrin(e, set1_intrin<VData>(const_ln2<Real>()), mul_intrin(f, eval_coeffs_intrin<VData, LogPolyCoeffs<n>>(f, std::make_integer_sequence<Integer, n + 1>())));
    } else { // K terms of the series, relative error below 0.0295^K/(2K+1)
      static constexpr Integer K = log_atanh_terms(DIGITS);
      const VData s2 = div_intrin(add_intrin(f, f), add_intrin(f, set1_intrin<VData>((Real)2))); // 2s
      const VData z = mul_intrin(mul_intrin(s2, s2), set1_intrin<VData>((Real)0.25));
      return fma_intrin(e, set1_intrin<VData>(const_ln2<Real>()), fma_intrin(mul_intrin(s2, z), atanh_series_intrin(z, std::make_integer_sequence<Integer, K - 1>()), s2));
    }
  }
  // log(x) to DIGITS digits for normal x > 0, by approx_log_split_poly_intrin
  template <Integer DIGITS, class VData> inline VData approx_log_normal_intrin(const VData& x) {
    VData e, f;
    log_split_intrin<false>(e, f, x);
    return approx_log_split_poly_intrin<DIGITS>(e, f);
  }
  template <Integer DIGITS, class VData> inline VData approx_log_intrin(const VData& x) { // approx_log_normal_intrin; log_intrin if some x is not positive, normal and finite
    if (mask_count_intrin(positive_normal_mask_intrin(x)) < VData::Size) return log_intrin(x);
    return approx_log_normal_intrin<DIGITS>(x);
  }
  template <Integer LOG_DIGITS, Integer ORDER, class VData> inline VData approx_pow_intrin(const VData& x, const VData& y) { // e^(y log x) with approx_log_normal_intrin<LOG_DIGITS> and approx_exp_intrin<ORDER>; pow_intrin if some x is not positive, normal and finite, or |y log x| >= 708.4 (float: 87.3)
    using Real = typename VData::ScalarType;
    const VData T = mul_intrin(y, approx_log_normal_intrin<LOG_DIGITS>(x));
    if (mask_count_intrin(positive_normal_mask_intrin(x) & comp_intrin<ComparisonType::lt>(fabs_intrin(T), set1_intrin<VData>(exp_normal_lim<Real>()))) < VData::Size) return pow_intrin(x, y);
    return approx_exp_intrin<ORDER, false>(T);
  }
  template <class VData> inline VData sqrt_intrin(const VData& x) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) x_.x[i] = sqrt(x_.x[i]);
    return x_.v;
  }
  template <class VData> inline VData rsqrt_intrin(const VData& x) {
    return div_intrin(set1_intrin<VData>((typename VData::ScalarType)1), sqrt_intrin(x));
  }
  template <class VData> inline VData rsqrt_full_intrin(const VData& x) { // 1/sqrt(x): rsqrt_approx_intrin at full precision for 2 min <= x <= max; if any x is outside, sqrt and division there
    using Real = typename VData::ScalarType;
    const VData r = rsqrt_approx_intrin<(Integer)(TypeTraits<Real>::SigBits*0.3010299957), VData>::eval(x);
    const Mask<VData> in_range = comp_intrin<ComparisonType::ge>(x, set1_intrin<VData>(2 * std::numeric_limits<Real>::min())) & comp_intrin<ComparisonType::le>(x, set1_intrin<VData>(std::numeric_limits<Real>::max()));
    if (mask_count_intrin(in_range) == VData::Size) return r;
    return select_intrin(in_range, r, div_intrin(set1_intrin<VData>((Real)1), sqrt_intrin(x))); // 0, inf, subnormal and negative x, NaN
  }
  template <class VData> inline VData fabs_intrin(const VData& x) {
    if constexpr (TypeTraits<typename VData::ScalarType>::Type == DataType::Real) {
      return andnot_intrin(x, set1_intrin<VData>((typename VData::ScalarType)-0.0)); // clears the sign bit
    } else {
      return max_intrin(x, unary_minus_intrin(x));
    }
  }
  template <class VData> inline VData floor_intrin(const VData& x) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) x_.x[i] = floor(x_.x[i]);
    return x_.v;
  }
  template <class VData> inline VData ceil_intrin(const VData& x) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) x_.x[i] = ceil(x_.x[i]);
    return x_.v;
  }
  template <class VData> inline VData copysign_intrin(const VData& x, const VData& y) {
    const VData sign = set1_intrin<VData>((typename VData::ScalarType)-0.0);
    return or_intrin(andnot_intrin(x, sign), and_intrin(y, sign));
  }
  template <class VData> inline Mask<VData> isnan_intrin(const VData& x) {
    using IntType = typename IntegerType<sizeof(typename VData::ScalarType)>::value;
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } x_ = {x};
    union {
      MaskIntVec<VData> v;
      IntType q[VData::Size];
    } m_;
    for (Integer i = 0; i < VData::Size; i++) m_.q[i] = (isnan(x_.x[i]) ? ~(IntType)0 : (IntType)0);
    return int2mask_intrin<VData>(m_.v);
  }
  template <bool SpecialValues = true, class VData> inline VData atan2_intrin(const VData& y, const VData& x) { // SpecialValues: x, y both infinite or both zero
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData zero = zero_intrin<VData>();
      const VData one = set1_intrin<VData>((Real)1);
      const VData ay = fabs_intrin(y);
      const VData ax = fabs_intrin(x);
      const Mask<VData> swap = comp_intrin<ComparisonType::gt>(ay, ax); // angle above pi/4
      VData num = select_intrin(swap, ax, ay);
      VData den = select_intrin(swap, ay, ax);
      if constexpr (SpecialValues) { // both infinite: atan(1)
        const VData inf = set1_intrin<VData>((Real)INFINITY);
        const Mask<VData> inf_inf = comp_intrin<ComparisonType::eq>(ay, inf) & comp_intrin<ComparisonType::eq>(ax, inf);
        num = select_intrin(inf_inf, one, num);
        den = select_intrin(inf_inf, one, den);
      }

      // atan(num/den) with num/den in [0, 1]: reduce to |t| <= tan(pi/8), then Cephes atan rational
      const Mask<VData> big = comp_intrin<ComparisonType::gt>(num, mul_intrin(den, set1_intrin<VData>((Real)0.41421356237309504880)));
      const VData n2 = select_intrin(big, sub_intrin(num, den), num);
      VData d2 = select_intrin(big, add_intrin(num, den), den);
      if constexpr (SpecialValues) d2 = select_intrin(comp_intrin<ComparisonType::eq>(d2, zero), one, d2); // 0 for atan2(0, 0)
      const VData t = div_intrin(n2, d2);
      const VData z = mul_intrin(t, t);
      VData p = set1_intrin<VData>((Real)-8.750608600031904122785e-1);
      p = fma_intrin(p, z, set1_intrin<VData>((Real)-1.615753718733365076637e1));
      p = fma_intrin(p, z, set1_intrin<VData>((Real)-7.500855792314704667340e1));
      p = fma_intrin(p, z, set1_intrin<VData>((Real)-1.228866684490136173410e2));
      p = fma_intrin(p, z, set1_intrin<VData>((Real)-6.485021904942025371773e1));
      VData q = add_intrin(z, set1_intrin<VData>((Real)2.485846490142306297962e1));
      q = fma_intrin(q, z, set1_intrin<VData>((Real)1.650270098316988542046e2));
      q = fma_intrin(q, z, set1_intrin<VData>((Real)4.328810604912902668951e2));
      q = fma_intrin(q, z, set1_intrin<VData>((Real)4.853903996359136964868e2));
      q = fma_intrin(q, z, set1_intrin<VData>((Real)1.945506571482613964425e2));
      VData r = fma_intrin(mul_intrin(t, z), div_intrin(p, q), t);
      r = add_intrin(r, select_intrin(big, set1_intrin<VData>((Real)0.78539816339744830962), zero));

      // quadrant
      r = select_intrin(swap, sub_intrin(set1_intrin<VData>((Real)1.57079632679489661923), r), r);
      r = select_intrin(comp_intrin<ComparisonType::lt>(copysign_intrin(one, x), zero), sub_intrin(set1_intrin<VData>((Real)3.14159265358979323846), r), r);
      return copysign_intrin(r, y);
    } else {
      union U {
        VData v;
        Real x[VData::Size];
      };
      U y_ = {y};
      U x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) y_.x[i] = atan2(y_.x[i], x_.x[i]);
      return y_.v;
    }
  }
  template <class VData> inline VData atan_intrin(const VData& x) { // atan2 without SpecialValues: its arguments are never both infinite or both zero, here and below
    return atan2_intrin<false>(x, set1_intrin<VData>((typename VData::ScalarType)1));
  }
  template <class VData> inline VData asin_intrin(const VData& x) {
    const VData one = set1_intrin<VData>((typename VData::ScalarType)1);
    return atan2_intrin<false>(x, sqrt_intrin(mul_intrin(sub_intrin(one, x), add_intrin(one, x))));
  }
  template <class VData> inline VData acos_intrin(const VData& x) {
    const VData one = set1_intrin<VData>((typename VData::ScalarType)1);
    return atan2_intrin<false>(sqrt_intrin(mul_intrin(sub_intrin(one, x), add_intrin(one, x))), x);
  }

  template <class VData> inline VData trunc_intrin(const VData& x) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) x_.x[i] = trunc(x_.x[i]);
    return x_.v;
  }
  template <class VData> inline VData round_intrin(const VData& x) { // halfway cases away from zero
    using Real = typename VData::ScalarType;
    const VData t = trunc_intrin(x);
    const Mask<VData> up = comp_intrin<ComparisonType::ge>(fabs_intrin(sub_intrin(x, t)), set1_intrin<VData>((Real)0.5));
    return add_intrin(t, select_intrin(up, copysign_intrin(set1_intrin<VData>((Real)1), x), zero_intrin<VData>()));
  }
  template <class VData> inline Mask<VData> isinf_intrin(const VData& x) {
    return comp_intrin<ComparisonType::eq>(fabs_intrin(x), set1_intrin<VData>((typename VData::ScalarType)INFINITY));
  }
  template <class VData> inline Mask<VData> isfinite_intrin(const VData& x) {
    return comp_intrin<ComparisonType::lt>(fabs_intrin(x), set1_intrin<VData>((typename VData::ScalarType)INFINITY));
  }

  template <bool AvoidOverflow = true, class VData> inline VData hypot_intrin(const VData& x, const VData& y) {
    using Real = typename VData::ScalarType;
    if constexpr (!AvoidOverflow) {
      return sqrt_intrin(fma_intrin(x, x, mul_intrin(y, y)));
    } else { // m sqrt(1 + (s/m)^2), m = max(|x|,|y|)
      const VData zero = zero_intrin<VData>();
      const VData inf = set1_intrin<VData>((Real)INFINITY);
      const VData a = fabs_intrin(x);
      const VData b = fabs_intrin(y);
      const VData m = max_intrin(a, b);
      const VData r = div_intrin(min_intrin(a, b), m);
      VData h = mul_intrin(m, sqrt_intrin(fma_intrin(r, r, set1_intrin<VData>((Real)1))));
      h = select_intrin(comp_intrin<ComparisonType::eq>(m, zero), zero, h);
      h = select_intrin(isnan_intrin(add_intrin(a, b)), add_intrin(a, b), h);
      return select_intrin(comp_intrin<ComparisonType::eq>(a, inf) | comp_intrin<ComparisonType::eq>(b, inf), inf, h); // inf even with NaN, as std::hypot
    }
  }

  template <class VData> inline VData exp2_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) { // 2^f 2^n1 2^n2, f = x - n, n = n1 + n2
      using Int = typename IntegerType<sizeof(Real)>::value;
      using IntVec = VecData<Int,VData::Size>;
      static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
      static constexpr Int max_exp = (((Int)1) << (sizeof(Real)*8 - SigBits - 2)) - 1;
      const VData lim = set1_intrin<VData>((Real)(2*(max_exp-1)));
      const VData xc = min_intrin(lim, max_intrin(unary_minus_intrin(lim), x)); // keeps NaN, the second operand
      const VData n = rint_intrin(xc);
      const VData e = approx_exp_intrin<(Integer)(SigBits/3.8), false>(mul_intrin(sub_intrin(xc, n), set1_intrin<VData>(const_ln2<Real>()))); // |x| <= ln2/2: no range check
      const IntVec ni = lrint_intrin<IntVec>(n);
      const IntVec n1 = bitshiftright_intrin(ni, 1);
      const IntVec n2 = sub_intrin(ni, n1);
      const VData p1 = reinterpret_intrin<VData>(bitshiftleft_intrin(add_intrin(n1, set1_intrin<IntVec>(max_exp)), SigBits));
      const VData p2 = reinterpret_intrin<VData>(bitshiftleft_intrin(add_intrin(n2, set1_intrin<IntVec>(max_exp)), SigBits));
      return mul_intrin(mul_intrin(e, p1), p2);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = pow((Real)2, x_.x[i]);
      return x_.v;
    }
  }
  template <Integer ORDER, bool RangeCheck = true, class VData> inline VData approx_exp10_intrin(const VData& x) { // as approx_exp_intrin
    return approx_exp_intrin<ORDER, RangeCheck, ExpArg::Base10>(x, x);
  }
  template <class VData> inline VData exp10_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      return approx_exp10_intrin<(Integer)(TypeTraits<Real>::SigBits/3.8)>(x);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = pow((Real)10, x_.x[i]);
      return x_.v;
    }
  }
  template <class VData> inline VData log2_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      return mul_intrin(log_intrin(x), set1_intrin<VData>((Real)1.44269504088896340736));
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = log2(x_.x[i]);
      return x_.v;
    }
  }
  template <class VData> inline VData log10_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      return mul_intrin(log_intrin(x), set1_intrin<VData>((Real)0.43429448190325182765));
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = log(x_.x[i]) / log((Real)10);
      return x_.v;
    }
  }

  // expm1(r) = r + r^2/2 + r^3 P(r), |r| <= ln2/2, P of degree n, lowest degree first: minimax for the relative error
  // of expm1
  template <Integer n> struct Expm1PolyCoeffs;
  template <> struct Expm1PolyCoeffs<3> { static constexpr double c[] = {0.16666548755272226, 0.04166685412507088, 0.008366034281379001, 0.0013898223046526785}; }; // 2^-26.1
  template <> struct Expm1PolyCoeffs<8> { static constexpr double c[] = {0.16666666666666602, 0.041666666666620035, 0.00833333333338835, 0.0013888888920480966, 0.00019841269709401925, 2.4801516389454372e-05, 2.7557425050216553e-06, 2.762220823946014e-07, 2.5045897154552257e-08}; }; // 2^-55.2
  // e^x - 1 = 2^n expm1(r) + (2^n - 1), x = n ln2 + r, rounded once; 2^n - 1 is exact for |n| <= SigBits
  template <class VData> inline VData expm1_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Int = typename IntegerType<sizeof(Real)>::value;
      using IntVec = VecData<Int,VData::Size>;
      static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
      static constexpr Int Bias = (((Int)1) << (sizeof(Real)*8 - SigBits - 2)) - 1;
      static constexpr Real magic = (Real)1.5 * pow<SigBits,Real>((Real)2) + (Real)Bias; // y + magic: y rounded to an integer n, with n + Bias in the low bits
      static constexpr Real lo_lim = -(Real)(SigBits + 4) * const_ln2<Real>(); // below, e^x - 1 rounds to -1
      static constexpr Real hi_lim = exp_normal_lim<Real>(); // up to here, 2^n is normal
      using Coeffs = Expm1PolyCoeffs<std::is_same<Real,float>::value ? 3 : 8>;

      const VData xc = min_intrin(set1_intrin<VData>(hi_lim), max_intrin(set1_intrin<VData>(lo_lim), x)); // keeps NaN, the second operand
      const VData t = fma_intrin(xc, set1_intrin<VData>(1 / const_ln2<Real>()), set1_intrin<VData>(magic));
      const VData n = sub_intrin(t, set1_intrin<VData>(magic));
      VData r = fma_intrin(n, set1_intrin<VData>(-Ln2Split<Real>::hi), xc); // n Ln2Split::hi is exact
      r = fma_intrin(n, set1_intrin<VData>(-Ln2Split<Real>::lo), r);
      const VData er = fma_intrin(mul_intrin(r, r), fma_intrin(r, eval_poly_intrin(r, Coeffs::c), set1_intrin<VData>((Real)0.5)), r); // expm1(r)
      const VData p2 = reinterpret_intrin<VData>(bitshiftleft_intrin(reinterpret_intrin<IntVec>(t), SigBits)); // 2^n
      const VData e = select_intrin(comp_intrin<ComparisonType::eq>(x, zero_intrin<VData>()), x, fma_intrin(p2, er, sub_intrin(p2, set1_intrin<VData>((Real)1)))); // x for +-0
      const Mask<VData> big = comp_intrin<ComparisonType::gt>(x, set1_intrin<VData>(hi_lim));
      if (mask_count_intrin(big) == 0) return e;
      return select_intrin(big, exp_intrin(x), e); // e^x - 1 rounds to e^x
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = expm1(x_.x[i]);
      return x_.v;
    }
  }
  // 1 + x = u + c, u = 1 + x rounded, c its rounding error; u = 2^e (1+f): log(1+x) = e ln2 + log(1 + f + c 2^-e)
  template <class VData> inline VData log1p_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData one = set1_intrin<VData>((Real)1);
      const VData u = add_intrin(one, x);
      const VData c = sub_intrin(x, sub_intrin(u, one)); // 0 where u >= 2^(Bias-1)
      VData e, f;
      log_split_intrin<false>(e, f, u);
      const VData r = select_intrin(comp_intrin<ComparisonType::eq>(u, one), x, log_split_poly_intrin(e, log_split_add_intrin(e, f, c))); // x where u = 1, also +-0
      if (mask_count_intrin(positive_normal_mask_intrin(u)) == VData::Size) return r;

      // u = 0 (x = -1), negative (x < -1), inf or NaN
      const VData rs = select_intrin(comp_intrin<ComparisonType::eq>(u, zero_intrin<VData>()), set1_intrin<VData>((Real)-INFINITY), select_intrin(comp_intrin<ComparisonType::eq>(u, set1_intrin<VData>((Real)INFINITY)), u, r));
      return select_intrin(comp_intrin<ComparisonType::ge>(u, zero_intrin<VData>()), rs, set1_intrin<VData>((Real)NAN));
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = log1p(x_.x[i]);
      return x_.v;
    }
  }

  template <class Real> inline Real log1p_generic(const Real w) { // log(1 + w) as log(u) w/(u-1), u = 1 + w rounded
    const Real u = 1 + w;
    return (u == 1 ? w : log(u) * w / (u - 1));
  }
  // sqrt(g + g_err) - s for s = sqrt(g) rounded and a small g_err: one Newton step, scaled by a 3-digit estimate of
  // 1/sqrt(g)
  template <class VData> inline VData sqrt_err_intrin(const VData& s, const VData& g, const VData& g_err) {
    using Real = typename VData::ScalarType;
    const VData r = rsqrt_approx_intrin<3, VData>::eval(g);
    return mul_intrin(sub_intrin(g_err, mul_sub_exact_intrin(s, s, g)), mul_intrin(r, set1_intrin<VData>((Real)0.5)));
  }
  // asinh(x) = x + x^3 Q(x^2), |x| <= 1/2: minimax Q of degree n, lowest degree first
  template <Integer n> struct AsinhPolyCoeffs;
  template <> struct AsinhPolyCoeffs<1> { static constexpr double c[] = {-0.1657223575385526, 0.06197997143321759}; };
  template <> struct AsinhPolyCoeffs<2> { static constexpr double c[] = {-0.1666058688864166, 0.07345595446324983, -0.033061359122391186}; };
  template <> struct AsinhPolyCoeffs<3> { static constexpr double c[] = {-0.16666288130295978, 0.07484758508588933, -0.04269933336328035, 0.020121986705147754}; };
  template <> struct AsinhPolyCoeffs<4> { static constexpr double c[] = {-0.1666664358086002, 0.0749865214096116, -0.04438451680631739, 0.02816912441132975, -0.013237527570641}; };
  template <> struct AsinhPolyCoeffs<5> { static constexpr double c[] = {-0.16666665278425186, 0.0749988928179296, -0.044613250291783146, 0.030012952236921492, -0.019979983993331386, 0.009167712263326848}; };
  template <> struct AsinhPolyCoeffs<6> { static constexpr double c[] = {-0.16666666584036027, 0.07499991375775812, -0.04463979805874086, 0.03033004859448526, -0.02189360874356594, 0.01484580274900371, -0.006585170172216225}; };
  template <> struct AsinhPolyCoeffs<7> { static constexpr double c[] = {-0.16666666661786464, 0.07499999354676105, -0.04464256445281436, 0.030375485787985487, -0.022292495478573295, 0.016769337639984463, -0.011390610035294, 0.004860905739454674}; };
  template <> struct AsinhPolyCoeffs<8> { static constexpr double c[] = {-0.16666666666380198, 0.07499999953208214, -0.04464283075516439, 0.030381211726470198, -0.02236056885603899, 0.017240727971745248, -0.013283400656374015, 0.008945810937544606, -0.003664837779870902}; };
  template <> struct AsinhPolyCoeffs<9> { static constexpr double c[] = {-0.16666666666649935, 0.07499999996692068, -0.04464285487410398, 0.03038186715686761, -0.022370638802534055, 0.01733412175430722, -0.013816769658281845, 0.01078045378128307, -0.007150650856619543, 0.0028101369867868362}; };
  template <> struct AsinhPolyCoeffs<10> { static constexpr double c[] = {-0.16666666666665694, 0.07499999999770973, -0.044642856955217046, 0.030381936758338737, -0.0223719754666556, 0.017349989917625097, -0.013937130674525018, 0.011364956876270108, -0.008909098636713858, 0.0057946862371079984, -0.002184703164205562}; };
  inline constexpr double asinh_poly_digits[] = {4.69, 6.13, 7.53, 8.9, 10.26, 11.6, 12.93, 14.26, 15.57, 16.81}; // degrees 1 to 10
  // asinh(x) = sign(x) log(|x| + sqrt(x^2 + 1)), with the rounding errors for |x| < 5/4; x + x^3 Q(x^2) for |x| < 1/2;
  // log(2|x|) from 2^(SigBits/2). digits = -1: full. FullRange = false: |x| < 2^(SigBits/2)
  template <Integer digits = -1, bool FullRange = true, class VData> inline VData asinh_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      static constexpr Real lim = pow<TypeTraits<Real>::SigBits/2,Real>((Real)2); // below, 1 - g is exact
      const VData one = set1_intrin<VData>((Real)1);
      const VData sgn = and_intrin(x, set1_intrin<VData>((Real)-0.0));
      const VData ax = xor_intrin(x, sgn);
      const Mask<VData> small = comp_intrin<ComparisonType::lt>(ax, set1_intrin<VData>((Real)0.5));
      const Integer n_small = mask_count_intrin(small);
      VData r = ax;
      if (n_small < VData::Size) {
        const VData g = fma_intrin(ax, ax, one);
        const VData s = sqrt_intrin(g);
        const VData v = add_intrin(ax, s);
        VData e, f;
        log_split_intrin<false>(e, f, v);
        if (mask_count_intrin(comp_intrin<ComparisonType::lt>(ax, set1_intrin<VData>((Real)1.25))) == 0) {
          r = approx_log_split_poly_intrin<digits>(e, f);
        } else {
          const VData g_err = mul_sub_exact_intrin(ax, ax, sub_intrin(g, one)); // g - 1 is exact
          const VData c = add_intrin(sub_intrin(ax, sub_intrin(v, s)), sqrt_err_intrin(s, g, g_err)); // s >= |x|
          r = approx_log_split_poly_intrin<digits>(e, log_split_add_intrin(e, f, c));
        }
      }
      if (n_small > 0) {
        const VData x2 = mul_intrin(ax, ax);
        r = select_intrin(small, fma_intrin(mul_intrin(ax, x2), eval_poly_horner_intrin(x2, AsinhPolyCoeffs<poly_degree(asinh_poly_digits, 1, (std::is_same<Real,float>::value ? 4 : 10), digits)>::c), ax), r);
      }
      if (FullRange && mask_count_intrin(comp_intrin<ComparisonType::lt>(ax, set1_intrin<VData>(lim))) < VData::Size) { // |x| >= lim, inf, NaN
        const VData big = add_intrin(log_intrin(ax), set1_intrin<VData>(const_ln2<Real>()));
        r = select_intrin(comp_intrin<ComparisonType::ge>(ax, set1_intrin<VData>(lim)), big, select_intrin(comp_intrin<ComparisonType::lt>(ax, set1_intrin<VData>(lim)), r, ax));
      }
      return xor_intrin(r, sgn);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        const Real a = fabs(x_.x[i]);
        const Real r = (a > pow<TypeTraits<Real>::SigBits/2,Real>((Real)2) ? log(a) + const_ln2<Real>() : log1p_generic(a + a * a / (1 + sqrt(1 + a * a))));
        x_.x[i] = (r == 0 ? x_.x[i] : (x_.x[i] < 0 ? -r : r));
      }
      return x_.v;
    }
  }
  // acosh(x) = log(x + sqrt(x^2 - 1)), for x < 2 with the rounding errors of x^2 - 1, the square root and the sum; log(2x)
  // from 2^(SigBits/2). digits = -1: full. FullRange = false: 1 < x < 2^(SigBits/2)
  template <Integer digits = -1, bool FullRange = true, class VData> inline VData acosh_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      static constexpr Real lim = pow<TypeTraits<Real>::SigBits/2,Real>((Real)2); // below, ph - 1 is exact
      const VData one = set1_intrin<VData>((Real)1);
      const VData g = (fused_fma<VData> ? fma_intrin(x, x, unary_minus_intrin(one)) : mul_intrin(sub_intrin(x, one), add_intrin(x, one))); // x - 1 exact for x <= 2
      const VData s = sqrt_intrin(g);
      const VData v = add_intrin(x, s);
      VData e, f;
      log_split_intrin<false>(e, f, v);
      if (mask_count_intrin(comp_intrin<ComparisonType::lt>(x, set1_intrin<VData>((Real)2))) > 0) {
        VData g_err;
        if constexpr (fused_fma<VData>) {
          const VData ph = mul_intrin(x, x);
          g_err = add_intrin(sub_intrin(sub_intrin(ph, one), g), mul_sub_exact_intrin(x, x, ph)); // x^2 = ph + pl
        } else {
          const VData z = add_intrin(x, one);
          g_err = fma_intrin(sub_intrin(x, one), sub_intrin(one, sub_intrin(z, x)), mul_sub_exact_intrin(sub_intrin(x, one), z, g)); // (x - 1)(z + z_err) - g
        }
        f = log_split_add_intrin(e, f, add_intrin(sub_intrin(s, sub_intrin(v, x)), sqrt_err_intrin(s, g, g_err))); // x >= s
      }
      const VData r = approx_log_split_poly_intrin<digits>(e, f);
      const VData vlim = set1_intrin<VData>(lim);
      if (!FullRange || mask_count_intrin(comp_intrin<ComparisonType::gt>(x, one) & comp_intrin<ComparisonType::lt>(x, vlim)) == VData::Size) return r;

      // x = 1, x < 1, x >= lim, inf or NaN
      const VData big = add_intrin(log_intrin(x), set1_intrin<VData>(const_ln2<Real>()));
      const VData rs = select_intrin(comp_intrin<ComparisonType::ge>(x, vlim), big, select_intrin(comp_intrin<ComparisonType::gt>(x, one), r, zero_intrin<VData>()));
      return select_intrin(comp_intrin<ComparisonType::ge>(x, one), rs, set1_intrin<VData>((Real)NAN));
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        const Real y = x_.x[i] - 1;
        x_.x[i] = (x_.x[i] > pow<TypeTraits<Real>::SigBits/2,Real>((Real)2) ? log(x_.x[i]) + const_ln2<Real>() : log1p_generic(y + sqrt(y * (y + 2))));
      }
      return x_.v;
    }
  }
  // atanh(x) = x + x^3 Q(x^2), |x| <= 1/2: minimax Q of degree n, lowest degree first
  template <Integer n> struct AtanhPolyCoeffs;
  template <> struct AtanhPolyCoeffs<1> { static constexpr double c[] = {0.3280017677789941, 0.26360947271605506}; };
  template <> struct AtanhPolyCoeffs<2> { static constexpr double c[] = {0.33384489990285793, 0.18822076209106758, 0.21632268596900586}; };
  template <> struct AtanhPolyCoeffs<3> { static constexpr double c[] = {0.3332873471565446, 0.20171765188740268, 0.12336189734808355, 0.1933203138409883}; };
  template <> struct AtanhPolyCoeffs<4> { static constexpr double c[] = {0.33333730030328274, 0.199782164503241, 0.14669143165616635, 0.08243703056405131, 0.18174007749956445}; };
  template <> struct AtanhPolyCoeffs<5> { static constexpr double c[] = {0.333333000715283, 0.20002518278104284, 0.1422276216534014, 0.11823729600830118, 0.051358254086960146, 0.17669116831155193}; };
  template <> struct AtanhPolyCoeffs<6> { static constexpr double c[] = {0.33333336064815317, 0.1999972753891939, 0.1429485304062475, 0.10967181892218987, 0.10282689994774834, 0.024521907743698612, 0.17594255725078464}; };
  template <> struct AtanhPolyCoeffs<7> { static constexpr double c[] = {0.3333333311255742, 0.20000028044309062, 0.14284501834136498, 0.111362930679137, 0.08804667126869078, 0.09552867840528878, -0.0008787088266464228, 0.1783638880901236}; };
  template <> struct AtanhPolyCoeffs<8> { static constexpr double c[] = {0.3333333335095571, 0.19999997223693067, 0.14285864413557667, 0.1110715043810897, 0.09149580462923146, 0.0717360160644831, 0.09433900326839616, -0.02653100494315919, 0.18335258800885093}; };
  template <> struct AtanhPolyCoeffs<9> { static constexpr double c[] = {0.33333333331940895, 0.20000000266379014, 0.1428569668719607, 0.11111684609565606, 0.09080229565292998, 0.07814304300984987, 0.05787274326264876, 0.09852763312464456, -0.053626364971385954, 0.1905893319946147}; };
  template <> struct AtanhPolyCoeffs<10> { static constexpr double c[] = {0.3333333333344244, 0.19999999975090232, 0.14285716258740408, 0.11111033398221813, 0.0909268037247804, 0.07667067905556711, 0.06900252171658791, 0.044642957368387484, 0.10804203109337505, -0.0831328856878387, 0.1999221341008489}; };
  template <> struct AtanhPolyCoeffs<11> { static constexpr double c[] = {0.33333333333324844, 0.2000000000227984, 0.14285714072552522, 0.11111121083181769, 0.09090636585700751, 0.07697027490259652, 0.06612503367535068, 0.06302446967547175, 0.03064016482306227, 0.12326737634941974, -0.11593193499138618, 0.21130716453134404}; };
  template <> struct AtanhPolyCoeffs<12> { static constexpr double c[] = {0.3333333333333399, 0.19999999999795112, 0.14285714308034816, 0.11111109888609831, 0.09090948485686924, 0.07691494750224992, 0.06677944404159813, 0.05774395536479686, 0.059824096280954496, 0.014564543866772328, 0.14492849982535477, -0.15289542825875824, 0.22477676335121333}; };
  inline constexpr double atanh_poly_digits[] = {3.9, 5.18, 6.42, 7.65, 8.86, 10.07, 11.26, 12.45, 13.64, 14.82, 16.0, 16.96}; // degrees 1 to 12
  // atanh(x) = x + x^3 P(x^2)/Q(x^2), |x| <= 1/2: Cephes's double coefficients, lowest degree first
  struct AtanhRationalCoeffs {
    static constexpr double p[] = {-3.09092539379866942570E1, 6.54566728676544377376E1, -4.61252884198732692637E1, 1.20426861384072379242E1, -8.54074331929669305196E-1};
    static constexpr double q[] = {-9.27277618139601130017E1, 2.52006675691344555838E2, -2.49839401325893582852E2, 1.08938092147140262656E2, -1.95638849376911654834E1, 1.0};
  };
  // atanh(x) = sign(x) log(q)/2, q = (1+|x|)/(1-|x|): for double with fused FMA with the rounding errors of 1 +- |x| and q;
  // else without, and x + x^3 Q(x^2) for |x| < 1/2 (P/Q for double). digits = -1: full. FullRange = false: |x| < 1
  template <Integer digits = -1, bool FullRange = true, class VData> inline VData atanh_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData one = set1_intrin<VData>((Real)1);
      const VData sgn = and_intrin(x, set1_intrin<VData>((Real)-0.0));
      const VData ax = xor_intrin(x, sgn);
      VData r;
      if constexpr (std::is_same<Real,double>::value && fused_fma<VData>) { // faster than the polynomial here
        const VData a = add_intrin(one, ax);
        const VData ea = sub_intrin(ax, sub_intrin(a, one)); // a + ea = 1 + |x|
        const VData b = sub_intrin(one, ax);
        const VData eb = sub_intrin(sub_intrin(one, b), ax); // b + eb = 1 - |x|
        const VData ib = div_intrin(one, b);
        const VData q = mul_intrin(a, ib);
        const VData c = mul_intrin(sub_intrin(ea, fma_intrin(q, eb, mul_sub_exact_intrin(q, b, a))), ib); // (a + ea)/(b + eb) - q
        VData e, f;
        log_split_intrin<false>(e, f, q);
        r = mul_intrin(approx_log_split_poly_intrin<digits>(e, log_split_add_intrin(e, f, c)), set1_intrin<VData>((Real)0.5));
      } else {
        const Mask<VData> small = comp_intrin<ComparisonType::lt>(ax, set1_intrin<VData>((Real)0.5));
        const Integer n_small = mask_count_intrin(small);
        r = ax;
        if (n_small < VData::Size) {
          VData e, f;
          log_split_intrin<false>(e, f, div_intrin(add_intrin(one, ax), sub_intrin(one, ax)));
          r = mul_intrin(approx_log_split_poly_intrin<digits>(e, f), set1_intrin<VData>((Real)0.5));
        }
        if (n_small > 0) {
          const VData x2 = mul_intrin(ax, ax);
          VData t;
          if constexpr (std::is_same<Real,double>::value && digits < 0) { // a division in place of 12 products without fused FMA
            t = div_intrin(eval_poly_horner_intrin(x2, AtanhRationalCoeffs::p), eval_poly_horner_intrin(x2, AtanhRationalCoeffs::q));
          } else {
            t = eval_poly_horner_intrin(x2, AtanhPolyCoeffs<poly_degree(atanh_poly_digits, 1, (std::is_same<Real,float>::value ? 5 : 12), digits)>::c);
          }
          r = select_intrin(small, fma_intrin(mul_intrin(ax, x2), t, ax), r);
        }
      }
      r = xor_intrin(r, sgn);
      if (!FullRange || mask_count_intrin(comp_intrin<ComparisonType::lt>(ax, one)) == VData::Size) return r;

      // |x| = 1, |x| > 1 or NaN
      return select_intrin(comp_intrin<ComparisonType::lt>(ax, one), r, select_intrin(comp_intrin<ComparisonType::eq>(ax, one), or_intrin(set1_intrin<VData>((Real)INFINITY), sgn), set1_intrin<VData>((Real)NAN)));
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        const Real a = fabs(x_.x[i]);
        const Real r = (a == 1 ? (Real)INFINITY : log1p_generic(2 * a / (1 - a)) / 2);
        x_.x[i] = (x_.x[i] < 0 ? -r : (r == 0 ? x_.x[i] : r));
      }
      return x_.v;
    }
  }

  // erf(x) = x + x Q(x^2), |x| <= 1: minimax Q of degree n, lowest degree first
  template <Integer n> struct ErfPolyCoeffs;
  template <> struct ErfPolyCoeffs<2> { static constexpr double c[] = {0.12771143407390823, -0.36434310773011075, 0.07983114574072726}; };
  template <> struct ErfPolyCoeffs<3> { static constexpr double c[] = {0.1283474151683488, -0.3751365417937819, 0.10783366841996714, -0.018367461950288377}; };
  template <> struct ErfPolyCoeffs<4> { static constexpr double c[] = {0.12837788781796353, -0.3760641422947072, 0.11234174300591122, -0.0254478102265399, 0.0034940700424334107}; };
  template <> struct ErfPolyCoeffs<5> { static constexpr double c[] = {0.12837912249389816, -0.3761232618671417, 0.11280180112663808, -0.026711311332550714, 0.0049175514493777165, -0.000563142230068688}; };
  template <> struct ErfPolyCoeffs<6> { static constexpr double c[] = {0.12837916572671065, -0.37612625824234047, 0.112835851486209, -0.026853811935397306, 0.005188327685588843, -0.0008010193621267156, 7.853861332517134e-05}; };
  template <> struct ErfPolyCoeffs<7> { static constexpr double c[] = {0.12837916705802271, -0.3761263843465518, 0.1128378197417088, -0.02686540004602393, 0.005220945443805413, -0.0008482829005328025, 0.00011256949066594243, -9.641519377778401e-06}; };
  template <> struct ErfPolyCoeffs<8> { static constexpr double c[] = {0.12837916709458586, -0.3761263888850547, 0.11283791285400693, -0.026866131411195447, 0.005223776413541104, -0.0008542489078869606, 0.00011954820078366617, -1.3898707871346391e-05, 1.0562994978617774e-06}; };
  template <> struct ErfPolyCoeffs<9> { static constexpr double c[] = {0.12837916709549171, -0.37612638902775164, 0.11283791657674758, -0.02686616896234274, 0.005223966745806345, -0.0008547920949485678, 0.00012046046391363172, -1.4792924109333755e-05, 1.5295216780453957e-06, -1.0444478572724252e-07}; };
  template <> struct ErfPolyCoeffs<10> { static constexpr double c[] = {0.12837916709551214, -0.3761263890317352, 0.11283791670551899, -0.026866170582901992, 0.005223977131040907, -0.0008548304020210763, 0.00012054662748637538, -1.4913033504820654e-05, 1.6308100195376958e-06, -1.5177732006451187e-07, 9.407620393468255e-09}; };
  template <> struct ErfPolyCoeffs<11> { static constexpr double c[] = {0.12837916709551256, -0.3761263890318352, 0.11283791670944185, -0.02686617064311147, 0.005223977606118472, -0.0008548325929314416, 0.00012055293576898172, -1.492471230196162e-05, 1.6447131570692952e-06, -1.6206313754170574e-07, 1.3710980380255295e-08, -7.779468457668766e-10}; };
  inline constexpr double erf_poly_digits[] = {3.23, 4.55, 5.95, 7.4, 8.92, 10.48, 12.09, 13.73, 15.39, 16.53}; // degrees 2 to 11
  // e^(x^2) erfc(x) = 1/(s x + P(x)/Q(x)) on [13/16, 27.5], s = sqrt(pi) rounded to double: minimax P/Q of degrees n,
  // n + 1, Q(0) = 1, lowest degree first
  template <Integer n> struct ErfcRationalCoeffs;
  template <> struct ErfcRationalCoeffs<1> { static constexpr double p[] = {0.9793030924700097, 0.42288936170840685}; static constexpr double q[] = {1.0, 0.9893711728659675, 0.4866411428373295}; };
  template <> struct ErfcRationalCoeffs<2> { static constexpr double p[] = {1.0026085224236734, 0.6273078682410291, 0.1874267755450747}; static constexpr double q[] = {1.0, 1.2841227654125438, 0.7139553650922655, 0.21124494297149213}; };
  template <> struct ErfcRationalCoeffs<3> { static constexpr double p[] = {0.9996951439268782, 0.8527665297863873, 0.3511403040704106, 0.07102392688056385}; static constexpr double q[] = {1.0, 1.495027167004807, 1.0456590119940157, 0.3960233358200523, 0.08014650011259258}; };
  template <> struct ErfcRationalCoeffs<4> { static constexpr double p[] = {1.0000317531399276, 1.0520435581200038, 0.5528099550068913, 0.16178732187790418, 0.024035505680954874}; static constexpr double q[] = {1.0, 1.696340192468083, 1.3714321008545873, 0.6507708278262218, 0.18256267214267471, 0.02712107825684264}; };
  template <> struct ErfcRationalCoeffs<5> { static constexpr double p[] = {0.9999969607119215, 1.2373352468770282, 0.7714360638269159, 0.28840933813157793, 0.06464478303669897, 0.007408209175780033}; static constexpr double q[] = {1.0, 1.881385612644955, 1.7100391936255317, 0.9433374013747173, 0.3337985231622568, 0.07294370145734384, 0.008359270480400931}; };
  template <> struct ErfcRationalCoeffs<6> { static constexpr double p[] = {1.0000002724799562, 1.4108382913499842, 1.006724340002467, 0.4462183030169594, 0.12857429554941077, 0.023036306295101394, 0.00211196475435353}; static constexpr double q[] = {1.0, 2.0549154238390877, 2.0569935490106483, 1.2774268507327922, 0.5294999660476738, 0.14746353780316065, 0.02599369085245512, 0.0023830970022584567}; };
  template <> struct ErfcRationalCoeffs<7> { static constexpr double p[] = {0.9999999769071137, 1.5745913557853748, 1.2556282160586023, 0.6350820624133369, 0.21765414783566892, 0.05056690316760704, 0.007461104100531597, 0.0005625057240145419}; static constexpr double q[] = {1.0, 2.21866580983234, 2.411376223193842, 1.6497666559761162, 0.7727217182739154, 0.2540152725786248, 0.05769336270708144, 0.008418954374339196, 0.0006347197408355777}; };
  template <> struct ErfcRationalCoeffs<8> { static constexpr double p[] = {1.0000000018668276, 1.7301586167186165, 1.5163885527850856, 0.8542208424641358, 0.33441685345061367, 0.09286375148410286, 0.017929724156315274, 0.002227764627242449, 0.00014113643680621653}; static constexpr double q[] = {1.0, 2.374233320862175, 2.772332479051487, 2.058884492027341, 1.064900953800876, 0.39734159494254195, 0.10729928815248248, 0.020390782559139904, 0.0025137631956481526, 0.00015925541500232633}; };
  inline constexpr double erfc_rational_digits[] = {4.76, 6.58, 8.37, 10.2, 12, 13.8, 15.7, 17.1}; // degrees 1 to 8
  // erf(x) = x + x Q(x^2), for |x| <= 1
  template <Integer digits, class VData> inline VData erf_poly_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    return fma_intrin(x, eval_poly_intrin(mul_intrin(x, x), ErfPolyCoeffs<poly_degree(erf_poly_digits, 2, (std::is_same<Real,float>::value ? 6 : 11), digits)>::c), x);
  }
  // erfc(a) = e^(-a^2) Q(a)/(s a Q(a) + P(a)) for a >= 13/16, with a^2 = hi + lo exactly; a clamped to 27.5, where
  // e^(-a^2) is 0; keeps NaN
  template <Integer digits, class VData> inline VData erfc_rational_intrin(const VData& a) {
    using Real = typename VData::ScalarType;
    static constexpr Integer digits1 = (digits < 0 ? digits : digits + 1); // each of the two factors to one more digit
    static constexpr Integer n = poly_degree(erfc_rational_digits, 1, (std::is_same<Real,float>::value ? 3 : 8), digits1);
    static constexpr Integer order = (digits < 0 ? (Integer)(TypeTraits<Real>::SigBits/3.8) : exp_taylor_order(digits1));
    const VData b = min_intrin(set1_intrin<VData>((Real)27.5), a);
    const VData hi = mul_intrin(b, b);
    const VData e = approx_exp_intrin<order, true, ExpArg::Sum>(unary_minus_intrin(hi), unary_minus_intrin(mul_sub_exact_intrin(b, b, hi)));
    const VData q = eval_poly_intrin(b, ErfcRationalCoeffs<n>::q);
    const VData d = fma_intrin(mul_intrin(b, set1_intrin<VData>((Real)1.7724538509055160273L)), q, eval_poly_intrin(b, ErfcRationalCoeffs<n>::p));
    return mul_intrin(e, div_intrin(q, d));
  }
  // e^(-x^2) of one value, x^2 = hi + lo exactly by Dekker's split of x; 0 for |x| > 1000
  template <class Real> inline Real exp_neg_sq_generic(const Real x) {
    if (fabs(x) > 1000) return 0;
    static constexpr Real split = (Real)(((Long)1) << ((TypeTraits<Real>::SigBits + 2) / 2)) + 1;
    const Real c = split * x;
    const Real xh = c - (c - x);
    const Real xl = x - xh;
    const Real hi = x * x;
    return exp<Real>(-hi) * (1 - (((xh * xh - hi) + 2 * xh * xl) + xl * xl));
  }
  // erf(x) of one value, small |x|: 2/sqrt(pi) e^(-x^2) sum_k (2x^2)^k x/(1 3 ... (2k+1)), terms of one sign
  template <class Real> inline Real erf_series_generic(const Real x) {
    const Real x2 = x * x;
    Real term = x;
    Real sum = x;
    for (Integer k = 1; fabs(term) > machine_eps<Real>() * fabs(sum); k++) {
      term *= 2 * x2 / (2 * k + 1);
      sum += term;
    }
    return 2 / sqrt<Real>(const_pi<Real>()) * exp_neg_sq_generic(x) * sum;
  }
  // erfc(a) of one value, a >= 1/2: e^(-a^2)/sqrt(pi) / (a + (1/2)/(a + 1/(a + (3/2)/(a + ...)))), from the last of
  // about 0.75 d^2/a^2 terms for d digits
  template <class Real> inline Real erfc_cf_generic(const Real a) {
    static constexpr Integer d = (Integer)(TypeTraits<Real>::SigBits * 0.30103) + 2;
    const Integer n = (a >= (Real)0.5 && a < d ? (Integer)(d * d / (a * a)) : 0) + 8; // NaN: 8
    Real t = 0;
    for (Integer k = n; k > 0; k--) t = (Real)k / 2 / (a + t);
    return exp_neg_sq_generic(a) / sqrt<Real>(const_pi<Real>()) / (a + t);
  }
  // erf(x) = x + x Q(x^2) for |x| <= 1, else sign(x) (1 - erfc(|x|)). digits = -1: full
  template <Integer digits = -1, class VData> inline VData erf_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData one = set1_intrin<VData>((Real)1);
      const VData sgn = and_intrin(x, set1_intrin<VData>((Real)-0.0));
      const VData ax = xor_intrin(x, sgn);
      const Mask<VData> small = comp_intrin<ComparisonType::le>(ax, one);
      const Integer n_small = mask_count_intrin(small);
      if (n_small == VData::Size) return erf_poly_intrin<digits>(x);
      const VData r = xor_intrin(sub_intrin(one, erfc_rational_intrin<digits>(ax)), sgn);
      return (n_small ? select_intrin(small, erf_poly_intrin<digits>(x), r) : r);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        const Real a = fabs(x_.x[i]);
        x_.x[i] = (a < 1 ? erf_series_generic(x_.x[i]) : (x_.x[i] < 0 ? -1 : 1) * (1 - erfc_cf_generic(a)));
      }
      return x_.v;
    }
  }
  // erfc(x) = 1 - erf(x) for -1 <= x < 13/16 (where erfc > 1/4), else erfc_rational_intrin(|x|), subtracted from 2 for
  // x < 0. digits = -1: full
  template <Integer digits = -1, class VData> inline VData erfc_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData one = set1_intrin<VData>((Real)1);
      const Mask<VData> small = comp_intrin<ComparisonType::ge>(x, set1_intrin<VData>((Real)-1)) & comp_intrin<ComparisonType::lt>(x, set1_intrin<VData>((Real)13/16));
      const Integer n_small = mask_count_intrin(small);
      static constexpr Integer digits1 = (digits < 0 ? digits : digits + 1); // 1 - erf: up to 3 times the relative error of erf
      if (n_small == VData::Size) return sub_intrin(one, erf_poly_intrin<digits1>(x));
      const VData c = erfc_rational_intrin<digits>(fabs_intrin(x));
      const VData r = select_intrin(comp_intrin<ComparisonType::lt>(x, zero_intrin<VData>()), sub_intrin(set1_intrin<VData>((Real)2), c), c);
      return (n_small ? select_intrin(small, sub_intrin(one, erf_poly_intrin<digits1>(x)), r) : r);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        const Real a = x_.x[i];
        x_.x[i] = (a < (Real)0.5 ? (a > -3 ? 1 - erf_series_generic(a) : 2 - erfc_cf_generic(-a)) : erfc_cf_generic(a)); // NaN: the last
      }
      return x_.v;
    }
  }
  // erfinv(y) of one value, t = (1 - |y|)/2 given more precisely: Newton's method on erf(z) = |y| for |y| <= 0.85, else
  // on erfc(z) = 2t
  template <class Real> inline Real erfinv_generic(const Real y, const Real t) {
    if (!(t > 0)) return (t == 0 ? (y < 0 ? -(Real)INFINITY : (Real)INFINITY) : (Real)NAN); // also NaN
    if (y == 0) return y;
    const Real ay = fabs(y);
    const Real c = 2 / sqrt<Real>(const_pi<Real>());
    const bool central = (ay <= (Real)0.85);
    const Real u = sqrt<Real>(-2 * log(t));
    Real z = (central ? ay / c : (u - (2.515517 + 0.802853 * u + 0.010328 * u * u) / (1 + 1.432788 * u + 0.189269 * u * u + 0.001308 * u * u * u)) / sqrt<Real>((Real)2)); // Abramowitz and Stegun 26.2.23, error below 5e-4
    for (Integer k = 0; k < 100; k++) {
      const Real r = (central ? erf_series_generic(z) - ay : (z < (Real)0.5 ? 1 - erf_series_generic(z) : erfc_cf_generic(z)) - 2 * t);
      const Real dz = r / (central ? c : -c) / exp<Real>(-z * z);
      z -= dz;
      if (fabs(dz) <= machine_eps<Real>() * z) break;
    }
    return (y < 0 ? -z : z);
  }
  // Minimax coefficients of ndtri for the given digits, lowest degree first; minimax.py ndtri. c = sqrt(2 pi), a = sqrt(2)
  // rounded to double
  template <Integer digits> struct NdtriCoeffs;
  template <> struct NdtriCoeffs<7> {
    static constexpr double p0[] = {0.8805045814976765, 7.505611139468174, -35.35711021253784, -183.7219247484125}; // ndtri(1/2 + q) = q (c + p0(r)/q0(r)), r = 0.180625 - q^2, |q| <= 0.425
    static constexpr double q0[] = {1.0, 20.084840481351875, 107.19082725442411, 127.0972692689975, -15.903298231941674};
    static constexpr double p1[] = {-0.8393045825961606, -0.526598973398611, -0.06393715152892002, -0.0003490320067989722}; // ndtri(t) = -(a u + p1(s)/q1(s)), u = sqrt(-log t) in [1.6, 5], s = u - 1.6
    static constexpr double q1[] = {1.0, 0.9771585376142381, 0.27130299973203253, 0.01853277938138593};
    static constexpr double p2[] = {-0.41316313942633054, -0.06741718853790997, -0.00198822815257825, -2.098635073699726e-06}; // as p1, q1 for u > 5, s = u - 5
    static constexpr double q2[] = {1.0, 0.29858043429691744, 0.024216132140794384, 0.0004510690126461227};
  };
  template <> struct NdtriCoeffs<16> {
    static constexpr double p0[] = {0.880504598165366, 27.0778750783357, 249.0685268758211, 210.44927209638936, -7253.1419312903845, -31264.50381461276, -38582.54746523508, -10591.794084564011}; // ndtri(1/2 + q) = q (c + p0(r)/q0(r)), r = 0.180625 - q^2, |q| <= 0.425
    static constexpr double q0[] = {1.0, 42.313328790192564, 687.1869394075856, 5394.195138469813, 21213.78918659876, 39307.88270057975, 28729.073062315376, 5226.492330320501};
    static constexpr double p1[] = {-0.8393045890472687, -1.4224023958229681, -0.9247258467831433, -0.29088013150488945, -0.045381505185371665, -0.0031144374873346287, -6.611016465678944e-05, -7.037921733600937e-08}; // ndtri(t) = -(a u + p1(s)/q1(s)), u = sqrt(-log t) in [1.6, 5], s = u - 1.6
    static constexpr double q1[] = {1.0, 2.0444753253613954, 1.670174001443069, 0.6974481813956763, 0.15736473035764298, 0.01843048179093454, 0.0009585114615217301, 1.4746843780033622e-05};
    static constexpr double p2[] = {-0.41316316836437206, -0.1966189876546412, -0.033795467095799676, -0.002594558288012549, -8.96827790196168e-05, -1.2422452094540746e-06, -4.9296658523336505e-09, -7.941865589400861e-13}; // as p1, q1 for u > 5, s = u - 5
    static constexpr double q2[] = {1.0, 0.6112953003423831, 0.14354228752237055, 0.01630987080431471, 0.0009322459281457903, 2.5430086628035514e-05, 2.8354433355158357e-07, 8.735983700824607e-10};
  };
  // ndtri(p), or erfinv(p) for Erfinv, from NdtriCoeffs: central in q = p - 1/2 (Erfinv: p/2), tails in sqrt(-log t),
  // t = min(p, 1 - p) (Erfinv: (1 - |p|)/2)
  template <bool Erfinv, Integer digits, class VData> inline VData ndtri_erfinv_intrin(const VData& p) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Coeffs = NdtriCoeffs<((digits < 0 ? std::is_same<Real,double>::value : digits > 7) ? 16 : 7)>;
      const VData one = set1_intrin<VData>((Real)1);
      const VData half = set1_intrin<VData>((Real)0.5);
      const VData q = (Erfinv ? mul_intrin(p, half) : sub_intrin(p, half)); // exact, as t
      const Mask<VData> central = comp_intrin<ComparisonType::le>(fabs_intrin(q), set1_intrin<VData>((Real)0.425));
      const Integer n_central = mask_count_intrin(central);
      const auto central_value = [&q, &p]() {
        SCTL_UNUSED(p); // read for erfinv only
        const VData r = fma_intrin(unary_minus_intrin(q), q, set1_intrin<VData>((Real)0.180625));
        const VData T = div_intrin(eval_poly_intrin(r, Coeffs::p0), eval_poly_intrin(r, Coeffs::q0));
        if constexpr (Erfinv) return mul_intrin(p, fma_intrin(T, set1_intrin<VData>((Real)0.35355339059327376220L), set1_intrin<VData>((Real)0.88622692545275801365L)));
        else return mul_intrin(q, add_intrin(set1_intrin<VData>((Real)2.5066282746310005024L), T));
      };
      if (n_central == VData::Size) return central_value();

      const VData t = (Erfinv ? mul_intrin(sub_intrin(one, fabs_intrin(p)), half) : min_intrin(p, sub_intrin(one, p))); // keeps NaN
      const VData u = sqrt_intrin(unary_minus_intrin(log_intrin(t)));
      const auto tail_value = [&u](const auto& P, const auto& Q, const Real shift) { // a u + U, divided by sqrt(2) for Erfinv
        const VData s = sub_intrin(u, set1_intrin<VData>(shift));
        const VData den = eval_poly_intrin(s, Q);
        const VData num = fma_intrin(mul_intrin(u, set1_intrin<VData>((Real)1.4142135623730950488L)), den, eval_poly_intrin(s, P));
        return div_intrin(num, (Erfinv ? mul_intrin(den, set1_intrin<VData>((Real)1.4142135623730950488L)) : den));
      };
      VData x = tail_value(Coeffs::p1, Coeffs::q1, (Real)1.6);
      const Mask<VData> far = comp_intrin<ComparisonType::gt>(u, set1_intrin<VData>((Real)5));
      if (mask_count_intrin(far)) x = select_intrin(far, tail_value(Coeffs::p2, Coeffs::q2, (Real)5), x);
      x = select_intrin(comp_intrin<ComparisonType::eq>(t, zero_intrin<VData>()), set1_intrin<VData>((Real)INFINITY), x); // +-inf at p = 0, 1
      x = xor_intrin(x, and_intrin(q, set1_intrin<VData>((Real)-0.0)));
      return (n_central ? select_intrin(central, central_value(), x) : x);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {p};
      for (Integer i = 0; i < VData::Size; i++) {
        const Real a = x_.x[i];
        x_.x[i] = (Erfinv ? erfinv_generic(a, (1 - fabs(a)) / 2) : sqrt<Real>((Real)2) * erfinv_generic(2 * a - 1, (a < 1 - a ? a : 1 - a)));
      }
      return x_.v;
    }
  }
  template <Integer digits = -1, class VData> inline VData ndtri_intrin(const VData& p) {
    return ndtri_erfinv_intrin<false, digits>(p);
  }
  template <Integer digits = -1, class VData> inline VData erfinv_intrin(const VData& y) {
    return ndtri_erfinv_intrin<true, digits>(y);
  }

  // sin(pi w/2) = (pi/2) w + w^3 Q(w^2), |w| <= 1/2: minimax Q of degree n, lowest degree first
  template <Integer n> struct SinPiPolyCoeffs;
  template <> struct SinPiPolyCoeffs<0> { static constexpr double c[] = {-0.6295356107990434}; };
  template <> struct SinPiPolyCoeffs<1> { static constexpr double c[] = {-0.6458371155860916, 0.07806640499920597}; };
  template <> struct SinPiPolyCoeffs<2> { static constexpr double c[] = {-0.645963630198311, 0.07968141280991331, -0.004604834191653698}; };
  template <> struct SinPiPolyCoeffs<3> { static constexpr double c[] = {-0.6459640965035954, 0.07969258772124267, -0.004681292225850496, 0.00015825147962984448}; };
  template <> struct SinPiPolyCoeffs<4> { static constexpr double c[] = {-0.6459640975048387, 0.07969262616721817, -0.0046817526929786545, 0.00016042965956748935, -3.556945908113985e-06}; };
  template <> struct SinPiPolyCoeffs<5> { static constexpr double c[] = {-0.6459640975062448, 0.0796926262460598, -0.0046817541325625996, 0.00016044115216852262, -3.5986477759320474e-06, 5.63446316045072e-08}; };
  // cos(pi w/2) = 1 + w^2 P(w^2), |w| <= 1/2: minimax P of degree n, lowest degree first
  template <Integer n> struct CosPiPolyCoeffs;
  template <> struct CosPiPolyCoeffs<0> { static constexpr double c[] = {-1.1806822290618728}; };
  template <> struct CosPiPolyCoeffs<1> { static constexpr double c[] = {-1.2331097484257911, 0.24631381622482182}; };
  template <> struct CosPiPolyCoeffs<2> { static constexpr double c[] = {-1.2336977063547245, 0.2536032111226791, -0.020417283010921}; };
  template <> struct CosPiPolyCoeffs<3> { static constexpr double c[] = {-1.233700542597666, 0.25366922596542296, -0.02086016511085469, 0.0009037665417802849}; };
  template <> struct CosPiPolyCoeffs<4> { static constexpr double c[] = {-1.2337005501235707, 0.2536695072116397, -0.02086346840130368, 0.0009191629222590091, -2.4852207181205153e-05}; };
  template <> struct CosPiPolyCoeffs<5> { static constexpr double c[] = {-1.2337005501361553, 0.25366950789995935, -0.02086348073586745, 0.0009192599541479634, -2.520014305985118e-05, 4.655337261110899e-07}; };
  inline constexpr double sinpi_poly_digits[] = {3.24, 5.73, 8.42, 11.29, 14.3, 17.12}; // correct digits of degrees 0 to 5
  inline constexpr double cospi_poly_digits[] = {2.49, 4.83, 7.41, 10.2, 13.14, 16.11};
  // sin(pi x), cos(pi x) to the digits (-1: full), divided by pi with DivPi: x = (n + w)/2, sin(pi w/2) and cos(pi w/2)
  // by n mod 4; FullRange as sincos_pi_reduce_intrin
  template <Integer digits, bool FullRange = true, bool Sin = true, bool Cos = true, bool DivPi = false, class VData> inline void approx_sincospi_intrin(VData& sinx, VData& cosx, const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Int = typename IntegerType<sizeof(Real)>::value;
      using IntVec = VecData<Int, VData::Size>;
      static constexpr bool F = std::is_same<Real,float>::value;
      static constexpr Integer Bits = sizeof(Real) * 8;
      static constexpr Real pi_hi = (F ? (Real)1.57079625129699707031 : (Real)1.5707963267948966); // pi/2 rounded down: pi_lo > 0 keeps -0
      static constexpr Real pi_lo = (Real)(1.570796326794896619231321691639751442L - (long double)pi_hi);
      VData w, t;
      sincos_pi_reduce_intrin<FullRange>(w, t, x);
      const VData w2 = mul_intrin(w, w);
      static constexpr double scale = (DivPi ? 0.318309886183790671537767526745028724 : 1.0); // of the coefficients
      const auto poly = [](const VData& v, const auto& cf) { return eval_poly_horner_intrin(v, cf, scale); };
      using SinC = SinPiPolyCoeffs<poly_degree(sinpi_poly_digits, 0, (F ? 2 : 5), digits)>;
      using CosC = CosPiPolyCoeffs<poly_degree(cospi_poly_digits, 0, (F ? 3 : 5), digits)>;
      VData s, c;
      if constexpr (DivPi) { // w/2 + w^3 Q/pi and 1/pi + w^2 P/pi, 1/pi = ipi_hi + ipi_lo
        static constexpr Real ipi_hi = (Real)0.318309886183790671537767526745028724L;
        static constexpr Real ipi_lo = (Real)(0.318309886183790671537767526745028724L - (long double)ipi_hi);
        s = fma_intrin(mul_intrin(w, w2), poly(w2, SinC::c), mul_intrin(w, set1_intrin<VData>((Real)0.5)));
        c = add_intrin(fma_intrin(w2, poly(w2, CosC::c), set1_intrin<VData>(ipi_lo)), set1_intrin<VData>(ipi_hi));
      } else {
        s = fma_intrin(w, set1_intrin<VData>(pi_hi), mul_intrin(w, fma_intrin(w2, poly(w2, SinC::c), set1_intrin<VData>(pi_lo))));
        c = fma_intrin(w2, poly(w2, CosC::c), set1_intrin<VData>((Real)1));
      }
      const IntVec ti = reinterpret_intrin<IntVec>(t);
      const Mask<VData> odd = reinterpret_mask<Mask<VData>>(comp_intrin<ComparisonType::ne>(and_intrin(ti, set1_intrin<IntVec>(1)), zero_intrin<IntVec>()));
      const IntVec sign = set1_intrin<IntVec>(((Int)1) << (Bits - 1));
      const IntVec t1 = bitshiftleft_intrin(ti, Bits - 2); // bit 1 of n at the sign
      if constexpr (Sin) sinx = xor_intrin(select_intrin(odd, c, s), reinterpret_intrin<VData>(and_intrin(t1, sign))); // n mod 4 = 0: s, 1: c, 2: -s, 3: -c
      if constexpr (Cos) cosx = xor_intrin(select_intrin(odd, s, c), reinterpret_intrin<VData>(and_intrin(xor_intrin(t1, bitshiftleft_intrin(ti, Bits - 1)), sign))); // c, -s, -c, s
    } else { // sin(pi w/2) and cos(pi w/2) by sin and cos, one element at a time
      static constexpr Real c = (Real)1.5 * pow<TypeTraits<Real>::SigBits,Real>((Real)2);
      union U {
        VData v;
        Real x[VData::Size];
      };
      U w, t, s, co;
      sincos_pi_reduce_intrin(w.v, t.v, x);
      for (Integer i = 0; i < VData::Size; i++) {
        const Real n = t.x[i] - c;
        const Real m = n - 4 * floor(n / 4); // n mod 4; NaN for inf and NaN x, where the result is NaN
        const Integer q = (m >= 0 && m < 4 ? (Integer)m : 0);
        const Real sw = sin(const_pi<Real>() / 2 * w.x[i]) / (DivPi ? const_pi<Real>() : (Real)1);
        const Real cw = cos(const_pi<Real>() / 2 * w.x[i]) / (DivPi ? const_pi<Real>() : (Real)1);
        s.x[i] = (q == 0 ? sw : (q == 1 ? cw : (q == 2 ? -sw : -cw)));
        co.x[i] = (q == 0 ? cw : (q == 1 ? -sw : (q == 2 ? -cw : sw)));
      }
      sinx = s.v;
      cosx = co.v;
    }
  }

  // sin(pi x)/(pi x) to the digits (-1: full), 1 at x = 0; FullRange as sincos_pi_reduce_intrin
  template <Integer digits, bool FullRange = true, class VData> inline VData approx_sincpi_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    VData s, c;
    approx_sincospi_intrin<digits, FullRange, true, false, true>(s, c, x);
    return select_intrin(comp_intrin<ComparisonType::eq>(x, zero_intrin<VData>()), set1_intrin<VData>((Real)1), div_intrin(s, x));
  }

  // Minimax coefficients of tgamma, lgamma and digamma for the given digits (7: float), lowest degree first; minimax.py
  // gamma
  template <Integer digits> struct GammaCoeffs;
  template <> struct GammaCoeffs<7> {
    static constexpr double tp[] = {-0.5772156647932086, 0.3535063035602683, 0.2347721166356489, -0.014168638283016337, 0.0031058826771362645}; // Gamma(1+z) = 1 + z tp(z)/tq(z), z in [0, 1]
    static constexpr double tq[] = {1.0, 1.1010610247931785, -0.09223513593325558, -0.16376234074817483, 0.03052198370441707};
    static constexpr double lp[] = {0.07721565145317047, -0.1347738947057954, -0.153532419797404, -0.022216882081451075}; // lgamma(1+z) = z (z-1) (1/2 + lp(z)/lq(z)), z in [0, 2]
    static constexpr double lq[] = {1.0, 1.430750043268914, 0.5431894783640754, 0.047565462686810535};
    static constexpr double dp[] = {0.2503801374139148, -0.39656506018397564, -0.5568685820142112, -0.15723957069988823, -0.009945382838752015}; // digamma(1+z) = (z - z0) (1 + dp(z)/dq(z)), z in [-1/2, 2]
    static constexpr double dq[] = {1.0, 1.8297310299335572, 1.0165427267152791, 0.19697570425751226, 0.010169327525103255};
    static constexpr double s[] = {0.08333327385367327, -0.0027624721006410434}; // lgamma(x) - (x - 1/2) log x + x - log(2 pi)/2 = s(w)/x, w = 1/x^2, x >= 8
    static constexpr double a[] = {0.08333288938489247, -0.008245830346701113}; // log x - 1/(2x) - digamma(x) = a(w) w, w = 1/x^2, x >= 8
  };
  template <> struct GammaCoeffs<16> {
    static constexpr double tp[] = {-0.5772156649015329, 0.29064333099394446, 0.28563499099867845, 0.0038869861075239392, -0.005842817405732664, 0.0029665165143425206, -8.07167982465156e-05, 7.374491022673913e-06}; // Gamma(1+z) = 1 + z tp(z)/tq(z), z in [0, 1]
    static constexpr double tq[] = {1.0, 1.2099683130622767, 0.006258227784187291, -0.19748266604553685, 0.01854939266931247, 0.009877303810033473, -0.002297049003040702, 0.00014762238974642733};
    static constexpr double lp[] = {0.07721566490153285, 0.0006387929430960195, -0.3199129596725109, -0.4031182170084245, -0.19610452957106866, -0.04149342543332065, -0.003443534399528478, -7.688680260007995e-05}; // lgamma(1+z) = z (z-1) (1/2 + lp(z)/lq(z)), z in [0, 2]
    static constexpr double lq[] = {1.0, 3.184459549486992, 3.9583397665480424, 2.4326985769804454, 0.7731201784977569, 0.1213038073243519, 0.008099036046113608, 0.00015642352357189964};
    static constexpr double dp[] = {0.2503801375034054, -0.27496469605001, -0.7362060356179951, -0.4483065976471182, -0.11647753373438799, -0.013873387364272589, -0.0006867206378828349, -9.857947806206267e-06}; // digamma(1+z) = (z - z0) (1 + dp(z)/dq(z)), z in [-1/2, 2]
    static constexpr double dq[] = {1.0, 2.3153939858503856, 1.9581327970518727, 0.7891069342104603, 0.16220887818740337, 0.01656915244689783, 0.0007383170067187312, 9.919731367862371e-06};
    static constexpr double s[] = {0.0833333333333331, -0.0027777777773586353, 0.0007936505764882301, -0.0005951896854021093, 0.0008364587473822234, -0.0016334355591652535}; // lgamma(x) - (x - 1/2) log x + x - log(2 pi)/2 = s(w)/x, w = 1/x^2, x >= 8
    static constexpr double a[] = {0.08333333333332363, -0.008333333322128397, 0.003968249524774464, -0.004165839418727212, 0.007496258858457115, -0.017201963505816913}; // log x - 1/(2x) - digamma(x) = a(w) w, w = 1/x^2, x >= 8
  };
  // Two-part constants of the gamma functions: z0 + 1 is the zero of digamma, c = log(2 pi)/2
  template <class Real> struct GammaConsts {
    static constexpr Real z0_hi = (Real)0.46163214496836236;
    static constexpr Real z0_lo = (Real)((0.46163214496836236 - (double)z0_hi) - 1.5522348162858677e-17);
    static constexpr Real c_hi = (Real)0.9189385332046728;
    static constexpr Real c_lo = (Real)((0.9189385332046728 - (double)c_hi) - 3.8782941580672414e-17);
  };
  // GammaCoeffs<16> for double at full precision or digits > 7, else <7>
  template <class Real, Integer digits> using GammaCoeffsOf = GammaCoeffs<((digits < 0 ? std::is_same<Real,double>::value : digits > 7) ? 16 : 7)>;
  // y = x - n, p = (x-1) ... (x-n) for the least n >= 0 with y < hi, in lanes with x < 8; elsewhere n = 0
  template <class VData> inline void gamma_shift_intrin(VData& y, VData& p, const VData& x, const typename VData::ScalarType hi) {
    using Real = typename VData::ScalarType;
    const Mask<VData> mid = comp_intrin<ComparisonType::lt>(x, set1_intrin<VData>((Real)8));
    y = x;
    p = set1_intrin<VData>((Real)1);
    for (Integer k = 0; k < 6; k++) { // x - k is exact
      const Mask<VData> m = mid & comp_intrin<ComparisonType::ge>(y, set1_intrin<VData>(hi));
      if (!mask_count_intrin(m)) break;
      y = select_intrin(m, sub_intrin(y, set1_intrin<VData>((Real)1)), y);
      p = select_intrin(m, mul_intrin(p, y), p);
    }
  }
  // lgamma(x) = L_hi + L_lo, 8 <= x <= 2^SigBits, by Stirling's series given log x = lh + ll; Exact: (x - 1/2) lh
  // exactly also without fused FMA
  template <class Coeffs, bool Exact = true, class VData> inline void lgamma_stirling_intrin(VData& L_hi, VData& L_lo, const VData& x, const VData& lh, const VData& ll) {
    using Real = typename VData::ScalarType;
    const VData a = sub_intrin(x, set1_intrin<VData>((Real)0.5));
    const VData t_hi = mul_intrin(a, lh); // (x - 1/2) log x = t_hi + t_lo
    VData t_lo = mul_intrin(a, ll);
    if constexpr (Exact || fused_fma<VData>) t_lo = fma_intrin(a, ll, mul_sub_exact_intrin(a, lh, t_hi));
    const VData s = sub_intrin(t_hi, x); // t_hi > x: s + s_lo = t_hi - x exactly
    const VData s_lo = sub_intrin(sub_intrin(t_hi, s), x);
    const VData r = div_intrin(set1_intrin<VData>((Real)1), x);
    const VData S = mul_intrin(r, eval_poly_intrin(mul_intrin(r, r), Coeffs::s));
    const VData c = add_intrin(set1_intrin<VData>(GammaConsts<Real>::c_hi), S); // c + c_err = c_hi + S exactly
    const VData c_err = add_intrin(sub_intrin(set1_intrin<VData>(GammaConsts<Real>::c_hi), c), S);
    L_hi = add_intrin(s, c);
    L_lo = add_intrin(add_intrin(sub_intrin(s, L_hi), c), add_intrin(add_intrin(s_lo, t_lo), add_intrin(c_err, set1_intrin<VData>(GammaConsts<Real>::c_lo))));
  }
  // lgamma(1 + z) for -1/1024 <= a < 8; lgamma(a) adds log(arg) xor sgn: arg = p of the shift to [2, 3), else |a| and
  // sgn = -0
  template <class Coeffs, class VData> inline VData lgamma_mid_intrin(VData& arg, VData& sgn, const VData& a) {
    using Real = typename VData::ScalarType;
    const VData one = set1_intrin<VData>((Real)1);
    VData y, p;
    gamma_shift_intrin(y, p, a, (Real)3);
    const Mask<VData> below1 = comp_intrin<ComparisonType::lt>(y, one);
    const VData z = select_intrin(below1, y, sub_intrin(y, one));
    const VData u = fma_intrin(z, div_intrin(eval_poly_intrin(z, Coeffs::lp), eval_poly_intrin(z, Coeffs::lq)), mul_intrin(z, set1_intrin<VData>((Real)0.5)));
    arg = select_intrin(below1, fabs_intrin(a), p);
    sgn = select_intrin(below1, set1_intrin<VData>((Real)-0.0), zero_intrin<VData>());
    return mul_intrin(u, sub_intrin(z, one)); // z - 1 exact for z >= 1/2
  }
  // sum_k B_2k/(2k (2k-1)) w^(k-1), k = 1 .. 14, w = 1/x^2, of one value (Bernoulli numbers B_2k); Digamma: B_2k/(2k)
  template <bool Digamma, class Real> inline Real gamma_series_generic(const Real w) {
    static constexpr double b[14][2] = {{1, 6}, {-1, 30}, {1, 42}, {-1, 30}, {5, 66}, {-691, 2730}, {7, 6}, {-3617, 510}, {43867, 798}, {-174611, 330}, {854513, 138}, {-236364091, 2730}, {8553103, 6}, {-23749461029, 870}};
    Real s = 0;
    for (Integer k = 14; k >= 1; k--) s = s * w + (Real)b[k - 1][0] / (Real)b[k - 1][1] / (Real)(Digamma ? 2 * k : 2 * k * (2 * k - 1));
    return s;
  }
  // lgamma(x) of one value: the reflection formula below 1/2; else Stirling's series at y = x + n >= 32, less the log of
  // x (x+1) ... (y-1)
  template <class Real> inline Real lgamma_generic(const Real x) {
    if (!(fabs(x) < (Real)INFINITY)) return fabs(x); // inf, NaN
    if (x < (Real)0.5) {
      if (x == floor(x)) return (Real)INFINITY;
      return log(const_pi<Real>() / fabs(sin(const_pi<Real>() * (x - round(x))))) - lgamma_generic(1 - x);
    }
    Real y = x;
    Real p = 1;
    for (; y < 32; y += 1) p *= y;
    return (y - (Real)0.5) * log(y) - y + log(2 * const_pi<Real>()) / 2 + gamma_series_generic<false>(1 / (y * y)) / y - log(p);
  }
  // Gamma(x) of one value: the reflection formula below 1/2; else e^lgamma(y) / (x (x+1) ... (y-1)), y = x + n >= 32
  template <class Real> inline Real tgamma_generic(const Real x) {
    if (x == 0) return (std::signbit((double)x) ? -(Real)INFINITY : (Real)INFINITY); // 1/x
    if (!(x == x)) return x;
    if (x < (Real)0.5) {
      if (x == floor(x)) return (Real)NAN; // also -inf
      const Real n = round(x);
      const Real s = sin(const_pi<Real>() * (x - n)) * (n - 2 * floor(n / 2) == 0 ? 1 : -1); // sin(pi x)
      return const_pi<Real>() / (s * tgamma_generic(1 - x));
    }
    if (x > 2000) return (Real)INFINITY;
    Real y = x;
    Real p = 1;
    for (; y < 32; y += 1) p *= y;
    return exp((y - (Real)0.5) * log(y) - y + log(2 * const_pi<Real>()) / 2 + gamma_series_generic<false>(1 / (y * y)) / y) / p;
  }
  // digamma(x) of one value: the reflection formula below 0; else the asymptotic series at y = x + n >= 32, less
  // 1/x + 1/(x+1) + ... + 1/(y-1)
  template <class Real> inline Real digamma_generic(const Real x) {
    if (x == 0) return (std::signbit((double)x) ? (Real)INFINITY : -(Real)INFINITY); // -1/x
    if (!(x == x)) return x;
    if (x < 0) {
      if (x == floor(x)) return (Real)NAN; // also -inf
      const Real r = const_pi<Real>() * (x - round(x)); // cot(pi x) = cot(r)
      return digamma_generic(1 - x) - const_pi<Real>() * cos(r) / sin(r);
    }
    if (x == (Real)INFINITY) return x;
    Real y = x;
    Real sum = 0;
    for (; y < 32; y += 1) sum += 1 / y;
    const Real w = 1 / (y * y);
    return log(y) - 1 / (2 * y) - gamma_series_generic<true>(w) * w - sum;
  }
  // Gamma(x): a rational on [1, 2] (GammaCoeffs) with the shift to it below 8, Stirling's series beyond, and the
  // reflection formula below -1/1024. digits = -1: full
  template <Integer digits = -1, class VData> inline VData tgamma_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Coeffs = GammaCoeffsOf<Real, digits>;
      const VData one = set1_intrin<VData>((Real)1);
      const VData eight = set1_intrin<VData>((Real)8);
      const VData tiny = set1_intrin<VData>((Real)-1 / 1024);
      const auto mid_value = [&one](const VData& a) { // -1/1024 <= a < 8: Gamma(1 + z) = (Q + z P)/Q, times p, over a below 1
        VData y, p;
        gamma_shift_intrin(y, p, a, (Real)2);
        const Mask<VData> below1 = comp_intrin<ComparisonType::lt>(y, one);
        const VData z = select_intrin(below1, y, sub_intrin(y, one));
        const VData q = eval_poly_intrin(z, Coeffs::tq);
        const VData num = fma_intrin(z, eval_poly_intrin(z, Coeffs::tp), q);
        return mul_intrin(div_intrin(num, select_intrin(below1, mul_intrin(z, q), q)), p);
      };
      if (mask_count_intrin(comp_intrin<ComparisonType::ge>(x, tiny) & comp_intrin<ComparisonType::lt>(x, eight)) == VData::Size) return mid_value(x);

      // x >= 8 or x < -1/1024, inf or NaN: Gamma(a), a = |x|; Gamma(x) = -pi/(x sinpi(x) Gamma(-x)) for x < -1/1024
      const Mask<VData> neg = comp_intrin<ComparisonType::lt>(x, tiny);
      const VData a = select_intrin(neg, unary_minus_intrin(x), x);
      const Mask<VData> big = comp_intrin<ComparisonType::ge>(a, eight);
      const Integer n_big = mask_count_intrin(big);
      VData r = (n_big < VData::Size ? mid_value(a) : one);
      VData L_hi = one;
      VData L_lo = one;
      if (n_big) {
        const VData ab = min_intrin(set1_intrin<VData>((Real)200), max_intrin(eight, a)); // Gamma(200) overflows
        VData e, f, lh, ll;
        log_split_intrin<false>(e, f, ab);
        log_hi_lo_intrin(lh, ll, e, f);
        lgamma_stirling_intrin<Coeffs>(L_hi, L_lo, ab, lh, ll);
        const VData E = exp_intrin(L_hi);
        r = select_intrin(big, select_intrin(comp_intrin<ComparisonType::eq>(E, set1_intrin<VData>((Real)INFINITY)), E, fma_intrin(E, L_lo, E)), r);
      }
      if (mask_count_intrin(neg)) {
        VData s, c;
        approx_sincospi_intrin<-1, true, true, false>(s, c, x);
        const Mask<VData> ovf = big & comp_intrin<ComparisonType::gt>(L_hi, set1_intrin<VData>(std::is_same<Real,float>::value ? (Real)85 : (Real)704)); // log(max/a), a <= 40 (double: 200) where Gamma(a) is finite
        VData h = one; // Gamma(a) = g h, h = e^(L_hi/2) where x sinpi(x) Gamma(a) could overflow
        if (mask_count_intrin(neg & ovf)) h = select_intrin(ovf, exp_intrin(mul_intrin(L_hi, set1_intrin<VData>((Real)0.5))), one);
        const VData g = select_intrin(ovf, fma_intrin(h, L_lo, h), r);
        const VData rr = div_intrin(div_intrin(set1_intrin<VData>(-const_pi<Real>()), mul_intrin(mul_intrin(x, s), g)), h);
        r = select_intrin(neg, select_intrin(comp_intrin<ComparisonType::eq>(s, zero_intrin<VData>()), set1_intrin<VData>((Real)NAN), rr), r);
      }
      return r;
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        if constexpr (std::is_floating_point<Real>::value) x_.x[i] = std::tgamma(x_.x[i]);
        else x_.x[i] = tgamma_generic(x_.x[i]);
      }
      return x_.v;
    }
  }
  // lgamma(x) for x >= 8 or x < -1/1024, inf or NaN: one log for the shift and Stirling's series, and the reflection
  // formula; arguments and result by value
  template <class Coeffs, class VData> [[gnu::noinline]] VData lgamma_general_intrin(const VData x) {
    using Real = typename VData::ScalarType;
    const VData zero = zero_intrin<VData>();
    const VData one = set1_intrin<VData>((Real)1);
    const VData eight = set1_intrin<VData>((Real)8);
    const VData tiny = set1_intrin<VData>((Real)-1 / 1024);
    VData arg = one;
    VData sgn = zero;
    const Mask<VData> neg = comp_intrin<ComparisonType::lt>(x, tiny);
    const VData a = select_intrin(neg, unary_minus_intrin(x), x);
    const Mask<VData> big = comp_intrin<ComparisonType::ge>(a, eight);
    const Integer n_big = mask_count_intrin(big);
    static constexpr Real big_x = pow<TypeTraits<Real>::SigBits,Real>((Real)2); // beyond, a (log a - 1)
    const VData ab = min_intrin(set1_intrin<VData>(big_x), max_intrin(eight, a));
    VData r = (n_big < VData::Size ? lgamma_mid_intrin<Coeffs>(arg, sgn, a) : zero);
    const VData lh = log_intrin(select_intrin(big, ab, arg));
    const VData ll = zero;
    r = add_intrin(r, xor_intrin(lh, sgn));
    if (n_big) {
      VData L_hi, L_lo;
      lgamma_stirling_intrin<Coeffs, false>(L_hi, L_lo, ab, lh, ll);
      r = select_intrin(big, add_intrin(L_hi, L_lo), r);
    }
    const Mask<VData> huge = comp_intrin<ComparisonType::gt>(a, set1_intrin<VData>(big_x));
    if (mask_count_intrin(huge)) r = select_intrin(huge, mul_intrin(a, sub_intrin(log_intrin(a), one)), r);
    if (mask_count_intrin(neg)) { // lgamma(x) = log(pi) - log|x sinpi(x)| - lgamma(-x)
      VData s, c;
      approx_sincospi_intrin<-1, true, true, false>(s, c, x);
      const VData rr = sub_intrin(sub_intrin(set1_intrin<VData>((Real)1.14472988584940017414342735135305871L), log_intrin(fabs_intrin(mul_intrin(x, s)))), r);
      r = select_intrin(neg, select_intrin(comp_intrin<ComparisonType::eq>(x, set1_intrin<VData>(-(Real)INFINITY)), set1_intrin<VData>((Real)INFINITY), rr), r);
    }
    return r;
  }
  // log|Gamma(x)|: a rational on [1, 3] (GammaCoeffs) with the shift to it below 8, Stirling's series beyond, and the
  // reflection formula below -1/1024. digits = -1: full
  template <Integer digits = -1, class VData> inline VData lgamma_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Coeffs = GammaCoeffsOf<Real, digits>;
      const VData zero = zero_intrin<VData>();
      const VData one = set1_intrin<VData>((Real)1);
      const VData eight = set1_intrin<VData>((Real)8);
      const VData tiny = set1_intrin<VData>((Real)-1 / 1024);
      VData arg = one;
      VData sgn = zero;
      if (mask_count_intrin(comp_intrin<ComparisonType::ge>(x, tiny) & comp_intrin<ComparisonType::lt>(x, eight)) == VData::Size) {
        const VData base = lgamma_mid_intrin<Coeffs>(arg, sgn, x);
        return add_intrin(base, xor_intrin(log_intrin(arg), sgn));
      }
      return lgamma_general_intrin<Coeffs>(x);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        if constexpr (std::is_floating_point<Real>::value) x_.x[i] = std::lgamma(x_.x[i]);
        else x_.x[i] = lgamma_generic(x_.x[i]);
      }
      return x_.v;
    }
  }
  // digamma(x): a rational on [1/2, 3] (GammaCoeffs) with the shift to it below 8, the asymptotic series beyond, and the
  // reflection formula below 0. digits = -1: full
  template <Integer digits = -1, class VData> inline VData digamma_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Coeffs = GammaCoeffsOf<Real, digits>;
      const VData zero = zero_intrin<VData>();
      const VData one = set1_intrin<VData>((Real)1);
      const VData eight = set1_intrin<VData>((Real)8);
      const auto mid_value = [&zero, &one](const VData& a) { // 0 <= a < 8, also -0: (d (Q + P) D + N Q)/(Q D)
        const Mask<VData> mid = comp_intrin<ComparisonType::lt>(a, set1_intrin<VData>((Real)8));
        VData y = a;
        VData N = zero; // N/D = 1/(a-1) + 1/(a-2) + ... + 1/y for the shift down to y < 3, or -1/a below 1/2
        VData D = one;
        for (Integer k = 0; k < 5; k++) {
          const Mask<VData> m = mid & comp_intrin<ComparisonType::ge>(y, set1_intrin<VData>((Real)3));
          if (!mask_count_intrin(m)) break;
          y = select_intrin(m, sub_intrin(y, one), y);
          N = select_intrin(m, fma_intrin(N, y, D), N);
          D = select_intrin(m, mul_intrin(D, y), D);
        }
        const Mask<VData> below = comp_intrin<ComparisonType::lt>(a, set1_intrin<VData>((Real)0.5));
        N = select_intrin(below, set1_intrin<VData>((Real)-1), N);
        D = select_intrin(below, a, D);
        const VData z = select_intrin(below, a, sub_intrin(y, one));
        const VData d = sub_intrin(sub_intrin(z, set1_intrin<VData>(GammaConsts<Real>::z0_hi)), set1_intrin<VData>(GammaConsts<Real>::z0_lo));
        const VData P = eval_poly_intrin(z, Coeffs::dp);
        const VData Q = eval_poly_intrin(z, Coeffs::dq);
        return div_intrin(fma_intrin(mul_intrin(d, add_intrin(Q, P)), D, mul_intrin(N, Q)), mul_intrin(Q, D));
      };
      if (mask_count_intrin(comp_intrin<ComparisonType::ge>(x, zero) & comp_intrin<ComparisonType::lt>(x, eight)) == VData::Size) return mid_value(x);

      // x >= 8 or x < 0, inf or NaN: digamma(a), a = x or 1 - x; digamma(x) = digamma(1 - x) - pi cospi(x)/sinpi(x)
      const Mask<VData> neg = comp_intrin<ComparisonType::lt>(x, zero);
      const VData a = select_intrin(neg, sub_intrin(one, x), x);
      const Mask<VData> big = comp_intrin<ComparisonType::ge>(a, eight);
      const Integer n_big = mask_count_intrin(big);
      VData r = (n_big < VData::Size ? mid_value(a) : zero);
      if (n_big) {
        const VData ri = div_intrin(one, a);
        const VData w = mul_intrin(ri, ri);
        r = select_intrin(big, sub_intrin(log_intrin(a), fma_intrin(w, eval_poly_intrin(w, Coeffs::a), mul_intrin(ri, set1_intrin<VData>((Real)0.5)))), r);
      }
      if (mask_count_intrin(neg)) {
        VData s, c;
        approx_sincospi_intrin<-1>(s, c, x);
        const VData rr = sub_intrin(r, div_intrin(mul_intrin(set1_intrin<VData>(const_pi<Real>()), c), s));
        r = select_intrin(neg, select_intrin(comp_intrin<ComparisonType::eq>(s, zero), set1_intrin<VData>((Real)NAN), rr), r);
      }
      return r;
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = digamma_generic(x_.x[i]);
      return x_.v;
    }
  }

  // Minimax coefficients of J_n and Y_n of order n = 0, 1, 2 for the given digits (7: float), lowest degree first, from
  // minimax.py bessel
  template <Integer digits, Integer n> struct BesselCoeffs;
  template <> struct BesselCoeffs<7, 0> {
    static constexpr double j[] = {0.003187355638722544, -0.00015368697111522376, 3.1211684312205528e-06, -3.646271661237845e-08, 2.8079483703369514e-10, -1.5476463123200065e-12, 6.3078292704098234e-15}; // J_n = x^n (z - z1) (z - z2) j(z - 25/2), z = x^2 <= 25
    static constexpr double y[] = {0.4837248577722562, -0.034612466604259096, -0.0030116887294084774, 0.00019892046666336222, -4.567729458929836e-06, 5.778588535392605e-08, -4.725888784942176e-10, 2.7389413243148166e-12, -1.1764631586313662e-14}; // Y_n = x^n y(z - 25/2) + 2/pi (log(x) J_n - h_n)
    static constexpr double pp[] = {-0.002812497637976186, 0.00017930821499641096, -3.536632827203849e-05, 1.0922051254024614e-05, -2.382850787994649e-06}; // P_n = 1 + w pp(w)/pq(w), w = 25/x^2 < 1
    static constexpr double pq[] = {1.0};
    static constexpr double qp[] = {0.0029296765326287575, -0.00036273899528400916, 0.00010449660321156088, -3.9576420654209763e-05, 9.466001802055049e-06}; // x Q_n = (4n^2 - 1)/8 + w qp(w)/qq(w)
    static constexpr double qq[] = {1.0};
  };
  template <> struct BesselCoeffs<7, 1> {
    static constexpr double j[] = {0.0004323698737371032, -1.67461381572909e-05, 2.8299956637321187e-07, -2.8239126654918323e-09, 1.8948619263226792e-11, -9.238391490840878e-14, 3.3868220068672783e-16}; // J_n = x^n (z - z1) (z - z2) j(z - 25/2), z = x^2 <= 25
    static constexpr double y[] = {0.13974939197560665, 0.007286921640085689, -0.0010439697196727017, 3.3969579096620435e-05, -5.502175233952695e-07, 5.469244709394378e-09, -3.7261451238537214e-11, 1.796220419835852e-13}; // Y_n = x^n y(z - 25/2) + 2/pi (log(x) J_n - h_n)
    static constexpr double pp[] = {0.004687497385189045, -0.00023056320149796692, 4.189192099918531e-05, -1.2512171143573935e-05, 2.6951709665446858e-06}; // P_n = 1 + w pp(w)/pq(w), w = 25/x^2 < 1
    static constexpr double pq[] = {1.0};
    static constexpr double qp[] = {-0.004101550452257082, 0.00044342560690108115, -0.00012090843653232402, 4.4722824253863605e-05, -1.059828709863941e-05}; // x Q_n = (4n^2 - 1)/8 + w qp(w)/qq(w)
    static constexpr double qq[] = {1.0};
  };
  template <> struct BesselCoeffs<7, 2> {
    static constexpr double j[] = {4.4865147979947535e-05, -1.4643735869771537e-06, 2.1317358891054606e-08, -1.8644977724611279e-10, 1.1120765267503724e-12, -4.872986103327627e-15, 1.625230313970517e-17}; // J_n = x^n (z - z1) (z - z2) j(z - 25/2), z = x^2 <= 25
    static constexpr double y[] = {0.00912670473250634, 0.003204768253714236, -0.00018248145188007588, 4.120021616640511e-06, -5.2220669118471704e-08, 4.297059906595613e-10, -2.5015154243125832e-12, 1.0410699796420331e-14}; // Y_n = x^n y(z - 25/2) + 2/pi (log(x) J_n - h_n)
    static constexpr double pp[] = {-0.032812496437927365, 0.0005073643989641904, -7.025374050396454e-05, 1.8862303001006003e-05, -3.902533782540046e-06}; // P_n = 1 + w pp(w)/pq(w), w = 25/x^2 < 1
    static constexpr double pq[] = {1.0};
    static constexpr double qp[] = {0.012304671492116339, -0.0008238717174245483, 0.0001883127450438455, -6.465315181512669e-05, 1.4882783060313914e-05}; // x Q_n = (4n^2 - 1)/8 + w qp(w)/qq(w)
    static constexpr double qq[] = {1.0};
  };
  template <> struct BesselCoeffs<16, 0> {
    static constexpr double j[] = {0.0031873556303932387, -0.00015368696349731236, 3.1211693936997383e-06, -3.646313083174819e-08, 2.807789505412937e-10, -1.5421576526531107e-12, 6.366486955227784e-15, -2.0513164152600907e-17, 5.307848171067075e-20, -1.1279968456660394e-22, 2.006655830065238e-25, -3.012868259515063e-28}; // J_n = x^n (z - z1) (z - z2) j(z - 25/2), z = x^2 <= 25
    static constexpr double y[] = {0.4837248578102995, -0.03461246578284103, -0.003011688739390264, 0.00019892039657764854, -4.567729022701026e-06, 5.778749977151347e-08, -4.725957372279627e-10, 2.7251712449352726e-12, -1.1719731578016508e-14, 3.9113944366226573e-17, -1.0437794621378697e-19, 2.283023354338751e-22, -4.157816687325966e-25}; // Y_n = x^n y(z - 25/2) + 2/pi (log(x) J_n - h_n)
    static constexpr double pp[] = {-0.0028125, -0.026587563394949693, -0.08528320776581748, -0.11651485337344465, -0.06971133349576555, -0.016765520348913952, -0.0012434313289263871, -9.989190600808682e-06}; // P_n = 1 + w pp(w)/pq(w), w = 25/x^2 < 1
    static constexpr double pq[] = {1.0, 9.517157957093195, 30.91710523392409, 43.28162226218273, 27.19354646047617, 7.269458316050991, 0.7022895844532588, 0.016069189801276397};
    static constexpr double qp[] = {0.0029296874999999996, 0.030654070151767077, 0.10948526940123919, 0.16753388214568266, 0.11277777030519183, 0.03050999740819592, 0.0024971089612372457, 1.660598928903254e-05}; // x Q_n = (4n^2 - 1)/8 + w qp(w)/qq(w)
    static constexpr double qq[] = {1.0, 10.587287195136332, 38.646383637369865, 61.599968054272146, 44.88279261064361, 14.30081888686441, 1.7233854342947574, 0.053822004023966606};
  };
  template <> struct BesselCoeffs<16, 1> {
    static constexpr double j[] = {0.0004323698734106474, -1.6746137771888152e-05, 2.829996036544929e-07, -2.823933163314925e-09, 1.8948010865408882e-11, -9.211567323238196e-14, 3.408637110659347e-16, -9.94652867132839e-19, 2.3509934749991738e-21, -4.60159304791747e-24, 7.539169825622234e-27}; // J_n = x^n (z - z1) (z - z2) j(z - 25/2), z = x^2 <= 25
    static constexpr double y[] = {0.13974939033568806, 0.007286920670758755, -0.001043969302919598, 3.3969640216469146e-05, -5.502334460238883e-07, 5.468334105760503e-09, -3.7071974025823e-11, 1.8314634593070199e-13, -6.901774241479743e-16, 2.0520396902102843e-18, -4.940672084225741e-21, 9.845598975820887e-24, -1.6331571308703538e-26}; // Y_n = x^n y(z - 25/2) + 2/pi (log(x) J_n - h_n)
    static constexpr double pp[] = {0.0046875, 0.043860427067950104, 0.13907713558784354, 0.18761211795718716, 0.110773474910082, 0.026341274425374903, 0.001956242356992678, 1.802970144970018e-05}; // P_n = 1 + w pp(w)/pq(w), w = 25/x^2 < 1
    static constexpr double pq[] = {1.0, 9.406109857829337, 30.123508150834166, 41.42342821295195, 25.42491700396801, 6.57934539833196, 0.6046872269487391, 0.012618933160073897};
    static constexpr double qp[] = {-0.004101562499999999, -0.042525925094907255, -0.15035693214398083, -0.227550131124614, -0.15142909829784024, -0.040544916551061724, -0.0033109419743992926, -2.5013599793140416e-05}; // x Q_n = (4n^2 - 1)/8 + w qp(w)/qq(w)
    static constexpr double qq[] = {1.0, 10.476506796948685, 37.76175458966129, 59.258897076889056, 42.324943404849705, 13.128866950499521, 1.5215757547778297, 0.04453053839891092};
  };
  template <> struct BesselCoeffs<16, 2> {
    static constexpr double j[] = {4.486514796785987e-05, -1.4643735697056801e-06, 2.1317360261500442e-08, -1.8645068552754918e-10, 1.1120543402311745e-12, -4.8611769275487106e-15, 1.633003118432335e-17, -4.360898008775409e-20, 9.498105879108946e-23, -1.7230322847483982e-25, 2.632670851578099e-28}; // J_n = x^n (z - z1) (z - z2) j(z - 25/2), z = x^2 <= 25
    static constexpr double y[] = {0.009126704723589382, 0.0032047681920615327, -0.0001824814446808372, 4.120026282890154e-06, -5.222113217895149e-08, 4.296240495168117e-10, -2.493780827267649e-12, 1.0792181934091186e-14, -3.622441597979738e-17, 9.716408906610672e-20, -2.134011846917858e-22, 3.831805762143809e-25}; // Y_n = x^n y(z - 25/2) + 2/pi (log(x) J_n - h_n)
    static constexpr double pp[] = {-0.0328125, -0.29803160194673267, -0.9137290854149711, -1.1870041120777062, -0.6730754850760635, -0.15419163418765874, -0.011356858334499412, -0.0001334052404848676}; // P_n = 1 + w pp(w)/pq(w), w = 25/x^2 < 1
    static constexpr double pq[] = {1.0, 9.098336618852802, 27.985522082424854, 36.589037964481975, 21.02383865246782, 4.961736125735502, 0.39500950975321075, 0.006221914988442327};
    static constexpr double qp[] = {0.0123046875, 0.12426087137836113, 0.42667643451598697, 0.6254722133654043, 0.4027086355379197, 0.1047245300475318, 0.008512266481928902, 8.484832992412702e-05}; // x Q_n = (4n^2 - 1)/8 + w qp(w)/qq(w)
    static constexpr double qq[] = {1.0, 10.165692542971511, 35.34132052677422, 53.0460354375923, 35.79154730555239, 10.281609251057063, 1.0644201137787792, 0.02565899800156392};
  };
  // The squares z1 and z2 of the first two zeros of J_n, n = 0, 1, 2, each the sum of two doubles
  inline constexpr double bessel_zeros_sq[3][2][2] = {{{5.783185962946784, 4.123247343506204e-16}, {30.471262343662087, -2.366534485750659e-16}},
                                                      {{14.681970642123893, -9.858177825793294e-17}, {49.2184563216946, 5.086354069436341e-16}},
                                                      {{26.374616427163392, -1.5036561242658726e-15}, {70.84999891909585, 6.850461574080647e-15}}};
  // BesselCoeffs<16, n> for double at full precision or digits > 7, else <7, n>
  template <class Real, Integer digits, Integer n> using BesselCoeffsOf = BesselCoeffs<((digits < 0 ? std::is_same<Real,double>::value : digits > 7) ? 16 : 7), n>;
  // J_n(x) and Y_n(x), n = 0, 1, 2, of one value x >= 0 by power series below 2, Miller's recurrence with Neumann series
  // below 0.4 SigBits, and the Hankel expansion beyond
  template <class Real> inline void bessel_jy_generic(Real (&J)[3], Real (&Y)[3], const Real x) {
    if (x == 0 || !(x < (Real)INFINITY)) { // 0, inf, NaN
      for (Integer n = 0; n < 3; n++) {
        J[n] = (x == 0 ? (Real)(n == 0) : (x == x ? (Real)0 : x));
        Y[n] = (x == 0 ? -(Real)INFINITY : (x == x ? (Real)0 : x));
      }
      return;
    }
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    const Real eps = machine_eps<Real>();
    const Real pi = const_pi<Real>();
    const Real L = log(x / 2) - digamma_generic<Real>(1); // log(x/2) + Euler's constant
    if (x < 2) { // in q = -x^2/4: J_n = (x/2)^n sum q^k/(k! (k+n)!), Y_0 and Y_1 with the harmonic numbers H_k
      const Real q = -x * x / 4;
      for (Integer n = 0; n < 3; n++) {
        Real t = (n == 0 ? (Real)1 : (n == 1 ? x / 2 : x * x / 8));
        Real s = t;
        for (Integer k = 1; fabs(t) > eps * fabs(s); k++) {
          t *= q / (Real)(k * (k + n));
          s += t;
        }
        J[n] = s;
      }
      Real H = 0;
      Real t0 = 1;
      Real t1 = 1;
      Real s0 = 0; // sum H_k q^k/(k!)^2
      Real s1 = 1; // sum (H_k + H_(k+1)) q^k/(k! (k+1)!)
      for (Integer k = 1; k < 200; k++) {
        H += (Real)1 / (Real)k;
        t0 *= q / (Real)(k * k);
        t1 *= q / (Real)(k * (k + 1));
        s0 += H * t0;
        s1 += (2 * H + (Real)1 / (Real)(k + 1)) * t1;
        if (fabs(t0) <= eps * fabs(s0) && fabs(t1) <= eps * fabs(s1)) break;
      }
      Y[0] = 2 / pi * (L * J[0] - s0);
      Y[1] = 2 / pi * (L * J[1] - 1 / x) - x / (2 * pi) * s1;
    } else if (x < (Real)0.4 * SigBits) { // J_(k-1) = (2k/x) J_k - J_(k+1) from k = N, with J_0 + 2 (J_2 + J_4 + ...) = 1
      const Integer N = 2 * (Integer)((x + SigBits) / 2) + 2;
      Real jp = 0;
      Real jk = 1;
      Real norm = 0;
      Real s0 = 0; // sum (-1)^m J_2m/m
      Real s1 = 0; // sum (-1)^m (2m+1)/(m (m+1)) J_(2m+1)
      for (Integer k = N; k > 0; k--) {
        const Integer m = k / 2;
        if (k % 2 == 0) {
          norm += 2 * jk;
          s0 += (m % 2 ? -jk : jk) / (Real)m;
        } else if (k >= 3) {
          s1 += (m % 2 ? -jk : jk) * (Real)(2 * m + 1) / (Real)(m * (m + 1));
        }
        if (k <= 2) J[k] = jk;
        const Real jm = 2 * (Real)k / x * jk - jp;
        jp = jk;
        jk = jm;
      }
      norm += jk;
      J[0] = jk / norm;
      J[1] /= norm;
      J[2] /= norm;
      Y[0] = 2 / pi * (L * J[0] - 2 * s0 / norm);
      Y[1] = 2 / pi * ((L - 1) * J[1] - J[0] / x - s1 / norm);
    } else { // sqrt(2/(pi x)) (P cos xi - Q sin xi) and (P sin xi + Q cos xi), xi = x - pi/4 - n pi/2, by sin x and cos x
      const Real r = sqrt(2 / (pi * x));
      const Real sin_x = sin(x);
      const Real cos_x = cos(x);
      const Real u = (cos_x + sin_x) / sqrt<Real>((Real)2); // cos(x - pi/4)
      const Real v = (sin_x - cos_x) / sqrt<Real>((Real)2); // sin(x - pi/4)
      for (Integer n = 0; n < 2; n++) {
        Real P = 1;
        Real Q = 0;
        Real a = 1;
        for (Integer k = 1; k < 1000; k++) { // the terms a_k = a_(k-1) (4n^2 - (2k-1)^2)/(8 k x), to the smallest
          const Real a_next = a * (Real)(4 * n * n - (2 * k - 1) * (2 * k - 1)) / (8 * (Real)k * x);
          if (fabs(a_next) > fabs(a) || fabs(a_next) < eps) break;
          a = a_next;
          if (k % 2) Q += ((k / 2) % 2 ? -a : a);
          else P += ((k / 2) % 2 ? -a : a);
        }
        const Real c = (n == 0 ? u : v);
        const Real s = (n == 0 ? v : -u);
        J[n] = r * (P * c - Q * s);
        Y[n] = r * (P * s + Q * c);
      }
      J[2] = 2 * J[1] / x - J[0];
    }
    Y[2] = 2 * Y[1] / x - Y[0];
  }
  // J_n(a) and Y_n(a) of order n = 0, 1, 2 for a >= 0, inf or NaN by the forms of BesselCoeffs, each computed only when
  // its flag J or Y is set
  template <Integer n, bool J, bool Y, Integer digits, class VData> inline void bessel_jy_intrin(VData& j, VData& y, const VData& a) {
    using Real = typename VData::ScalarType;
    using Coeffs = BesselCoeffsOf<Real, digits, n>;
    const VData one = set1_intrin<VData>((Real)1);
    const Mask<VData> small = comp_intrin<ComparisonType::le>(a, set1_intrin<VData>((Real)5));
    const Integer n_small = mask_count_intrin(small);
    VData jr = zero_intrin<VData>();
    VData yr = jr;
    if (n_small) { // z = x^2 = zh + zl
      const VData zh = mul_intrin(a, a);
      const VData zl = mul_sub_exact_intrin(a, a, zh);
      const auto factor = [&zh, &zl](const double (&c)[2]) { // z - c to about twice the precision
        const Real hi = (Real)c[0];
        const Real lo = (Real)((c[0] - (double)hi) + c[1]);
        return add_intrin(sub_intrin(zh, set1_intrin<VData>(hi)), sub_intrin(zl, set1_intrin<VData>(lo)));
      };
      const VData t = sub_intrin(zh, set1_intrin<VData>((Real)12.5));
      const VData xn = (n == 0 ? one : (n == 1 ? a : zh));
      const VData f = mul_intrin(mul_intrin(factor(bessel_zeros_sq[n][0]), factor(bessel_zeros_sq[n][1])), xn);
      jr = mul_intrin(f, eval_poly_intrin(t, Coeffs::j));
      if constexpr (Y) { // -2/pi h_n = 0, -2/(pi x), -4/(pi x^2) - 1/pi, without x^2, which can be subnormal
        const VData log_a = (digits < 0 ? log_intrin(a) : approx_log_intrin<digits>(a));
        VData mh = zero_intrin<VData>();
        if constexpr (n == 1) mh = div_intrin(set1_intrin<VData>(-2 / const_pi<Real>()), a);
        if constexpr (n == 2) mh = sub_intrin(div_intrin(div_intrin(set1_intrin<VData>(-4 / const_pi<Real>()), a), a), set1_intrin<VData>(1 / const_pi<Real>()));
        const VData ys = fma_intrin(xn, eval_poly_intrin(t, Coeffs::y), fma_intrin(mul_intrin(log_a, set1_intrin<VData>(2 / const_pi<Real>())), jr, mh));
        yr = select_intrin(comp_intrin<ComparisonType::eq>(a, zero_intrin<VData>()), set1_intrin<VData>(-(Real)INFINITY), ys);
      }
    }
    if (n_small < VData::Size) { // J_n = r (P c - Q s), Y_n = r (P s + Q c), c and s sqrt(2) cos and sin of x - pi/4 - n pi/2
      const VData r = div_intrin(set1_intrin<VData>((Real)0.56418958354775628694807945156077259L), sqrt_intrin(a)); // 1/sqrt(pi x), not from 1/x, which can be subnormal
      const VData d = mul_intrin(mul_intrin(r, r), set1_intrin<VData>(const_pi<Real>())); // 1/x
      const VData w = mul_intrin(mul_intrin(d, d), set1_intrin<VData>((Real)25));
      VData sin_a, cos_a;
      approx_sincos_intrin<digits>(sin_a, cos_a, a);
      const VData u = add_intrin(cos_a, sin_a); // sqrt(2) cos(x - pi/4)
      const VData v = sub_intrin(sin_a, cos_a); // sqrt(2) sin(x - pi/4)
      const VData c = (n == 0 ? u : (n == 1 ? v : unary_minus_intrin(u)));
      const VData s = (n == 0 ? v : (n == 1 ? unary_minus_intrin(u) : unary_minus_intrin(v)));
      const VData ms = (n == 0 ? unary_minus_intrin(v) : (n == 1 ? u : v)); // -s
      const VData P = fma_intrin(w, eval_rational_intrin(w, Coeffs::pp, Coeffs::pq), one);
      const VData Q = mul_intrin(d, fma_intrin(w, eval_rational_intrin(w, Coeffs::qp, Coeffs::qq), set1_intrin<VData>((Real)(4 * n * n - 1) / 8)));
      const Mask<VData> inf = comp_intrin<ComparisonType::eq>(a, set1_intrin<VData>((Real)INFINITY));
      if constexpr (J) jr = select_intrin(small, jr, select_intrin(inf, zero_intrin<VData>(), mul_intrin(r, fma_intrin(P, c, mul_intrin(Q, ms)))));
      if constexpr (Y) yr = select_intrin(small, yr, select_intrin(inf, zero_intrin<VData>(), mul_intrin(r, fma_intrin(P, s, mul_intrin(Q, c)))));
    }
    if constexpr (J) j = jr;
    if constexpr (Y) y = yr;
  }
  // J_n(x), n = 0, 1, 2, to the given digits (-1: full)
  template <Integer n, Integer digits = -1, class VData> inline VData cyl_bessel_j_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData sgn = and_intrin(x, set1_intrin<VData>((Real)-0.0));
      VData j, y;
      bessel_jy_intrin<n, true, false, digits>(j, y, xor_intrin(x, sgn));
      return (n == 1 ? xor_intrin(j, sgn) : j);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        Real J[3], Y[3];
        bessel_jy_generic(J, Y, fabs(x_.x[i]));
        x_.x[i] = (n == 1 && x_.x[i] == 0 ? x_.x[i] : (n == 1 && x_.x[i] < 0 ? -J[n] : J[n])); // J_1 odd, also at -0
      }
      return x_.v;
    }
  }
  // Y_n(x), n = 0, 1, 2, to the given digits (-1: full), NaN for x < 0
  template <Integer n, Integer digits = -1, class VData> inline VData cyl_neumann_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      VData j, y;
      bessel_jy_intrin<n, false, true, digits>(j, y, x);
      return select_intrin(comp_intrin<ComparisonType::lt>(x, zero_intrin<VData>()), set1_intrin<VData>((Real)NAN), y);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        Real J[3], Y[3];
        bessel_jy_generic(J, Y, fabs(x_.x[i]));
        x_.x[i] = (x_.x[i] < 0 ? (Real)NAN : Y[n]);
      }
      return x_.v;
    }
  }
  // Minimax coefficients of I_n and K_n of order n = 0, 1, 2 for the given digits (7: float), lowest degree first,
  // from minimax.py bessel_ik
  template <Integer digits, Integer n> struct BesselIKCoeffs;
  template <> struct BesselIKCoeffs<7, 0> {
    static constexpr double i_lo = 9; // the small form of I_n below
    static constexpr double i_hi = 9; // the large form of I_n from
    static constexpr double is[] = {0.24999998949324265, 0.015625011666181205, 0.000434023754350706, 6.782314522942107e-06, 6.776387246761016e-08, 4.735488366932798e-10, 2.3252676362009085e-12, 1.0800299103434303e-14, 1.3794041044946004e-17, 1.5692084666117342e-19}; // I_n = x^n (c_n + z is(z)), c_n = 1/(2^n n!), z = x^2, x < i_lo
    static constexpr double il[] = {0.013888857094620304, 0.000868623489852085, 9.714254501959026e-05, 2.5774695540283596e-05, -6.871869805820262e-06, 6.554223842586333e-06}; // I_n = e^x/sqrt(2 pi x) (1 + t il(t)/il_q(t)), t = i_hi/x <= 1
    static constexpr double il_q[] = {1.0};
    static constexpr double ki[] = {0.24999999552363172, 0.015625044669084112, 0.0004338904458703124, 6.946942071661232e-06}; // I_n = x^n (c_n + z ki(z)) in K_n for x < 1
    static constexpr double ks[] = {0.11593151690948186, 0.27898284813768, 0.025249115569834196, 0.0008455955425064991, 1.5361912259433705e-05}; // K_n = (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/z - 1/2, x < 1
    static constexpr double kl[] = {-0.1249997134424807, -0.3582255942806568, -0.18607795489270473, -0.005294875846376951}; // K_n = e^-x sqrt(pi/(2x)) (1 + t kl(t)/kl_q(t)), t = 1/x <= 1
    static constexpr double kl_q[] = {1.0, 3.4281923208453082, 2.8329960867065616, 0.5061533858267837};
  };
  template <> struct BesselIKCoeffs<7, 1> {
    static constexpr double i_lo = 9; // the small form of I_n below
    static constexpr double i_hi = 9; // the large form of I_n from
    static constexpr double is[] = {0.062499999031606114, 0.0026041675688371416, 5.4253201382783656e-05, 6.782063743003412e-07, 5.648489424310336e-09, 3.3772003394884886e-11, 1.4645316969765351e-13, 5.859250823592211e-16, 7.855656053263318e-19, 6.860335551712824e-21}; // I_n = x^n (c_n + z is(z)), c_n = 1/(2^n n!), z = x^2, x < i_lo
    static constexpr double il[] = {-0.0416666304425583, -0.001447402785957528, -0.00013690268977506115, -3.1746718397786e-05, 7.3425787259988336e-06, -7.4334130276771476e-06}; // I_n = e^x/sqrt(2 pi x) (1 + t il(t)/il_q(t)), t = i_hi/x <= 1
    static constexpr double il_q[] = {1.0};
    static constexpr double ki[] = {0.06249999961425392, 0.002604170464345117, 5.4241905617459677e-05, 6.919947766167127e-07}; // I_n = x^n (c_n + z ki(z)) in K_n for x < 1
    static constexpr double ks[] = {-0.30796575824904937, -0.08537071411815494, -0.004642207524219038, -0.00011248793925601319, -1.6019658139721434e-06}; // K_n = (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/z - 1/2, x < 1
    static constexpr double kl[] = {0.37499955686887604, 1.0469119387957126, 0.5628434006265577, 0.03131164886711576}; // K_n = e^-x sqrt(pi/(2x)) (1 + t kl(t)/kl_q(t)), t = 1/x <= 1
    static constexpr double kl_q[] = {1.0, 3.1042078701525244, 2.1985201687671894, 0.29733718445336776};
  };
  template <> struct BesselIKCoeffs<7, 2> {
    static constexpr double i_lo = 9; // the small form of I_n below
    static constexpr double i_hi = 9; // the large form of I_n from
    static constexpr double is[] = {0.010416666607474563, 0.0003255208834101739, 5.4253333333065294e-06, 5.651585715018534e-08, 4.0353918300352434e-10, 2.108218576951932e-12, 8.187607219311191e-15, 2.867876401709957e-17, 3.9758915049152666e-20, 2.74695378903463e-22}; // I_n = x^n (c_n + z is(z)), c_n = 1/(2^n n!), z = x^2, x < i_lo
    static constexpr double il[] = {-0.2083333870202193, 0.010128253361721695, 0.0004165593880309206, 6.2320783158957e-05, -8.425572880100588e-06, 1.0931043846780383e-05}; // I_n = e^x/sqrt(2 pi x) (1 + t il(t)/il_q(t)), t = i_hi/x <= 1
    static constexpr double il_q[] = {1.0};
    static constexpr double ki[] = {0.010416666638819298, 0.0003255211062311442, 5.424518674721966e-06, 5.7502369870560336e-08}; // I_n = x^n (c_n + z ki(z)) in K_n for x < 1
    static constexpr double ks[] = {0.1082414395275003, 0.01596456369727948, 0.0006209654967752508, 1.1791847586214284e-05, 1.380658444193391e-07}; // K_n = (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/z - 1/2, x < 1
    static constexpr double kl[] = {1.875001309733612, 5.267282041539186, 3.6480751404252363, 0.5524996657495119}; // K_n = e^-x sqrt(pi/(2x)) (1 + t kl(t)/kl_q(t)), t = 1/x <= 1
    static constexpr double kl_q[] = {1.0, 2.3717506618201534, 1.071505672678339, 0.05061525030155449};
  };
  template <> struct BesselIKCoeffs<16, 0> {
    static constexpr double i_lo = 10; // the small form of I_n below
    static constexpr double i_hi = 15; // the large form of I_n from
    static constexpr double is[] = {0.24999999999999994, 0.01562500000000013, 0.0004340277777776944, 6.781684027803576e-06, 6.781684027322852e-08, 4.709502802110302e-10, 2.4028075121497727e-12, 9.385968919892172e-15, 2.896896266577768e-17, 7.242449683964901e-20, 1.4959591871369148e-22, 2.6031701195589564e-25, 3.7877598256166685e-28, 5.2957072847943906e-31, 3.6083078316642037e-34, 1.0186582713400755e-36}; // I_n = x^n (c_n + z is(z)), c_n = 1/(2^n n!), z = x^2, x < i_lo
    static constexpr double im[] = {0.008763133441829449, 0.005639374927901772, -0.010184854921460554, 0.01646210881609423, -0.0043091100338171675, 8.41268692511977e-05}; // I_n = e^x/sqrt(2 pi x) (1 + t im(s)/im_q(s)), s = t - (1 + i_hi/i_lo)/2, t = i_hi/x, i_lo <= x < i_hi
    static constexpr double im_q[] = {1.0, 0.6001270190475448, -1.1922214070992467, 1.9273408178543934, -0.5712311869670811, 0.02738788511770857};
    static constexpr double il[] = {0.008333333333330507, -0.0020319429654162215, -0.0034262166700781667, 0.0013020797803199547, -0.00010863181644195718, 8.744376982368183e-07}; // I_n = e^x/sqrt(2 pi x) (1 + t il(t)/il_q(t)), t = i_hi/x <= 1
    static constexpr double il_q[] = {1.0, -0.28133315586467433, -0.4032001735108669, 0.17183637495005316, -0.01839077351358782, 0.0004583330553854738};
    static constexpr double ki[] = {0.25, 0.015624999999999393, 0.0004340277777831026, 6.7816840056890995e-06, 6.781688915274609e-08, 4.708909864674292e-10, 2.439987601973322e-12}; // I_n = x^n (c_n + z ki(z)) in K_n for x < 1
    static constexpr double ks[] = {0.11593151565841245, 0.2789828789146034, 0.02524892993215875, 0.0008460350907346958, 1.491471920745097e-05, 1.6271073919471184e-07, 1.208230956608941e-09, 6.621479016797831e-12}; // K_n = (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/z - 1/2, x < 1
    static constexpr double kl[] = {-0.1249999999999972, -2.012760571839541, -11.963851865937736, -33.32485521112417, -45.69426533179854, -29.70081077257051, -8.041036359816939, -0.6546635045199246, -0.0028344594225496803}; // K_n = e^-x sqrt(pi/(2x)) (1 + t kl(t)/kl_q(t)), t = 1/x <= 1
    static constexpr double kl_q[] = {1.0, 16.664584574708623, 104.49870625169679, 316.5121756718166, 495.4973940773092, 398.9280842094767, 155.01817170672314, 25.043215608146454, 1.1635455494860518};
  };
  template <> struct BesselIKCoeffs<16, 1> {
    static constexpr double i_lo = 10; // the small form of I_n below
    static constexpr double i_hi = 15; // the large form of I_n from
    static constexpr double is[] = {0.06249999999999999, 0.0026041666666666748, 5.425347222221769e-05, 6.781684027790427e-07, 5.651403356277039e-09, 3.363930571436781e-11, 1.5017547038479584e-13, 5.214426818165533e-16, 1.4484491932901303e-18, 3.2919999276935237e-21, 6.233514363735131e-24, 1.0008292236176047e-26, 1.3557964046940927e-29, 1.749697884011255e-32, 1.1750627824326902e-35, 2.930799504458559e-38}; // I_n = x^n (c_n + z is(z)), c_n = 1/(2^n n!), z = x^2, x < i_lo
    static constexpr double im[] = {-0.025705214673601498, -0.0014623372525136044, 0.009732115952000082, -0.027647564785448154, 0.00751870372541552, -0.0001852767651770167}; // I_n = e^x/sqrt(2 pi x) (1 + t im(s)/im_q(s)), s = t - (1 + i_hi/i_lo)/2, t = i_hi/x, i_lo <= x < i_hi
    static constexpr double im_q[] = {1.0, 0.033000355164859864, -0.3811985730200789, 1.0843521431304797, -0.31778375590647084, 0.012912660047938876};
    static constexpr double il[] = {-0.02499999999999718, 0.013198408423778172, 0.003413486293853544, -0.002054020788778204, 0.00018966558309078705, -2.2438910248277577e-06}; // I_n = e^x/sqrt(2 pi x) (1 + t il(t)/il_q(t)), t = i_hi/x <= 1
    static constexpr double il_q[] = {1.0, -0.5487696702892899, -0.12632202799606984, 0.08534551524355462, -0.009163235158050986, 0.0001969727301670331};
    static constexpr double ki[] = {0.0625, 0.002604166666666632, 5.425347222252514e-05, 6.781684015297088e-07, 5.651406102632881e-09, 3.363598964391962e-11, 1.5224684378087743e-13}; // I_n = x^n (c_n + z ki(z)) in K_n for x < 1
    static constexpr double ks[] = {-0.3079657578292062, -0.08537071972865083, -0.00464218276647104, -0.0001125360703690081, -1.559288762392873e-06, -1.4030176905251253e-08, -8.870598024438757e-11, -4.2304894080785285e-13}; // K_n = (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/z - 1/2, x < 1
    static constexpr double kl[] = {0.37499999999999584, 5.894774568737717, 34.196045130396975, 93.08236041946775, 125.26024060421042, 80.7292355229297, 22.218683796625385, 1.992514208084416, 0.024592682192945782}; // K_n = e^-x sqrt(pi/(2x)) (1 + t kl(t)/kl_q(t)), t = 1/x <= 1
    static constexpr double kl_q[] = {1.0, 16.03189884996348, 95.92598457211724, 274.19729707974045, 398.9086088920535, 291.7840282753018, 99.39563370899981, 13.213747164861601, 0.4402498874500776};
  };
  template <> struct BesselIKCoeffs<16, 2> {
    static constexpr double i_lo = 10; // the small form of I_n below
    static constexpr double i_hi = 15; // the large form of I_n from
    static constexpr double is[] = {0.010416666666666666, 0.0003255208333333337, 5.425347222222041e-06, 5.651403356486246e-08, 4.036716683127968e-10, 2.102456606551816e-12, 8.343081721298338e-15, 2.6072132781658764e-17, 6.583863682008219e-20, 1.3716589787436715e-22, 2.397620396434177e-25, 3.573151270761884e-28, 4.528670505072395e-31, 5.420971772804902e-34, 3.5958831560574525e-37, 7.952937020062928e-40}; // I_n = x^n (c_n + z is(z)), c_n = 1/(2^n n!), z = x^2, x < i_lo
    static constexpr double im[] = {-0.12028599744590364, 0.08088111808573983, -0.06497195015085086, -0.017888875460170654, 0.008373880296723599, -0.0003678045182839588}; // I_n = e^x/sqrt(2 pi x) (1 + t im(s)/im_q(s)), s = t - (1 + i_hi/i_lo)/2, t = i_hi/x, i_lo <= x < i_hi
    static constexpr double im_q[] = {1.0, -0.6399001821705688, 0.5203760934897422, 0.16508045644893657, -0.06375968073976554, 0.0012040286010117798};
    static constexpr double il[] = {-0.12500000000000305, 0.10979199512961965, -0.024595828834890667, 0.0008084900970517447, 0.00011470544133191579, -4.263477067042346e-06}; // I_n = e^x/sqrt(2 pi x) (1 + t il(t)/il_q(t)), t = i_hi/x <= 1
    static constexpr double il_q[] = {1.0, -0.8491692943692646, 0.1727283595784068, -0.001999065917151219, -0.0008871405502266585, 1.1615516415656665e-05};
    static constexpr double ki[] = {0.010416666666666666, 0.0003255208333333316, 5.425347222237485e-06, 5.6514033502075034e-08, 4.0367180610275344e-10, 2.1022904913353443e-12, 8.446717300669922e-15}; // I_n = x^n (c_n + z ki(z)) in K_n for x < 1
    static constexpr double ks[] = {0.10824143945730155, 0.01596456439921958, 0.0006209629499755581, 1.1796141759082927e-05, 1.346502330733896e-07, 1.0309891024969893e-09, 5.6755701997945905e-12, 2.395922641322016e-14}; // K_n = (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/z - 1/2, x < 1
    static constexpr double kl[] = {1.8750000000000098, 28.342994636547537, 159.7356462863714, 431.3252484697625, 598.3552246518645, 425.05186827425223, 144.6518602881335, 19.911677056414202, 0.7125516042561155}; // K_n = e^-x sqrt(pi/(2x)) (1 + t kl(t)/kl_q(t)), t = 1/x <= 1
    static constexpr double kl_q[] = {1.0, 14.678763806160434, 78.93444802066675, 197.74535675399665, 243.35081618449985, 142.75033889779252, 35.67928827786996, 2.9082445568607764, 0.03277017661576984};
  };
  // BesselIKCoeffs<16, n> for double at full precision or digits > 7, else <7, n>
  template <class Real, Integer digits, Integer n> using BesselIKCoeffsOf = BesselIKCoeffs<((digits < 0 ? std::is_same<Real,double>::value : digits > 7) ? 16 : 7), n>;
  // I_n(x) and K_n(x), n = 0, 1, 2, of one value x >= 0: I by its power series below 0.4 SigBits and the asymptotic
  // series beyond, K by its power series up to 1/2 and the trapezoidal rule beyond
  template <class Real> inline void bessel_ik_generic(Real (&I)[3], Real (&K)[3], const Real x) {
    if (x == 0 || !(x < (Real)INFINITY)) { // 0, inf, NaN
      for (Integer n = 0; n < 3; n++) {
        I[n] = (x == 0 ? (Real)(n == 0) : x);
        K[n] = (x == 0 ? (Real)INFINITY : (x == x ? (Real)0 : x));
      }
      return;
    }
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    const Real eps = machine_eps<Real>();
    const Real pi = const_pi<Real>();
    const Real q = x * x / 4;
    if (x < (Real)0.4 * SigBits) { // I_n = (x/2)^n sum q^k/(k! (k+n)!)
      for (Integer n = 0; n < 3; n++) {
        Real t = (n == 0 ? (Real)1 : (n == 1 ? x / 2 : x * x / 8));
        Real s = t;
        for (Integer k = 1; t > eps * s; k++) {
          t *= q / (Real)(k * (k + n));
          s += t;
        }
        I[n] = s;
      }
    } else { // e^x/sqrt(2 pi x) sum b_k, b_k = b_(k-1) ((2k-1)^2 - 4n^2)/(8 k x), to the smallest term, with e^(x/2) twice
      const Real e = exp(x / 2);
      for (Integer n = 0; n < 3; n++) {
        Real s = 1;
        Real b = 1;
        for (Integer k = 1; k < 1000; k++) {
          const Real b_next = b * (Real)((2 * k - 1) * (2 * k - 1) - 4 * n * n) / (8 * (Real)k * x);
          if (fabs(b_next) > fabs(b) || fabs(b_next) < eps) break;
          b = b_next;
          s += b;
        }
        I[n] = e * (s / sqrt(2 * pi * x)) * e;
      }
    }
    if (x <= (Real)0.5) { // K_0 and K_1 by their series in q with the harmonic numbers H_k, L = log(x/2) + Euler's constant
      const Real L = log(x / 2) - digamma_generic<Real>(1);
      Real H = 0;
      Real t0 = 1;
      Real t1 = 1;
      Real s0 = 0; // sum H_k q^k/(k!)^2
      Real s1 = 1; // sum (H_k + H_(k+1)) q^k/(k! (k+1)!)
      for (Integer k = 1; k < 200; k++) {
        H += (Real)1 / (Real)k;
        t0 *= q / (Real)(k * k);
        t1 *= q / (Real)(k * (k + 1));
        s0 += H * t0;
        s1 += (2 * H + (Real)1 / (Real)(k + 1)) * t1;
        if (t0 <= eps * s0 && t1 <= eps * s1) break;
      }
      K[0] = s0 - L * I[0];
      K[1] = 1 / x + L * I[1] - x / 4 * s1;
    } else { // e^-x h (1/2 + sum_j e^(-x (cosh(jh) - 1)) cosh(n jh)), the trapezoidal rule for the integral of e^(-x cosh t) cosh(nt)
      const Real c = (Real)(SigBits + 8) * log((Real)2) + (Real)0.3 * x; // error e^-c: within |Im t| < pi/4 the integrand grows by e^(0.3 x)
      const Real h = pi * pi / (2 * c);
      Real s0 = (Real)0.5;
      Real s1 = (Real)0.5;
      for (Integer j = 1; j < 100000; j++) { // u = e^(jh/2): cosh(jh) - 1 = 2 sinh(jh/2)^2 = (u - 1/u)^2/2
        const Real u = exp((Real)j * h / 2);
        const Real d = u - 1 / u;
        const Real f = exp(-x * d * d / 2);
        s0 += f;
        s1 += f * (u * u + 1 / (u * u)) / 2;
        if (f < eps * eps) break;
      }
      const Real e = exp(-x) * h;
      K[0] = e * s0;
      K[1] = e * s1;
    }
    K[2] = K[0] + 2 * K[1] / x;
  }
  // I_n(a) of order n = 0, 1, 2 for a >= 0, inf or NaN by the forms of BesselIKCoeffs
  template <Integer n, Integer digits, class VData> inline VData bessel_i_intrin(const VData& a) {
    using Real = typename VData::ScalarType;
    using Coeffs = BesselIKCoeffsOf<Real, digits, n>;
    const VData one = set1_intrin<VData>((Real)1);
    const VData cn = set1_intrin<VData>((Real)(n == 0 ? 1 : (n == 1 ? 0.5 : 0.125)));
    const Mask<VData> small = comp_intrin<ComparisonType::lt>(a, set1_intrin<VData>((Real)Coeffs::i_lo));
    const Integer n_small = mask_count_intrin(small);
    VData r = zero_intrin<VData>();
    if (n_small) {
      const VData z = mul_intrin(a, a);
      const VData xn = (n == 0 ? one : (n == 1 ? a : z));
      r = mul_intrin(xn, fma_intrin(z, eval_poly_intrin(z, Coeffs::is), cn));
    }
    if (n_small < VData::Size) { // e^x/sqrt(2 pi x) (1 + t R), t = i_hi/x, with e^(x/2) twice where e^x can overflow
      const auto exp_ = [](const VData& v) {
        if constexpr (digits < 0) return exp_intrin(v);
        else return approx_exp_intrin<exp_taylor_order(digits), true>(v);
      };
      const VData t = div_intrin(set1_intrin<VData>((Real)Coeffs::i_hi), a);
      VData R;
      if constexpr (Coeffs::i_hi > Coeffs::i_lo) { // im on [i_lo, i_hi), in s = t - (1 + i_hi/i_lo)/2
        const Mask<VData> mid = comp_intrin<ComparisonType::gt>(t, one); // also the lanes of small
        const Integer n_mid = mask_count_intrin(mid) - n_small;
        R = (n_small + n_mid < VData::Size ? eval_rational_intrin(t, Coeffs::il, Coeffs::il_q) : one);
        if (n_mid) R = select_intrin(mid, eval_rational_intrin(sub_intrin(t, set1_intrin<VData>((Real)((1 + Coeffs::i_hi / Coeffs::i_lo) / 2))), Coeffs::im, Coeffs::im_q), R);
      } else {
        R = eval_rational_intrin(t, Coeffs::il, Coeffs::il_q);
      }
      const VData g = mul_intrin(fma_intrin(t, R, one), sqrt_intrin(mul_intrin(t, set1_intrin<VData>((Real)(0.1591549430918953357688837633725143620345L / Coeffs::i_hi))))); // (1 + t R)/sqrt(2 pi x), 1/x = t/i_hi
      static constexpr Real e_max = (std::is_same<Real,float>::value ? (Real)88 : (Real)709); // e^x is finite below
      VData rb;
      if (mask_count_intrin(comp_intrin<ComparisonType::gt>(a, set1_intrin<VData>(e_max)))) {
        const VData e = exp_(mul_intrin(a, set1_intrin<VData>((Real)0.5)));
        rb = mul_intrin(mul_intrin(e, g), e);
      } else {
        rb = mul_intrin(exp_(a), g);
      }
      rb = select_intrin(comp_intrin<ComparisonType::eq>(a, set1_intrin<VData>((Real)INFINITY)), a, rb);
      r = select_intrin(small, r, rb);
    }
    return r;
  }
  // K_n(a) of order n = 0, 1, 2 for a >= 0, inf or NaN by the forms of BesselIKCoeffs
  template <Integer n, Integer digits, class VData> inline VData bessel_k_intrin(const VData& a) {
    using Real = typename VData::ScalarType;
    using Coeffs = BesselIKCoeffsOf<Real, digits, n>;
    const VData one = set1_intrin<VData>((Real)1);
    const VData two = set1_intrin<VData>((Real)2);
    const Mask<VData> small = comp_intrin<ComparisonType::lt>(a, one);
    const Integer n_small = mask_count_intrin(small);
    VData r = zero_intrin<VData>();
    if (n_small) { // (-1)^(n+1) log(x) I_n + h_n + x^n ks(z), h_n = 0, 1/x, 2/x^2 - 1/2 without x^2, which can be subnormal
      const VData z = mul_intrin(a, a);
      const VData xn = (n == 0 ? one : (n == 1 ? a : z));
      const VData I = mul_intrin(xn, fma_intrin(z, eval_poly_intrin(z, Coeffs::ki), set1_intrin<VData>((Real)(n == 0 ? 1 : (n == 1 ? 0.5 : 0.125)))));
      VData L = (digits < 0 ? log_intrin(a) : approx_log_intrin<digits>(a));
      if constexpr (n != 1) L = unary_minus_intrin(L);
      VData h = zero_intrin<VData>();
      if constexpr (n == 1) h = div_intrin(one, a);
      if constexpr (n == 2) h = sub_intrin(div_intrin(div_intrin(two, a), a), set1_intrin<VData>((Real)0.5));
      r = fma_intrin(L, I, fma_intrin(xn, eval_poly_intrin(z, Coeffs::ks), h));
      r = select_intrin(comp_intrin<ComparisonType::eq>(a, zero_intrin<VData>()), set1_intrin<VData>((Real)INFINITY), r);
    }
    if (n_small < VData::Size) { // e^-x sqrt(pi/(2x)) (1 + t R), t = 1/x
      VData e;
      if constexpr (digits < 0) e = exp_intrin(unary_minus_intrin(a));
      else e = approx_exp_intrin<exp_taylor_order(digits), true>(unary_minus_intrin(a));
      const VData t = div_intrin(one, a);
      const VData g = mul_intrin(fma_intrin(t, eval_rational_intrin(t, Coeffs::kl, Coeffs::kl_q), one), sqrt_intrin(mul_intrin(t, set1_intrin<VData>((Real)1.570796326794896619231321691639751442099L)))); // (1 + t R) sqrt(pi/(2x))
      const VData rb = select_intrin(comp_intrin<ComparisonType::eq>(a, set1_intrin<VData>((Real)INFINITY)), zero_intrin<VData>(), mul_intrin(e, g));
      r = select_intrin(small, r, rb);
    }
    return r;
  }
  // I_n(x), n = 0, 1, 2, to the given digits (-1: full)
  template <Integer n, Integer digits = -1, class VData> inline VData cyl_bessel_i_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData sgn = and_intrin(x, set1_intrin<VData>((Real)-0.0));
      const VData r = bessel_i_intrin<n, digits>(xor_intrin(x, sgn));
      return (n == 1 ? xor_intrin(r, sgn) : r);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        Real I[3], K[3];
        bessel_ik_generic(I, K, fabs(x_.x[i]));
        x_.x[i] = (n == 1 && x_.x[i] == 0 ? x_.x[i] : (n == 1 && x_.x[i] < 0 ? -I[n] : I[n])); // I_1 odd, also at -0
      }
      return x_.v;
    }
  }
  // K_n(x), n = 0, 1, 2, to the given digits (-1: full), NaN for x < 0
  template <Integer n, Integer digits = -1, class VData> inline VData cyl_bessel_k_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      return select_intrin(comp_intrin<ComparisonType::lt>(x, zero_intrin<VData>()), set1_intrin<VData>((Real)NAN), bessel_k_intrin<n, digits>(x));
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        Real I[3], K[3];
        bessel_ik_generic(I, K, fabs(x_.x[i]));
        x_.x[i] = (x_.x[i] < 0 ? (Real)NAN : K[n]);
      }
      return x_.v;
    }
  }

  // Minimax coefficients of the spherical Bessel functions j_1 and j_2 for x < 2 and the given digits (7: float), lowest
  // degree first, from minimax.py sph_bessel
  template <Integer digits> struct SphBesselCoeffs;
  template <> struct SphBesselCoeffs<7> {
    static constexpr double j1[] = {0.33333333330846193, -0.033333332912193765, 0.00119047501321979, -2.2044637075596085e-05, 2.4994573198436495e-07, -1.8005195322060004e-09}; // j_1 = x j1(z), z = x^2 < 4
    static constexpr double j2[] = {0.06666666638639736, -0.004761901425636761, 0.00013226866049139796, -1.999722272691627e-06, 1.8009557019454032e-08}; // j_2 = z j2(z)
  };
  template <> struct SphBesselCoeffs<16> {
    static constexpr double j1[] = {0.3333333333333333, -0.0333333333333333, 0.0011904761904759594, -2.2045855378611925e-05, 2.5052108312941436e-07, -1.9270847464352463e-09, 1.0705814406494299e-11, -4.4930789235126065e-14, 1.4100555340568715e-16}; // j_1 = x j1(z), z = x^2 < 4
    static constexpr double j2[] = {0.06666666666666667, -0.004761904761904761, 0.00013227513227512258, -2.00416867081094e-06, 1.9270852573300347e-08, -1.2847232866692714e-10, 6.297571716925392e-13, -2.3652706827617343e-15, 6.7438976692145565e-18}; // j_2 = z j2(z)
  };
  // j_n(x) and y_n(x), n = 0, 1, 2, of one value x >= 0 by their forms in sin x and cos x, with the power series of
  // j_1 and j_2 below 2
  template <class Real> inline void sph_bessel_generic(Real (&j)[3], Real (&y)[3], const Real x) {
    if (x == 0 || !(x < (Real)INFINITY)) { // 0, inf, NaN
      for (Integer n = 0; n < 3; n++) {
        j[n] = (x == 0 ? (Real)(n == 0) : (x == x ? (Real)0 : x));
        y[n] = (x == 0 ? -(Real)INFINITY : (x == x ? (Real)0 : x));
      }
      return;
    }
    const Real s = sin(x);
    const Real c = cos(x);
    j[0] = s / x;
    j[1] = (s / x - c) / x;
    j[2] = (3 * j[1] - s) / x;
    y[0] = -c / x;
    y[1] = -(c / x + s) / x;
    y[2] = (3 * y[1] + c) / x;
    if (x < 2) { // j_n = x^n/(2n+1)!! sum (-x^2/2)^k/(k! (2n+3) (2n+5) ... (2n+2k+1))
      const Real eps = machine_eps<Real>();
      for (Integer n = 1; n < 3; n++) {
        Real t = (n == 1 ? x / 3 : x * x / 15);
        Real sum = t;
        for (Integer k = 1; fabs(t) > eps * fabs(sum); k++) {
          t *= -x * x / (Real)(2 * k * (2 * n + 2 * k + 1));
          sum += t;
        }
        j[n] = sum;
      }
    }
  }
  // j_n(a) and y_n(a) of order n = 0, 1, 2 for a >= 0, inf or NaN from one sincos, with SphBesselCoeffs for j_1 and j_2
  // below 2, each computed only when its flag J or Y is set
  template <Integer n, bool J, bool Y, Integer digits, class VData> inline void sph_bessel_jy_intrin(VData& j, VData& y, const VData& a) {
    using Real = typename VData::ScalarType;
    using Coeffs = SphBesselCoeffs<((digits < 0 ? std::is_same<Real,double>::value : digits > 7) ? 16 : 7)>;
    VData s, c;
    approx_sincos_intrin<digits>(s, c, a);
    const Mask<VData> inf = comp_intrin<ComparisonType::eq>(a, set1_intrin<VData>((Real)INFINITY));
    if constexpr (n == 0) { // j_0 = s/x, y_0 = -c/x, by division: 1/x can overflow
      if constexpr (J) j = select_intrin(comp_intrin<ComparisonType::eq>(a, zero_intrin<VData>()), set1_intrin<VData>((Real)1), select_intrin(inf, zero_intrin<VData>(), div_intrin(s, a)));
      if constexpr (Y) y = select_intrin(comp_intrin<ComparisonType::eq>(a, zero_intrin<VData>()), set1_intrin<VData>(-(Real)INFINITY), select_intrin(inf, zero_intrin<VData>(), div_intrin(unary_minus_intrin(c), a)));
    } else { // j_(n+1) = (2n+1)/x j_n - j_(n-1) and y_(n+1) likewise, d = 1/x
      const VData d = div_intrin(set1_intrin<VData>((Real)1), a);
      if constexpr (J) {
        VData jr = mul_intrin(fma_intrin(s, d, unary_minus_intrin(c)), d);
        if constexpr (n == 2) jr = mul_intrin(fma_intrin(jr, set1_intrin<VData>((Real)3), unary_minus_intrin(s)), d);
        const Mask<VData> small = comp_intrin<ComparisonType::lt>(a, set1_intrin<VData>((Real)2));
        if (mask_count_intrin(small)) {
          const VData z = mul_intrin(a, a);
          jr = select_intrin(small, (n == 1 ? mul_intrin(a, eval_poly_intrin(z, Coeffs::j1)) : mul_intrin(z, eval_poly_intrin(z, Coeffs::j2))), jr);
        }
        j = select_intrin(inf, zero_intrin<VData>(), jr);
      }
      if constexpr (Y) {
        VData yr = mul_intrin(unary_minus_intrin(fma_intrin(c, d, s)), d);
        if constexpr (n == 2) yr = mul_intrin(fma_intrin(yr, set1_intrin<VData>((Real)3), c), d);
        y = select_intrin(comp_intrin<ComparisonType::eq>(a, zero_intrin<VData>()), set1_intrin<VData>(-(Real)INFINITY), select_intrin(inf, zero_intrin<VData>(), yr)); // also at -0
      }
    }
  }
  // j_n(x), n = 0, 1, 2, to the given digits (-1: full)
  template <Integer n, Integer digits = -1, class VData> inline VData sph_bessel_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      const VData sgn = and_intrin(x, set1_intrin<VData>((Real)-0.0));
      VData j, y;
      sph_bessel_jy_intrin<n, true, false, digits>(j, y, xor_intrin(x, sgn));
      return (n == 1 ? xor_intrin(j, sgn) : j);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        Real j[3], y[3];
        sph_bessel_generic(j, y, fabs(x_.x[i]));
        x_.x[i] = (n == 1 && x_.x[i] == 0 ? x_.x[i] : (n == 1 && x_.x[i] < 0 ? -j[n] : j[n])); // j_1 odd, also at -0
      }
      return x_.v;
    }
  }
  // y_n(x), n = 0, 1, 2, to the given digits (-1: full), NaN for x < 0
  template <Integer n, Integer digits = -1, class VData> inline VData sph_neumann_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      VData j, y;
      sph_bessel_jy_intrin<n, false, true, digits>(j, y, x);
      return select_intrin(comp_intrin<ComparisonType::lt>(x, zero_intrin<VData>()), set1_intrin<VData>((Real)NAN), y);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {x};
      for (Integer i = 0; i < VData::Size; i++) {
        Real j[3], y[3];
        sph_bessel_generic(j, y, fabs(x_.x[i]));
        x_.x[i] = (x_.x[i] < 0 ? (Real)NAN : y[n]);
      }
      return x_.v;
    }
  }

  // Minimax coefficients of zeta(s), s >= 0, on pieces of [0, inf) for the given digits (7: float), lowest degree
  // first, from minimax.py zeta
  template <Integer digits> struct ZetaCoeffs;
  template <> struct ZetaCoeffs<7> {
    static constexpr double bound[] = {0.0, 2.0, 8.0, 24.0}; // the pieces [bound[k], bound[k+1]], the last also beyond
    static constexpr double p[][4] = {{0.07721566491880283, 0.003456362714870402, -0.00010479644523559637, 8.85370300715344e-06}, {1.1816880582160783, 0.28537816420258116, 0.024487970817811715, 0.0007597628745129845}, {1.001539057072135, 0.16721014857494867, 0.010078667822946497, 0.00022105575454182567}}; // zeta = 1/(s-1) + 1/2 + s p/q on piece 0, 1 + 2^-s p/q on the others, in s minus the middle of the piece
    static constexpr double q[][4] = {{1.0, 0.10174337468187815, 0.010208011426600137, -5.151831713216234e-05}, {1.0, 0.3221286569368545, 0.02148777098179688, 0.0008639796388433123}, {1.0, 0.16758047511417035, 0.010039944922421648, 0.00022346419177095744}};
  };
  template <> struct ZetaCoeffs<16> {
    static constexpr double bound[] = {0.0, 2.0, 6.0, 16.0, 56.0}; // the pieces [bound[k], bound[k+1]], the last also beyond
    static constexpr double p[][7] = {{0.07721566490153287, 0.01627728668004219, 0.001570400209994251, 0.00010665422527260776, 6.318141926388837e-06, 1.0920227894874557e-07, 8.264217507560348e-10}, {1.317171739378211, 0.4223613741628973, 0.056632161827418294, 0.0042055620904412425, 0.00020038705662212325, 6.621329678410656e-06, 1.2104559491923736e-07}, {1.0120982612366625, 0.2127738803846578, 0.01930020706084039, 0.0010149043644009576, 3.37885388912219e-05, 6.502394460673083e-07, 4.86451539396763e-09}, {1.0000004578687474, 0.15336406528877736, 0.010229731485684563, 0.00037971817670110316, 8.263984399757733e-06, 9.98482542548548e-08, 5.224691293178791e-10}}; // zeta = 1/(s-1) + 1/2 + s p/q on piece 0, 1 + 2^-s p/q on the others, in s minus the middle of the piece
    static constexpr double q[][7] = {{1.0, 0.267783825008387, 0.041364182587911115, 0.0039480784792415916, 0.00026782377715759694, 1.1386657282537153e-05, 3.1292678350619924e-07}, {1.0, 0.4645922264673974, 0.05472183640830018, 0.004209864129399663, 0.00020302381805482382, 6.532166103557671e-06, 1.2190716731550882e-07}, {1.0, 0.2152422782577208, 0.019071481714031356, 0.0010273198658372034, 3.3368397438079974e-05, 6.587324166991156e-07, 4.783417075541603e-09}, {1.0, 0.15336418064290988, 0.010229717580991417, 0.00037971919961093416, 8.263927462554568e-06, 9.984989722583438e-08, 5.223990080059563e-10}};
  };
  // Two-part constants of zeta: log(2 pi) and log(pi)
  template <class Real> struct ZetaConsts {
    static constexpr Real l2pi_hi = (Real)1.8378770664093456;
    static constexpr Real l2pi_lo = (Real)((1.8378770664093456 - (double)l2pi_hi) - 7.756588316134483e-17);
    static constexpr Real lpi_hi = (Real)1.1447298858494002;
    static constexpr Real lpi_lo = (Real)((1.1447298858494002 - (double)lpi_hi) + 1.0265951162707826e-17);
  };
  // zeta(s) of one value: the alternating series of Borwein for s >= 0, with 1 - 2^(1-s) from u = s - 1, and the
  // reflection formula below 0
  template <class Real> inline Real riemann_zeta_generic(const Real s) {
    const Real pi = const_pi<Real>();
    const auto zeta_pos = [](const Real t, const Real u) { // zeta(t), t = 1 + u >= 0: eta(t)/(1 - 2^-u)
      if (u == 0) return (Real)INFINITY;
      if (!(t < (Real)INFINITY)) return (t == t ? (Real)1 : t);
      static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
      static constexpr Integer n = SigBits * 2 / 5 + 2; // error about (3 + sqrt(8))^-n
      Real d[n + 1]; // d_k = n sum_(i <= k) (n+i-1)! 4^i/((n-i)! (2i)!)
      Real term = (Real)1 / n;
      d[0] = n * term;
      for (Integer i = 1; i <= n; i++) {
        term *= (Real)(4 * (n + i - 1) * (n - i + 1)) / (Real)(2 * i * (2 * i - 1));
        d[i] = d[i - 1] + n * term;
      }
      Real sum = 0;
      for (Integer k = n - 1; k >= 0; k--) sum += (k % 2 ? -1 : 1) * (d[k] - d[n]) / pow((Real)(k + 1), t);
      const Real x = -u * log((Real)2); // 1 - 2^-u = -expm1(x)
      Real em1 = exp(x) - 1;
      if (fabs(x) < (Real)0.5) {
        Real xk = x;
        em1 = 0;
        for (Integer k = 1; k < 200 && xk != 0; k++) {
          em1 += xk;
          xk *= x / (k + 1);
        }
      }
      return -sum / d[n] / (-em1);
    };
    if (!(s < 0)) return zeta_pos(s, s - 1); // also NaN
    if (s == -(Real)INFINITY) return (Real)NAN;
    const Real h = s / 2; // sinpi(s/2) = (-1)^k sin(pi (h - k)), k = round(h)
    const Real k = round(h);
    const Real sn = sin(pi * (h - k)) * (fmod(k, (Real)2) == 0 ? 1 : -1);
    if (sn == 0) return 0;
    Real g; // (2 pi)^s Gamma(1-s), with Gamma(1-s) = -s Gamma(-s) as 1-s can be inexact
    if constexpr (std::is_floating_point<Real>::value) g = -s * std::tgamma(-s) * pow(2 * pi, s);
    else g = -s * tgamma_generic(-s) * pow(2 * pi, s);
    if (!(g < (Real)INFINITY)) g = exp(s * log(2 * pi) + log(-s) + lgamma_generic(-s));
    return g / pi * sn * zeta_pos(1 - s, -s);
  }
  // zeta(s) for s >= 0, inf or NaN, given u = s - 1, by ZetaCoeffs with the coefficients of each lane's piece
  template <Integer digits, class VData> inline VData zeta_pos_intrin(const VData& s, const VData& u) {
    using Real = typename VData::ScalarType;
    using Coeffs = ZetaCoeffs<((digits < 0 ? std::is_same<Real,double>::value : digits > 7) ? 16 : 7)>;
    static constexpr Integer K = std::extent<decltype(Coeffs::p), 0>::value; // number of pieces
    static constexpr Integer M = std::extent<decltype(Coeffs::p), 1>::value; // number of coefficients
    const VData one = set1_intrin<VData>((Real)1);
    Mask<VData> below[K - 1]; // s < bound[k+1]
    bool any[K - 1]; // some lane in piece k
    Integer n_prev = 0;
    for (Integer k = 0; k < K - 1; k++) {
      below[k] = comp_intrin<ComparisonType::lt>(s, set1_intrin<VData>((Real)Coeffs::bound[k + 1]));
      const Integer n_k = mask_count_intrin(below[k]);
      any[k] = (n_k > n_prev);
      n_prev = n_k;
    }
    const auto piecewise = [&below, &any](const auto& c) { // c(k) of the piece k of each lane
      VData r = set1_intrin<VData>((Real)c(K - 1));
      for (Integer k = K - 2; k >= 0; k--) {
        if (any[k]) r = select_intrin(below[k], set1_intrin<VData>((Real)c(k)), r);
      }
      return r;
    };
    const VData t = sub_intrin(min_intrin(set1_intrin<VData>((Real)Coeffs::bound[K]), s), piecewise([](const Integer k) { return (Coeffs::bound[k] + Coeffs::bound[k + 1]) / 2; })); // keeps NaN
    const auto eval = [&piecewise, &t](const auto& c) { // sum_i c[k][i] t^i by Horner's scheme
      VData r = piecewise([&c](const Integer k) { return c[k][M - 1]; });
      for (Integer i = M - 2; i >= 0; i--) r = fma_intrin(r, t, piecewise([&c, i](const Integer k) { return c[k][i]; }));
      return r;
    };
    const VData R = div_intrin(eval(Coeffs::p), eval(Coeffs::q));
    const Integer n_pole = mask_count_intrin(below[0]);
    VData r = zero_intrin<VData>();
    if (n_pole < VData::Size) r = fma_intrin(exp2_intrin(unary_minus_intrin(s)), R, one);
    if (n_pole) r = select_intrin(below[0], add_intrin(div_intrin(one, u), fma_intrin(s, R, set1_intrin<VData>((Real)0.5))), r);
    return r;
  }
  // s1 + s2 = a + b exactly
  template <class VData> inline void two_sum_intrin(VData& s1, VData& s2, const VData& a, const VData& b) {
    s1 = add_intrin(a, b);
    const VData bb = sub_intrin(s1, a);
    s2 = add_intrin(sub_intrin(a, sub_intrin(s1, bb)), sub_intrin(b, bb));
  }
  // (2 pi)^s/pi sinpi(s/2) Gamma(1-s) for s < 0 where Gamma(1-s) overflows, in one exponential of
  // lgamma(-s) + log(-s) + s log(2 pi) - log(pi) + log|sinpi(s/2)|, as 1-s can be inexact; arguments and result by value
  template <Integer digits, class VData> [[gnu::noinline]] VData zeta_reflection_big_intrin(const VData s, const VData sn) {
    using Real = typename VData::ScalarType;
    using Consts = ZetaConsts<Real>;
    static constexpr Real big_x = pow<TypeTraits<Real>::SigBits,Real>((Real)2);
    const VData x = min_intrin(set1_intrin<VData>(big_x), unary_minus_intrin(s));
    VData e, f, lh, ll, L_hi, L_lo;
    log_split_intrin<false>(e, f, x);
    log_hi_lo_intrin(lh, ll, e, f);
    lgamma_stirling_intrin<GammaCoeffsOf<Real, digits>>(L_hi, L_lo, x, lh, ll);
    const VData p_hi = mul_intrin(s, set1_intrin<VData>(Consts::l2pi_hi));
    const VData p_lo = fma_intrin(s, set1_intrin<VData>(Consts::l2pi_lo), mul_sub_exact_intrin(s, set1_intrin<VData>(Consts::l2pi_hi), p_hi));
    VData h, h_err, g, g_err, l, l_err;
    two_sum_intrin(l, l_err, L_hi, lh); // lgamma(1-s) = lgamma(-s) + log(-s)
    two_sum_intrin(h, h_err, l, p_hi);
    two_sum_intrin(g, g_err, h, set1_intrin<VData>(-Consts::lpi_hi));
    const VData a = fabs_intrin(sn);
    const VData la = (digits < 0 ? log_intrin(a) : approx_log_intrin<digits>(a));
    VData k, k_err;
    two_sum_intrin(k, k_err, g, la);
    const VData lo = add_intrin(add_intrin(add_intrin(h_err, g_err), add_intrin(k_err, l_err)), add_intrin(add_intrin(L_lo, ll), sub_intrin(p_lo, set1_intrin<VData>(Consts::lpi_lo))));
    const VData T = add_intrin(k, lo);
    static constexpr Integer order = (digits < 0 ? (Integer)(TypeTraits<Real>::SigBits / 3.8) : exp_taylor_order(digits));
    return copysign_intrin(approx_exp_intrin<order, true, ExpArg::Sum>(T, sub_intrin(lo, sub_intrin(T, k))), sn);
  }
  // zeta(s), the Riemann zeta function, to the given digits (-1: full): ZetaCoeffs for s >= 0 and the reflection
  // formula zeta(s) = (2 pi)^s/pi sinpi(s/2) Gamma(1-s) zeta(1-s) below
  template <Integer digits = -1, class VData> inline VData riemann_zeta_intrin(const VData& s) {
    using Real = typename VData::ScalarType;
    if constexpr (std::is_same<Real,float>::value || std::is_same<Real,double>::value) {
      using Consts = ZetaConsts<Real>;
      const VData zero = zero_intrin<VData>();
      const VData one = set1_intrin<VData>((Real)1);
      const Mask<VData> neg = comp_intrin<ComparisonType::lt>(s, set1_intrin<VData>((Real)-1 / (1 << 20))); // above, s/2 can underflow and ZetaCoeffs hold
      const Integer n_neg = mask_count_intrin(neg);
      if (!n_neg) return zeta_pos_intrin<digits>(s, sub_intrin(s, one));
      const VData z = zeta_pos_intrin<digits>(select_intrin(neg, sub_intrin(one, s), s), select_intrin(neg, unary_minus_intrin(s), sub_intrin(s, one))); // zeta(1-s) with 1-s - 1 = -s exact
      VData sn, cs;
      approx_sincospi_intrin<digits, true, true, false>(sn, cs, mul_intrin(s, set1_intrin<VData>((Real)0.5)));
      static constexpr Real s_big = (std::is_same<Real,float>::value ? (Real)-34 : (Real)-170); // Gamma(1-s) is finite above
      const Mask<VData> big = comp_intrin<ComparisonType::lt>(s, set1_intrin<VData>(s_big));
      const Integer n_big = mask_count_intrin(big);
      VData g = zero; // (2 pi)^s/pi sinpi(s/2) Gamma(1-s), with Gamma(1-s) = -s Gamma(-s) as 1-s can be inexact
      if (n_big < n_neg) {
        const VData p_hi = mul_intrin(s, set1_intrin<VData>(Consts::l2pi_hi));
        const VData p_lo = fma_intrin(s, set1_intrin<VData>(Consts::l2pi_lo), mul_sub_exact_intrin(s, set1_intrin<VData>(Consts::l2pi_hi), p_hi));
        const VData p = add_intrin(p_hi, p_lo);
        static constexpr Integer order = (digits < 0 ? (Integer)(TypeTraits<Real>::SigBits / 3.8) : exp_taylor_order(digits));
        const VData e = approx_exp_intrin<order, true, ExpArg::Sum>(p, sub_intrin(p_lo, sub_intrin(p, p_hi)));
        const VData ms = select_intrin(neg, unary_minus_intrin(s), one);
        g = mul_intrin(mul_intrin(mul_intrin(tgamma_intrin<digits>(ms), ms), e), mul_intrin(sn, set1_intrin<VData>(1 / const_pi<Real>())));
      }
      if (n_big) g = select_intrin(big, zeta_reflection_big_intrin<digits>(s, sn), g);
      const VData r = select_intrin(comp_intrin<ComparisonType::eq>(sn, zero), zero, mul_intrin(g, z)); // the zeros at negative even s
      return select_intrin(neg, r, z);
    } else {
      union {
        VData v;
        Real x[VData::Size];
      } x_ = {s};
      for (Integer i = 0; i < VData::Size; i++) x_.x[i] = riemann_zeta_generic(x_.x[i]);
      return x_.v;
    }
  }
  template <class VData> inline VData cbrt_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    union {
      VData v;
      Real x[VData::Size];
    } x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) {
      if constexpr (std::is_floating_point<Real>::value) {
        x_.x[i] = std::cbrt(x_.x[i]);
      } else { // one Newton step corrects the rounding of 1/3
        const Real a = fabs(x_.x[i]);
        Real y = pow(a, 1/(Real)3);
        y = (y == 0 || isinf(y) ? y : y - (y*y*y - a) / (3*y*y));
        x_.x[i] = (x_.x[i] < 0 ? -y : y);
      }
    }
    return x_.v;
  }
  template <class VData> inline VData fmod_intrin(const VData& x, const VData& y) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U x_ = {x};
    U y_ = {y};
    for (Integer i = 0; i < VData::Size; i++) x_.x[i] = fmod(x_.x[i], y_.x[i]);
    return x_.v;
  }

  // Hyperbolic: e^|x|/2 = exp(|x| - 1) e/2 avoids overflow; sinh, tanh by series for |x| < 1
  template <class VData> inline VData sinh_series_intrin(const VData& x) { // x (1 + x^2/3! + x^4/5! + ...), |x| < 1
    using Real = typename VData::ScalarType;
    constexpr Integer K = [] { // terms until 1/(2k+1)! < 2^-(SigBits+1)
      Integer k = 1;
      double term = 1.0/6;
      while (term > pow<-(TypeTraits<Real>::SigBits+1),double>(2.0)) {
        k++;
        term /= (double)((2*k)*(2*k+1));
      }
      return k;
    }();
    Real c[K+1]; // c[k] = 1/(2k+1)!
    c[0] = 1;
    for (Integer k = 1; k <= K; k++) c[k] = c[k-1] / (Real)((2*k)*(2*k+1));
    const VData z = mul_intrin(x, x);
    VData p = set1_intrin<VData>(c[K]);
    for (Integer k = K-1; k >= 1; k--) p = fma_intrin(p, z, set1_intrin<VData>(c[k]));
    return fma_intrin(mul_intrin(x, z), p, x);
  }
  template <class VData> inline VData exp_half_intrin(const VData& a) { // e^a / 2 for a >= 0; inf beyond the overflow
    using Real = typename VData::ScalarType;
    return mul_intrin(exp_intrin(sub_intrin(a, set1_intrin<VData>((Real)1))), set1_intrin<VData>(const_e<Real>()/2));
  }
  template <class VData> inline VData cosh_intrin(const VData& x) {
    const VData h = exp_half_intrin(fabs_intrin(x));
    return add_intrin(h, div_intrin(set1_intrin<VData>((typename VData::ScalarType)0.25), h));
  }
  template <class VData> inline VData sinh_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    const VData a = fabs_intrin(x);
    const VData h = exp_half_intrin(a);
    const VData big = copysign_intrin(sub_intrin(h, div_intrin(set1_intrin<VData>((Real)0.25), h)), x);
    return select_intrin(comp_intrin<ComparisonType::lt>(a, set1_intrin<VData>((Real)1)), sinh_series_intrin(x), big);
  }
  template <class VData> inline VData tanh_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    const VData one = set1_intrin<VData>((Real)1);
    const VData a = fabs_intrin(x);
    const VData u = exp_intrin(mul_intrin(a, set1_intrin<VData>((Real)-2)));
    const VData big = copysign_intrin(div_intrin(sub_intrin(one, u), add_intrin(one, u)), x);
    const VData s = sinh_series_intrin(x);
    const VData small = div_intrin(s, sqrt_intrin(fma_intrin(s, s, one))); // sinh / cosh
    return select_intrin(comp_intrin<ComparisonType::lt>(a, one), small, big);
  }

  template <Integer DIGITS, class VData> inline VData approx_sin_intrin(const VData& x) {
    VData sinx, cosx;
    approx_sincos_intrin<DIGITS>(sinx, cosx, x);
    return sinx;
  }
  template <Integer DIGITS, class VData> inline VData approx_cos_intrin(const VData& x) {
    VData sinx, cosx;
    approx_sincos_intrin<DIGITS>(sinx, cosx, x);
    return cosx;
  }
  template <Integer DIGITS, class VData> inline VData approx_tan_intrin(const VData& x) {
    VData sinx, cosx;
    approx_sincos_intrin<DIGITS>(sinx, cosx, x);
    return div_intrin(sinx, cosx);
    //VData cos2_x = mul_intrin(cosx, cosx);
    //VData cos4_x = mul_intrin(cos2_x, cos2_x);
    //VData sec2_x = rsqrt_approx_intrin<digits,VData>::eval(cos4_x);
    //return mul_intrin(sinx, mul_intrin(cosx, sec2_x));
  }
}

namespace sctl { // SSE
#if defined(__SSE4_2__) || defined(__ARM_NEON)
  template <> struct alignas(sizeof(int8_t) * 16) VecData<int8_t,16> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 16;
    VecData() = default;
    inline VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(int16_t) * 8) VecData<int16_t,8> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 8;
    VecData() = default;
    inline VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(int32_t) * 4) VecData<int32_t,4> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 4;
    VecData() = default;
    inline VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(int64_t) * 2) VecData<int64_t,2> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 2;
    VecData() = default;
    inline VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(float) * 4) VecData<float,4> {
    using ScalarType = float;
    static constexpr Integer Size = 4;
    VecData() = default;
    inline VecData(__m128 v_) : v(v_) {}
    __m128 v;
  };
  template <> struct alignas(sizeof(double) * 2) VecData<double,2> {
    using ScalarType = double;
    static constexpr Integer Size = 2;
    VecData() = default;
    inline VecData(__m128d v_) : v(v_) {}
    __m128d v;
  };

  // Select between two sources, byte by byte. Used in various functions and operators
  // Corresponds to this pseudocode:
  // for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
  // Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
  // The implementation depends on the instruction set:
  // If SSE4.1 is supported then only bit 7 in each byte of s is checked,
  // otherwise all bits in s are used.
  static inline __m128i selectb (__m128i const & s, __m128i const & a, __m128i const & b) {
    #if defined(__SSE4_1__) || defined(__ARM_NEON)
    return _mm_blendv_epi8 (b, a, s);
    #else
    return _mm_or_si128(_mm_and_si128(s,a), _mm_andnot_si128(s,b));
    #endif
  }


  template <> inline VecData<int8_t,16> zero_intrin<VecData<int8_t,16>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<int16_t,8> zero_intrin<VecData<int16_t,8>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<int32_t,4> zero_intrin<VecData<int32_t,4>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<int64_t,2> zero_intrin<VecData<int64_t,2>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<float,4> zero_intrin<VecData<float,4>>() {
    return _mm_setzero_ps();
  }
  template <> inline VecData<double,2> zero_intrin<VecData<double,2>>() {
    return _mm_setzero_pd();
  }

#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14 || (__GNUC__ >= 12 && defined(__AVX__)))
  // Two equal 64-bit halves known at compile time, by one load (movddup), for constants that GCC would make with two
  // or three instructions (a load of one element, or a move from a general register, and a shuffle): of 4 floats or
  // equal integers with GCC 14 and later without AVX, and of equal integers with GCC 12 and later with AVX. The empty
  // asm keeps GCC from seeing them as the constants. Not for 0, nor -1 of integers: GCC makes them by one instruction
  // (pxor, pcmpeqd), and knows their value.
  inline __m128i dup64_const_intrin(const uint64_t bits) {
    double d;
    __builtin_memcpy(&d, &bits, sizeof(d));
    __m128d t = _mm_set1_pd(d);
    asm("" : "+x"(t));
    return _mm_castpd_si128(t);
  }
#endif
  template <> inline VecData<int8_t,16> set1_intrin<VecData<int8_t,16>>(int8_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14 || (__GNUC__ >= 12 && defined(__AVX__)))
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64_const_intrin((uint8_t)a * 0x0101010101010101ULL);
#endif
    return _mm_set1_epi8(a);
  }
  template <> inline VecData<int16_t,8> set1_intrin<VecData<int16_t,8>>(int16_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14 || (__GNUC__ >= 12 && defined(__AVX__)))
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64_const_intrin((uint16_t)a * 0x0001000100010001ULL);
#endif
    return _mm_set1_epi16(a);
  }
  template <> inline VecData<int32_t,4> set1_intrin<VecData<int32_t,4>>(int32_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14 || (__GNUC__ >= 12 && defined(__AVX__)))
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64_const_intrin((((uint64_t)(uint32_t)a) << 32) | (uint32_t)a);
#endif
    return _mm_set1_epi32(a);
  }
  template <> inline VecData<int64_t,2> set1_intrin<VecData<int64_t,2>>(int64_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14 || (__GNUC__ >= 12 && defined(__AVX__)))
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64_const_intrin((uint64_t)a);
#endif
    return _mm_set1_epi64x(a);
  }
  template <> inline VecData<float,4> set1_intrin<VecData<float,4>>(float a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) && !defined(__AVX__)
    if (__builtin_constant_p(a)) { // GCC 12 and later make 4 equal floats by a load of one float and a shuffle (movss, shufps)
      uint32_t b;
      __builtin_memcpy(&b, &a, sizeof(b));
#if __GNUC__ >= 14
      if (b != 0) return _mm_castsi128_ps(dup64_const_intrin((((uint64_t)b) << 32) | b));
#else // as 4 equal integers, by one load (movdqa): in exp and log up to 1.1x faster than movddup on Ice Lake and Zen 2, 1.2x on Zen 4
      if (b != 0) {
        __m128i t = _mm_set1_epi32((int32_t)b);
        asm("" : "+x"(t));
        return _mm_castsi128_ps(t);
      }
#endif
    }
#endif
    return _mm_set1_ps(a);
  }
  template <> inline VecData<double,2> set1_intrin<VecData<double,2>>(double a) {
    return _mm_set1_pd(a);
  }

  template <> inline VecData<int8_t,16> set_intrin<VecData<int8_t,16>,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t>(int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16) {
    return _mm_set_epi8(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int16_t,8> set_intrin<VecData<int16_t,8>,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t>(int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8) {
    return _mm_set_epi16(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int32_t,4> set_intrin<VecData<int32_t,4>,int32_t,int32_t,int32_t,int32_t>(int32_t v1, int32_t v2, int32_t v3, int32_t v4) {
    return _mm_set_epi32(v4,v3,v2,v1);
  }
  template <> inline VecData<int64_t,2> set_intrin<VecData<int64_t,2>,int64_t,int64_t>(int64_t v1, int64_t v2) {
    return _mm_set_epi64x(v2,v1);
  }
  template <> inline VecData<float,4> set_intrin<VecData<float,4>,float,float,float,float>(float v1, float v2, float v3, float v4) {
    return _mm_set_ps(v4,v3,v2,v1);
  }
  template <> inline VecData<double,2> set_intrin<VecData<double,2>,double,double>(double v1, double v2) {
    return _mm_set_pd(v2,v1);
  }

  template <> inline VecData<int8_t,16> load1_intrin<VecData<int8_t,16>>(int8_t const* p) {
    return _mm_set1_epi8(p[0]);
  }
  template <> inline VecData<int16_t,8> load1_intrin<VecData<int16_t,8>>(int16_t const* p) {
    return _mm_set1_epi16(p[0]);
  }
  template <> inline VecData<int32_t,4> load1_intrin<VecData<int32_t,4>>(int32_t const* p) {
    return _mm_set1_epi32(p[0]);
  }
  template <> inline VecData<int64_t,2> load1_intrin<VecData<int64_t,2>>(int64_t const* p) {
    return _mm_set1_epi64x(p[0]);
  }
  template <> inline VecData<float,4> load1_intrin<VecData<float,4>>(float const* p) {
    return _mm_load1_ps(p);
  }
  template <> inline VecData<double,2> load1_intrin<VecData<double,2>>(double const* p) {
    return _mm_load1_pd(p);
  }

  template <> inline VecData<int8_t,16> loadu_intrin<VecData<int8_t,16>>(int8_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<int16_t,8> loadu_intrin<VecData<int16_t,8>>(int16_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<int32_t,4> loadu_intrin<VecData<int32_t,4>>(int32_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<int64_t,2> loadu_intrin<VecData<int64_t,2>>(int64_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<float,4> loadu_intrin<VecData<float,4>>(float const* p) {
    return _mm_loadu_ps(p);
  }
  template <> inline VecData<double,2> loadu_intrin<VecData<double,2>>(double const* p) {
    return _mm_loadu_pd(p);
  }

  template <> inline VecData<int8_t,16> load_intrin<VecData<int8_t,16>>(int8_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<int16_t,8> load_intrin<VecData<int16_t,8>>(int16_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<int32_t,4> load_intrin<VecData<int32_t,4>>(int32_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<int64_t,2> load_intrin<VecData<int64_t,2>>(int64_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<float,4> load_intrin<VecData<float,4>>(float const* p) {
    return _mm_load_ps(p);
  }
  template <> inline VecData<double,2> load_intrin<VecData<double,2>>(double const* p) {
    return _mm_load_pd(p);
  }

  template <> inline void storeu_intrin<VecData<int8_t,16>>(int8_t* p, VecData<int8_t,16> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int16_t,8>>(int16_t* p, VecData<int16_t,8> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<float,4>>(float* p, VecData<float,4> vec) {
    _mm_storeu_ps(p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<double,2>>(double* p, VecData<double,2> vec) {
    _mm_storeu_pd(p, vec.v);
  }

  template <> inline void store_intrin<VecData<int8_t,16>>(int8_t* p, VecData<int8_t,16> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int16_t,8>>(int16_t* p, VecData<int16_t,8> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<float,4>>(float* p, VecData<float,4> vec) {
    _mm_store_ps(p, vec.v);
  }
  template <> inline void store_intrin<VecData<double,2>>(double* p, VecData<double,2> vec) {
    _mm_store_pd(p, vec.v);
  }

#if defined(__AVX512VBMI2__)
  template <> inline int8_t extract_intrin<VecData<int8_t,16>>(VecData<int8_t,16> vec, Integer i) {
    __m128i x = _mm_maskz_compress_epi8(__mmask16(1u<<i), vec.v);
    return (int8_t)_mm_cvtsi128_si32(x);
  }
  template <> inline int16_t extract_intrin<VecData<int16_t,8>>(VecData<int16_t,8> vec, Integer i) {
    __m128i x = _mm_maskz_compress_epi16(__mmask8(1u<<i), vec.v);
    return (int16_t)_mm_cvtsi128_si32(x);
  }
  template <> inline int32_t extract_intrin<VecData<int32_t,4>>(VecData<int32_t,4> vec, Integer i) {
    __m128i x = _mm_maskz_compress_epi32(__mmask8(1u<<i), vec.v);
    return (int32_t)_mm_cvtsi128_si32(x);
  }
  //template <> inline int64_t extract_intrin<VecData<int64_t,2>>(VecData<int64_t,2> vec, Integer i) {}
  template <> inline float extract_intrin<VecData<float,4>>(VecData<float,4> vec, Integer i) {
    __m128 x = _mm_maskz_compress_ps(__mmask8(1u<<i), vec.v);
    return _mm_cvtss_f32(x);
  }
  template <> inline double extract_intrin<VecData<double,2>>(VecData<double,2> vec, Integer i) {
    __m128d x = _mm_mask_unpackhi_pd(vec.v, __mmask8(i), vec.v, vec.v);
    return _mm_cvtsd_f64(x);
  }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
  template <> inline void insert_intrin<VecData<int8_t,16>>(VecData<int8_t,16>& vec, Integer i, int8_t value) {
    vec.v = _mm_mask_set1_epi8(vec.v, __mmask16(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<int16_t,8>>(VecData<int16_t,8>& vec, Integer i, int16_t value) {
    vec.v = _mm_mask_set1_epi16(vec.v, __mmask8(1u<<i), value);
  }
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline void insert_intrin<VecData<int32_t,4>>(VecData<int32_t,4>& vec, Integer i, int32_t value) {
    vec.v = _mm_mask_set1_epi32(vec.v, __mmask8(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<int64_t,2>>(VecData<int64_t,2>& vec, Integer i, int64_t value) {
    vec.v = _mm_mask_set1_epi64(vec.v, __mmask8(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<float,4>>(VecData<float,4>& vec, Integer i, float value) {
    vec.v = _mm_mask_broadcastss_ps(vec.v, __mmask8(1u<<i), _mm_set_ss(value));
  }
  template <> inline void insert_intrin<VecData<double,2>>(VecData<double,2>& vec, Integer i, double value) {
    vec.v = _mm_mask_movedup_pd(vec.v, __mmask8(1u<<i), _mm_set_sd(value));
  }
#endif

  // Arithmetic operators
  template <> inline VecData<int8_t,16> unary_minus_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a) {
    return _mm_sub_epi8(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<int16_t,8> unary_minus_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a) {
    return _mm_sub_epi16(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<int32_t,4> unary_minus_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a) {
    return _mm_sub_epi32(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<int64_t,2> unary_minus_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a) {
    return _mm_sub_epi64(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<float,4> unary_minus_intrin<VecData<float,4>>(const VecData<float,4>& a) {
    return _mm_xor_ps(a.v, _mm_castsi128_ps(set1_intrin<VecData<int32_t,4>>(0x80000000).v));
  }
  template <> inline VecData<double,2> unary_minus_intrin<VecData<double,2>>(const VecData<double,2>& a) {
    return _mm_xor_pd(a.v, _mm_castsi128_pd(_mm_setr_epi32(0,0x80000000,0,0x80000000)));
  }

  template <> inline VecData<int8_t,16> mul_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    // There is no 8-bit multiply in SSE2. Split into two 16-bit multiplies
    __m128i aodd    = _mm_srli_epi16(a.v,8);               // odd numbered elements of a
    __m128i bodd    = _mm_srli_epi16(b.v,8);               // odd numbered elements of b
    __m128i muleven = _mm_mullo_epi16(a.v,b.v);            // product of even numbered elements
    __m128i mulodd  = _mm_mullo_epi16(aodd,bodd);          // product of odd  numbered elements
            mulodd  = _mm_slli_epi16(mulodd,8);            // put odd numbered elements back in place
    #if defined(__AVX512VL__) && defined(__AVX512BW__)
    return _mm_mask_mov_epi8(mulodd, 0x5555, muleven);
    #else
    __m128i mask    = set1_intrin<VecData<int32_t,4>>(0x00FF00FF).v; // mask for even positions
    return selectb(mask,muleven,mulodd);                   // interleave even and odd
    #endif
  }
  template <> inline VecData<int16_t,8> mul_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_mullo_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> mul_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    #if defined(__SSE4_1__) || defined(__ARM_NEON)
    return _mm_mullo_epi32(a.v, b.v);
    #else
    __m128i a13    = _mm_shuffle_epi32(a.v, 0xF5);        // (-,a3,-,a1)
    __m128i b13    = _mm_shuffle_epi32(b.v, 0xF5);        // (-,b3,-,b1)
    __m128i prod02 = _mm_mul_epu32(a.v, b.v);             // (-,a2*b2,-,a0*b0)
    __m128i prod13 = _mm_mul_epu32(a13, b13);             // (-,a3*b3,-,a1*b1)
    __m128i prod01 = _mm_unpacklo_epi32(prod02,prod13);   // (-,-,a1*b1,a0*b0)
    __m128i prod23 = _mm_unpackhi_epi32(prod02,prod13);   // (-,-,a3*b3,a2*b2)
    return           _mm_unpacklo_epi64(prod01,prod23);   // (ab3,ab2,ab1,ab0)
    #endif
  }
  template <> inline VecData<int64_t,2> mul_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    #if defined(__AVX512DQ__) && defined(__AVX512VL__)
    return _mm_mullo_epi64(a.v, b.v);
    #elif defined(__SSE4_1__) || defined(__ARM_NEON)
    // Split into 32-bit multiplies
    __m128i bswap   = _mm_shuffle_epi32(b.v,0xB1);         // b0H,b0L,b1H,b1L (swap H<->L)
    __m128i prodlh  = _mm_mullo_epi32(a.v,bswap);          // a0Lb0H,a0Hb0L,a1Lb1H,a1Hb1L, 32 bit L*H products
    __m128i zero    = _mm_setzero_si128();                 // 0
    __m128i prodlh2 = _mm_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m128i prodlh3 = _mm_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m128i prodll  = _mm_mul_epu32(a.v,b.v);              // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m128i prod    = _mm_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
    #else               // SSE2
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] *= b_.x[i];
    return a_.v;
    #endif
  }
  template <> inline VecData<float,4> mul_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_mul_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> mul_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_mul_pd(a.v, b.v);
  }

  template <> inline VecData<int16_t,8> div_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { // through float, exact for 16 bits; the low 16 bits of the quotient, as the C++ conversion
    const auto q = [](const __m128i a4, const __m128i b4) { return _mm_and_si128(_mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a4)), _mm_cvtepi32_ps(_mm_cvtepi16_epi32(b4)))), set1_intrin<VecData<int32_t,4>>(0xFFFF).v); };
    return _mm_packus_epi32(q(a.v, b.v), q(_mm_srli_si128(a.v, 8), _mm_srli_si128(b.v, 8)));
  }
  template <> inline VecData<int8_t,16> div_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { // through float, four lanes at a time; the low 8 bits of the quotient
    const auto q = [](const __m128i a4, const __m128i b4) { return _mm_and_si128(_mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(a4)), _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b4)))), set1_intrin<VecData<int32_t,4>>(0xFF).v); };
    const __m128i q01 = _mm_packus_epi32(q(a.v, b.v), q(_mm_srli_si128(a.v, 4), _mm_srli_si128(b.v, 4)));
    const __m128i q23 = _mm_packus_epi32(q(_mm_srli_si128(a.v, 8), _mm_srli_si128(b.v, 8)), q(_mm_srli_si128(a.v, 12), _mm_srli_si128(b.v, 12)));
    return _mm_packus_epi16(q01, q23);
  }
  template <> inline VecData<int32_t,4> div_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { // through double, exact: the rounding error of the quotient is below 2^-22/|b|, and a quotient that is not an integer is at least 1/|b| from one
#if defined(__AVX__)
    return _mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(a.v), _mm256_cvtepi32_pd(b.v)));
#else // two lanes at a time
    const __m128i lo = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(a.v), _mm_cvtepi32_pd(b.v)));
    const __m128i hi = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(_mm_unpackhi_epi64(a.v, a.v)), _mm_cvtepi32_pd(_mm_unpackhi_epi64(b.v, b.v))));
    return _mm_unpacklo_epi64(lo, hi);
#endif
  }
  template <> inline VecData<int64_t,2> div_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return div_int64_intrin(a, b); }
  template <> inline VecData<float,4> div_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_div_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> div_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_div_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> add_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_add_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> add_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_add_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> add_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_add_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> add_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_add_epi64(a.v, b.v);
  }
  template <> inline VecData<float,4> add_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_add_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> add_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_add_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> sub_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_sub_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> sub_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_sub_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> sub_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_sub_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> sub_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_sub_epi64(a.v, b.v);
  }
  template <> inline VecData<float,4> sub_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_sub_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> sub_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_sub_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,16> fma_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b, const VecData<int8_t,16>& c) {}
  //template <> inline VecData<int16_t,8> fma_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b, const VecData<int16_t,8>& c) {}
  //template <> inline VecData<int32_t,4> sub_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b, const VecData<int32_t,4>& c) {}
  //template <> inline VecData<int64_t,2> sub_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b, const VecData<int64_t,2>& c) {}
  template <> inline VecData<float,4> fma_intrin(const VecData<float,4>& a, const VecData<float,4>& b, const VecData<float,4>& c) {
    #ifdef __FMA__
    return _mm_fmadd_ps(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm_macc_ps(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }
  template <> inline VecData<double,2> fma_intrin(const VecData<double,2>& a, const VecData<double,2>& b, const VecData<double,2>& c) {
    #ifdef __FMA__
    return _mm_fmadd_pd(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm_macc_pd(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }

  // Bitwise operators
  template <> inline VecData<int8_t,16> not_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<int16_t,8> not_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<int32_t,4> not_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<int64_t,2> not_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<float,4> not_intrin<VecData<float,4>>(const VecData<float,4>& a) {
    return _mm_xor_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(-1)));
  }
  template <> inline VecData<double,2> not_intrin<VecData<double,2>>(const VecData<double,2>& a) {
    return _mm_xor_pd(a.v, _mm_castsi128_pd(_mm_set1_epi32(-1)));
  }

  template <> inline VecData<int8_t,16> and_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> and_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> and_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> and_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<float,4> and_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_and_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> and_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_and_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> xor_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> xor_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> xor_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> xor_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<float,4> xor_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_xor_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> xor_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_xor_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> or_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> or_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> or_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> or_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<float,4> or_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_or_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> or_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_or_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> andnot_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<int16_t,8> andnot_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<int32_t,4> andnot_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<int64_t,2> andnot_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<float,4> andnot_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_andnot_ps(b.v, a.v);
  }
  template <> inline VecData<double,2> andnot_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_andnot_pd(b.v, a.v);
  }

  // Bitshift
  template <> inline VecData<int8_t,16> bitshiftleft_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a, const Integer& rhs) { // 16-bit bit shift, then the bits from the next byte cleared
    return _mm_and_si128(_mm_slli_epi16(a.v, (int)rhs), _mm_set1_epi8((char)(rhs < 8 ? (0xFF << rhs) & 0xFF : 0)));
  }
  template <> inline VecData<int16_t,8> bitshiftleft_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a, const Integer& rhs) { return _mm_slli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,4> bitshiftleft_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a, const Integer& rhs) { return _mm_slli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t,2> bitshiftleft_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const Integer& rhs) { return _mm_slli_epi64(a.v , rhs); }
  template <> inline VecData<float  ,4> bitshiftleft_intrin<VecData<float  ,4>>(const VecData<float  ,4>& a, const Integer& rhs) { return _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(a.v), rhs)); }
  template <> inline VecData<double ,2> bitshiftleft_intrin<VecData<double ,2>>(const VecData<double ,2>& a, const Integer& rhs) { return _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(a.v), rhs)); }

  template <> inline VecData<int8_t,16> bitshiftright_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a, const Integer& rhs) { // logical 16-bit bit shift, the bits from the next byte cleared, then the sign extended: (u ^ m) - m
    const int n = (int)(rhs < 7 ? rhs : 7); // larger bit shifts also give 0 or -1
    const __m128i u = _mm_and_si128(_mm_srli_epi16(a.v, n), _mm_set1_epi8((char)(0xFF >> n)));
    const __m128i m = _mm_set1_epi8((char)(0x80 >> n));
    return _mm_sub_epi8(_mm_xor_si128(u, m), m);
  }
  template <> inline VecData<int16_t,8> bitshiftright_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a, const Integer& rhs) { return _mm_srai_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,4> bitshiftright_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a, const Integer& rhs) { return _mm_srai_epi32(a.v , rhs); }
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> bitshiftright_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const Integer& rhs) { return _mm_srai_epi64(a.v , rhs); }
  #else
  template <> inline VecData<int64_t,2> bitshiftright_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const Integer& rhs) { // negative lanes: complement, logical bit shift, complement back
    const __m128i s = _mm_cmpgt_epi64(_mm_setzero_si128(), a.v);
    return _mm_xor_si128(_mm_srli_epi64(_mm_xor_si128(a.v, s), rhs), s);
  }
  #endif
  template <> inline VecData<float  ,4> bitshiftright_intrin<VecData<float  ,4>>(const VecData<float  ,4>& a, const Integer& rhs) { return _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(a.v), rhs)); }
  template <> inline VecData<double ,2> bitshiftright_intrin<VecData<double ,2>>(const VecData<double ,2>& a, const Integer& rhs) { return _mm_castsi128_pd(_mm_srli_epi64(_mm_castpd_si128(a.v), rhs)); }

  #if defined(__AVX2__)
  template <> inline VecData<int32_t,4> bitshiftleft_intrin <VecData<int32_t,4>>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& rhs) { return _mm_sllv_epi32(a.v, rhs.v); }
  template <> inline VecData<int64_t,2> bitshiftleft_intrin <VecData<int64_t,2>>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& rhs) { return _mm_sllv_epi64(a.v, rhs.v); }
  template <> inline VecData<int32_t,4> bitshiftright_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& rhs) { return _mm_srav_epi32(a.v, rhs.v); }
  #else // without AVX2: a multiply, or one bit shift for the count of each lane and blends
  template <> inline VecData<int32_t,4> bitshiftleft_intrin <VecData<int32_t,4>>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& rhs) { // a times 2^rhs, made from the exponent bits of a float; 2^31 becomes 0x80000000
    return _mm_mullo_epi32(a.v, _mm_cvttps_epi32(_mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(rhs.v, 23), set1_intrin<VecData<int32_t,4>>(127 << 23).v))));
  }
  template <> inline VecData<int64_t,2> bitshiftleft_intrin <VecData<int64_t,2>>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& rhs) { return _mm_blend_epi16(_mm_sll_epi64(a.v, rhs.v), _mm_sll_epi64(a.v, _mm_unpackhi_epi64(rhs.v, rhs.v)), 0xF0); }
  template <> inline VecData<int32_t,4> bitshiftright_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& rhs) { // the count of lane i in the low 64 bits of the count of _mm_sra_epi32
    const __m128i z = _mm_setzero_si128();
    const __m128i r0 = _mm_sra_epi32(a.v, _mm_blend_epi16(rhs.v, z, 0xFC));
    const __m128i r1 = _mm_sra_epi32(a.v, _mm_srli_epi64(rhs.v, 32));
    const __m128i r2 = _mm_sra_epi32(a.v, _mm_unpackhi_epi32(rhs.v, z));
    const __m128i r3 = _mm_sra_epi32(a.v, _mm_srli_si128(rhs.v, 12));
    return _mm_blend_epi16(_mm_blend_epi16(r0, r1, 0x0C), _mm_blend_epi16(r2, r3, 0xC0), 0xF0);
  }
  #endif
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> bitshiftright_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& rhs) { return _mm_srav_epi64(a.v, rhs.v); }
  #elif defined(__AVX2__) || !defined(__AVX__) // negative lanes: complement, logical bit shift, complement back; with AVX and without AVX2 the generic code is faster
  template <> inline VecData<int64_t,2> bitshiftright_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& rhs) {
    const __m128i s = _mm_cmpgt_epi64(_mm_setzero_si128(), a.v);
    const __m128i t = _mm_xor_si128(a.v, s);
    #if defined(__AVX2__)
    return _mm_xor_si128(_mm_srlv_epi64(t, rhs.v), s);
    #else
    return _mm_xor_si128(_mm_blend_epi16(_mm_srl_epi64(t, rhs.v), _mm_srl_epi64(t, _mm_unpackhi_epi64(rhs.v, rhs.v)), 0xF0), s);
    #endif
  }
  #endif
  #if defined(__AVX512BW__) && defined(__AVX512VL__)
  template <> inline VecData<int16_t,8> bitshiftleft_intrin <VecData<int16_t,8>>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& rhs) { return _mm_sllv_epi16(a.v, rhs.v); }
  template <> inline VecData<int16_t,8> bitshiftright_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& rhs) { return _mm_srav_epi16(a.v, rhs.v); }
  #else
  template <> inline VecData<int16_t,8> bitshiftleft_intrin <VecData<int16_t,8>>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& rhs) { // a times 2^rhs, made from the exponent bits of floats and packed with unsigned saturation (2^15 stays 0x8000)
    const __m128i z = _mm_setzero_si128();
    const auto pow2 = [](const __m128i k) { return _mm_cvttps_epi32(_mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(k, 23), set1_intrin<VecData<int32_t,4>>(127 << 23).v))); };
    return _mm_mullo_epi16(a.v, _mm_packus_epi32(pow2(_mm_unpacklo_epi16(rhs.v, z)), pow2(_mm_unpackhi_epi16(rhs.v, z))));
  }
  #if defined(__AVX2__) // without AVX2, the generic code is faster
  template <> inline VecData<int16_t,8> bitshiftright_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& rhs) { // the int32 bit shift of each half, sign extended
    const __m128i z = _mm_setzero_si128();
    const __m128i lo = _mm_srav_epi32(_mm_srai_epi32(_mm_unpacklo_epi16(a.v, a.v), 16), _mm_unpacklo_epi16(rhs.v, z));
    const __m128i hi = _mm_srav_epi32(_mm_srai_epi32(_mm_unpackhi_epi16(a.v, a.v), 16), _mm_unpackhi_epi16(rhs.v, z));
    return _mm_packs_epi32(lo, hi);
  }
  #endif
  #endif

  // Other functions
  template <> inline VecData<int8_t,16> max_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_max_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> max_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_max_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> max_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_max_epi32(a.v, b.v);
  }
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> max_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_max_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,4> max_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_max_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> max_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_max_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> min_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_min_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> min_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_min_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> min_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_min_epi32(a.v, b.v);
  }
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> min_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_min_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,4> min_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_min_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> min_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_min_pd(a.v, b.v);
  }

  // Interleave two 128-bit vectors at a given element granularity. Used to build
  // the transpose network for element widths that have no float counterpart.
  template <Integer Bits> struct UnpackIntrin128;
  template <> struct UnpackIntrin128< 8> {
    static inline __m128i lo(const __m128i& a, const __m128i& b) { return _mm_unpacklo_epi8 (a, b); }
    static inline __m128i hi(const __m128i& a, const __m128i& b) { return _mm_unpackhi_epi8 (a, b); }
  };
  template <> struct UnpackIntrin128<16> {
    static inline __m128i lo(const __m128i& a, const __m128i& b) { return _mm_unpacklo_epi16(a, b); }
    static inline __m128i hi(const __m128i& a, const __m128i& b) { return _mm_unpackhi_epi16(a, b); }
  };
  template <> struct UnpackIntrin128<32> {
    static inline __m128i lo(const __m128i& a, const __m128i& b) { return _mm_unpacklo_epi32(a, b); }
    static inline __m128i hi(const __m128i& a, const __m128i& b) { return _mm_unpackhi_epi32(a, b); }
  };
  template <> struct UnpackIntrin128<64> {
    static inline __m128i lo(const __m128i& a, const __m128i& b) { return _mm_unpacklo_epi64(a, b); }
    static inline __m128i hi(const __m128i& a, const __m128i& b) { return _mm_unpackhi_epi64(a, b); }
  };

  // One stage of the NxN transpose network: interleave at granularity Bits,
  // pairing vectors Stride apart. Recurses with both doubled until Stride == N.
  // A 128-bit vector is a single lane, so no cross-lane stage is needed.
  template <Integer N, Integer Bits, Integer Stride> struct TransposeNet128 {
    static inline void apply(__m128i (&v)[N]) {
      __m128i w[N];
      for (Integer k = 0; k < N; k += 2*Stride) {
        for (Integer j = 0; j < Stride; j++) {
          w[k+2*j+0] = UnpackIntrin128<Bits>::lo(v[k+j], v[k+j+Stride]);
          w[k+2*j+1] = UnpackIntrin128<Bits>::hi(v[k+j], v[k+j+Stride]);
        }
      }
      for (Integer i = 0; i < N; i++) v[i] = w[i];
      TransposeNet128<N,Bits*2,Stride*2>::apply(v);
    }
  };
  template <Integer N, Integer Bits> struct TransposeNet128<N,Bits,N> {
    static inline void apply(__m128i (&)[N]) {}
  };

  template <> inline void transpose_intrin<VecData<int8_t,16>>(VecData<int8_t,16> (&v)[16]) {
    __m128i w[16];
    for (Integer i = 0; i < 16; i++) w[i] = v[i].v;
    TransposeNet128<16,8,1>::apply(w);
    for (Integer i = 0; i < 16; i++) v[i].v = w[i];
  }
  template <> inline void transpose_intrin<VecData<int16_t,8>>(VecData<int16_t,8> (&v)[8]) {
    __m128i w[8];
    for (Integer i = 0; i < 8; i++) w[i] = v[i].v;
    TransposeNet128<8,16,1>::apply(w);
    for (Integer i = 0; i < 8; i++) v[i].v = w[i];
  }
  template <> inline void transpose_intrin<VecData<double,2>>(VecData<double,2> (&v)[2]) {
    const __m128d r0 = v[0].v, r1 = v[1].v;
    v[0].v = _mm_unpacklo_pd(r0, r1);
    v[1].v = _mm_unpackhi_pd(r0, r1);
  }
  template <> inline void transpose_intrin<VecData<float,4>>(VecData<float,4> (&v)[4]) {
    const __m128 r0 = v[0].v, r1 = v[1].v, r2 = v[2].v, r3 = v[3].v;

    const __m128 a0 = _mm_unpacklo_ps(r0, r1);
    const __m128 a1 = _mm_unpackhi_ps(r0, r1);
    const __m128 b0 = _mm_unpacklo_ps(r2, r3);
    const __m128 b1 = _mm_unpackhi_ps(r2, r3);

    v[0].v = _mm_movelh_ps(a0, b0);
    v[1].v = _mm_movehl_ps(b0, a0);
    v[2].v = _mm_movelh_ps(a1, b1);
    v[3].v = _mm_movehl_ps(b1, a1);
  }
  template <> inline void transpose_intrin<VecData<int32_t,4>>(VecData<int32_t,4> (&v)[4]) { transpose_reinterpret_intrin<VecData<float ,4>>(v); }
  template <> inline void transpose_intrin<VecData<int64_t,2>>(VecData<int64_t,2> (&v)[2]) { transpose_reinterpret_intrin<VecData<double,2>>(v); }

  template <> inline VecData<float,4> swap_pairs_intrin(const VecData<float,4>& vec) { return _mm_shuffle_ps(vec.v, vec.v, 0xB1); }
  template <> inline VecData<double,2> swap_pairs_intrin(const VecData<double,2>& vec) { return _mm_shuffle_pd(vec.v, vec.v, 0x1); }

  // The rows transposed and evaluated by Estrin's scheme together: one step of the scheme after each stage of the
  // transpose, q_k = c_2k + c_2k+1 t after the first, s_k = q_2k + q_2k+1 t^2 after the second.
  template <> inline VecData<double,2> eval_poly_rows_intrin<2, VecData<double,2>>(const double* const (&p)[2], const VecData<double,2>& t) {
    using V = VecData<double,2>;
    const __m128d r0 = _mm_loadu_pd(p[0]);
    const __m128d r1 = _mm_loadu_pd(p[1]);
    return fma_intrin(V(_mm_unpackhi_pd(r0, r1)), t, V(_mm_unpacklo_pd(r0, r1)));
  }
  template <> inline VecData<float,4> eval_poly_rows_intrin<4, VecData<float,4>>(const float* const (&p)[4], const VecData<float,4>& t) {
    using V = VecData<float,4>;
    const __m128 tp[2] = {_mm_unpacklo_ps(t.v, t.v), _mm_unpackhi_ps(t.v, t.v)}; // (t_a, t_a, t_b, t_b) of lanes a, b = 2j, 2j+1
    V q[2]; // (q_0, q_1 of lane 2j, q_0, q_1 of lane 2j+1)
    for (Integer j = 0; j < 2; j++) {
      const __m128 ra = _mm_loadu_ps(p[2 * j]);
      const __m128 rb = _mm_loadu_ps(p[2 * j + 1]);
      q[j] = fma_intrin(V(_mm_shuffle_ps(ra, rb, _MM_SHUFFLE(3,1,3,1))), V(tp[j]), V(_mm_shuffle_ps(ra, rb, _MM_SHUFFLE(2,0,2,0))));
    }
    return fma_intrin(V(_mm_shuffle_ps(q[0].v, q[1].v, _MM_SHUFFLE(3,1,3,1))), mul_intrin(t, t), V(_mm_shuffle_ps(q[0].v, q[1].v, _MM_SHUFFLE(2,0,2,0))));
  }

  // Conversion operators
  template <> inline VecData<float ,4> convert_int2real_intrin<VecData<float ,4>,VecData<int32_t,4>>(const VecData<int32_t,4>& x) {
    return _mm_cvtepi32_ps(x.v);
  }
  #if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<double,2> convert_int2real_intrin<VecData<double,2>,VecData<int64_t,2>>(const VecData<int64_t,2>& x) {
    return _mm_cvtepi64_pd(x.v);
  }
  #endif

  template <> inline VecData<int32_t,4> lrint_intrin<VecData<int32_t,4>,VecData<float ,4>>(const VecData<float ,4>& x) {
    return _mm_cvtps_epi32(x.v);
  }
  #if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> lrint_intrin<VecData<int64_t,2>,VecData<double,2>>(const VecData<double,2>& x) {
    return _mm_cvtpd_epi64(x.v);
  }
  #endif

  template <> inline VecData<float ,4> rint_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_round_ps(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }
  template <> inline VecData<double,2> rint_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_round_pd(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }

  template <> inline VecData<float  ,4> convert_intrin<VecData<float  ,4>,VecData<int32_t,4>>(const VecData<int32_t,4>& a) { return _mm_cvtepi32_ps (a.v); }
  template <> inline VecData<int32_t,4> convert_intrin<VecData<int32_t,4>,VecData<float  ,4>>(const VecData<float  ,4>& a) { return _mm_cvttps_epi32(a.v); }
  #if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<double ,2> convert_intrin<VecData<double ,2>,VecData<int64_t,2>>(const VecData<int64_t,2>& a) { return _mm_cvtepi64_pd (a.v); }
  template <> inline VecData<int64_t,2> convert_intrin<VecData<int64_t,2>,VecData<double ,2>>(const VecData<double ,2>& a) { return _mm_cvttpd_epi64(a.v); }
  #elif !defined(__AVX__) && !defined(__clang__) // with AVX, and with clang, the generic code is faster
  template <> inline VecData<int64_t,2> convert_intrin<VecData<int64_t,2>,VecData<double ,2>>(const VecData<double ,2>& a) { // trunc(a) = hi 2^32 + lo with lo in [0, 2^32), all exact: hi by cvttpd2dq, lo as the low bits of 2^52 + lo
    const __m128d t = _mm_round_pd(a.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m128d hi = _mm_floor_pd(_mm_mul_pd(t, _mm_set1_pd(0x1p-32)));
    const __m128d lo = _mm_add_pd(_mm_sub_pd(t, _mm_mul_pd(hi, _mm_set1_pd(0x1p32))), _mm_set1_pd(0x1p52));
    return _mm_blend_epi16(_mm_castpd_si128(lo), _mm_shuffle_epi32(_mm_cvttpd_epi32(hi), 0x50), 0xCC);
  }
  #endif


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return Mask<VecData<int8_t,16>>(_mm_cmplt_epi8(a.v,b.v)); }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::le>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return Mask<VecData<int8_t,16>>(_mm_cmpgt_epi8(a.v,b.v)); }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return Mask<VecData<int8_t,16>>(_mm_cmpeq_epi8(a.v,b.v)); }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return Mask<VecData<int16_t,8>>(_mm_cmplt_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::le>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return Mask<VecData<int16_t,8>>(_mm_cmpgt_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return Mask<VecData<int16_t,8>>(_mm_cmpeq_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return Mask<VecData<int32_t,4>>(_mm_cmplt_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::le>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return Mask<VecData<int32_t,4>>(_mm_cmpgt_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return Mask<VecData<int32_t,4>>(_mm_cmpeq_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return Mask<VecData<int64_t,2>>(_mm_cmpgt_epi64(b.v,a.v));}
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::le>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return Mask<VecData<int64_t,2>>(_mm_cmpgt_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return Mask<VecData<int64_t,2>>(_mm_cmpeq_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::lt>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmplt_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::le>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmple_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::gt>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpgt_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::ge>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpge_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::eq>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpeq_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::ne>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpneq_ps(a.v,b.v));}

  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::lt>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmplt_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::le>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmple_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::gt>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpgt_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::ge>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpge_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::eq>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpeq_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::ne>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpneq_pd(a.v,b.v));}

  template <> inline VecData<int8_t,16> select_intrin(const Mask<VecData<int8_t,16>>& s, const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int16_t,8> select_intrin(const Mask<VecData<int16_t,8>>& s, const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int32_t,4> select_intrin(const Mask<VecData<int32_t,4>>& s, const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int64_t,2> select_intrin(const Mask<VecData<int64_t,2>>& s, const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<float  ,4> select_intrin(const Mask<VecData<float  ,4>>& s, const VecData<float  ,4>& a, const VecData<float  ,4>& b) { return _mm_blendv_ps  (b.v, a.v, s.v); }
  template <> inline VecData<double ,2> select_intrin(const Mask<VecData<double ,2>>& s, const VecData<double ,2>& a, const VecData<double ,2>& b) { return _mm_blendv_pd  (b.v, a.v, s.v); }

  // Masked load and store
#if defined(__AVX512DQ__) && defined(__AVX512VL__) // with AVX-512VL by a mask register from the sign bits of the lanes: vmaskmovps and vmaskmovpd take about 6 cycles on AMD Zen 4
  template <> inline VecData<float ,4> loadu_mask_intrin<VecData<float ,4>>(float  const* p, const Mask<VecData<float ,4>>& m) { return _mm_maskz_loadu_ps(_mm_movepi32_mask(_mm_castps_si128(m.v)), p); }
  template <> inline VecData<double,2> loadu_mask_intrin<VecData<double,2>>(double const* p, const Mask<VecData<double,2>>& m) { return _mm_maskz_loadu_pd(_mm_movepi64_mask(_mm_castpd_si128(m.v)), p); }
  template <> inline void storeu_mask_intrin<VecData<float ,4>>(float * p, VecData<float ,4> vec, const Mask<VecData<float ,4>>& m) { _mm_mask_storeu_ps(p, _mm_movepi32_mask(_mm_castps_si128(m.v)), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<double,2>>(double* p, VecData<double,2> vec, const Mask<VecData<double,2>>& m) { _mm_mask_storeu_pd(p, _mm_movepi64_mask(_mm_castpd_si128(m.v)), vec.v); }
#elif defined(__AVX__)
  template <> inline VecData<float ,4> loadu_mask_intrin<VecData<float ,4>>(float  const* p, const Mask<VecData<float ,4>>& m) { return _mm_maskload_ps(p, _mm_castps_si128(m.v)); }
  template <> inline VecData<double,2> loadu_mask_intrin<VecData<double,2>>(double const* p, const Mask<VecData<double,2>>& m) { return _mm_maskload_pd(p, _mm_castpd_si128(m.v)); }
  template <> inline void storeu_mask_intrin<VecData<float ,4>>(float * p, VecData<float ,4> vec, const Mask<VecData<float ,4>>& m) { _mm_maskstore_ps(p, _mm_castps_si128(m.v), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<double,2>>(double* p, VecData<double,2> vec, const Mask<VecData<double,2>>& m) { _mm_maskstore_pd(p, _mm_castpd_si128(m.v), vec.v); }
#endif
  // The first n lanes by their count: in the leftover columns of SmallGEMM, the generic loop over the lanes of a mask
  // was 5-8x slower, and tests of each bit of a movemask 1.4-2x. n = 3 of 4 lanes with AVX-512VL by a mask register;
  // with AVX, except on AMD Zen, by the masked load and store (1.4-1.7 cycles with an add on Intel, against 2.1 for two
  // loads and two stores). With clang and AVX-512VL, except on AMD Zen, n = 1 by a mask register: from the load and
  // store of one element, in a loop over rows, clang makes its own vector loop with scatter stores (add to 1 of 4 floats
  // on Sapphire Rapids: 0.57 cycles, against 1.56). On AMD Zen 4, SmallGEMM with one column was 12-31% slower with the
  // mask register.
  template <> inline VecData<float,4> loadu_first_intrin<VecData<float,4>>(float const* p, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) return _mm_maskz_loadu_ps(__mmask8(1), p);
#endif
    if (n >= 4) return _mm_loadu_ps(p);
#if defined(__AVX512F__) && defined(__AVX512VL__)
    if (n == 3) return _mm_maskz_loadu_ps(__mmask8(0x7), p);
#elif defined(__AVX__) && !defined(SCTL_TUNE_ZEN)
    if (n == 3) return _mm_maskload_ps(p, _mm_setr_epi32(-1, -1, -1, 0));
#else
    if (n == 3) return _mm_movelh_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64 const*)p), _mm_load_ss(p + 2));
#endif
    if (n == 2) return _mm_loadl_pi(_mm_setzero_ps(), (__m64 const*)p);
    if (n == 1) return _mm_load_ss(p);
    return _mm_setzero_ps();
  }
  template <> inline void storeu_first_intrin<VecData<float,4>>(float* p, VecData<float,4> vec, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) {
      _mm_mask_storeu_ps(p, __mmask8(1), vec.v);
      return;
    }
#endif
    if (n >= 4) {
      _mm_storeu_ps(p, vec.v);
    } else if (n == 3) {
#if defined(__AVX512F__) && defined(__AVX512VL__)
      _mm_mask_storeu_ps(p, __mmask8(0x7), vec.v);
#elif defined(__AVX__) && !defined(SCTL_TUNE_ZEN)
      _mm_maskstore_ps(p, _mm_setr_epi32(-1, -1, -1, 0), vec.v);
#else
      _mm_storel_pi((__m64*)p, vec.v);
      _mm_store_ss(p + 2, _mm_movehl_ps(vec.v, vec.v));
#endif
    } else if (n == 2) {
      _mm_storel_pi((__m64*)p, vec.v);
    } else if (n == 1) {
      _mm_store_ss(p, vec.v);
    }
  }
  template <> inline VecData<double,2> loadu_first_intrin<VecData<double,2>>(double const* p, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) return _mm_maskz_loadu_pd(__mmask8(1), p);
#endif
    if (n >= 2) return _mm_loadu_pd(p);
    if (n == 1) return _mm_load_sd(p);
    return _mm_setzero_pd();
  }
  template <> inline void storeu_first_intrin<VecData<double,2>>(double* p, VecData<double,2> vec, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) {
      _mm_mask_storeu_pd(p, __mmask8(1), vec.v);
      return;
    }
#endif
    if (n >= 2) {
      _mm_storeu_pd(p, vec.v);
    } else if (n == 1) {
      _mm_store_sd(p, vec.v);
    }
  }
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<int32_t,4> loadu_mask_intrin<VecData<int32_t,4>>(int32_t const* p, const Mask<VecData<int32_t,4>>& m) { return _mm_maskz_loadu_epi32(_mm_movepi32_mask(m.v), p); }
  template <> inline VecData<int64_t,2> loadu_mask_intrin<VecData<int64_t,2>>(int64_t const* p, const Mask<VecData<int64_t,2>>& m) { return _mm_maskz_loadu_epi64(_mm_movepi64_mask(m.v), p); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec, const Mask<VecData<int32_t,4>>& m) { _mm_mask_storeu_epi32(p, _mm_movepi32_mask(m.v), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec, const Mask<VecData<int64_t,2>>& m) { _mm_mask_storeu_epi64(p, _mm_movepi64_mask(m.v), vec.v); }
#elif defined(__AVX2__)
  template <> inline VecData<int32_t,4> loadu_mask_intrin<VecData<int32_t,4>>(int32_t const* p, const Mask<VecData<int32_t,4>>& m) { return _mm_maskload_epi32((int const*)p, m.v); }
  template <> inline VecData<int64_t,2> loadu_mask_intrin<VecData<int64_t,2>>(int64_t const* p, const Mask<VecData<int64_t,2>>& m) { return _mm_maskload_epi64((long long const*)p, m.v); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec, const Mask<VecData<int32_t,4>>& m) { _mm_maskstore_epi32((int*)p, m.v, vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec, const Mask<VecData<int64_t,2>>& m) { _mm_maskstore_epi64((long long*)p, m.v, vec.v); }
#elif defined(__AVX__) // the masked load and store of float and double, on the bits of the integers
  template <> inline VecData<int32_t,4> loadu_mask_intrin<VecData<int32_t,4>>(int32_t const* p, const Mask<VecData<int32_t,4>>& m) { return _mm_castps_si128(_mm_maskload_ps((float const*)p, m.v)); }
  template <> inline VecData<int64_t,2> loadu_mask_intrin<VecData<int64_t,2>>(int64_t const* p, const Mask<VecData<int64_t,2>>& m) { return _mm_castpd_si128(_mm_maskload_pd((double const*)p, m.v)); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec, const Mask<VecData<int32_t,4>>& m) { _mm_maskstore_ps((float*)p, m.v, _mm_castsi128_ps(vec.v)); }
  template <> inline void storeu_mask_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec, const Mask<VecData<int64_t,2>>& m) { _mm_maskstore_pd((double*)p, m.v, _mm_castsi128_pd(vec.v)); }
#endif
  // The first n lanes by their count, as for float and double
  template <> inline VecData<int32_t,4> loadu_first_intrin<VecData<int32_t,4>>(int32_t const* p, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) return _mm_maskz_loadu_epi32(__mmask8(1), p);
#endif
    if (n >= 4) return _mm_loadu_si128((__m128i const*)p);
#if defined(__AVX512F__) && defined(__AVX512VL__)
    if (n == 3) return _mm_maskz_loadu_epi32(__mmask8(0x7), p);
#elif defined(__AVX__) && !defined(SCTL_TUNE_ZEN)
    if (n == 3) return _mm_castps_si128(_mm_maskload_ps((float const*)p, _mm_setr_epi32(-1, -1, -1, 0)));
#else
    if (n == 3) return _mm_insert_epi32(_mm_loadl_epi64((__m128i const*)p), p[2], 2);
#endif
    if (n == 2) return _mm_loadl_epi64((__m128i const*)p);
    if (n == 1) return _mm_cvtsi32_si128(p[0]);
    return _mm_setzero_si128();
  }
  template <> inline void storeu_first_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) {
      _mm_mask_storeu_epi32(p, __mmask8(1), vec.v);
      return;
    }
#endif
    if (n >= 4) {
      _mm_storeu_si128((__m128i*)p, vec.v);
    } else if (n == 3) {
#if defined(__AVX512F__) && defined(__AVX512VL__)
      _mm_mask_storeu_epi32(p, __mmask8(0x7), vec.v);
#elif defined(__AVX__) && !defined(SCTL_TUNE_ZEN)
      _mm_maskstore_ps((float*)p, _mm_setr_epi32(-1, -1, -1, 0), _mm_castsi128_ps(vec.v));
#else
      _mm_storel_epi64((__m128i*)p, vec.v);
      p[2] = _mm_extract_epi32(vec.v, 2);
#endif
    } else if (n == 2) {
      _mm_storel_epi64((__m128i*)p, vec.v);
    } else if (n == 1) {
      p[0] = _mm_cvtsi128_si32(vec.v);
    }
  }
  template <> inline VecData<int64_t,2> loadu_first_intrin<VecData<int64_t,2>>(int64_t const* p, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) return _mm_maskz_loadu_epi64(__mmask8(1), p);
#endif
    if (n >= 2) return _mm_loadu_si128((__m128i const*)p);
    if (n == 1) return _mm_loadl_epi64((__m128i const*)p);
    return _mm_setzero_si128();
  }
  template <> inline void storeu_first_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec, Integer n) {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__clang__) && !defined(SCTL_TUNE_ZEN)
    if (n == 1) {
      _mm_mask_storeu_epi64(p, __mmask8(1), vec.v);
      return;
    }
#endif
    if (n >= 2) {
      _mm_storeu_si128((__m128i*)p, vec.v);
    } else if (n == 1) {
      _mm_storel_epi64((__m128i*)p, vec.v);
    }
  }

  // Number of selected lanes
#if defined(__POPCNT__) || defined(__ARM_NEON)
  template <> inline Integer mask_count_intrin<VecData<int8_t ,16>>(const Mask<VecData<int8_t ,16>>& m) { return _mm_popcnt_u32(_mm_movemask_epi8(m.v)); }
  template <> inline Integer mask_count_intrin<VecData<int16_t, 8>>(const Mask<VecData<int16_t, 8>>& m) { return _mm_popcnt_u32(_mm_movemask_epi8(m.v)) / 2; } // two bits per lane
  template <> inline Integer mask_count_intrin<VecData<int32_t, 4>>(const Mask<VecData<int32_t, 4>>& m) { return _mm_popcnt_u32(_mm_movemask_ps(_mm_castsi128_ps(m.v))); }
  template <> inline Integer mask_count_intrin<VecData<int64_t, 2>>(const Mask<VecData<int64_t, 2>>& m) { return _mm_popcnt_u32(_mm_movemask_pd(_mm_castsi128_pd(m.v))); }
  template <> inline Integer mask_count_intrin<VecData<float  , 4>>(const Mask<VecData<float  , 4>>& m) { return _mm_popcnt_u32(_mm_movemask_ps(m.v)); }
  template <> inline Integer mask_count_intrin<VecData<double , 2>>(const Mask<VecData<double , 2>>& m) { return _mm_popcnt_u32(_mm_movemask_pd(m.v)); }
#endif

  // Math functions
  template <> inline VecData<float ,4> sqrt_intrin <VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_sqrt_ps (x.v); }
  template <> inline VecData<double,2> sqrt_intrin <VecData<double,2>>(const VecData<double,2>& x) { return _mm_sqrt_pd (x.v); }
  template <> inline VecData<float ,4> floor_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_floor_ps(x.v); }
  template <> inline VecData<double,2> floor_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_floor_pd(x.v); }
  template <> inline VecData<float ,4> ceil_intrin <VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_ceil_ps (x.v); }
  template <> inline VecData<double,2> ceil_intrin <VecData<double,2>>(const VecData<double,2>& x) { return _mm_ceil_pd (x.v); }
  template <> inline VecData<float ,4> trunc_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_round_ps(x.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
  template <> inline VecData<double,2> trunc_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_round_pd(x.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
  template <> inline Mask<VecData<float ,4>> isnan_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return Mask<VecData<float ,4>>(_mm_cmpunord_ps(x.v, x.v)); }
  template <> inline Mask<VecData<double,2>> isnan_intrin<VecData<double,2>>(const VecData<double,2>& x) { return Mask<VecData<double,2>>(_mm_cmpunord_pd(x.v, x.v)); }
  template <> inline VecData<int8_t ,16> fabs_intrin<VecData<int8_t ,16>>(const VecData<int8_t ,16>& x) { return _mm_abs_epi8 (x.v); }
  template <> inline VecData<int16_t, 8> fabs_intrin<VecData<int16_t, 8>>(const VecData<int16_t, 8>& x) { return _mm_abs_epi16(x.v); }
  template <> inline VecData<int32_t, 4> fabs_intrin<VecData<int32_t, 4>>(const VecData<int32_t, 4>& x) { return _mm_abs_epi32(x.v); }
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t, 2> fabs_intrin<VecData<int64_t, 2>>(const VecData<int64_t, 2>& x) { return _mm_abs_epi64(x.v); }
  #endif


  // Special functions
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<float,4>> {
    static inline VecData<float,4> eval(const VecData<float,4>& a) {
      #if defined(__AVX512F__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(_mm_maskz_rsqrt14_ps(~__mmask8(0), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(_mm_rsqrt_ps(a.v), a.v);
      #endif
    }
    static inline VecData<float,4> eval(const VecData<float,4>& a, const Mask<VecData<float,4>>& m) {
      #if defined(__AVX512DQ__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(_mm_maskz_rsqrt14_ps(_mm_movepi32_mask(_mm_castps_si128(m.v)), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(and_intrin(VecData<float,4>(_mm_rsqrt_ps(a.v)), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<double,2>> {
    static inline VecData<double,2> eval(const VecData<double,2>& a) {
      #if defined(__AVX512F__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,2>>::eval(_mm_maskz_rsqrt14_pd(~__mmask8(0), a.v), a.v);
      #else
      return rsqrt_bittrick_intrin<digits,VecData<double,2>>::eval(a); // a float estimate would limit x to the float range
      #endif
    }
    static inline VecData<double,2> eval(const VecData<double,2>& a, const Mask<VecData<double,2>>& m) {
      #if defined(__AVX512DQ__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,2>>::eval(_mm_maskz_rsqrt14_pd(_mm_movepi64_mask(_mm_castpd_si128(m.v)), a.v), a.v);
      #else
      return rsqrt_bittrick_intrin<digits,VecData<double,2>>::eval(a, m);
      #endif
    }
  };

  #ifdef SCTL_HAVE_SVML
  template <> inline void sincos_intrin<VecData<float ,4>>(VecData<float ,4>& sinx, VecData<float ,4>& cosx, const VecData<float ,4>& x) { sinx = _mm_sincos_ps(&cosx.v, x.v); }
  template <> inline void sincos_intrin<VecData<double,2>>(VecData<double,2>& sinx, VecData<double,2>& cosx, const VecData<double,2>& x) { sinx = _mm_sincos_pd(&cosx.v, x.v); }

  template <> inline VecData<float ,4> log_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_log_ps(x.v); }
  template <> inline VecData<double,2> log_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_log_pd(x.v); }

  template <> inline VecData<float ,4> exp_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_exp_ps(x.v); }
  template <> inline VecData<double,2> exp_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_exp_pd(x.v); }

  template <> inline VecData<float ,4> pow_intrin<VecData<float ,4>>(const VecData<float ,4>& x, const VecData<float ,4>& y) { return _mm_pow_ps(x.v, y.v); }
  template <> inline VecData<double,2> pow_intrin<VecData<double,2>>(const VecData<double,2>& x, const VecData<double,2>& y) { return _mm_pow_pd(x.v, y.v); }
  #else
  template <> inline void sincos_intrin<VecData<float ,4>>(VecData<float ,4>& sinx, VecData<float ,4>& cosx, const VecData<float ,4>& x) {
    approx_sincos_intrin<-1>(sinx, cosx, x);
  }
  template <> inline void sincos_intrin<VecData<double,2>>(VecData<double,2>& sinx, VecData<double,2>& cosx, const VecData<double,2>& x) {
    approx_sincos_intrin<-1>(sinx, cosx, x);
  }

  // The bits of x by integer instructions, as in Agner Fog's vectorclass (log, w5-3435X, independent /
  // dependent calls; 4 floats: x86-64-v2 18.0 / 57.1 -> 17.7 / 53.1 cycles, haswell 15.9 / 56.6 -> 15.5 /
  // 48.9; 2 doubles: x86-64-v2 20.8 / 63.0 -> 19.4 / 60.9, haswell 18.1 / 66.7 -> 16.4 / 60.5)
  template <> inline Mask<VecData<float,4>> positive_normal_mask_intrin<VecData<float,4>>(const VecData<float,4>& x) {
    const __m128i t = _mm_sub_epi32(_mm_castps_si128(x.v), set1_intrin<VecData<int32_t,4>>(0x00800000).v); // below 0x7f000000 as unsigned for normal x > 0
    return Mask<VecData<float,4>>(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_min_epu32(t, set1_intrin<VecData<int32_t,4>>(0x7effffff).v), t)));
  }
  template <> inline void log_mant_intrin<VecData<float,4>>(VecData<float,4>& e, VecData<float,4>& m, const VecData<float,4>& x) {
    const __m128i t = _mm_castps_si128(x.v);
    const __m128i m2 = _mm_or_si128(_mm_and_si128(t, set1_intrin<VecData<int32_t,4>>(0x007fffff).v), set1_intrin<VecData<int32_t,4>>(0x3f000000).v); // in [1/2, 1)
    const __m128i e1 = _mm_sub_epi32(_mm_srli_epi32(_mm_slli_epi32(t, 1), 24), set1_intrin<VecData<int32_t,4>>(127).v); // x = 2^e1 (2 m2)
    const __m128i big = _mm_cmpgt_epi32(m2, set1_intrin<VecData<int32_t,4>>(0x3f3504f3).v); // m2 > sqrt(1/2), on the bits: positive floats are in the order of their bits
    m = _mm_add_ps(_mm_castsi128_ps(m2), _mm_andnot_ps(_mm_castsi128_ps(big), _mm_castsi128_ps(m2))); // 2 m2 where not big
    e = _mm_cvtepi32_ps(_mm_sub_epi32(e1, big)); // e1 + 1 where big: its lanes are -1
  }
  template <> inline void log_mant_intrin<VecData<double,2>>(VecData<double,2>& e, VecData<double,2>& m, const VecData<double,2>& x) {
    const __m128i t = _mm_castpd_si128(x.v);
    const __m128i m2 = _mm_or_si128(_mm_and_si128(t, set1_intrin<VecData<int64_t,2>>(0x000fffffffffffffLL).v), set1_intrin<VecData<int64_t,2>>(0x3fe0000000000000LL).v); // in [1/2, 1)
    const __m128i e1 = _mm_add_epi64(_mm_srli_epi64(_mm_slli_epi64(t, 1), 53), set1_intrin<VecData<int64_t,2>>(0x4338000000000000LL - 1023).v); // the bits of 1.5 2^52 + e1, x = 2^e1 (2 m2)
    const __m128i big = _mm_cmpgt_epi64(m2, set1_intrin<VecData<int64_t,2>>(0x3fe6a09e667f3bcdLL).v); // m2 > sqrt(1/2)
    m = _mm_add_pd(_mm_castsi128_pd(m2), _mm_andnot_pd(_mm_castsi128_pd(big), _mm_castsi128_pd(m2))); // 2 m2 where not big
    e = _mm_sub_pd(_mm_castsi128_pd(_mm_sub_epi64(e1, big)), _mm_set1_pd(0x1.8p52)); // e1 + 1 where big
  }

#ifdef SCTL_HAVE_LIBMVEC
  template <> inline VecData<float, 4> log_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _ZGVbN4v_logf(x.v); }
  template <> inline VecData<double,2> log_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _ZGVbN2v_log(x.v); }
  template <> inline VecData<float, 4> pow_intrin<VecData<float ,4>>(const VecData<float ,4>& x, const VecData<float ,4>& y) { return _ZGVbN4vv_powf(x.v, y.v); }
  template <> inline VecData<double,2> pow_intrin<VecData<double,2>>(const VecData<double,2>& x, const VecData<double,2>& y) { return _ZGVbN2vv_pow(x.v, y.v); }
#else
  template <> inline VecData<float ,4> log_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return log_poly_intrin(x); }
  template <> inline VecData<double,2> log_intrin<VecData<double,2>>(const VecData<double,2>& x) { return log_poly_intrin(x); }
#endif

  template <> inline VecData<float ,4> exp_intrin<VecData<float ,4>>(const VecData<float ,4>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<float>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  template <> inline VecData<double,2> exp_intrin<VecData<double,2>>(const VecData<double,2>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<double>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  // pow: glibc's pow for each element; pow_poly_intrin is slower for double and less accurate at 128 bits
  template <> inline VecData<float ,4> cbrt_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return cbrt_poly_intrin(x); }
  template <> inline VecData<double,2> cbrt_intrin<VecData<double,2>>(const VecData<double,2>& x) { return cbrt_poly_intrin(x); }
  template <> inline VecData<float ,4> fmod_intrin<VecData<float ,4>>(const VecData<float ,4>& x, const VecData<float ,4>& y) { return fmod_poly_intrin(x, y); }
  template <> inline VecData<double,2> fmod_intrin<VecData<double,2>>(const VecData<double,2>& x, const VecData<double,2>& y) { return fmod_poly_intrin(x, y); }
  #endif


#endif
}

namespace sctl { // AVX
#ifdef __AVX__
  template <> struct alignas(sizeof(int8_t) * 32) VecData<int8_t,32> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 32;
    VecData() = default;
    inline VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(int16_t) * 16) VecData<int16_t,16> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 16;
    VecData() = default;
    inline VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(int32_t) * 8) VecData<int32_t,8> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 8;
    VecData() = default;
    inline VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(int64_t) * 4) VecData<int64_t,4> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 4;
    VecData() = default;
    inline VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(float) * 8) VecData<float,8> {
    using ScalarType = float;
    static constexpr Integer Size = 8;
    VecData() = default;
    inline VecData(__m256 v_) : v(v_) {}
    __m256 v;
  };
  template <> struct alignas(sizeof(double) * 4) VecData<double,4> {
    using ScalarType = double;
    static constexpr Integer Size = 4;
    VecData() = default;
    inline VecData(__m256d v_) : v(v_) {}
    __m256d v;
  };

  // Select between two sources, byte by byte. Used in various functions and operators
  // Corresponds to this pseudocode:
  // for (int i = 0; i < 32; i++) result[i] = s[i] ? a[i] : b[i];
  // Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
  // Only bit 7 in each byte of s is checked,
  #if defined(__AVX2__)
  static inline __m256i selectb (__m256i const & s, __m256i const & a, __m256i const & b) {
    return _mm256_blendv_epi8(b, a, s);

    //union U {
    //  __m256i  v;
    //  int8_t x[32];
    //};
    //U s_ = {s};
    //U a_ = {a};
    //U b_ = {b};
    //for (Integer i = 0; i < 32; i++) {
    //  a_.x[i] = (s_.x[i] ? a_.x[i] : b_.x[i]);
    //}
    //return a_.v;
  }
  #endif


  template <> inline VecData<int8_t,32> zero_intrin<VecData<int8_t,32>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<int16_t,16> zero_intrin<VecData<int16_t,16>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<int32_t,8> zero_intrin<VecData<int32_t,8>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<int64_t,4> zero_intrin<VecData<int64_t,4>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<float,8> zero_intrin<VecData<float,8>>() {
    return _mm256_setzero_ps();
  }
  template <> inline VecData<double,4> zero_intrin<VecData<double,4>>() {
    return _mm256_setzero_pd();
  }

#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
  // As dup64_const_intrin, for 256 bits, by one load (vbroadcastsd). With EachUse the asm is volatile: each use has
  // its own load, and GCC does not keep the value in a register through a function, for code with many values at
  // once (log in pow of 8 floats at haswell, GCC 15 and 16: 112 -> 118 cycles per dependent call without it); but
  // in a loop the volatile asm stays inside and copies the value at each pass.
  template <bool EachUse = false> inline __m256i dup64x4_const_intrin(const uint64_t bits) {
    double d;
    __builtin_memcpy(&d, &bits, sizeof(d));
    __m256d t = _mm256_set1_pd(d);
    if (EachUse) asm volatile("" : "+x"(t));
    else asm("" : "+x"(t));
    return _mm256_castpd_si256(t);
  }
#endif
  template <> inline VecData<int8_t,32> set1_intrin<VecData<int8_t,32>>(int8_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64x4_const_intrin((uint8_t)a * 0x0101010101010101ULL);
#endif
    return _mm256_set1_epi8(a);
  }
  template <> inline VecData<int16_t,16> set1_intrin<VecData<int16_t,16>>(int16_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64x4_const_intrin((uint16_t)a * 0x0001000100010001ULL);
#endif
    return _mm256_set1_epi16(a);
  }
  template <> inline VecData<int32_t,8> set1_intrin<VecData<int32_t,8>>(int32_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64x4_const_intrin((((uint64_t)(uint32_t)a) << 32) | (uint32_t)a);
#endif
    return _mm256_set1_epi32(a);
  }
  template <> inline VecData<int64_t,4> set1_intrin<VecData<int64_t,4>>(int64_t a) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
    if (__builtin_constant_p(a) && a != 0 && a != -1) return dup64x4_const_intrin((uint64_t)a);
#endif
    return _mm256_set1_epi64x(a);
  }
  template <> inline VecData<float,8> set1_intrin<VecData<float,8>>(float a) {
    return _mm256_set1_ps(a);
  }
  template <> inline VecData<double,4> set1_intrin<VecData<double,4>>(double a) {
    return _mm256_set1_pd(a);
  }

  template <> inline VecData<int8_t,32> set_intrin<VecData<int8_t,32>,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t>(int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23, int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31, int8_t v32) {
    return _mm256_set_epi8(v32,v31,v30,v29,v28,v27,v26,v25,v24,v23,v22,v21,v20,v19,v18,v17,v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int16_t,16> set_intrin<VecData<int16_t,16>,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t>(int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15, int16_t v16) {
    return _mm256_set_epi16(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int32_t,8> set_intrin<VecData<int32_t,8>,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t>(int32_t v1, int32_t v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8) {
    return _mm256_set_epi32(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int64_t,4> set_intrin<VecData<int64_t,4>,int64_t,int64_t,int64_t,int64_t>(int64_t v1, int64_t v2, int64_t v3, int64_t v4) {
    return _mm256_set_epi64x(v4,v3,v2,v1);
  }
  template <> inline VecData<float,8> set_intrin<VecData<float,8>,float,float,float,float,float,float,float,float>(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8) {
    return _mm256_set_ps(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<double,4> set_intrin<VecData<double,4>,double,double,double,double>(double v1, double v2, double v3, double v4) {
    return _mm256_set_pd(v4,v3,v2,v1);
  }

  template <> inline VecData<int8_t,32> load1_intrin<VecData<int8_t,32>>(int8_t const* p) {
    return _mm256_set1_epi8(p[0]);
  }
  template <> inline VecData<int16_t,16> load1_intrin<VecData<int16_t,16>>(int16_t const* p) {
    return _mm256_set1_epi16(p[0]);
  }
  template <> inline VecData<int32_t,8> load1_intrin<VecData<int32_t,8>>(int32_t const* p) {
    return _mm256_set1_epi32(p[0]);
  }
  template <> inline VecData<int64_t,4> load1_intrin<VecData<int64_t,4>>(int64_t const* p) {
    return _mm256_set1_epi64x(p[0]);
  }
  template <> inline VecData<float,8> load1_intrin<VecData<float,8>>(float const* p) {
    return _mm256_broadcast_ss(p);
  }
  template <> inline VecData<double,4> load1_intrin<VecData<double,4>>(double const* p) {
    return _mm256_broadcast_sd(p);
  }

  template <> inline VecData<int8_t,32> loadu_intrin<VecData<int8_t,32>>(int8_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<int16_t,16> loadu_intrin<VecData<int16_t,16>>(int16_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<int32_t,8> loadu_intrin<VecData<int32_t,8>>(int32_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<int64_t,4> loadu_intrin<VecData<int64_t,4>>(int64_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<float,8> loadu_intrin<VecData<float,8>>(float const* p) {
    return _mm256_loadu_ps(p);
  }
  template <> inline VecData<double,4> loadu_intrin<VecData<double,4>>(double const* p) {
    return _mm256_loadu_pd(p);
  }

  template <> inline VecData<int8_t,32> load_intrin<VecData<int8_t,32>>(int8_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<int16_t,16> load_intrin<VecData<int16_t,16>>(int16_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<int32_t,8> load_intrin<VecData<int32_t,8>>(int32_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<int64_t,4> load_intrin<VecData<int64_t,4>>(int64_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<float,8> load_intrin<VecData<float,8>>(float const* p) {
    return _mm256_load_ps(p);
  }
  template <> inline VecData<double,4> load_intrin<VecData<double,4>>(double const* p) {
    return _mm256_load_pd(p);
  }

  template <> inline void storeu_intrin<VecData<int8_t,32>>(int8_t* p, VecData<int8_t,32> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int16_t,16>>(int16_t* p, VecData<int16_t,16> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<float,8>>(float* p, VecData<float,8> vec) {
    _mm256_storeu_ps(p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<double,4>>(double* p, VecData<double,4> vec) {
    _mm256_storeu_pd(p, vec.v);
  }

  template <> inline void store_intrin<VecData<int8_t,32>>(int8_t* p, VecData<int8_t,32> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int16_t,16>>(int16_t* p, VecData<int16_t,16> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<float,8>>(float* p, VecData<float,8> vec) {
    _mm256_store_ps(p, vec.v);
  }
  template <> inline void store_intrin<VecData<double,4>>(double* p, VecData<double,4> vec) {
    _mm256_store_pd(p, vec.v);
  }

  //template <> inline int8_t extract_intrin<VecData<int8_t,32>>(VecData<int8_t,32> vec, Integer i) {}
  //template <> inline int16_t extract_intrin<VecData<int16_t,16>>(VecData<int16_t,16> vec, Integer i) {}
  //template <> inline int32_t extract_intrin<VecData<int32_t,8>>(VecData<int32_t,8> vec, Integer i) {}
  //template <> inline int64_t extract_intrin<VecData<int64_t,4>>(VecData<int64_t,4> vec, Integer i) {}
  //template <> inline float extract_intrin<VecData<float,8>>(VecData<float,8> vec, Integer i) {}
  //template <> inline double extract_intrin<VecData<double,4>>(VecData<double,4> vec, Integer i) {}

  //template <> inline void insert_intrin<VecData<int8_t,32>>(VecData<int8_t,32>& vec, Integer i, int8_t value) {}
  //template <> inline void insert_intrin<VecData<int16_t,16>>(VecData<int16_t,16>& vec, Integer i, int16_t value) {}
  //template <> inline void insert_intrin<VecData<int32_t,8>>(VecData<int32_t,8>& vec, Integer i, int32_t value) {}
  //template <> inline void insert_intrin<VecData<int64_t,4>>(VecData<int64_t,4>& vec, Integer i, int64_t value) {}
  //template <> inline void insert_intrin<VecData<float,8>>(VecData<float,8>& vec, Integer i, float value) {}
  //template <> inline void insert_intrin<VecData<double,4>>(VecData<double,4>& vec, Integer i, double value) {}

  #ifndef __AVX2__
  template <class F, class... A> inline __m256i avx_halves_intrin(const F& f, const A... a) { // f applied to each 128-bit half of the a
    return _mm256_insertf128_si256(_mm256_castsi128_si256(f(_mm256_castsi256_si128(a)...)), f(_mm256_extractf128_si256(a, 1)...), 1);
  }
  #endif

  // Arithmetic operators
  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> unary_minus_intrin<VecData<int8_t,32>>(const VecData<int8_t,32>& a) {
    return _mm256_sub_epi8(_mm256_setzero_si256(), a.v);
  }
  template <> inline VecData<int16_t,16> unary_minus_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a) {
    return _mm256_sub_epi16(_mm256_setzero_si256(), a.v);
  }
  template <> inline VecData<int32_t,8> unary_minus_intrin<VecData<int32_t,8>>(const VecData<int32_t,8>& a) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), a.v);
  }
  template <> inline VecData<int64_t,4> unary_minus_intrin<VecData<int64_t,4>>(const VecData<int64_t,4>& a) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), a.v);
  }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int8_t ,32> unary_minus_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a) { return avx_halves_intrin([](const __m128i x) { return _mm_sub_epi8 (_mm_setzero_si128(), x); }, a.v); }
  template <> inline VecData<int16_t,16> unary_minus_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a) { return avx_halves_intrin([](const __m128i x) { return _mm_sub_epi16(_mm_setzero_si128(), x); }, a.v); }
  template <> inline VecData<int32_t ,8> unary_minus_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a) { return avx_halves_intrin([](const __m128i x) { return _mm_sub_epi32(_mm_setzero_si128(), x); }, a.v); }
  template <> inline VecData<int64_t ,4> unary_minus_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a) { return avx_halves_intrin([](const __m128i x) { return _mm_sub_epi64(_mm_setzero_si128(), x); }, a.v); }
  #endif
  template <> inline VecData<float,8> unary_minus_intrin<VecData<float,8>>(const VecData<float,8>& a) {
    return _mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f));
  }
  template <> inline VecData<double,4> unary_minus_intrin<VecData<double,4>>(const VecData<double,4>& a) {
    static constexpr union {
      int32_t i[8];
      __m256  ymm;
    } u = {{0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000}};
    return _mm256_xor_pd(a.v, _mm256_castps_pd(u.ymm));
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> mul_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    // There is no 8-bit multiply in SSE2. Split into two 16-bit multiplies
    __m256i aodd    = _mm256_srli_epi16(a.v,8);               // odd numbered elements of a
    __m256i bodd    = _mm256_srli_epi16(b.v,8);               // odd numbered elements of b
    __m256i muleven = _mm256_mullo_epi16(a.v,b.v);            // product of even numbered elements
    __m256i mulodd  = _mm256_mullo_epi16(aodd,bodd);          // product of odd  numbered elements
            mulodd  = _mm256_slli_epi16(mulodd,8);            // put odd numbered elements back in place
    #if defined(__AVX512VL__) && defined(__AVX512BW__)
    return _mm256_mask_mov_epi8(mulodd, 0x55555555, muleven);
    #else
    __m256i mask    = set1_intrin<VecData<int32_t,8>>(0x00FF00FF).v; // mask for even positions
    __m256i product = selectb(mask,muleven,mulodd);           // interleave even and odd
    return product;
    #endif
  }
  template <> inline VecData<int16_t,16> mul_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_mullo_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> mul_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_mullo_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> mul_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    #if defined(__AVX512DQ__) && defined(__AVX512VL__)
    return _mm256_mullo_epi64(a.v, b.v);
    #else
    // Split into 32-bit multiplies
    __m256i bswap   = _mm256_shuffle_epi32(b.v,0xB1);         // swap H<->L
    __m256i prodlh  = _mm256_mullo_epi32(a.v,bswap);          // 32 bit L*H products
    __m256i zero    = _mm256_setzero_si256();                 // 0
    __m256i prodlh2 = _mm256_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m256i prodll  = _mm256_mul_epu32(a.v,b.v);              // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m256i prod    = _mm256_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
    #endif
  }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int8_t ,32> mul_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return mul_intrin(VecData<int8_t ,16>(x), VecData<int8_t ,16>(y)).v; }, a.v, b.v); }
  template <> inline VecData<int16_t,16> mul_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return mul_intrin(VecData<int16_t, 8>(x), VecData<int16_t, 8>(y)).v; }, a.v, b.v); }
  template <> inline VecData<int32_t ,8> mul_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return mul_intrin(VecData<int32_t, 4>(x), VecData<int32_t, 4>(y)).v; }, a.v, b.v); }
  template <> inline VecData<int64_t ,4> mul_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return mul_intrin(VecData<int64_t, 2>(x), VecData<int64_t, 2>(y)).v; }, a.v, b.v); }
  #endif
  template <> inline VecData<float,8> mul_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_mul_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> mul_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_mul_pd(a.v, b.v);
  }

  template <> inline VecData<float,8> div_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_div_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> div_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_div_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> add_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_add_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> add_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_add_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> add_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_add_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> add_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_add_epi64(a.v, b.v);
  }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int8_t ,32> add_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_add_epi8 (x, y); }, a.v, b.v); }
  template <> inline VecData<int16_t,16> add_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_add_epi16(x, y); }, a.v, b.v); }
  template <> inline VecData<int32_t ,8> add_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_add_epi32(x, y); }, a.v, b.v); }
  template <> inline VecData<int64_t ,4> add_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_add_epi64(x, y); }, a.v, b.v); }
  #endif
  template <> inline VecData<float,8> add_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_add_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> add_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_add_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> sub_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_sub_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> sub_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_sub_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> sub_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_sub_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> sub_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_sub_epi64(a.v, b.v);
  }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int8_t ,32> sub_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_sub_epi8 (x, y); }, a.v, b.v); }
  template <> inline VecData<int16_t,16> sub_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_sub_epi16(x, y); }, a.v, b.v); }
  template <> inline VecData<int32_t ,8> sub_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_sub_epi32(x, y); }, a.v, b.v); }
  template <> inline VecData<int64_t ,4> sub_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_sub_epi64(x, y); }, a.v, b.v); }
  #endif
  template <> inline VecData<float,8> sub_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_sub_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> sub_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_sub_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,32> fma_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b, const VecData<int8_t,32>& c) {}
  //template <> inline VecData<int16_t,16> fma_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b, const VecData<int16_t,16>& c) {}
  //template <> inline VecData<int32_t,8> sub_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b, const VecData<int32_t,8>& c) {}
  //template <> inline VecData<int64_t,4> sub_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b, const VecData<int64_t,4>& c) {}
  template <> inline VecData<float,8> fma_intrin(const VecData<float,8>& a, const VecData<float,8>& b, const VecData<float,8>& c) {
    #ifdef __FMA__
    return _mm256_fmadd_ps(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm256_macc_ps(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }
  template <> inline VecData<double,4> fma_intrin(const VecData<double,4>& a, const VecData<double,4>& b, const VecData<double,4>& c) {
    #ifdef __FMA__
    return _mm256_fmadd_pd(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm256_macc_pd(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }

  // Bitwise operators
  template <> inline VecData<int8_t,32> not_intrin<VecData<int8_t,32>>(const VecData<int8_t,32>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<int16_t,16> not_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<int32_t,8> not_intrin<VecData<int32_t,8>>(const VecData<int32_t,8>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<int64_t,4> not_intrin<VecData<int64_t,4>>(const VecData<int64_t,4>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<float,8> not_intrin<VecData<float,8>>(const VecData<float,8>& a) {
    return _mm256_xor_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
  }
  template <> inline VecData<double,4> not_intrin<VecData<double,4>>(const VecData<double,4>& a) {
    return _mm256_xor_pd(a.v, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> and_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> and_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> and_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> and_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  #else // AVX without AVX2: the instruction for float on the bits
  template <> inline VecData<int8_t ,32> and_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int16_t,16> and_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int32_t ,8> and_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int64_t ,4> and_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  #endif
  template <> inline VecData<float,8> and_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_and_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> and_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_and_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> xor_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> xor_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> xor_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> xor_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  #else // AVX without AVX2: the instruction for float on the bits
  template <> inline VecData<int8_t ,32> xor_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int16_t,16> xor_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int32_t ,8> xor_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int64_t ,4> xor_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  #endif
  template <> inline VecData<float,8> xor_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_xor_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> xor_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_xor_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> or_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> or_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> or_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> or_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  #else // AVX without AVX2: the instruction for float on the bits
  template <> inline VecData<int8_t ,32> or_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int16_t,16> or_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int32_t ,8> or_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  template <> inline VecData<int64_t ,4> or_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v))); }
  #endif
  template <> inline VecData<float,8> or_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_or_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> or_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_or_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> andnot_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  template <> inline VecData<int16_t,16> andnot_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  template <> inline VecData<int32_t,8> andnot_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  template <> inline VecData<int64_t,4> andnot_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  #else // AVX without AVX2: the instruction for float on the bits
  template <> inline VecData<int8_t ,32> andnot_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v))); }
  template <> inline VecData<int16_t,16> andnot_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v))); }
  template <> inline VecData<int32_t ,8> andnot_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v))); }
  template <> inline VecData<int64_t ,4> andnot_intrin(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v))); }
  #endif
  template <> inline VecData<float,8> andnot_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_andnot_ps(b.v, a.v);
  }
  template <> inline VecData<double,4> andnot_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_andnot_pd(b.v, a.v);
  }

  // Bitshift
  #ifdef __AVX2__
  template <> inline VecData<int8_t ,32> bitshiftleft_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a, const Integer& rhs) { // 16-bit bit shift, then the bits from the next byte cleared
    return _mm256_and_si256(_mm256_slli_epi16(a.v, (int)rhs), _mm256_set1_epi8((char)(rhs < 8 ? (0xFF << rhs) & 0xFF : 0)));
  }
  template <> inline VecData<int16_t,16> bitshiftleft_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const Integer& rhs) { return _mm256_slli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t ,8> bitshiftleft_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const Integer& rhs) { return _mm256_slli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,4> bitshiftleft_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { return _mm256_slli_epi64(a.v , rhs); }
  template <> inline VecData<float   ,8> bitshiftleft_intrin<VecData<float   ,8>>(const VecData<float   ,8>& a, const Integer& rhs) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(a.v), rhs)); }
  template <> inline VecData<double  ,4> bitshiftleft_intrin<VecData<double  ,4>>(const VecData<double  ,4>& a, const Integer& rhs) { return _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_castpd_si256(a.v), rhs)); }

  template <> inline VecData<int8_t ,32> bitshiftright_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a, const Integer& rhs) { // logical 16-bit bit shift, the bits from the next byte cleared, then the sign extended: (u ^ m) - m
    const int n = (int)(rhs < 7 ? rhs : 7); // larger bit shifts also give 0 or -1
    const __m256i u = _mm256_and_si256(_mm256_srli_epi16(a.v, n), _mm256_set1_epi8((char)(0xFF >> n)));
    const __m256i m = _mm256_set1_epi8((char)(0x80 >> n));
    return _mm256_sub_epi8(_mm256_xor_si256(u, m), m);
  }
  template <> inline VecData<int16_t,16> bitshiftright_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const Integer& rhs) { return _mm256_srai_epi16(a.v , rhs); }
  template <> inline VecData<int32_t ,8> bitshiftright_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const Integer& rhs) { return _mm256_srai_epi32(a.v , rhs); }
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t ,4> bitshiftright_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { return _mm256_srai_epi64(a.v , rhs); }
  #else
  template <> inline VecData<int64_t ,4> bitshiftright_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { // negative lanes: complement, logical bit shift, complement back
    const __m256i s = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.v);
    return _mm256_xor_si256(_mm256_srli_epi64(_mm256_xor_si256(a.v, s), rhs), s);
  }
  #endif
  template <> inline VecData<float   ,8> bitshiftright_intrin<VecData<float   ,8>>(const VecData<float   ,8>& a, const Integer& rhs) { return _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_castps_si256(a.v), rhs)); }
  template <> inline VecData<double  ,4> bitshiftright_intrin<VecData<double  ,4>>(const VecData<double  ,4>& a, const Integer& rhs) { return _mm256_castsi256_pd(_mm256_srli_epi64(_mm256_castpd_si256(a.v), rhs)); }

  template <> inline VecData<int32_t ,8> bitshiftleft_intrin <VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& rhs) { return _mm256_sllv_epi32(a.v, rhs.v); }
  template <> inline VecData<int64_t ,4> bitshiftleft_intrin <VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& rhs) { return _mm256_sllv_epi64(a.v, rhs.v); }
  template <> inline VecData<int32_t ,8> bitshiftright_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& rhs) { return _mm256_srav_epi32(a.v, rhs.v); }
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t ,4> bitshiftright_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& rhs) { return _mm256_srav_epi64(a.v, rhs.v); }
  #else
  template <> inline VecData<int64_t ,4> bitshiftright_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& rhs) { // negative lanes: complement, logical bit shift, complement back
    const __m256i s = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.v);
    return _mm256_xor_si256(_mm256_srlv_epi64(_mm256_xor_si256(a.v, s), rhs.v), s);
  }
  #endif
  #if defined(__AVX512BW__) && defined(__AVX512VL__)
  template <> inline VecData<int16_t,16> bitshiftleft_intrin <VecData<int16_t,16>>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& rhs) { return _mm256_sllv_epi16(a.v, rhs.v); }
  template <> inline VecData<int16_t,16> bitshiftright_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& rhs) { return _mm256_srav_epi16(a.v, rhs.v); }
  #else // as for 8 lanes; unpack and pack work within each 128-bit half, in the same order
  template <> inline VecData<int16_t,16> bitshiftleft_intrin <VecData<int16_t,16>>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& rhs) {
    const __m256i z = _mm256_setzero_si256();
    const auto pow2 = [](const __m256i k) { return _mm256_cvttps_epi32(_mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(k, 23), set1_intrin<VecData<int32_t,8>>(127 << 23).v))); };
    return _mm256_mullo_epi16(a.v, _mm256_packus_epi32(pow2(_mm256_unpacklo_epi16(rhs.v, z)), pow2(_mm256_unpackhi_epi16(rhs.v, z))));
  }
  template <> inline VecData<int16_t,16> bitshiftright_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& rhs) {
    const __m256i z = _mm256_setzero_si256();
    const __m256i lo = _mm256_srav_epi32(_mm256_srai_epi32(_mm256_unpacklo_epi16(a.v, a.v), 16), _mm256_unpacklo_epi16(rhs.v, z));
    const __m256i hi = _mm256_srav_epi32(_mm256_srai_epi32(_mm256_unpackhi_epi16(a.v, a.v), 16), _mm256_unpackhi_epi16(rhs.v, z));
    return _mm256_packs_epi32(lo, hi);
  }
  #endif
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int8_t ,32> bitshiftleft_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return bitshiftleft_intrin(VecData<int8_t,16>(h), rhs).v; }, a.v); }
  template <> inline VecData<int16_t,16> bitshiftleft_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return _mm_slli_epi16(h, (int)rhs); }, a.v); }
  template <> inline VecData<int32_t ,8> bitshiftleft_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return _mm_slli_epi32(h, (int)rhs); }, a.v); }
  template <> inline VecData<int64_t ,4> bitshiftleft_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return _mm_slli_epi64(h, (int)rhs); }, a.v); }
  template <> inline VecData<float   ,8> bitshiftleft_intrin<VecData<float   ,8>>(const VecData<float   ,8>& a, const Integer& rhs) { return _mm256_castsi256_ps(avx_halves_intrin([rhs](const __m128i h) { return _mm_slli_epi32(h, (int)rhs); }, _mm256_castps_si256(a.v))); }
  template <> inline VecData<double  ,4> bitshiftleft_intrin<VecData<double  ,4>>(const VecData<double  ,4>& a, const Integer& rhs) { return _mm256_castsi256_pd(avx_halves_intrin([rhs](const __m128i h) { return _mm_slli_epi64(h, (int)rhs); }, _mm256_castpd_si256(a.v))); }
  template <> inline VecData<int8_t ,32> bitshiftright_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return bitshiftright_intrin(VecData<int8_t,16>(h), rhs).v; }, a.v); }
  template <> inline VecData<int16_t,16> bitshiftright_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return _mm_srai_epi16(h, (int)rhs); }, a.v); }
  template <> inline VecData<int32_t ,8> bitshiftright_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const Integer& rhs) { return avx_halves_intrin([rhs](const __m128i h) { return _mm_srai_epi32(h, (int)rhs); }, a.v); }
  template <> inline VecData<int64_t ,4> bitshiftright_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { // negative lanes: complement, logical bit shift, complement back
    return avx_halves_intrin([rhs](const __m128i h) {
      const __m128i s = _mm_cmpgt_epi64(_mm_setzero_si128(), h);
      return _mm_xor_si128(_mm_srli_epi64(_mm_xor_si128(h, s), (int)rhs), s);
    }, a.v);
  }
  template <> inline VecData<float   ,8> bitshiftright_intrin<VecData<float   ,8>>(const VecData<float   ,8>& a, const Integer& rhs) { return _mm256_castsi256_ps(avx_halves_intrin([rhs](const __m128i h) { return _mm_srli_epi32(h, (int)rhs); }, _mm256_castps_si256(a.v))); }
  template <> inline VecData<double  ,4> bitshiftright_intrin<VecData<double  ,4>>(const VecData<double  ,4>& a, const Integer& rhs) { return _mm256_castsi256_pd(avx_halves_intrin([rhs](const __m128i h) { return _mm_srli_epi64(h, (int)rhs); }, _mm256_castpd_si256(a.v))); }

  template <> inline VecData<int16_t,16> bitshiftleft_intrin <VecData<int16_t,16>>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& rhs) { return avx_halves_intrin([](const __m128i h, const __m128i k) { return bitshiftleft_intrin(VecData<int16_t,8>(h), VecData<int16_t,8>(k)).v; }, a.v, rhs.v); }
  template <> inline VecData<int32_t ,8> bitshiftleft_intrin <VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& rhs) { return avx_halves_intrin([](const __m128i h, const __m128i k) { return bitshiftleft_intrin(VecData<int32_t,4>(h), VecData<int32_t,4>(k)).v; }, a.v, rhs.v); } // the other bit shifts by a vector are faster in the generic code
  #endif

  // Other functions
  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> max_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_max_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> max_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_max_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> max_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_max_epi32(a.v, b.v);
  }
  #elif !defined(__clang__) // AVX without AVX2: each 128-bit half with SSE; with clang the generic loop is faster
  template <> inline VecData<int8_t ,32> max_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_max_epi8 (x, y); }, a.v, b.v); }
  template <> inline VecData<int16_t,16> max_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_max_epi16(x, y); }, a.v, b.v); }
  template <> inline VecData<int32_t ,8> max_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_max_epi32(x, y); }, a.v, b.v); }
  #endif
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,4> max_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_max_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> max_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_max_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> max_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_max_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> min_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_min_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> min_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_min_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> min_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_min_epi32(a.v, b.v);
  }
  #elif !defined(__clang__) // AVX without AVX2: each 128-bit half with SSE; with clang the generic loop is faster
  template <> inline VecData<int8_t ,32> min_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_min_epi8 (x, y); }, a.v, b.v); }
  template <> inline VecData<int16_t,16> min_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_min_epi16(x, y); }, a.v, b.v); }
  template <> inline VecData<int32_t ,8> min_intrin(const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_min_epi32(x, y); }, a.v, b.v); }
  #endif
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,4> min_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_min_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> min_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_min_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> min_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_min_pd(a.v, b.v);
  }

  template <> inline void transpose_intrin<VecData<double,4>>(VecData<double,4> (&v)[4]) {
    const __m256d r0 = v[0].v, r1 = v[1].v, r2 = v[2].v, r3 = v[3].v;

    const __m256d t0 = _mm256_shuffle_pd(r0, r1, 0x0);
    const __m256d t1 = _mm256_shuffle_pd(r2, r3, 0x0);
    const __m256d t2 = _mm256_shuffle_pd(r0, r1, 0xF);
    const __m256d t3 = _mm256_shuffle_pd(r2, r3, 0xF);

    v[0].v = _mm256_permute2f128_pd(t0, t1, 0x20);
    v[1].v = _mm256_permute2f128_pd(t2, t3, 0x20);
    v[2].v = _mm256_permute2f128_pd(t0, t1, 0x31);
    v[3].v = _mm256_permute2f128_pd(t2, t3, 0x31);
  }
  template <> inline void transpose_intrin<VecData<float,8>>(VecData<float,8> (&v)[8]) {
    const __m256 r0 = v[0].v, r1 = v[1].v, r2 = v[2].v, r3 = v[3].v;
    const __m256 r4 = v[4].v, r5 = v[5].v, r6 = v[6].v, r7 = v[7].v;

    const __m256 a0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 a1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 a2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 a3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 a4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 a5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 a6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 a7 = _mm256_unpackhi_ps(r6, r7);

    const __m256 t0 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(1,0,1,0));
    const __m256 t1 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(3,2,3,2));
    const __m256 t2 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(1,0,1,0));
    const __m256 t3 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(3,2,3,2));
    const __m256 t4 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(1,0,1,0));
    const __m256 t5 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(3,2,3,2));
    const __m256 t6 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(1,0,1,0));
    const __m256 t7 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(3,2,3,2));

    v[0].v = _mm256_permute2f128_ps(t0, t4, 0x20);
    v[1].v = _mm256_permute2f128_ps(t1, t5, 0x20);
    v[2].v = _mm256_permute2f128_ps(t2, t6, 0x20);
    v[3].v = _mm256_permute2f128_ps(t3, t7, 0x20);
    v[4].v = _mm256_permute2f128_ps(t0, t4, 0x31);
    v[5].v = _mm256_permute2f128_ps(t1, t5, 0x31);
    v[6].v = _mm256_permute2f128_ps(t2, t6, 0x31);
    v[7].v = _mm256_permute2f128_ps(t3, t7, 0x31);
  }
  template <> inline void transpose_intrin<VecData<int32_t,8>>(VecData<int32_t,8> (&v)[8]) { transpose_reinterpret_intrin<VecData<float ,8>>(v); }
  template <> inline void transpose_intrin<VecData<int64_t,4>>(VecData<int64_t,4> (&v)[4]) { transpose_reinterpret_intrin<VecData<double,4>>(v); }

  // The rows transposed and evaluated by Estrin's scheme together: one step of the scheme after each stage of the
  // transpose, q_k = c_2k + c_2k+1 t after the first, s_k = q_2k + q_2k+1 t^2 after the second.
  template <> inline VecData<double,4> eval_poly_rows_intrin<4, VecData<double,4>>(const double* const (&p)[4], const VecData<double,4>& t) {
    using V = VecData<double,4>;
    const __m256d tp[2] = {_mm256_permute2f128_pd(t.v, t.v, 0x00), _mm256_permute2f128_pd(t.v, t.v, 0x11)}; // (t_a, t_b, t_a, t_b) of lanes a, b = 2j, 2j+1
    V q[2]; // (q_0 of lanes 2j, 2j+1, q_1 of lanes 2j, 2j+1)
    for (Integer j = 0; j < 2; j++) {
      const __m256d ra = _mm256_loadu_pd(p[2 * j]);
      const __m256d rb = _mm256_loadu_pd(p[2 * j + 1]);
      q[j] = fma_intrin(V(_mm256_unpackhi_pd(ra, rb)), V(tp[j]), V(_mm256_unpacklo_pd(ra, rb)));
    }
    return fma_intrin(V(_mm256_permute2f128_pd(q[0].v, q[1].v, 0x31)), mul_intrin(t, t), V(_mm256_permute2f128_pd(q[0].v, q[1].v, 0x20)));
  }
  template <> inline VecData<float,8> eval_poly_rows_intrin<8, VecData<float,8>>(const float* const (&p)[8], const VecData<float,8>& t) {
    using V = VecData<float,8>;
    const __m256 tl = _mm256_permute2f128_ps(t.v, t.v, 0x00);
    const __m256 th = _mm256_permute2f128_ps(t.v, t.v, 0x11);
    const __m256 tp[4] = {_mm256_unpacklo_ps(tl, tl), _mm256_unpackhi_ps(tl, tl), _mm256_unpacklo_ps(th, th), _mm256_unpackhi_ps(th, th)}; // (t_a, t_a, t_b, t_b) of lanes a, b = 2j, 2j+1, in each 128-bit lane
    V q[4]; // (q_0, q_1 of lane 2j, q_0, q_1 of lane 2j+1, then q_2, q_3 of the two lanes)
    for (Integer j = 0; j < 4; j++) {
      const __m256 ra = _mm256_loadu_ps(p[2 * j]);
      const __m256 rb = _mm256_loadu_ps(p[2 * j + 1]);
      q[j] = fma_intrin(V(_mm256_shuffle_ps(ra, rb, _MM_SHUFFLE(3,1,3,1))), V(tp[j]), V(_mm256_shuffle_ps(ra, rb, _MM_SHUFFLE(2,0,2,0))));
    }
    const V t2 = mul_intrin(t, t);
    const __m256 t2p[2] = {_mm256_permute2f128_ps(t2.v, t2.v, 0x00), _mm256_permute2f128_ps(t2.v, t2.v, 0x11)}; // t^2 of lanes 4j..4j+3, twice
    V s[2]; // (s_0 of lanes 4j..4j+3, s_1 of lanes 4j..4j+3)
    for (Integer j = 0; j < 2; j++) {
      s[j] = fma_intrin(V(_mm256_shuffle_ps(q[2 * j].v, q[2 * j + 1].v, _MM_SHUFFLE(3,1,3,1))), V(t2p[j]), V(_mm256_shuffle_ps(q[2 * j].v, q[2 * j + 1].v, _MM_SHUFFLE(2,0,2,0))));
    }
    return fma_intrin(V(_mm256_permute2f128_ps(s[0].v, s[1].v, 0x31)), mul_intrin(t2, t2), V(_mm256_permute2f128_ps(s[0].v, s[1].v, 0x20)));
  }

  template <> inline VecData<float,8> swap_pairs_intrin(const VecData<float,8>& vec) { return _mm256_permute_ps(vec.v, 0xB1); }
  template <> inline VecData<double,4> swap_pairs_intrin(const VecData<double,4>& vec) { return _mm256_permute_pd(vec.v, 0x5); }

  #ifdef __AVX2__
  template <Integer Bits> struct UnpackIntrin256;
  template <> struct UnpackIntrin256< 8> {
    static inline __m256i lo(const __m256i& a, const __m256i& b) { return _mm256_unpacklo_epi8 (a, b); }
    static inline __m256i hi(const __m256i& a, const __m256i& b) { return _mm256_unpackhi_epi8 (a, b); }
  };
  template <> struct UnpackIntrin256<16> {
    static inline __m256i lo(const __m256i& a, const __m256i& b) { return _mm256_unpacklo_epi16(a, b); }
    static inline __m256i hi(const __m256i& a, const __m256i& b) { return _mm256_unpackhi_epi16(a, b); }
  };
  template <> struct UnpackIntrin256<32> {
    static inline __m256i lo(const __m256i& a, const __m256i& b) { return _mm256_unpacklo_epi32(a, b); }
    static inline __m256i hi(const __m256i& a, const __m256i& b) { return _mm256_unpackhi_epi32(a, b); }
  };
  template <> struct UnpackIntrin256<64> {
    static inline __m256i lo(const __m256i& a, const __m256i& b) { return _mm256_unpacklo_epi64(a, b); }
    static inline __m256i hi(const __m256i& a, const __m256i& b) { return _mm256_unpackhi_epi64(a, b); }
  };

  // 256-bit unpacks act within each 128-bit lane, so this transposes the MxM
  // block held in each lane independently. Same stride-doubling network as
  // TransposeNet128.
  template <Integer M, Integer Bits, Integer Stride> struct TransposeNet256 {
    static inline void apply(__m256i* v) {
      __m256i w[M];
      for (Integer k = 0; k < M; k += 2*Stride) {
        for (Integer j = 0; j < Stride; j++) {
          w[k+2*j+0] = UnpackIntrin256<Bits>::lo(v[k+j], v[k+j+Stride]);
          w[k+2*j+1] = UnpackIntrin256<Bits>::hi(v[k+j], v[k+j+Stride]);
        }
      }
      for (Integer i = 0; i < M; i++) v[i] = w[i];
      TransposeNet256<M,Bits*2,Stride*2>::apply(v);
    }
  };
  template <Integer M, Integer Bits> struct TransposeNet256<M,Bits,M> {
    static inline void apply(__m256i*) {}
  };

  // Viewing the NxN matrix as 2x2 blocks of HxH, the lane-wise networks above
  // give [[A^T,B^T],[C^T,D^T]]; the final permute stage swaps B^T and C^T.
  template <class VData> inline void transpose256_intrin(VData (&v)[VData::Size]) {
    static constexpr Integer N = VData::Size;
    static constexpr Integer H = N/2;
    static constexpr Integer W = sizeof(typename VData::ScalarType)*8;
    __m256i w[N];
    for (Integer i = 0; i < N; i++) w[i] = v[i].v;
    TransposeNet256<H,W,1>::apply(w);
    TransposeNet256<H,W,1>::apply(w+H);
    for (Integer i = 0; i < H; i++) {
      v[i  ].v = _mm256_permute2x128_si256(w[i], w[i+H], 0x20);
      v[i+H].v = _mm256_permute2x128_si256(w[i], w[i+H], 0x31);
    }
  }
  template <> inline void transpose_intrin<VecData<int16_t,16>>(VecData<int16_t,16> (&v)[16]) { transpose256_intrin(v); }
  template <> inline void transpose_intrin<VecData<int8_t ,32>>(VecData<int8_t ,32> (&v)[32]) { transpose256_intrin(v); }
  #else // AVX without AVX2: [[A,B],[C,D]] -> [[A^T,C^T],[B^T,D^T]] by the 128-bit transposes of the HxH blocks
  template <class VData> inline void transpose256_intrin(VData (&v)[VData::Size]) {
    using Half = VecData<typename VData::ScalarType, VData::Size/2>;
    static constexpr Integer H = VData::Size/2;
    Half a[H], b[H], c[H], d[H];
    for (Integer i = 0; i < H; i++) {
      a[i] = _mm256_castsi256_si128(v[i].v);
      b[i] = _mm256_extractf128_si256(v[i].v, 1);
      c[i] = _mm256_castsi256_si128(v[i+H].v);
      d[i] = _mm256_extractf128_si256(v[i+H].v, 1);
    }
    transpose_intrin(a);
    transpose_intrin(b);
    transpose_intrin(c);
    transpose_intrin(d);
    for (Integer i = 0; i < H; i++) {
      v[i  ].v = _mm256_insertf128_si256(_mm256_castsi128_si256(a[i].v), c[i].v, 1);
      v[i+H].v = _mm256_insertf128_si256(_mm256_castsi128_si256(b[i].v), d[i].v, 1);
    }
  }
  template <> inline void transpose_intrin<VecData<int16_t,16>>(VecData<int16_t,16> (&v)[16]) { transpose256_intrin(v); }
  template <> inline void transpose_intrin<VecData<int8_t ,32>>(VecData<int8_t ,32> (&v)[32]) { transpose256_intrin(v); }
  #endif

  // Conversion operators
  template <> inline VecData<float ,8> convert_int2real_intrin<VecData<float ,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& x) {
    return _mm256_cvtepi32_ps(x.v);
  }
  #if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<double,4> convert_int2real_intrin<VecData<double,4>,VecData<int64_t,4>>(const VecData<int64_t,4>& x) {
    return _mm256_cvtepi64_pd(x.v);
  }
  #endif

  template <> inline VecData<int32_t,8> lrint_intrin<VecData<int32_t,8>,VecData<float ,8>>(const VecData<float ,8>& x) {
    return _mm256_cvtps_epi32(x.v);
  }
  #if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t,4> lrint_intrin<VecData<int64_t,4>,VecData<double,4>>(const VecData<double,4>& x) {
    return _mm256_cvtpd_epi64(x.v);
  }
  #endif

  template <> inline VecData<float ,8> rint_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_round_ps(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }
  template <> inline VecData<double,4> rint_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_round_pd(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  #ifdef __AVX2__
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(_mm256_cmpgt_epi8(b.v,a.v));}
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::le>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));             }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(_mm256_cmpgt_epi8(a.v,b.v));}
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));             }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(_mm256_cmpeq_epi8(a.v,b.v));}
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));             }

  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(_mm256_cmpgt_epi16(b.v,a.v));}
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::le>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));               }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(_mm256_cmpgt_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));               }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(_mm256_cmpeq_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));               }

  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(_mm256_cmpgt_epi32(b.v,a.v));}
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::le>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));              }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(_mm256_cmpgt_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));              }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(_mm256_cmpeq_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));              }

  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(_mm256_cmpgt_epi64(b.v,a.v));}
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::le>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));              }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(_mm256_cmpgt_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));              }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(_mm256_cmpeq_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));              }
  #else // AVX without AVX2: each 128-bit half with SSE
  #if !defined(__clang__) // with clang the generic loop is faster, but not for int64: div_int64_intrin was then 1.7x slower
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi8(y, x); }, a.v, b.v)); }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::le>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a)); }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi8(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a)); }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpeq_epi8(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b)); }

  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi16(y, x); }, a.v, b.v)); }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::le>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a)); }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi16(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a)); }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpeq_epi16(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b)); }

  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi32(y, x); }, a.v, b.v)); }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::le>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a)); }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi32(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a)); }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpeq_epi32(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b)); }

  #endif
  // The first n lanes from the 128-bit halves: with clang, from the generic comparison, the in-place copy of 4 of 8
  // int32 by LoadPartial and StorePartial took 2.13 cycles with clang 18, against 1.36
  template <class VData> inline Mask<VData> mask_first_halves_intrin(Integer n) {
    using HalfVec = VecData<typename VData::ScalarType, VData::Size/2>;
    return Mask<VData>(_mm256_insertf128_si256(_mm256_castsi128_si256(mask_first_intrin<HalfVec>(n).v), mask_first_intrin<HalfVec>(n > HalfVec::Size ? n - HalfVec::Size : 0).v, 1));
  }
  template <> inline Mask<VecData<int8_t ,32>> mask_first_intrin<VecData<int8_t ,32>>(Integer n) { return mask_first_halves_intrin<VecData<int8_t ,32>>(n); }
  template <> inline Mask<VecData<int16_t,16>> mask_first_intrin<VecData<int16_t,16>>(Integer n) { return mask_first_halves_intrin<VecData<int16_t,16>>(n); }
  template <> inline Mask<VecData<int32_t, 8>> mask_first_intrin<VecData<int32_t, 8>>(Integer n) { return mask_first_halves_intrin<VecData<int32_t, 8>>(n); }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi64(y, x); }, a.v, b.v)); }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::le>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a)); }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpgt_epi64(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a)); }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(avx_halves_intrin([](const __m128i x, const __m128i y) { return _mm_cmpeq_epi64(x, y); }, a.v, b.v)); }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b)); }
  #endif

  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::lt>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::le>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::gt>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::ge>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::eq>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::ne>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_UQ));}

  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::lt>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::le>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::gt>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::ge>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::eq>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::ne>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_UQ));}

  #if defined(__AVX2__)
  template <> inline VecData<int8_t ,32> select_intrin(const Mask<VecData<int8_t ,32>>& s, const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int16_t,16> select_intrin(const Mask<VecData<int16_t,16>>& s, const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int32_t ,8> select_intrin(const Mask<VecData<int32_t ,8>>& s, const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int64_t ,4> select_intrin(const Mask<VecData<int64_t ,4>>& s, const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  #else // AVX without AVX2: the instructions for float and double on the bits
  template <> inline VecData<int8_t ,32> select_intrin(const Mask<VecData<int8_t ,32>>& s, const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return or_intrin(and_intrin(VecData<int8_t ,32>(s.v), a), andnot_intrin(b, VecData<int8_t ,32>(s.v))); }
  template <> inline VecData<int16_t,16> select_intrin(const Mask<VecData<int16_t,16>>& s, const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return or_intrin(and_intrin(VecData<int16_t,16>(s.v), a), andnot_intrin(b, VecData<int16_t,16>(s.v))); }
  template <> inline VecData<int32_t ,8> select_intrin(const Mask<VecData<int32_t ,8>>& s, const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v), _mm256_castsi256_ps(s.v))); }
  template <> inline VecData<int64_t ,4> select_intrin(const Mask<VecData<int64_t ,4>>& s, const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(b.v), _mm256_castsi256_pd(a.v), _mm256_castsi256_pd(s.v))); }
  #endif
  template <> inline VecData<float   ,8> select_intrin(const Mask<VecData<float   ,8>>& s, const VecData<float   ,8>& a, const VecData<float   ,8>& b) { return _mm256_blendv_ps  (b.v, a.v, s.v); }
  template <> inline VecData<double  ,4> select_intrin(const Mask<VecData<double  ,4>>& s, const VecData<double  ,4>& a, const VecData<double  ,4>& b) { return _mm256_blendv_pd  (b.v, a.v, s.v); }

  // Masked load and store
  #if defined(__AVX512DQ__) && defined(__AVX512VL__) // with AVX-512VL by a mask register from the sign bits of the lanes: vmaskmovps and vmaskmovpd take about 6 cycles on AMD Zen 4
  template <> inline VecData<float  ,8> loadu_mask_intrin<VecData<float  ,8>>(float   const* p, const Mask<VecData<float  ,8>>& m) { return _mm256_maskz_loadu_ps   (_mm256_movepi32_mask(_mm256_castps_si256(m.v)), p); }
  template <> inline VecData<double ,4> loadu_mask_intrin<VecData<double ,4>>(double  const* p, const Mask<VecData<double ,4>>& m) { return _mm256_maskz_loadu_pd   (_mm256_movepi64_mask(_mm256_castpd_si256(m.v)), p); }
  template <> inline VecData<int32_t,8> loadu_mask_intrin<VecData<int32_t,8>>(int32_t const* p, const Mask<VecData<int32_t,8>>& m) { return _mm256_maskz_loadu_epi32(_mm256_movepi32_mask(m.v), p); }
  template <> inline VecData<int64_t,4> loadu_mask_intrin<VecData<int64_t,4>>(int64_t const* p, const Mask<VecData<int64_t,4>>& m) { return _mm256_maskz_loadu_epi64(_mm256_movepi64_mask(m.v), p); }
  template <> inline void storeu_mask_intrin<VecData<float  ,8>>(float  * p, VecData<float  ,8> vec, const Mask<VecData<float  ,8>>& m) { _mm256_mask_storeu_ps   (p, _mm256_movepi32_mask(_mm256_castps_si256(m.v)), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<double ,4>>(double * p, VecData<double ,4> vec, const Mask<VecData<double ,4>>& m) { _mm256_mask_storeu_pd   (p, _mm256_movepi64_mask(_mm256_castpd_si256(m.v)), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec, const Mask<VecData<int32_t,8>>& m) { _mm256_mask_storeu_epi32(p, _mm256_movepi32_mask(m.v), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec, const Mask<VecData<int64_t,4>>& m) { _mm256_mask_storeu_epi64(p, _mm256_movepi64_mask(m.v), vec.v); }
  #else
  template <> inline VecData<float ,8> loadu_mask_intrin<VecData<float ,8>>(float  const* p, const Mask<VecData<float ,8>>& m) { return _mm256_maskload_ps(p, _mm256_castps_si256(m.v)); }
  template <> inline VecData<double,4> loadu_mask_intrin<VecData<double,4>>(double const* p, const Mask<VecData<double,4>>& m) { return _mm256_maskload_pd(p, _mm256_castpd_si256(m.v)); }
  template <> inline void storeu_mask_intrin<VecData<float ,8>>(float * p, VecData<float ,8> vec, const Mask<VecData<float ,8>>& m) { _mm256_maskstore_ps(p, _mm256_castps_si256(m.v), vec.v); }
  template <> inline void storeu_mask_intrin<VecData<double,4>>(double* p, VecData<double,4> vec, const Mask<VecData<double,4>>& m) { _mm256_maskstore_pd(p, _mm256_castpd_si256(m.v), vec.v); }
  #if defined(__AVX2__)
  template <> inline VecData<int32_t,8> loadu_mask_intrin<VecData<int32_t,8>>(int32_t const* p, const Mask<VecData<int32_t,8>>& m) { return _mm256_maskload_epi32((int const*)p, m.v); }
  template <> inline VecData<int64_t,4> loadu_mask_intrin<VecData<int64_t,4>>(int64_t const* p, const Mask<VecData<int64_t,4>>& m) { return _mm256_maskload_epi64((long long const*)p, m.v); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec, const Mask<VecData<int32_t,8>>& m) { _mm256_maskstore_epi32((int*)p, m.v, vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec, const Mask<VecData<int64_t,4>>& m) { _mm256_maskstore_epi64((long long*)p, m.v, vec.v); }
  #else
  template <> inline VecData<int32_t,8> loadu_mask_intrin<VecData<int32_t,8>>(int32_t const* p, const Mask<VecData<int32_t,8>>& m) { return _mm256_castps_si256(_mm256_maskload_ps((float const*)p, m.v)); }
  template <> inline VecData<int64_t,4> loadu_mask_intrin<VecData<int64_t,4>>(int64_t const* p, const Mask<VecData<int64_t,4>>& m) { return _mm256_castpd_si256(_mm256_maskload_pd((double const*)p, m.v)); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec, const Mask<VecData<int32_t,8>>& m) { _mm256_maskstore_ps((float*)p, m.v, _mm256_castsi256_ps(vec.v)); }
  template <> inline void storeu_mask_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec, const Mask<VecData<int64_t,4>>& m) { _mm256_maskstore_pd((double*)p, m.v, _mm256_castsi256_pd(vec.v)); }
  #endif
  #endif

  // Number of selected lanes
  #if defined(__POPCNT__)
  #if defined(__AVX2__)
  template <> inline Integer mask_count_intrin<VecData<int8_t ,32>>(const Mask<VecData<int8_t ,32>>& m) { return _mm_popcnt_u32(_mm256_movemask_epi8(m.v)); }
  template <> inline Integer mask_count_intrin<VecData<int16_t,16>>(const Mask<VecData<int16_t,16>>& m) { return _mm_popcnt_u32(_mm256_movemask_epi8(m.v)) / 2; } // two bits per lane
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline Integer mask_count_intrin<VecData<int8_t ,32>>(const Mask<VecData<int8_t ,32>>& m) { return _mm_popcnt_u32(_mm_movemask_epi8(_mm256_castsi256_si128(m.v))) + _mm_popcnt_u32(_mm_movemask_epi8(_mm256_extractf128_si256(m.v, 1))); }
  #if !defined(__clang__) // with clang the generic loop is faster
  template <> inline Integer mask_count_intrin<VecData<int16_t,16>>(const Mask<VecData<int16_t,16>>& m) { return (_mm_popcnt_u32(_mm_movemask_epi8(_mm256_castsi256_si128(m.v))) + _mm_popcnt_u32(_mm_movemask_epi8(_mm256_extractf128_si256(m.v, 1)))) / 2; } // two bits per lane
  #endif
  #endif
  template <> inline Integer mask_count_intrin<VecData<int32_t, 8>>(const Mask<VecData<int32_t, 8>>& m) { return _mm_popcnt_u32(_mm256_movemask_ps(_mm256_castsi256_ps(m.v))); }
  template <> inline Integer mask_count_intrin<VecData<int64_t, 4>>(const Mask<VecData<int64_t, 4>>& m) { return _mm_popcnt_u32(_mm256_movemask_pd(_mm256_castsi256_pd(m.v))); }
  template <> inline Integer mask_count_intrin<VecData<float  , 8>>(const Mask<VecData<float  , 8>>& m) { return _mm_popcnt_u32(_mm256_movemask_ps(m.v)); }
  template <> inline Integer mask_count_intrin<VecData<double , 4>>(const Mask<VecData<double , 4>>& m) { return _mm_popcnt_u32(_mm256_movemask_pd(m.v)); }
  #endif

  // Math functions
  template <> inline VecData<float ,8> sqrt_intrin <VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_sqrt_ps (x.v); }
  template <> inline VecData<double,4> sqrt_intrin <VecData<double,4>>(const VecData<double,4>& x) { return _mm256_sqrt_pd (x.v); }
  template <> inline VecData<float ,8> floor_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_floor_ps(x.v); }
  template <> inline VecData<double,4> floor_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_floor_pd(x.v); }
  template <> inline VecData<float ,8> ceil_intrin <VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_ceil_ps (x.v); }
  template <> inline VecData<double,4> ceil_intrin <VecData<double,4>>(const VecData<double,4>& x) { return _mm256_ceil_pd (x.v); }
  template <> inline VecData<float ,8> trunc_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_round_ps(x.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
  template <> inline VecData<double,4> trunc_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_round_pd(x.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
  template <> inline Mask<VecData<float ,8>> isnan_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return Mask<VecData<float ,8>>(_mm256_cmp_ps(x.v, x.v, _CMP_UNORD_Q)); }
  template <> inline Mask<VecData<double,4>> isnan_intrin<VecData<double,4>>(const VecData<double,4>& x) { return Mask<VecData<double,4>>(_mm256_cmp_pd(x.v, x.v, _CMP_UNORD_Q)); }
  #if defined(__AVX2__)
  template <> inline VecData<int8_t ,32> fabs_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& x) { return _mm256_abs_epi8 (x.v); }
  template <> inline VecData<int16_t,16> fabs_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& x) { return _mm256_abs_epi16(x.v); }
  template <> inline VecData<int32_t, 8> fabs_intrin<VecData<int32_t, 8>>(const VecData<int32_t, 8>& x) { return _mm256_abs_epi32(x.v); }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int8_t ,32> fabs_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& x) { return avx_halves_intrin([](const __m128i h) { return _mm_abs_epi8 (h); }, x.v); }
  template <> inline VecData<int16_t,16> fabs_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& x) { return avx_halves_intrin([](const __m128i h) { return _mm_abs_epi16(h); }, x.v); }
  template <> inline VecData<int32_t, 8> fabs_intrin<VecData<int32_t, 8>>(const VecData<int32_t, 8>& x) { return avx_halves_intrin([](const __m128i h) { return _mm_abs_epi32(h); }, x.v); }
  #endif
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int64_t, 4> fabs_intrin<VecData<int64_t, 4>>(const VecData<int64_t, 4>& x) { return _mm256_abs_epi64(x.v); }
  #elif defined(__AVX2__) // negative lanes: complement and add 1
  template <> inline VecData<int64_t, 4> fabs_intrin<VecData<int64_t, 4>>(const VecData<int64_t, 4>& x) {
    const __m256i s = _mm256_cmpgt_epi64(_mm256_setzero_si256(), x.v);
    return _mm256_sub_epi64(_mm256_xor_si256(x.v, s), s);
  }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int64_t, 4> fabs_intrin<VecData<int64_t, 4>>(const VecData<int64_t, 4>& x) {
    return avx_halves_intrin([](const __m128i h) {
      const __m128i s = _mm_cmpgt_epi64(_mm_setzero_si128(), h);
      return _mm_sub_epi64(_mm_xor_si128(h, s), s);
    }, x.v);
  }
  #endif

  // Gather and scatter
  #if defined(__AVX2__)
  template <> inline VecData<double,2> gather_intrin<VecData<double,2>,VecData<int64_t,2>>(double const* p, const VecData<int64_t,2>& idx) { return _mm_i64gather_pd   (p, idx.v, 8); }
  template <> inline VecData<double,4> gather_intrin<VecData<double,4>,VecData<int32_t,4>>(double const* p, const VecData<int32_t,4>& idx) { return _mm256_i32gather_pd(p, idx.v, 8); }
  template <> inline VecData<double,4> gather_intrin<VecData<double,4>,VecData<int64_t,4>>(double const* p, const VecData<int64_t,4>& idx) { return _mm256_i64gather_pd(p, idx.v, 8); }
  template <> inline VecData<float ,4> gather_intrin<VecData<float ,4>,VecData<int32_t,4>>(float  const* p, const VecData<int32_t,4>& idx) { return _mm_i32gather_ps   (p, idx.v, 4); }
  template <> inline VecData<float ,4> gather_intrin<VecData<float ,4>,VecData<int64_t,4>>(float  const* p, const VecData<int64_t,4>& idx) { return _mm256_i64gather_ps(p, idx.v, 4); }
  template <> inline VecData<float ,8> gather_intrin<VecData<float ,8>,VecData<int32_t,8>>(float  const* p, const VecData<int32_t,8>& idx) { return _mm256_i32gather_ps(p, idx.v, 4); }
  #endif
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline void scatter_intrin<VecData<double,2>,VecData<int64_t,2>>(double* p, const VecData<double,2>& vec, const VecData<int64_t,2>& idx) { _mm_i64scatter_pd   (p, idx.v, vec.v, 8); }
  template <> inline void scatter_intrin<VecData<double,4>,VecData<int32_t,4>>(double* p, const VecData<double,4>& vec, const VecData<int32_t,4>& idx) { _mm256_i32scatter_pd(p, idx.v, vec.v, 8); }
  template <> inline void scatter_intrin<VecData<double,4>,VecData<int64_t,4>>(double* p, const VecData<double,4>& vec, const VecData<int64_t,4>& idx) { _mm256_i64scatter_pd(p, idx.v, vec.v, 8); }
  template <> inline void scatter_intrin<VecData<float ,4>,VecData<int32_t,4>>(float * p, const VecData<float ,4>& vec, const VecData<int32_t,4>& idx) { _mm_i32scatter_ps   (p, idx.v, vec.v, 4); }
  template <> inline void scatter_intrin<VecData<float ,4>,VecData<int64_t,4>>(float * p, const VecData<float ,4>& vec, const VecData<int64_t,4>& idx) { _mm256_i64scatter_ps(p, idx.v, vec.v, 4); }
  template <> inline void scatter_intrin<VecData<float ,8>,VecData<int32_t,8>>(float * p, const VecData<float ,8>& vec, const VecData<int32_t,8>& idx) { _mm256_i32scatter_ps(p, idx.v, vec.v, 4); }
  #endif

  // Conversion between element types
  template <> inline VecData<double ,4> convert_intrin<VecData<double ,4>,VecData<int32_t,4>>(const VecData<int32_t,4>& a) { return _mm256_cvtepi32_pd (a.v); }
  template <> inline VecData<int32_t,4> convert_intrin<VecData<int32_t,4>,VecData<double ,4>>(const VecData<double ,4>& a) { return _mm256_cvttpd_epi32(a.v); }
  template <> inline VecData<double ,4> convert_intrin<VecData<double ,4>,VecData<float  ,4>>(const VecData<float  ,4>& a) { return _mm256_cvtps_pd    (a.v); }
  template <> inline VecData<float  ,4> convert_intrin<VecData<float  ,4>,VecData<double ,4>>(const VecData<double ,4>& a) { return _mm256_cvtpd_ps    (a.v); }
  template <> inline VecData<float  ,8> convert_intrin<VecData<float  ,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& a) { return _mm256_cvtepi32_ps (a.v); }
  template <> inline VecData<int32_t,8> convert_intrin<VecData<int32_t,8>,VecData<float  ,8>>(const VecData<float  ,8>& a) { return _mm256_cvttps_epi32(a.v); }
  #ifdef __AVX2__ // integers: the low bits; sign extension is as fast in the generic code
  template <> inline VecData<int16_t,8> convert_intrin<VecData<int16_t,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& a) {
    const __m256i lo = _mm256_and_si256(a.v, set1_intrin<VecData<int32_t,8>>(0xFFFF).v);
    return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(lo, lo), 0x08));
  }
  template <> inline VecData<int32_t,4> convert_intrin<VecData<int32_t,4>,VecData<int64_t,4>>(const VecData<int64_t,4>& a) { return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(a.v, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6))); }
  template <> inline VecData<int8_t,16> convert_intrin<VecData<int8_t,16>,VecData<int16_t,16>>(const VecData<int16_t,16>& a) {
    const __m256i lo = _mm256_and_si256(a.v, set1_intrin<VecData<int16_t,16>>(0xFF).v);
    return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi16(lo, lo), 0x08));
  }
  template <> inline VecData<float  ,8> convert_intrin<VecData<float  ,8>,VecData<int16_t,8>>(const VecData<int16_t,8>& a) { return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(a.v)); }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int16_t,8> convert_intrin<VecData<int16_t,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& a) {
    const __m128i m = set1_intrin<VecData<int32_t,4>>(0xFFFF).v;
    return _mm_packus_epi32(_mm_and_si128(_mm256_castsi256_si128(a.v), m), _mm_and_si128(_mm256_extractf128_si256(a.v, 1), m));
  }
  template <> inline VecData<int8_t,16> convert_intrin<VecData<int8_t,16>,VecData<int16_t,16>>(const VecData<int16_t,16>& a) {
    const __m128i m = set1_intrin<VecData<int16_t,8>>(0xFF).v;
    return _mm_packus_epi16(_mm_and_si128(_mm256_castsi256_si128(a.v), m), _mm_and_si128(_mm256_extractf128_si256(a.v, 1), m));
  }
  #if !defined(__clang__) // with clang the generic loop is faster
  template <> inline VecData<float  ,8> convert_intrin<VecData<float  ,8>,VecData<int16_t,8>>(const VecData<int16_t,8>& a) { return _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_cvtepi16_epi32(a.v)), _mm_cvtepi16_epi32(_mm_unpackhi_epi64(a.v, a.v)), 1)); }
  #endif
  #endif
  template <> inline VecData<int16_t,8> convert_intrin<VecData<int16_t,8>,VecData<float  ,8>>(const VecData<float  ,8>& a) { return convert_intrin<VecData<int16_t,8>>(VecData<int32_t,8>(_mm256_cvttps_epi32(a.v))); }
  #if defined(__AVX512DQ__) && defined(__AVX512VL__)
  template <> inline VecData<double ,4> convert_intrin<VecData<double ,4>,VecData<int64_t,4>>(const VecData<int64_t,4>& a) { return _mm256_cvtepi64_pd (a.v); }
  template <> inline VecData<int64_t,4> convert_intrin<VecData<int64_t,4>,VecData<double ,4>>(const VecData<double ,4>& a) { return _mm256_cvttpd_epi64(a.v); }
  #else
  template <> inline VecData<double ,4> convert_intrin<VecData<double ,4>,VecData<int64_t,4>>(const VecData<int64_t,4>& a) { // the high 32 bits times 2^32 plus the low 32 bits, both exact: one rounding, as static_cast
    const __m128 l = _mm_castsi128_ps(_mm256_castsi256_si128(a.v));
    const __m128 h = _mm_castsi128_ps(_mm256_extractf128_si256(a.v, 1));
    const __m256d hi = _mm256_cvtepi32_pd(_mm_castps_si128(_mm_shuffle_ps(l, h, 0xDD)));
    const __m256d lo = _mm256_sub_pd(_mm256_castps_pd(_mm256_blend_ps(_mm256_castsi256_ps(a.v), _mm256_castpd_ps(_mm256_set1_pd(0x1p52)), 0xAA)), _mm256_set1_pd(0x1p52)); // 2^52 + the low bits, minus 2^52
    return _mm256_add_pd(_mm256_mul_pd(hi, _mm256_set1_pd(0x1p32)), lo);
  }
  #endif

  // Halves of a vector
  template <> inline VecData<int8_t ,16> get_low_intrin <VecData<int8_t ,32>>(const VecData<int8_t ,32>& a) { return _mm256_castsi256_si128(a.v); }
  template <> inline VecData<int16_t, 8> get_low_intrin <VecData<int16_t,16>>(const VecData<int16_t,16>& a) { return _mm256_castsi256_si128(a.v); }
  template <> inline VecData<int32_t, 4> get_low_intrin <VecData<int32_t, 8>>(const VecData<int32_t, 8>& a) { return _mm256_castsi256_si128(a.v); }
  template <> inline VecData<int64_t, 2> get_low_intrin <VecData<int64_t, 4>>(const VecData<int64_t, 4>& a) { return _mm256_castsi256_si128(a.v); }
  template <> inline VecData<float  , 4> get_low_intrin <VecData<float  , 8>>(const VecData<float  , 8>& a) { return _mm256_castps256_ps128(a.v); }
  template <> inline VecData<double , 2> get_low_intrin <VecData<double , 4>>(const VecData<double , 4>& a) { return _mm256_castpd256_pd128(a.v); }

  template <> inline VecData<int8_t ,16> get_high_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a) { return _mm256_extractf128_si256(a.v, 1); }
  template <> inline VecData<int16_t, 8> get_high_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a) { return _mm256_extractf128_si256(a.v, 1); }
  template <> inline VecData<int32_t, 4> get_high_intrin<VecData<int32_t, 8>>(const VecData<int32_t, 8>& a) { return _mm256_extractf128_si256(a.v, 1); }
  template <> inline VecData<int64_t, 2> get_high_intrin<VecData<int64_t, 4>>(const VecData<int64_t, 4>& a) { return _mm256_extractf128_si256(a.v, 1); }
  template <> inline VecData<float  , 4> get_high_intrin<VecData<float  , 8>>(const VecData<float  , 8>& a) { return _mm256_extractf128_ps(a.v, 1); }
  template <> inline VecData<double , 2> get_high_intrin<VecData<double , 4>>(const VecData<double , 4>& a) { return _mm256_extractf128_pd(a.v, 1); }

  template <> inline VecData<int8_t ,32> concat_intrin<VecData<int8_t ,16>>(const VecData<int8_t ,16>& lo, const VecData<int8_t ,16>& hi) { return _mm256_insertf128_si256(_mm256_castsi128_si256(lo.v), hi.v, 1); }
  template <> inline VecData<int16_t,16> concat_intrin<VecData<int16_t, 8>>(const VecData<int16_t, 8>& lo, const VecData<int16_t, 8>& hi) { return _mm256_insertf128_si256(_mm256_castsi128_si256(lo.v), hi.v, 1); }
  template <> inline VecData<int32_t, 8> concat_intrin<VecData<int32_t, 4>>(const VecData<int32_t, 4>& lo, const VecData<int32_t, 4>& hi) { return _mm256_insertf128_si256(_mm256_castsi128_si256(lo.v), hi.v, 1); }
  template <> inline VecData<int64_t, 4> concat_intrin<VecData<int64_t, 2>>(const VecData<int64_t, 2>& lo, const VecData<int64_t, 2>& hi) { return _mm256_insertf128_si256(_mm256_castsi128_si256(lo.v), hi.v, 1); }
  template <> inline VecData<float  , 8> concat_intrin<VecData<float  , 4>>(const VecData<float  , 4>& lo, const VecData<float  , 4>& hi) { return _mm256_insertf128_ps(_mm256_castps128_ps256(lo.v), hi.v, 1); }
  template <> inline VecData<double , 4> concat_intrin<VecData<double , 2>>(const VecData<double , 2>& lo, const VecData<double , 2>& hi) { return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo.v), hi.v, 1); }

  // The first n lanes by their count: with AVX-512VL by a mask register, on AMD Zen by those of the 128-bit halves,
  // else by the masked load and store (with the halves, SmallGEMM was up to 1.6x slower on Ice Lake)
  #if defined(__AVX512F__) && defined(__AVX512VL__)
  template <> inline VecData<int32_t,8> loadu_first_intrin<VecData<int32_t,8>>(int32_t const* p, Integer n) { return _mm256_maskz_loadu_epi32(__mmask8((1u << (n < 8 ? n : 8)) - 1), p); }
  template <> inline VecData<int64_t,4> loadu_first_intrin<VecData<int64_t,4>>(int64_t const* p, Integer n) { return _mm256_maskz_loadu_epi64(__mmask8((1u << (n < 4 ? n : 4)) - 1), p); }
  template <> inline VecData<float  ,8> loadu_first_intrin<VecData<float  ,8>>(float   const* p, Integer n) { return _mm256_maskz_loadu_ps   (__mmask8((1u << (n < 8 ? n : 8)) - 1), p); }
  template <> inline VecData<double ,4> loadu_first_intrin<VecData<double ,4>>(double  const* p, Integer n) { return _mm256_maskz_loadu_pd   (__mmask8((1u << (n < 4 ? n : 4)) - 1), p); }
  template <> inline void storeu_first_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec, Integer n) { _mm256_mask_storeu_epi32(p, __mmask8((1u << (n < 8 ? n : 8)) - 1), vec.v); }
  template <> inline void storeu_first_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec, Integer n) { _mm256_mask_storeu_epi64(p, __mmask8((1u << (n < 4 ? n : 4)) - 1), vec.v); }
  template <> inline void storeu_first_intrin<VecData<float  ,8>>(float  * p, VecData<float  ,8> vec, Integer n) { _mm256_mask_storeu_ps   (p, __mmask8((1u << (n < 8 ? n : 8)) - 1), vec.v); }
  template <> inline void storeu_first_intrin<VecData<double ,4>>(double * p, VecData<double ,4> vec, Integer n) { _mm256_mask_storeu_pd   (p, __mmask8((1u << (n < 4 ? n : 4)) - 1), vec.v); }
  #elif defined(SCTL_TUNE_ZEN)
  template <class VData> inline VData loadu_first_halves_intrin(typename VData::ScalarType const* p, Integer n) {
    using HalfVec = VecData<typename VData::ScalarType, VData::Size/2>;
    if (n >= VData::Size) return loadu_intrin<VData>(p);
    if (n > HalfVec::Size) return concat_intrin(loadu_intrin<HalfVec>(p), loadu_first_intrin<HalfVec>(p + HalfVec::Size, n - HalfVec::Size));
    return concat_intrin(loadu_first_intrin<HalfVec>(p, n), zero_intrin<HalfVec>());
  }
  template <class VData> inline void storeu_first_halves_intrin(typename VData::ScalarType* p, const VData& vec, Integer n) {
    using HalfVec = VecData<typename VData::ScalarType, VData::Size/2>;
    if (n >= VData::Size) {
      storeu_intrin(p, vec);
    } else if (n > HalfVec::Size) {
      storeu_intrin(p, get_low_intrin(vec));
      storeu_first_intrin(p + HalfVec::Size, get_high_intrin(vec), n - HalfVec::Size);
    } else {
      storeu_first_intrin(p, get_low_intrin(vec), n);
    }
  }
  template <> inline VecData<int32_t,8> loadu_first_intrin<VecData<int32_t,8>>(int32_t const* p, Integer n) { return loadu_first_halves_intrin<VecData<int32_t,8>>(p, n); }
  template <> inline VecData<int64_t,4> loadu_first_intrin<VecData<int64_t,4>>(int64_t const* p, Integer n) { return loadu_first_halves_intrin<VecData<int64_t,4>>(p, n); }
  template <> inline VecData<float  ,8> loadu_first_intrin<VecData<float  ,8>>(float   const* p, Integer n) { return loadu_first_halves_intrin<VecData<float  ,8>>(p, n); }
  template <> inline VecData<double ,4> loadu_first_intrin<VecData<double ,4>>(double  const* p, Integer n) { return loadu_first_halves_intrin<VecData<double ,4>>(p, n); }
  template <> inline void storeu_first_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec, Integer n) { storeu_first_halves_intrin(p, vec, n); }
  template <> inline void storeu_first_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec, Integer n) { storeu_first_halves_intrin(p, vec, n); }
  template <> inline void storeu_first_intrin<VecData<float  ,8>>(float  * p, VecData<float  ,8> vec, Integer n) { storeu_first_halves_intrin(p, vec, n); }
  template <> inline void storeu_first_intrin<VecData<double ,4>>(double * p, VecData<double ,4> vec, Integer n) { storeu_first_halves_intrin(p, vec, n); }
  #endif

  #ifdef __AVX2__
  template <> inline VecData<int16_t,16> div_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { // through float, exact for 16 bits; the low 16 bits of the quotient, as the C++ conversion
    const auto q = [](const __m128i a8, const __m128i b8) { return _mm256_and_si256(_mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(a8)), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(b8)))), set1_intrin<VecData<int32_t,8>>(0xFFFF).v); };
    const __m256i lo = q(_mm256_castsi256_si128(a.v), _mm256_castsi256_si128(b.v));
    const __m256i hi = q(_mm256_extracti128_si256(a.v, 1), _mm256_extracti128_si256(b.v, 1));
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8); // packus works within 128-bit lanes
  }
  template <> inline VecData<int8_t,32> div_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { // through float, eight lanes at a time; the low 8 bits of the quotient
    const auto q = [](const __m128i a8, const __m128i b8) { return _mm256_and_si256(_mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a8)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8)))), set1_intrin<VecData<int32_t,8>>(0xFF).v); };
    const __m128i al = _mm256_castsi256_si128(a.v);
    const __m128i ah = _mm256_extracti128_si256(a.v, 1);
    const __m128i bl = _mm256_castsi256_si128(b.v);
    const __m128i bh = _mm256_extracti128_si256(b.v, 1);
    const __m256i w01 = _mm256_permute4x64_epi64(_mm256_packus_epi32(q(al, bl), q(_mm_srli_si128(al, 8), _mm_srli_si128(bl, 8))), 0xD8);
    const __m256i w23 = _mm256_permute4x64_epi64(_mm256_packus_epi32(q(ah, bh), q(_mm_srli_si128(ah, 8), _mm_srli_si128(bh, 8))), 0xD8);
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(w01, w23), 0xD8);
  }
  #else // AVX without AVX2: each 128-bit half with SSE
  template <> inline VecData<int16_t,16> div_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return div_intrin(VecData<int16_t, 8>(x), VecData<int16_t, 8>(y)).v; }, a.v, b.v); }
  template <> inline VecData<int8_t ,32> div_intrin(const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return avx_halves_intrin([](const __m128i x, const __m128i y) { return div_intrin(VecData<int8_t ,16>(x), VecData<int8_t ,16>(y)).v; }, a.v, b.v); }
  #endif
  template <> inline VecData<int64_t,4> div_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return div_int64_intrin(a, b); }
  template <> inline VecData<int32_t,8> div_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { // through float on Intel (12.8 cycles against 16.0 on a w5-3435X), through double on AMD Zen (10.0 against 17.7 on Zen 2)
    #if defined(__AVX2__) && !defined(SCTL_TUNE_ZEN)
    return div_int32_intrin(a, b);
    #elif defined(__AVX512F__)
    return _mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(a.v), _mm512_cvtepi32_pd(b.v)));
    #else
    return concat_intrin(div_intrin(get_low_intrin(a), get_low_intrin(b)), div_intrin(get_high_intrin(a), get_high_intrin(b)));
    #endif
  }



  // Special functions
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<float,8>> {
    static inline VecData<float,8> eval(const VecData<float,8>& a) {
      #if defined(__AVX512F__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(_mm256_maskz_rsqrt14_ps(~__mmask8(0), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(_mm256_rsqrt_ps(a.v), a.v);
      #endif
    }
    static inline VecData<float,8> eval(const VecData<float,8>& a, const Mask<VecData<float,8>>& m) {
      #if defined(__AVX512DQ__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(_mm256_maskz_rsqrt14_ps(_mm256_movepi32_mask(_mm256_castps_si256(m.v)), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(and_intrin(VecData<float,8>(_mm256_rsqrt_ps(a.v)), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<double,4>> {
    static inline VecData<double,4> eval(const VecData<double,4>& a) {
      #if defined(__AVX512F__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,4>>::eval(_mm256_maskz_rsqrt14_pd(~__mmask8(0), a.v), a.v);
      #else
      return rsqrt_bittrick_intrin<digits,VecData<double,4>>::eval(a); // a float estimate would limit x to the float range
      #endif
    }
    static inline VecData<double,4> eval(const VecData<double,4>& a, const Mask<VecData<double,4>>& m) {
      #if defined(__AVX512DQ__) && defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,4>>::eval(_mm256_maskz_rsqrt14_pd(_mm256_movepi64_mask(_mm256_castpd_si256(m.v)), a.v), a.v);
      #else
      return rsqrt_bittrick_intrin<digits,VecData<double,4>>::eval(a, m);
      #endif
    }
  };

  #if defined(__AVX2__) // log_mant_intrin by integer instructions, as for 4 floats and 2 doubles (log, haswell: 8 floats 17.0 / 56.5 -> 14.5 / 48.5 cycles, 4 doubles 20.6 / 71.1 -> 18.4 / 67.3)
  template <> inline void log_mant_intrin<VecData<float,8>>(VecData<float,8>& e, VecData<float,8>& m, const VecData<float,8>& x) {
    #if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) // a load at each use, see dup64x4_const_intrin
    const auto c = [](const int32_t a) { return dup64x4_const_intrin<true>((((uint64_t)(uint32_t)a) << 32) | (uint32_t)a); };
    #else
    const auto c = [](const int32_t a) { return _mm256_set1_epi32(a); };
    #endif
    const __m256i t = _mm256_castps_si256(x.v);
    const __m256i m2 = _mm256_or_si256(_mm256_and_si256(t, c(0x007fffff)), c(0x3f000000));
    const __m256i e1 = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_slli_epi32(t, 1), 24), c(127));
    const __m256i big = _mm256_cmpgt_epi32(m2, c(0x3f3504f3));
    m = _mm256_add_ps(_mm256_castsi256_ps(m2), _mm256_andnot_ps(_mm256_castsi256_ps(big), _mm256_castsi256_ps(m2)));
    e = _mm256_cvtepi32_ps(_mm256_sub_epi32(e1, big));
  }
  template <> inline void log_mant_intrin<VecData<double,4>>(VecData<double,4>& e, VecData<double,4>& m, const VecData<double,4>& x) {
    #if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) // a load at each use, see dup64x4_const_intrin
    const auto c = [](const int64_t a) { return dup64x4_const_intrin<true>((uint64_t)a); };
    #else
    const auto c = [](const int64_t a) { return _mm256_set1_epi64x(a); };
    #endif
    const __m256i t = _mm256_castpd_si256(x.v);
    const __m256i m2 = _mm256_or_si256(_mm256_and_si256(t, c(0x000fffffffffffffLL)), c(0x3fe0000000000000LL));
    const __m256i e1 = _mm256_add_epi64(_mm256_srli_epi64(_mm256_slli_epi64(t, 1), 53), c(0x4338000000000000LL - 1023));
    const __m256i big = _mm256_cmpgt_epi64(m2, c(0x3fe6a09e667f3bcdLL));
    m = _mm256_add_pd(_mm256_castsi256_pd(m2), _mm256_andnot_pd(_mm256_castsi256_pd(big), _mm256_castsi256_pd(m2)));
    e = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_sub_epi64(e1, big)), _mm256_set1_pd(0x1.8p52));
  }
  #endif

  #ifdef SCTL_HAVE_SVML
  template <> inline void sincos_intrin<VecData<float ,8>>(VecData<float ,8>& sinx, VecData<float ,8>& cosx, const VecData<float ,8>& x) { sinx = _mm256_sincos_ps(&cosx.v, x.v); }
  template <> inline void sincos_intrin<VecData<double,4>>(VecData<double,4>& sinx, VecData<double,4>& cosx, const VecData<double,4>& x) { sinx = _mm256_sincos_pd(&cosx.v, x.v); }

  template <> inline VecData<float ,8> log_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_log_ps(x.v); }
  template <> inline VecData<double,4> log_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_log_pd(x.v); }

  template <> inline VecData<float ,8> exp_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_exp_ps(x.v); }
  template <> inline VecData<double,4> exp_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_exp_pd(x.v); }

  template <> inline VecData<float ,8> pow_intrin<VecData<float ,8>>(const VecData<float ,8>& x, const VecData<float ,8>& y) { return _mm256_pow_ps(x.v, y.v); }
  template <> inline VecData<double,4> pow_intrin<VecData<double,4>>(const VecData<double,4>& x, const VecData<double,4>& y) { return _mm256_pow_pd(x.v, y.v); }
  #else
  template <> inline void sincos_intrin<VecData<float ,8>>(VecData<float ,8>& sinx, VecData<float ,8>& cosx, const VecData<float ,8>& x) {
    approx_sincos_intrin<-1>(sinx, cosx, x);
  }
  template <> inline void sincos_intrin<VecData<double,4>>(VecData<double,4>& sinx, VecData<double,4>& cosx, const VecData<double,4>& x) {
    approx_sincos_intrin<-1>(sinx, cosx, x);
  }

#ifdef SCTL_HAVE_LIBMVEC
#ifdef __AVX2__
  template <> inline VecData<float ,8> log_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _ZGVdN8v_logf(x.v); }
  template <> inline VecData<double,4> log_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _ZGVdN4v_log(x.v); }
  template <> inline VecData<float ,8> pow_intrin<VecData<float ,8>>(const VecData<float ,8>& x, const VecData<float ,8>& y) { return _ZGVdN8vv_powf(x.v, y.v); }
  template <> inline VecData<double,4> pow_intrin<VecData<double,4>>(const VecData<double,4>& x, const VecData<double,4>& y) { return _ZGVdN4vv_pow(x.v, y.v); }
#else
  template <> inline VecData<float ,8> log_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _ZGVcN8v_logf(x.v); }
  template <> inline VecData<double,4> log_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _ZGVcN4v_log(x.v); }
  template <> inline VecData<float ,8> pow_intrin<VecData<float ,8>>(const VecData<float ,8>& x, const VecData<float ,8>& y) { return _ZGVcN8vv_powf(x.v, y.v); }
  template <> inline VecData<double,4> pow_intrin<VecData<double,4>>(const VecData<double,4>& x, const VecData<double,4>& y) { return _ZGVcN4vv_pow(x.v, y.v); }
#endif
#else
  template <> inline VecData<float ,8> log_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return log_poly_intrin(x); }
  template <> inline VecData<double,4> log_intrin<VecData<double,4>>(const VecData<double,4>& x) { return log_poly_intrin(x); }
#endif

  template <> inline VecData<float ,8> exp_intrin<VecData<float ,8>>(const VecData<float ,8>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<float>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  template <> inline VecData<double,4> exp_intrin<VecData<double,4>>(const VecData<double,4>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<double>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
#if !defined(SCTL_HAVE_LIBMVEC)
  template <> inline VecData<float,8> pow_intrin<VecData<float,8>>(const VecData<float,8>& x, const VecData<float,8>& y) { return pow_poly_intrin(x, y); }
  template <> inline VecData<double,4> pow_intrin<VecData<double,4>>(const VecData<double,4>& x, const VecData<double,4>& y) { return pow_poly_intrin(x, y); }
#endif
  template <> inline VecData<float ,8> cbrt_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return cbrt_poly_intrin(x); }
  template <> inline VecData<double,4> cbrt_intrin<VecData<double,4>>(const VecData<double,4>& x) { return cbrt_poly_intrin(x); }
  template <> inline VecData<float ,8> fmod_intrin<VecData<float ,8>>(const VecData<float ,8>& x, const VecData<float ,8>& y) { return fmod_poly_intrin(x, y); }
  template <> inline VecData<double,4> fmod_intrin<VecData<double,4>>(const VecData<double,4>& x, const VecData<double,4>& y) { return fmod_poly_intrin(x, y); }
  #endif


  template <> inline double reduce_add_intrin<VecData<double,4>>(const VecData<double,4>& a) {
    __m128d t = _mm_add_pd(_mm256_castpd256_pd128(a.v), _mm256_extractf128_pd(a.v, 1));
    t = _mm_hadd_pd(t, t);
    return _mm_cvtsd_f64(t);
  }

#ifdef __AVX2__
  // Left-pack permutation LUTs for mask_compress_iota_store: idx[m][k] = index of the k-th set
  // bit of mask m (lanes past the popcount stay 0 and are dropped by the count-limited store).
  struct Avx2IotaLUT8 {
    int32_t idx[256][8];
    constexpr Avx2IotaLUT8() : idx{} {
      for (int m = 0; m < 256; ++m) {
        int k = 0;
        for (int i = 0; i < 8; ++i) if (m & (1 << i)) idx[m][k++] = i;
      }
    }
  };
  struct Avx2IotaLUT4 {
    int32_t idx[16][4];
    constexpr Avx2IotaLUT4() : idx{} {
      for (int m = 0; m < 16; ++m) {
        int k = 0;
        for (int i = 0; i < 4; ++i) if (m & (1 << i)) idx[m][k++] = i;
      }
    }
  };
  inline constexpr Avx2IotaLUT8 avx2_iota_lut8{};
  inline constexpr Avx2IotaLUT4 avx2_iota_lut4{};

  // No AVX2 vcompress: left-pack the base+lane iota via a mask-indexed permute, then a
  // count-limited masked store so only the surviving lanes are written (no buffer overrun).
  template <> inline Integer mask_compress_iota_store<VecData<float,8>>(const Mask<VecData<float,8>>& mask, int32_t base, int32_t* ptr) {
    const unsigned m = (unsigned)_mm256_movemask_ps(mask.v);
    const __m256i iota = _mm256_add_epi32(_mm256_set1_epi32(base), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
    const __m256i perm = _mm256_loadu_si256((const __m256i*)avx2_iota_lut8.idx[m]);
    const __m256i packed = _mm256_permutevar8x32_epi32(iota, perm);
    const Integer cnt = (Integer)_mm_popcnt_u32(m);
    const __m256i smask = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)cnt), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
    _mm256_maskstore_epi32((int*)ptr, smask, packed);
    return cnt;
  }
  template <> inline Integer mask_compress_iota_store<VecData<double,4>>(const Mask<VecData<double,4>>& mask, int32_t base, int32_t* ptr) {
    const unsigned m = (unsigned)_mm256_movemask_pd(mask.v);
    const __m128i iota = _mm_add_epi32(_mm_set1_epi32(base), _mm_setr_epi32(0,1,2,3));
    const __m128i perm = _mm_loadu_si128((const __m128i*)avx2_iota_lut4.idx[m]);
    const __m128i packed = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(iota), perm));
    const Integer cnt = (Integer)_mm_popcnt_u32(m);
    const __m128i smask = _mm_cmpgt_epi32(_mm_set1_epi32((int)cnt), _mm_setr_epi32(0,1,2,3));
    _mm_maskstore_epi32((int*)ptr, smask, packed);
    return cnt;
  }
  // Fuse two 4-lane masks into one 8-lane compress (mirrors the AVX512 double-8 store2).
  template <> inline Integer mask_compress_iota_store2<VecData<double,4>>(const Mask<VecData<double,4>>& mask_lo, const Mask<VecData<double,4>>& mask_hi, int32_t base, int32_t* ptr) {
    const unsigned m = (unsigned)_mm256_movemask_pd(mask_lo.v) | ((unsigned)_mm256_movemask_pd(mask_hi.v) << 4);
    const __m256i iota = _mm256_add_epi32(_mm256_set1_epi32(base), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
    const __m256i perm = _mm256_loadu_si256((const __m256i*)avx2_iota_lut8.idx[m]);
    const __m256i packed = _mm256_permutevar8x32_epi32(iota, perm);
    const Integer cnt = (Integer)_mm_popcnt_u32(m);
    const __m256i smask = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)cnt), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
    _mm256_maskstore_epi32((int*)ptr, smask, packed);
    return cnt;
  }
#endif

#endif
}

namespace sctl { // AVX512
#if defined(__AVX512F__)
  template <> struct alignas(sizeof(int8_t) * 64) VecData<int8_t,64> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 64;
    VecData() = default;
    inline VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(int16_t) * 32) VecData<int16_t,32> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 32;
    VecData() = default;
    inline VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(int32_t) * 16) VecData<int32_t,16> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 16;
    VecData() = default;
    inline VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(int64_t) * 8) VecData<int64_t,8> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 8;
    VecData() = default;
    inline VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(float) * 16) VecData<float,16> {
    using ScalarType = float;
    static constexpr Integer Size = 16;
    VecData() = default;
    inline VecData(__m512 v_) : v(v_) {}
    __m512 v;
  };
  template <> struct alignas(sizeof(double) * 8) VecData<double,8> {
    using ScalarType = double;
    static constexpr Integer Size = 8;
    inline VecData(__m512d v_) : v(v_) {}
    VecData() = default;
    __m512d v;
  };



  template <> inline VecData<int8_t,64> zero_intrin<VecData<int8_t,64>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<int16_t,32> zero_intrin<VecData<int16_t,32>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<int32_t,16> zero_intrin<VecData<int32_t,16>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<int64_t,8> zero_intrin<VecData<int64_t,8>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<float,16> zero_intrin<VecData<float,16>>() {
    return _mm512_setzero_ps();
  }
  template <> inline VecData<double,8> zero_intrin<VecData<double,8>>() {
    return _mm512_setzero_pd();
  }

  template <> inline VecData<int8_t,64> set1_intrin<VecData<int8_t,64>>(int8_t a) {
    return _mm512_set1_epi8(a);
  }
  template <> inline VecData<int16_t,32> set1_intrin<VecData<int16_t,32>>(int16_t a) {
    return _mm512_set1_epi16(a);
  }
  template <> inline VecData<int32_t,16> set1_intrin<VecData<int32_t,16>>(int32_t a) {
    return _mm512_set1_epi32(a);
  }
  template <> inline VecData<int64_t,8> set1_intrin<VecData<int64_t,8>>(int64_t a) {
    return _mm512_set1_epi64(a);
  }
  template <> inline VecData<float,16> set1_intrin<VecData<float,16>>(float a) {
    return _mm512_set1_ps(a);
  }
  template <> inline VecData<double,8> set1_intrin<VecData<double,8>>(double a) {
    return _mm512_set1_pd(a);
  }

  template <> inline VecData<int8_t,64> set_intrin<VecData<int8_t,64>,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t>(int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23, int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31, int8_t v32, int8_t v33, int8_t v34, int8_t v35, int8_t v36, int8_t v37, int8_t v38, int8_t v39, int8_t v40, int8_t v41, int8_t v42, int8_t v43, int8_t v44, int8_t v45, int8_t v46, int8_t v47, int8_t v48, int8_t v49, int8_t v50, int8_t v51, int8_t v52, int8_t v53, int8_t v54, int8_t v55, int8_t v56, int8_t v57, int8_t v58, int8_t v59, int8_t v60, int8_t v61, int8_t v62, int8_t v63, int8_t v64) {
    return _mm512_set_epi8(v64,v63,v62,v61,v60,v59,v58,v57,v56,v55,v54,v53,v52,v51,v50,v49,v48,v47,v46,v45,v44,v43,v42,v41,v40,v39,v38,v37,v36,v35,v34,v33,v32,v31,v30,v29,v28,v27,v26,v25,v24,v23,v22,v21,v20,v19,v18,v17,v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int16_t,32> set_intrin<VecData<int16_t,32>,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t>(int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15, int16_t v16, int16_t v17, int16_t v18, int16_t v19, int16_t v20, int16_t v21, int16_t v22, int16_t v23, int16_t v24, int16_t v25, int16_t v26, int16_t v27, int16_t v28, int16_t v29, int16_t v30, int16_t v31, int16_t v32) {
    return _mm512_set_epi16(v32,v31,v30,v29,v28,v27,v26,v25,v24,v23,v22,v21,v20,v19,v18,v17,v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int32_t,16> set_intrin<VecData<int32_t,16>,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t>(int32_t v1, int32_t v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, int32_t v12, int32_t v13, int32_t v14, int32_t v15, int32_t v16) {
    return _mm512_set_epi32(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int64_t,8> set_intrin<VecData<int64_t,8>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t>(int64_t v1, int64_t v2, int64_t v3, int64_t v4, int64_t v5, int64_t v6, int64_t v7, int64_t v8) {
    return _mm512_set_epi64(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<float,16> set_intrin<VecData<float,16>,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16) {
    return _mm512_set_ps(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<double,8> set_intrin<VecData<double,8>,double,double,double,double,double,double,double,double>(double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8) {
    return _mm512_set_pd(v8,v7,v6,v5,v4,v3,v2,v1);
  }

  template <> inline VecData<int8_t,64> load1_intrin<VecData<int8_t,64>>(int8_t const* p) {
    return _mm512_set1_epi8(p[0]);
  }
  template <> inline VecData<int16_t,32> load1_intrin<VecData<int16_t,32>>(int16_t const* p) {
    return _mm512_set1_epi16(p[0]);
  }
  template <> inline VecData<int32_t,16> load1_intrin<VecData<int32_t,16>>(int32_t const* p) {
    return _mm512_set1_epi32(p[0]);
  }
  template <> inline VecData<int64_t,8> load1_intrin<VecData<int64_t,8>>(int64_t const* p) {
    return _mm512_set1_epi64(p[0]);
  }
  template <> inline VecData<float,16> load1_intrin<VecData<float,16>>(float const* p) {
    return _mm512_set1_ps(p[0]);
  }
  template <> inline VecData<double,8> load1_intrin<VecData<double,8>>(double const* p) {
    return _mm512_set1_pd(p[0]);
  }

  template <> inline VecData<int8_t,64> loadu_intrin<VecData<int8_t,64>>(int8_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<int16_t,32> loadu_intrin<VecData<int16_t,32>>(int16_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<int32_t,16> loadu_intrin<VecData<int32_t,16>>(int32_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<int64_t,8> loadu_intrin<VecData<int64_t,8>>(int64_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<float,16> loadu_intrin<VecData<float,16>>(float const* p) {
    return _mm512_loadu_ps(p);
  }
  template <> inline VecData<double,8> loadu_intrin<VecData<double,8>>(double const* p) {
    return _mm512_loadu_pd(p);
  }

  template <> inline VecData<int8_t,64> load_intrin<VecData<int8_t,64>>(int8_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<int16_t,32> load_intrin<VecData<int16_t,32>>(int16_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<int32_t,16> load_intrin<VecData<int32_t,16>>(int32_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<int64_t,8> load_intrin<VecData<int64_t,8>>(int64_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<float,16> load_intrin<VecData<float,16>>(float const* p) {
    return _mm512_load_ps(p);
  }
  template <> inline VecData<double,8> load_intrin<VecData<double,8>>(double const* p) {
    return _mm512_load_pd(p);
  }

  template <> inline void storeu_intrin<VecData<int8_t,64>>(int8_t* p, VecData<int8_t,64> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int16_t,32>>(int16_t* p, VecData<int16_t,32> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int32_t,16>>(int32_t* p, VecData<int32_t,16> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int64_t,8>>(int64_t* p, VecData<int64_t,8> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<float,16>>(float* p, VecData<float,16> vec) {
    _mm512_storeu_ps(p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<double,8>>(double* p, VecData<double,8> vec) {
    _mm512_storeu_pd(p, vec.v);
  }

  template <> inline void store_intrin<VecData<int8_t,64>>(int8_t* p, VecData<int8_t,64> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int16_t,32>>(int16_t* p, VecData<int16_t,32> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int32_t,16>>(int32_t* p, VecData<int32_t,16> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int64_t,8>>(int64_t* p, VecData<int64_t,8> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<float,16>>(float* p, VecData<float,16> vec) {
    _mm512_store_ps(p, vec.v);
  }
  template <> inline void store_intrin<VecData<double,8>>(double* p, VecData<double,8> vec) {
    _mm512_store_pd(p, vec.v);
  }

  //template <> inline int8_t extract_intrin<VecData<int8_t,64>>(VecData<int8_t,64> vec, Integer i) {}
  //template <> inline int16_t extract_intrin<VecData<int16_t,32>>(VecData<int16_t,32> vec, Integer i) {}
  //template <> inline int32_t extract_intrin<VecData<int32_t,16>>(VecData<int32_t,16> vec, Integer i) {}
  //template <> inline int64_t extract_intrin<VecData<int64_t,8>>(VecData<int64_t,8> vec, Integer i) {}
  //template <> inline float extract_intrin<VecData<float,16>>(VecData<float,16> vec, Integer i) {}
  //template <> inline double extract_intrin<VecData<double,8>>(VecData<double,8> vec, Integer i) {}

  //template <> inline void insert_intrin<VecData<int8_t,64>>(VecData<int8_t,64>& vec, Integer i, int8_t value) {}
  //template <> inline void insert_intrin<VecData<int16_t,32>>(VecData<int16_t,32>& vec, Integer i, int16_t value) {}
  //template <> inline void insert_intrin<VecData<int32_t,16>>(VecData<int32_t,16>& vec, Integer i, int32_t value) {}
  //template <> inline void insert_intrin<VecData<int64_t,8>>(VecData<int64_t,8>& vec, Integer i, int64_t value) {}
  //template <> inline void insert_intrin<VecData<float,16>>(VecData<float,16>& vec, Integer i, float value) {}
  //template <> inline void insert_intrin<VecData<double,8>>(VecData<double,8>& vec, Integer i, double value) {}

  // Arithmetic operators
  //template <> inline VecData<int8_t,64> unary_minus_intrin<VecData<int8_t,64>>(const VecData<int8_t,64>& a) {}
  //template <> inline VecData<int16_t,32> unary_minus_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) {}
  template <> inline VecData<int32_t,16> unary_minus_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) {
    return _mm512_sub_epi32(_mm512_setzero_epi32(), a.v);
  }
  template <> inline VecData<int64_t,8> unary_minus_intrin<VecData<int64_t,8>>(const VecData<int64_t,8>& a) {
    return _mm512_sub_epi64(_mm512_setzero_epi32(), a.v);
  }
  template <> inline VecData<float,16> unary_minus_intrin<VecData<float,16>>(const VecData<float,16>& a) {
    #ifdef __AVX512DQ__
    return _mm512_xor_ps(a.v, _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000)));
    #else
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(0x80000000)));
    #endif
  }
  template <> inline VecData<double,8> unary_minus_intrin<VecData<double,8>>(const VecData<double,8>& a) {
    #ifdef __AVX512DQ__
    return _mm512_xor_pd(a.v, _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000)));
    #else
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.v), _mm512_set1_epi64(0x8000000000000000)));
    #endif
  }

  //template <> inline VecData<int8_t,64> mul_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {}
  //template <> inline VecData<int16_t,32> mul_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {}
  template <> inline VecData<int32_t,16> mul_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_mullo_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> mul_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    #if defined(__AVX512DQ__)
    return _mm512_mullo_epi64(a.v, b.v);
    #elif defined (__INTEL_COMPILER)
    return _mm512_mullox_epi64(a.v, b.v);                  // _mm512_mullox_epi64 missing in gcc
    #else
    // instruction does not exist. Split into 32-bit multiplies
    //__m512i ahigh = _mm512_shuffle_epi32(a, 0xB1);       // swap H<->L
    __m512i ahigh   = _mm512_srli_epi64(a.v, 32);          // high 32 bits of each a
    __m512i bhigh   = _mm512_srli_epi64(b.v, 32);          // high 32 bits of each b
    __m512i prodahb = _mm512_mul_epu32(ahigh, b.v);        // ahigh*b
    __m512i prodbha = _mm512_mul_epu32(bhigh, a.v);        // bhigh*a
    __m512i prodhl  = _mm512_add_epi64(prodahb, prodbha);  // sum of high*low products
    __m512i prodhi  = _mm512_slli_epi64(prodhl, 32);       // same, shifted high
    __m512i prodll  = _mm512_mul_epu32(a.v, b.v);          // alow*blow = 64 bit unsigned products
    __m512i prod    = _mm512_add_epi64(prodll, prodhi);    // low*low+(high*low)<<32
    return  prod;
    #endif
  }
  template <> inline VecData<float,16> mul_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_mul_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> mul_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_mul_pd(a.v, b.v);
  }

  template <> inline VecData<float,16> div_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_div_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> div_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_div_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,64> add_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {}
  //template <> inline VecData<int16_t,32> add_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {}
  template <> inline VecData<int32_t,16> add_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_add_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> add_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_add_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> add_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_add_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> add_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_add_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,64> sub_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {}
  //template <> inline VecData<int16_t,32> sub_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {}
  template <> inline VecData<int32_t,16> sub_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_sub_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> sub_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_sub_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> sub_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_sub_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> sub_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_sub_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,64> fma_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b, const VecData<int8_t,64>& c) {}
  //template <> inline VecData<int16_t,32> fma_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b, const VecData<int16_t,32>& c) {}
  //template <> inline VecData<int32_t,16> sub_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b, const VecData<int32_t,16>& c) {}
  //template <> inline VecData<int64_t,8> sub_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b, const VecData<int64_t,8>& c) {}
  template <> inline VecData<float,16> fma_intrin(const VecData<float,16>& a, const VecData<float,16>& b, const VecData<float,16>& c) {
    return _mm512_fmadd_ps(a.v, b.v, c.v);
  }
  template <> inline VecData<double,8> fma_intrin(const VecData<double,8>& a, const VecData<double,8>& b, const VecData<double,8>& c) {
    return _mm512_fmadd_pd(a.v, b.v, c.v);
  }

  // Bitwise operators
  template <> inline VecData<int8_t,64> not_intrin<VecData<int8_t,64>>(const VecData<int8_t,64>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<int16_t,32> not_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<int32_t,16> not_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<int64_t,8> not_intrin<VecData<int64_t,8>>(const VecData<int64_t,8>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<float,16> not_intrin<VecData<float,16>>(const VecData<float,16>& a) {
    #ifdef __AVX512DQ__
    return _mm512_xor_ps(a.v, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
    #else
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(-1)));
    #endif
  }
  template <> inline VecData<double,8> not_intrin<VecData<double,8>>(const VecData<double,8>& a) {
    #ifdef __AVX512DQ__
    return _mm512_xor_pd(a.v, _mm512_castsi512_pd(_mm512_set1_epi32(-1)));
    #else
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.v), _mm512_set1_epi32(-1)));
    #endif
  }

  template <> inline VecData<int8_t,64> and_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> and_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> and_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> and_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<float,16> and_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    #ifdef __AVX512DQ__
    return _mm512_and_ps(a.v, b.v);
    #else
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a.v), _mm512_castps_si512(b.v)));
    #endif
  }
  template <> inline VecData<double,8> and_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    #ifdef __AVX512DQ__
    return _mm512_and_pd(a.v, b.v);
    #else
    return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a.v), _mm512_castpd_si512(b.v)));
    #endif
  }

  template <> inline VecData<int8_t,64> xor_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> xor_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> xor_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> xor_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<float,16> xor_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    #ifdef __AVX512DQ__
    return _mm512_xor_ps(a.v, b.v);
    #else
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_castps_si512(b.v)));
    #endif
  }
  template <> inline VecData<double,8> xor_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    #ifdef __AVX512DQ__
    return _mm512_xor_pd(a.v, b.v);
    #else
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.v), _mm512_castpd_si512(b.v)));
    #endif
  }

  template <> inline VecData<int8_t,64> or_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> or_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> or_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> or_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<float,16> or_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    #ifdef __AVX512DQ__
    return _mm512_or_ps(a.v, b.v);
    #else
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a.v), _mm512_castps_si512(b.v)));
    #endif
  }
  template <> inline VecData<double,8> or_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    #ifdef __AVX512DQ__
    return _mm512_or_pd(a.v, b.v);
    #else
    return _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(a.v), _mm512_castpd_si512(b.v)));
    #endif
  }

  template <> inline VecData<int8_t,64> andnot_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<int16_t,32> andnot_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<int32_t,16> andnot_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<int64_t,8> andnot_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<float,16> andnot_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    #ifdef __AVX512DQ__
    return _mm512_andnot_ps(b.v, a.v);
    #else
    return _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(b.v), _mm512_castps_si512(a.v)));
    #endif
  }
  template <> inline VecData<double,8> andnot_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    #ifdef __AVX512DQ__
    return _mm512_andnot_pd(b.v, a.v);
    #else
    return _mm512_castsi512_pd(_mm512_andnot_si512(_mm512_castpd_si512(b.v), _mm512_castpd_si512(a.v)));
    #endif
  }

  // Bitshift
  #if defined(__AVX512BW__)
  template <> inline VecData<int8_t ,64> bitshiftleft_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a, const Integer& rhs) { // 16-bit bit shift, then the bits from the next byte cleared
    return _mm512_and_si512(_mm512_slli_epi16(a.v, (unsigned int)rhs), _mm512_set1_epi8((char)(rhs < 8 ? (0xFF << rhs) & 0xFF : 0)));
  }
  #endif
  template <> inline VecData<int16_t,32> bitshiftleft_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a, const Integer& rhs) { return _mm512_slli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,16> bitshiftleft_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a, const Integer& rhs) { return _mm512_slli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,8> bitshiftleft_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a, const Integer& rhs) { return _mm512_slli_epi64(a.v , rhs); }
  template <> inline VecData<float  ,16> bitshiftleft_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a, const Integer& rhs) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(a.v), rhs)); }
  template <> inline VecData<double  ,8> bitshiftleft_intrin<VecData<double  ,8>>(const VecData<double  ,8>& a, const Integer& rhs) { return _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_castpd_si512(a.v), rhs)); }

  #if defined(__AVX512BW__)
  template <> inline VecData<int8_t ,64> bitshiftright_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a, const Integer& rhs) { // logical 16-bit bit shift, the bits from the next byte cleared, then the sign extended: (u ^ m) - m
    const unsigned int n = (unsigned int)(rhs < 7 ? rhs : 7); // larger bit shifts also give 0 or -1
    const __m512i u = _mm512_and_si512(_mm512_srli_epi16(a.v, n), _mm512_set1_epi8((char)(0xFF >> n)));
    const __m512i m = _mm512_set1_epi8((char)(0x80 >> n));
    return _mm512_sub_epi8(_mm512_xor_si512(u, m), m);
  }
  #endif
  template <> inline VecData<int16_t,32> bitshiftright_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a, const Integer& rhs) { return _mm512_srai_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,16> bitshiftright_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a, const Integer& rhs) { return _mm512_srai_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,8> bitshiftright_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a, const Integer& rhs) { return _mm512_srai_epi64(a.v , rhs); }
  template <> inline VecData<float  ,16> bitshiftright_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a, const Integer& rhs) { return _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(a.v), rhs)); }
  template <> inline VecData<double  ,8> bitshiftright_intrin<VecData<double  ,8>>(const VecData<double  ,8>& a, const Integer& rhs) { return _mm512_castsi512_pd(_mm512_srli_epi64(_mm512_castpd_si512(a.v), rhs)); }

  template <> inline VecData<int32_t,16> bitshiftleft_intrin <VecData<int32_t,16>>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& rhs) { return _mm512_sllv_epi32(a.v, rhs.v); }
  template <> inline VecData<int64_t ,8> bitshiftleft_intrin <VecData<int64_t ,8>>(const VecData<int64_t ,8>& a, const VecData<int64_t ,8>& rhs) { return _mm512_sllv_epi64(a.v, rhs.v); }
  template <> inline VecData<int32_t,16> bitshiftright_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& rhs) { return _mm512_srav_epi32(a.v, rhs.v); }
  template <> inline VecData<int64_t ,8> bitshiftright_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a, const VecData<int64_t ,8>& rhs) { return _mm512_srav_epi64(a.v, rhs.v); }
#if defined(__AVX512BW__)
  template <> inline VecData<int16_t,32> bitshiftleft_intrin <VecData<int16_t,32>>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& rhs) { return _mm512_sllv_epi16(a.v, rhs.v); }
  template <> inline VecData<int16_t,32> bitshiftright_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& rhs) { return _mm512_srav_epi16(a.v, rhs.v); }
#endif

  // Other functions
#if defined(__AVX512BW__)
  template <> inline VecData<int8_t,64> max_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_max_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> max_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_max_epi16(a.v, b.v);
  }
#endif
  template <> inline VecData<int32_t,16> max_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_max_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> max_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_max_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> max_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_max_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> max_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_max_pd(a.v, b.v);
  }

#if defined(__AVX512BW__)
  template <> inline VecData<int8_t,64> min_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_min_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> min_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_min_epi16(a.v, b.v);
  }
#endif
  template <> inline VecData<int32_t,16> min_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_min_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> min_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_min_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> min_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_min_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> min_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_min_pd(a.v, b.v);
  }

  template <> inline void transpose_intrin<VecData<double,8>>(VecData<double,8> (&v)[8]) {
    const __m512d r0 = v[0].v, r1 = v[1].v, r2 = v[2].v, r3 = v[3].v;
    const __m512d r4 = v[4].v, r5 = v[5].v, r6 = v[6].v, r7 = v[7].v;

    // interleave within 128-bit lanes, forming 2x2 blocks
    const __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    const __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    const __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    const __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    const __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    const __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    const __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    const __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    // 0x88/0xDD select the low/high 128-bit lane of each operand, forming 4x4 blocks
    const __m512d s0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    const __m512d s1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    const __m512d s2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    const __m512d s3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    const __m512d s4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    const __m512d s5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    const __m512d s6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    const __m512d s7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

    v[0].v = _mm512_shuffle_f64x2(s0, s4, 0x88);
    v[1].v = _mm512_shuffle_f64x2(s1, s5, 0x88);
    v[2].v = _mm512_shuffle_f64x2(s2, s6, 0x88);
    v[3].v = _mm512_shuffle_f64x2(s3, s7, 0x88);
    v[4].v = _mm512_shuffle_f64x2(s0, s4, 0xDD);
    v[5].v = _mm512_shuffle_f64x2(s1, s5, 0xDD);
    v[6].v = _mm512_shuffle_f64x2(s2, s6, 0xDD);
    v[7].v = _mm512_shuffle_f64x2(s3, s7, 0xDD);
  }
  template <> inline void transpose_intrin<VecData<float,16>>(VecData<float,16> (&v)[16]) {
    const __m512 r0 = v[ 0].v, r1 = v[ 1].v, r2  = v[ 2].v, r3  = v[ 3].v;
    const __m512 r4 = v[ 4].v, r5 = v[ 5].v, r6  = v[ 6].v, r7  = v[ 7].v;
    const __m512 r8 = v[ 8].v, r9 = v[ 9].v, r10 = v[10].v, r11 = v[11].v;
    const __m512 r12= v[12].v, r13= v[13].v, r14 = v[14].v, r15 = v[15].v;

    // interleave 32-bit elements within 128-bit lanes
    const __m512 t0  = _mm512_unpacklo_ps(r0 , r1 ), t1  = _mm512_unpackhi_ps(r0 , r1 );
    const __m512 t2  = _mm512_unpacklo_ps(r2 , r3 ), t3  = _mm512_unpackhi_ps(r2 , r3 );
    const __m512 t4  = _mm512_unpacklo_ps(r4 , r5 ), t5  = _mm512_unpackhi_ps(r4 , r5 );
    const __m512 t6  = _mm512_unpacklo_ps(r6 , r7 ), t7  = _mm512_unpackhi_ps(r6 , r7 );
    const __m512 t8  = _mm512_unpacklo_ps(r8 , r9 ), t9  = _mm512_unpackhi_ps(r8 , r9 );
    const __m512 t10 = _mm512_unpacklo_ps(r10, r11), t11 = _mm512_unpackhi_ps(r10, r11);
    const __m512 t12 = _mm512_unpacklo_ps(r12, r13), t13 = _mm512_unpackhi_ps(r12, r13);
    const __m512 t14 = _mm512_unpacklo_ps(r14, r15), t15 = _mm512_unpackhi_ps(r14, r15);

    // gather 2-wide groups into 4-wide, still within 128-bit lanes
    const __m512 s0  = _mm512_shuffle_ps(t0 , t2 , 0x44), s1  = _mm512_shuffle_ps(t0 , t2 , 0xEE);
    const __m512 s2  = _mm512_shuffle_ps(t1 , t3 , 0x44), s3  = _mm512_shuffle_ps(t1 , t3 , 0xEE);
    const __m512 s4  = _mm512_shuffle_ps(t4 , t6 , 0x44), s5  = _mm512_shuffle_ps(t4 , t6 , 0xEE);
    const __m512 s6  = _mm512_shuffle_ps(t5 , t7 , 0x44), s7  = _mm512_shuffle_ps(t5 , t7 , 0xEE);
    const __m512 s8  = _mm512_shuffle_ps(t8 , t10, 0x44), s9  = _mm512_shuffle_ps(t8 , t10, 0xEE);
    const __m512 s10 = _mm512_shuffle_ps(t9 , t11, 0x44), s11 = _mm512_shuffle_ps(t9 , t11, 0xEE);
    const __m512 s12 = _mm512_shuffle_ps(t12, t14, 0x44), s13 = _mm512_shuffle_ps(t12, t14, 0xEE);
    const __m512 s14 = _mm512_shuffle_ps(t13, t15, 0x44), s15 = _mm512_shuffle_ps(t13, t15, 0xEE);

    // move 128-bit lanes across the 512-bit vectors
    const __m512 u0  = _mm512_shuffle_f32x4(s0 , s4 , 0x88);
    const __m512 u1  = _mm512_shuffle_f32x4(s1 , s5 , 0x88);
    const __m512 u2  = _mm512_shuffle_f32x4(s2 , s6 , 0x88);
    const __m512 u3  = _mm512_shuffle_f32x4(s3 , s7 , 0x88);
    const __m512 u4  = _mm512_shuffle_f32x4(s0 , s4 , 0xDD);
    const __m512 u5  = _mm512_shuffle_f32x4(s1 , s5 , 0xDD);
    const __m512 u6  = _mm512_shuffle_f32x4(s2 , s6 , 0xDD);
    const __m512 u7  = _mm512_shuffle_f32x4(s3 , s7 , 0xDD);
    const __m512 u8  = _mm512_shuffle_f32x4(s8 , s12, 0x88);
    const __m512 u9  = _mm512_shuffle_f32x4(s9 , s13, 0x88);
    const __m512 u10 = _mm512_shuffle_f32x4(s10, s14, 0x88);
    const __m512 u11 = _mm512_shuffle_f32x4(s11, s15, 0x88);
    const __m512 u12 = _mm512_shuffle_f32x4(s8 , s12, 0xDD);
    const __m512 u13 = _mm512_shuffle_f32x4(s9 , s13, 0xDD);
    const __m512 u14 = _mm512_shuffle_f32x4(s10, s14, 0xDD);
    const __m512 u15 = _mm512_shuffle_f32x4(s11, s15, 0xDD);

    v[ 0].v = _mm512_shuffle_f32x4(u0 , u8 , 0x88);
    v[ 1].v = _mm512_shuffle_f32x4(u1 , u9 , 0x88);
    v[ 2].v = _mm512_shuffle_f32x4(u2 , u10, 0x88);
    v[ 3].v = _mm512_shuffle_f32x4(u3 , u11, 0x88);
    v[ 4].v = _mm512_shuffle_f32x4(u4 , u12, 0x88);
    v[ 5].v = _mm512_shuffle_f32x4(u5 , u13, 0x88);
    v[ 6].v = _mm512_shuffle_f32x4(u6 , u14, 0x88);
    v[ 7].v = _mm512_shuffle_f32x4(u7 , u15, 0x88);
    v[ 8].v = _mm512_shuffle_f32x4(u0 , u8 , 0xDD);
    v[ 9].v = _mm512_shuffle_f32x4(u1 , u9 , 0xDD);
    v[10].v = _mm512_shuffle_f32x4(u2 , u10, 0xDD);
    v[11].v = _mm512_shuffle_f32x4(u3 , u11, 0xDD);
    v[12].v = _mm512_shuffle_f32x4(u4 , u12, 0xDD);
    v[13].v = _mm512_shuffle_f32x4(u5 , u13, 0xDD);
    v[14].v = _mm512_shuffle_f32x4(u6 , u14, 0xDD);
    v[15].v = _mm512_shuffle_f32x4(u7 , u15, 0xDD);
  }
  template <> inline void transpose_intrin<VecData<int32_t,16>>(VecData<int32_t,16> (&v)[16]) { transpose_reinterpret_intrin<VecData<float ,16>>(v); }
  template <> inline void transpose_intrin<VecData<int64_t, 8>>(VecData<int64_t, 8> (&v)[ 8]) { transpose_reinterpret_intrin<VecData<double, 8>>(v); }

  // The rows transposed and evaluated by Estrin's scheme together: one step of the scheme after each stage of the
  // transpose, q_k = c_2k + c_2k+1 t after the first, s_k = q_2k + q_2k+1 t^2 after the second.
  template <> inline VecData<double,8> eval_poly_rows_intrin<8, VecData<double,8>>(const double* const (&p)[8], const VecData<double,8>& t) {
    __m512d q[4]; // (q_0 of lanes 2j, 2j+1, q_1 of the two lanes, ..., q_3 of the two lanes)
    for (Integer j = 0; j < 4; j++) {
      const __m512d ra = _mm512_loadu_pd(p[2 * j]);
      const __m512d rb = _mm512_loadu_pd(p[2 * j + 1]);
      const __m512d tj = _mm512_permutexvar_pd(_mm512_setr_epi64(2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1), t.v);
      q[j] = _mm512_fmadd_pd(_mm512_unpackhi_pd(ra, rb), tj, _mm512_unpacklo_pd(ra, rb));
    }
    const __m512d t2 = _mm512_mul_pd(t.v, t.v);
    __m512d s[2]; // (s_0, s_1 of lanes 4j, 4j+1, then s_0, s_1 of lanes 4j+2, 4j+3), each pair of lanes in a 128-bit lane
    for (Integer j = 0; j < 2; j++) {
      const __m512d tj = _mm512_permutexvar_pd(_mm512_setr_epi64(4 * j, 4 * j + 1, 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3, 4 * j + 2, 4 * j + 3), t2);
      s[j] = _mm512_fmadd_pd(_mm512_shuffle_f64x2(q[2 * j], q[2 * j + 1], 0xDD), tj, _mm512_shuffle_f64x2(q[2 * j], q[2 * j + 1], 0x88));
    }
    return _mm512_fmadd_pd(_mm512_shuffle_f64x2(s[0], s[1], 0xDD), _mm512_mul_pd(t2, t2), _mm512_shuffle_f64x2(s[0], s[1], 0x88));
  }
  // Rows of 8 values: the 256-bit halves hold lanes 0..7 and 8..15, each as in the 8-lane kernel for AVX.
  template <> inline VecData<float,16> eval_poly_rows_intrin<8, VecData<float,16>>(const float* const (&p)[16], const VecData<float,16>& t) {
    const auto rows = [&p](const Integer l) { // the rows of lanes l and l + 8
      return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(_mm256_loadu_ps(p[l]))), _mm256_castps_pd(_mm256_loadu_ps(p[l + 8])), 1));
    };
    __m512 q[4]; // in each half: (q_0, q_1 of lane 2j, q_0, q_1 of lane 2j+1, then q_2, q_3 of the two lanes)
    for (Integer j = 0; j < 4; j++) {
      const int a = (int)(2 * j);
      const __m512 ra = rows(a);
      const __m512 rb = rows(a + 1);
      const __m512 tj = _mm512_permutexvar_ps(_mm512_setr_epi32(a, a, a + 1, a + 1, a, a, a + 1, a + 1, a + 8, a + 8, a + 9, a + 9, a + 8, a + 8, a + 9, a + 9), t.v);
      q[j] = _mm512_fmadd_ps(_mm512_shuffle_ps(ra, rb, _MM_SHUFFLE(3,1,3,1)), tj, _mm512_shuffle_ps(ra, rb, _MM_SHUFFLE(2,0,2,0)));
    }
    const __m512 t2 = _mm512_mul_ps(t.v, t.v);
    __m512 s[2]; // in each half: (s_0 of lanes 4j..4j+3, s_1 of lanes 4j..4j+3)
    for (Integer j = 0; j < 2; j++) {
      const int a = (int)(4 * j);
      const __m512 tj = _mm512_permutexvar_ps(_mm512_setr_epi32(a, a + 1, a + 2, a + 3, a, a + 1, a + 2, a + 3, a + 8, a + 9, a + 10, a + 11, a + 8, a + 9, a + 10, a + 11), t2);
      s[j] = _mm512_fmadd_ps(_mm512_shuffle_ps(q[2 * j], q[2 * j + 1], _MM_SHUFFLE(3,1,3,1)), tj, _mm512_shuffle_ps(q[2 * j], q[2 * j + 1], _MM_SHUFFLE(2,0,2,0)));
    }
    const __m512i even = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27); // s_0 of lanes 0..3, 4..7, 8..11, 12..15
    const __m512i odd = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
    return _mm512_fmadd_ps(_mm512_permutex2var_ps(s[0], odd, s[1]), _mm512_mul_ps(t2, t2), _mm512_permutex2var_ps(s[0], even, s[1]));
  }

  template <> inline VecData<float,16> swap_pairs_intrin(const VecData<float,16>& vec) { return _mm512_permute_ps(vec.v, 0xB1); }
  template <> inline VecData<double,8> swap_pairs_intrin(const VecData<double,8>& vec) { return _mm512_permute_pd(vec.v, 0x55); }

#if defined(__AVX512BW__)
  template <Integer Bits> struct UnpackIntrin512;
  template <> struct UnpackIntrin512< 8> {
    static inline __m512i lo(const __m512i& a, const __m512i& b) { return _mm512_unpacklo_epi8 (a, b); }
    static inline __m512i hi(const __m512i& a, const __m512i& b) { return _mm512_unpackhi_epi8 (a, b); }
  };
  template <> struct UnpackIntrin512<16> {
    static inline __m512i lo(const __m512i& a, const __m512i& b) { return _mm512_unpacklo_epi16(a, b); }
    static inline __m512i hi(const __m512i& a, const __m512i& b) { return _mm512_unpackhi_epi16(a, b); }
  };
  template <> struct UnpackIntrin512<32> {
    static inline __m512i lo(const __m512i& a, const __m512i& b) { return _mm512_unpacklo_epi32(a, b); }
    static inline __m512i hi(const __m512i& a, const __m512i& b) { return _mm512_unpackhi_epi32(a, b); }
  };
  template <> struct UnpackIntrin512<64> {
    static inline __m512i lo(const __m512i& a, const __m512i& b) { return _mm512_unpacklo_epi64(a, b); }
    static inline __m512i hi(const __m512i& a, const __m512i& b) { return _mm512_unpackhi_epi64(a, b); }
  };

  template <Integer M, Integer Bits, Integer Stride> struct TransposeNet512 {
    static inline void apply(__m512i* v) {
      __m512i w[M];
      for (Integer k = 0; k < M; k += 2*Stride) {
        for (Integer j = 0; j < Stride; j++) {
          w[k+2*j+0] = UnpackIntrin512<Bits>::lo(v[k+j], v[k+j+Stride]);
          w[k+2*j+1] = UnpackIntrin512<Bits>::hi(v[k+j], v[k+j+Stride]);
        }
      }
      for (Integer i = 0; i < M; i++) v[i] = w[i];
      TransposeNet512<M,Bits*2,Stride*2>::apply(v);
    }
  };
  template <Integer M, Integer Bits> struct TransposeNet512<M,Bits,M> {
    static inline void apply(__m512i*) {}
  };

  // A 512-bit register is four 128-bit lanes, so the lane-wise networks leave a
  // 4x4 block matrix to transpose; that takes two shuffle_i32x4 stages.
  template <class VData> inline void transpose512_intrin(VData (&v)[VData::Size]) {
    static constexpr Integer N = VData::Size;
    static constexpr Integer Q = N/4;
    static constexpr Integer W = sizeof(typename VData::ScalarType)*8;
    __m512i w[N];
    for (Integer i = 0; i < N; i++) w[i] = v[i].v;
    for (Integer g = 0; g < 4; g++) TransposeNet512<Q,W,1>::apply(w + g*Q);
    for (Integer i = 0; i < Q; i++) {
      const __m512i a0 = w[i], a1 = w[Q+i], a2 = w[2*Q+i], a3 = w[3*Q+i];
      const __m512i t0 = _mm512_shuffle_i32x4(a0, a1, 0x88);
      const __m512i t1 = _mm512_shuffle_i32x4(a2, a3, 0x88);
      const __m512i t2 = _mm512_shuffle_i32x4(a0, a1, 0xDD);
      const __m512i t3 = _mm512_shuffle_i32x4(a2, a3, 0xDD);
      v[      i].v = _mm512_shuffle_i32x4(t0, t1, 0x88);
      v[  Q + i].v = _mm512_shuffle_i32x4(t2, t3, 0x88);
      v[2*Q + i].v = _mm512_shuffle_i32x4(t0, t1, 0xDD);
      v[3*Q + i].v = _mm512_shuffle_i32x4(t2, t3, 0xDD);
    }
  }
  template <> inline void transpose_intrin<VecData<int16_t,32>>(VecData<int16_t,32> (&v)[32]) { transpose512_intrin(v); }
  template <> inline void transpose_intrin<VecData<int8_t ,64>>(VecData<int8_t ,64> (&v)[64]) { transpose512_intrin(v); }
#endif

  // Conversion operators
  template <> inline VecData<float,16> convert_int2real_intrin<VecData<float,16>,VecData<int32_t,16>>(const VecData<int32_t,16>& x) { return _mm512_cvtepi32_ps(x.v); }
  template <> inline VecData<int32_t,16> lrint_intrin<VecData<int32_t,16>,VecData<float,16>>(const VecData<float,16>& x) { return _mm512_cvtps_epi32(x.v); }
#if defined(__AVX512DQ__)
  template <> inline VecData<double,8> convert_int2real_intrin<VecData<double,8>,VecData<int64_t, 8>>(const VecData<int64_t, 8>& x) { return _mm512_cvtepi64_pd(x.v); }
  template <> inline VecData<int64_t, 8> lrint_intrin<VecData<int64_t, 8>,VecData<double,8>>(const VecData<double,8>& x) { return _mm512_cvtpd_epi64(x.v); }
#endif
  template <> inline VecData<float,16> rint_intrin<VecData<float,16>>(const VecData<float,16>& x) { return _mm512_roundscale_ps(x.v, _MM_FROUND_TO_NEAREST_INT); }
  template <> inline VecData<double,8> rint_intrin<VecData<double,8>>(const VecData<double,8>& x) { return _mm512_roundscale_pd(x.v, _MM_FROUND_TO_NEAREST_INT); }

  template <> inline VecData<double ,8> convert_intrin<VecData<double ,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& a) { return _mm512_cvtepi32_pd (a.v); }
  template <> inline VecData<int32_t,8> convert_intrin<VecData<int32_t,8>,VecData<double ,8>>(const VecData<double ,8>& a) { return _mm512_cvttpd_epi32(a.v); }
  template <> inline VecData<double ,8> convert_intrin<VecData<double ,8>,VecData<float  ,8>>(const VecData<float  ,8>& a) { return _mm512_cvtps_pd    (a.v); }
  template <> inline VecData<float  ,8> convert_intrin<VecData<float  ,8>,VecData<double ,8>>(const VecData<double ,8>& a) { return _mm512_cvtpd_ps    (a.v); }
  template <> inline VecData<float ,16> convert_intrin<VecData<float ,16>,VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return _mm512_cvtepi32_ps (a.v); }
  template <> inline VecData<int32_t,16> convert_intrin<VecData<int32_t,16>,VecData<float ,16>>(const VecData<float ,16>& a) { return _mm512_cvttps_epi32(a.v); }
  // integers: sign extension, or the low bits
  template <> inline VecData<int32_t,16> convert_intrin<VecData<int32_t,16>,VecData<int8_t ,16>>(const VecData<int8_t ,16>& a) { return _mm512_cvtepi8_epi32(a.v); }
  template <> inline VecData<int8_t ,16> convert_intrin<VecData<int8_t ,16>,VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return _mm512_cvtepi32_epi8(a.v); }
  template <> inline VecData<int32_t,16> convert_intrin<VecData<int32_t,16>,VecData<int16_t,16>>(const VecData<int16_t,16>& a) { return _mm512_cvtepi16_epi32(a.v); }
  template <> inline VecData<int16_t,16> convert_intrin<VecData<int16_t,16>,VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return _mm512_cvtepi32_epi16(a.v); }
  template <> inline VecData<int64_t,8> convert_intrin<VecData<int64_t,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& a) { return _mm512_cvtepi32_epi64(a.v); }
  template <> inline VecData<int32_t,8> convert_intrin<VecData<int32_t,8>,VecData<int64_t,8>>(const VecData<int64_t,8>& a) { return _mm512_cvtepi64_epi32(a.v); }
  template <> inline VecData<int64_t,8> convert_intrin<VecData<int64_t,8>,VecData<int16_t,8>>(const VecData<int16_t,8>& a) { return _mm512_cvtepi16_epi64(a.v); }
  template <> inline VecData<int16_t,8> convert_intrin<VecData<int16_t,8>,VecData<int64_t,8>>(const VecData<int64_t,8>& a) { return _mm512_cvtepi64_epi16(a.v); }
  template <> inline VecData<float ,16> convert_intrin<VecData<float ,16>,VecData<int16_t,16>>(const VecData<int16_t,16>& a) { return _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a.v)); }
  template <> inline VecData<int16_t,16> convert_intrin<VecData<int16_t,16>,VecData<float ,16>>(const VecData<float ,16>& a) { return _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(a.v)); }
  template <> inline VecData<float ,16> convert_intrin<VecData<float ,16>,VecData<int8_t ,16>>(const VecData<int8_t ,16>& a) { return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a.v)); }
  template <> inline VecData<int8_t ,16> convert_intrin<VecData<int8_t ,16>,VecData<float ,16>>(const VecData<float ,16>& a) { return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(a.v)); }
  template <> inline VecData<double ,8> convert_intrin<VecData<double ,8>,VecData<int16_t,8>>(const VecData<int16_t,8>& a) { return _mm512_cvtepi32_pd(_mm256_cvtepi16_epi32(a.v)); }
  #if defined(__AVX512BW__)
  template <> inline VecData<int16_t,32> convert_intrin<VecData<int16_t,32>,VecData<int8_t ,32>>(const VecData<int8_t ,32>& a) { return _mm512_cvtepi8_epi16(a.v); }
  template <> inline VecData<int8_t ,32> convert_intrin<VecData<int8_t ,32>,VecData<int16_t,32>>(const VecData<int16_t,32>& a) { return _mm512_cvtepi16_epi8(a.v); }
  #endif
  #if defined(__AVX512VL__)
  template <> inline VecData<int16_t,8> convert_intrin<VecData<int16_t,8>,VecData<double ,8>>(const VecData<double ,8>& a) { return _mm256_cvtepi32_epi16(_mm512_cvttpd_epi32(a.v)); }
  #endif
  #if defined(__AVX512DQ__)
  template <> inline VecData<float  ,8> convert_intrin<VecData<float  ,8>,VecData<int64_t,8>>(const VecData<int64_t,8>& a) { return _mm512_cvtepi64_ps(a.v); }
  template <> inline VecData<int64_t,8> convert_intrin<VecData<int64_t,8>,VecData<float  ,8>>(const VecData<float  ,8>& a) { return _mm512_cvttps_epi64(a.v); }
  #endif
#if defined(__AVX512DQ__)
  template <> inline VecData<double ,8> convert_intrin<VecData<double ,8>,VecData<int64_t,8>>(const VecData<int64_t,8>& a) { return _mm512_cvtepi64_pd (a.v); }
  template <> inline VecData<int64_t,8> convert_intrin<VecData<int64_t,8>,VecData<double ,8>>(const VecData<double ,8>& a) { return _mm512_cvttpd_epi64(a.v); }
#endif

  // Halves of a vector
  template <> inline VecData<int8_t ,32> get_low_intrin <VecData<int8_t ,64>>(const VecData<int8_t ,64>& a) { return _mm512_castsi512_si256(a.v); }
  template <> inline VecData<int16_t,16> get_low_intrin <VecData<int16_t,32>>(const VecData<int16_t,32>& a) { return _mm512_castsi512_si256(a.v); }
  template <> inline VecData<int32_t, 8> get_low_intrin <VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return _mm512_castsi512_si256(a.v); }
  template <> inline VecData<int64_t, 4> get_low_intrin <VecData<int64_t, 8>>(const VecData<int64_t, 8>& a) { return _mm512_castsi512_si256(a.v); }
  template <> inline VecData<float  , 8> get_low_intrin <VecData<float  ,16>>(const VecData<float  ,16>& a) { return _mm512_castps512_ps256(a.v); }
  template <> inline VecData<double , 4> get_low_intrin <VecData<double , 8>>(const VecData<double , 8>& a) { return _mm512_castpd512_pd256(a.v); }

  template <> inline VecData<int8_t ,32> get_high_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a) { return _mm512_extracti64x4_epi64(a.v, 1); }
  template <> inline VecData<int16_t,16> get_high_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) { return _mm512_extracti64x4_epi64(a.v, 1); }
  template <> inline VecData<int32_t, 8> get_high_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return _mm512_extracti64x4_epi64(a.v, 1); }
  template <> inline VecData<int64_t, 4> get_high_intrin<VecData<int64_t, 8>>(const VecData<int64_t, 8>& a) { return _mm512_extracti64x4_epi64(a.v, 1); }
  template <> inline VecData<float  , 8> get_high_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a) { return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.v), 1)); }
  template <> inline VecData<double , 4> get_high_intrin<VecData<double , 8>>(const VecData<double , 8>& a) { return _mm512_extractf64x4_pd(a.v, 1); }

  template <> inline VecData<int8_t ,64> concat_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& lo, const VecData<int8_t ,32>& hi) { return _mm512_inserti64x4(_mm512_castsi256_si512(lo.v), hi.v, 1); }
  template <> inline VecData<int16_t,32> concat_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& lo, const VecData<int16_t,16>& hi) { return _mm512_inserti64x4(_mm512_castsi256_si512(lo.v), hi.v, 1); }
  template <> inline VecData<int32_t,16> concat_intrin<VecData<int32_t, 8>>(const VecData<int32_t, 8>& lo, const VecData<int32_t, 8>& hi) { return _mm512_inserti64x4(_mm512_castsi256_si512(lo.v), hi.v, 1); }
  template <> inline VecData<int64_t, 8> concat_intrin<VecData<int64_t, 4>>(const VecData<int64_t, 4>& lo, const VecData<int64_t, 4>& hi) { return _mm512_inserti64x4(_mm512_castsi256_si512(lo.v), hi.v, 1); }
  template <> inline VecData<float  ,16> concat_intrin<VecData<float  , 8>>(const VecData<float  , 8>& lo, const VecData<float  , 8>& hi) { return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(lo.v)), _mm256_castps_pd(hi.v), 1)); }
  template <> inline VecData<double , 8> concat_intrin<VecData<double , 4>>(const VecData<double , 4>& lo, const VecData<double , 4>& hi) { return _mm512_insertf64x4(_mm512_castpd256_pd512(lo.v), hi.v, 1); }

  template <> inline VecData<int32_t,16> div_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return div_int32_intrin(a, b); }
  template <> inline VecData<int64_t,8> div_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return div_int64_intrin(a, b); }
  #if defined(__AVX512BW__)
  template <> inline VecData<int16_t,32> div_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { // through float, exact for 16 bits; the low 16 bits of the quotient, as the C++ conversion
    const auto q = [](const __m256i a16, const __m256i b16) { return _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a16)), _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(b16))))); };
    return _mm512_inserti64x4(_mm512_castsi256_si512(q(_mm512_castsi512_si256(a.v), _mm512_castsi512_si256(b.v))), q(_mm512_extracti64x4_epi64(a.v, 1), _mm512_extracti64x4_epi64(b.v, 1)), 1);
  }
  template <> inline VecData<int8_t,64> div_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { // through float, sixteen lanes at a time; the low 8 bits of the quotient
    const auto q = [](const __m128i a16, const __m128i b16) { return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a16)), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b16))))); };
    __m512i r = _mm512_castsi128_si512(q(_mm512_extracti32x4_epi32(a.v, 0), _mm512_extracti32x4_epi32(b.v, 0)));
    r = _mm512_inserti32x4(r, q(_mm512_extracti32x4_epi32(a.v, 1), _mm512_extracti32x4_epi32(b.v, 1)), 1);
    r = _mm512_inserti32x4(r, q(_mm512_extracti32x4_epi32(a.v, 2), _mm512_extracti32x4_epi32(b.v, 2)), 2);
    return _mm512_inserti32x4(r, q(_mm512_extracti32x4_epi32(a.v, 3), _mm512_extracti32x4_epi32(b.v, 3)), 3);
  }
  #endif


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512BW__)
  template <> struct Mask<VecData<int8_t ,64>> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 64;

    static inline Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const __mmask64& v_) : v(v_) {}

    __mmask64 v;
  };
  template <> struct Mask<VecData<int16_t,32>> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 32;

    static inline Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const __mmask32& v_) : v(v_) {}

    __mmask32 v;
  };
#endif

#if defined(__AVX512DQ__)
  template <> struct Mask<VecData<int32_t,16>> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 16;

    static inline Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const __mmask16& v_) : v(v_) {}

    __mmask16 v;
  };
  template <> struct Mask<VecData<int64_t ,8>> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 8;

    static inline Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const __mmask8& v_) : v(v_) {}

    __mmask8  v;
  };
  template <> struct Mask<VecData<float  ,16>> {
    using ScalarType = float;
    static constexpr Integer Size = 16;

    static inline Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const __mmask16& v_) : v(v_) {}

    __mmask16 v;
  };
  template <> struct Mask<VecData<double  ,8>> {
    using ScalarType = double;
    static constexpr Integer Size = 8;

    static inline Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    inline explicit Mask(const __mmask8& v_) : v(v_) {}

    __mmask8  v;
  };

  // GCC 12 to 13.3 and 14.0 to 14.2 spill a compare's mask with a 16-bit store and reload it as 32 bits
  // (GCC bug 117159); an explicit kmov to a general register keeps the zero-extension.
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER) && \
    (__GNUC__ == 12 || (__GNUC__ == 13 && __GNUC_MINOR__ < 4) || (__GNUC__ == 14 && __GNUC_MINOR__ < 3))
  inline unsigned mask_to_u32(__mmask16 k) {
    unsigned m;
    asm("kmovw %1, %0" : "=r"(m) : "k"(k));
    return m;
  }
  inline unsigned mask_to_u32(__mmask8 k) {
    unsigned m;
    asm("kmovb %1, %0" : "=r"(m) : "k"(k));
    return m;
  }
#else
  inline unsigned mask_to_u32(__mmask16 k) { return _cvtmask16_u32(k); }
  inline unsigned mask_to_u32(__mmask8 k) { return _cvtmask8_u32(k); }
#endif

  template <> inline Integer mask_popcnt_intrin<VecData<float, 16>>(const Mask<VecData<float, 16>>& v) { return (Integer)_mm_popcnt_u32(mask_to_u32(v.v)); }
  template <> inline Integer mask_popcnt_intrin<VecData<double, 8>>(const Mask<VecData<double, 8>>& v) { return (Integer)_mm_popcnt_u32(mask_to_u32(v.v)); }
  template <> inline bool mask_any<VecData<float, 16>>(const Mask<VecData<float, 16>>& v) { return mask_to_u32(v.v) != 0; }
  template <> inline bool mask_any<VecData<double, 8>>(const Mask<VecData<double, 8>>& v) { return mask_to_u32(v.v) != 0; }
  template <> inline void mask_compress_store<VecData<float, 16>>(const Mask<VecData<float, 16>>& mask, const VecData<float, 16>& v, float* ptr) { _mm512_mask_compressstoreu_ps(ptr, mask.v, v.v); }
  template <> inline void mask_compress_store<VecData<double, 8>>(const Mask<VecData<double, 8>>& mask, const VecData<double, 8>& v, double* ptr) { _mm512_mask_compressstoreu_pd(ptr, mask.v, v.v); }
  template <> inline Integer mask_compress_iota_store<VecData<float, 16>>(const Mask<VecData<float, 16>>& mask, int32_t base, int32_t* ptr) {
    const __m512i iota = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
    _mm512_mask_compressstoreu_epi32(ptr, mask.v, iota);
    return (Integer)_mm_popcnt_u32(mask_to_u32(mask.v));
  }
  template <> inline Integer mask_compress_iota_store<VecData<double, 8>>(const Mask<VecData<double, 8>>& mask, int32_t base, int32_t* ptr) {
    // 512-bit epi32 compress (needs no AVX512VL): low 8 lanes hold the iota, high 8 masked off.
    const __m512i iota = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_setr_epi32(0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0));
    _mm512_mask_compressstoreu_epi32(ptr, (__mmask16)mask.v, iota);
    return (Integer)_mm_popcnt_u32(mask_to_u32(mask.v));
  }
  template <> inline Integer mask_compress_iota_store2<VecData<double, 8>>(const Mask<VecData<double, 8>>& mask_lo, const Mask<VecData<double, 8>>& mask_hi, int32_t base, int32_t* ptr) {
    // Both 8-lane masks packed into one 16-lane int32 compress -> one vpcompressd for 16 sources.
    // kunpackb builds the fused mask once in a k-register (consumed directly by the compress and
    // by a single kmov for the popcount), avoiding a redundant GPR shift/or materialization.
    const __m512i iota = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
    const __mmask16 m = _mm512_kunpackb(mask_hi.v, mask_lo.v);
    _mm512_mask_compressstoreu_epi32(ptr, m, iota);
    return (Integer)_mm_popcnt_u32(mask_to_u32(m));
  }
  template <> inline float reduce_add_intrin<VecData<float, 16>>(const VecData<float, 16>& a) { return _mm512_reduce_add_ps(a.v); }
  template <> inline double reduce_add_intrin<VecData<double, 8>>(const VecData<double, 8>& a) { return _mm512_reduce_add_pd(a.v); }
  template <> inline VecData<float, 16> mask_expand_load<VecData<float, 16>>(const Mask<VecData<float, 16>> &mask, const VecData<float, 16> &zero, const float *ptr) {
    VecData<float, 16> result;
    result.v = _mm512_mask_expandloadu_ps(zero.v, mask.v, ptr);
    return result;
  }
  template <> inline VecData<double, 8> mask_expand_load<VecData<double, 8>>(const Mask<VecData<double, 8>>& mask, const VecData<double, 8>& zero, const double* ptr) {
    VecData<double, 8> result;
    result.v = _mm512_mask_expandloadu_pd(zero.v, mask.v, ptr);
    return result;
  }
#endif

  // Bitwise operators
#if defined(__AVX512BW__)
  template <> inline Mask<VecData<int8_t ,64>> operator~<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& vec) { return Mask<VecData<int8_t ,64>>(_knot_mask64(vec.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator~<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& vec) { return Mask<VecData<int16_t,32>>(_knot_mask32(vec.v)); }

  template <> inline Mask<VecData<int8_t ,64>> operator&<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kand_mask64(a.v,b.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator&<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kand_mask32(a.v,b.v)); }

  template <> inline Mask<VecData<int8_t ,64>> operator^<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kxor_mask64(a.v,b.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator^<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kxor_mask32(a.v,b.v)); }

  template <> inline Mask<VecData<int8_t ,64>> operator|<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kor_mask64(a.v,b.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator|<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kor_mask32(a.v,b.v)); }

  template <> inline Mask<VecData<int8_t ,64>> AndNot   <VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kandn_mask64(b.v,a.v)); }
  template <> inline Mask<VecData<int16_t,32>> AndNot   <VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kandn_mask32(b.v,a.v)); }

  template <> inline VecData<int8_t ,64> convert_mask2vec_intrin<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a) { return _mm512_movm_epi8 (a.v); }
  template <> inline VecData<int16_t,32> convert_mask2vec_intrin<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a) { return _mm512_movm_epi16(a.v); }

  template <> inline Mask<VecData<int8_t ,64>> convert_vec2mask_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a) { return Mask<VecData<int8_t ,64>>(_mm512_movepi8_mask (a.v)); }
  template <> inline Mask<VecData<int16_t,32>> convert_vec2mask_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) { return Mask<VecData<int16_t,32>>(_mm512_movepi16_mask(a.v)); }
#endif

#if defined(__AVX512DQ__)
  template <> inline Mask<VecData<int32_t,16>> operator~<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& vec) { return Mask<VecData<int32_t,16>>(_knot_mask16(vec.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator~<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& vec) { return Mask<VecData<int64_t ,8>>(_knot_mask8 (vec.v)); }
  template <> inline Mask<VecData<float  ,16>> operator~<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& vec) { return Mask<VecData<float  ,16>>(_knot_mask16(vec.v)); }
  template <> inline Mask<VecData<double  ,8>> operator~<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& vec) { return Mask<VecData<double  ,8>>(_knot_mask8 (vec.v)); }

  template <> inline Mask<VecData<int32_t,16>> operator&<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kand_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator&<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kand_mask8 (a.v,b.v)); }
  template <> inline Mask<VecData<float  ,16>> operator&<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kand_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<double  ,8>> operator&<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kand_mask8 (a.v,b.v)); }

  template <> inline Mask<VecData<int32_t,16>> operator^<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kxor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator^<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kxor_mask8 (a.v,b.v)); }
  template <> inline Mask<VecData<float  ,16>> operator^<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kxor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<double  ,8>> operator^<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kxor_mask8 (a.v,b.v)); }

  template <> inline Mask<VecData<int32_t,16>> operator|<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator|<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kor_mask8 (a.v,b.v)); }
  template <> inline Mask<VecData<float  ,16>> operator|<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<double  ,8>> operator|<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kor_mask8 (a.v,b.v)); }

  template <> inline Mask<VecData<int32_t,16>> AndNot   <VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kandn_mask16(b.v,a.v)); }
  template <> inline Mask<VecData<int64_t ,8>> AndNot   <VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kandn_mask8 (b.v,a.v)); }
  template <> inline Mask<VecData<float  ,16>> AndNot   <VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kandn_mask16(b.v,a.v)); }
  template <> inline Mask<VecData<double  ,8>> AndNot   <VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kandn_mask8 (b.v,a.v)); }

  template <> inline VecData<int32_t,16> convert_mask2vec_intrin<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a) { return _mm512_movm_epi32(a.v); }
  template <> inline VecData<int64_t ,8> convert_mask2vec_intrin<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a) { return _mm512_movm_epi64(a.v); }
  template <> inline VecData<float  ,16> convert_mask2vec_intrin<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a) { return _mm512_castsi512_ps(_mm512_movm_epi32(a.v)); }
  template <> inline VecData<double  ,8> convert_mask2vec_intrin<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a) { return _mm512_castsi512_pd(_mm512_movm_epi64(a.v)); }

  template <> inline Mask<VecData<int32_t,16>> convert_vec2mask_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return Mask<VecData<int32_t,16>>(_mm512_movepi32_mask(a.v)); }
  template <> inline Mask<VecData<int64_t ,8>> convert_vec2mask_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a) { return Mask<VecData<int64_t ,8>>(_mm512_movepi64_mask(a.v)); }
  template <> inline Mask<VecData<float  ,16>> convert_vec2mask_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a) { return Mask<VecData<float  ,16>>(_mm512_movepi32_mask(_mm512_castps_si512(a.v))); }
  template <> inline Mask<VecData<double  ,8>> convert_vec2mask_intrin<VecData<double  ,8>>(const VecData<double  ,8>& a) { return Mask<VecData<double  ,8>>(_mm512_movepi64_mask(_mm512_castpd_si512(a.v))); }
#endif

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
#if defined(__AVX512BW__)
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::le>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::le>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE)); }
#endif

#if defined(__AVX512DQ__)
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::le>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::le>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::lt>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::le>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::gt>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::ge>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::eq>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::ne>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_UQ));}

  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::lt>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::le>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::gt>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::ge>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::eq>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::ne>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_UQ));}
#endif

#if defined(__AVX512BW__)
  template <> inline VecData<int8_t ,64> select_intrin(const Mask<VecData<int8_t ,64>>& s, const VecData<int8_t ,64>& a, const VecData<int8_t ,64>& b) { return _mm512_mask_blend_epi8 (s.v, b.v, a.v); }
  template <> inline VecData<int16_t,32> select_intrin(const Mask<VecData<int16_t,32>>& s, const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return _mm512_mask_blend_epi16(s.v, b.v, a.v); }
#endif
#if defined(__AVX512DQ__)
  template <> inline VecData<int32_t,16> select_intrin(const Mask<VecData<int32_t,16>>& s, const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return _mm512_mask_blend_epi32(s.v, b.v, a.v); }
  template <> inline VecData<int64_t ,8> select_intrin(const Mask<VecData<int64_t ,8>>& s, const VecData<int64_t ,8>& a, const VecData<int64_t ,8>& b) { return _mm512_mask_blend_epi64(s.v, b.v, a.v); }
  template <> inline VecData<float  ,16> select_intrin(const Mask<VecData<float  ,16>>& s, const VecData<float  ,16>& a, const VecData<float  ,16>& b) { return _mm512_mask_blend_ps   (s.v, b.v, a.v); }
  template <> inline VecData<double  ,8> select_intrin(const Mask<VecData<double  ,8>>& s, const VecData<double  ,8>& a, const VecData<double  ,8>& b) { return _mm512_mask_blend_pd   (s.v, b.v, a.v); }
#endif

  // Masked load and store
#if defined(__AVX512BW__)
  template <> inline VecData<int8_t ,64> loadu_mask_intrin<VecData<int8_t ,64>>(int8_t  const* p, const Mask<VecData<int8_t ,64>>& m) { return _mm512_maskz_loadu_epi8 (m.v, p); }
  template <> inline VecData<int16_t,32> loadu_mask_intrin<VecData<int16_t,32>>(int16_t const* p, const Mask<VecData<int16_t,32>>& m) { return _mm512_maskz_loadu_epi16(m.v, p); }
  template <> inline void storeu_mask_intrin<VecData<int8_t ,64>>(int8_t * p, VecData<int8_t ,64> vec, const Mask<VecData<int8_t ,64>>& m) { _mm512_mask_storeu_epi8 (p, m.v, vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int16_t,32>>(int16_t* p, VecData<int16_t,32> vec, const Mask<VecData<int16_t,32>>& m) { _mm512_mask_storeu_epi16(p, m.v, vec.v); }
#endif
#if defined(__AVX512DQ__)
  template <> inline VecData<int32_t,16> loadu_mask_intrin<VecData<int32_t,16>>(int32_t const* p, const Mask<VecData<int32_t,16>>& m) { return _mm512_maskz_loadu_epi32(m.v, p); }
  template <> inline VecData<int64_t ,8> loadu_mask_intrin<VecData<int64_t ,8>>(int64_t const* p, const Mask<VecData<int64_t ,8>>& m) { return _mm512_maskz_loadu_epi64(m.v, p); }
  template <> inline VecData<float  ,16> loadu_mask_intrin<VecData<float  ,16>>(float   const* p, const Mask<VecData<float  ,16>>& m) { return _mm512_maskz_loadu_ps   (m.v, p); }
  template <> inline VecData<double  ,8> loadu_mask_intrin<VecData<double  ,8>>(double  const* p, const Mask<VecData<double  ,8>>& m) { return _mm512_maskz_loadu_pd   (m.v, p); }
  template <> inline void storeu_mask_intrin<VecData<int32_t,16>>(int32_t* p, VecData<int32_t,16> vec, const Mask<VecData<int32_t,16>>& m) { _mm512_mask_storeu_epi32(p, m.v, vec.v); }
  template <> inline void storeu_mask_intrin<VecData<int64_t ,8>>(int64_t* p, VecData<int64_t ,8> vec, const Mask<VecData<int64_t ,8>>& m) { _mm512_mask_storeu_epi64(p, m.v, vec.v); }
  template <> inline void storeu_mask_intrin<VecData<float  ,16>>(float  * p, VecData<float  ,16> vec, const Mask<VecData<float  ,16>>& m) { _mm512_mask_storeu_ps   (p, m.v, vec.v); }
  template <> inline void storeu_mask_intrin<VecData<double  ,8>>(double * p, VecData<double  ,8> vec, const Mask<VecData<double  ,8>>& m) { _mm512_mask_storeu_pd   (p, m.v, vec.v); }
#endif

  // Number of selected lanes
#if defined(__POPCNT__) && defined(__AVX512BW__)
  template <> inline Integer mask_count_intrin<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& m) { return (Integer)_mm_popcnt_u64(m.v); }
  template <> inline Integer mask_count_intrin<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& m) { return _mm_popcnt_u32(m.v); }
#endif
#if defined(__POPCNT__) && defined(__AVX512DQ__)
  template <> inline Integer mask_count_intrin<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& m) { return _mm_popcnt_u32(m.v); }
  template <> inline Integer mask_count_intrin<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& m) { return _mm_popcnt_u32(m.v); }
  template <> inline Integer mask_count_intrin<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& m) { return _mm_popcnt_u32(m.v); }
  template <> inline Integer mask_count_intrin<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& m) { return _mm_popcnt_u32(m.v); }
#endif

  // Math functions
  template <> inline VecData<float ,16> sqrt_intrin <VecData<float ,16>>(const VecData<float ,16>& x) { return _mm512_sqrt_ps(x.v); }
  template <> inline VecData<double, 8> sqrt_intrin <VecData<double, 8>>(const VecData<double, 8>& x) { return _mm512_sqrt_pd(x.v); }
  template <> inline VecData<float ,16> floor_intrin<VecData<float ,16>>(const VecData<float ,16>& x) { return _mm512_roundscale_ps(x.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
  template <> inline VecData<double, 8> floor_intrin<VecData<double, 8>>(const VecData<double, 8>& x) { return _mm512_roundscale_pd(x.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
  template <> inline VecData<float ,16> ceil_intrin <VecData<float ,16>>(const VecData<float ,16>& x) { return _mm512_roundscale_ps(x.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC); }
  template <> inline VecData<double, 8> ceil_intrin <VecData<double, 8>>(const VecData<double, 8>& x) { return _mm512_roundscale_pd(x.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC); }
  template <> inline VecData<float ,16> trunc_intrin<VecData<float ,16>>(const VecData<float ,16>& x) { return _mm512_roundscale_ps(x.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
  template <> inline VecData<double, 8> trunc_intrin<VecData<double, 8>>(const VecData<double, 8>& x) { return _mm512_roundscale_pd(x.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
  template <> inline VecData<int32_t,16> fabs_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& x) { return _mm512_abs_epi32(x.v); }
  template <> inline VecData<int64_t, 8> fabs_intrin<VecData<int64_t, 8>>(const VecData<int64_t, 8>& x) { return _mm512_abs_epi64(x.v); }
#if defined(__AVX512BW__)
  template <> inline VecData<int8_t ,64> fabs_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& x) { return _mm512_abs_epi8 (x.v); }
  template <> inline VecData<int16_t,32> fabs_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& x) { return _mm512_abs_epi16(x.v); }
#endif
#if defined(__AVX512DQ__)
  template <> inline Mask<VecData<float ,16>> isnan_intrin<VecData<float ,16>>(const VecData<float ,16>& x) { return Mask<VecData<float ,16>>(_mm512_cmp_ps_mask(x.v, x.v, _CMP_UNORD_Q)); }
  template <> inline Mask<VecData<double, 8>> isnan_intrin<VecData<double, 8>>(const VecData<double, 8>& x) { return Mask<VecData<double, 8>>(_mm512_cmp_pd_mask(x.v, x.v, _CMP_UNORD_Q)); }
#endif

  // Gather and scatter
  template <> inline VecData<double, 8> gather_intrin<VecData<double, 8>,VecData<int32_t, 8>>(double const* p, const VecData<int32_t, 8>& idx) { return _mm512_i32gather_pd(idx.v, p, 8); }
  template <> inline VecData<double, 8> gather_intrin<VecData<double, 8>,VecData<int64_t, 8>>(double const* p, const VecData<int64_t, 8>& idx) { return _mm512_i64gather_pd(idx.v, p, 8); }
  template <> inline VecData<float , 8> gather_intrin<VecData<float , 8>,VecData<int64_t, 8>>(float  const* p, const VecData<int64_t, 8>& idx) { return _mm512_i64gather_ps(idx.v, p, 4); }
  template <> inline VecData<float ,16> gather_intrin<VecData<float ,16>,VecData<int32_t,16>>(float  const* p, const VecData<int32_t,16>& idx) { return _mm512_i32gather_ps(idx.v, p, 4); }
  template <> inline void scatter_intrin<VecData<double, 8>,VecData<int32_t, 8>>(double* p, const VecData<double, 8>& vec, const VecData<int32_t, 8>& idx) { _mm512_i32scatter_pd(p, idx.v, vec.v, 8); }
  template <> inline void scatter_intrin<VecData<double, 8>,VecData<int64_t, 8>>(double* p, const VecData<double, 8>& vec, const VecData<int64_t, 8>& idx) { _mm512_i64scatter_pd(p, idx.v, vec.v, 8); }
  template <> inline void scatter_intrin<VecData<float , 8>,VecData<int64_t, 8>>(float * p, const VecData<float , 8>& vec, const VecData<int64_t, 8>& idx) { _mm512_i64scatter_ps(p, idx.v, vec.v, 4); }
  template <> inline void scatter_intrin<VecData<float ,16>,VecData<int32_t,16>>(float * p, const VecData<float ,16>& vec, const VecData<int32_t,16>& idx) { _mm512_i32scatter_ps(p, idx.v, vec.v, 4); }


  // Special functions
  // log_mant_intrin: getmant and getexp, then m/2 and e + 1 where m > sqrt(2), as masked operations
  template <> inline void log_mant_intrin<VecData<float,16>>(VecData<float,16>& e, VecData<float,16>& m, const VecData<float,16>& x) {
    const __m512 m1 = _mm512_getmant_ps(x.v, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    const __mmask16 big = _mm512_cmp_ps_mask(m1, _mm512_set1_ps(1.41421356237309504880f), _CMP_GT_OQ);
    const __m512 e1 = _mm512_getexp_ps(x.v);
    m = _mm512_mask_mul_ps(m1, big, m1, _mm512_set1_ps(0.5f));
    e = _mm512_mask_add_ps(e1, big, e1, _mm512_set1_ps(1.0f));
  }
  template <> inline void log_mant_intrin<VecData<double,8>>(VecData<double,8>& e, VecData<double,8>& m, const VecData<double,8>& x) {
    const __m512d m1 = _mm512_getmant_pd(x.v, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    const __mmask8 big = _mm512_cmp_pd_mask(m1, _mm512_set1_pd(1.41421356237309504880), _CMP_GT_OQ);
    const __m512d e1 = _mm512_getexp_pd(x.v);
    m = _mm512_mask_mul_pd(m1, big, m1, _mm512_set1_pd(0.5));
    e = _mm512_mask_add_pd(e1, big, e1, _mm512_set1_pd(1.0));
  }
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<float,16>> {
    static constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
    static inline VecData<float,16> eval(const VecData<float,16>& a) {
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,16>>::eval(_mm512_rsqrt14_ps(a.v), a.v);
    }
    static inline VecData<float,16> eval(const VecData<float,16>& a, const Mask<VecData<float,16>>& m) {
      #if defined(__AVX512DQ__)
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,16>>::eval(_mm512_maskz_rsqrt14_ps(m.v, a.v), a.v);
      #else
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,16>>::eval(and_intrin(VecData<float,16>(_mm512_rsqrt14_ps(a.v)), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<double,8>> {
    static constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
    static inline VecData<double,8> eval(const VecData<double,8>& a) {
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,8>>::eval(_mm512_rsqrt14_pd(a.v), a.v);
    }
    static inline VecData<double,8> eval(const VecData<double,8>& a, const Mask<VecData<double,8>>& m) {
      #if defined(__AVX512DQ__)
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,8>>::eval(_mm512_maskz_rsqrt14_pd(m.v, a.v), a.v);
      #else
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,8>>::eval(and_intrin(VecData<double,8>(_mm512_rsqrt14_pd(a.v)), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };
  template <> inline VecData<float,16> rsqrt_intrin<VecData<float,16>>(const VecData<float,16>& x) { return rsqrt_full_intrin(x); } // faster than sqrt and division here, for dependent and independent calls
  template <> inline VecData<double,8> rsqrt_intrin<VecData<double,8>>(const VecData<double,8>& x) { return rsqrt_full_intrin(x); }

  #ifdef SCTL_HAVE_SVML
  template <> inline void sincos_intrin<VecData<float,16>>(VecData<float,16>& sinx, VecData<float,16>& cosx, const VecData<float,16>& x) { sinx = _mm512_sincos_ps(&cosx.v, x.v); }
  template <> inline void sincos_intrin<VecData<double,8>>(VecData<double,8>& sinx, VecData<double,8>& cosx, const VecData<double,8>& x) { sinx = _mm512_sincos_pd(&cosx.v, x.v); }

  template <> inline VecData<float,16> log_intrin<VecData<float,16>>(const VecData<float,16>& x) { return _mm512_log_ps(x.v); }
  template <> inline VecData<double,8> log_intrin<VecData<double,8>>(const VecData<double,8>& x) { return _mm512_log_pd(x.v); }

  template <> inline VecData<float,16> exp_intrin<VecData<float,16>>(const VecData<float,16>& x) { return _mm512_exp_ps(x.v); }
  template <> inline VecData<double,8> exp_intrin<VecData<double,8>>(const VecData<double,8>& x) { return _mm512_exp_pd(x.v); }

  template <> inline VecData<float,16> pow_intrin<VecData<float,16>>(const VecData<float,16>& x, const VecData<float,16>& y) { return _mm512_pow_ps(x.v, y.v); }
  template <> inline VecData<double,8> pow_intrin<VecData<double,8>>(const VecData<double,8>& x, const VecData<double,8>& y) { return _mm512_pow_pd(x.v, y.v); }
  #else
  template <> inline void sincos_intrin<VecData<float,16>>(VecData<float,16>& sinx, VecData<float,16>& cosx, const VecData<float,16>& x) {
    approx_sincos_intrin<-1>(sinx, cosx, x);
  }
  template <> inline void sincos_intrin<VecData<double,8>>(VecData<double,8>& sinx, VecData<double,8>& cosx, const VecData<double,8>& x) {
    approx_sincos_intrin<-1>(sinx, cosx, x);
  }

#ifdef SCTL_HAVE_LIBMVEC
  template <> inline VecData<float ,16> log_intrin<VecData<float ,16>>(const VecData<float ,16>& x) { return _ZGVeN16v_logf(x.v); }
  template <> inline VecData<double,8> log_intrin<VecData<double,8>>(const VecData<double,8>& x) { return _ZGVeN8v_log(x.v); }
  template <> inline VecData<float,16> pow_intrin<VecData<float,16>>(const VecData<float,16>& x, const VecData<float,16>& y) { return _ZGVeN16vv_powf(x.v, y.v); }
  template <> inline VecData<double,8> pow_intrin<VecData<double,8>>(const VecData<double,8>& x, const VecData<double,8>& y) { return _ZGVeN8vv_pow(x.v, y.v); }
#else
  template <> inline VecData<float,16> log_intrin<VecData<float,16>>(const VecData<float,16>& x) { return log_poly_intrin(x); }
  template <> inline VecData<double,8> log_intrin<VecData<double,8>>(const VecData<double,8>& x) { return log_poly_intrin(x); }
#endif

  template <> inline VecData<float,16> exp_intrin<VecData<float,16>>(const VecData<float,16>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<float>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  template <> inline VecData<double,8> exp_intrin<VecData<double,8>>(const VecData<double,8>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<double>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
#ifndef SCTL_HAVE_LIBMVEC
  template <> inline VecData<float,16> pow_intrin<VecData<float,16>>(const VecData<float,16>& x, const VecData<float,16>& y) { return pow_poly_intrin(x, y); }
  template <> inline VecData<double,8> pow_intrin<VecData<double,8>>(const VecData<double,8>& x, const VecData<double,8>& y) { return pow_poly_intrin(x, y); }
#endif
  template <> inline VecData<float,16> cbrt_intrin<VecData<float,16>>(const VecData<float,16>& x) { return cbrt_poly_intrin(x); }
  template <> inline VecData<double,8> cbrt_intrin<VecData<double,8>>(const VecData<double,8>& x) { return cbrt_poly_intrin(x); }
  template <> inline VecData<float,16> fmod_intrin<VecData<float,16>>(const VecData<float,16>& x, const VecData<float,16>& y) { return fmod_poly_intrin(x, y); }
  template <> inline VecData<double,8> fmod_intrin<VecData<double,8>>(const VecData<double,8>& x, const VecData<double,8>& y) { return fmod_poly_intrin(x, y); }
  #endif

#endif
}

#endif // _SCTL_INTRIN_WRAPPER_HPP_
