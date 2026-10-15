#ifndef _SCTL_INTRIN_WRAPPER_HPP_
#define _SCTL_INTRIN_WRAPPER_HPP_

#include <stdint.h>             // for int8_t, int16_t, int32_t, int64_t, uint8_t, ...
#include <cmath>                // for signbit, copysign
#include <cstring>              // for memcpy
#include <limits>               // for numeric_limits
#include <type_traits>          // for is_same
#include <utility>              // for integer_sequence

#include "sctl/common.hpp"      // for Integer, sctl, SCTL_ALIGN_B...
#include "sctl/math_utils.hpp"  // for const_pi, QuadReal, cos, exp, sin, sqrt
#include "sctl/math_utils.txx"  // for significant_bits

#if defined(__ARM_NEON)
#  ifndef SSE2NEON_PRECISE_MINMAX
#    define SSE2NEON_PRECISE_MINMAX 1 // _mm_max_* and _mm_min_* give the second operand if either is NaN, as on x86
#  endif
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

  /**
   * Promises about the arguments of a special function of Vec, given as its template argument A, e.g. log<Assume::All>(x)
   * or approx_exp<8, Assume::Finite | Assume::Moderate>(x): with all the promises that a function lists, it leaves out
   * its tests and code for other arguments, and its results for them are unspecified. Bits a function does not list
   * change nothing. Each function documents its Domain and Moderate.
   */
  enum class Assume : unsigned {
    None     = 0,
    Finite   = 1, ///< no inf or NaN
    Normal   = 2, ///< no zero or subnormal
    Domain   = 4, ///< inside the open domain that the function documents, e.g. x > 0 for log
    Moderate = 8, ///< inside the range that the function documents, e.g. |x| < 708.4 for exp of double
    All      = 15
  };
  constexpr Assume operator|(Assume a, Assume b) { return (Assume)((unsigned)a | (unsigned)b); }
  constexpr Assume operator&(Assume a, Assume b) { return (Assume)((unsigned)a & (unsigned)b); }
  constexpr bool assumes(Assume a, Assume b) { return ((unsigned)a & (unsigned)b) == (unsigned)b; } ///< a has every bit of b

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

  template <class ValueType, Integer N> struct VecData;

  template <class VData, class = void> inline constexpr bool array_lanes = false; // VData holds its lanes in an array
  template <class VData> inline constexpr bool array_lanes<VData, std::enable_if_t<VData::ArrayLanes>> = true;

  // VecData<ValueType,N> holds two halves when the half is in vector registers, or is itself two halves
  template <class ValueType, Integer N, bool = (N % 2 == 0)> struct SplitLanes : std::false_type {};
  template <class ValueType, Integer N> struct SplitLanes<ValueType, N, true> : std::bool_constant<!array_lanes<VecData<ValueType, N/2>>> {};

  template <class ValueType, Integer N, bool = SplitLanes<ValueType, N>::value> struct VecDataLanes {
    static constexpr bool ArrayLanes = true; // the lanes in an array, not in a vector register
    ValueType v[N];
  };
  template <class ValueType, Integer N> struct VecDataLanes<ValueType, N, true> {
    static constexpr bool SplitLanes = true;
    VecData<ValueType, N/2> lo, hi;
  };

  template <class ValueType, Integer N> struct alignas(sizeof(ValueType)*N>SCTL_ALIGN_BYTES?SCTL_ALIGN_BYTES:sizeof(ValueType)*N) VecData : VecDataLanes<ValueType, N> {
    using ScalarType = ValueType;
    static constexpr Integer Size = N;
  };

  template <class VData, class = void> inline constexpr bool split_lanes = false; // VData holds two halves, lo and hi; not a Mask derived from one
  template <class VData> inline constexpr bool split_lanes<VData, std::enable_if_t<VData::SplitLanes>> = std::is_same<VData, VecData<typename VData::ScalarType, VData::Size>>::value;
  template <class VData> using HalfVData = VecData<typename VData::ScalarType, VData::Size/2>;

  /**
   * The lanes of a VData as an array of T, copied by memcpy or directly, not through a union, whose accesses GCC 11 to
   * 14 can reorder.
   */
  template <class VData, class T = typename VData::ScalarType> class LaneArray {
    static constexpr bool Direct = [] { // the lanes are an array of T
      if constexpr (array_lanes<VData>) {
        return std::is_same<T, std::remove_extent_t<decltype(VData::v)>>::value;
      } else {
        return false;
      }
    }();

   public:
    LaneArray() = default;
    explicit LaneArray(const VData& v) {
      if constexpr (Direct) {
        a_ = v;
      } else {
        std::memcpy(a_, &v, sizeof(VData));
      }
    }
    T& operator[](Integer i) {
      if constexpr (Direct) {
        return a_.v[i];
      } else {
        return a_[i];
      }
    }
    const T& operator[](Integer i) const {
      if constexpr (Direct) {
        return a_.v[i];
      } else {
        return a_[i];
      }
    }
    VData value() const {
      if constexpr (Direct) {
        return a_;
      } else {
        VData v;
        std::memcpy(&v, a_, sizeof(VData));
        return v;
      }
    }

   private:
    // aligned as VData, which GCC's vector copies to it may assume
    alignas(VData) alignas(T) std::conditional_t<Direct, VData, T[sizeof(VData) / sizeof(T)]> a_;
  };

  template <class VData> inline VData zero_intrin();
  template <class VData> inline VData set1_intrin(typename VData::ScalarType a);

  template <Integer k, Integer Size, class Data, class T> inline void SetHelper(Data& vec, T x);
  template <Integer k, Integer Size, class Data, class T, class... T2> inline void SetHelper(Data& vec, T x, T2... rest);
  template <class VData, class T, class ...T2> inline VData set_intrin(T x, T2 ...args);

  template <class VData> inline VData load1_intrin(typename VData::ScalarType const* p);
  template <class VData> inline VData loadu_intrin(typename VData::ScalarType const* p);
  template <class VData> inline VData load_intrin(typename VData::ScalarType const* p);

  template <class VData> inline void storeu_intrin(typename VData::ScalarType* p, VData vec);
  template <class VData> inline void store_intrin(typename VData::ScalarType* p, VData vec);

  template <class VData> inline typename VData::ScalarType extract_intrin(VData vec, Integer i);
  template <class VData> inline void insert_intrin(VData& vec, Integer i, typename VData::ScalarType value);

  template <class VData> inline VData unary_minus_intrin(const VData& vec);
  template <class VData> inline VData mul_intrin(const VData& a, const VData& b);
  template <class VData> inline VData div_intrin(const VData& a, const VData& b);
  template <class VData> inline VData add_intrin(const VData& a, const VData& b);
  template <class VData> inline VData sub_intrin(const VData& a, const VData& b);
  template <class VData> inline VData rem_intrin(const VData& a, const VData& b); // remainder of the integer division, a - (a/b)*b
  template <class VData> inline VData fma_intrin(const VData& a, const VData& b, const VData& c);

  template <class VData> inline VData not_intrin(const VData& vec);
  template <class VData> inline VData and_intrin(const VData& a, const VData& b);
  template <class VData> inline VData xor_intrin(const VData& a, const VData& b);
  template <class VData> inline VData or_intrin(const VData& a, const VData& b);
  template <class VData> inline VData andnot_intrin(const VData& a, const VData& b);

  template <class VData> inline VData bitshiftleft_intrin(const VData& a, const Integer& rhs);
  template <class VData> inline VData bitshiftright_intrin(const VData& a, const Integer& rhs);
  template <class VData> inline VData bitshiftleft_intrin(const VData& a, const VData& rhs); // lane i bit shifted by lane i of rhs
  template <class VData> inline VData bitshiftright_intrin(const VData& a, const VData& rhs); // lane i bit shifted by lane i of rhs

  template <class VData> inline VData max_intrin(const VData& a, const VData& b);
  template <class VData> inline VData min_intrin(const VData& a, const VData& b);

  template <class VData> inline void transpose_intrin(VData (&v)[VData::Size]);

  template <Integer... I, class VData> inline VData blend_intrin(const VData& a, const VData& b);
  template <class VData, Integer... K> inline VData swap_pairs_lanes(const VData& vec, std::integer_sequence<Integer, K...>);
  template <class VData> inline VData swap_pairs_intrin(const VData& vec); // as a permute, for the shuffle of the compiler
  template <Integer... I, class VData> inline VData permute_intrin(const VData& a);

  template <class RetType, class ValueType, Integer N> inline RetType reinterpret_intrin(const VecData<ValueType,N>& v);
  template <class VData0, class VData> inline void transpose_reinterpret_intrin(VData (&v)[VData::Size]);
  template <class RealVec, class IntVec> inline RealVec convert_int2real_intrin(const IntVec& x);
  template <class IntVec, class RealVec> inline IntVec lrint_intrin(const RealVec& x); // as rint_intrin, to Int, for |x| < 2^(SigBits-1)
  template <class VData> inline VData rint_intrin(const VData& x); // nearest integer, halves to even, as std::rint; generic: |x| < 2^(SigBits-1), zero results are +0
  template <class VDataTo, class VData> inline VDataTo convert_intrin(const VData& a); // static_cast of each lane

  template <class VData> inline VecData<typename VData::ScalarType,VData::Size/2> get_low_intrin(const VData& a);
  template <class VData> inline VecData<typename VData::ScalarType,VData::Size/2> get_high_intrin(const VData& a);
  template <class VData> inline VecData<typename VData::ScalarType,VData::Size*2> concat_intrin(const VData& lo, const VData& hi);

  template <class VData> inline typename VData::ScalarType reduce_add_intrin(const VData& a);
  template <class VData> inline typename VData::ScalarType reduce_min_intrin(const VData& a);
  template <class VData> inline typename VData::ScalarType reduce_max_intrin(const VData& a);


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Storage of the lanes of a Mask<VData>, each all ones or all zeros: VData itself where it is a vector
   * register, the masks of the halves where VData holds two halves, and integers of the lane size where VData
   * holds an array, whose element copies need not keep all bytes (a copy of an x87 long double keeps 10 of its 16).
   */
  template <class VData, class = void> struct MaskLanesType {
    using type = VData;
  };
  template <class VData> struct MaskLanesType<VData, std::enable_if_t<VData::ArrayLanes>> {
    using type = VecData<typename IntegerType<sizeof(typename VData::ScalarType)>::value, VData::Size>;
  };
  template <class VData> struct Mask;
  template <class VData> struct MaskLanesType<VData, std::enable_if_t<VData::SplitLanes>> {
    struct type {
      Mask<HalfVData<VData>> lo, hi;
    };
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
      if constexpr (split_lanes<VData>) {
        MaskLanes<VData> m;
        m.lo = convert_vec2mask_intrin(v.lo);
        m.hi = convert_vec2mask_intrin(v.hi);
        return m;
      } else if constexpr (std::is_same<MaskLanes<VData>, VData>::value) {
        return v;
      } else {
        using Int = typename MaskLanes<VData>::ScalarType;
        constexpr Integer K = (sizeof(Int) < 8 ? sizeof(Int) : 8); // bytes read per lane
        LaneArray<MaskLanes<VData>> m; // the integer lanes can be two halves, as for 8-byte long double
        for (Integer i = 0; i < VData::Size; i++) {
          typename IntegerType<K>::value w;
          std::memcpy(&w, &v.v[i], K);
          m[i] = (w ? ~(Int)0 : (Int)0);
        }
        return m.value();
      }
    }
  };

  template <class VData> inline Mask<VecData<typename VData::ScalarType,VData::Size*2>> concat_mask_intrin(const Mask<VData>& lo, const Mask<VData>& hi);

  template <class RetType, class VData> inline RetType reinterpret_mask(const Mask<VData>& v);

  // Bitwise operators
  // the type whose bitwise intrinsics a Mask<VData> uses: VData where the Mask is a VData, for the intrinsics of the
  // target; deduced from the Mask itself, they would be the generic ones, byte by byte
  template <class VData> using MaskBits = std::conditional_t<std::is_base_of<VData, Mask<VData>>::value, VData, Mask<VData>>;
  template <class VData> inline Mask<VData> operator~(const Mask<VData>& vec);
  template <class VData> inline Mask<VData> operator&(const Mask<VData>& a, const Mask<VData>& b) {
    if constexpr (split_lanes<VData>) {
      return concat_mask_intrin(a.lo & b.lo, a.hi & b.hi);
    } else {
      return Mask<VData>(and_intrin<MaskBits<VData>>(a,b));
    }
  }
  template <class VData> inline Mask<VData> operator^(const Mask<VData>& a, const Mask<VData>& b);
  template <class VData> inline Mask<VData> operator|(const Mask<VData>& a, const Mask<VData>& b) {
    if constexpr (split_lanes<VData>) {
      return concat_mask_intrin(a.lo | b.lo, a.hi | b.hi);
    } else {
      return Mask<VData>(or_intrin<MaskBits<VData>>(a,b));
    }
  }
  template <class VData> inline Mask<VData> AndNot(const Mask<VData>& a, const Mask<VData>& b);

  template <class VData> inline VData convert_mask2vec_intrin(const Mask<VData>& v);
  template <class VData> inline Mask<VData> convert_vec2mask_intrin(const VData& v);
  template <class VData> using MaskIntVec = VecData<typename IntegerType<sizeof(typename VData::ScalarType)>::value, VData::Size>; // a mask as integers of all ones or all zeros
  template <class VData> inline MaskIntVec<VData> mask2int_intrin(const Mask<VData>& m);
  template <class VData> inline Mask<VData> int2mask_intrin(const MaskIntVec<VData>& q);
  template <class VDataTo, class VData> inline Mask<VDataTo> convert_mask_intrin(const Mask<VData>& m); // same lanes, another element type
  template <class VData> inline Integer mask_count_intrin(const Mask<VData>& m); // number of selected lanes
  template <class VData> inline bool mask_all_intrin(const Mask<VData>& m); // whether every lane is selected

  template <class VData> inline Integer mask_popcnt_intrin(const Mask<VData>& v);

  template <class VData> inline bool mask_any(const Mask<VData>& v);

  template <class VData> inline void mask_compress_store(const Mask<VData>& mask, const VData& v, typename VData::ScalarType* ptr);

  template <class VData> inline Integer mask_compress_iota_store(const Mask<VData>& mask, int32_t base, int32_t* ptr);

  template <class VData> inline Integer mask_compress_iota_store2(const Mask<VData>& mask_lo, const Mask<VData>& mask_hi, int32_t base, int32_t* ptr);

  template <class VData> inline VData mask_expand_load(const Mask<VData>& mask, const VData& zero, const typename VData::ScalarType* ptr);


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  enum class ComparisonType { lt, le, gt, ge, eq, ne};
  template <ComparisonType TYPE, class VData> inline Mask<VData> comp_intrin(const VData& a, const VData& b);

  template <class VData> inline VData select_intrin(const Mask<VData>& s, const VData& a, const VData& b);

  template <class VData> inline Mask<VData> mask_first_intrin(Integer n); // true in lanes i < n, for n >= 0
  template <class VData> inline VData loadu_mask_intrin(typename VData::ScalarType const* p, const Mask<VData>& m); // lanes not in m are zero and not read
  template <class VData> inline void storeu_mask_intrin(typename VData::ScalarType* p, VData vec, const Mask<VData>& m); // lanes not in m are not written
  template <class VData> inline VData loadu_first_intrin(typename VData::ScalarType const* p, Integer n); // lanes i < n, for n >= 0; the others are zero and not read
  template <class VData> inline void storeu_first_intrin(typename VData::ScalarType* p, VData vec, Integer n); // lanes i < n, for n >= 0; the others are not written

  template <class VData, class IdxVData> inline VData gather_intrin(typename VData::ScalarType const* p, const IdxVData& idx); // lane i is p[idx lane i]
  template <class VData, class IdxVData> inline void scatter_intrin(typename VData::ScalarType* p, const VData& vec, const IdxVData& idx); // in lane order: of equal indices, the last lane is stored


  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13, const CType& c14);
  template <class VData, class CType> inline VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13, const CType& c14, const CType& c15);

  template <Integer K> inline constexpr Integer poly_degree(const double (&d)[K], const Integer first, const Integer full, const Integer digits);


  template <bool FullRange = true, class VData> inline void sincos_pi_reduce_intrin(VData& w, VData& t, const VData& x);
  // Two VData returned by value; not std::pair, whose passing GCC for ARM notes as changed in GCC 10.1 (an empty base in
  // C++17)
  template <class VData> struct VDataPair {
    VData first;
    VData second;
  };
  template <class VData> [[gnu::noinline, gnu::cold]] VDataPair<VData> sincos_reduce_double_intrin(VData x1, VData x_int, const VData x, const Mask<VData> in_range);
  template <class VData> [[gnu::noinline, gnu::cold]] VDataPair<VData> sincos_beyond_intrin(const VData sinx, const VData cosx, const VData x, const typename VData::ScalarType lim);
  template <Integer DIGITS, bool FullRange = true, bool ExactReduction = FullRange, class VData> inline void approx_sincos_intrin(VData& sinx, VData& cosx, const VData& x);
  template <bool Cos, bool FullRange = true, Integer DIGITS = -1, class VData> inline VData sin_or_cos_intrin(const VData& x); // sin or cos by one polynomial, float and double
  template <bool FullRange = true, Integer DIGITS = -1, class VData> inline VData tan_poly_intrin(const VData& x); // tan, float and double
  template <bool Cos, bool FullRange = true, Integer DIGITS = -1, class VData> inline VData sinpi_or_cospi_intrin(const VData& x); // sinpi or cospi by one polynomial, float and double
  template <class VData> inline void sincos_intrin(VData& sinx, VData& cosx, const VData& x);

  template <class Real> inline constexpr Real exp_normal_lim(); // for |x| below, e^x and 2^round(x/ln2) are normal and finite
  enum class ExpArg { Plain, Sum, Base10, Base2, Half }; // approx_exp_intrin of x: e^x, e^(x + x_lo), 10^x, 2^x, e^x/2
  enum class LogBase { E, Two, Ten }; // log_split_poly_intrin of x: log x, log2 x, log10 x
  template <Integer ORDER, bool RangeCheck = true, ExpArg Arg = ExpArg::Plain, class VData> inline VData approx_exp_intrin(const VData& x, const VData& x_lo);
  template <Integer ORDER, bool RangeCheck = true, class VData> inline VData approx_exp_intrin(const VData& x);
  template <class VData> inline Mask<VData> positive_normal_mask_intrin(const VData& x); // x normal, positive and finite
  template <class VData> inline void exp_mant_intrin(VData& e, VData& m, const VData& x); // x = 2^e m, m in [1, 2), for normal x > 0
  template <class VData> inline void log_mant_intrin(VData& e, VData& m, const VData& x); // x = 2^e m, m in [sqrt(1/2), sqrt(2)), for normal x > 0
  template <bool Subnormal = true, class VData> inline void log_split_intrin(VData& e, VData& f, const VData& x); // x = 2^e (1+f), 1+f in [sqrt(1/2), sqrt(2)), for x > 0; without Subnormal, for normal x
  template <LogBase B = LogBase::E, class VData> inline VData log_split_poly_intrin(const VData& e, const VData& f);
  template <class VData> inline VData log_split_add_intrin(const VData& e, const VData& f, const VData& c);
  template <bool Check = true, LogBase B = LogBase::E, class VData> inline VData log_poly_intrin(const VData& x); // Check = false: only for positive normal x
  template <class VData> inline void split_half_intrin(VData& hi, VData& lo, const VData& x); // x = hi + lo exactly, hi with the low half of the significand cleared
  template <class VData> inline VData mul_sub_exact_intrin(const VData& a, const VData& b, const VData& c); // a b - c, the product to about twice the precision, for c near a b: by FMA, or else with a and b split in two halves
  template <class VData> inline void log_hi_lo_intrin(VData& L_hi, VData& L_lo, const VData& e, const VData& f);
  template <bool Check = true, class VData> inline VData pow_poly_intrin(const VData& x, const VData& y); // exp(y log|x|), log|x| = L_hi + L_lo to about 1.5x the precision of Real; std::pow at signs, zeros, inf, NaN. Check = false: only for positive normal x and |y log x| < exp_normal_lim
  template <bool Subnormal = true, bool Special = true, Integer DIGITS = -1, class VData> inline VData cbrt_poly_intrin(const VData& x); // cbrt(2^e m) = 2^q cbrt(2^r m), e = 3q + r, r in {-1, 0, 1}; (2^r m)^(-1/3) by a polynomial, then a series in its residual; no division. Without Subnormal, for no subnormal x; without Special, for x not 0, inf or NaN
  template <class VData> [[gnu::noinline, gnu::cold]] VData fmod_lanes_intrin(const VData x, const VData y, const VData r, const Mask<VData> m);
  template <class VData> inline VData fmod_poly_intrin(const VData& x, const VData& y); // |x| - q |y|, q = trunc(fl(|x|/|y|)), the quotient or one more; exact
  template <class IntVec> inline IntVec div_int64_intrin(const IntVec& a, const IntVec& b); // through double, exact for |a|, |b| < 2^51; other lanes one element at a time
  template <class IntVec> inline IntVec div_int32_intrin(const IntVec& a, const IntVec& b); // through float, exact: |a| / |b| by two estimates from a reciprocal scaled down so that they are at most the quotient, then the sign
  template <class VData> inline VData exp_intrin(const VData& x);
  // exp_intrin, log_intrin, pow_intrin, cbrt_intrin or sincos_intrin of VData is SCTL's polynomial (approx_exp_intrin,
  // log_poly_intrin, pow_poly_intrin, cbrt_poly_intrin, approx_sincos_intrin), not SVML, libmvec or std one element at a
  // time: an Assume can leave out its tests
  enum class PolyOf { Exp, Log, Pow, Cbrt, Sincos };
  template <PolyOf F, class VData> inline constexpr bool sctl_poly();
  template <class VData> inline VData exp_normal_intrin(const VData& x);
  template <Assume A, class VData> inline VData exp_assume_intrin(const VData& x);
  template <Assume A, class VData> inline VData log_assume_intrin(const VData& x);
  template <Integer DIGITS, Assume A, class VData> inline VData log_lanes_intrin(const VData& x); // log of the x > 0 among lanes of any value, without a branch
  template <Assume A, class VData> inline VData pow_assume_intrin(const VData& x, const VData& y);
  template <Assume A, Integer DIGITS = -1, class VData> inline VData cbrt_assume_intrin(const VData& x); // Normal: no subnormal handling; Finite | Normal: also no fix of 0, inf, NaN
  template <class VData> inline VData log_intrin(const VData& x);
  template <class VData> inline VData pow_intrin(const VData& x, const VData& y);
  inline constexpr Integer log_poly_degree(const Integer digits); // the lowest degree of LogPolyCoeffs for the digits; 0 if none
  inline constexpr Integer exp_taylor_order(const Integer digits); // Taylor order k of e^r, |r| <= ln2/2: (ln2/2)^(k+1)/(k+1)! < 10^-digits
  inline constexpr Integer log_atanh_terms(const Integer digits); // terms K of 2 atanh(s), s^2 <= 0.0295: 0.0295^K/(2K+1) < 10^-digits
  template <class VData, class Coeffs, Integer... k> inline VData eval_coeffs_intrin(const VData& x, std::integer_sequence<Integer, k...>); // sum Coeffs::c[k] x^k
  template <Integer Lo, Integer Len, class VData, Integer K, class CType, Integer N> inline VData eval_poly_estrin_intrin(const VData (&xp)[K], const CType (&c)[N]);
  template <class VData, class CType, Integer N> inline VData eval_poly_intrin(const VData& x, const CType (&c)[N]);
  template <class VData, class CType, Integer N> inline VData eval_poly_horner_intrin(const VData& x, const CType (&c)[N], const double scale = 1);
  template <Integer B, class VData> inline VData eval_poly_rows_intrin(const typename VData::ScalarType* const (&p)[VData::Size], const VData& t);
  template <class VData, class CType, Integer N, Integer M> inline VData eval_rational_intrin(const VData& x, const CType (&p)[N], const CType (&q)[M]);
  template <class VData, Integer... k> inline VData atanh_series_intrin(const VData& z, std::integer_sequence<Integer, k...>); // sum 1/(2k+3) z^k
  template <Integer DIGITS, LogBase B = LogBase::E, class VData> inline VData approx_log_split_poly_intrin(const VData& e, const VData& f);
  template <Integer DIGITS, LogBase B = LogBase::E, class VData> inline VData approx_log_normal_intrin(const VData& x);
  template <Integer DIGITS, bool Check = true, LogBase B = LogBase::E, class VData> inline VData approx_log_intrin(const VData& x); // approx_log_normal_intrin; log_intrin if some x is not positive, normal and finite, unless !Check
  template <Integer LOG_DIGITS, Integer ORDER, bool Check = true, class VData> inline VData approx_pow_intrin(const VData& x, const VData& y); // e^(y log x) with approx_log_normal_intrin<LOG_DIGITS> and approx_exp_intrin<ORDER>; pow_intrin if some x is not positive, normal and finite, or |y log x| >= 708.4 (float: 87.3)
  template <class VData> inline VData sqrt_intrin(const VData& x);
  template <class VData> inline VData rsqrt_intrin(const VData& x);
  template <class VData> inline VData rsqrt_full_intrin(const VData& x); // 1/sqrt(x): rsqrt_approx_intrin at full precision for 2 min <= x <= max; if any x is outside, sqrt and division there
  template <class VData> inline VData fabs_intrin(const VData& x);
  template <class VData> inline VData floor_intrin(const VData& x);
  template <class VData> inline VData ceil_intrin(const VData& x);
  template <class VData> inline VData copysign_intrin(const VData& x, const VData& y);
  template <class VData> inline Mask<VData> isnan_intrin(const VData& x);
  template <bool SpecialValues = true, Integer DIGITS = -1, class VData> inline VData atan2_intrin(const VData& y, const VData& x); // SpecialValues: x, y both infinite or both zero; DIGITS = -1: full
  template <Integer DIGITS = -1, class VData> inline VData atan_intrin(const VData& x); // atan2 without SpecialValues: its arguments are never both infinite or both zero, here and below
  template <Integer DIGITS = -1, class VData> inline VData asin_intrin(const VData& x);
  template <Integer DIGITS = -1, class VData> inline VData acos_intrin(const VData& x);

  template <class VData> inline VData trunc_intrin(const VData& x);
  template <class VData> inline VData round_intrin(const VData& x); // halfway cases away from zero
  template <class VData> inline Mask<VData> isinf_intrin(const VData& x);
  template <class VData> inline Mask<VData> isfinite_intrin(const VData& x);

  template <bool AvoidOverflow = true, bool SpecialValues = true, class VData> inline VData hypot_intrin(const VData& x, const VData& y); // SpecialValues: zeros, inf, NaN

  template <bool RangeCheck = true, Integer DIGITS = -1, class VData> inline VData exp2_intrin(const VData& x); // as approx_exp_intrin; RangeCheck = false: only for |x| < 1022 (float: 126); DIGITS = -1: full
  template <Integer ORDER, bool RangeCheck = true, class VData> inline VData approx_exp10_intrin(const VData& x); // as approx_exp_intrin
  template <Assume A = Assume::None, class VData> inline VData exp10_intrin(const VData& x); // Finite | Moderate: |x| < exp_normal_lim / ln10
  template <Assume A = Assume::None, class VData> inline VData log2_intrin(const VData& x);
  template <Assume A = Assume::None, class VData> inline VData log10_intrin(const VData& x);

  template <Integer DIGITS = -1, bool Upper = true, class VData> inline VData expm1_intrin(const VData& x); // Upper = false: only for x <= 0; DIGITS = -1: full
  template <class VData> inline VData log1p_intrin(const VData& x);

  template <class Real> inline Real log1p_generic(const Real w); // log(1 + w) as log(u) w/(u-1), u = 1 + w rounded
  template <class VData> inline VData sqrt_err_intrin(const VData& s, const VData& g, const VData& g_err);
  template <Integer digits = -1, bool FullRange = true, class VData> inline VData asinh_intrin(const VData& x);
  template <Integer digits = -1, bool FullRange = true, class VData> inline VData acosh_intrin(const VData& x);
  template <Integer digits = -1, bool FullRange = true, class VData> inline VData atanh_intrin(const VData& x);

  template <Integer digits, class VData> inline VData erf_poly_intrin(const VData& x);
  template <Integer digits, bool Erf = false, bool RangeCheck = !Erf, class VData> inline VData erfc_rational_intrin(const VData& a);
  template <class Real> inline Real exp_neg_sq_generic(const Real x);
  template <class Real> inline Real erf_series_generic(const Real x);
  template <class Real> inline Real erfc_cf_generic(const Real a);
  template <Integer digits = -1, class VData> inline VData erf_intrin(const VData& x);
  template <Integer digits = -1, Assume A = Assume::None, class VData> inline VData erfc_intrin(const VData& x); // A: Finite | Moderate, x^2 < exp_normal_lim
  template <class Real> inline Real erfinv_generic(const Real y, const Real t);
  template <bool Erfinv, Integer digits, bool Check = true, class VData> inline VData ndtri_erfinv_intrin(const VData& p); // Check = false: only for t > 0 normal
  template <Integer digits = -1, Assume A = Assume::None, class VData> inline VData ndtri_intrin(const VData& p); // A: Normal | Domain, 0 < p < 1 normal
  template <Integer digits = -1, Assume A = Assume::None, class VData> inline VData erfinv_intrin(const VData& y); // A: Domain, |y| < 1

  template <Integer digits, bool FullRange = true, bool Sin = true, bool Cos = true, bool DivPi = false, class VData> inline void approx_sincospi_intrin(VData& sinx, VData& cosx, const VData& x);

  template <Integer digits, bool FullRange = true, class VData> inline VData approx_sincpi_intrin(const VData& x);

  template <class VData> inline void gamma_shift_intrin(VData& y, VData& p, const VData& x, const typename VData::ScalarType hi);
  template <class Coeffs, bool Exact = true, class VData> inline void lgamma_stirling_intrin(VData& L_hi, VData& L_lo, const VData& x, const VData& lh, const VData& ll);
  template <class Coeffs, class VData> inline VData lgamma_mid_intrin(VData& arg, VData& sgn, const VData& a);
  template <bool Digamma, class Real> inline Real gamma_series_generic(const Real w);
  template <class Real> inline Real lgamma_generic(const Real x);
  template <class Real> inline Real tgamma_generic(const Real x);
  template <class Real> inline Real digamma_generic(const Real x);
  template <Integer digits = -1, Assume A = Assume::None, class VData> inline VData tgamma_intrin(const VData& x); // A: Moderate, |x| < 170 (float: 33); Domain, not a pole
  template <class Coeffs, Assume A, class VData> [[gnu::noinline]] VData lgamma_general_intrin(const VData x);
  template <Integer digits = -1, Assume A = Assume::None, class VData> inline VData lgamma_intrin(const VData& x); // A: Finite | Normal, the logs without tests; Moderate, |x| < 2^SigBits; Domain, not a pole
  template <Integer digits = -1, class VData> inline VData digamma_intrin(const VData& x);

  template <class Real> inline void bessel_jy_generic(Real (&J)[3], Real (&Y)[3], const Real x);
  template <Integer n, bool J, bool Y, Integer digits, Assume A = Assume::None, class VData> inline void bessel_jy_intrin(VData& j, VData& y, const VData& a); // A as cyl_neumann_intrin
  template <Integer n, Integer digits = -1, Assume A = Assume::None, class VData> inline VData cyl_bessel_j_intrin(const VData& x); // A: Finite | Moderate, |x| < PiOver2Split<Real>::lim
  template <Integer n, Integer digits = -1, Assume A = Assume::None, class VData> inline VData cyl_neumann_intrin(const VData& x); // A: all four, 0 < x < PiOver2Split<Real>::lim normal
  template <class Real> inline void bessel_ik_generic(Real (&I)[3], Real (&K)[3], const Real x);
  template <Integer n, Integer digits, Assume A = Assume::None, class VData> inline VData bessel_i_intrin(const VData& a); // A as cyl_bessel_i_intrin
  template <Integer n, Integer digits, Assume A = Assume::None, class VData> inline VData bessel_k_intrin(const VData& a); // A as cyl_bessel_k_intrin
  template <Integer n, Integer digits = -1, Assume A = Assume::None, class VData> inline VData cyl_bessel_i_intrin(const VData& x); // A: Finite | Moderate, |x| < exp_normal_lim
  template <Integer n, Integer digits = -1, Assume A = Assume::None, class VData> inline VData cyl_bessel_k_intrin(const VData& x); // A: all four, 0 < x < exp_normal_lim normal

  template <class Real> inline void sph_bessel_generic(Real (&j)[3], Real (&y)[3], const Real x);
  template <Integer n, bool J, bool Y, Integer digits, Assume A = Assume::None, class VData> inline void sph_bessel_jy_intrin(VData& j, VData& y, const VData& a); // A: Finite | Moderate, Normal
  template <Integer n, Integer digits = -1, Assume A = Assume::None, class VData> inline VData sph_bessel_intrin(const VData& x); // A: Finite | Moderate, |x| < PiOver2Split<Real>::lim; Normal for n = 0
  template <Integer n, Integer digits = -1, Assume A = Assume::None, class VData> inline VData sph_neumann_intrin(const VData& x); // A: all four, 0 < x < PiOver2Split<Real>::lim normal

  template <class Real> inline Real riemann_zeta_generic(const Real s);
  template <Integer digits, class VData> inline VData zeta_pos_intrin(const VData& s, const VData& u);
  template <class VData> inline void two_sum_intrin(VData& s1, VData& s2, const VData& a, const VData& b);
  template <Integer digits, class VData> [[gnu::noinline]] VData zeta_reflection_big_intrin(const VData s, const VData sn);
  template <Integer digits = -1, class VData> inline VData riemann_zeta_intrin(const VData& s);
  template <class VData> inline VData cbrt_intrin(const VData& x);
  template <class VData> inline VData fmod_intrin(const VData& x, const VData& y);

  template <Integer DIGITS = -1, class VData> inline VData sinh_series_intrin(const VData& x); // x (1 + x^2/3! + x^4/5! + ...), |x| < 1
  template <Assume A = Assume::None, Integer DIGITS = -1, class VData> inline VData exp_half_intrin(const VData& a); // e^a / 2 for a >= 0; inf beyond the overflow. Finite | Moderate: a < exp_normal_lim + 1
  template <Assume A = Assume::None, Integer DIGITS = -1, class VData> inline VData cosh_intrin(const VData& x);
  template <Assume A = Assume::None, Integer DIGITS = -1, class VData> inline VData sinh_intrin(const VData& x);
  template <Integer DIGITS = -1, class VData> inline VData tanh_intrin(const VData& x); // -v/(2+v), v = expm1(-2|x|)

  template <Integer DIGITS, class VData> inline VData approx_sin_intrin(const VData& x);
  template <Integer DIGITS, class VData> inline VData approx_cos_intrin(const VData& x);
  template <Integer DIGITS, class VData> inline VData approx_tan_intrin(const VData& x);
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


#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 14 || (__GNUC__ >= 12 && defined(__AVX__)))
  inline __m128i dup64_const_intrin(const uint64_t bits);
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


#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
  template <bool EachUse = false> inline __m256i dup64x4_const_intrin(const uint64_t bits);
#endif


  #ifndef __AVX2__
  template <class F, class... A> inline __m256i avx_halves_intrin(const F& f, const A... a); // f applied to each 128-bit half of the a
  #endif


  #ifdef __AVX2__


  template <class VData> inline void transpose256_intrin(VData (&v)[VData::Size]);
  #else // AVX without AVX2: [[A,B],[C,D]] -> [[A^T,C^T],[B^T,D^T]] by the 128-bit transposes of the HxH blocks
  template <class VData> inline void transpose256_intrin(VData (&v)[VData::Size]);
  #endif


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  #ifdef __AVX2__


  #else // AVX without AVX2: each 128-bit half with SSE
  template <class VData> inline Mask<VData> mask_first_halves_intrin(Integer n);
  #endif


  #if defined(__AVX512F__) && defined(__AVX512VL__)
  #elif defined(SCTL_TUNE_ZEN)
  template <class VData> inline VData loadu_first_halves_intrin(typename VData::ScalarType const* p, Integer n);
  template <class VData> inline void storeu_first_halves_intrin(typename VData::ScalarType* p, const VData& vec, Integer n);
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


#if defined(__AVX512BW__)


  template <class VData> inline void transpose512_intrin(VData (&v)[VData::Size]);
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

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER) && \
    (__GNUC__ == 12 || (__GNUC__ == 13 && __GNUC_MINOR__ < 4) || (__GNUC__ == 14 && __GNUC_MINOR__ < 3))
  inline unsigned mask_to_u32(__mmask16 k);
  inline unsigned mask_to_u32(__mmask8 k);
#else
  inline unsigned mask_to_u32(__mmask16 k);
  inline unsigned mask_to_u32(__mmask8 k);
#endif

#endif


#endif
}

#endif // _SCTL_INTRIN_WRAPPER_HPP_
