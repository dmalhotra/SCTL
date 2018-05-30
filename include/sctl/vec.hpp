#ifndef _SCTL_VEC_WRAPPER_HPP_
#define _SCTL_VEC_WRAPPER_HPP_

#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(common.hpp)
#include <cstdint>
#include <ostream>

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#if defined(__MIC__)
#include <immintrin.h>
#endif

// TODO: Check alignment when SCTL_MEMDEBUG is defined
// TODO: Replace pointers with iterators

namespace SCTL_NAMESPACE {

  template <class ValueType> class TypeTraits {
    public:
      static constexpr Integer SigBits = 0;
  };
  template <> class TypeTraits<float> {
    public:
      static constexpr Integer SigBits = 23;
  };
  template <> class TypeTraits<double> {
    public:
      static constexpr Integer SigBits = 52;
  };


  template <class ValueType, Integer N> class alignas(sizeof(ValueType) * N) Vec {
    public:

      typedef ValueType ScalarType;

      static constexpr Integer Size() {
        return N;
      }

      static Vec Zero() {
        Vec r;
        for (Integer i = 0; i < N; i++) r.v[i] = 0;
        return r;
      }

      static Vec Load1(ValueType const* p) {
        Vec r;
        for (Integer i = 0; i < N; i++) r.v[i] = p[0];
        return r;
      }
      static Vec Load(ValueType const* p) {
        Vec r;
        for (Integer i = 0; i < N; i++) r.v[i] = p[i];
        return r;
      }
      static Vec LoadAligned(ValueType const* p) {
        Vec r;
        for (Integer i = 0; i < N; i++) r.v[i] = p[i];
        return r;
      }

      Vec() {}

      Vec(const ValueType& a) {
        for (Integer i = 0; i < N; i++) v[i] = a;
      }

      void Store(ValueType* p) const {
        for (Integer i = 0; i < N; i++) p[i] = v[i];
      }
      void StoreAligned(ValueType* p) const {
        for (Integer i = 0; i < N; i++) p[i] = v[i];
      }

      // Bitwise NOT
      Vec operator~() const {
        Vec r;
        char* vo = (char*)r.v;
        const char* vi = (const char*)this->v;
        for (Integer i = 0; i < (Integer)(N*sizeof(ValueType)); i++) vo[i] = ~vi[i];
        return r;
      }

      // Unary plus and minus
      Vec operator+() const {
        return *this;
      }
      Vec operator-() const {
        Vec r;
        for (Integer i = 0; i < N; i++) r.v[i] = -v[i];
        return r;
      }

      // C-style cast
      template <class RetValueType> explicit operator Vec<RetValueType,N>() const {
        Vec<RetValueType,N> r;
        for (Integer i = 0; i < N; i++) r.v[i] = (RetValueType)v[i];
        return r;
      }

      // Arithmetic operators
      friend Vec operator*(Vec lhs, const Vec& rhs) {
        for (Integer i = 0; i < N; i++) lhs.v[i] *= rhs.v[i];
        return lhs;
      }
      friend Vec operator+(Vec lhs, const Vec& rhs) {
        for (Integer i = 0; i < N; i++) lhs.v[i] += rhs.v[i];
        return lhs;
      }
      friend Vec operator-(Vec lhs, const Vec& rhs) {
        for (Integer i = 0; i < N; i++) lhs.v[i] -= rhs.v[i];
        return lhs;
      }

      // Comparison operators
      friend Vec operator< (Vec lhs, const Vec& rhs) {
        static const ValueType value_zero = const_zero();
        static const ValueType value_one = const_one();
        for (Integer i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] < rhs.v[i] ? value_one : value_zero);
        return lhs;
      }
      friend Vec operator<=(Vec lhs, const Vec& rhs) {
        static const ValueType value_zero = const_zero();
        static const ValueType value_one = const_one();
        for (Integer i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] <= rhs.v[i] ? value_one : value_zero);
        return lhs;
      }
      friend Vec operator>=(Vec lhs, const Vec& rhs) {
        static const ValueType value_zero = const_zero();
        static const ValueType value_one = const_one();
        for (Integer i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] >= rhs.v[i] ? value_one : value_zero);
        return lhs;
      }
      friend Vec operator> (Vec lhs, const Vec& rhs) {
        static const ValueType value_zero = const_zero();
        static const ValueType value_one = const_one();
        for (Integer i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] > rhs.v[i] ? value_one : value_zero);
        return lhs;
      }
      friend Vec operator==(Vec lhs, const Vec& rhs) {
        static const ValueType value_zero = const_zero();
        static const ValueType value_one = const_one();
        for (Integer i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] == rhs.v[i] ? value_one : value_zero);
        return lhs;
      }
      friend Vec operator!=(Vec lhs, const Vec& rhs) {
        static const ValueType value_zero = const_zero();
        static const ValueType value_one = const_one();
        for (Integer i = 0; i < N; i++) lhs.v[i] = (lhs.v[i] != rhs.v[i] ? value_one : value_zero);
        return lhs;
      }

      // Bitwise operators
      friend Vec operator&(Vec lhs, const Vec& rhs) {
        char* vo = (char*)lhs.v;
        const char* vi = (const char*)rhs.v;
        for (Integer i = 0; i < (Integer)sizeof(ValueType)*N; i++) vo[i] &= vi[i];
        return lhs;
      }
      friend Vec operator^(Vec lhs, const Vec& rhs) {
        char* vo = (char*)lhs.v;
        const char* vi = (const char*)rhs.v;
        for (Integer i = 0; i < (Integer)sizeof(ValueType)*N; i++) vo[i] ^= vi[i];
        return lhs;
      }
      friend Vec operator|(Vec lhs, const Vec& rhs) {
        char* vo = (char*)lhs.v;
        const char* vi = (const char*)rhs.v;
        for (Integer i = 0; i < (Integer)sizeof(ValueType)*N; i++) vo[i] |= vi[i];
        return lhs;
      }
      friend Vec AndNot(Vec lhs, const Vec& rhs) {
        return lhs & (~rhs);
      }

      // Assignment operators
      Vec& operator+=(const Vec& rhs) {
        for (Integer i = 0; i < N; i++) v[i] += rhs.v[i];
        return *this;
      }
      Vec& operator-=(const Vec& rhs) {
        for (Integer i = 0; i < N; i++) v[i] -= rhs.v[i];
        return *this;
      }
      Vec& operator*=(const Vec& rhs) {
        for (Integer i = 0; i < N; i++) v[i] *= rhs.v[i];
        return *this;
      }
      Vec& operator&=(const Vec& rhs) {
        char* vo = (char*)this->v;
        const char* vi = (const char*)rhs.v;
        for (Integer i = 0; i < (Integer)sizeof(ValueType)*N; i++) vo[i] &= vi[i];
        return *this;
      }
      Vec& operator^=(const Vec& rhs) {
        char* vo = (char*)this->v;
        const char* vi = (const char*)rhs.v;
        for (Integer i = 0; i < (Integer)sizeof(ValueType)*N; i++) vo[i] ^= vi[i];
        return *this;
      }
      Vec& operator|=(const Vec& rhs) {
        char* vo = (char*)this->v;
        const char* vi = (const char*)rhs.v;
        for (Integer i = 0; i < (Integer)sizeof(ValueType)*N; i++) vo[i] != vi[i];
        return *this;
      }

      // Other operators
      friend Vec max(Vec lhs, const Vec& rhs) {
        for (Integer i = 0; i < N; i++) {
          if (lhs.v[i] < rhs.v[i]) lhs.v[i] = rhs.v[i];
        }
        return lhs;
      }
      friend Vec min(Vec lhs, const Vec& rhs) {
        for (Integer i = 0; i < N; i++) {
          if (lhs.v[i] > rhs.v[i]) lhs.v[i] = rhs.v[i];
        }
        return lhs;
      }

      friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
        //for (Integer i = 0; i < (Integer)sizeof(ValueType)*8; i++) os << ((*(uint64_t*)in.v) & (1UL << i) ? '1' : '0');
        //os << '\n';
        for (Integer i = 0; i < N; i++) os << in.v[i] << ' ';
        return os;
      }
      friend Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x) {
        Vec<ValueType,N> r;
        for (int i = 0; i < N; i++) r.v[i] = 1.0 / sqrt(x.v[i]);
        return r;
      }

    private:

      static const ValueType const_zero() {
        union {
          ValueType value;
          unsigned char cvalue[sizeof(ValueType)];
        };
        for (Integer i = 0; i < (Integer)sizeof(ValueType); i++) cvalue[i] = 0;
        return value;
      }
      static const ValueType const_one() {
        union {
          ValueType value;
          unsigned char cvalue[sizeof(ValueType)];
        };
        for (Integer i = 0; i < (Integer)sizeof(ValueType); i++) cvalue[i] = ~(unsigned char)0;
        return value;
      }

      ValueType v[N];
  };

  // Other operators
  template <class Vec, Integer ORDER = 13> void sincos_intrin(Vec& sinx, Vec& cosx, const Vec& x) {
    // ORDER    ERROR
    //     1 8.81e-02
    //     3 2.45e-03
    //     5 3.63e-05
    //     7 3.11e-07
    //     9 1.75e-09
    //    11 6.93e-12
    //    13 2.09e-14
    //    15 6.66e-16
    //    17 6.66e-16

    using Real = typename Vec::ScalarType;
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Real coeff3  = -1/(((Real)2)*3);
    static constexpr Real coeff5  =  1/(((Real)2)*3*4*5);
    static constexpr Real coeff7  = -1/(((Real)2)*3*4*5*6*7);
    static constexpr Real coeff9  =  1/(((Real)2)*3*4*5*6*7*8*9);
    static constexpr Real coeff11 = -1/(((Real)2)*3*4*5*6*7*8*9*10*11);
    static constexpr Real coeff13 =  1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13);
    static constexpr Real coeff15 = -1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13*14*15);
    static constexpr Real coeff17 =  1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16*17);
    static constexpr Real coeff19 = -1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16*17*18*19);
    static constexpr Real x0 = (Real)1.570796326794896619231321691639l;
    static constexpr Real invx0 = 1 / x0;

    Vec x_ = RoundReal2Real(x * invx0); // 4.5 - cycles
    Vec x1 = x - x_ * x0; // 2 - cycles
    Vec x2, x3, x5, x7, x9, x11, x13, x15, x17, x19;

    Vec s1 = x1;
    if (ORDER >= 3) { // 5 - cycles
      x2 = x1 * x1;
      x3 = x1 * x2;
      s1 += x3 * coeff3;
    }
    if (ORDER >= 5) { // 3 - cycles
      x5 = x3 * x2;
      s1 += x5 * coeff5;
    }
    if (ORDER >= 7) {
      x7 = x5 * x2;
      s1 += x7 * coeff7;
    }
    if (ORDER >= 9) {
      x9 = x7 * x2;
      s1 += x9 * coeff9;
    }
    if (ORDER >= 11) {
      x11 = x9 * x2;
      s1 += x11 * coeff11;
    }
    if (ORDER >= 13) {
      x13 = x11 * x2;
      s1 += x13 * coeff13;
    }
    if (ORDER >= 15) {
      x15 = x13 * x2;
      s1 += x15 * coeff15;
    }
    if (ORDER >= 17) {
      x17 = x15 * x2;
      s1 += x17 * coeff17;
    }
    if (ORDER >= 19) {
      x19 = x17 * x2;
      s1 += x19 * coeff19;
    }

    Vec cos_squared = (Real)1.0 - s1 * s1;
    Vec inv_cos = approx_rsqrt(cos_squared); // 1.5 - cycles
    if (ORDER < 5) {
    } else if (ORDER < 9) {
      inv_cos *= ((3.0) - cos_squared * inv_cos * inv_cos) * 0.5; // 7 - cycles
    } else if (ORDER < 15) {
      inv_cos *= ((3.0) - cos_squared * inv_cos * inv_cos); // 7 - cycles
      inv_cos *= ((3.0 * pow<pow<0>(3)*3-1>(2.0)) - cos_squared * inv_cos * inv_cos) * (pow<(pow<0>(3)*3-1)*3/2+1>(0.5)); // 8 - cycles
    } else {
      inv_cos *= ((3.0) - cos_squared * inv_cos * inv_cos); // 7 - cycles
      inv_cos *= ((3.0 * pow<pow<0>(3)*3-1>(2.0)) - cos_squared * inv_cos * inv_cos); // 7 - cycles
      inv_cos *= ((3.0 * pow<pow<1>(3)*3-1>(2.0)) - cos_squared * inv_cos * inv_cos) * (pow<(pow<1>(3)*3-1)*3/2+1>(0.5)); // 8 - cycles
    }
    Vec c1 = cos_squared * inv_cos; // 1 - cycle

    union {
      unsigned char int_one = 1;
      Real real_one;
    };
    union {
      unsigned char int_two = 2;
      Real real_two;
    };
    union {
      int64_t Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    Vec x_offset(Creal);
    Vec xAnd1 = (((x_+x_offset) & Vec(real_one)) == Vec::Zero());
    Vec xAnd2 = (((x_+x_offset) & Vec(real_two)) == Vec::Zero());

    Vec s2 = AndNot( c1,xAnd1) | (s1 & xAnd1);
    Vec c2 = AndNot(-s1,xAnd1) | (c1 & xAnd1);
    Vec s3 = AndNot(-s2,xAnd2) | (s2 & xAnd2);
    Vec c3 = AndNot(-c2,xAnd2) | (c2 & xAnd2);

    sinx = s3;
    cosx = c3;
  }
  template <class RealVec, class IntVec> RealVec ConvertInt2Real(const IntVec& x) {
    typedef typename RealVec::ScalarType Real;
    typedef typename IntVec::ScalarType Int;
    assert(sizeof(RealVec) == sizeof(IntVec));
    assert(sizeof(Real) == sizeof(Int));
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    union {
      Int Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    IntVec l(x + IntVec(Cint));
    return *(RealVec*)&l - RealVec(Creal);
  }
  template <class IntVec, class RealVec> IntVec RoundReal2Int(const RealVec& x) {
    typedef typename RealVec::ScalarType Real;
    typedef typename IntVec::ScalarType Int;
    assert(sizeof(RealVec) == sizeof(IntVec));
    assert(sizeof(Real) == sizeof(Int));
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    union {
      Int Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    RealVec d(x + RealVec(Creal));
    return *(IntVec*)&d - IntVec(Cint);
  }
  template <class Vec> Vec RoundReal2Real(const Vec& x) {
    typedef typename Vec::ScalarType Real;
    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    union {
      int64_t Cint = (1UL << (SigBits - 1)) + ((SigBits + ((1UL<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    Vec Vreal(Creal);
    return (x + Vreal) - Vreal;
  }

#ifdef __AVX__
  template <> class alignas(sizeof(double)*4) Vec<double,4> {
    typedef double ValueType;
    static constexpr Integer N = 4;
    public:

      typedef ValueType ScalarType;

      static constexpr Integer Size() {
        return N;
      }

      static Vec Zero() {
        Vec r;
        r.v = _mm256_setzero_pd();
        return r;
      }

      static Vec Load1(ValueType const* p) {
        Vec r;
        r.v = _mm256_broadcast_sd(p);
        return r;
      }
      static Vec Load(ValueType const* p) {
        Vec r;
        r.v = _mm256_loadu_pd(p);
        return r;
      }
      static Vec LoadAligned(ValueType const* p) {
        Vec r;
        r.v = _mm256_load_pd(p);
        return r;
      }

      Vec() {}

      Vec(const ValueType& a) {
        v = _mm256_set1_pd(a);
      }

      void Store(ValueType* p) const {
        _mm256_storeu_pd(p, v);
      }
      void StoreAligned(ValueType* p) const {
        _mm256_store_pd(p, v);
      }

      // Bitwise NOT
      Vec operator~() const {
        Vec r;
        static constexpr ScalarType Creal = -1.0;
        r.v = _mm256_xor_pd(v, _mm256_set1_pd(Creal));
        return r;
      }

      // Unary plus and minus
      Vec operator+() const {
        return *this;
      }
      Vec operator-() const {
        return Zero() - (*this);
      }

      // C-style cast
      template <class RetValueType> explicit operator Vec<RetValueType,N>() const {
        Vec<RetValueType,N> r;
        __m256d& ret_v = *(__m256d*)&r.v;
        ret_v = v;
        return r;
      }

      // Arithmetic operators
      friend Vec operator*(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_mul_pd(lhs.v, rhs.v);
        return lhs;
      }
      friend Vec operator+(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_add_pd(lhs.v, rhs.v);
        return lhs;
      }
      friend Vec operator-(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_sub_pd(lhs.v, rhs.v);
        return lhs;
      }

      // Comparison operators
      friend Vec operator< (Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LT_OS);
        return lhs;
      }
      friend Vec operator<=(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LE_OS);
        return lhs;
      }
      friend Vec operator>=(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GE_OS);
        return lhs;
      }
      friend Vec operator> (Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GT_OS);
        return lhs;
      }
      friend Vec operator==(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_EQ_OS);
        return lhs;
        return lhs;
      }
      friend Vec operator!=(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NEQ_OS);
        return lhs;
      }

      // Bitwise operators
      friend Vec operator&(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_and_pd(lhs.v, rhs.v);
        return lhs;
      }
      friend Vec operator^(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_xor_pd(lhs.v, rhs.v);
        return lhs;
      }
      friend Vec operator|(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_or_pd(lhs.v, rhs.v);
        return lhs;
      }
      friend Vec AndNot(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_andnot_pd(rhs.v, lhs.v);
        return lhs;
      }

      // Assignment operators
      Vec& operator*=(const Vec& rhs) {
        v = _mm256_mul_pd(v, rhs.v);
        return *this;
      }
      Vec& operator+=(const Vec& rhs) {
        v = _mm256_add_pd(v, rhs.v);
        return *this;
      }
      Vec& operator-=(const Vec& rhs) {
        v = _mm256_sub_pd(v, rhs.v);
        return *this;
      }
      Vec& operator&=(const Vec& rhs) {
        v = _mm256_and_pd(v, rhs.v);
        return *this;
      }
      Vec& operator^=(const Vec& rhs) {
        v = _mm256_xor_pd(v, rhs.v);
        return *this;
      }
      Vec& operator|=(const Vec& rhs) {
        v = _mm256_or_pd(v, rhs.v);
        return *this;
      }

      // Other operators
      friend Vec max(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_max_pd(lhs.v, rhs.v);
        return lhs;
      }
      friend Vec min(Vec lhs, const Vec& rhs) {
        lhs.v = _mm256_min_pd(lhs.v, rhs.v);
        return lhs;
      }

      friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
        union {
          __m256d vec;
          ValueType val[N];
        };
        vec = in.v;
        for (Integer i = 0; i < N; i++) os << val[i] << ' ';
        return os;
      }
      friend Vec RoundReal2Real(const Vec& x) {
        Vec r;
        r.v = _mm256_round_pd(x.v,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        return r;
      }
      friend Vec approx_rsqrt(const Vec& x) {
        Vec r;
        r.v = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x.v)));
        return r;
      }

    private:

      __m256d v;
  };
#endif

}

#endif  //_SCTL_VEC_WRAPPER_HPP_
