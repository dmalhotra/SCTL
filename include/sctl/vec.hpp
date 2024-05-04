#ifndef _SCTL_VEC_WRAPPER_HPP_
#define _SCTL_VEC_WRAPPER_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(intrin-wrapper.hpp)

#include <cassert>
#include <cstdint>
#include <ostream>

namespace SCTL_NAMESPACE {

  template <class ScalarType> constexpr Integer DefaultVecLen();

  template <class ValueType, Integer N = DefaultVecLen<ValueType>()> class alignas(sizeof(ValueType) * N) Vec {
    public:
      using ScalarType = ValueType;
      using VData = VecData<ScalarType,N>;
      using MaskType = Mask<VData>;

      static constexpr Integer Size();

      static inline Vec Zero();

      static inline Vec Load1(ScalarType const* p);
      static inline Vec Load(ScalarType const* p);
      static inline Vec LoadAligned(ScalarType const* p);

      Vec() = default;
      Vec(const Vec&) = default;
      Vec& operator=(const Vec&) = default;
      ~Vec() = default;

      inline Vec(const VData& v_);
      inline Vec(const ScalarType& a);
      template <class T,class ...T1> inline Vec(T x, T1... args);

      inline void Store(ScalarType* p) const;
      inline void StoreAligned(ScalarType* p) const;

      // Element access
      inline ScalarType operator[](Integer i) const;
      inline void insert(Integer i, ScalarType value);

      // Arithmetic operators
      inline Vec operator+() const;
      inline Vec operator-() const;

      // Bitwise operators
      inline Vec operator~() const;

      // Assignment operators
      inline Vec& operator=(const ScalarType& a);
      inline Vec& operator*=(const Vec& rhs);
      inline Vec& operator/=(const Vec& rhs);
      inline Vec& operator+=(const Vec& rhs);
      inline Vec& operator-=(const Vec& rhs);
      inline Vec& operator&=(const Vec& rhs);
      inline Vec& operator^=(const Vec& rhs);
      inline Vec& operator|=(const Vec& rhs);

      inline void set(const VData& v_) { v = v_; }
      inline const VData& get() const { return v; }
      inline VData& get() { return v; }

    private:

      template <class T, class... T2> struct InitVec;
      //template <class T> struct InitVec<T>;

      VData v;
  };

  // Conversion operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType convert2mask(const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> RoundReal2Real(const Vec<ValueType,N>& x);
  template <class RealVec, class IntVec> inline RealVec ConvertInt2Real(const IntVec& x);
  template <class IntVec, class RealVec> inline IntVec RoundReal2Int(const RealVec& x);
  template <class MaskType> inline Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a);
  //template <class Vec1, class Vec2> friend Vec1 reinterpret(const Vec2& x);


  // Arithmetic operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> FMA(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b, const Vec<ValueType,N>& c);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const ValueType& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const ValueType& a, const Vec<ValueType,N>& b);


  // Comparison operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const ValueType& b);

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const ValueType& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const ValueType& a, const Vec<ValueType,N>& b);


  // Bitwise operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const ValueType& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const ValueType& b, const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const ValueType& b, const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const ValueType& b, const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const ValueType& b, const Vec<ValueType,N>& a);


  // Bitshift
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Integer& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Integer& rhs);


  // Other operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const ValueType& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const ValueType& rhs);

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const ValueType& lhs, const Vec<ValueType,N>& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const ValueType& lhs, const Vec<ValueType,N>& rhs);


  // Special functions
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m);

  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m);

  template <class ValueType, Integer N> inline void sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline void approx_sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x);

  template <class ValueType, Integer N> inline Vec<ValueType,N> exp(const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_exp(const Vec<ValueType,N>& x);

  #if defined(SCTL_HAVE_SVML) || defined(SCTL_HAVE_LIBMVEC)
  template <class ValueType, Integer N> inline Vec<ValueType,N> log(const Vec<ValueType,N>& x);
  #endif


  // Print
  template <Integer digits, class ValueType, Integer N> inline std::ostream& operator<<(std::ostream& os, const Vec<ValueType,N>& in);


  // Other operators
  template <class ValueType> inline void printb(const ValueType& x);

}

#include SCTL_INCLUDE(vec.txx)

#endif  //_SCTL_VEC_WRAPPER_HPP_
