#ifndef _SCTL_VEC_TXX_
#define _SCTL_VEC_TXX_

#include <ostream>                  // for ostream
#include <cstdint>                  // for uintptr_t, uint8_t
#include <iostream>                 // for basic_ostream, cout, operator<<
#include <type_traits>              // for is_same

#include "sctl/common.hpp"          // for Integer, SCTL_ASSERT, SCTL_ALIGN_...
#include "sctl/vec.hpp"             // for Vec, AndNot, max, min, operator!=
#include "sctl/intrin-wrapper.hpp"  // for ComparisonType, comp_intrin, Type...

namespace sctl {

  template <class ScalarType> constexpr Integer DefaultVecLen() {
    #if defined(__AVX512__) || defined(__AVX512F__)
    static_assert(SCTL_ALIGN_BYTES >= 64, "Insufficient memory alignment for SIMD vector types");
    return 64/sizeof(ScalarType);
    #elif defined(__AVX__)
    static_assert(SCTL_ALIGN_BYTES >= 32, "Insufficient memory alignment for SIMD vector types");
    return 32/sizeof(ScalarType);
    #elif defined(__SSE4_2__) || defined(__ARM_NEON)
    static_assert(SCTL_ALIGN_BYTES >= 16, "Insufficient memory alignment for SIMD vector types");
    return 16/sizeof(ScalarType);
    #else
    static_assert(SCTL_ALIGN_BYTES >= 8, "Insufficient memory alignment for SIMD vector types");
    return 1;
    #endif
  }

  template <class ValueType, Integer N> constexpr Integer Vec<ValueType,N>::Size() {
    return N;
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::Zero() noexcept {
    Vec<ValueType,N> r;
    r.v = zero_intrin<VData>();
    return r;
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::Load1(ScalarType const* p) {
    Vec<ValueType,N> r;
    r.v = load1_intrin<VData>(p);
    return r;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::Load(ScalarType const* p) {
    Vec<ValueType,N> r;
    r.v = loadu_intrin<VData>(p);
    return r;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::LoadAligned(ScalarType const* p) {
    #ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(((uintptr_t)p) % (sizeof(ValueType)*N) == 0);
    #endif
    Vec<ValueType,N> r;
    r.v = load_intrin<VData>(p);
    return r;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::Load(ScalarType const* p, const MaskType& m) {
    Vec<ValueType,N> r;
    r.v = loadu_mask_intrin<VData>(p, m);
    return r;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::LoadPartial(ScalarType const* p, Integer n) {
    Vec<ValueType,N> r;
    r.v = loadu_first_intrin<VData>(p, n);
    return r;
  }
  template <class ValueType, Integer N> template <class IndexType> inline Vec<ValueType,N> Vec<ValueType,N>::Gather(ScalarType const* p, const Vec<IndexType,N>& idx) {
    Vec<ValueType,N> r;
    r.v = gather_intrin<VData>(p, idx.get());
    return r;
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N>::Vec(const VData& v_) : v(v_) {}
  template <class ValueType, Integer N> inline Vec<ValueType,N>::Vec(const ScalarType& a) : Vec(set1_intrin<VData>(a)) {}
  template <class ValueType, Integer N> template <class T0, class T1, class ...T2> inline Vec<ValueType,N>::Vec(T0 x0, T1 x1, T2... args) : Vec(set_intrin<VData>((ScalarType)x0, (ScalarType)x1, ((ScalarType)args)...)) {}
  template <class ValueType, Integer N> inline Vec<ValueType,N>::Vec(const Vec<ValueType,N/2>& lo, const Vec<ValueType,N/2>& hi) : Vec(concat_intrin(lo.get(), hi.get())) {}


  template <class ValueType, Integer N> inline void Vec<ValueType,N>::Store(ScalarType* p) const {
    storeu_intrin(p,v);
  }
  template <class ValueType, Integer N> inline void Vec<ValueType,N>::StoreAligned(ScalarType* p) const {
    #ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(((uintptr_t)p) % (sizeof(ValueType)*N) == 0);
    #endif
    store_intrin(p,v);
  }
  template <class ValueType, Integer N> inline void Vec<ValueType,N>::Store(ScalarType* p, const MaskType& m) const {
    storeu_mask_intrin(p, v, m);
  }
  template <class ValueType, Integer N> inline void Vec<ValueType,N>::StorePartial(ScalarType* p, Integer n) const {
    storeu_first_intrin(p, v, n);
  }
  template <class ValueType, Integer N> template <class IndexType> inline void Vec<ValueType,N>::Scatter(ScalarType* p, const Vec<IndexType,N>& idx) const {
    scatter_intrin(p, v, idx.get());
  }

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::ScalarType Vec<ValueType,N>::operator[](Integer i) const {
    return extract_intrin(v,i);
  }
  template <class ValueType, Integer N> inline void Vec<ValueType,N>::insert(Integer i, ScalarType value) {
    insert_intrin(v,i,value);
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::operator+() const {
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::operator-() const {
    return unary_minus_intrin(v);
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::operator~() const {
    return not_intrin(v);
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator= (const ValueType& a) {
    v = set1_intrin<VData>(a);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator*=(const Vec<ValueType,N>& rhs) {
    v = mul_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator/=(const Vec<ValueType,N>& rhs) {
    v = div_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator%=(const Vec<ValueType,N>& rhs) {
    static_assert(TypeTraits<ValueType>::Type == DataType::Integer, "The remainder requires an integer type.");
    v = rem_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator+=(const Vec<ValueType,N>& rhs) {
    v = add_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator-=(const Vec<ValueType,N>& rhs) {
    v = sub_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator&=(const Vec<ValueType,N>& rhs) {
    v = and_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator^=(const Vec<ValueType,N>& rhs) {
    v = xor_intrin(v, rhs.v);
    return *this;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N>& Vec<ValueType,N>::operator|=(const Vec<ValueType,N>& rhs) {
    v = or_intrin(v, rhs.v);
    return *this;
  }


  template <class ValueType, Integer N> inline void Vec<ValueType,N>::set(const VData& v_) {
    v = v_;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::VData& Vec<ValueType,N>::get() {
    return v;
  }
  template <class ValueType, Integer N> inline const typename Vec<ValueType,N>::VData& Vec<ValueType,N>::get() const {
    return v;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N/2> Vec<ValueType,N>::get_low() const {
    static_assert(N >= 2, "get_low requires at least two elements.");
    return get_low_intrin(v);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N/2> Vec<ValueType,N>::get_high() const {
    static_assert(N >= 2, "get_high requires at least two elements.");
    return get_high_intrin(v);
  }




  // Conversion operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType convert2mask(const Vec<ValueType,N>& a) {
    return convert_vec2mask_intrin(a.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> rint(const Vec<ValueType,N>& x) {
    return rint_intrin(x.get());
  }
  template <class RealVec, class IntVec> inline RealVec ConvertInt2Real(const IntVec& x) {
    return convert_int2real_intrin<typename RealVec::VData>(x.get());
  }
  template <class IntVec, class RealVec> inline IntVec lrint(const RealVec& x) {
    return lrint_intrin<typename IntVec::VData>(x.get());
  }
  template <class MaskType> inline Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a) {
    return convert_mask2vec_intrin(a);
  }
  template <class VecTo, class ValueType, Integer N> inline VecTo Convert(const Vec<ValueType,N>& x) {
    return convert_intrin<typename VecTo::VData>(x.get());
  }
  template <class VecTo, class MaskType> inline typename VecTo::MaskType ConvertMask(const MaskType& m) {
    return convert_mask_intrin<typename VecTo::VData>(m);
  }
  //template <class Vec1, class Vec2> friend Vec1 reinterpret(const Vec2& x);


  // Arithmetic operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> FMA(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b, const Vec<ValueType,N>& c) {
    return fma_intrin(a.get(), b.get(), c.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return mul_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return div_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return add_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return sub_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator%(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    static_assert(TypeTraits<ValueType>::Type == DataType::Integer, "The remainder requires an integer type.");
    return rem_intrin(a.get(), b.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a * Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a / Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a + Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a - Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator%(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a % Vec<ValueType,N>(b);
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) * b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) / b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) + b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) - b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator%(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) % b;
  }


  // Comparison operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return comp_intrin<ComparisonType::lt>(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return comp_intrin<ComparisonType::le>(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return comp_intrin<ComparisonType::ge>(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return comp_intrin<ComparisonType::gt>(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return comp_intrin<ComparisonType::eq>(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return comp_intrin<ComparisonType::ne>(a.get(), b.get());
  }

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a <  Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a <= Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a >= Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a > Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a == Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a != Vec<ValueType,N>(b);
  }

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) <  b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) <= b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) >= b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) >  b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) == b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) != b;
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return select_intrin(m, a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return select(m, a, Vec<ValueType,N>(b));
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return select(m, Vec<ValueType,N>(a), b);
  }


  // Bitwise operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return and_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return xor_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return or_intrin(a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) { // return a & ~b
    return andnot_intrin(a.get(), b.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a & Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a ^ Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) {
    return a | Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b) { // return a & ~b
    return AndNot(a, Vec<ValueType,N>(b));
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) & b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) ^ b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) | b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b) { // return a & ~b
    return AndNot(Vec<ValueType,N>(a), b);
  }


  // Bitshift
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Integer& rhs) {
    return bitshiftleft_intrin(lhs.get(), rhs);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Integer& rhs) {
    return bitshiftright_intrin(lhs.get(), rhs);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs) {
    static_assert(TypeTraits<ValueType>::Type == DataType::Integer, "Bit shift counts per element require an integer type.");
    return bitshiftleft_intrin(lhs.get(), rhs.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs) {
    static_assert(TypeTraits<ValueType>::Type == DataType::Integer, "Bit shift counts per element require an integer type.");
    return bitshiftright_intrin(lhs.get(), rhs.get());
  }


  // Other operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs) {
    return max_intrin(lhs.get(), rhs.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs) {
    return min_intrin(lhs.get(), rhs.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const typename Vec<ValueType,N>::ScalarType& rhs) {
    return max(lhs, Vec<ValueType,N>(rhs));
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const typename Vec<ValueType,N>::ScalarType& rhs) {
    return min(lhs, Vec<ValueType,N>(rhs));
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const typename Vec<ValueType,N>::ScalarType& lhs, const Vec<ValueType,N>& rhs) {
    return max(Vec<ValueType,N>(lhs), rhs);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const typename Vec<ValueType,N>::ScalarType& lhs, const Vec<ValueType,N>& rhs) {
    return min(Vec<ValueType,N>(lhs), rhs);
  }

  template <class ValueType, Integer N> inline void transpose(Vec<ValueType,N> (&v)[N]) {
    VecData<ValueType,N> w[N];
    for (Integer i = 0; i < N; i++) w[i] = v[i].get();
    transpose_intrin(w);
    for (Integer i = 0; i < N; i++) v[i].set(w[i]);
  }

  template <class ValueType, Integer N, class ...T> inline void transpose(Vec<ValueType,N>& v0, T&... vs) {
    static_assert(sizeof...(T)+1 == N, "transpose requires exactly N vectors.");
    static_assert(((std::is_same<T,Vec<ValueType,N>>::value) && ...), "all arguments must have the same Vec type.");
    VecData<ValueType,N> w[N] = {v0.get(), vs.get()...};
    transpose_intrin(w);
    Integer i = 0;
    v0.set(w[i++]);
    ((vs.set(w[i++])), ...);
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> swap_pairs(const Vec<ValueType,N>& x) {
    return swap_pairs_intrin(x.get());
  }


  // Reductions
  template <class ValueType, Integer N> inline ValueType reduce(const Vec<ValueType,N>& v) {
    return reduce_add_intrin(v.get());
  }
  template <class ValueType, Integer N> inline ValueType reduce_min(const Vec<ValueType,N>& v) {
    return reduce_min_intrin(v.get());
  }
  template <class ValueType, Integer N> inline ValueType reduce_max(const Vec<ValueType,N>& v) {
    return reduce_max_intrin(v.get());
  }
  template <class VData> inline Integer reduce_count(const Mask<VData>& m) {
    return mask_count_intrin(m);
  }
  template <class VData> inline bool all_of(const Mask<VData>& m) {
    return mask_count_intrin(m) == VData::Size;
  }
  template <class VData> inline bool any_of(const Mask<VData>& m) {
    return mask_count_intrin(m) != 0;
  }
  template <class VData> inline bool none_of(const Mask<VData>& m) {
    return mask_count_intrin(m) == 0;
  }


  // Special functions
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x) {
    static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<ValueType>::SigBits*0.3010299957) : digits);
    return rsqrt_approx_intrin<digits_, typename Vec<ValueType,N>::VData>::eval(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m) {
    static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<ValueType>::SigBits*0.3010299957) : digits);
    return rsqrt_approx_intrin<digits_, typename Vec<ValueType,N>::VData>::eval(x.get(), m);
  }

  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x) {
    return x*approx_rsqrt<digits>(x);
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m) {
    return x*approx_rsqrt<digits>(x, m);
  }

  namespace detail_approx_digits { // which of the approximate or the full-precision routines serves the given digits
    template <class ValueType> inline constexpr bool full(const Integer digits) { // the full-precision routine is as accurate, or the type is not float or double
      return digits < 0 || digits >= (std::is_same<ValueType,float>::value ? 7 : 15) || !(std::is_same<ValueType,float>::value || std::is_same<ValueType,double>::value);
    }
    template <class ValueType> inline constexpr bool full_exp(const Integer digits) { // as full, but the Taylor polynomial of approx_exp_intrin also serves other types, up to order 13
      return digits < 0 || exp_taylor_order(digits) > 13 || ((std::is_same<ValueType,float>::value || std::is_same<ValueType,double>::value) && full<ValueType>(digits));
    }
  }
  template <bool FullRange, class ValueType, Integer N> inline void sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x) {
#ifndef SCTL_HAVE_SVML
    if constexpr (!FullRange && (std::is_same<ValueType,float>::value || std::is_same<ValueType,double>::value)) {
      approx_sincos_intrin<-1, false>(sinx.get(), cosx.get(), x.get());
      return;
    }
#endif
    sincos_intrin(sinx.get(), cosx.get(), x.get());
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline void approx_sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x) {
    if constexpr (detail_approx_digits::full<ValueType>(digits)) sincos<FullRange>(sinx, cosx, x);
    else approx_sincos_intrin<digits, FullRange>(sinx.get(), cosx.get(), x.get());
  }
  template <class ValueType, Integer N> inline void sincospi(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x) {
    approx_sincospi_intrin<-1>(sinx.get(), cosx.get(), x.get());
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline void approx_sincospi(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x) {
    approx_sincospi_intrin<digits, FullRange>(sinx.get(), cosx.get(), x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> sinpi(const Vec<ValueType,N>& x) {
    return approx_sinpi<-1>(x);
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_sinpi(const Vec<ValueType,N>& x) {
    typename Vec<ValueType,N>::VData sinx, cosx;
    approx_sincospi_intrin<digits, FullRange, true, false>(sinx, cosx, x.get());
    return sinx;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> cospi(const Vec<ValueType,N>& x) {
    return approx_cospi<-1>(x);
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_cospi(const Vec<ValueType,N>& x) {
    typename Vec<ValueType,N>::VData sinx, cosx;
    approx_sincospi_intrin<digits, FullRange, false, true>(sinx, cosx, x.get());
    return cosx;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> sincpi(const Vec<ValueType,N>& x) {
    return approx_sincpi_intrin<-1>(x.get());
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_sincpi(const Vec<ValueType,N>& x) {
    return approx_sincpi_intrin<digits, FullRange>(x.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> exp(const Vec<ValueType,N>& x) {
    return exp_intrin(x.get());
  }

  template <Integer digits, bool RangeCheck, class ValueType, Integer N> inline Vec<ValueType,N> approx_exp(const Vec<ValueType,N>& x) {
    if constexpr (detail_approx_digits::full_exp<ValueType>(digits)) return exp(x);
    else return approx_exp_intrin<exp_taylor_order(digits), RangeCheck>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> exp10(const Vec<ValueType,N>& x) {
    return exp10_intrin(x.get());
  }
  template <Integer digits, bool RangeCheck, class ValueType, Integer N> inline Vec<ValueType,N> approx_exp10(const Vec<ValueType,N>& x) {
    if constexpr (detail_approx_digits::full_exp<ValueType>(digits)) return exp10(x);
    else return approx_exp10_intrin<exp_taylor_order(digits), RangeCheck>(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_log(const Vec<ValueType,N>& x) {
    if constexpr (detail_approx_digits::full<ValueType>(digits) || log_poly_degree(digits) == 0) return log(x); // beyond the polynomials, log is as fast
    else return approx_log_intrin<digits>(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_pow(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y) {
    if constexpr (detail_approx_digits::full<ValueType>(digits) || digits >= (std::is_same<ValueType,float>::value ? 6 : 14)) return pow(x, y); // from these digits, y log x rounded to ValueType is not accurate enough
    else return approx_pow_intrin<digits + 2, exp_taylor_order(digits)>(x.get(), y.get()); // two more digits in log: |y log x| up to 100
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> log(const Vec<ValueType,N>& x) {
    return log_intrin(x.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> sqrt(const Vec<ValueType,N>& x) {
    return sqrt_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> rsqrt(const Vec<ValueType,N>& x) {
    return rsqrt_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> fabs(const Vec<ValueType,N>& x) {
    return fabs_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> floor(const Vec<ValueType,N>& x) {
    return floor_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> ceil(const Vec<ValueType,N>& x) {
    return ceil_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> copysign(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y) {
    static_assert(TypeTraits<ValueType>::Type == DataType::Real, "copysign requires a real type.");
    return copysign_intrin(x.get(), y.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType isnan(const Vec<ValueType,N>& x) {
    return isnan_intrin(x.get());
  }

  template <bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> sin(const Vec<ValueType,N>& x) {
    Vec<ValueType,N> sinx, cosx;
    sincos<FullRange>(sinx, cosx, x);
    return sinx;
  }
  template <bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> cos(const Vec<ValueType,N>& x) {
    Vec<ValueType,N> sinx, cosx;
    sincos<FullRange>(sinx, cosx, x);
    return cosx;
  }
  template <bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> tan(const Vec<ValueType,N>& x) {
    Vec<ValueType,N> sinx, cosx;
    sincos<FullRange>(sinx, cosx, x);
    return sinx / cosx;
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_sin(const Vec<ValueType,N>& x) {
    Vec<ValueType,N> sinx, cosx;
    approx_sincos<digits, FullRange>(sinx, cosx, x);
    return sinx;
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_cos(const Vec<ValueType,N>& x) {
    Vec<ValueType,N> sinx, cosx;
    approx_sincos<digits, FullRange>(sinx, cosx, x);
    return cosx;
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_tan(const Vec<ValueType,N>& x) {
    Vec<ValueType,N> sinx, cosx;
    approx_sincos<digits, FullRange>(sinx, cosx, x);
    return sinx / cosx;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> sinc(const Vec<ValueType,N>& x) {
    return approx_sinc<-1>(x);
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_sinc(const Vec<ValueType,N>& x) {
    return select(x == Vec<ValueType,N>((ValueType)0), Vec<ValueType,N>((ValueType)1), approx_sin<digits, FullRange>(x) / x);
  }

  template <bool SpecialValues, class ValueType, Integer N> inline Vec<ValueType,N> atan2(const Vec<ValueType,N>& y, const Vec<ValueType,N>& x) {
    return atan2_intrin<SpecialValues>(y.get(), x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> pow(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y) {
    return pow_intrin(x.get(), y.get());
  }
  template <Long e, class ValueType, Integer N> inline Vec<ValueType,N> pow(const Vec<ValueType,N>& x) {
    if constexpr (e < 0) {
      return (ValueType)1 / pow<-e>(x);
    } else if constexpr (e == 0) {
      return Vec<ValueType,N>((ValueType)1);
    } else { // the product of the set bits from the lowest, alongside the squarings
      constexpr Long low = e & -e; // the lowest set bit
      Vec<ValueType,N> r = x;
      for (Long k = 1; k < low; k <<= 1) r = r * r;
      Vec<ValueType,N> p = r;
      for (Long k = e / low; k > 1;) {
        k >>= 1;
        p = p * p;
        if (k & 1) r = r * p;
      }
      return r;
    }
  }
  template <Long e, Long d, class ValueType, Integer N> inline Vec<ValueType,N> pow(const Vec<ValueType,N>& x) {
    static_assert(d == 1 || d == 2, "pow<e, d> requires d = 1 or 2.");
    if constexpr (d == 1 || e % 2 == 0) return pow<e / d>(x);
    else if constexpr (e > 0) return pow<(e - 1) / 2>(x) * sqrt(x);
    else return (ValueType)1 / (pow<(-e - 1) / 2>(x) * sqrt(x));
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> hermite(const Vec<ValueType,N>& x) {
    static_assert(n >= 0, "hermite<n> requires n >= 0.");
    Vec<ValueType,N> h0((ValueType)1);
    if constexpr (n == 0) {
      return h0;
    } else {
      const Vec<ValueType,N> x2 = x + x;
      Vec<ValueType,N> h1 = x2;
      for (Integer k = 1; k < n; k++) {
        const Vec<ValueType,N> h2 = FMA(x2, h1, h0 * (ValueType)(-2 * k));
        h0 = h1;
        h1 = h2;
      }
      return h1;
    }
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> trunc(const Vec<ValueType,N>& x) {
    return trunc_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> round(const Vec<ValueType,N>& x) {
    static_assert(TypeTraits<ValueType>::Type == DataType::Real, "round requires a real type.");
    return round_intrin(x.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType isinf(const Vec<ValueType,N>& x) {
    return isinf_intrin(x.get());
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType isfinite(const Vec<ValueType,N>& x) {
    return isfinite_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> atan(const Vec<ValueType,N>& x) {
    return atan_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> asin(const Vec<ValueType,N>& x) {
    return asin_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> acos(const Vec<ValueType,N>& x) {
    return acos_intrin(x.get());
  }
  template <bool AvoidOverflow, class ValueType, Integer N> inline Vec<ValueType,N> hypot(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y) {
    return hypot_intrin<AvoidOverflow>(x.get(), y.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> exp2(const Vec<ValueType,N>& x) {
    return exp2_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> log2(const Vec<ValueType,N>& x) {
    return log2_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> log10(const Vec<ValueType,N>& x) {
    return log10_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> cbrt(const Vec<ValueType,N>& x) {
    return cbrt_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> sinh(const Vec<ValueType,N>& x) {
    return sinh_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> cosh(const Vec<ValueType,N>& x) {
    return cosh_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> tanh(const Vec<ValueType,N>& x) {
    return tanh_intrin(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> asinh(const Vec<ValueType,N>& x) {
    return asinh_intrin(x.get());
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_asinh(const Vec<ValueType,N>& x) {
    return asinh_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits), FullRange>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> acosh(const Vec<ValueType,N>& x) {
    return acosh_intrin(x.get());
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_acosh(const Vec<ValueType,N>& x) {
    return acosh_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits), FullRange>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> atanh(const Vec<ValueType,N>& x) {
    return atanh_intrin(x.get());
  }
  template <Integer digits, bool FullRange, class ValueType, Integer N> inline Vec<ValueType,N> approx_atanh(const Vec<ValueType,N>& x) {
    return atanh_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits), FullRange>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> erf(const Vec<ValueType,N>& x) {
    return erf_intrin(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_erf(const Vec<ValueType,N>& x) {
    return erf_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> erfc(const Vec<ValueType,N>& x) {
    return erfc_intrin(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_erfc(const Vec<ValueType,N>& x) {
    return erfc_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> ndtri(const Vec<ValueType,N>& p) {
    return ndtri_intrin(p.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_ndtri(const Vec<ValueType,N>& p) {
    return ndtri_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(p.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> erfinv(const Vec<ValueType,N>& y) {
    return erfinv_intrin(y.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_erfinv(const Vec<ValueType,N>& y) {
    return erfinv_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(y.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> tgamma(const Vec<ValueType,N>& x) {
    return tgamma_intrin(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_tgamma(const Vec<ValueType,N>& x) {
    return tgamma_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> lgamma(const Vec<ValueType,N>& x) {
    return lgamma_intrin(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_lgamma(const Vec<ValueType,N>& x) {
    return lgamma_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> digamma(const Vec<ValueType,N>& x) {
    return digamma_intrin(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_digamma(const Vec<ValueType,N>& x) {
    return digamma_intrin<(detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> cyl_bessel_j(const Vec<ValueType,N>& x) {
    return cyl_bessel_j_intrin<n>(x.get());
  }
  template <Integer n, Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_cyl_bessel_j(const Vec<ValueType,N>& x) {
    return cyl_bessel_j_intrin<n, (detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> cyl_neumann(const Vec<ValueType,N>& x) {
    return cyl_neumann_intrin<n>(x.get());
  }
  template <Integer n, Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_cyl_neumann(const Vec<ValueType,N>& x) {
    return cyl_neumann_intrin<n, (detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> cyl_bessel_i(const Vec<ValueType,N>& x) {
    return cyl_bessel_i_intrin<n>(x.get());
  }
  template <Integer n, Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_cyl_bessel_i(const Vec<ValueType,N>& x) {
    return cyl_bessel_i_intrin<n, (detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> cyl_bessel_k(const Vec<ValueType,N>& x) {
    return cyl_bessel_k_intrin<n>(x.get());
  }
  template <Integer n, Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_cyl_bessel_k(const Vec<ValueType,N>& x) {
    return cyl_bessel_k_intrin<n, (detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> sph_bessel(const Vec<ValueType,N>& x) {
    return sph_bessel_intrin<n>(x.get());
  }
  template <Integer n, Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sph_bessel(const Vec<ValueType,N>& x) {
    return sph_bessel_intrin<n, (detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <Integer n, class ValueType, Integer N> inline Vec<ValueType,N> sph_neumann(const Vec<ValueType,N>& x) {
    return sph_neumann_intrin<n>(x.get());
  }
  template <Integer n, Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sph_neumann(const Vec<ValueType,N>& x) {
    return sph_neumann_intrin<n, (detail_approx_digits::full<ValueType>(digits) ? -1 : digits)>(x.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> fmod(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y) {
    return fmod_intrin(x.get(), y.get());
  }


  // Print
  template <class ValueType, Integer N> inline std::ostream& operator<<(std::ostream& os, const Vec<ValueType,N>& in) {
    for (Integer i = 0; i < N; i++) os << in[i] << ' ';
    return os;
  }


  // Other operators
  template <class ValueType> inline void printb(const ValueType& x) { // print binary
    union {
      ValueType v;
      uint8_t c[sizeof(ValueType)];
    } u = {x};
    //std::cout<<std::setw(10)<<x<<' ';
    for (Integer i = 0; i < (Integer)sizeof(ValueType); i++) {
      for (Integer j = 0; j < 8; j++) {
        std::cout<<((u.c[i] & (1U<<j))?'1':'0');
      }
    }
    std::cout<<'\n';
  }

}

#endif // _SCTL_VEC_TXX_
