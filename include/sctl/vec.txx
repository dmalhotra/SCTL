
namespace SCTL_NAMESPACE {

  template <class ScalarType> constexpr Integer DefaultVecLen() {
    #if defined(__AVX512__) || defined(__AVX512F__)
    static_assert(SCTL_ALIGN_BYTES >= 64, "Insufficient memory alignment for SIMD vector types");
    return 64/sizeof(ScalarType);
    #elif defined(__AVX__)
    static_assert(SCTL_ALIGN_BYTES >= 32, "Insufficient memory alignment for SIMD vector types");
    return 32/sizeof(ScalarType);
    #elif defined(__SSE4_2__)
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

  template <class ValueType, Integer N> inline Vec<ValueType,N> Vec<ValueType,N>::Zero() {
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

  template <class ValueType, Integer N> inline Vec<ValueType,N>::Vec(const VData& v_) : v(v_) {}
  template <class ValueType, Integer N> inline Vec<ValueType,N>::Vec(const ScalarType& a) : Vec(set1_intrin<VData>(a)) {}
  template <class ValueType, Integer N> template <class T,class ...T1> inline Vec<ValueType,N>::Vec(T x, T1... args) : Vec(InitVec<T1...>::template apply<ScalarType>((ScalarType)x,args...)) {}


  template <class ValueType, Integer N> inline void Vec<ValueType,N>::Store(ScalarType* p) const {
    storeu_intrin(p,v);
  }
  template <class ValueType, Integer N> inline void Vec<ValueType,N>::StoreAligned(ScalarType* p) const {
    #ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(((uintptr_t)p) % (sizeof(ValueType)*N) == 0);
    #endif
    store_intrin(p,v);
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

  template <class ValueType, Integer N> template <class T, class... T2> struct Vec<ValueType,N>::InitVec {
    template <class... T1> static inline VData apply(T1... start, T x, T2... rest) {
      return InitVec<T2...>::template apply<ScalarType, T1...>(start..., (ScalarType)x, rest...);
    }
  };
  template <class ValueType, Integer N> template <class T> struct Vec<ValueType,N>::InitVec<T> {
    template <class... T1> static inline VData apply(T1... start, T x) {
      return set_intrin<VData>(start..., (ScalarType)x);
    }
  };




  // Conversion operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType convert2mask(const Vec<ValueType,N>& a) {
    return convert_vec2mask_intrin(a.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> RoundReal2Real(const Vec<ValueType,N>& x) {
    return round_real2real_intrin(x.get());
  }
  template <class RealVec, class IntVec> inline RealVec ConvertInt2Real(const IntVec& x) {
    return convert_int2real_intrin<typename RealVec::VData>(x.get());
  }
  template <class IntVec, class RealVec> inline IntVec RoundReal2Int(const RealVec& x) {
    return round_real2int_intrin<typename IntVec::VData>(x.get());
  }
  template <class MaskType> inline Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a) {
    return convert_mask2vec_intrin(a);
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

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const ValueType& b) {
    return a * Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const ValueType& b) {
    return a / Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const ValueType& b) {
    return a + Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const ValueType& b) {
    return a - Vec<ValueType,N>(b);
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) * b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) / b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) + b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) - b;
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

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const ValueType& b) {
    return a <  Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const ValueType& b) {
    return a <= Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const ValueType& b) {
    return a >= Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const ValueType& b) {
    return a > Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const ValueType& b) {
    return a == Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const ValueType& b) {
    return a != Vec<ValueType,N>(b);
  }

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) <  b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) <= b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) >= b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) >  b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) == b;
  }
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const ValueType& a, const Vec<ValueType,N>& b) {
    return Vec<ValueType,N>(a) != b;
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const Vec<ValueType,N>& b) {
    return select_intrin(m, a.get(), b.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const ValueType& b) {
    return select(m, a, Vec<ValueType,N>(b));
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const ValueType& a, const Vec<ValueType,N>& b) {
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

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const ValueType& b) {
    return a & Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const ValueType& b) {
    return a ^ Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const ValueType& b) {
    return a | Vec<ValueType,N>(b);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const ValueType& b) { // return a & ~b
    return AndNot(a, Vec<ValueType,N>(b));
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const ValueType& b, const Vec<ValueType,N>& a) {
    return Vec<ValueType,N>(a) & b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const ValueType& b, const Vec<ValueType,N>& a) {
    return Vec<ValueType,N>(a) ^ b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const ValueType& b, const Vec<ValueType,N>& a) {
    return Vec<ValueType,N>(a) | b;
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const ValueType& b, const Vec<ValueType,N>& a) { // return a & ~b
    return AndNot(Vec<ValueType,N>(a), b);
  }


  // Bitshift
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Integer& rhs) {
    return bitshiftleft_intrin(lhs.get(), rhs);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Integer& rhs) {
    return bitshiftright_intrin(lhs.get(), rhs);
  }


  // Other operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs) {
    return max_intrin(lhs.get(), rhs.get());
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs) {
    return min_intrin(lhs.get(), rhs.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const ValueType& rhs) {
    return max(lhs, Vec<ValueType,N>(rhs));
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const ValueType& rhs) {
    return min(lhs, Vec<ValueType,N>(rhs));
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const ValueType& lhs, const Vec<ValueType,N>& rhs) {
    return max(Vec<ValueType,N>(lhs), rhs);
  }
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const ValueType& lhs, const Vec<ValueType,N>& rhs) {
    return min(Vec<ValueType,N>(lhs), rhs);
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

  template <class ValueType, Integer N> inline void sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x) {
    sincos_intrin(sinx.get(), cosx.get(), x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline void approx_sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x) {
    constexpr Integer ORDER = (digits>1?digits>9?digits>14?digits>17?digits-1:digits:digits+1:digits+2:1);
    if (digits == -1 || ORDER > 20) sincos(sinx, cosx, x);
    else approx_sincos_intrin<ORDER>(sinx.get(), cosx.get(), x.get());
  }

  template <class ValueType, Integer N> inline Vec<ValueType,N> exp(const Vec<ValueType,N>& x) {
    return exp_intrin(x.get());
  }
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_exp(const Vec<ValueType,N>& x) {
    constexpr Integer ORDER = digits;
    if (digits == -1 || ORDER > 13) return exp(x);
    else return approx_exp_intrin<ORDER>(x.get());
  }

  #if defined(SCTL_HAVE_SVML) || defined(SCTL_HAVE_LIBMVEC)
  template <class ValueType, Integer N> inline Vec<ValueType,N> log(const Vec<ValueType,N>& x) {
    return log_intrin(x.get());
  }
  #endif


  // Print
  template <Integer digits, class ValueType, Integer N> inline std::ostream& operator<<(std::ostream& os, const Vec<ValueType,N>& in) {
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
