#ifndef _SCTL_VEC_WRAPPER_HPP_
#define _SCTL_VEC_WRAPPER_HPP_

#include SCTL_INCLUDE(intrin-wrapper.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(common.hpp)

#include <cassert>
#include <cstdint>
#include <ostream>

namespace SCTL_NAMESPACE { // Vec

  #if defined(__AVX512__) || defined(__AVX512F__)
  static_assert(SCTL_ALIGN_BYTES >= 64);
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 64/sizeof(ScalarType); }
  #elif defined(__AVX__)
  static_assert(SCTL_ALIGN_BYTES >= 32);
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 32/sizeof(ScalarType); }
  #elif defined(__SSE4_2__)
  static_assert(SCTL_ALIGN_BYTES >= 16);
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 16/sizeof(ScalarType); }
  #else
  static_assert(SCTL_ALIGN_BYTES >= 8);
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 1; }
  #endif

  template <class ValueType, Integer N = DefaultVecLen<ValueType>()> class alignas(sizeof(ValueType) * N) Vec {
    public:
      using ScalarType = ValueType;
      using VData = VecData<ScalarType,N>;
      using MaskType = Mask<VData>;

      static constexpr Integer Size() {
        return N;
      }

      static Vec Zero() {
        Vec r;
        r.v = zero_intrin<VData>();
        return r;
      }

      static Vec Load1(ScalarType const* p) {
        Vec r;
        r.v = load1_intrin<VData>(p);
        return r;
      }
      static Vec Load(ScalarType const* p) {
        Vec r;
        r.v = loadu_intrin<VData>(p);
        return r;
      }
      static Vec LoadAligned(ScalarType const* p) {
        Vec r;
        r.v = load_intrin<VData>(p);
        return r;
      }

      Vec() = default;
      Vec(const Vec&) = default;
      Vec& operator=(const Vec&) = default;
      ~Vec() = default;

      Vec(const VData& v_) : v(v_) {}
      Vec(const ScalarType& a) : Vec(set1_intrin<VData>(a)) {}
      template <class T,class ...T1> Vec(T x, T1... args) : Vec(InitVec<T1...>::template apply<ScalarType>((ScalarType)x,args...)) {}

      void Store(ScalarType* p) const {
        storeu_intrin(p,v);
      }
      void StoreAligned(ScalarType* p) const {
        store_intrin(p,v);
      }

      // Conversion operators
      friend Mask<VData> convert2mask(const Vec& a) {
        return convert_vec2mask_intrin(a.v);
      }
      friend Vec RoundReal2Real(const Vec& x) {
        return round_real2real_intrin(x.v);
      }
      template <class IntVec, class RealVec> friend IntVec RoundReal2Int(const RealVec& x);
      template <class RealVec, class IntVec> friend RealVec ConvertInt2Real(const IntVec& x);

      // Element access
      ScalarType operator[](Integer i) const {
        return extract_intrin(v,i);
      }
      void insert(Integer i, ScalarType value) {
        insert_intrin(v,i,value);
      }

      // Arithmetic operators
      Vec operator+() const {
        return *this;
      }
      Vec operator-() const {
        return unary_minus_intrin(v); // Zero() - (*this);
      }
      friend Vec operator*(const Vec& a, const Vec& b) {
        return mul_intrin(a.v, b.v);
      }
      friend Vec operator/(const Vec& a, const Vec& b) {
        return div_intrin(a.v, b.v);
      }
      friend Vec operator+(const Vec& a, const Vec& b) {
        return add_intrin(a.v, b.v);
      }
      friend Vec operator-(const Vec& a, const Vec& b) {
        return sub_intrin(a.v, b.v);
      }
      friend Vec FMA(const Vec& a, const Vec& b, const Vec& c) {
        return fma_intrin(a.v, b.v, c.v);
      }

      // Comparison operators
      friend Mask<VData> operator< (const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::lt>(a.v, b.v);
      }
      friend Mask<VData> operator<=(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::le>(a.v, b.v);
      }
      friend Mask<VData> operator>=(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::ge>(a.v, b.v);
      }
      friend Mask<VData> operator> (const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::gt>(a.v, b.v);
      }
      friend Mask<VData> operator==(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::eq>(a.v, b.v);
      }
      friend Mask<VData> operator!=(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::ne>(a.v, b.v);
      }
      friend Vec select(const Mask<VData>& m, const Vec& a, const Vec& b) {
        return select_intrin(m, a.v, b.v);
      }

      // Bitwise operators
      Vec operator~() const {
        return not_intrin(v);
      }
      friend Vec operator&(const Vec& a, const Vec& b) {
        return and_intrin(a.v, b.v);
      }
      friend Vec operator^(const Vec& a, const Vec& b) {
        return xor_intrin(a.v, b.v);
      }
      friend Vec operator|(const Vec& a, const Vec& b) {
        return or_intrin(a.v, b.v);
      }
      friend Vec AndNot(const Vec& a, const Vec& b) { // return a & ~b
        return andnot_intrin(a.v, b.v);
      }

      // Bitshift
      friend Vec operator<<(const Vec& lhs, const Integer& rhs) {
        return bitshiftleft_intrin(lhs.v, rhs);
      }
      friend Vec operator>>(const Vec& lhs, const Integer& rhs) {
        return bitshiftright_intrin(lhs.v, rhs);
      }

      // Assignment operators
      Vec& operator=(const ScalarType& a) {
        v = set1_intrin<VData>(a);
        return *this;
      }
      Vec& operator*=(const Vec& rhs) {
        v = mul_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator/=(const Vec& rhs) {
        v = div_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator+=(const Vec& rhs) {
        v = add_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator-=(const Vec& rhs) {
        v = sub_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator&=(const Vec& rhs) {
        v = and_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator^=(const Vec& rhs) {
        v = xor_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator|=(const Vec& rhs) {
        v = or_intrin(v, rhs.v);
        return *this;
      }

      // Other operators
      friend Vec max(const Vec& lhs, const Vec& rhs) {
        return max_intrin(lhs.v, rhs.v);
      }
      friend Vec min(const Vec& lhs, const Vec& rhs) {
        return min_intrin(lhs.v, rhs.v);
      }

      // Special functions
      template <Integer digits, class RealVec> friend RealVec approx_rsqrt(const RealVec& x);
      template <Integer digits, class RealVec> friend RealVec approx_rsqrt(const RealVec& x, const typename RealVec::MaskType& m);

      friend void sincos(Vec& sinx, Vec& cosx, const Vec& x) {
        sincos_intrin(sinx.v, cosx.v, x.v);
      }
      friend Vec exp(const Vec& x) {
        return exp_intrin(x.v);
      }


      //template <class Vec1, class Vec2> friend Vec1 reinterpret(const Vec2& x);
      //template <class Vec> friend Vec RoundReal2Real(const Vec& x);
      //template <class Vec> friend void exp_intrin(Vec& expx, const Vec& x);

      // Print
      friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
        for (Integer i = 0; i < Size(); i++) os << in[i] << ' ';
        return os;
      }

      VData& get() { return v; }

    private:

      template <class T, class... T2> struct InitVec {
        template <class... T1> static VData apply(T1... start, T x, T2... rest) {
          return InitVec<T2...>::template apply<ScalarType, T1...>(start..., (ScalarType)x, rest...);
        }
      };
      template <class T> struct InitVec<T> {
        template <class... T1> static VData apply(T1... start, T x) {
          return set_intrin<VData>(start..., (ScalarType)x);
        }
      };

      VData v;
  };

  // Conversion operators
  template <class RealVec, class IntVec> RealVec ConvertInt2Real(const IntVec& x) {
    return convert_int2real_intrin<typename RealVec::VData>(x.v);
  }
  template <class IntVec, class RealVec> IntVec RoundReal2Int(const RealVec& x) {
    return round_real2int_intrin<typename IntVec::VData>(x.v);
  }
  template <class MaskType> Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a) {
    return convert_mask2vec_intrin(a);
  }

  // Special functions
  template <Integer digits, class RealVec> RealVec approx_rsqrt(const RealVec& x) {
    return rsqrt_approx_intrin<digits, typename RealVec::VData>::eval(x.v);
  }
  template <Integer digits, class RealVec> RealVec approx_rsqrt(const RealVec& x, const typename RealVec::MaskType& m) {
    return rsqrt_approx_intrin<digits, typename RealVec::VData>::eval(x.v, m);
  }

  // Other operators
  template <class ValueType> void printb(const ValueType& x) { // print binary
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

  // Verify Vec class
  template <class ValueType = double, Integer N = 1> class VecTest {
    public:
      using VecType = Vec<ValueType,N>;
      using ScalarType = typename VecType::ScalarType;
      using MaskType = Mask<typename VecType::VData>;

      static void test() {
        for (Integer i = 0; i < 1000; i++) {
          VecTest<ScalarType, 1>::test_all_types();
          VecTest<ScalarType, 2>::test_all_types();
          VecTest<ScalarType, 4>::test_all_types();
          VecTest<ScalarType, 8>::test_all_types();
          VecTest<ScalarType,16>::test_all_types();
          VecTest<ScalarType,32>::test_all_types();
          VecTest<ScalarType,64>::test_all_types();
        }
      }

      static void test_all_types() {
        VecTest< int8_t,N>::test_all();
        VecTest<int16_t,N>::test_all();
        VecTest<int32_t,N>::test_all();
        VecTest<int64_t,N>::test_all();

        VecTest<float,N>::test_all();
        VecTest<float,N>::test_reals();

        VecTest<double,N>::test_all();
        VecTest<double,N>::test_reals();

        //VecTest<long double,N>::test_all();
        //VecTest<long double,N>::test_reals();

        #ifdef SCTL_QUAD_T
        VecTest<QuadReal,N>::test_all();
        VecTest<QuadReal,N>::test_reals();
        #endif
      }

      static void test_all() {
        if (N*sizeof(ScalarType)*8<=512) {
          test_init();
          test_bitwise(); // TODO: fails for 'long double'
          test_arithmetic();
          test_maxmin();
          test_mask(); // TODO: fails for 'long double'
          test_comparison(); // TODO: fails for 'long double'
        }
      }

      static void test_reals() {
        if (N*sizeof(ScalarType)*8<=512) {
          test_reals_convert(); // TODO: fails for 'long double'
          test_reals_specialfunc();
          test_reals_rsqrt();
        }
      }

    private:

      static void test_init() {
        sctl::Vector<ScalarType> x(N+1), y(N+1), z(N);

        // Constructor: Vec(v)
        VecType v1((ScalarType)2);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v1[i] == (ScalarType)2);
        }

        // Constructor: Vec(v1,..,vn)
        VecType v2 = InitVec<N>::apply();
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v2[i] == (ScalarType)(i+1));
        }

        // insert, operator[]
        for (Integer i = 0; i < N; i++) {
          v1.insert(i, (ScalarType)(i+2));
        }
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v1[i] == (ScalarType)(i+2));
        }

        // Load1
        for (Integer i = 0; i < N+1; i++) {
          x[i] = (ScalarType)(i+7);
        }
        v1 = VecType::Load1(&x[1]);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v1[i] == (ScalarType)8);
        }

        // Load, Store
        v1 = VecType::Load(&x[1]);
        v1.Store(&y[1]);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(y[i+1] == (ScalarType)(i+8));
        }

        // LoadAligned, StoreAligned
        v1 = VecType::LoadAligned(&x[0]);
        v1.StoreAligned(&z[0]);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(z[i] == (ScalarType)(i+7));
        }

        // SetZero
        v1 = VecType::Zero();
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v1[i] == (ScalarType)0);
        }

        // Assignment operators
        v1 = (ScalarType)3;
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v1[i] == (ScalarType)3);
        }


        //// get_low, get_high
        //auto v_low = v2.get_low();
        //auto v_high = v2.get_high();
        //for (Integer i = 0; i < N/2; i++) {
        //  SCTL_ASSERT(v_low[i] == (ScalarType)(N-i));
        //  SCTL_ASSERT(v_high[i] == (ScalarType)(N-(i+N/2)));
        //}

        //// Constructor: Vec(v1, v2)
        //VecType v3(v_low,v_high);
        //for (Integer i = 0; i < N; i++) {
        //  SCTL_ASSERT(v3[i] == (ScalarType)(N-i));
        //}
      }

      static void test_bitwise() {
        UnionType u1, u2, u3, u4, u5, u6, u7;
        for (Integer i = 0; i < SizeBytes; i++) {
          u1.c[i] = rand();
          u2.c[i] = rand();
        }

        u3.v = ~u1.v;
        u4.v = u1.v & u2.v;
        u5.v = u1.v ^ u2.v;
        u6.v = u1.v | u2.v;
        u7.v = AndNot(u1.v, u2.v);

        for (Integer i = 0; i < SizeBytes; i++) {
          SCTL_ASSERT(u3.c[i] == (int8_t)~u1.c[i]);
          SCTL_ASSERT(u4.c[i] == (int8_t)(u1.c[i] & u2.c[i]));
          SCTL_ASSERT(u5.c[i] == (int8_t)(u1.c[i] ^ u2.c[i]));
          SCTL_ASSERT(u6.c[i] == (int8_t)(u1.c[i] | u2.c[i]));
          SCTL_ASSERT(u7.c[i] == (int8_t)(u1.c[i] & (~u2.c[i])));
        }
      }

      static void test_arithmetic() {
        UnionType u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17;
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)(rand()%100)+1;
          u2.x[i] = (ScalarType)(rand()%100)+2;
          u3.x[i] = (ScalarType)(rand()%100)+5;
        }

        u4.v = -u1.v;
        u5.v = u1.v + u2.v;
        u6.v = u1.v - u2.v;
        u7.v = u1.v * u2.v;
        u8.v = u1.v / u2.v;
        u9.v = FMA(u1.v, u2.v, u3.v);

        u10.v = u1.v; u10.v += u2.v;
        u11.v = u1.v; u11.v -= u2.v;
        u12.v = u1.v; u12.v *= u2.v;
        u13.v = u1.v; u13.v /= u2.v;

        u14.v = u1.v; u14.v += u2.v[0];
        u15.v = u1.v; u15.v -= u2.v[0];
        u16.v = u1.v; u16.v *= u2.v[0];
        u17.v = u1.v; u17.v /= u2.v[0];

        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(u4.x[i] == (ScalarType)-u1.x[i]);
          SCTL_ASSERT(u5.x[i] == (ScalarType)(u1.x[i] + u2.x[i]));
          SCTL_ASSERT(u6.x[i] == (ScalarType)(u1.x[i] - u2.x[i]));
          SCTL_ASSERT(u7.x[i] == (ScalarType)(u1.x[i] * u2.x[i]));
          SCTL_ASSERT(u8.x[i] == (ScalarType)(u1.x[i] / u2.x[i]));

          SCTL_ASSERT(u10.x[i] == (ScalarType)(u1.x[i] + u2.x[i]));
          SCTL_ASSERT(u11.x[i] == (ScalarType)(u1.x[i] - u2.x[i]));
          SCTL_ASSERT(u12.x[i] == (ScalarType)(u1.x[i] * u2.x[i]));
          SCTL_ASSERT(u13.x[i] == (ScalarType)(u1.x[i] / u2.x[i]));

          SCTL_ASSERT(u14.x[i] == (ScalarType)(u1.x[i] + u2.x[0]));
          SCTL_ASSERT(u15.x[i] == (ScalarType)(u1.x[i] - u2.x[0]));
          SCTL_ASSERT(u16.x[i] == (ScalarType)(u1.x[i] * u2.x[0]));
          SCTL_ASSERT(u17.x[i] == (ScalarType)(u1.x[i] / u2.x[0]));

          if (TypeTraits<ScalarType>::Type == DataType::Integer) {
            SCTL_ASSERT(u9.x[i] == (ScalarType)(u1.x[i]*u2.x[i] + u3.x[i]));
          } else {
            auto myabs = [](ScalarType a) {
              return (a < 0 ? -a : a);
            };
            auto machine_eps = [](){
              ScalarType eps = 1;
              while ((ScalarType)(1+eps/2) > 1) {
                eps = eps/2;
              }
              return eps;
            };
            static const ScalarType eps = machine_eps();
            ScalarType err = myabs(u9.x[i] - (ScalarType)(u1.x[i]*u2.x[i] + u3.x[i]));
            ScalarType max_val = myabs(u1.x[i]*u2.x[i]) + myabs(u3.x[i]);
            ScalarType rel_err = err / max_val;
            SCTL_ASSERT(rel_err < eps);
          }
        }
      }

      static void test_maxmin() {
        UnionType u1, u2, u3, u4;
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)rand();
          u2.x[i] = (ScalarType)rand();
        }

        u3.v = max(u1.v, u2.v);
        u4.v = min(u1.v, u2.v);

        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(u3.x[i] == (u1.x[i] < u2.x[i] ? u2.x[i] : u1.x[i]));
          SCTL_ASSERT(u4.x[i] == (u1.x[i] < u2.x[i] ? u1.x[i] : u2.x[i]));
        }
      }

      static void test_mask() {
        union {
          MaskType v;
          int8_t c[sizeof(MaskType)];
        } u1, u2, u3, u4, u5, u6, u7;
        for (Integer i = 0; i < (Integer)sizeof(MaskType); i++) {
          u1.c[i] = rand();
          u2.c[i] = rand();
        }

        u3.v = ~u1.v;
        u4.v = u1.v & u2.v;
        u5.v = u1.v ^ u2.v;
        u6.v = u1.v | u2.v;
        u7.v = AndNot(u1.v, u2.v);

        for (Integer i = 0; i < (Integer)sizeof(MaskType); i++) {
          SCTL_ASSERT(u3.c[i] == (int8_t)~u1.c[i]);
          SCTL_ASSERT(u4.c[i] == (int8_t)(u1.c[i] & u2.c[i]));
          SCTL_ASSERT(u5.c[i] == (int8_t)(u1.c[i] ^ u2.c[i]));
          SCTL_ASSERT(u6.c[i] == (int8_t)(u1.c[i] | u2.c[i]));
          SCTL_ASSERT(u7.c[i] == (int8_t)(u1.c[i] & (~u2.c[i])));
        }
      }

      static void test_comparison() {
        UnionType u1, u2, u3, u4, u5, u6, u7, u8, u9, u10;
        for (Integer i = 0; i < SizeBytes; i++) {
          u1.c[i] = rand()%4;
          u2.c[i] = rand()%4;
          u3.c[i] = rand()%4;
          u4.c[i] = rand()%4;
        }

        u5 .v = select((u1.v <  u2.v), u3.v, u4.v);
        u6 .v = select((u1.v <= u2.v), u3.v, u4.v);
        u7 .v = select((u1.v >  u2.v), u3.v, u4.v);
        u8 .v = select((u1.v >= u2.v), u3.v, u4.v);
        u9 .v = select((u1.v == u2.v), u3.v, u4.v);
        u10.v = select((u1.v != u2.v), u3.v, u4.v);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(u5 .x[i] == (u1.x[i] <  u2.x[i] ? u3.x[i] : u4.x[i]));
          SCTL_ASSERT(u6 .x[i] == (u1.x[i] <= u2.x[i] ? u3.x[i] : u4.x[i]));
          SCTL_ASSERT(u7 .x[i] == (u1.x[i] >  u2.x[i] ? u3.x[i] : u4.x[i]));
          SCTL_ASSERT(u8 .x[i] == (u1.x[i] >= u2.x[i] ? u3.x[i] : u4.x[i]));
          SCTL_ASSERT(u9 .x[i] == (u1.x[i] == u2.x[i] ? u3.x[i] : u4.x[i]));
          SCTL_ASSERT(u10.x[i] == (u1.x[i] != u2.x[i] ? u3.x[i] : u4.x[i]));
        }

        MaskType m0 = (u1.v < u2.v);
        VecType v1 = convert2vec(m0);
        MaskType m1 = convert2mask(v1);
        VecType v2 = select(m1, u3.v, u4.v);
        VecType v3 = (u3.v & v1) | AndNot(u4.v, v1);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v2[i] == (u1.x[i] <  u2.x[i] ? u3.x[i] : u4.x[i]));
          SCTL_ASSERT(v3[i] == (u1.x[i] <  u2.x[i] ? u3.x[i] : u4.x[i]));
        }
      }

      static void test_reals_convert() {
        using IntVec = Vec<typename IntegerType<sizeof(ScalarType)>::value,N>;
        using RealVec = Vec<ScalarType,N>;
        static_assert(TypeTraits<ScalarType>::Type == DataType::Real, "Expected real type!");

        RealVec a = RealVec::Zero();
        for (Integer i = 0; i < N; i++) a.insert(i, (ScalarType)(drand48()-0.5)*100);
        IntVec b = RoundReal2Int<IntVec>(a);
        RealVec c = RoundReal2Real(a);
        RealVec d = ConvertInt2Real<RealVec>(b);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(b[i] == (typename IntVec::ScalarType)round(a[i]));
          SCTL_ASSERT(c[i] == (ScalarType)(typename IntVec::ScalarType)round(a[i]));
          SCTL_ASSERT(d[i] == (ScalarType)b[i]);
        }
      }

      static void test_reals_specialfunc() {
        VecType v0 = VecType::Zero(), v1, v2, v3;
        for (Integer i = 0; i < N; i++) {
          v0.insert(i, (ScalarType)(drand48()-0.5)*4*const_pi<ScalarType>());
        }
        sincos(v1, v2, v0);
        v3 = exp(v0);
        for (Integer i = 0; i < N; i++) {
          ScalarType err_tol = std::max<ScalarType>((ScalarType)1.77e-15, (pow<TypeTraits<ScalarType>::SigBits-3,ScalarType>((ScalarType)0.5))); // TODO: fix for accuracy greater than 1.77e-15
          SCTL_ASSERT(fabs(v1[i] - sin<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v2[i] - cos<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v3[i] - exp<ScalarType>(v0[i]))/fabs(exp<ScalarType>(v0[i])) < err_tol);
        }
      }

      static void test_reals_rsqrt() {
        UnionType u1, u2, u3;
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)rand();
        }

        u2.v = approx_rsqrt<4>(u1.v);
        u3.v = approx_rsqrt<7>(u1.v);
        for (Integer i = 0; i < N; i++) {
          ScalarType err = fabs(u2.x[i] - 1/sqrt<ScalarType>(u1.x[i]));
          ScalarType max_val = fabs(1/sqrt<ScalarType>(u1.x[i]));
          ScalarType rel_err = err / max_val;
          SCTL_ASSERT(rel_err < (pow<11,ScalarType>((ScalarType)0.5)));
        }
        for (Integer i = 0; i < N; i++) {
          ScalarType err = fabs((ScalarType)(u3.x[i] - 1/sqrt((double)u1.x[i]))); // float is not accurate enough to compute reference solution with 7-digits
          ScalarType max_val = fabs(1/sqrt<ScalarType>(u1.x[i]));
          ScalarType rel_err = err / max_val;
          SCTL_ASSERT(rel_err < (pow<22,ScalarType>((ScalarType)0.5)));
        }
      }


      template <Integer k, class... T2> struct InitVec {
        static VecType apply(T2... rest) {
          return InitVec<k-1, ScalarType, T2...>::apply((ScalarType)k, rest...);
        }
      };
      template <class... T2> struct InitVec<0, T2...> {
        static VecType apply(T2... rest) {
          return VecType(rest...);
        }
      };

      static constexpr Integer SizeBytes = VecType::Size()*sizeof(ScalarType);
      union UnionType {
        VecType v;
        ScalarType x[N];
        int8_t c[SizeBytes];
      };
  };

}

#endif  //_SCTL_VEC_WRAPPER_HPP_
