#ifndef _SCTL_VEC_TEST_HPP_
#define _SCTL_VEC_TEST_HPP_

#include <stdlib.h>                 // for rand, drand48
#include <algorithm>                // for max
#include <cmath>                    // for pow
#include <cstdint>                  // for int8_t, int16_t, int32_t, int64_t
#include <limits>                   // for numeric_limits
#include <type_traits>              // for is_same

#include "sctl/common.hpp"          // for SCTL_ASSERT, Integer, sctl
#include "sctl/intrin-wrapper.hpp"  // for IntegerType, TypeTraits, DataType
#include "sctl/math_utils.hpp"      // for fabs, cos, sin, sqrt, QuadReal (p...
#include "sctl/math_utils.txx"      // for pow, machine_eps
#include "sctl/vec.hpp"             // for Vec
#include "sctl/vec.txx"             // for select, approx_exp, approx_rsqrt
#include "sctl/vector.hpp"          // for Vector

namespace sctl {

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
        VecTest< int8_t,N>::test_ints();
        VecTest<int16_t,N>::test_all();
        VecTest<int16_t,N>::test_ints();
        VecTest<int32_t,N>::test_all();
        VecTest<int32_t,N>::test_ints();
        VecTest<int64_t,N>::test_all();
        VecTest<int64_t,N>::test_ints();

        VecTest<float,N>::test_all();
        VecTest<float,N>::test_reals();

        VecTest<double,N>::test_all();
        VecTest<double,N>::test_reals();

        VecTest<long double,N>::test_all();
        VecTest<long double,N>::test_reals();
        VecTest<long double,N>::test_reals_long_double();

        #ifdef SCTL_QUAD_T
        VecTest<QuadReal,N>::test_all();
        VecTest<QuadReal,N>::test_reals();
        #endif
      }

      static void test_all() {
        if (N*sizeof(ScalarType)*8<=512) {
          test_align();
          test_init();
          test_load_store_mask();
          test_gather_scatter();
          test_convert();
          test_reduce();
          test_bitwise();
          test_arithmetic();
          test_maxmin();
          test_transpose();
          test_swap_pairs();
          test_mask();
          test_comparison();
        }
      }

      static void test_ints() {
        if (N*sizeof(ScalarType)*8<=512) {
          test_bitshift();
          test_div_rem();
          test_fabs();
        }
      }

      static void test_reals() {
        if (N*sizeof(ScalarType)*8<=512) {
          test_reals_convert();
          test_reals_specialfunc();
          test_reals_rsqrt();
          test_mask_helpers();
          test_reals_math();
          test_reals_math2();
        }
      }

      static void test_reals_long_double() { // approx_exp over the whole range of the result, and approx_sincos
        if (N*sizeof(ScalarType)*8<=512) {
          const ScalarType max_x = std::log(std::numeric_limits<ScalarType>::max());
          UnionType u, u1;
          for (Integer i = 0; i < N; i++) {
            u.x[i] = (ScalarType)((drand48()-0.5)*2) * max_x;
            u1.x[i] = (ScalarType)((drand48()-0.5)*200);
          }
          const VecType e = approx_exp<12>(u.v);
          VecType sn, cs;
          approx_sincos<12>(sn, cs, u1.v);
          for (Integer i = 0; i < N; i++) {
            const ScalarType ref = std::exp(u.x[i]);
            SCTL_ASSERT(fabs(e[i] - ref) <= (ScalarType)1e-12 * ref || ref < std::numeric_limits<ScalarType>::min());
            SCTL_ASSERT(fabs(sn[i] - std::sin(u1.x[i])) <= (ScalarType)1e-12 && fabs(cs[i] - std::cos(u1.x[i])) <= (ScalarType)1e-12);
          }
          SCTL_ASSERT(isinf(approx_exp<12>(VecType((ScalarType)INFINITY))[0]) && approx_exp<12>(VecType(-(ScalarType)INFINITY))[0] == 0);
          SCTL_ASSERT(isnan(approx_exp<12>(VecType((ScalarType)NAN))[0]));
        }
      }

    private:

      static void test_align() {
        // NOTE: alignas not respected on some GCC versions when compiled with address-sanitizer
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110027

        VecType v1;
        char c;
        VecType v2;
        auto addr1 = reinterpret_cast<uintptr_t>(&v1);
        auto addr2 = reinterpret_cast<uintptr_t>(&v2);
        SCTL_ASSERT(addr1 % (N*sizeof(ValueType)) == 0);
        SCTL_ASSERT(addr2 % (N*sizeof(ValueType)) == 0);
        SCTL_UNUSED(c);
      }

      static void test_init() {
        sctl::Vector<ScalarType> x(N+1), y(N+1), z(N);

        // Constructor: Vec(v)
        VecType v1((ScalarType)2);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v1[i] == (ScalarType)2);
        }

        // Constructor: Vec(v) from another scalar type
        VecType v_int(2);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(v_int[i] == (ScalarType)2);
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

        if constexpr (N >= 2) {
          // get_low, get_high
          const auto v_low = v2.get_low();
          const auto v_high = v2.get_high();
          for (Integer i = 0; i < N/2; i++) {
            SCTL_ASSERT(v_low[i] == (ScalarType)(i+1));
            SCTL_ASSERT(v_high[i] == (ScalarType)(i+N/2+1));
          }

          // Constructor: Vec(v_low, v_high)
          const VecType v3(v_low, v_high);
          for (Integer i = 0; i < N; i++) {
            SCTL_ASSERT(v3[i] == (ScalarType)(i+1));
          }
        }
      }

      static void test_load_store_mask() {
        sctl::Vector<ScalarType> x(N), y(N);
        UnionType u;
        for (Integer i = 0; i < N; i++) {
          x[i] = (ScalarType)(i+1);
          u.x[i] = (ScalarType)(rand()%2);
        }
        const MaskType m = (u.v == (ScalarType)1);
        const VecType v1((ScalarType)2);

        // Load, Store with a mask
        const VecType v2 = VecType::Load(&x[0], m);
        for (Integer i = 0; i < N; i++) {
          y[i] = (ScalarType)-1;
        }
        v1.Store(&y[0], m);
        for (Integer i = 0; i < N; i++) {
          const bool sel = (u.x[i] == (ScalarType)1);
          SCTL_ASSERT(v2[i] == (sel ? x[i] : (ScalarType)0));
          SCTL_ASSERT(y[i] == (sel ? (ScalarType)2 : (ScalarType)-1));
        }

        // LoadPartial, StorePartial: the first n elements
        for (Integer n = 0; n <= N+1; n++) {
          const VecType v3 = VecType::LoadPartial(&x[0], n);
          for (Integer i = 0; i < N; i++) {
            y[i] = (ScalarType)-1;
          }
          v1.StorePartial(&y[0], n);
          for (Integer i = 0; i < N; i++) {
            SCTL_ASSERT(v3[i] == (i < n ? x[i] : (ScalarType)0));
            SCTL_ASSERT(y[i] == (i < n ? (ScalarType)2 : (ScalarType)-1));
          }
        }
      }

      static void test_gather_scatter() {
        test_gather_scatter_idx<int32_t>();
        test_gather_scatter_idx<int64_t>();
      }

      template <class IndexType> static void test_gather_scatter_idx() {
        constexpr Integer M = 4*N;
        sctl::Vector<ScalarType> x(M), y(M);
        IndexType idx[N];
        ScalarType val[N];
        for (Integer j = 0; j < M; j++) {
          x[j] = (ScalarType)(rand()%100);
          y[j] = (ScalarType)-1;
        }
        for (Integer i = 0; i < N; i++) {
          idx[i] = (IndexType)(rand() % M); // repeated indices too
          val[i] = (ScalarType)(rand()%100);
        }
        const Vec<IndexType,N> vidx = Vec<IndexType,N>::Load(idx);
        const VecType g = VecType::Gather(&x[0], vidx);
        VecType::Load(val).Scatter(&y[0], vidx);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(g[i] == x[idx[i]]);
        }
        for (Integer j = 0; j < M; j++) {
          ScalarType expected = (ScalarType)-1;
          for (Integer i = 0; i < N; i++) {
            if (idx[i] == (IndexType)j) expected = val[i]; // the last element with index j
          }
          SCTL_ASSERT(y[j] == expected);
        }
      }

      static void test_convert() {
        test_convert_to<int8_t>();
        test_convert_to<int16_t>();
        test_convert_to<int32_t>();
        test_convert_to<int64_t>();
        test_convert_to<float>();
        test_convert_to<double>();
      }

      template <class ValueTo> static void test_convert_to() {
        using VecTo = Vec<ValueTo,N>;
        UnionType u;
        for (Integer i = 0; i < N; i++) {
          u.x[i] = (ScalarType)(rand()%201-100) + (ScalarType)0.25 * (ScalarType)(rand()%4); // fractions only in the real types
        }

        // Convert
        const VecTo w = Convert<VecTo>(u.v);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(w[i] == (ValueTo)u.x[i]);
        }

        // ConvertMask
        const auto m = ConvertMask<VecTo>(u.v < (ScalarType)0);
        const VecTo s = select(m, VecTo((ValueTo)1), VecTo((ValueTo)0));
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(s[i] == (u.x[i] < (ScalarType)0 ? (ValueTo)1 : (ValueTo)0));
        }
      }

      static void test_reduce() {
        UnionType u;
        for (Integer i = 0; i < N; i++) {
          u.x[i] = (TypeTraits<ScalarType>::Type == DataType::Real ? (ScalarType)(drand48()*200-100) : (ScalarType)(rand()%201-100));
        }

        // reduce, reduce_min, reduce_max
        ScalarType x[N]; // pairwise sums in the order of reduce
        ScalarType min_val = u.x[0];
        ScalarType max_val = u.x[0];
        for (Integer i = 0; i < N; i++) {
          x[i] = u.x[i];
          min_val = (u.x[i] < min_val ? u.x[i] : min_val);
          max_val = (max_val < u.x[i] ? u.x[i] : max_val);
        }
        for (Integer len = N/2; len > 0; len /= 2) {
          for (Integer i = 0; i < len; i++) x[i] = (ScalarType)(x[i] + x[i+len]);
        }
        SCTL_ASSERT(reduce(u.v) == x[0]);
        SCTL_ASSERT(reduce_min(u.v) == min_val);
        SCTL_ASSERT(reduce_max(u.v) == max_val);

        // reduce_count, all_of, any_of, none_of
        for (const ScalarType t : {(ScalarType)-101, (ScalarType)0, (ScalarType)101}) {
          const MaskType m = (u.v < t);
          Integer count = 0;
          for (Integer i = 0; i < N; i++) count += (u.x[i] < t ? 1 : 0);
          SCTL_ASSERT(reduce_count(m) == count);
          SCTL_ASSERT(all_of(m) == (count == N));
          SCTL_ASSERT(any_of(m) == (count > 0));
          SCTL_ASSERT(none_of(m) == (count == 0));
        }
      }

      static void test_bitwise() {
        UnionType u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15;
        for (Integer i = 0; i < SizeBytes; i++) {
          u1.c[i] = rand();
          u2.c[i] = rand();
        }

        u3.v = ~u1.v;
        u4.v = u1.v & u2.v;
        u5.v = u1.v ^ u2.v;
        u6.v = u1.v | u2.v;
        u7.v = AndNot(u1.v, u2.v);
        u8.v = AndNot(u1.v, u2.x[0]);
        u9.v = AndNot(u2.x[0], u1.v);
        u10.v = u1.v & u2.x[0];
        u11.v = u2.x[0] & u1.v;
        u12.v = u1.v ^ u2.x[0];
        u13.v = u2.x[0] ^ u1.v;
        u14.v = u1.v | u2.x[0];
        u15.v = u2.x[0] | u1.v;

        constexpr Integer ValueBytes = (std::is_same<ScalarType, long double>::value && std::numeric_limits<long double>::digits == 64 ? 10 : (Integer)sizeof(ScalarType)); // bytes of the value of a lane: 10 of the 16 of an x87 long double, the others need not be copied
        for (Integer i = 0; i < SizeBytes; i++) {
          if (i % (Integer)sizeof(ScalarType) >= ValueBytes) continue;
          const int8_t s = u2.c[i % (Integer)sizeof(ScalarType)]; // byte i of the broadcast u2.x[0]
          SCTL_ASSERT(u3.c[i] == (int8_t)~u1.c[i]);
          SCTL_ASSERT(u4.c[i] == (int8_t)(u1.c[i] & u2.c[i]));
          SCTL_ASSERT(u5.c[i] == (int8_t)(u1.c[i] ^ u2.c[i]));
          SCTL_ASSERT(u6.c[i] == (int8_t)(u1.c[i] | u2.c[i]));
          SCTL_ASSERT(u7.c[i] == (int8_t)(u1.c[i] & (~u2.c[i])));
          SCTL_ASSERT(u8.c[i] == (int8_t)(u1.c[i] & (~s)));
          SCTL_ASSERT(u9.c[i] == (int8_t)(s & (~u1.c[i])));
          SCTL_ASSERT(u10.c[i] == (int8_t)(u1.c[i] & s));
          SCTL_ASSERT(u11.c[i] == (int8_t)(s & u1.c[i]));
          SCTL_ASSERT(u12.c[i] == (int8_t)(u1.c[i] ^ s));
          SCTL_ASSERT(u13.c[i] == (int8_t)(s ^ u1.c[i]));
          SCTL_ASSERT(u14.c[i] == (int8_t)(u1.c[i] | s));
          SCTL_ASSERT(u15.c[i] == (int8_t)(s | u1.c[i]));
        }
      }

      static void test_bitshift() {
        using UInt = typename IntegerType<sizeof(ScalarType)>::unsigned_value;
        UnionType u1, u2, u3;
        for (Integer i = 0; i < SizeBytes; i++) {
          u1.c[i] = rand();
        }
        u1.x[0] = std::numeric_limits<ScalarType>::min();
        for (Integer k = 0; k < (Integer)sizeof(ScalarType)*8; k++) {
          u2.v = u1.v >> k;
          u3.v = u1.v << k;
          for (Integer i = 0; i < N; i++) {
            SCTL_ASSERT(u2.x[i] == (ScalarType)(u1.x[i] >> k));
            SCTL_ASSERT(u3.x[i] == (ScalarType)((UInt)u1.x[i] * ((UInt)1 << k))); // x * 2^k modulo 2^(bit width)
          }
        }

        // counts per element
        UnionType k_;
        for (Integer i = 0; i < N; i++) {
          k_.x[i] = (ScalarType)(rand() % (sizeof(ScalarType)*8));
        }
        u2.v = u1.v >> k_.v;
        u3.v = u1.v << k_.v;
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(u2.x[i] == (ScalarType)(u1.x[i] >> k_.x[i]));
          SCTL_ASSERT(u3.x[i] == (ScalarType)((UInt)u1.x[i] * ((UInt)1 << k_.x[i])));
        }
      }

      static void test_div_rem() {
        UnionType u1, u2, u3, u4, u5, u6, u7;
        for (Integer i = 0; i < SizeBytes; i++) {
          u1.c[i] = rand();
          u2.c[i] = rand();
        }
        for (Integer i = 0; i < N; i++) { // divisors of every magnitude; not 0, and not -1 for the minimum value
          u2.x[i] = (ScalarType)(u2.x[i] >> (rand() % (sizeof(ScalarType)*8)));
          if (u2.x[i] == 0 || (u2.x[i] == -1 && u1.x[i] == std::numeric_limits<ScalarType>::min())) u2.x[i] = 1;
        }

        u3.v = u1.v / u2.v;
        u4.v = u1.v % u2.v;
        u5.v = u1.v;
        u5.v %= u2.v;
        u6.v = u1.v % 7;
        u7.v = 100 % u2.v;
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(u3.x[i] == (ScalarType)(u1.x[i] / u2.x[i]));
          SCTL_ASSERT(u4.x[i] == (ScalarType)(u1.x[i] % u2.x[i]));
          SCTL_ASSERT(u5.x[i] == u4.x[i]);
          SCTL_ASSERT(u6.x[i] == (ScalarType)(u1.x[i] % 7));
          SCTL_ASSERT(u7.x[i] == (ScalarType)(100 % u2.x[i]));
        }
      }

      static void test_fabs() {
        UnionType u1, u2;
        for (Integer i = 0; i < SizeBytes; i++) {
          u1.c[i] = rand();
        }
        for (Integer i = 0; i < N; i++) {
          if (u1.x[i] == std::numeric_limits<ScalarType>::min()) u1.x[i] = 0;
        }
        u2.v = fabs(u1.v);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(u2.x[i] == (ScalarType)(u1.x[i] < 0 ? -u1.x[i] : u1.x[i]));
        }
      }

      static void test_arithmetic() {
        UnionType u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19;
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

        u18.v = u1.v * 2;
        u19.v = 2 - u1.v;

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

          SCTL_ASSERT(u18.x[i] == (ScalarType)(u1.x[i] * 2));
          SCTL_ASSERT(u19.x[i] == (ScalarType)(2 - u1.x[i]));

          if (TypeTraits<ScalarType>::Type == DataType::Integer) {
            SCTL_ASSERT(u9.x[i] == (ScalarType)(u1.x[i]*u2.x[i] + u3.x[i]));
          } else {
            auto myabs = [](ScalarType a) {
              return (a < 0 ? -a : a);
            };
            static const ScalarType eps = machine_eps<ScalarType>();
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

      static void test_transpose() {
        UnionType u[N], w[N];
        for (Integer i = 0; i < N; i++) {
          for (Integer j = 0; j < N; j++) u[i].x[j] = (ScalarType)rand();
        }

        VecType v[N];
        for (Integer i = 0; i < N; i++) v[i] = u[i].v;
        transpose(v);
        for (Integer i = 0; i < N; i++) w[i].v = v[i];

        for (Integer i = 0; i < N; i++) {
          for (Integer j = 0; j < N; j++) SCTL_ASSERT(w[i].x[j] == u[j].x[i]);
        }

        // variadic overload must agree with the array overload
        for (Integer i = 0; i < N; i++) v[i] = u[i].v;
        TransposeVa<N>::apply(v);
        for (Integer i = 0; i < N; i++) {
          for (Integer j = 0; j < N; j++) SCTL_ASSERT(v[i][j] == w[i].x[j]);
        }
      }

      // expand v[0],..,v[N-1] into the variadic transpose
      template <Integer k, class ...T> struct TransposeVa {
        static void apply(VecType (&v)[N], T&... args) { TransposeVa<k-1,T...,VecType>::apply(v, args..., v[N-k]); }
      };
      template <class ...T> struct TransposeVa<0,T...> {
        static void apply(VecType (&)[N], T&... args) { transpose(args...); }
      };

      static void test_swap_pairs() {
        if constexpr (N % 2 == 0) {
          UnionType u, w;
          for (Integer i = 0; i < N; i++) u.x[i] = (ScalarType)rand();
          w.v = swap_pairs(u.v);
          for (Integer i = 0; i < N; i++) SCTL_ASSERT(w.x[i] == u.x[i ^ 1]);
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
        if constexpr (std::is_same<ScalarType, long double>::value) { // random bytes make x87 unnormals, which compare unordered even with themselves
          for (Integer i = 0; i < N; i++) {
            u1.x[i] = (ScalarType)(rand()%4);
            u2.x[i] = (ScalarType)(rand()%4);
            u3.x[i] = (ScalarType)(rand()%4);
            u4.x[i] = (ScalarType)(rand()%4);
          }
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

      static void test_mask_helpers() {
        for (Integer trial = 0; trial < 16; trial++) {
          ScalarType a[N], b[N], src[N];
          for (Integer i = 0; i < N; i++) {
            a[i] = (ScalarType)(rand()%4);
            b[i] = (trial == 0 ? a[i] : trial == 1 ? a[i] + 1 : (ScalarType)(rand()%4)); // no lane, every lane, then random lanes
            src[i] = (ScalarType)(100 + i);
          }
          const VecType va = VecType::Load(a);
          const VecType vb = VecType::Load(b);
          const MaskType lt = (va < vb);
          const MaskType gt = (va > vb);
          Integer cnt_lt = 0;
          Integer cnt_gt = 0;
          ScalarType sum = 0;
          for (Integer i = 0; i < N; i++) {
            cnt_lt += (a[i] < b[i]);
            cnt_gt += (a[i] > b[i]);
            sum += a[i];
          }

          SCTL_ASSERT(reduce(va) == sum); // small integers, so the summation order does not matter
          SCTL_ASSERT(mask_popcnt_intrin(lt) == cnt_lt);
          SCTL_ASSERT(mask_any(lt) == (cnt_lt > 0));

          { // compress store: the selected elements in order, nothing written past them
            ScalarType out[N+1];
            for (Integer i = 0; i <= N; i++) out[i] = -1;
            mask_compress_store(lt, va.get(), out);
            Integer k = 0;
            for (Integer i = 0; i < N; i++) {
              if (a[i] < b[i]) {
                SCTL_ASSERT(out[k] == a[i]);
                k++;
              }
            }
            for (Integer i = cnt_lt; i <= N; i++) SCTL_ASSERT(out[i] == -1);
          }
          { // iota stores: the selected indices in order, nothing written past them
            const int32_t base = 1000;
            int32_t idx[2*N+1];
            for (Integer i = 0; i <= 2*N; i++) idx[i] = -1;
            SCTL_ASSERT(mask_compress_iota_store(lt, base, idx) == cnt_lt);
            Integer k = 0;
            for (Integer i = 0; i < N; i++) {
              if (a[i] < b[i]) {
                SCTL_ASSERT(idx[k] == base + i);
                k++;
              }
            }
            for (Integer i = cnt_lt; i <= 2*N; i++) SCTL_ASSERT(idx[i] == -1);

            for (Integer i = 0; i <= 2*N; i++) idx[i] = -1;
            SCTL_ASSERT(mask_compress_iota_store2(lt, gt, base, idx) == cnt_lt + cnt_gt);
            k = 0;
            for (Integer i = 0; i < N; i++) {
              if (a[i] < b[i]) {
                SCTL_ASSERT(idx[k] == base + i);
                k++;
              }
            }
            for (Integer i = 0; i < N; i++) {
              if (a[i] > b[i]) {
                SCTL_ASSERT(idx[k] == base + N + i);
                k++;
              }
            }
            for (Integer i = cnt_lt + cnt_gt; i <= 2*N; i++) SCTL_ASSERT(idx[i] == -1);
          }
          { // expand load: the inverse of the compress store
            const VecType v(mask_expand_load(lt, vb.get(), src));
            ScalarType out[N];
            v.Store(out);
            Integer k = 0;
            for (Integer i = 0; i < N; i++) {
              if (a[i] < b[i]) {
                SCTL_ASSERT(out[i] == src[k]);
                k++;
              } else {
                SCTL_ASSERT(out[i] == b[i]);
              }
            }
          }
        }
      }

      static void test_reals_convert() {
        using IntVec = Vec<typename IntegerType<sizeof(ScalarType)>::value,N>;
        using RealVec = Vec<ScalarType,N>;
        static_assert(TypeTraits<ScalarType>::Type == DataType::Real, "Expected real type!");

        RealVec a = RealVec::Zero();
        for (Integer i = 0; i < N; i++) a.insert(i, (ScalarType)(drand48()-0.5)*100);
        IntVec b = lrint<IntVec>(a);
        RealVec c = rint(a);
        RealVec d = ConvertInt2Real<RealVec>(b);
        for (Integer i = 0; i < N; i++) {
          SCTL_ASSERT(b[i] == (typename IntVec::ScalarType)round(a[i]));
          SCTL_ASSERT(c[i] == (ScalarType)(typename IntVec::ScalarType)round(a[i]));
          SCTL_ASSERT(d[i] == (ScalarType)b[i]);
        }

        const ScalarType xs[] = {(ScalarType)2.5, (ScalarType)-2.5, (ScalarType)3.5, (ScalarType)-0.5}; // halves go to the even integer
        const ScalarType rs[] = {(ScalarType)2, (ScalarType)-2, (ScalarType)4, (ScalarType)0};
        for (Integer k = 0; k < 4; k++) {
          SCTL_ASSERT(rint(RealVec(xs[k]))[0] == rs[k]);
          SCTL_ASSERT(lrint<IntVec>(RealVec(xs[k]))[0] == (typename IntVec::ScalarType)rs[k]);
        }
      }

      static void test_reals_specialfunc() {
        VecType v0 = VecType::Zero(), v1, v2, v3;
        for (Integer i = 0; i < N; i++) {
          v0.insert(i, (ScalarType)(drand48()-0.5)*4*const_pi<ScalarType>());
        }
        sincos(v1, v2, v0);
        v3 = exp(v0);
#if defined(SCTL_HAVE_SVML) || defined(SCTL_HAVE_LIBMVEC)
        VecType v4 = log(v0 + ScalarType(2.01) * const_pi<ScalarType>());
#endif
        for (Integer i = 0; i < N; i++) {
          ScalarType err_tol = pow<TypeTraits<ScalarType>::SigBits-3,ScalarType>((ScalarType)0.5);
          SCTL_ASSERT(fabs(v1[i] - sin<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v2[i] - cos<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v3[i] - exp<ScalarType>(v0[i]))/fabs(exp<ScalarType>(v0[i])) < err_tol);
#if defined(SCTL_HAVE_SVML) || defined(SCTL_HAVE_LIBMVEC)
          SCTL_ASSERT(fabs(v4[i] - log<ScalarType>(v0[i] + ScalarType(2.01) * const_pi<ScalarType>())) < err_tol);
#endif
        }

        approx_sincos<3>(v1, v2, v0);
        v3 = approx_exp<3>(v0);
        for (Integer i = 0; i < N; i++) {
          ScalarType err_tol = (ScalarType)1e-3;
          SCTL_ASSERT(fabs(v1[i] - sin<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v2[i] - cos<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v3[i] - exp<ScalarType>(v0[i]))/fabs(exp<ScalarType>(v0[i])) < err_tol);
        }

        approx_sincos<5>(v1, v2, v0);
        v3 = approx_exp<5>(v0);
        for (Integer i = 0; i < N; i++) {
          ScalarType err_tol = (ScalarType)1e-5;
          SCTL_ASSERT(fabs(v1[i] - sin<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v2[i] - cos<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v3[i] - exp<ScalarType>(v0[i]))/fabs(exp<ScalarType>(v0[i])) < err_tol);
        }

        { // approx_log, approx_pow to the given digits; zero and negative x as log and pow
          const VecType xp = fabs(v0) + (ScalarType)0.5;
          const auto check = [&xp, &v0](const VecType& l, const VecType& p, const ScalarType err_tol) {
            for (Integer i = 0; i < N; i++) {
              SCTL_ASSERT(fabs(l[i] - log<ScalarType>(xp[i])) <= err_tol * fabs(log<ScalarType>(xp[i])));
              SCTL_ASSERT(fabs(p[i] - pow<ScalarType>(xp[i], v0[i])) <= err_tol * pow<ScalarType>(xp[i], v0[i]));
            }
          };
          check(approx_log<3>(xp), approx_pow<3>(xp, v0), (ScalarType)1e-3);
          check(approx_log<5>(xp), approx_pow<5>(xp, v0), (ScalarType)1e-5);
          if constexpr (std::is_same<ScalarType,double>::value) {
            check(approx_log<9>(xp), approx_pow<9>(xp, v0), (ScalarType)1e-9);
            check(approx_log<12>(xp), approx_pow<12>(xp, v0), (ScalarType)1e-12);
          }
          SCTL_ASSERT(isinf(approx_log<5>(VecType((ScalarType)0))[0]) && approx_log<5>(VecType((ScalarType)0))[0] < 0 && isnan(approx_log<5>(VecType((ScalarType)-1))[0]));
          SCTL_ASSERT(fabs(approx_pow<5>(VecType((ScalarType)-2), VecType((ScalarType)3))[0] + 8) <= (ScalarType)8e-5);
        }

        { // exp10, and approx_exp10 to 5 digits
          const VecType e10 = exp10(v0), a10 = approx_exp10<5>(v0);
          for (Integer i = 0; i < N; i++) {
            const ScalarType ref = pow<ScalarType>((ScalarType)10, v0[i]);
            SCTL_ASSERT(fabs(e10[i] - ref) <= 8 * machine_eps<ScalarType>() * ref);
            SCTL_ASSERT(fabs(a10[i] - ref) <= (ScalarType)1e-5 * ref);
          }
          SCTL_ASSERT(isinf(exp10(VecType((ScalarType)INFINITY))[0]) && exp10(VecType(-(ScalarType)INFINITY))[0] == 0 && isnan(exp10(VecType((ScalarType)NAN))[0]));
        }

        { // sinpi, cospi, sincospi, and approx_sinpi, approx_cospi to 5 digits
          VecType s, c;
          sincospi(s, c, v0);
          const VecType sp = sinpi(v0), cp = cospi(v0), as = approx_sinpi<5>(v0), ac = approx_cospi<5>(v0);
          for (Integer i = 0; i < N; i++) {
            const ScalarType tol = (4 + 4 * fabs(v0[i])) * machine_eps<ScalarType>(); // with the rounding of pi x in the reference
            const ScalarType s_ref = sin<ScalarType>(const_pi<ScalarType>() * v0[i]), c_ref = cos<ScalarType>(const_pi<ScalarType>() * v0[i]);
            SCTL_ASSERT(fabs(sp[i] - s_ref) <= tol && fabs(cp[i] - c_ref) <= tol && fabs(s[i] - s_ref) <= tol && fabs(c[i] - c_ref) <= tol);
            SCTL_ASSERT(fabs(as[i] - s_ref) <= (ScalarType)1e-5 && fabs(ac[i] - c_ref) <= (ScalarType)1e-5);
          }
          const ScalarType h = pow<TypeTraits<ScalarType>::SigBits - 1, ScalarType>((ScalarType)2); // beyond 2^(SigBits-2): the range step
          SCTL_ASSERT(sinpi(VecType((ScalarType)3))[0] == 0 && cospi(VecType((ScalarType)3))[0] == -1 && sinpi(VecType((ScalarType)2.5))[0] == 1 && cospi(VecType((ScalarType)2.5))[0] == 0);
          SCTL_ASSERT(sinpi(VecType(h + (ScalarType)0.5))[0] == 1 && cospi(VecType(2 * h + 1))[0] == -1 && isnan(sinpi(VecType((ScalarType)INFINITY))[0]) && isnan(cospi(VecType((ScalarType)NAN))[0]));
        }

        { // sincpi, and approx_sincpi to 5 digits
          const VecType sc = sincpi(v0), ac = approx_sincpi<5>(v0);
          for (Integer i = 0; i < N; i++) {
            const ScalarType px = const_pi<ScalarType>() * v0[i];
            const ScalarType ref = sin<ScalarType>(px) / px;
            SCTL_ASSERT(fabs(sc[i] - ref) <= (4 + 4 * fabs(v0[i])) * machine_eps<ScalarType>() / fabs(px) + 4 * machine_eps<ScalarType>()); // with the rounding of pi x in the reference
            SCTL_ASSERT(fabs(ac[i] - ref) <= (ScalarType)1e-5);
          }
          SCTL_ASSERT(sincpi(VecType((ScalarType)0))[0] == 1 && sincpi(VecType((ScalarType)3))[0] == 0 && isnan(sincpi(VecType((ScalarType)NAN))[0]));
        }

        if (sizeof(ScalarType) < 16) return;

        approx_sincos<8>(v1, v2, v0);
        v3 = approx_exp<8>(v0);
        for (Integer i = 0; i < N; i++) {
          ScalarType err_tol = (ScalarType)1e-8;
          SCTL_ASSERT(fabs(v1[i] - sin<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v2[i] - cos<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v3[i] - exp<ScalarType>(v0[i]))/fabs(exp<ScalarType>(v0[i])) < err_tol);
        }

        approx_sincos<12>(v1, v2, v0);
        v3 = approx_exp<12>(v0);
        for (Integer i = 0; i < N; i++) {
          ScalarType err_tol = (ScalarType)1e-12;
          SCTL_ASSERT(fabs(v1[i] - sin<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v2[i] - cos<ScalarType>(v0[i])) < err_tol);
          SCTL_ASSERT(fabs(v3[i] - exp<ScalarType>(v0[i]))/fabs(exp<ScalarType>(v0[i])) < err_tol);
        }
      }

      static void test_reals_math() {
        UnionType u1, u2, u3, u4;
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)((drand48()-0.5)*20);
          u2.x[i] = (ScalarType)((drand48()-0.5)*20);
          u3.x[i] = (i % 2 ? (ScalarType)NAN : u1.x[i]);
          u4.x[i] = (ScalarType)(0.1 + 10*drand48());
        }
        const ScalarType eps = machine_eps<ScalarType>();
        const ScalarType err_tol = pow<TypeTraits<ScalarType>::SigBits-3,ScalarType>((ScalarType)0.5);

        const VecType a = fabs(u1.v);
        const VecType s = sqrt(a);
        const VecType r = rsqrt(a);
        const VecType f = floor(u1.v);
        const VecType c = ceil(u1.v);
        const VecType cs = copysign(u1.v, u2.v);
        const VecType nan = select(isnan(u3.v), VecType((ScalarType)1), VecType((ScalarType)0));
        const VecType ne = select(u3.v != u3.v, VecType((ScalarType)1), VecType((ScalarType)0));
        const VecType eq = select(u3.v == u3.v, VecType((ScalarType)1), VecType((ScalarType)0));
        const VecType mx0 = max(u3.v, u1.v);
        const VecType mx1 = max(u1.v, u3.v);
        const VecType mn0 = min(u3.v, u1.v);
        const VecType mn1 = min(u1.v, u3.v);
        const VecType sn = sin(u1.v);
        const VecType cn = cos(u1.v);
        const VecType tn = tan(u1.v);
        const VecType sn5 = approx_sin<5>(u1.v);
        const VecType cn5 = approx_cos<5>(u1.v);
        const VecType tn5 = approx_tan<5>(u1.v);
        const VecType at = atan2(u1.v, u2.v);
        const VecType pw = pow(u4.v, u2.v / (ScalarType)2);
        SCTL_ASSERT(1/fabs(VecType((ScalarType)-0.0))[0] > 0);
        for (Integer i = 0; i < N; i++) {
          const ScalarType x = u1.x[i];
          const ScalarType tan_x = tan<ScalarType>(x);
          SCTL_ASSERT(a[i] == fabs(x));
          SCTL_ASSERT(s[i] == sqrt<ScalarType>(a[i])); // correctly rounded, as the scalar sqrt
          SCTL_ASSERT(r[i] == 1/sqrt<ScalarType>(a[i]) || fabs(r[i] - 1/sqrt<ScalarType>(a[i])) <= 4*eps/sqrt<ScalarType>(a[i])); // within 2 ulp
          SCTL_ASSERT(f[i] == floor<ScalarType>(x));
          SCTL_ASSERT(c[i] == ceil<ScalarType>(x));
          SCTL_ASSERT(cs[i] == (u2.x[i] < 0 ? -a[i] : a[i]));
          SCTL_ASSERT(nan[i] == (ScalarType)(i % 2));
          SCTL_ASSERT(ne[i] == (ScalarType)(i % 2)); // NaN != NaN, as in C++
          SCTL_ASSERT(eq[i] == (ScalarType)(1 - i % 2));
          SCTL_ASSERT(mx0[i] == x && mn0[i] == x); // the second operand where the first is NaN
          SCTL_ASSERT(i % 2 ? (isnan(mx1[i]) && isnan(mn1[i])) : (mx1[i] == x && mn1[i] == x));
          SCTL_ASSERT(fabs(sn[i] - sin<ScalarType>(x)) < err_tol);
          SCTL_ASSERT(fabs(cn[i] - cos<ScalarType>(x)) < err_tol);
          SCTL_ASSERT(fabs(tn[i] - tan_x) < 4*err_tol*(1 + tan_x*tan_x));
          SCTL_ASSERT(fabs(sn5[i] - sin<ScalarType>(x)) < (ScalarType)1e-5);
          SCTL_ASSERT(fabs(cn5[i] - cos<ScalarType>(x)) < (ScalarType)1e-5);
          SCTL_ASSERT(fabs(tn5[i] - tan_x) < (ScalarType)4e-5*(1 + tan_x*tan_x));
          SCTL_ASSERT(fabs(at[i] - atan2<ScalarType>(x, u2.x[i])) <= 8*eps*fabs(atan2<ScalarType>(x, u2.x[i])));
          SCTL_ASSERT(fabs(pw[i] - pow<ScalarType>(u4.x[i], u2.x[i]/2)) <= 8*eps*pow<ScalarType>(u4.x[i], u2.x[i]/2));
        }
        for (const ScalarType x : {(ScalarType)100000.25, (ScalarType)-3e7, (ScalarType)1e20}) { // beyond the reduction of FullRange = false
          SCTL_ASSERT(fabs(sin(VecType(x))[0] - sin<ScalarType>(x)) < err_tol);
          SCTL_ASSERT(fabs(cos(VecType(x))[0] - cos<ScalarType>(x)) < err_tol);
          SCTL_ASSERT(fabs(approx_sin<8>(VecType(x))[0] - sin<ScalarType>(x)) < (ScalarType)1e-6);
        }
        { // the faster versions, for moderate arguments
          const VecType snf = sin<false>(u1.v);
          const VecType atf = atan2<false>(u1.v, u2.v);
          for (Integer i = 0; i < N; i++) {
            SCTL_ASSERT(fabs(snf[i] - sin<ScalarType>(u1.x[i])) < err_tol);
            SCTL_ASSERT(fabs(atf[i] - atan2<ScalarType>(u1.x[i], u2.x[i])) <= 8*eps*fabs(atan2<ScalarType>(u1.x[i], u2.x[i])));
          }
        }

        if constexpr (sizeof(ScalarType) <= sizeof(double)) { // rsqrt at zeros, inf, a subnormal, a negative value and NaN, as 1/sqrt
          const ScalarType sv[] = {(ScalarType)0, (ScalarType)-0.0, (ScalarType)INFINITY, std::numeric_limits<ScalarType>::denorm_min() * 8, (ScalarType)-1, (ScalarType)NAN};
          for (const ScalarType x : sv) {
            const ScalarType ref = 1/std::sqrt(x);
            const ScalarType r0 = rsqrt(VecType(x))[0];
            SCTL_ASSERT(std::isnan(ref) ? std::isnan(r0) : (r0 == ref || fabs(r0 - ref) <= 4*eps*ref));
          }
        }

        if constexpr (sizeof(ScalarType) <= sizeof(double)) { // atan2 at zeros, infinities and NaN
          const ScalarType sv[] = {(ScalarType)0, (ScalarType)-0.0, (ScalarType)1, (ScalarType)-1, (ScalarType)INFINITY, (ScalarType)-INFINITY, (ScalarType)NAN};
          for (const ScalarType y : sv) {
            for (const ScalarType x : sv) {
              const ScalarType ref = std::atan2(y, x);
              const ScalarType at0 = atan2(VecType(y), VecType(x))[0];
              SCTL_ASSERT(std::isnan(ref) ? std::isnan(at0) : (fabs(at0 - ref) <= 8*eps*fabs(ref) && std::signbit(at0) == std::signbit(ref)));
            }
          }
        }
      }

      static void test_reals_math2() {
        UnionType u1, u2, u3, u4, u5;
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)((drand48()-0.5)*20);
          u2.x[i] = (ScalarType)((drand48()-0.5)*20);
          u3.x[i] = (i % 3 == 0 ? u1.x[i] : (i % 3 == 1 ? (ScalarType)INFINITY : (ScalarType)NAN));
          u4.x[i] = (ScalarType)(0.1 + 10*drand48());
          u5.x[i] = (ScalarType)(2*drand48()-1);
        }
        const ScalarType eps = machine_eps<ScalarType>();
        const ScalarType err_tol = pow<TypeTraits<ScalarType>::SigBits-3,ScalarType>((ScalarType)0.5);
        const auto rel = [](ScalarType a, ScalarType b) { return fabs(a - b) / fabs(b); };
        const ScalarType tol = (TypeTraits<ScalarType>::SigBits > 64 ? 64 : 16)*eps; // for QuadReal, both sides have up to about 26 eps error
        const auto sinh_ref = [](const ScalarType x) -> ScalarType { // long double std::sinh where it has enough digits; else exp, or a series near 0
          if (TypeTraits<ScalarType>::SigBits <= 64) return (ScalarType)std::sinh((long double)x);
          if (fabs(x) >= (ScalarType)0.5) return (exp<ScalarType>(x) - exp<ScalarType>(-x)) / 2;
          ScalarType term = x;
          ScalarType sum = x;
          for (Integer k = 1; k < 20; k++) {
            term *= x * x / (ScalarType)((2*k) * (2*k+1));
            sum += term;
          }
          return sum;
        };
        const auto cosh_ref = [](const ScalarType x) -> ScalarType {
          if (TypeTraits<ScalarType>::SigBits <= 64) return (ScalarType)std::cosh((long double)x);
          return (exp<ScalarType>(x) + exp<ScalarType>(-x)) / 2;
        };

        const VecType tr = trunc(u1.v);
        const VecType rn = round(u1.v);
        const VecType inf = select(isinf(u3.v), VecType((ScalarType)1), VecType((ScalarType)0));
        const VecType fin = select(isfinite(u3.v), VecType((ScalarType)1), VecType((ScalarType)0));
        const VecType at = atan(u1.v);
        const VecType as = asin(u5.v);
        const VecType ac = acos(u5.v);
        const VecType hy = hypot(u1.v, u2.v);
        const VecType e2 = exp2(u1.v);
        const VecType l2 = log2(u4.v);
        const VecType l10 = log10(u4.v);
        const VecType cr = cbrt(u1.v);
        const VecType sh = sinh(u1.v);
        const VecType ch = cosh(u1.v);
        const VecType th = tanh(u1.v);
        const VecType fm = fmod(u1.v, u4.v);
        SCTL_ASSERT(round(VecType((ScalarType)2.5))[0] == 3 && round(VecType((ScalarType)-0.5))[0] == -1);
        for (Integer i = 0; i < N; i++) {
          const ScalarType x = u1.x[i];
          SCTL_ASSERT(tr[i] == trunc<ScalarType>(x));
          SCTL_ASSERT(rn[i] == round<ScalarType>(x));
          SCTL_ASSERT(inf[i] == (ScalarType)(i % 3 == 1) && fin[i] == (ScalarType)(i % 3 == 0));
          SCTL_ASSERT(rel(at[i], atan<ScalarType>(x)) <= tol);
          SCTL_ASSERT(rel(as[i], asin<ScalarType>(u5.x[i])) <= tol);
          SCTL_ASSERT(rel(ac[i], acos<ScalarType>(u5.x[i])) <= tol);
          SCTL_ASSERT(rel(hy[i], hypot<ScalarType>(x, u2.x[i])) <= tol);
          SCTL_ASSERT(rel(e2[i], pow<ScalarType>((ScalarType)2, x)) <= err_tol);
          SCTL_ASSERT(fabs(l2[i] - log2<ScalarType>(u4.x[i])) <= 8*eps*fabs(log2<ScalarType>(u4.x[i])));
          SCTL_ASSERT(fabs(l10[i] - log<ScalarType>(u4.x[i])/log<ScalarType>((ScalarType)10)) <= 8*eps*fabs(log<ScalarType>(u4.x[i])/log<ScalarType>((ScalarType)10)));
          SCTL_ASSERT(fabs(cr[i]*cr[i]*cr[i] - x) <= 16*eps*fabs(x));
          SCTL_ASSERT(rel(sh[i], sinh_ref(x)) <= tol);
          SCTL_ASSERT(rel(ch[i], cosh_ref(x)) <= tol);
          SCTL_ASSERT(rel(th[i], (TypeTraits<ScalarType>::SigBits <= 64 ? (ScalarType)std::tanh((long double)x) : sinh_ref(x) / cosh_ref(x))) <= tol);
          SCTL_ASSERT(fm[i] == fmod<ScalarType>(x, u4.x[i]));
        }

        const ScalarType big = (ScalarType)(sizeof(ScalarType) == 4 ? 1e30 : 1e200); // big^2 overflows float
        SCTL_ASSERT(rel(hypot(VecType(big), VecType(big))[0], big * sqrt<ScalarType>((ScalarType)2)) <= tol);
        SCTL_ASSERT(isinf(hypot(VecType((ScalarType)INFINITY), VecType((ScalarType)NAN))[0]));
        const VecType hyf = hypot<false>(u1.v, u2.v);
        for (Integer i = 0; i < N; i++) SCTL_ASSERT(rel(hyf[i], hypot<ScalarType>(u1.x[i], u2.x[i])) <= tol);

        for (const ScalarType x : {(ScalarType)1e6, (ScalarType)INFINITY}) { // exp and approx_exp beyond the range of the result
          SCTL_ASSERT(isinf(exp(VecType(x))[0]) && exp(VecType(x))[0] > 0);
          SCTL_ASSERT(isinf(approx_exp<8>(VecType(x))[0]) && approx_exp<8>(VecType(x))[0] > 0);
          SCTL_ASSERT(exp(VecType(-x))[0] == 0 && approx_exp<8>(VecType(-x))[0] == 0);
        }
        SCTL_ASSERT(isnan(exp(VecType((ScalarType)NAN))[0]) && isnan(approx_exp<8>(VecType((ScalarType)NAN))[0]));
        if constexpr (sizeof(ScalarType) <= sizeof(double)) { // exp, log and pow near the overflow and with subnormal values, as std
          const ScalarType big = std::log(std::numeric_limits<ScalarType>::max()) - (ScalarType)0.3; // e^big within a factor 1.35 of the largest value
          const ScalarType sub = std::log(std::numeric_limits<ScalarType>::denorm_min()) + (ScalarType)8; // e^sub subnormal
          const ScalarType xsub = std::numeric_limits<ScalarType>::denorm_min() * 1000;
          SCTL_ASSERT(rel(exp(VecType(big))[0], std::exp(big)) <= 8*eps);
          SCTL_ASSERT(fabs(exp(VecType(sub))[0] - std::exp(sub)) <= std::numeric_limits<ScalarType>::denorm_min());
          SCTL_ASSERT(rel(log(VecType(xsub))[0], std::log(xsub)) <= 8*eps);
          SCTL_ASSERT(rel(pow(VecType((ScalarType)2), VecType(big / std::log((ScalarType)2)))[0], std::pow((ScalarType)2, big / std::log((ScalarType)2))) <= 64*eps);
          SCTL_ASSERT(fabs(pow(VecType((ScalarType)0.5), VecType(-sub / std::log((ScalarType)2)))[0] - std::pow((ScalarType)0.5, -sub / std::log((ScalarType)2))) <= std::numeric_limits<ScalarType>::denorm_min());
        }
        SCTL_ASSERT(fmod(VecType((ScalarType)5.5), VecType((ScalarType)INFINITY))[0] == (ScalarType)5.5);
        SCTL_ASSERT(cbrt(VecType((ScalarType)0))[0] == 0 && isinf(cbrt(VecType(-(ScalarType)INFINITY))[0]) && fabs(cbrt(VecType((ScalarType)-27))[0] + 3) <= 8*eps);
        if constexpr (sizeof(ScalarType) <= sizeof(double)) { // log and pow at zeros, infinities, NaN and signs, as std
          const auto same = [eps](ScalarType a, ScalarType b) { return std::isnan(b) ? std::isnan(a) : (a == b ? std::signbit(a) == std::signbit(b) : std::isfinite(b) && fabs(a - b) <= 16*eps*fabs(b)); };
          const ScalarType sv[] = {(ScalarType)0, (ScalarType)-0.0, (ScalarType)1, (ScalarType)-1, (ScalarType)2, (ScalarType)-2, (ScalarType)0.5, (ScalarType)3, (ScalarType)INFINITY, -(ScalarType)INFINITY, (ScalarType)NAN};
          for (const ScalarType x : sv) {
            SCTL_ASSERT(same(log(VecType(x))[0], std::log(x)));
            for (const ScalarType y : sv) SCTL_ASSERT(same(pow(VecType(x), VecType(y))[0], std::pow(x, y)));
          }
        }
        const VecType ae = approx_exp<8>(u1.v);
        const VecType ae_unchecked = approx_exp<8, false>(u1.v);
        for (Integer i = 0; i < N; i++) SCTL_ASSERT(fabs(ae_unchecked[i] - ae[i]) <= err_tol * ae[i]); // equal in the range, up to FMA contraction in generic code
      }

      static void test_reals_rsqrt() {
        UnionType u1, b1, b2, u2, u3, u4, u5;
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)rand();
          b1.x[i] = (ScalarType)rand();
          b2.x[i] = (ScalarType)rand();
        }

        u2.v = approx_rsqrt<4>(u1.v);
        u3.v = approx_rsqrt<7>(u1.v);
        u4.v = approx_rsqrt<4>(u1.v, b1.v>b2.v);
        u5.v = approx_rsqrt<7>(u1.v, b1.v>b2.v);
        for (Integer i = 0; i < N; i++) {
          ScalarType err = fabs(u2.x[i] - 1/sqrt<ScalarType>(u1.x[i]));
          ScalarType max_val = fabs(1/sqrt<ScalarType>(u1.x[i]));
          ScalarType rel_err = err / max_val;
          SCTL_ASSERT(rel_err < (pow<11,ScalarType>((ScalarType)0.5)));

          err = u4.x[i] - (b1.x[i]>b2.x[i] ? u2.x[i] : 0);
          rel_err = err / max_val;
          SCTL_ASSERT(rel_err < (pow<11,ScalarType>((ScalarType)0.5)));
          SCTL_ASSERT(b1.x[i]>b2.x[i] || (u4.x[i]==0));
        }
        for (Integer i = 0; i < N; i++) {
          ScalarType err = fabs((ScalarType)(u3.x[i] - 1/sqrt((double)u1.x[i]))); // float is not accurate enough to compute reference solution with 7-digits
          ScalarType max_val = fabs(1/sqrt<ScalarType>(u1.x[i]));
          ScalarType rel_err = err / max_val;
          SCTL_ASSERT(rel_err < (pow<22,ScalarType>((ScalarType)0.5)));

          err = u5.x[i] - (b1.x[i]>b2.x[i] ? u3.x[i] : 0);
          rel_err = err / max_val;
          SCTL_ASSERT(rel_err < (pow<22,ScalarType>((ScalarType)0.5)));
          SCTL_ASSERT(b1.x[i]>b2.x[i] || (u5.x[i]==0));
        }

        // x = 10^e over the exponent range: |e| <= 300 for double, 30 for float
        const double max_exp = (sizeof(ScalarType) >= sizeof(double) ? 300 : 30);
        for (Integer i = 0; i < N; i++) {
          u1.x[i] = (ScalarType)std::pow(10.0, (2*drand48()-1) * max_exp);
        }
        u2.v = approx_rsqrt<4>(u1.v);
        u3.v = approx_rsqrt<-1>(u1.v);
        for (Integer i = 0; i < N; i++) {
          const ScalarType r = 1/sqrt<ScalarType>(u1.x[i]);
          SCTL_ASSERT(fabs(u2.x[i]/r - 1) < (ScalarType)1e-4);
          SCTL_ASSERT(fabs(u3.x[i]/r - 1) < 8*machine_eps<ScalarType>());
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

#endif // _SCTL_VEC_TEST_HPP_
