// Per-function tests for sctl/intrin-wrapper.hpp.
//
// intrin-wrapper provides VData-level SIMD primitives parameterized over
// (ScalarType, lane count, ISA). The full (T,N) sweep is already covered by
// `test-vec` via the existing VecTest framework on the high-level Vec<T,N>.
// This test exercises the *generic / scalar-fallback* path that's compiled
// regardless of host ISA: VecData<double, 1> and VecData<int32_t, 1>.

#include <cstdint>
#include <cstdio>

#include "sctl/common.hpp"
#include "sctl/intrin-wrapper.hpp"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Integer;

template <class T> using VD = sctl::VecData<T, 1>;

int main() {
  using D = VD<double>;
  using I = VD<int32_t>;

  // --- zero_intrin / set1_intrin / extract_intrin ---
  std::printf("zero / set1 / extract :\n");
  {
    D z = sctl::zero_intrin<D>();
    CHECK(sctl::extract_intrin(z, 0) == 0.0);
    D one = sctl::set1_intrin<D>(1.5);
    CHECK(sctl::extract_intrin(one, 0) == 1.5);
  }

  // --- set_intrin (variadic) ---
  std::printf("set_intrin :\n");
  {
    D v = sctl::set_intrin<D>(7.25);
    CHECK(sctl::extract_intrin(v, 0) == 7.25);
  }

  // --- load{,u}_intrin / store{,u}_intrin / load1_intrin round-trip ---
  std::printf("load / store round-trip :\n");
  {
    alignas(64) double buf_in [1] = {3.14};
    alignas(64) double buf_out[1] = {0.0};
    D v = sctl::load_intrin<D>(buf_in);
    sctl::store_intrin(buf_out, v);
    CHECK(buf_out[0] == 3.14);
    // unaligned variant
    double misc_in[1] = {2.71}, misc_out[1] = {0.0};
    D vu = sctl::loadu_intrin<D>(misc_in);
    sctl::storeu_intrin(misc_out, vu);
    CHECK(misc_out[0] == 2.71);
    // load1: replicate the scalar across all lanes (N=1, so just stores it)
    double s = 9.81;
    D v1 = sctl::load1_intrin<D>(&s);
    CHECK(sctl::extract_intrin(v1, 0) == 9.81);
  }

  // --- insert_intrin ---
  std::printf("insert :\n");
  {
    D v = sctl::zero_intrin<D>();
    sctl::insert_intrin(v, 0, 4.2);
    CHECK(sctl::extract_intrin(v, 0) == 4.2);
  }

  // --- arithmetic: add / sub / mul / div / unary_minus / fma ---
  std::printf("arithmetic :\n");
  {
    D a = sctl::set1_intrin<D>(6.0);
    D b = sctl::set1_intrin<D>(2.0);
    CHECK(sctl::extract_intrin(sctl::add_intrin(a, b),         0) ==  8.0);
    CHECK(sctl::extract_intrin(sctl::sub_intrin(a, b),         0) ==  4.0);
    CHECK(sctl::extract_intrin(sctl::mul_intrin(a, b),         0) == 12.0);
    CHECK(sctl::extract_intrin(sctl::div_intrin(a, b),         0) ==  3.0);
    CHECK(sctl::extract_intrin(sctl::unary_minus_intrin(a),    0) == -6.0);
    // fma: a*b + c = 6*2 + 5 = 17
    D c = sctl::set1_intrin<D>(5.0);
    CHECK(sctl::extract_intrin(sctl::fma_intrin(a, b, c), 0) == 17.0);
  }

  // --- min / max ---
  std::printf("min / max :\n");
  {
    D a = sctl::set1_intrin<D>(3.0);
    D b = sctl::set1_intrin<D>(5.0);
    CHECK(sctl::extract_intrin(sctl::min_intrin(a, b), 0) == 3.0);
    CHECK(sctl::extract_intrin(sctl::max_intrin(a, b), 0) == 5.0);
  }

  // --- bitwise ops on integer lanes: and / or / xor / not / andnot ---
  std::printf("bitwise integer :\n");
  {
    I x = sctl::set1_intrin<I>(0x0F0F);
    I y = sctl::set1_intrin<I>(0x00FF);
    CHECK((sctl::extract_intrin(sctl::and_intrin   (x, y), 0) & 0xFFFF) == 0x000F);
    CHECK((sctl::extract_intrin(sctl::or_intrin    (x, y), 0) & 0xFFFF) == 0x0FFF);
    CHECK((sctl::extract_intrin(sctl::xor_intrin   (x, y), 0) & 0xFFFF) == 0x0FF0);
    CHECK((sctl::extract_intrin(sctl::andnot_intrin(x, y), 0) & 0xFFFF) == 0x0F00);
    // not_intrin: all bits flipped
    I z = sctl::set1_intrin<I>(0x00000000);
    CHECK(sctl::extract_intrin(sctl::not_intrin(z), 0) == (int32_t)-1);
  }

  // --- bitshift ---
  std::printf("bitshift :\n");
  {
    I x = sctl::set1_intrin<I>(1);
    CHECK(sctl::extract_intrin(sctl::bitshiftleft_intrin (x, 4), 0) == 16);
    I y = sctl::set1_intrin<I>(64);
    CHECK(sctl::extract_intrin(sctl::bitshiftright_intrin(y, 2), 0) == 16);
  }

  // --- reinterpret_intrin (bitcast between same-size vecs) ---
  std::printf("reinterpret :\n");
  {
    using D64 = VD<double>;
    using I64 = VD<int64_t>;
    D64 d = sctl::set1_intrin<D64>(1.0);
    I64 i = sctl::reinterpret_intrin<I64>(d);
    // 1.0 in IEEE-754 double has the bit pattern 0x3FF0000000000000.
    CHECK(sctl::extract_intrin(i, 0) == (int64_t)0x3FF0000000000000ll);
  }

  TEST_SUMMARY_RETURN();
}
