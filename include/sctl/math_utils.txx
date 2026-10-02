#ifndef _SCTL_MATH_UTILS_TXX_
#define _SCTL_MATH_UTILS_TXX_

#include <istream>              // for istream
#include <ostream>              // for ostream
#include <stdlib.h>             // for abs
#include <cctype>               // for isspace
#include <ctype.h>              // for isspace
#include <algorithm>            // for max
#include <cmath>                // for acos, asin, atan, log, sqrt, NAN, pow
#include <cstring>              // for strlen
#include <istream>              // for basic_istream, ws
#include <ostream>              // for basic_ostream, operator<<
#include <string>               // for basic_string, string, to_string
#include <charconv>             // for from_chars (defines __cpp_lib_to_chars, tested below)
#include <system_error>         // for errc
#include <vector>               // for vector

#include "sctl/common.hpp"      // for Long, Integer, SCTL_ASSERT, SCTL_NAME...
#include "sctl/math_utils.hpp"  // for QuadReal, operator/, cos, operator*

namespace sctl {

template <class Real, Integer bits = sizeof(Real)*8> struct GetSigBits {
  static constexpr Integer value() {
    return ((Real)(pow<bits>((Real)0.5)+1) == (Real)1 ? GetSigBits<Real,bits-1>::value() : bits);
  }
};
template <class Real> struct GetSigBits<Real,0> {
  static constexpr Integer value() {
    return 0;
  }
};
template <class Real> inline constexpr Integer significant_bits() {
  return GetSigBits<Real>::value();
}

template <class Real> inline constexpr Real machine_eps() {
  return pow<-GetSigBits<Real>::value()-1,Real>(2);
}

namespace detail {

template <class Real> inline Real atoreal_generic(const char* str) { // Warning: does not do correct rounding
  const auto get_num = [](const char* str, int& end) {
    Real val = 0, exp = 1;
    for (int i = end; i >= 0; i--) {
      char c = str[i];
      if ('0' <= c && c <= '9') {
        val += (c - '0') * exp;
        exp *= 10;
      } else if (c == '.') {
        val /= exp;
        exp = 1;
      } else if (c == '-') {
        val = -val;
      } else if (c == '+') {
      } else {
        end = i;
        break;
      }
      end = i - 1;
    }
    return val;
  };

  Real val = 0;
  int i = std::strlen(str)-1;
  for (; i >= 0; i--) { // ignore trailing non-numeric characters
    if ('0' <= str[i] && str[i] <= '9') break;
  }
  val = get_num(str, i);
  if (i>0 && (str[i] == 'e' || str[i] == 'E')) {
    i--;
    val = get_num(str, i) * sctl::pow<Real,Real>((Real)10, val);
  }
  for (; i >= 0; i--) { // ignore leading whitespace
    SCTL_ASSERT(str[i] == ' ');
  }
  return val;
}

}  // namespace detail

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
namespace detail {

// std::from_chars is correctly rounded and, unlike strtod / std::stod, independent of the
// LC_NUMERIC decimal separator. It takes no leading whitespace and no leading '+', both of which
// atoreal accepts, and it leaves the value at 0 when nothing parses. For a value out of range it sets
// none, so those take atoreal_generic, which gives inf or 0.
template <class ValueType> inline ValueType from_chars_parse(const char* str) {
  const char* p = str;
  while (std::isspace((unsigned char)*p)) p++; // cast: isspace is undefined for negative char
  if (*p == '+') p++;
  ValueType v = 0;
  if (std::from_chars(p, p + std::strlen(p), v).ec == std::errc::result_out_of_range) return atoreal_generic<ValueType>(str);
  return v;
}

}  // namespace detail

template <> inline float atoreal<float>(const char* str) { return detail::from_chars_parse<float>(str); }
template <> inline double atoreal<double>(const char* str) { return detail::from_chars_parse<double>(str); }
template <> inline long double atoreal<long double>(const char* str) { return detail::from_chars_parse<long double>(str); }
#endif

template <class Real> inline Real atoreal(const char* str) { return detail::atoreal_generic<Real>(str); }

template <class Real> static inline constexpr bool isinf_generic(const Real a) {
  return (a==2*a && a!=0);
}
template <class Real> static inline constexpr bool isnan_generic(const Real a) {
  return (a!=a);
}

template <class Real> static inline constexpr Real fabs_generic(const Real a) {
  return (a<0?-a:a);
}

template <class Real> static inline Real round_generic(const Real& x) {
  return trunc(x+(Real)0.5) - (x<(Real)-0.5);
}

template <class Real> static inline Real floor_generic(const Real& x) {
  return trunc(x) - (x<0);
}

template <class Real> static inline Real ceil_generic(const Real& a) {
  const auto trunc_a = trunc(a);
  return (trunc_a == a ? trunc_a : trunc_a + (a>0));
}

template <class Real> static inline Real sqrt_generic(const Real a) {
  Real b = ::sqrt((double)a);
  if (a > 0) { // Newton iterations for greater accuracy
    b = (b + a / b) * 0.5;
    b = (b + a / b) * 0.5;
  }
  return b;
}

// Scale-by-max formulation: keeps the radicand in [1, 2] so the result is
// representable whenever the true Euclidean norm is.
template <class Real> static inline Real hypot_generic(const Real a, const Real b) {
  const Real abs_a = fabs<Real>(a);
  const Real abs_b = fabs<Real>(b);
  const Real m = (abs_a > abs_b) ? abs_a : abs_b;
  if (m == (Real)0) return (Real)0;
  const Real inv_m = (Real)1 / m;
  const Real ra = abs_a * inv_m;
  const Real rb = abs_b * inv_m;
  return m * sqrt_generic(ra * ra + rb * rb);
}

template <class Real> static inline void sincos_generic(const Real a, Real& sin_a, Real& cos_a) {
  const int N = 200;
  static std::vector<Real> theta;
  static std::vector<Real> sinval;
  static std::vector<Real> cosval;
  if (theta.size() == 0) {
#pragma omp critical(SCTL_QUAD_SINCOS)
    if (theta.size() == 0) {
      sinval.resize(N);
      cosval.resize(N);

      Real t = 1.0;
      std::vector<Real> theta_(N);
      for (int i = 0; i < N; i++) {
        theta_[i] = t;
        t = t * 0.5;
      }

      sinval[N - 1] = theta_[N - 1];
      cosval[N - 1] = 1.0 - sinval[N - 1] * sinval[N - 1] / 2;
      for (int i = N - 2; i >= 0; i--) {
        sinval[i] = 2.0 * sinval[i + 1] * cosval[i + 1];
        cosval[i] = cosval[i + 1] * cosval[i + 1] - sinval[i + 1] * sinval[i + 1];
        Real s = 1 / sqrt<Real>(cosval[i] * cosval[i] + sinval[i] * sinval[i]);
        sinval[i] *= s;
        cosval[i] *= s;
      }
      theta_.swap(theta);
    }
  }

  if (a == 0) { // keeps the sign of zero
    sin_a = a;
    cos_a = 1;
    return;
  }
#ifdef __SIZEOF_INT128__
  if (!(fabs<Real>(a) < (Real)INFINITY)) {
#else
  if (!(fabs<Real>(a) < (Real)0x1p62)) { // without 128-bit integers, no reduction beyond 2^62
#endif
    sin_a = (Real)NAN;
    cos_a = (Real)NAN;
    return;
  }
  // a = n pi/2 + r, q = n mod 4
  Real r;
  Integer q;
  if (fabs<Real>(a) < (Real)0x1p62) { // pi/2 = sum of pi2[i], each with 51 significant bits, so n * pi2[i] is exact
    const Real n = round<Real>(a * ((Real)2 / const_pi<Real>()));
    const Real pi2[6] = {(Real)0x1.921fb54442d18p+0, (Real)0x1.1a62633145c04p-54, (Real)0x1.707344a409380p-105, (Real)0x1.114cf98e80414p-156, (Real)0x1.bea63b139b224p-207, (Real)0x1.14a08798e3404p-259};
    r = a;
    for (const Real& p : pi2) r -= n * p;
    q = (Integer)((Long)n & 3);
  } else {
#ifdef __SIZEOF_INT128__
    // Payne-Hanek: |a| = M 2^E with an integer M < 2^114, and |a| 2/pi mod 4 from the bits of 2/pi that matter
    using U128 = unsigned __int128;
    static constexpr uint64_t two_over_pi[261] = { // bit i of 2/pi weighs 2^-i; word w holds bits 64w+1 .. 64w+64
        0xa2f9836e4e441529ull, 0xfc2757d1f534ddc0ull, 0xdb6295993c439041ull, 0xfe5163abdebbc561ull, 0xb7246e3a424dd2e0ull, 0x06492eea09d1921cull,
        0xfe1deb1cb129a73eull, 0xe88235f52ebb4484ull, 0xe99c7026b45f7e41ull, 0x3991d639835339f4ull, 0x9c845f8bbdf9283bull, 0x1ff897ffde05980full,
        0xef2f118b5a0a6d1full, 0x6d367ecf27cb09b7ull, 0x4f463f669e5fea2dull, 0x7527bac7ebe5f17bull, 0x3d0739f78a5292eaull, 0x6bfb5fb11f8d5d08ull,
        0x56033046fc7b6babull, 0xf0cfbc209af4361dull, 0xa9e391615ee61b08ull, 0x6599855f14a06840ull, 0x8dffd8804d732731ull, 0x06061556ca73a8c9ull,
        0x60e27bc08c6b47c4ull, 0x19c367cddce8092aull, 0x8359c4768b961ca6ull, 0xddaf44d15719053eull, 0xa5ff07053f7e33e8ull, 0x32c2de4f98327dbbull,
        0xc33d26ef6b1e5ef8ull, 0x9f3a1f35caf27f1dull, 0x87f121907c7c246aull, 0xfa6ed5772d30433bull, 0x15c614b59d19c3c2ull, 0xc4ad414d2c5d000cull,
        0x467d862d71e39ac6ull, 0x9b0062337cd2b497ull, 0xa7b4d55537f63ed7ull, 0x1810a3fc764d2a9dull, 0x64abd770f87c6357ull, 0xb07ae715175649c0ull,
        0xd9d63b3884a7cb23ull, 0x24778ad623545ab9ull, 0x1f001b0af1dfce19ull, 0xff319f6a1e666157ull, 0x9947fbacd87f7eb7ull, 0x652289e83260bfe6ull,
        0xcdc4ef09366cd43full, 0x5dd7de16de3b5892ull, 0x9bde2822d2e88628ull, 0x4d58e232cac616e3ull, 0x08cb7de050c017a7ull, 0x1df35be01834132eull,
        0x6212830148835b8eull, 0xf57fb0adf2e91e43ull, 0x4a48d36710d8ddaaull, 0x425faece616aa428ull, 0x0ab499d3f2a6067full, 0x775c83c2a3883c61ull,
        0x78738a5a8cafbdd7ull, 0x6f63a62dcbbff4efull, 0x818d67c12645ca55ull, 0x36d9cad2a8288d61ull, 0xc277c9121426049bull, 0x4612c459c444c5c8ull,
        0x91b24df31700ad43ull, 0xd4e5492910d5fdfcull, 0xbe00cc941eeece70ull, 0xf53e1380f1ecc3e7ull, 0xb328f8c79405933eull, 0x71c1b3092ef3450bull,
        0x9c12887b20ab9fb5ull, 0x2ec292472f327b6dull, 0x550c90a7721fe76bull, 0x96cb314a1679e279ull, 0x4189dff49794e884ull, 0xe6e29731996bed88ull,
        0x365f5f0efdbbb49aull, 0x486ca46742727132ull, 0x5d8db8159f09e5bcull, 0x25318d3974f71c05ull, 0x30010c0d68084b58ull, 0xee2c90aa4702e774ull,
        0x24d6bda67df77248ull, 0x6eef169fa6948ef6ull, 0x91b45153d1f20acfull, 0x3398207e4bf56863ull, 0xb25f3edd035d407full, 0x8985295255c06437ull,
        0x10d86d324832754cull, 0x5bd4714e6e5445c1ull, 0x090b69f52ad56614ull, 0x9d072750045ddb3bull, 0xb4c576ea17f9877dull, 0x6b49ba271d296996ull,
        0xacccc65414ad6ae2ull, 0x9089d98850722cbeull, 0xa4049407777030f3ull, 0x27fc00a871ea49c2ull, 0x663de06483dd9797ull, 0x3fa3fd94438c860dull,
        0xde41319d39928c70ull, 0xdde7b7173bdf082bull, 0x3715a0805c93805aull, 0x921110d8e80faf80ull, 0x6c4bffdb0f903876ull, 0x185915a562bbcb61ull,
        0xb989c7bd401004f2ull, 0xd2277549f6b6ebbbull, 0x22dbaa140a2f2689ull, 0x768364333b091a94ull, 0x0eaa3a51c2a31daeull, 0xedaf12265c4dc26dull,
        0x9c7a2d9756c0833full, 0x03f6f0098c402b99ull, 0x316d07b43915200cull, 0x5bc3d8c492f54badull, 0xc6a5ca4ecd37a736ull, 0xa9e69492ab6842ddull,
        0xde6319ef8c76528bull, 0x6837dbfcaba1ae31ull, 0x15dfa1ae00dafb0cull, 0x664d64b705ed3065ull, 0x29bf56573aff47b9ull, 0xf96af3be75df9328ull,
        0x3080abf68c6615cbull, 0x040622fa1de4d9a4ull, 0xb33d8f1b5709cd36ull, 0xe9424ea4be13b523ull, 0x331aaaf0a8654fa5ull, 0xc1d20f3f0bcd785bull,
        0x76f923048b7b7217ull, 0x8953a6c6e26e6f00ull, 0xebef584a9bb7dac4ull, 0xba66aacfcf761d02ull, 0xd12df1b1c1998c77ull, 0xadc3da4886a05df7ull,
        0xf480c62ff0ac9aecull, 0xddbc5c3f6dded01full, 0xc790b6db2a3a25a3ull, 0x9aaf009353ad0457ull, 0xb6b42d297e804ba7ull, 0x07da0eaa76a1597bull,
        0x2a12162db7dcfde5ull, 0xfafedb89fdbe896cull, 0x76e4fca90670803eull, 0x156e85ff87fd073eull, 0x2833676186182aeaull, 0xbd4dafe7b36e6d8full,
        0x3967955bbf3148d7ull, 0x8416df30432dc735ull, 0x6125ce70c9b8cb30ull, 0xfd6cbfa200a4e46cull, 0x05a0dd5a476f21d2ull, 0x1262845cb9496170ull,
        0xe0566b0152993755ull, 0x50b7d51ec4f1335full, 0x6e13e4305da92e85ull, 0xc3b21d3632a1a4b7ull, 0x08d4b1ea21f716e4ull, 0x698f77ff2780030cull,
        0x2d408da0cd4f99a5ull, 0x20d3a2b30a5d2f42ull, 0xf9b4cbda11d0be7dull, 0xc1db9bbd17ab81a2ull, 0xca5c6a0817552e55ull, 0x0027f0147f8607e1ull,
        0x640b148d4196debeull, 0x872afddab6256b34ull, 0x897bfef3059ebfb9ull, 0x4f6a68a82a4a5ac4ull, 0x4fbcf82d985ad795ull, 0xc7f48d4d0da63a20ull,
        0x5f57a4b13f149538ull, 0x800120cc86dd71b6ull, 0xdec9f560bf11654dull, 0x6b0701acb08cd0c0ull, 0xb24855510efb1ec3ull, 0x72953b06a33540c0ull,
        0x7bdc06cc45e0fa29ull, 0x4ec8cad641f3e8deull, 0x647cd8649b31bed9ull, 0xc397a4d45877c5e3ull, 0x6913daf03c3aba46ull, 0x18465f7555f5bdd2ull,
        0xc6926e5d2eaced44ull, 0x0e423e1c87c461e9ull, 0xfd29f3d6e7ca7c22ull, 0x35916fc5e0088dd7ull, 0xffe26a6ec6fdb0c1ull, 0x0893745d7cb2ad6bull,
        0x9d6ecd7b723e6a11ull, 0xc6a9cff7df7329baull, 0xc9b55100b70db2e2ull, 0x24ba74607de58ad8ull, 0x742c150d0c188194ull, 0x667e162901767a9full,
        0xbefdfdef4556367eull, 0xd913d9ecb9ba8bfcull, 0x97c427a831c36ef1ull, 0x36c59456a8d8b5a8ull, 0xb40ecccf2d891234ull, 0x576f89562ce3ce99ull,
        0xb920d6aa5e6b9c2aull, 0x3ecc5f114a0bfdfbull, 0xf4e16d3b8e2c86e2ull, 0x84d4e9a9b4fcd1eeull, 0xefc9352e61392f44ull, 0x2138c8d91b0afc81ull,
        0x6a4afbd81c2f84b4ull, 0x538c994ecc2254dcull, 0x552ad6c6c096190bull, 0xb8701a649569605aull, 0x26ee523f0f117f11ull, 0xb5f4f5cbfc2dbc34ull,
        0xeebc34cc5de8605eull, 0xdd9b8e67ef3392b8ull, 0x17c99b5861bc57e1ull, 0xc68351103ed84871ull, 0xdddd1c2da118af46ull, 0x2c21d7f359987ad9ull,
        0xc0549efa864ffc06ull, 0x56ae79e536228922ull, 0xad38dc9367aae855ull, 0x3826829be7caa40dull, 0x51b133990ed7a948ull, 0x0569f0b265a7887full,
        0x974c8836d1f9b392ull, 0x214a827b21cf98dcull, 0x9f405547dc3a74e1ull, 0x42eb67df9dfe5fd4ull, 0x5ea4677b7aacbaa2ull, 0xf65523882b55ba41ull,
        0x086e59862a218347ull, 0x39e6e389d49ee540ull, 0xfb49e956ffca0f1cull, 0x8a59c52bfa94c5c1ull, 0xd3cfc50fae5adb86ull, 0xc5476243853b8621ull,
        0x94792c8761107b4cull, 0x2a1a2c8012bf4390ull, 0x2688893c78e4c4a8ull, 0x7bdbe5c23ac4eaf4ull, 0x268a67f7bf920d2bull, 0xa365b1933d0b7cbdull,
        0xdc51a463dd27dde1ull, 0x6919949a9529a828ull, 0xce68b4ed09209f44ull, 0xca984e638270237cull, 0x7e32b90f8ef5a7e7ull, 0x561408f1212a9db5ull,
        0x4d7e6f5119a5abf9ull, 0xb5d6df8261dd9602ull, 0x36169f3ac4a1a283ull, 0x6ded727a8d39a9b8ull, 0x825c326b5b2746edull, 0x34007700d255f4fcull,
        0x4d59018071e0e13full, 0x89b295f364a8f1aeull, 0xa74b38fc4ceab2bbull
    };
    Real m = fabs<Real>(a);
    Long E = 0;
    while (m > (Real)0x1p1000) {
      m *= (Real)0x1p-1000;
      E += 1000;
    }
    const int e = std::ilogb((double)m); // the exponent of m, or one more
    m *= (Real)std::ldexp(1.0, 113 - e);
    E += e - 113;
    const U128 M = (U128)m;
    const auto bits64 = [](const Long t) -> uint64_t { // bits t .. t+63 of 2/pi; bits before the first are 0
      const Long o = t - 1;
      if (o <= -64) return 0;
      if (o < 0) return two_over_pi[0] >> (-o);
      const Long w = o / 64;
      const Long b = o % 64;
      return (b == 0 ? two_over_pi[w] : (two_over_pi[w] << b) | (two_over_pi[w+1] >> (64 - b)));
    };

    // bits of 2/pi before bit E-1 give multiples of 4; with the 320 bits W from bit E-1, |a| 2/pi = M W 2^-318
    const uint64_t Mw[2] = {(uint64_t)M, (uint64_t)(M >> 64)};
    uint64_t W[5];
    for (Integer j = 0; j < 5; j++) W[j] = bits64(E - 1 + 64 * (4 - j));
    uint64_t P[7] = {0, 0, 0, 0, 0, 0, 0};
    for (Integer i = 0; i < 2; i++) {
      U128 carry = 0;
      for (Integer j = 0; j < 5; j++) {
        const U128 t = (U128)Mw[i] * W[j] + P[i+j] + carry;
        P[i+j] = (uint64_t)t;
        carry = t >> 64;
      }
      P[i+5] = (uint64_t)carry;
    }
    q = (Integer)(P[4] >> 62); // bits 318, 319
    const U128 F = ((U128)(P[4] & ((((uint64_t)1) << 62) - 1)) << 66) | ((U128)P[3] << 2) | (P[2] >> 62); // bits 190 .. 317
    Real f = (Real)F * (Real)0x1p-128;
    if (f >= 0.5) {
      f -= 1;
      q = (q + 1) & 3;
    }
    r = f * (const_pi<Real>() / 2);
    if (a < 0) {
      r = -r;
      q = (-q) & 3;
    }
#endif
  }

  Real t = (r < 0.0 ? -r : r);
  Real sval = 0.0;
  Real cval = 1.0;
  for (int i = 0; i < N; i++) {
    while (theta[i] <= t) {
      Real sval_ = sval * cosval[i] + cval * sinval[i];
      Real cval_ = cval * cosval[i] - sval * sinval[i];
      sval = sval_;
      cval = cval_;
      t = t - theta[i];
    }
  }
  { // remaining angle t < theta[N-1], where sin(t) = t and cos(t) = 1
    const Real sval_ = sval + cval * t;
    cval = cval - sval * t;
    sval = sval_;
  }
  if (r < 0.0) sval = -sval;
  sin_a = (q == 0 ? sval : (q == 1 ? cval : (q == 2 ? -sval : -cval)));
  cos_a = (q == 0 ? cval : (q == 1 ? -sval : (q == 2 ? -cval : sval)));
}

template <class Real> static inline Real sin_generic(const Real a) {
  Real sin_a, cos_a;
  sincos_generic(a, sin_a, cos_a);
  return sin_a;
}

template <class Real> static inline Real cos_generic(const Real a) {
  Real sin_a, cos_a;
  sincos_generic(a, sin_a, cos_a);
  return cos_a;
}

template <class Real> static inline Real tan_generic(const Real a) {
  return sin(a) / cos(a);
}

template <class Real> static inline Real asin_generic(const Real a) {
  if (fabs<Real>(a) > 0.5) { // Newton below divides by cos(b), which is near 0 at a = +-1
    const Real b = const_pi<Real>()/2 - 2*asin_generic(sqrt<Real>((1-fabs<Real>(a))/2));
    return (a < 0 ? -b : b);
  }
  Real b = ::asin((double)a);
  if (!(b!=b)) { // Newton iterations for greater accuracy; b -= keeps b = -0 at a = -0, b += would not
    b -= (sin<Real>(b)-a)/cos<Real>(b);
    b -= (sin<Real>(b)-a)/cos<Real>(b);
  }
  return b;
}

template <class Real> static inline Real acos_generic(const Real a) {
  if (a > 0.5) return 2*asin_generic(sqrt<Real>((1-a)/2)); // Newton below divides by sin(b), which is near 0 at a = +-1
  if (a < -0.5) return const_pi<Real>() - 2*asin_generic(sqrt<Real>((1+a)/2));
  Real b = ::acos((double)a);
  if (!(b!=b)) { // Newton iterations for greater accuracy
    b += (cos<Real>(b)-a)/sin<Real>(b);
    b += (cos<Real>(b)-a)/sin<Real>(b);
  }
  return b;
}

template <class Real> static inline Real atan_generic(const Real a) {
  Real b = ::atan((double)a);
  if (!(b!=b)) { // Newton iterations for greater accuracy; b -= keeps b = -0 at a = -0, b += would not
    const auto cos_b0 = cos<Real>(b);
    b -= (tan<Real>(b)-a) * cos_b0 * cos_b0;
    const auto cos_b1 = cos<Real>(b);
    b -= (tan<Real>(b)-a) * cos_b1 * cos_b1;
  }
  return b;
}

template <class Real> static inline Real atan2_generic(const Real y, const Real x) {
  if (x + y > 0) {
    if (x - y > 0) return atan(y/x);
    else return atan(-x/y) + const_pi<Real>()/2;
  } else {
    if (x - y > 0) return -atan(x/y) - const_pi<Real>()/2;
    else {
      if (y >= 0) return atan(y/x) + const_pi<Real>();
      else return atan(y/x) - const_pi<Real>();
    }
  }
}

template <class Real> static inline Real fmod_generic(const Real a, const Real b) {
  if (isinf<Real>(b) && !isinf<Real>(a)) return a; // trunc(a/b) b would be 0 inf
  return a - trunc<Real>(a/b) * b;
}

template <class Real> static inline Real exp_generic(const Real a) {
  if (!(a == a)) return a;
  if (fabs<Real>(a) > (Real)0x1p14) return (a < 0.0 ? (Real)0 : (Real)INFINITY); // beyond the range of QuadReal
  // a = k ln2 + r, |r| <= ln2/2; ln2 = sum of ln2p[i], each with 51 significant bits, so k * ln2p[i] is exact
  const Real k = round<Real>(a * (Real)1.44269504088896340736);
  const Real ln2p[4] = {(Real)0x1.62e42fefa39ecp-1, (Real)0x1.9abc9e3b39800p-52, (Real)0x1.f97b57a079a18p-103, (Real)0x1.3394c5b16c508p-155};
  Real r = a;
  for (const Real& p : ln2p) r -= k * p;

  static const std::vector<Real> coeff = [] { // 1/n!; the first term left out, (ln2/2)^26/26!, is below 2^-128
    std::vector<Real> c(26);
    c[0] = 1;
    for (Integer n = 1; n < 26; n++) c[n] = c[n-1] / n;
    return c;
  }();
  Real e = coeff[25];
  for (Integer n = 24; n >= 0; n--) e = e * r + coeff[n];

  const auto pow2 = [](const Long j) { // 2^j, exact
    Real b = (j < 0 ? (Real)0.5 : (Real)2);
    Real p = 1;
    for (Long m = (j < 0 ? -j : j); m > 0; m >>= 1) {
      if (m & 1) p *= b;
      b *= b;
    }
    return p;
  };
  const Long k1 = (Long)k / 2;
  return e * pow2(k1) * pow2((Long)k - k1); // two factors, so that a subnormal result is rounded once
}

template <class Real> static inline Real log_generic(const Real a) {
  if (a == 0) return -(Real)INFINITY;
  if (!(a > 0)) return (Real)NAN;
  if (isinf<Real>(a)) return a;
  // a = 2^k m with m in [1/sqrt(2), sqrt(2)]; first scale m into the range of double
  Real m = a;
  Integer k = 0;
  while (m > (Real)0x1p1000) {
    m *= (Real)0x1p-1000;
    k += 1000;
  }
  while (m < (Real)0x1p-1000) {
    m *= (Real)0x1p1000;
    k -= 1000;
  }
  const int e = (int)std::lround(std::log2((double)m));
  m *= (Real)std::ldexp(1.0, -e);
  k += e;

  // log(m) = 2 atanh(s) with s = (m-1)/(m+1); m-1 is exact, |s| <= 0.172, so 22 terms reach 2^-117
  static const std::vector<Real> coeff = [] {
    std::vector<Real> c(22);
    for (Integer j = 0; j < 22; j++) c[j] = 1 / (Real)(2 * j + 1);
    return c;
  }();
  const Real s = (m - 1) / (m + 1);
  const Real s2 = s * s;
  Real p = 0;
  for (Integer j = 21; j >= 0; j--) p = p * s2 + coeff[j];
  const Real ln2 = (Real)0x1.62e42fefa39efp-1 + (Real)0x1.abc9e3b39803fp-56 + (Real)0x1.7b57a079a1934p-111;
  return 2 * s * p + (Real)k * ln2;
}

template <class Real> static inline Real log2_generic(const Real a) {
  static const Real recip_log2 = 1/log<Real>((Real)2);
  return log<Real>(a) * recip_log2;
}

template <class Real> static inline Real pow_generic(const Real b, const Real e) {
  if (e == 0) return 1;
  if (b == 0) return 0;
  if (b < 0) {
    Long e_ = (Long)e;
    SCTL_ASSERT(e == (Real)e_);
    return exp<Real>(log<Real>(-b) * e) * (e_ % 2 ? (Real)-1 : (Real)1.0);
  }
  return exp<Real>(log<Real>(b) * e);
}
template <class ValueType> static inline constexpr ValueType pow_integer_exp(ValueType b, Long e) {
  return (e > 0) ? ((e & 1) ? b : ValueType(1)) * pow_integer_exp(b*b, e>>1) : ValueType(1);
}
template <class Real, class ExpType> class pow_wrapper {
  public:
    static Real pow(Real b, ExpType e) {
      return (Real)std::pow(b, e);
    }
};
template <class ValueType> class pow_wrapper<ValueType,Long> {
  public:
    static constexpr ValueType pow(ValueType b, Long e) {
      return (e > 0) ? pow_integer_exp(b, e) : 1/pow_integer_exp(b, -e);
    }
};

template <Long e, class ValueType> inline constexpr ValueType pow(ValueType b) {
  return (e > 0) ? pow_integer_exp<ValueType>(b, e) : 1/pow_integer_exp<ValueType>(b, -e);
}

template <class Real> inline std::ostream& ostream_insertion_generic(std::ostream& output, const Real q_) {
  if (isnan(q_)) {
    output << "nan";
    return output;
  }
  if (isinf(q_)) {
    if (q_ > 0) output << "inf";
    else output << "-inf";
    return output;
  }

  int precision=output.precision();

  Real q = q_;
  std::string ss;
  if (q < 0.0) {
    ss += "-";
    q = -q;
  } else if (q > 0) {
    ss += " ";
  } else {
    ss += " 0";
    output << ss;
    return output;
  }

  int exp = 0;
  static const Real ONETENTH = (Real)1 / 10;
  while (q < 1.0 && abs(exp) < 10000) {
    q = q * 10;
    exp--;
  }
  while (q >= 10 && abs(exp) < 10000) {
    q = q * ONETENTH;
    exp++;
  }

  for (int i = 0; i < std::max(1,precision); i++) {
    if (i == 1) ss += ".";
    ss += ('0' + int(q));
    q = (q - int(q)) * 10;
    if (q == 0 && i > 0) break;
  }

  if (exp < 0) ss += "e";
  if (exp >= 0) ss += "e+";
  ss += std::to_string(exp);

  output << ss;
  return output;
}



#ifdef SCTL_QUAD_T
template <> inline bool isinf<QuadReal>(const QuadReal a) { return isinf_generic(a); }
template <> inline bool isnan<QuadReal>(const QuadReal a) { return isnan_generic(a); }

template <> inline QuadReal fabs<QuadReal>(const QuadReal a) { return fabs_generic(a); }

template <> inline QuadReal round<QuadReal>(const QuadReal a) { return round_generic(a); }

template <> inline QuadReal floor<QuadReal>(const QuadReal a) { return floor_generic(a); }

template <> inline QuadReal ceil<QuadReal>(const QuadReal a) { return ceil_generic(a); }

template <> inline QuadReal trunc<QuadReal>(const QuadReal x) {
  #ifdef __SIZEOF_INT128__
  return (QuadReal)(__int128)(x.val);
  #else
  return (QuadReal)(int64_t)(x.val);
  #endif
}

template <> inline QuadReal sqrt<QuadReal>(const QuadReal a) { return sqrt_generic(a); }

template <> inline QuadReal hypot<QuadReal>(const QuadReal a, const QuadReal b) { return hypot_generic(a, b); }

template <> inline QuadReal sin<QuadReal>(const QuadReal a) { return sin_generic(a); }

template <> inline QuadReal cos<QuadReal>(const QuadReal a) { return cos_generic(a); }

template <> inline QuadReal tan<QuadReal>(const QuadReal a) { return tan_generic(a); }

template <> inline QuadReal asin<QuadReal>(const QuadReal a) { return asin_generic(a); }

template <> inline QuadReal acos<QuadReal>(const QuadReal a) { return acos_generic(a); }

template <> inline QuadReal atan<QuadReal>(const QuadReal a) { return atan_generic(a); }

template <> inline QuadReal atan2<QuadReal>(const QuadReal a, const QuadReal b) { return atan2_generic(a, b); }

template <> inline QuadReal fmod<QuadReal>(const QuadReal a, const QuadReal b) { return fmod_generic(a, b); }

template <> inline QuadReal exp<QuadReal>(const QuadReal a) { return exp_generic(a); }

template <> inline QuadReal log<QuadReal>(const QuadReal a) { return log_generic(a); }

template <> inline QuadReal log2<QuadReal>(const QuadReal a) { return log2_generic(a); }

template <class ExpType> class pow_wrapper<QuadReal,ExpType> {
  public:
    static QuadReal pow(QuadReal b, ExpType e) {
      return pow_generic<QuadReal>(b, (QuadReal)e);
    }
};
template <> class pow_wrapper<QuadReal,Long> {
  public:
    static inline constexpr QuadReal pow(QuadReal b, Long e) {
      return (e > 0) ? pow_integer_exp(b, e) : 1/pow_integer_exp(b, -e);
    }
};

inline std::ostream& operator<<(std::ostream& output, const QuadReal& q) { return ostream_insertion_generic(output, q); }
inline std::istream& operator>>(std::istream& inputstream, QuadReal& x) {
  std::string str;
  inputstream >> std::ws;
  std::istream::sentry s(inputstream);
  if (s) while (inputstream.good()) {
    char c = inputstream.peek();
    if (std::isspace(c,inputstream.getloc()) || inputstream.eof()) {
      if (str.size()) {
        x = atoreal<QuadReal>(str.c_str());
        break;
      }
    }
    if (('0' <= c && c <= '9') || c == '.'  || c == '-' || c == '+'|| c == 'e' || c == 'E') {
      str += c;
      inputstream.get();
    } else {
      inputstream.setstate(std::istream::failbit);
    }
  }
  return inputstream;
}
#endif



template <class Real, class ExpType> inline Real pow(const Real b, const ExpType e) {
  return pow_wrapper<Real,ExpType>::pow(b, e);
}

} // end namespace

#endif // _SCTL_MATH_UTILS_TXX_
