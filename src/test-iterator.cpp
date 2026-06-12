// Per-function tests for sctl/iterator.{hpp,txx}.
//
// In release mode (no SCTL_MEMDEBUG) Iterator<T> is just `T*` and ConstIterator<T> is
// `const T*` (see common.hpp). In debug mode they are wrapper classes that perform
// bounds + alignment + lifetime checks. This test exercises the API surface that's
// available in both modes; the same source also compiles + runs under DEBUG=1 (where
// the wrapper class machinery is exercised by the underlying SCTL_ASSERTs).

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Iterator;
using sctl::ConstIterator;

int main() {
  // --- NullIterator ---
  std::printf("NullIterator:\n");
  {
    Iterator<double> it = sctl::NullIterator<double>();
    // In release: it == nullptr. In debug: it is a wrapper around nullptr with len=0.
    // The cross-mode invariant we can check: it equality-comparable, and converts
    // to bool false (or is comparable equal to a default-constructed Iterator).
    CHECK(it == sctl::NullIterator<double>());
    Iterator<int> it2 = sctl::NullIterator<int>();
    CHECK(it2 == sctl::NullIterator<int>());
  }

  // --- Ptr2Itr / Ptr2ConstItr ---
  std::printf("Ptr2Itr / Ptr2ConstItr:\n");
  {
    double buf[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    Iterator<double>      it  = sctl::Ptr2Itr     <double>(buf,    8);
    ConstIterator<double> cit = sctl::Ptr2ConstItr<double>(buf,    8);
    CHECK(it[0] == 10.0);
    CHECK(it[7] == 80.0);
    CHECK(cit[0] == 10.0);
    CHECK(cit[7] == 80.0);
  }

  // --- operator[], operator*, operator-> ---
  std::printf("operator[] / * / -> :\n");
  {
    struct S { int a; double b; };
    S sbuf[3] = {{1, 1.5}, {2, 2.5}, {3, 3.5}};
    Iterator<S> it = sctl::Ptr2Itr<S>(sbuf, 3);
    CHECK(it[0].a == 1);
    CHECK(it[1].a == 2);
    CHECK((*it).a == 1);
    CHECK(it->b == 1.5);
    it[2].a = 33;
    CHECK(sbuf[2].a == 33);
  }

  // --- ++ / -- (pre + post) ---
  std::printf("++ / -- pre+post:\n");
  {
    int buf[5] = {0, 1, 2, 3, 4};
    Iterator<int> it = sctl::Ptr2Itr<int>(buf, 5);
    CHECK(*it == 0);
    Iterator<int> it2 = ++it;        // pre-increment
    CHECK(*it == 1 && *it2 == 1);
    Iterator<int> it3 = it++;        // post-increment
    CHECK(*it == 2 && *it3 == 1);
    Iterator<int> it4 = --it;        // pre-decrement
    CHECK(*it == 1 && *it4 == 1);
    Iterator<int> it5 = it--;        // post-decrement
    CHECK(*it == 0 && *it5 == 1);
  }

  // --- arithmetic: + - += -= and difference ---
  std::printf("arithmetic + - += -= difference:\n");
  {
    int buf[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    Iterator<int> it = sctl::Ptr2Itr<int>(buf, 8);
    Iterator<int> jt = it + 3;
    CHECK(*jt == 3);
    Iterator<int> kt = jt - 2;
    CHECK(*kt == 1);
    Iterator<int> mt = 5 + it;       // free operator+
    CHECK(*mt == 5);
    CHECK((mt - it) == 5);           // difference
    CHECK((it - mt) == -5);
    Iterator<int> nt = it;
    nt += 4;
    CHECK(*nt == 4);
    nt -= 2;
    CHECK(*nt == 2);
  }

  // --- comparison: == != < <= > >= ---
  std::printf("comparison ops:\n");
  {
    char buf[16];
    Iterator<char> a = sctl::Ptr2Itr<char>(buf,     16);
    Iterator<char> b = a + 4;
    Iterator<char> c = a + 4;
    CHECK(a == a);
    CHECK(a != b);
    CHECK(a <  b);
    CHECK(b >  a);
    CHECK(b <= c);
    CHECK(b >= c);
    CHECK(!(a > b));
    CHECK(!(b < a));
  }

  // --- ConstIterator from Iterator (implicit) ---
  std::printf("ConstIterator from Iterator:\n");
  {
    double buf[3] = {1.1, 2.2, 3.3};
    Iterator<double>      it  = sctl::Ptr2Itr<double>(buf, 3);
    ConstIterator<double> cit = it;  // Iterator -> ConstIterator (via inheritance / implicit)
    CHECK(cit[0] == 1.1);
    CHECK(cit[2] == 3.3);
    // mutate through Iterator, read through ConstIterator
    it[1] = 99.9;
    CHECK(cit[1] == 99.9);
  }

  // --- omp_par::memcpy ---
  std::printf("omp_par::memcpy:\n");
  {
    int src[5] = {10, 20, 30, 40, 50};
    int dst[5] = {0,  0,  0,  0,  0};
    Iterator<int>      d = sctl::Ptr2Itr     <int>(dst, 5);
    ConstIterator<int> s = sctl::Ptr2ConstItr<int>(src, 5);
    sctl::omp_par::memcpy(d, s, 5);
    for (Long i = 0; i < 5; ++i) CHECK(dst[i] == src[i]);

    // num=0 is a no-op
    int dst2[3] = {1, 2, 3};
    Iterator<int> d2 = sctl::Ptr2Itr<int>(dst2, 3);
    sctl::omp_par::memcpy(d2, sctl::Ptr2ConstItr<int>(src, 5), 0);
    CHECK(dst2[0] == 1 && dst2[1] == 2 && dst2[2] == 3);

    // src == dst is a no-op (aliasing short-circuit).
    int dst3[3] = {7, 8, 9};
    Iterator<int>      d3  = sctl::Ptr2Itr<int>(dst3, 3);
    ConstIterator<int> d3c = d3;
    sctl::omp_par::memcpy(d3, d3c, 3);
    CHECK(dst3[0] == 7 && dst3[1] == 8 && dst3[2] == 9);
  }

  // --- memset ---
  std::printf("memset:\n");
  {
    char buf[8];
    Iterator<char> p = sctl::Ptr2Itr<char>(buf, 8);
    Iterator<char> r = sctl::memset(p, 0x5A, 8);
    CHECK(r == p);
    for (Long i = 0; i < 8; ++i) CHECK(buf[i] == (char)0x5A);
  }

  // --- difference_type semantics: iterator over arithmetic types ---
  std::printf("difference_type / iteration semantics:\n");
  {
    int buf[16];
    for (int i = 0; i < 16; ++i) buf[i] = i * i;
    Iterator<int> begin_it = sctl::Ptr2Itr<int>(buf,    16);
    Iterator<int> end_it   = begin_it + 16;
    Long sum = 0;
    for (Iterator<int> it = begin_it; it < end_it; ++it) sum += *it;
    Long expected = 0;
    for (int i = 0; i < 16; ++i) expected += i * i;
    CHECK(sum == expected);
    CHECK((end_it - begin_it) == 16);
  }

  // --- alignment: Ptr2Itr accepts aligned pointers (smoke test) ---
  std::printf("alignment smoke:\n");
  {
    alignas(64) double aligned_buf[8] = {0};
    Iterator<double> it = sctl::Ptr2Itr<double>(aligned_buf, 8);
    CHECK(((uintptr_t)&it[0] & (alignof(double) - 1)) == 0);
  }

  TEST_SUMMARY_RETURN();
}
