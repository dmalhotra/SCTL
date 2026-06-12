// Per-function tests for sctl/static-array.{hpp,txx}.
//
// In release mode (no SCTL_MEMDEBUG) StaticArray<T,N> is aliased to T[N] in
// common.hpp. In SCTL_MEMDEBUG mode it is a wrapper class with bounds-checked
// iterators and Iterator-compatible ops. Tests below exercise the surface
// available in both modes and compile/run under either build.

#include <cstdio>
#include <cstdint>

#include "sctl/common.hpp"
#include "sctl/static-array.hpp"
#include "sctl/static-array.txx"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::StaticArray;
using sctl::Iterator;
using sctl::ConstIterator;

int main() {
  // --- default ctor + operator[] ---
  std::printf("default ctor / operator[] :\n");
  {
    StaticArray<int, 4> a;
    a[0] = 10; a[1] = 20; a[2] = 30; a[3] = 40;
    CHECK(a[0] == 10);
    CHECK(a[1] == 20);
    CHECK(a[2] == 30);
    CHECK(a[3] == 40);
  }

  // --- const operator[] ---
  std::printf("const operator[] :\n");
  {
    StaticArray<int, 3> a;
    a[0] = 1; a[1] = 2; a[2] = 3;
    const StaticArray<int, 3>& cref = a;
    CHECK(cref[0] == 1);
    CHECK(cref[1] == 2);
    CHECK(cref[2] == 3);
  }

#ifdef SCTL_MEMDEBUG
  // --- copy / assign (class form only; T[N] in release isn't copyable) ---
  std::printf("copy / assign :\n");
  {
    StaticArray<int, 4> a;
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
    StaticArray<int, 4> b = a;
    StaticArray<int, 4> c;
    c = a;
    for (Long i = 0; i < 4; ++i) {
      CHECK(b[i] == a[i]);
      CHECK(c[i] == a[i]);
    }
    // mutating a doesn't affect b, c (independent copies)
    a[0] = 99;
    CHECK(b[0] == 1);
    CHECK(c[0] == 1);
  }
#endif

  // --- Iterator/pointer interop ---
  std::printf("Iterator / pointer interop :\n");
  {
    StaticArray<double, 5> a;
    for (Long i = 0; i < 5; ++i) a[i] = (double)i * 2.0;
    // implicit-convert to iterator and walk
    Iterator<double> it = a;             // StaticArray -> Iterator (release: array decay)
    CHECK(it[0] == 0.0);
    CHECK(it[4] == 8.0);
    // pointer arithmetic on the converted iterator
    Iterator<double> jt = it + 2;
    CHECK(*jt == 4.0);
    // mutate through iterator, read through array
    it[3] = 100.0;
    CHECK(a[3] == 100.0);
  }

  // --- iteration via begin-style pointer arithmetic ---
  std::printf("iteration :\n");
  {
    StaticArray<int, 8> a;
    for (Long i = 0; i < 8; ++i) a[i] = (int)(i * i);
    Long sum = 0;
    Iterator<int> b = a;
    for (Iterator<int> it = b; it < b + 8; ++it) sum += *it;
    Long expected = 0;
    for (Long i = 0; i < 8; ++i) expected += i * i;
    CHECK(sum == expected);
  }

  // --- initializer-list ctor (StaticArray class form, SCTL_MEMDEBUG path) ---
  // In release mode this constructor doesn't exist (StaticArray is just T[N]),
  // but plain brace-init of the array form works the same way.
  std::printf("brace-init :\n");
  {
#ifdef SCTL_MEMDEBUG
    sctl::StaticArray<int, 3> a({7, 8, 9});
#else
    sctl::StaticArray<int, 3> a = {7, 8, 9};
#endif
    CHECK(a[0] == 7);
    CHECK(a[1] == 8);
    CHECK(a[2] == 9);
  }

  // --- nested types ---
  std::printf("nested struct :\n");
  {
    struct Pt { int x, y; };
    StaticArray<Pt, 2> a;
    a[0].x = 1; a[0].y = 2;
    a[1].x = 3; a[1].y = 4;
    CHECK(a[0].x == 1 && a[0].y == 2);
    CHECK(a[1].x == 3 && a[1].y == 4);
  }

  TEST_SUMMARY_RETURN();
}
