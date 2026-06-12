// Per-function tests for sctl/vector.{hpp,txx}.
//
// Covers every public method of sctl::Vector<T>: constructors (default, sized,
// std::vector, init-list, copy, move, ScratchBuf view), Swap, ReInit, Dim,
// begin/end, operator[], SetZero, PushBack, all elementwise vector ops
// (+, -, *, /, +=, -=, *=, /=, unary -), all scalar broadcast ops, scalar/
// vector free-function operators, Write/Read round-trip, operator<<.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/scratch_pool.hpp"
#include "sctl/scratch_pool.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Vector;

int main() {
  // --- default ctor ---
  std::printf("default ctor :\n");
  {
    Vector<double> v;
    CHECK(v.Dim() == 0);
    CHECK(v.begin() == v.end());
  }

  // --- sized ctor ---
  std::printf("sized ctor :\n");
  {
    Vector<int> v(5);
    CHECK(v.Dim() == 5);
    for (Long i = 0; i < 5; ++i) v[i] = (int)(i * 3);
    for (Long i = 0; i < 5; ++i) CHECK(v[i] == (int)(i * 3));
  }

  // --- ctor from std::vector ---
  std::printf("ctor from std::vector :\n");
  {
    std::vector<double> src = {1.5, 2.5, 3.5, 4.5};
    Vector<double> v(src);
    CHECK(v.Dim() == 4);
    for (Long i = 0; i < 4; ++i) CHECK(v[i] == src[i]);
  }

  // --- ctor from initializer_list ---
  std::printf("ctor from init-list :\n");
  {
    Vector<int> v({10, 20, 30});
    CHECK(v.Dim() == 3);
    CHECK(v[0] == 10);
    CHECK(v[1] == 20);
    CHECK(v[2] == 30);
  }

  // --- copy ctor + copy assignment ---
  std::printf("copy ctor / assignment :\n");
  {
    Vector<int> a({1, 2, 3, 4});
    Vector<int> b(a);
    Vector<int> c;
    c = a;
    CHECK(b.Dim() == 4 && c.Dim() == 4);
    for (Long i = 0; i < 4; ++i) {
      CHECK(b[i] == a[i]);
      CHECK(c[i] == a[i]);
    }
    a[0] = 99;
    CHECK(b[0] == 1);  // deep copy
    CHECK(c[0] == 1);
  }

  // --- move ctor + move assignment ---
  std::printf("move ctor / assignment :\n");
  {
    Vector<int> a({5, 6, 7});
    Vector<int> b(std::move(a));
    CHECK(b.Dim() == 3 && b[0] == 5 && b[2] == 7);
    CHECK(a.Dim() == 0);  // moved-from is empty + valid

    Vector<int> c({100, 200});
    Vector<int> d;
    d = std::move(c);
    CHECK(d.Dim() == 2 && d[0] == 100 && d[1] == 200);
    CHECK(c.Dim() == 0);
  }

  // --- ScratchBuf view ctor ---
  std::printf("ScratchBuf view ctor :\n");
  {
    sctl::ScratchBuf<int> buf(5);
    for (Long i = 0; i < 5; ++i) buf[i] = (int)(i + 1);
    Vector<int> v(buf);
    CHECK(v.Dim() == 5);
    for (Long i = 0; i < 5; ++i) CHECK(v[i] == (int)(i + 1));
    // mutate through view -> see in buf
    v[2] = 999;
    CHECK(buf[2] == 999);
  }

  // --- Swap ---
  std::printf("Swap :\n");
  {
    Vector<int> a({1, 2, 3});
    Vector<int> b({10, 20});
    a.Swap(b);
    CHECK(a.Dim() == 2 && a[0] == 10 && a[1] == 20);
    CHECK(b.Dim() == 3 && b[0] == 1  && b[2] == 3);
  }

  // --- ReInit ---
  std::printf("ReInit :\n");
  {
    Vector<double> v(3);
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0;
    v.ReInit(10);  // grow
    CHECK(v.Dim() == 10);
    v.ReInit(2);   // shrink
    CHECK(v.Dim() == 2);
  }

  // --- SetZero ---
  std::printf("SetZero :\n");
  {
    Vector<double> v({1.0, 2.0, 3.0, 4.0});
    v.SetZero();
    for (Long i = 0; i < 4; ++i) CHECK(v[i] == 0.0);
  }

  // --- PushBack ---
  std::printf("PushBack :\n");
  {
    Vector<int> v;
    v.PushBack(10);
    v.PushBack(20);
    v.PushBack(30);
    CHECK(v.Dim() == 3);
    CHECK(v[0] == 10 && v[1] == 20 && v[2] == 30);
  }

  // --- begin / end iteration ---
  std::printf("begin / end :\n");
  {
    Vector<int> v({1, 2, 3, 4, 5});
    Long sum = 0;
    for (auto it = v.begin(); it != v.end(); ++it) sum += *it;
    CHECK(sum == 15);
    const Vector<int>& cref = v;
    Long csum = 0;
    for (auto it = cref.begin(); it != cref.end(); ++it) csum += *it;
    CHECK(csum == 15);
  }

  // --- operator[] (const + non-const) ---
  std::printf("operator[] :\n");
  {
    Vector<int> v({7, 8, 9});
    v[1] = 88;
    CHECK(v[1] == 88);
    const Vector<int>& c = v;
    CHECK(c[1] == 88);
  }

  // --- vector op vector (elementwise) ---
  std::printf("elementwise vector ops :\n");
  {
    Vector<int> a({1, 2, 3, 4});
    Vector<int> b({10, 10, 10, 10});
    Vector<int> sum  = a + b;
    Vector<int> diff = b - a;
    Vector<int> prod = a * b;
    Vector<int> quot = b / a;
    CHECK(sum.Dim()  == 4 && sum[0]  == 11 && sum[3]  == 14);
    CHECK(diff.Dim() == 4 && diff[0] == 9  && diff[3] == 6);
    CHECK(prod.Dim() == 4 && prod[0] == 10 && prod[3] == 40);
    CHECK(quot.Dim() == 4 && quot[0] == 10 && quot[3] == 2);

    Vector<int> neg = -a;
    CHECK(neg[0] == -1 && neg[3] == -4);

    Vector<int> c = a;
    c += b;
    CHECK(c[0] == 11 && c[3] == 14);
    c -= b;
    CHECK(c[0] == 1  && c[3] == 4);
    c *= b;
    CHECK(c[0] == 10 && c[3] == 40);
    c /= b;
    CHECK(c[0] == 1  && c[3] == 4);
  }

  // --- vector op scalar ---
  std::printf("scalar broadcast ops :\n");
  {
    Vector<double> a({1.0, 2.0, 4.0, 8.0});
    Vector<double> p = a + 1.0;
    Vector<double> m = a - 1.0;
    Vector<double> t = a * 2.0;
    Vector<double> d = a / 2.0;
    CHECK(p[0] == 2.0 && p[3] == 9.0);
    CHECK(m[0] == 0.0 && m[3] == 7.0);
    CHECK(t[0] == 2.0 && t[3] == 16.0);
    CHECK(d[0] == 0.5 && d[3] == 4.0);

    a += 1.0; CHECK(a[0] == 2.0 && a[3] == 9.0);
    a -= 1.0; CHECK(a[0] == 1.0 && a[3] == 8.0);
    a *= 2.0; CHECK(a[0] == 2.0 && a[3] == 16.0);
    a /= 2.0; CHECK(a[0] == 1.0 && a[3] == 8.0);
  }

  // --- scalar = (broadcast assignment) ---
  std::printf("scalar broadcast assignment :\n");
  {
    Vector<int> v(5);
    v = 7;
    for (Long i = 0; i < 5; ++i) CHECK(v[i] == 7);
  }

  // --- free-function scalar OP vector ---
  std::printf("free scalar op vector :\n");
  {
    Vector<double> a({1.0, 2.0, 3.0});
    Vector<double> p = 10.0 + a;
    Vector<double> m = 10.0 - a;
    Vector<double> t =  2.0 * a;
    Vector<double> d =  6.0 / a;
    CHECK(p[0] == 11.0 && p[2] == 13.0);
    CHECK(m[0] == 9.0  && m[2] == 7.0);
    CHECK(t[0] == 2.0  && t[2] == 6.0);
    CHECK(d[0] == 6.0  && d[2] == 2.0);
  }

  // --- operator<< (stream output) ---
  std::printf("operator<< :\n");
  {
    Vector<int> v({1, 2, 3});
    std::ostringstream os;
    os << v;
    const std::string s = os.str();
    CHECK(s.find('1') != std::string::npos);
    CHECK(s.find('2') != std::string::npos);
    CHECK(s.find('3') != std::string::npos);
  }

  // --- Write / Read round-trip ---
  std::printf("Write / Read :\n");
  {
    Vector<double> v({3.14, 2.71, 1.41, 1.73});
    const char* fname = "/tmp/sctl-test-vector.bin";
    v.Write(fname);
    Vector<double> r;
    r.Read(fname);
    CHECK(r.Dim() == v.Dim());
    for (Long i = 0; i < v.Dim(); ++i) CHECK(r[i] == v[i]);
    std::remove(fname);
  }

  TEST_SUMMARY_RETURN();
}
