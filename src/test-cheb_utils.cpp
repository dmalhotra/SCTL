// Per-function tests for sctl/cheb_utils.hpp.
//
// cheb_utils provides Chebyshev approximation utilities via the
// BasisInterface<Real, ChebBasis<Real>> CRTP. Coverage of the public API:
//   - Nodes<DIM>            : DIM-dimensional Chebyshev nodes (tensor product)
//   - Approx<DIM>           : fit Chebyshev coefficients to nodal values
//   - Eval<DIM>             : evaluate the approximation at arbitrary points
//   - TruncErr<DIM>         : truncation error estimate
//
// We test round-trip identity (polynomial -> Approx -> Eval -> recover values)
// for 1D, plus a 1D polynomial-exactness test for TruncErr.
// (The 1D-only helpers Nodes1D / EvalBasis1D are private to ChebBasis; users
// access them indirectly through Nodes<1> / Eval<1>.)

#include <cstdio>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"
#include "sctl/cheb_utils.hpp"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Vector;
using R = double;
using Cheb = sctl::BasisInterface<R, sctl::ChebBasis<R>>;

int main() {
  // --- Nodes<1>(order, nodes): right count, in [0, 1], strictly increasing ---
  std::printf("Nodes<1> :\n");
  {
    Vector<R> nodes;
    Cheb::template Nodes<1>(8, nodes);
    CHECK(nodes.Dim() == 8);
    for (Long i = 0; i < nodes.Dim(); ++i) {
      CHECK(nodes[i] >= R(0) && nodes[i] <= R(1));
    }
    for (Long i = 1; i < nodes.Dim(); ++i) CHECK(nodes[i] > nodes[i-1]);
  }

  // --- Nodes<2>(order, nodes): tensor-product, length 2 * order^2 ---
  std::printf("Nodes<2> :\n");
  {
    const sctl::Integer order = 5;
    Vector<R> nodes;
    Cheb::template Nodes<2>(order, nodes);
    // Layout: nodes is [x0_0, x0_1, ..., x0_{order^2-1}, x1_0, ...]
    // 2D tensor product yields order^2 points; with 2 coords each, the
    // vector length is 2 * order^2.
    CHECK(nodes.Dim() == 2 * order * order);
  }

  // --- Approx<1> + Eval<1> : recover polynomial value at arbitrary points ---
  // Use a polynomial of degree (order-1) so Chebyshev fit is exact.
  std::printf("Approx<1> + Eval<1> on polynomial :\n");
  {
    const sctl::Integer order = 8;
    // f(x) = 2 - 3x + 5x^2 - x^3 + 0.5x^5  (degree 5 < 8)
    auto f = [](R x) {
      R x2 = x*x, x3 = x2*x, x5 = x3*x2;
      return R(2) - R(3)*x + R(5)*x2 - x3 + R(0.5)*x5;
    };

    Vector<R> nodes;
    Cheb::template Nodes<1>(order, nodes);
    Vector<R> fn_v(nodes.Dim());
    for (Long i = 0; i < nodes.Dim(); ++i) fn_v[i] = f(nodes[i]);

    Vector<R> coeff;
    Cheb::template Approx<1>(order, fn_v, coeff);

    // Evaluate at a few non-node points
    Vector<R> probe;
    {
      Vector<R> p({R(0.15), R(0.42), R(0.73), R(0.93)});
      // Eval takes ConstIterator<Vector<R>>: a 1-element sequence of Vectors per dim.
      Vector<Vector<R>> coords_per_dim(1);
      coords_per_dim[0] = p;
      Cheb::template Eval<1>(order, coeff, coords_per_dim.begin(), probe);
      CHECK(probe.Dim() == p.Dim());
      for (Long i = 0; i < p.Dim(); ++i) {
        CHECK(test_utils::approx_eq(probe[i], f(p[i]), 1e-9));
      }
    }
  }

  // --- Eval at the node points recovers the input nodal values ---
  std::printf("Approx<1>+Eval<1> identity at nodes :\n");
  {
    const sctl::Integer order = 6;
    auto g = [](R x) { return R(7) - R(2)*x + R(3)*x*x*x; };
    Vector<R> nodes;
    Cheb::template Nodes<1>(order, nodes);
    Vector<R> fn_v(nodes.Dim());
    for (Long i = 0; i < nodes.Dim(); ++i) fn_v[i] = g(nodes[i]);
    Vector<R> coeff;
    Cheb::template Approx<1>(order, fn_v, coeff);

    Vector<Vector<R>> coords_per_dim(1);
    coords_per_dim[0] = nodes;
    Vector<R> out;
    Cheb::template Eval<1>(order, coeff, coords_per_dim.begin(), out);
    CHECK(out.Dim() == nodes.Dim());
    for (Long i = 0; i < nodes.Dim(); ++i) CHECK(test_utils::approx_eq(out[i], fn_v[i], 1e-10));
  }

  // --- TruncErr<1> on a polynomial of degree < order is tiny ---
  std::printf("TruncErr<1> :\n");
  {
    const sctl::Integer order = 10;
    auto h = [](R x) { return R(1) + R(2)*x + R(3)*x*x; };  // degree 2 << 10
    Vector<R> nodes;
    Cheb::template Nodes<1>(order, nodes);
    Vector<R> fn_v(nodes.Dim());
    for (Long i = 0; i < nodes.Dim(); ++i) fn_v[i] = h(nodes[i]);
    Vector<R> coeff;
    Cheb::template Approx<1>(order, fn_v, coeff);
    R err = Cheb::template TruncErr<1>(order, coeff);
    CHECK(err < 1e-10);
  }

  TEST_SUMMARY_RETURN();
}
