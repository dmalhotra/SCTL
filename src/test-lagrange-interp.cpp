// Per-function tests for sctl/lagrange-interp.{hpp,txx}.
//
// LagrangeInterp has two static functions:
//   Interpolate(wts, src_nds, trg_nds)   : Ns x Nt interpolation weight matrix
//   Derivative(df, f, nds)               : exact polynomial derivative at nodes
//
// Interpolation is exact for polynomials up to degree Ns-1; we verify by
// sampling a known polynomial at src_nds, applying the weights to recover
// values at trg_nds, and comparing against the polynomial.

#include <cstdio>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"
#include "sctl/lagrange-interp.hpp"
#include "sctl/lagrange-interp.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Vector;
using sctl::LagrangeInterp;

// Evaluate p(x) = sum_k a[k] * x^k.
static double poly_eval(const std::vector<double>& a, double x) {
  double v = 0, p = 1;
  for (double c : a) { v += c * p; p *= x; }
  return v;
}

// Symbolic derivative coefficients of poly a: a'[k-1] = k * a[k] for k>=1.
static std::vector<double> poly_deriv(const std::vector<double>& a) {
  if (a.size() <= 1) return std::vector<double>(1, 0.0);
  std::vector<double> d(a.size() - 1);
  for (size_t k = 1; k < a.size(); ++k) d[k-1] = (double)k * a[k];
  return d;
}

int main() {
  using R = double;
  const R tol = 1e-10;

  // --- Interpolate: polynomial of degree Ns-1 is recovered exactly. ---
  std::printf("Interpolate polynomial-recovery :\n");
  {
    // Ns = 4 source nodes, sample p(x) = 1 - 2x + 3x^2 - 0.5x^3 (degree 3).
    const std::vector<double> coeffs = {1.0, -2.0, 3.0, -0.5};
    Vector<R> src({0.0, 0.25, 0.5, 0.75});
    Vector<R> trg({0.1, 0.3, 0.6, 0.9, 1.2});
    Vector<R> wts;
    LagrangeInterp<R>::Interpolate(wts, src, trg);
    CHECK(wts.Dim() == src.Dim() * trg.Dim());

    // Sample p at src_nds.
    Vector<R> f_src(src.Dim());
    for (Long i = 0; i < src.Dim(); ++i) f_src[i] = poly_eval(coeffs, src[i]);

    // Apply weights: f_trg[j] = sum_i wts[i*Nt + j] * f_src[i]
    const Long Ns = src.Dim(), Nt = trg.Dim();
    for (Long j = 0; j < Nt; ++j) {
      R f_t = 0;
      for (Long i = 0; i < Ns; ++i) f_t += wts[i * Nt + j] * f_src[i];
      CHECK(test_utils::approx_eq(f_t, poly_eval(coeffs, trg[j]), tol));
    }
  }

  // --- Interpolate: identity weights when trg == src ---
  std::printf("Interpolate identity (trg==src) :\n");
  {
    Vector<R> nds({-1.0, -0.5, 0.5, 1.0});
    Vector<R> wts;
    LagrangeInterp<R>::Interpolate(wts, nds, nds);
    // Apply to constants -> recover constants
    Vector<R> f({2.0, 2.0, 2.0, 2.0});
    const Long N = nds.Dim();
    for (Long j = 0; j < N; ++j) {
      R s = 0;
      for (Long i = 0; i < N; ++i) s += wts[i * N + j] * f[i];
      CHECK(test_utils::approx_eq(s, 2.0, tol));
    }
  }

  // --- Derivative: polynomial of degree <= Ns-1 has its derivative exact. ---
  std::printf("Derivative polynomial-exactness :\n");
  {
    const std::vector<double> coeffs = {0.5, -1.0, 2.0, 3.0};  // p(x) = 0.5 - x + 2x^2 + 3x^3
    const std::vector<double> dcoeffs = poly_deriv(coeffs);
    Vector<R> nds({-1.0, -0.5, 0.5, 1.0});
    Vector<R> f(nds.Dim());
    for (Long i = 0; i < nds.Dim(); ++i) f[i] = poly_eval(coeffs, nds[i]);
    Vector<R> df;
    LagrangeInterp<R>::Derivative(df, f, nds);
    CHECK(df.Dim() == nds.Dim());
    for (Long i = 0; i < nds.Dim(); ++i) {
      CHECK(test_utils::approx_eq(df[i], poly_eval(dcoeffs, nds[i]), 1e-9));
    }
  }

  // --- Derivative: multi-field (dof > 1) ---
  std::printf("Derivative multi-field :\n");
  {
    Vector<R> nds({0.0, 1.0, 2.0});
    // Two scalar fields: f1(x) = x   -> df1 = 1
    //                    f2(x) = x^2 -> df2 = 2x
    Vector<R> f({0.0, 1.0, 2.0,     // f1 at nds
                 0.0, 1.0, 4.0});   // f2 at nds
    Vector<R> df;
    LagrangeInterp<R>::Derivative(df, f, nds);
    CHECK(df.Dim() == 6);
    // f1 derivative ≡ 1
    for (Long i = 0; i < 3; ++i) CHECK(test_utils::approx_eq(df[i], 1.0, 1e-10));
    // f2 derivative = 2 * nds[i]
    for (Long i = 0; i < 3; ++i) CHECK(test_utils::approx_eq(df[3 + i], 2 * nds[i], 1e-10));
  }

  TEST_SUMMARY_RETURN();
}
