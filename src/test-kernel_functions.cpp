// Per-function tests for sctl/kernel_functions.hpp.
//
// kernel_functions.hpp publishes a set of GenericKernel<...> aliases for the
// 3D Laplace and Stokes single/double-layer kernels. This test verifies each
// kernel produces the analytically expected value at a known probe point,
// confirms its CoordDim / SrcDim / TrgDim / NormalDim geometry, and confirms
// the published Name() string.

#include <cmath>
#include <cstdio>
#include <string>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"
#include "sctl/generic-kernel.hpp"
#include "sctl/generic-kernel.txx"
#include "sctl/kernel_functions.hpp"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Vector;
using sctl::Matrix;

int main() {
  using R = double;
  const R pi = 3.14159265358979323846;
  const R tol = 1e-9;

  // --- Laplace3D_FxU : K = 1/(4 pi r) ---
  std::printf("Laplace3D_FxU :\n");
  {
    sctl::Laplace3D_FxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 1);
    CHECK(K.TrgDim()    == 1);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("Laplace3D-FxU"));

    Vector<R> Xs({0,0,0}), Xt({2,0,0}), Xn, v_s({3.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(test_utils::approx_eq(v_t[0], 3.0 / (4 * pi * 2.0), tol));
  }

  // --- Laplace3D_DxU : double-layer; K = r·n / (4 pi r^3) ---
  std::printf("Laplace3D_DxU :\n");
  {
    sctl::Laplace3D_DxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 1);
    CHECK(K.TrgDim()    == 1);
    CHECK(K.NormalDim() == 3);
    CHECK(K.Name() == std::string("Laplace3D-DxU"));

    // Source at origin with outward normal +x. Target at (2,0,0).
    // r = t - s = (2,0,0); r.n = 2; r = 2; → K = 2/(4 pi * 8) = 1/(16 pi).
    Vector<R> Xs({0,0,0}), Xt({2,0,0}), Xn({1,0,0}), v_s({1.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(test_utils::approx_eq(v_t[0], 1.0 / (16 * pi), tol));
  }

  // --- Laplace3D_FxdU : gradient of single-layer potential ---
  std::printf("Laplace3D_FxdU :\n");
  {
    sctl::Laplace3D_FxdU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 1);
    CHECK(K.TrgDim()    == 3);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("Laplace3D-FxdU"));

    // ∇ (1/r) at target = -r̂ / r^2 in the convention -∇G where G = 1/(4πr).
    // Just verify the output magnitude matches the expected vector length.
    Vector<R> Xs({0,0,0}), Xt({3,0,0}), Xn, v_s({1.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(v_t.Dim() == 3);
    const R mag = std::sqrt(v_t[0]*v_t[0] + v_t[1]*v_t[1] + v_t[2]*v_t[2]);
    const R expected_mag = 1.0 / (4 * pi * 3.0 * 3.0);  // 1/(4π r^2)
    CHECK(test_utils::approx_eq(mag, expected_mag, tol));
    // along-x direction (only x-component nonzero by symmetry)
    CHECK(std::fabs(v_t[1]) < 1e-12);
    CHECK(std::fabs(v_t[2]) < 1e-12);
  }

  // --- Stokes3D_FxU : Stokes single-layer; velocity from point force ---
  std::printf("Stokes3D_FxU :\n");
  {
    sctl::Stokes3D_FxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 3);
    CHECK(K.TrgDim()    == 3);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("Stokes3D-FxU"));

    // Stokeslet G(r) = 1/(8πμ) (I/r + r⊗r/r^3), with μ = 1.
    // Source at origin with force f = (1,0,0). Target at (r,0,0).
    // u_x = f_x/(8π) * (1/r + r^2/r^3) = f_x/(8πr) * 2 = 1/(4πr).
    Vector<R> Xs({0,0,0}), Xn;
    Vector<R> Xt({2.0, 0.0, 0.0});
    Vector<R> v_s({1.0, 0.0, 0.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(v_t.Dim() == 3);
    CHECK(test_utils::approx_eq(v_t[0], 1.0 / (4 * pi * 2.0), tol));
    CHECK(std::fabs(v_t[1]) < 1e-12);
    CHECK(std::fabs(v_t[2]) < 1e-12);
  }

  // --- Stokes3D_DxU : double-layer Stokes; dimensions only ---
  std::printf("Stokes3D_DxU :\n");
  {
    sctl::Stokes3D_DxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 3);
    CHECK(K.TrgDim()    == 3);
    CHECK(K.NormalDim() == 3);
    CHECK(K.Name() == std::string("Stokes3D-DxU"));
    // Smoke: call Eval with a normal and ensure no crash.
    Vector<R> Xs({0,0,0}), Xt({3,0,0}), Xn({0,1,0}), v_s({1.0,0.0,0.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(v_t.Dim() == 3);
  }

  TEST_SUMMARY_RETURN();
}
