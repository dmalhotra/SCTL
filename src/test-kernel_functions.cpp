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

  // --- Laplace3D_Fxd2U : Hessian of single layer ---
  std::printf("Laplace3D_Fxd2U :\n");
  {
    sctl::Laplace3D_Fxd2U K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 1);
    CHECK(K.TrgDim()    == 9);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("Laplace3D-Fxd2U"));

    // d2/dxi dxj of 1/(4 pi r) at r = (2,0,0): diag(2, -1, -1) / (4 pi r^3)
    Vector<R> Xs({0,0,0}), Xt({2,0,0}), Xn, v_s({1.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    const R s = 1 / (4 * pi * 8.0);
    CHECK(test_utils::approx_eq(v_t[0], 2*s, tol));
    CHECK(test_utils::approx_eq(v_t[4], -s, tol));
    CHECK(test_utils::approx_eq(v_t[8], -s, tol));
    for (Long k : {1,2,3,5,6,7}) CHECK(std::fabs(v_t[k]) < 1e-12);
  }

  // --- Laplace3D_DxdU : gradient of the double layer ---
  std::printf("Laplace3D_DxdU :\n");
  {
    sctl::Laplace3D_DxdU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 1);
    CHECK(K.TrgDim()    == 3);
    CHECK(K.NormalDim() == 3);
    CHECK(K.Name() == std::string("Laplace3D-DxdU"));

    // n along x, r = (2,0,0): grad of (r.n)/(4 pi r^3) is diag-like; x-component
    // is -2/(4 pi r^3), transverse components vanish.
    Vector<R> Xs({0,0,0}), Xt({2,0,0}), Xn({1,0,0}), v_s({1.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(test_utils::approx_eq(v_t[0], -2.0 / (4 * pi * 8.0), tol));
    CHECK(std::fabs(v_t[1]) < 1e-12);
    CHECK(std::fabs(v_t[2]) < 1e-12);
  }

  // --- BiotSavart3D_FxU : K = f x r / (4 pi r^3) ---
  std::printf("BiotSavart3D_FxU :\n");
  {
    sctl::BiotSavart3D_FxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 3);
    CHECK(K.TrgDim()    == 3);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("BiotSavart3D-FxU"));

    // f = z_hat, r = (2,0,0)  ->  f x r = (0, 2, 0)
    Vector<R> Xs({0,0,0}), Xt({2,0,0}), Xn, v_s({0.0,0.0,1.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(std::fabs(v_t[0]) < 1e-12);
    CHECK(test_utils::approx_eq(v_t[1], 2.0 / (4 * pi * 8.0), tol));
    CHECK(std::fabs(v_t[2]) < 1e-12);
    // antisymmetry: f parallel to r gives zero
    Vector<R> v_s2({1.0,0.0,0.0}), v_t2;
    K.template Eval<R, false>(v_t2, Xt, Xs, Xn, v_s2);
    for (Long k = 0; k < 3; k++) CHECK(std::fabs(v_t2[k]) < 1e-12);
  }

  // --- BiotSavart3D_FxdU : gradient of Biot-Savart ---
  std::printf("BiotSavart3D_FxdU :\n");
  {
    sctl::BiotSavart3D_FxdU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 3);
    CHECK(K.TrgDim()    == 9);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("BiotSavart3D-FxdU"));

    // compare against a central finite difference of BiotSavart3D_FxU
    sctl::BiotSavart3D_FxU K0;
    Vector<R> Xs({0,0,0}), Xn, v_s({0.3,-0.7,1.1});
    const R x0[3] = {1.3, 0.7, -0.9}, h = 1e-5;
    Vector<R> dU;
    K.template Eval<R, false>(dU, Vector<R>({x0[0],x0[1],x0[2]}), Xs, Xn, v_s);
    for (Long j = 0; j < 3; j++) { // d/dx_j
      Vector<R> xp({x0[0],x0[1],x0[2]}), xm({x0[0],x0[1],x0[2]}), up, um;
      xp[j] += h; xm[j] -= h;
      K0.template Eval<R, false>(up, xp, Xs, Xn, v_s);
      K0.template Eval<R, false>(um, xm, Xs, Xn, v_s);
      for (Long i = 0; i < 3; i++) {
        CHECK(test_utils::approx_eq(dU[i*3+j], (up[i]-um[i])/(2*h), (R)1e-6));
      }
    }
  }

  // --- Helmholtz3D_FxU : K = exp(i mu r) / (4 pi r), complex as (re,im) pairs ---
  std::printf("Helmholtz3D_FxU :\n");
  {
    sctl::Helmholtz3D_FxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 2);
    CHECK(K.TrgDim()    == 2);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("Helmholtz3D-FxU"));

    R mu = 1.7;
    K.SetCtxPtr(&mu);
    const R r = 2.0;
    Vector<R> Xs({0,0,0}), Xt({r,0,0}), Xn, v_s({1.0, 0.0}); // unit real density
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(test_utils::approx_eq(v_t[0], std::cos(mu*r) / (4 * pi * r), tol));
    CHECK(test_utils::approx_eq(v_t[1], std::sin(mu*r) / (4 * pi * r), tol));

    // mu = 0 must reduce to the Laplace single layer
    R mu0 = 0;
    K.SetCtxPtr(&mu0);
    Vector<R> v_t0;
    K.template Eval<R, false>(v_t0, Xt, Xs, Xn, v_s);
    CHECK(test_utils::approx_eq(v_t0[0], 1.0 / (4 * pi * r), tol));
    CHECK(std::fabs(v_t0[1]) < 1e-12);
  }

  // --- Helmholtz3D_DxU / Helmholtz3D_FxdU : dims + consistency with FxdU ---
  std::printf("Helmholtz3D_DxU :\n");
  {
    sctl::Helmholtz3D_DxU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 2);
    CHECK(K.TrgDim()    == 2);
    CHECK(K.NormalDim() == 3);
    CHECK(K.Name() == std::string("Helmholtz3D-DxU"));

    // DxU is the normal derivative of FxU, i.e. n . FxdU
    R mu = 1.7;
    sctl::Helmholtz3D_FxdU Kg;
    K.SetCtxPtr(&mu);
    Kg.SetCtxPtr(&mu);
    Vector<R> Xs({0,0,0}), Xt({1.3,0.7,-0.9}), Xn({0,1,0}), v_s({0.4,-1.2});
    Vector<R> vd, vg;
    K.template Eval<R, false>(vd, Xt, Xs, Xn, v_s);
    Kg.template Eval<R, false>(vg, Xt, Xs, Vector<R>(), v_s);
    for (Long c = 0; c < 2; c++) { // n . grad, per complex component
      R acc = 0;
      for (Long i = 0; i < 3; i++) acc += Xn[i] * vg[i*2+c];
      CHECK(test_utils::approx_eq(vd[c], acc, tol));
    }
  }

  std::printf("Helmholtz3D_FxdU :\n");
  {
    sctl::Helmholtz3D_FxdU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 2);
    CHECK(K.TrgDim()    == 6);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("Helmholtz3D-FxdU"));

    // compare against a central finite difference of Helmholtz3D_FxU
    R mu = 1.7;
    sctl::Helmholtz3D_FxU K0;
    K.SetCtxPtr(&mu);
    K0.SetCtxPtr(&mu);
    Vector<R> Xs({0,0,0}), Xn, v_s({0.4,-1.2});
    const R x0[3] = {1.3, 0.7, -0.9}, h = 1e-5;
    Vector<R> dU;
    K.template Eval<R, false>(dU, Vector<R>({x0[0],x0[1],x0[2]}), Xs, Xn, v_s);
    for (Long j = 0; j < 3; j++) {
      Vector<R> xp({x0[0],x0[1],x0[2]}), xm({x0[0],x0[1],x0[2]}), up, um;
      xp[j] += h; xm[j] -= h;
      K0.template Eval<R, false>(up, xp, Xs, Xn, v_s);
      K0.template Eval<R, false>(um, xm, Xs, Xn, v_s);
      for (Long c = 0; c < 2; c++) {
        CHECK(test_utils::approx_eq(dU[j*2+c], (up[c]-um[c])/(2*h), (R)1e-6));
      }
    }
  }

  // --- HelmholtzDiff3D_FxdU : Helmholtz minus Laplace, accurate as mu r -> 0 ---
  std::printf("HelmholtzDiff3D_FxdU :\n");
  {
    sctl::HelmholtzDiff3D_FxdU K;
    CHECK(K.CoordDim()  == 3);
    CHECK(K.SrcDim()    == 2);
    CHECK(K.TrgDim()    == 6);
    CHECK(K.NormalDim() == 0);
    CHECK(K.Name() == std::string("HelmholtzDiff3D-FxdU"));

    // real part must equal Helmholtz3D_FxdU's real part minus Laplace3D_FxdU
    R mu = 1.7;
    sctl::Helmholtz3D_FxdU Kh;
    sctl::Laplace3D_FxdU Kl;
    K.SetCtxPtr(&mu);
    Kh.SetCtxPtr(&mu);
    Vector<R> Xs({0,0,0}), Xt({1.3,0.7,-0.9}), Xn, v_s({1.0,0.0}), v_l({1.0});
    Vector<R> vdiff, vh, vl;
    K.template Eval<R, false>(vdiff, Xt, Xs, Xn, v_s);
    Kh.template Eval<R, false>(vh, Xt, Xs, Xn, v_s);
    Kl.template Eval<R, false>(vl, Xt, Xs, Xn, v_l);
    for (Long i = 0; i < 3; i++) {
      CHECK(test_utils::approx_eq(vdiff[i*2+0], vh[i*2+0] - vl[i], tol));
      CHECK(test_utils::approx_eq(vdiff[i*2+1], vh[i*2+1], tol));
    }

    // the cancellation is resolved without catastrophic loss as mu r -> 0
    R mu_small = 1e-7;
    K.SetCtxPtr(&mu_small);
    Vector<R> vsmall;
    K.template Eval<R, false>(vsmall, Xt, Xs, Xn, v_s);
    for (Long i = 0; i < 3; i++) CHECK(std::fabs(vsmall[i*2+0]) < 1e-14);
  }

  // --- optional fused apply must agree with uKerMatrix ---
  std::printf("FUSED_APPLY consistency :\n");
  {
    using VecType = sctl::Vec<R, 1>;
    const R mu = 1.7;
    const VecType r[3] = {VecType((R)1.3), VecType((R)0.7), VecType((R)-0.9)};
    const VecType n[3] = {VecType((R)0.0), VecType((R)1.0), VecType((R)0.0)};

    // Laplace3D_FxdU: matrix apply vs fused apply
    {
      VecType u[1][3], v_fused[3];
      sctl::kernel_impl::Laplace3D_FxdU::uKerMatrix<15>(u, r, nullptr);
      for (Long k = 0; k < 3; k++) v_fused[k] = VecType::Zero();
      const VecType f[1] = {VecType((R)0.7)};
      sctl::kernel_impl::Laplace3D_FxdU::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k = 0; k < 3; k++) CHECK(test_utils::approx_eq(v_fused[k][0], (u[0][k]*f[0])[0], tol));
    }
    // BiotSavart3D_FxU
    {
      VecType u[3][3], v_fused[3];
      sctl::kernel_impl::BiotSavart3D_FxU::uKerMatrix<15>(u, r, nullptr);
      for (Long k = 0; k < 3; k++) v_fused[k] = VecType::Zero();
      const VecType f[3] = {VecType((R)0.3), VecType((R)-0.7), VecType((R)1.1)};
      sctl::kernel_impl::BiotSavart3D_FxU::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k1 = 0; k1 < 3; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 3; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    // Stokes3D_DxU
    {
      VecType u[3][3], v_fused[3];
      sctl::kernel_impl::Stokes3D_DxU::uKerMatrix<15>(u, r, n, nullptr);
      for (Long k = 0; k < 3; k++) v_fused[k] = VecType::Zero();
      const VecType f[3] = {VecType((R)0.3), VecType((R)-0.7), VecType((R)1.1)};
      sctl::kernel_impl::Stokes3D_DxU::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k1 = 0; k1 < 3; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 3; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }

    // Stokes3D_FxU / FxT / FSxU / FxUP
    {
      VecType u[3][3], v_fused[3];
      sctl::kernel_impl::Stokes3D_FxU::uKerMatrix<15>(u, r, nullptr);
      for (Long k = 0; k < 3; k++) v_fused[k] = VecType::Zero();
      const VecType f[3] = {VecType((R)0.3), VecType((R)-0.7), VecType((R)1.1)};
      sctl::kernel_impl::Stokes3D_FxU::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k1 = 0; k1 < 3; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 3; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    {
      VecType u[3][9], v_fused[9];
      sctl::kernel_impl::Stokes3D_FxT::uKerMatrix<15>(u, r, nullptr);
      for (Long k = 0; k < 9; k++) v_fused[k] = VecType::Zero();
      const VecType f[3] = {VecType((R)0.3), VecType((R)-0.7), VecType((R)1.1)};
      sctl::kernel_impl::Stokes3D_FxT::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k1 = 0; k1 < 9; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 3; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    {
      VecType u[4][3], v_fused[3];
      sctl::kernel_impl::Stokes3D_FSxU::uKerMatrix<15>(u, r, nullptr);
      for (Long k = 0; k < 3; k++) v_fused[k] = VecType::Zero();
      const VecType f[4] = {VecType((R)0.3), VecType((R)-0.7), VecType((R)1.1), VecType((R)0.5)};
      sctl::kernel_impl::Stokes3D_FSxU::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k1 = 0; k1 < 3; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 4; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    {
      VecType u[3][4], v_fused[4];
      sctl::kernel_impl::Stokes3D_FxUP::uKerMatrix<15>(u, r, nullptr);
      for (Long k = 0; k < 4; k++) v_fused[k] = VecType::Zero();
      const VecType f[3] = {VecType((R)0.3), VecType((R)-0.7), VecType((R)1.1)};
      sctl::kernel_impl::Stokes3D_FxUP::uKerApply<15,1>(v_fused, r, n, f, nullptr);
      for (Long k1 = 0; k1 < 4; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 3; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    // Helmholtz3D_FxU / DxU / FxdU
    {
      VecType u[2][2], v_fused[2];
      sctl::kernel_impl::Helmholtz3D_FxU::uKerMatrix<15>(u, r, &mu);
      for (Long k = 0; k < 2; k++) v_fused[k] = VecType::Zero();
      const VecType f[2] = {VecType((R)0.4), VecType((R)-1.2)};
      sctl::kernel_impl::Helmholtz3D_FxU::uKerApply<15,1>(v_fused, r, n, f, &mu);
      for (Long k1 = 0; k1 < 2; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 2; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    {
      VecType u[2][2], v_fused[2];
      sctl::kernel_impl::Helmholtz3D_DxU::uKerMatrix<15>(u, r, n, &mu);
      for (Long k = 0; k < 2; k++) v_fused[k] = VecType::Zero();
      const VecType f[2] = {VecType((R)0.4), VecType((R)-1.2)};
      sctl::kernel_impl::Helmholtz3D_DxU::uKerApply<15,1>(v_fused, r, n, f, &mu);
      for (Long k1 = 0; k1 < 2; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 2; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
    {
      VecType u[2][6], v_fused[6];
      sctl::kernel_impl::Helmholtz3D_FxdU::uKerMatrix<15>(u, r, &mu);
      for (Long k = 0; k < 6; k++) v_fused[k] = VecType::Zero();
      const VecType f[2] = {VecType((R)0.4), VecType((R)-1.2)};
      sctl::kernel_impl::Helmholtz3D_FxdU::uKerApply<15,1>(v_fused, r, n, f, &mu);
      for (Long k1 = 0; k1 < 6; k1++) {
        R acc = 0;
        for (Long k0 = 0; k0 < 2; k0++) acc += (u[k0][k1]*f[k0])[0];
        CHECK(test_utils::approx_eq(v_fused[k1][0], acc, tol));
      }
    }
  }


  // --- Eval takes the fused path for these; it must still agree with KernelMatrix ---
  std::printf("Eval vs KernelMatrix (fused kernels) :\n");
  {
    auto check = [&](auto K, const char* nm, Long K0, Long K1, Long ND, const void* ctx) {
      if (ctx) K.SetCtxPtr((void*)ctx);
      const Long Ns = 7, Nt = 5;
      Vector<R> Xs(Ns*3), Xn(Ns*ND), Vs(Ns*K0), Xt(Nt*3);
      for (Long i = 0; i < Xs.Dim(); i++) Xs[i] = 0.1*(i+1) - 0.35;
      for (Long i = 0; i < Vs.Dim(); i++) Vs[i] = 0.3*(i%5) - 0.4;
      for (Long i = 0; i < Xt.Dim(); i++) Xt[i] = 1.7 + 0.2*i;
      for (Long i = 0; i < Ns; i++) { // unit normals
        if (!ND) break;
        Xn[i*3+0] = 0; Xn[i*3+1] = 1; Xn[i*3+2] = 0;
      }
      Vector<R> Vt;
      K.template Eval<R,false>(Vt, Xt, Xs, Xn, Vs);
      Matrix<R> M;
      K.template KernelMatrix<R,false>(M, Xt, Xs, Xn);
      for (Long t = 0; t < Nt; t++) {
        for (Long k1 = 0; k1 < K1; k1++) {
          R acc = 0;
          for (Long is = 0; is < Ns; is++)
            for (Long k0 = 0; k0 < K0; k0++) acc += M[is*K0+k0][t*K1+k1] * Vs[is*K0+k0];
          CHECK(test_utils::approx_eq(Vt[t*K1+k1], acc, (R)1e-10));
        }
      }
      SCTL_UNUSED(nm);
    };
    const R mu = 1.7;
    check(sctl::Laplace3D_FxdU(),   "Laplace3D_FxdU",   1,3,0, nullptr);
    check(sctl::Stokes3D_FxU(),     "Stokes3D_FxU",     3,3,0, nullptr);
    check(sctl::Stokes3D_DxU(),     "Stokes3D_DxU",     3,3,3, nullptr);
    check(sctl::Stokes3D_FxT(),     "Stokes3D_FxT",     3,9,0, nullptr);
    check(sctl::Stokes3D_FSxU(),    "Stokes3D_FSxU",    4,3,0, nullptr);
    check(sctl::Stokes3D_FxUP(),    "Stokes3D_FxUP",    3,4,0, nullptr);
    check(sctl::BiotSavart3D_FxU(), "BiotSavart3D_FxU", 3,3,0, nullptr);
    check(sctl::Helmholtz3D_FxU(),  "Helmholtz3D_FxU",  2,2,0, &mu);
    check(sctl::Helmholtz3D_DxU(),  "Helmholtz3D_DxU",  2,2,3, &mu);
    check(sctl::Helmholtz3D_FxdU(), "Helmholtz3D_FxdU", 2,6,0, &mu);
  }

  TEST_SUMMARY_RETURN();
}
