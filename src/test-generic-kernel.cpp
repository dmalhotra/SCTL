// Per-function tests for sctl/generic-kernel.{hpp,txx}.
//
// GenericKernel<uKernel> is the CRTP wrapper class that turns a low-level uKer
// (uKerMatrix + uKerScaleFactor) into a high-level (Eval, KernelMatrix) API.
// We exercise it through the Laplace 3D single-layer kernel
// (sctl::Laplace3D_FxU) which has a clean analytical reference 1/(4π r).
//
// Coverage:
//   - GenericKernel::KernelMatrix    : M_ij = K(t_i, s_j) for a 1-source/1-target setup
//   - GenericKernel::Eval            : v_t = sum_s K(t, s) * v_s
//   - GenericKernel::CoordDim/SrcDim/TrgDim/NormalDim                 (geometry accessors)
//   - GenericKernel::ScaleFactor / Name                                (kernel metadata)

#include <cstdio>

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
  using Kernel = sctl::Laplace3D_FxU;
  Kernel K;

  // --- accessors ---
  std::printf("CoordDim / SrcDim / TrgDim / NormalDim :\n");
  {
    CHECK(K.CoordDim()  == 3);   // 3D
    CHECK(K.SrcDim()    == 1);   // scalar source density
    CHECK(K.TrgDim()    == 1);   // scalar potential
    CHECK(K.NormalDim() == 0);   // single-layer (no normal)
  }

  // --- Name + uKerScaleFactor (the underlying kernel's scale factor) ---
  std::printf("Name / uKerScaleFactor :\n");
  {
    CHECK(K.Name() == std::string("Laplace3D-FxU"));
    const R sf = sctl::kernel_impl::Laplace3D_FxU::template uKerScaleFactor<R>();
    CHECK(test_utils::approx_eq(sf, 1.0 / (4.0 * 3.14159265358979323846), 1e-12));
  }

  // --- KernelMatrix : single source at origin, single target at (1, 0, 0) ---
  // K(t, s) = 1/(4π |t - s|). |t - s| = 1, so the matrix entry is 1/(4π).
  std::printf("KernelMatrix (1 src x 1 trg) :\n");
  {
    Vector<R> Xt({1.0, 0.0, 0.0});  // 1 target
    Vector<R> Xs({0.0, 0.0, 0.0});  // 1 source
    Vector<R> Xn;                    // (no normals for FxU)
    Matrix<R> M;
    K.template KernelMatrix<R, false>(M, Xt, Xs, Xn);
    CHECK(M.Dim(0) == 1 && M.Dim(1) == 1);
    CHECK(test_utils::approx_eq(M(0, 0), 1.0 / (4.0 * 3.14159265358979323846), 1e-9));
  }

  // --- Eval : single source at origin, multiple targets, density = 1 ---
  std::printf("Eval (single source, varied targets) :\n");
  {
    Vector<R> Xs({0.0, 0.0, 0.0});
    Vector<R> v_s({1.0});       // density 1.0
    Vector<R> Xn;               // (no normals)
    // 4 targets at known distances 1, 2, 3, 5 from origin along x-axis.
    Vector<R> Xt({1.0, 0.0, 0.0,
                  2.0, 0.0, 0.0,
                  3.0, 0.0, 0.0,
                  5.0, 0.0, 0.0});
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(v_t.Dim() == 4);
    const R sf = 1.0 / (4.0 * 3.14159265358979323846);
    CHECK(test_utils::approx_eq(v_t[0], sf / 1.0, 1e-9));
    CHECK(test_utils::approx_eq(v_t[1], sf / 2.0, 1e-9));
    CHECK(test_utils::approx_eq(v_t[2], sf / 3.0, 1e-9));
    CHECK(test_utils::approx_eq(v_t[3], sf / 5.0, 1e-9));
  }

  // --- Eval : superposition (2 sources sum linearly) ---
  std::printf("Eval (2 sources, linearity) :\n");
  {
    Vector<R> Xs({0.0, 0.0, 0.0,
                  4.0, 0.0, 0.0});  // 2 sources at (0,0,0) and (4,0,0)
    Vector<R> v_s({1.0, 1.0});
    Vector<R> Xn;
    Vector<R> Xt({2.0, 0.0, 0.0});  // midpoint: r1=2, r2=2
    Vector<R> v_t;
    K.template Eval<R, false>(v_t, Xt, Xs, Xn, v_s);
    CHECK(v_t.Dim() == 1);
    const R sf = 1.0 / (4.0 * 3.14159265358979323846);
    const R expected = sf / 2.0 + sf / 2.0;  // two equal contributions
    CHECK(test_utils::approx_eq(v_t[0], expected, 1e-9));
  }

  TEST_SUMMARY_RETURN();
}
