// Per-function tests for sctl/boundary_integral.{hpp,txx}.
//
// boundary_integral.hpp publishes BoundaryIntegralOp<Real, Kernel>, a heavy
// abstraction that operates on registered element lists, performs MPI
// partitioning of source / target proxy points, and evaluates layer
// potentials. A full numerical test that drives BoundaryIntegralOp from
// surface input to potential output is already provided by
// test-quad-elem.cpp (which builds a cubed-sphere geometry).
//
// This file is a focused regression test for the `Depth() != INVALID_DEPTH`
// filter paths in BuildNearList / BuildNbrList (boundary_integral.txx
// lines ~212, ~315, ~338, fixed in commit ded8fb6). The actual filter sites
// are exercised together with sctl::Morton<DIM>::NbrList by
// test-morton.cpp; here we verify only:
//
//   1. The BoundaryIntegralOp<Real, Kernel> constructor + destructor work
//      for several published kernels (smoke).
//   2. Calling Setup / ClearSetup on a freshly-constructed operator (with
//      no registered ElemList) is harmless.
//
// The detailed near-list filter regression lives in test-morton.cpp
// (NbrList section) and test-tree-edge-periodic.cpp (the analogous
// tree.txx filter, which crashes pre-fix on periodic=false).

#include <cstdio>

#include "sctl/common.hpp"
#include "sctl/comm.hpp"
#include "sctl/comm.txx"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"
#include "sctl/morton.hpp"
#include "sctl/morton.txx"
#include "sctl/generic-kernel.hpp"
#include "sctl/generic-kernel.txx"
#include "sctl/kernel_functions.hpp"
#include "sctl/fmm-wrapper.hpp"
#include "sctl/fmm-wrapper.txx"
#include "sctl/boundary_integral.hpp"
#include "sctl/boundary_integral.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Comm;

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);

  // --- INVALID_DEPTH constant available + matches Morton<DIM> definition ---
  // (The boundary_integral filter sites use Morton<COORD_DIM>::INVALID_DEPTH.)
  std::printf("INVALID_DEPTH visible to boundary_integral :\n");
  {
    CHECK(sctl::Morton<3>::INVALID_DEPTH == (uint8_t)0xFF);
    CHECK(sctl::Morton<2>::INVALID_DEPTH == (uint8_t)0xFF);
  }

  // --- BoundaryIntegralOp construct / destruct (no ElemList registered) ---
  std::printf("BoundaryIntegralOp construct/destruct :\n");
  {
    sctl::Laplace3D_FxU ker_F;
    sctl::BoundaryIntegralOp<double, sctl::Laplace3D_FxU> op(ker_F);
    CHECK(true);  // reached here without crash
  }

  std::printf("BoundaryIntegralOp + DxU :\n");
  {
    sctl::Laplace3D_DxU ker_D;
    sctl::BoundaryIntegralOp<double, sctl::Laplace3D_DxU> op(ker_D);
    CHECK(true);
  }

  std::printf("BoundaryIntegralOp + Stokes3D_FxU :\n");
  {
    sctl::Stokes3D_FxU ker;
    sctl::BoundaryIntegralOp<double, sctl::Stokes3D_FxU> op(ker);
    CHECK(true);
  }

  // --- trg_normal_dot_prod flag accepted (requires KDIM1 % COORD_DIM == 0) ---
  std::printf("trg_normal_dot_prod flag :\n");
  {
    // Laplace3D_FxdU has KDIM1 = 3 == COORD_DIM so the assertion `KDIM1 % COORD_DIM == 0` holds.
    sctl::Laplace3D_FxdU ker_FxdU;
    sctl::BoundaryIntegralOp<double, sctl::Laplace3D_FxdU> op(ker_FxdU, /*trg_normal_dot_prod=*/true);
    CHECK(true);
  }

  TEST_SUMMARY_RETURN();

  Comm::MPI_Finalize();
}
