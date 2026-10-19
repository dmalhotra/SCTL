// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Authors: Dhairya Malhotra, Xuanzhao Gao
// SPDX-License-Identifier: Apache-2.0

#include "sctl.hpp"

#include "test-utils.hpp"

// Kernels that PVFMM cannot evaluate -- here one with a context pointer (the Helmholtz wavenumber) -- must use
// direct evaluation, also with enough targets (>= 40000) for Eval() to otherwise choose PVFMM.
static void test_pvfmm_incompatible_kernel(const sctl::Comm& comm) {
  using namespace sctl;
  std::printf("ParticleFMM: Helmholtz kernel (context pointer) above the PVFMM threshold :\n");
  Helmholtz3D_FxU ker;
  double wavenumber = 2.0;
  ker.SetCtxPtr(&wavenumber);

  const Long Ns = 64, Nt = 40000 / comm.Size() + 1;
  Vector<double> Xs(Ns * 3), Xt(Nt * 3), F(Ns * ker.SrcDim());
  srand48(comm.Rank() + 1);
  for (auto& a : Xs) a = drand48();
  for (auto& a : Xt) a = drand48();
  for (auto& a : F) a = drand48() - 0.5;

  ParticleFMM<double,3> fmm(comm);
  fmm.SetAccuracy(6);
  fmm.SetKernels(ker, ker, ker);
  fmm.AddSrc("src", ker, ker);
  fmm.AddTrg("trg", ker, ker);
  fmm.SetKernelS2T("src", "trg", ker);
  fmm.SetSrcCoord("src", Xs);
  fmm.SetSrcDensity("src", F);
  fmm.SetTrgCoord("trg", Xt);

  Vector<double> U, Uref;
  fmm.Eval(U, "trg");
  fmm.EvalDirect(Uref, "trg");
  CHECK(U.Dim() == Uref.Dim());
  double err = 0, max_val = 0;
  for (Long i = 0; i < std::min(U.Dim(), Uref.Dim()); i++) {
    err = std::max(err, std::fabs(U[i] - Uref[i]));
    max_val = std::max(max_val, std::fabs(Uref[i]));
  }
  CHECK(max_val > 0);
  CHECK(err <= 1e-12 * max_val);
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  sctl::ParticleFMM<double,3>::test(sctl::Comm::World());
  //sctl::ParticleFMM<float,2>::test(sctl::Comm::World());
  //sctl::ParticleFMM<sctl::QuadReal,2>::test(sctl::Comm::World());

  test_pvfmm_incompatible_kernel(sctl::Comm::World());

  sctl::Comm::MPI_Finalize();
  TEST_SUMMARY_RETURN();
}
