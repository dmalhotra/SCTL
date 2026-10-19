// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include "sctl.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  sctl::ParticleFMM<double,3>::test(sctl::Comm::World());
  //sctl::ParticleFMM<float,2>::test(sctl::Comm::World());
  //sctl::ParticleFMM<sctl::QuadReal,2>::test(sctl::Comm::World());

  sctl::Comm::MPI_Finalize();
  return 0;
}
