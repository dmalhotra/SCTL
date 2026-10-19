// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include "sctl.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  sctl::GMRES<long double>::test();

  sctl::Comm::MPI_Finalize();
  return 0;
}
