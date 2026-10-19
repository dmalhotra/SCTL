// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include "sctl.hpp"

int main(int argc, char** argv) {
  sctl::SphericalHarmonics<double>::test();
  sctl::SphericalHarmonics<double>::test_stokes();
  return 0;
}
