// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include "sctl.hpp"

int main(int argc, char** argv) {
  sctl::FFT<float>::test();
  sctl::FFT<double>::test();
  sctl::FFT<long double>::test();
#ifdef SCTL_QUAD_T
  sctl::FFT<sctl::QuadReal>::test();
#endif
  return 0;
}
