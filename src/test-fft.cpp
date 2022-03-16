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
