#include "sctl.hpp"

int main(int argc, char** argv) {

  sctl::InterpQuadRule<long double>::test();

  #ifdef SCTL_QUAD_T
  sctl::InterpQuadRule<sctl::QuadReal>::test();
  #endif

  return 0;
}


