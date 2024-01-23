#include "sctl.hpp"

int main(int argc, char** argv) {

  sctl::InterpQuadRule<long double,false>::test();

  #ifdef SCTL_QUAD_T
  sctl::InterpQuadRule<sctl::QuadReal,false>::test();
  #endif

  return 0;
}


