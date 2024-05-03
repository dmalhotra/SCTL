#include "sctl.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  sctl::ParticleFMM<double,2>::test(sctl::Comm::World());
  //sctl::ParticleFMM<float,2>::test(sctl::Comm::World());
  //sctl::ParticleFMM<sctl::QuadReal,2>::test(sctl::Comm::World());

  sctl::Comm::MPI_Finalize();
  return 0;
}
