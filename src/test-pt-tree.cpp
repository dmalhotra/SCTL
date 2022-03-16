#include <sctl.hpp>

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  sctl::PtTree<double,2>::test();
  sctl::Comm::MPI_Finalize();

  return 0;
}

