#include "sctl.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);

  sctl::GMRES<long double>::test();

  sctl::Comm::MPI_Finalize();
  return 0;
}
