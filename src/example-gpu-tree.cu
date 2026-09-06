// Minimal use of gpu_tree::PtTree: see PtTree::test. Swap the backend for gpu_tree::HostVector to run
// the same code on the host. Build: make gpu. Run: mpirun -np 2 bin/example-gpu-tree
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include "sctl/experimental/gpu-tree.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  gpu_tree::PtTree<double, 3, gpu_tree::DeviceVector>::test();
  sctl::Comm::MPI_Finalize();
  return 0;
}
