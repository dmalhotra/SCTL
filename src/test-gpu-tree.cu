// gpu_tree against sctl on every backend; see PtTree::test. Build: make gpu. Run under mpirun.
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include "sctl/experimental/gpu-tree.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const bool root = (sctl::Comm::World().Rank() == 0);
    if (root) printf("HostVector backend\n");
    gpu_tree::PtTree<double, 3, gpu_tree::HostVector>::test();
    if (root) printf("DeviceVector backend\n");
    gpu_tree::PtTree<double, 3, gpu_tree::DeviceVector>::test();
    if (root) printf("std::vector backend\n");
    gpu_tree::PtTree<double, 3, std::vector>::test();
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
