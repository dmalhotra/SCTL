/**
 * @file example-gpu-tree.cu
 * @brief Minimal use of gpu_tree::PtTree; see PtTree::test. Swap the backend for gpu_tree::HostVector
 * to run the same code on the host.
 *
 * Build (`make gpu`, or by hand with a CUDA-aware MPI):
 *
 *     nvcc -ccbin mpicxx -x cu -std=c++17 -O3 -arch=native -rdc=true --expt-relaxed-constexpr \
 *          -Xcompiler -fopenmp -DSCTL_HAVE_MPI -DSCTL_MAX_DEPTH=20 \
 *          -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -I include src/example-gpu-tree.cu -o bin/example-gpu-tree -ldl
 *
 * Run:
 *
 *     mpirun -np 2 bin/example-gpu-tree
 */
#include "sctl/experimental/gpu-tree.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  gpu_tree::PtTree<double, 3, gpu_tree::DeviceVector>::test();
  sctl::Comm::MPI_Finalize();
  return 0;
}
