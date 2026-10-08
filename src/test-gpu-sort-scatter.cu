/**
 * @file test-gpu-sort-scatter.cu
 * @brief gpu_tree::SortScatter round-tripped on both backends; see SortScatter::test.
 *
 * The counterpart of src/test-sort-scatter.cpp for the backend implementation. Both backends run
 * the same checks, so the host one also covers the shared plan and re-cut code without a GPU.
 *
 * Build (`make gpu`, or by hand with a CUDA-aware MPI):
 *
 *     nvcc -ccbin mpicxx -x cu -std=c++17 -O3 -arch=native -rdc=true --expt-relaxed-constexpr \
 *          -Xcompiler -fopenmp -DSCTL_HAVE_MPI -DSCTL_MAX_DEPTH=20 \
 *          -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -I include src/test-gpu-sort-scatter.cu -o bin/test-gpu-sort-scatter -ldl
 *
 * Run:
 *
 *     mpirun -np 4 bin/test-gpu-sort-scatter
 */
#include <cstdio>
#include "sctl/experimental/gpu-tree.hpp"

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const bool root = (sctl::Comm::World().Rank() == 0);
    if (root) std::printf("host backend\n");
    gpu_tree::SortScatter<sctl::Long, gpu_tree::HostVector>::test();
    if (root) std::printf("device backend\n");
    gpu_tree::SortScatter<sctl::Long, gpu_tree::DeviceVector>::test();
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
