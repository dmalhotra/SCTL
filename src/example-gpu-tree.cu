// Minimal use of gpu_tree::PtTree: random particles with one value each, a distributed tree with at
// most 64 particles per leaf, the values doubled in place through a view of the node data, and the
// result scattered back to the caller's order. Swap the backend for gpu_tree::HostVector to run the
// same code on the host. Build: make gpu. Run: mpirun -np 2 bin/example-gpu-tree
#include <cstdio>
#include <random>
#include <vector>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include "sctl/experimental/gpu-tree.hpp"

template <class T> using Vec = gpu_tree::DeviceVector<T>;

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    using Real = double;
    constexpr sctl::Integer DIM = 3;
    const sctl::Comm comm = sctl::Comm::World();
    const sctl::Long N = 100000;  // particles on this rank

    std::mt19937_64 rng(comm.Rank());
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> x(N * DIM), f(N);
    for (auto& v : x) v = U(rng);
    for (sctl::Long i = 0; i < N; i++) f[i] = x[i * DIM];  // any per-particle value
    const Vec<Real> xd(x.begin(), x.end()), fd(f.begin(), f.end());

    gpu_tree::PtTree<Real, DIM, Vec> tree(comm);
    tree.UpdateRefinement(xd, 64, true);  // leaves hold at most 64 particles; 2:1 balanced
    tree.AddParticles("pt", xd);          // sorted into node order and distributed by the partition
    tree.AddParticleData("f", "pt", fd);  // follows the particles

    { // work on the stored data in place: the view aliases the tree's storage, in node order
      gpu_tree::DataView<Real, Vec> v; sctl::Vector<sctl::Long> cnt;  // cnt[i]: particles in node i
      tree.GetData(v, cnt, "f");
      thrust::transform(v.begin(), v.end(), v.begin(), 2.0 * thrust::placeholders::_1);
    }

    Vec<Real> out;
    tree.GetParticleData(out, "f");  // back in the caller's order: out[i] == 2 * f[i]
    std::vector<Real> h(out.size());
    thrust::copy(out.begin(), out.end(), h.begin());
    sctl::Long bad = 0;
    for (sctl::Long i = 0; i < N; i++) bad += (h[i] != 2 * f[i]);

    sctl::Long b, e;
    tree.GetOwnedRange(b, e);
    printf("rank %d: %ld particles, %ld nodes of which %ld owned, %ld values wrong\n",
           (int)comm.Rank(), (long)N, (long)tree.GetNodeMID().size(), (long)(e - b), (long)bad);
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
