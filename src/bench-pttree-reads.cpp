// Reads of one PtTree, to measure whether building the inverse order maps once pays off.
// The maps are built on the first GetParticleData, so read 1 carries that cost and later reads
// show the steady state. Break-even is where the cumulative lines cross.
// argv: N(per rank) M K(reads)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include "sctl/experimental/gpu-tree.hpp"

using Real = double;
using sctl::Long;
static constexpr sctl::Integer kDim = 3;

static void gen_sphere(std::vector<Real>& x, Long i0, Long i1) {
  x.resize((i1 - i0) * kDim);
  const auto mix = [](uint64_t v) {
    v += 0x9E3779B97F4A7C15ull;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
  };
  #pragma omp parallel for schedule(static)
  for (Long i = i0; i < i1; i++) {
    const Real z = 2 * (Real)(mix(2 * (uint64_t)i) >> 11) * (1.0 / 9007199254740992.0) - 1;
    const Real th = 6.283185307179586 * (Real)(mix(2 * (uint64_t)i + 1) >> 11) * (1.0 / 9007199254740992.0);
    const Real r = std::sqrt(std::max<Real>(0, 1 - z * z));
    x[(i-i0)*kDim+0] = 0.5 + 0.3 * r * std::cos(th);
    x[(i-i0)*kDim+1] = 0.5 + 0.3 * r * std::sin(th);
    x[(i-i0)*kDim+2] = 0.5 + 0.3 * z;
  }
}
static double tick() { cudaDeviceSynchronize(); return SCTL_GET_WTIME(); }

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    sctl::Comm comm = sctl::Comm::World();
    const Long np = comm.Size(), rank = comm.Rank();
    const Long N = (argc > 1 ? atol(argv[1]) : 100000000), M = (argc > 2 ? atol(argv[2]) : 100);
    const int K = (argc > 3 ? atoi(argv[3]) : 6);

    std::vector<Real> xs; gen_sphere(xs, N * rank, N * (rank + 1));
    std::vector<Real> rho(N * 3);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < N * 3; i++) rho[i] = 0.5 + (Real)((i * 2654435761u) % 1000) / 1000.0;

    std::vector<double> best(K, 1e30);
    for (int rep = 0; rep < 2; rep++) {
      gpu_tree::PtTree<Real, kDim, thrust::device_vector> tr(comm);
      thrust::device_vector<Real> c(xs.begin(), xs.end()), d(rho.begin(), rho.end());
      tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0);
      tr.AddParticles("pt", c);
      tr.AddParticleData("rho", "pt", d);
      thrust::device_vector<Real> out;
      for (int k = 0; k < K; k++) {
        comm.Barrier();
        const double a = tick(); tr.GetParticleData(out, "rho"); const double b = tick();
        double t = (b - a) * 1e3, g = 0;
        comm.Allreduce(sctl::Ptr2ConstItr<double>(&t, 1), sctl::Ptr2Itr<double>(&g, 1), 1, sctl::CommOp::MAX);
        if (rep && g < best[k]) best[k] = g;
      }
    }
    if (!rank) {
      printf("  np=%ld  reads of one tree (ms, max over ranks):", (long)np);
      double cum = 0;
      for (int k = 0; k < K; k++) printf(" %7.1f", best[k]);
      printf("\n  np=%ld  cumulative:                          ", (long)np);
      for (int k = 0; k < K; k++) { cum += best[k]; printf(" %7.1f", cum); }
      printf("\n");
    }
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
