// PtTree end-to-end: build a tree on sphere-surface particles, attach dof=3 density, then refine
// onto a uniform distribution. Times each API stage for sctl::PtTree, gpu_tree::PtTree on the host
// backend, and the same on the device. GTPROF=1 additionally prints buildTreeDist's own stages.
//
// argv: N(per rank) M mode(s|c|g|all) reps
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 20
#endif
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
static void gen_uniform(std::vector<Real>& x, Long i0, Long i1) {
  x.resize((i1 - i0) * kDim);
  const auto mix = [](uint64_t v) {
    v += 0xD6E8FEB86659FD93ull;
    v = (v ^ (v >> 32)) * 0xD6E8FEB86659FD93ull;
    v = (v ^ (v >> 32)) * 0xD6E8FEB86659FD93ull;
    return v ^ (v >> 32);
  };
  #pragma omp parallel for schedule(static)
  for (Long i = i0; i < i1; i++)
    for (sctl::Integer d = 0; d < kDim; d++)
      x[(i-i0)*kDim+d] = (Real)(mix(3*(uint64_t)i + d) >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    sctl::Comm comm = sctl::Comm::World();
    const Long np = comm.Size(), rank = comm.Rank();
    const Long N = (argc > 1 ? atol(argv[1]) : 10000000);   // per rank
    const Long M = (argc > 2 ? atol(argv[2]) : 100);
    const std::string mode = (argc > 3 ? argv[3] : "all");
    const int reps = (argc > 4 ? atoi(argv[4]) : 2);
    const bool all = (mode == "all");

    { char host[64]; gethostname(host, sizeof(host));
      int dev = -1; cudaGetDevice(&dev);
      cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev);
      const int nt = omp_get_max_threads();
      std::vector<int> cpu(nt, -1);
      #pragma omp parallel num_threads(nt)
      cpu[omp_get_thread_num()] = sched_getcpu();
      int lo = cpu[0], hi = cpu[0];
      for (int i = 1; i < nt; i++) { lo = std::min(lo, cpu[i]); hi = std::max(hi, cpu[i]); }
      for (Long r = 0; r < np; r++) { comm.Barrier();
        if (r == rank) printf("  rank %ld: %s gpu=%02x:%02x (%s) threads=%d cores=%d-%d\n",
                              (long)rank, host, pr.pciBusID, pr.pciDeviceID, pr.name, nt, lo, hi);
        fflush(stdout); }
      comm.Barrier(); }

    std::vector<Real> xs, xu;
    gen_sphere (xs, N * rank, N * (rank + 1));
    gen_uniform(xu, N * rank, N * (rank + 1));
    std::vector<Real> rho(N * 3);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < N * 3; i++) rho[i] = 0.5 + (Real)((i * 2654435761u) % 1000) / 1000.0;

    if (!rank) printf("\n# np=%ld  N=%ld/rank (%ld total)  M=%ld  dof=3  balance21=1  %d threads/rank  %d timed reps\n",
                      (long)np, (long)N, (long)(N*np), (long)M, omp_get_max_threads(), reps);
    if (!rank) printf("  %-18s %13s %10s %10s %13s %10s %12s\n",
                      "variant", "build(sphere)", "AddPart", "AddData", "refine(unif)", "GetData", "nodes");

    const auto mx = [&comm](double t) { double g = 0; comm.Allreduce(sctl::Ptr2ConstItr<double>(&t,1), sctl::Ptr2Itr<double>(&g,1), 1, sctl::CommOp::MAX); return g; };
    const auto tick = [](){ cudaDeviceSynchronize(); return SCTL_GET_WTIME(); };

    const auto row = [&](const char* name, auto&& run) {
      double best[5] = {1e30,1e30,1e30,1e30,1e30}; long nodes = 0;
      for (int r = 0; r <= reps; r++) {
        double t[5] = {0,0,0,0,0};
        comm.Barrier();
        nodes = run(t);
        for (int i = 0; i < 5; i++) { const double g = mx(t[i]); if (r && g < best[i]) best[i] = g; }
      }
      long gn = 0; comm.Allreduce(sctl::Ptr2ConstItr<long>(&nodes,1), sctl::Ptr2Itr<long>(&gn,1), 1, sctl::CommOp::SUM);
      if (!rank) printf("  %-18s %13.1f %10.1f %10.1f %13.1f %10.1f %12ld\n", name, best[0], best[1], best[2], best[3], best[4], gn);
    };

    if (all || mode == "s") row("sctl::PtTree", [&](double* t) {
      sctl::PtTree<Real, kDim> tr(comm);
      sctl::Vector<Real> c(N*kDim), u(N*kDim), d(N*3);
      for (Long i = 0; i < N*kDim; i++) { c[i] = xs[i]; u[i] = xu[i]; }
      for (Long i = 0; i < N*3; i++) d[i] = rho[i];
      double a = tick(); tr.AddParticles("pt", c);                                          double b = tick(); t[1] = (b-a)*1e3;
      a = b; tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0);                   b = tick(); t[0] = (b-a)*1e3;
      a = b; tr.AddParticleData("rho", "pt", d);                                            b = tick(); t[2] = (b-a)*1e3;
      a = b; tr.UpdateRefinement(u, M, true, sctl::Periodicity::NONE, 0);                   b = tick(); t[3] = (b-a)*1e3;
      sctl::Vector<Real> out; a = b; tr.GetParticleData(out, "rho");                        b = tick(); t[4] = (b-a)*1e3;
      return (long)tr.GetNodeMID().Dim();
    });

    if (all || mode == "c") row("GPUTree PtTree CPU", [&](double* t) {
      gpu_tree::PtTree<Real, kDim, std::vector> tr(comm);
      std::vector<Real> c(xs), u(xu), d(rho);
      double a = tick(); tr.AddParticles("pt", c);                                          double b = tick(); t[1] = (b-a)*1e3;
      a = b; tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0);                   b = tick(); t[0] = (b-a)*1e3;
      a = b; tr.AddParticleData("rho", "pt", d);                                            b = tick(); t[2] = (b-a)*1e3;
      a = b; tr.UpdateRefinement(u, M, true, sctl::Periodicity::NONE, 0);                   b = tick(); t[3] = (b-a)*1e3;
      std::vector<Real> out; a = b; tr.GetParticleData(out, "rho");                         b = tick(); t[4] = (b-a)*1e3;
      return (long)tr.GetNodeMID().size();
    });

    if (all || mode == "g") row("GPUTree PtTree GPU", [&](double* t) {
      gpu_tree::PtTree<Real, kDim, thrust::device_vector> tr(comm);
      thrust::device_vector<Real> c(xs.begin(), xs.end()), u(xu.begin(), xu.end()), d(rho.begin(), rho.end());
      double a = tick(); tr.AddParticles("pt", c);                                          double b = tick(); t[1] = (b-a)*1e3;
      a = b; tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0);                   b = tick(); t[0] = (b-a)*1e3;
      a = b; tr.AddParticleData("rho", "pt", d);                                            b = tick(); t[2] = (b-a)*1e3;
      a = b; tr.UpdateRefinement(u, M, true, sctl::Periodicity::NONE, 0);                   b = tick(); t[3] = (b-a)*1e3;
      thrust::device_vector<Real> out; a = b; tr.GetParticleData(out, "rho");               b = tick(); t[4] = (b-a)*1e3;
      return (long)tr.GetNodeMID().size();
    });
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
