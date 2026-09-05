// PtTree end-to-end: refine on sphere-surface particles, then add them and attach dof=3 density,
// read it back, and finally refine on a uniform point set, which repartitions the sphere data onto
// a partition that ignores it. Refine-first is the idiomatic order: adding a non-uniform group
// before the first refinement concentrates it on the coarse starting partition -- measured 3.7x on
// one rank at np=16, 5.5x at np=24 -- so the add-first order runs out of device memory at scale.
// Times each API stage for sctl::PtTree, gpu_tree::PtTree on the host backend, and the same on the
// device. GTPROF=1 additionally prints buildTreeDist's own stages. When tools/pmpiprof.c is
// LD_PRELOADed, each stage is also split into mpi, shared (node-local direct reads between ranks)
// and compute rows; waiting on a slower rank counts as mpi. Peak resident memory (max over ranks) is printed after each stage, and the share
// of particles per rank (max/avg, min/avg) under the sphere partition and under the uniform one.
//
// argv: N(per rank) M mode(s|c|g|cpu|all) reps N_uniform(total, default 1M)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/resource.h>
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

// Seconds spent inside MPI so far, from the pmpiprof shim when preloaded; zero without it.
typedef double (*mpiprof_fn)(void);
static mpiprof_fn g_mpiprof = nullptr;
static double commsec() { return g_mpiprof ? g_mpiprof() : 0.0; }
static mpiprof_fn g_shmprof = nullptr;  // wall seconds with a node-local direct read in flight, same shim
static double shmsec() { return g_shmprof ? g_shmprof() : 0.0; }
static constexpr sctl::Integer kDim = 3;

static uint64_t mix(uint64_t v) {
  v += 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}
static Real unit(uint64_t v) { return (Real)(mix(v) >> 11) * (1.0 / 9007199254740992.0); }

static void gen_sphere(std::vector<Real>& x, Long i0, Long i1) {
  x.resize((i1 - i0) * kDim);
  #pragma omp parallel for schedule(static)
  for (Long i = i0; i < i1; i++) {
    const Real z = 2 * unit(2 * (uint64_t)i) - 1;
    const Real th = 6.283185307179586 * unit(2 * (uint64_t)i + 1);
    const Real r = std::sqrt(std::max<Real>(0, 1 - z * z));
    x[(i-i0)*kDim+0] = 0.5 + 0.3 * r * std::cos(th);
    x[(i-i0)*kDim+1] = 0.5 + 0.3 * r * std::sin(th);
    x[(i-i0)*kDim+2] = 0.5 + 0.3 * z;
  }
}

static void gen_uniform(std::vector<Real>& x, Long i0, Long i1) {
  x.resize((i1 - i0) * kDim);
  #pragma omp parallel for schedule(static)
  for (Long i = i0; i < i1; i++)
    for (int k = 0; k < kDim; k++) x[(i-i0)*kDim+k] = std::min<Real>(unit(3 * (uint64_t)i + k + (1ull << 40)), 1 - 1e-12);
}

/** Peak resident set of this process, GB (ru_maxrss is in KB on Linux). */
static double peakRssGB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / (1024.0 * 1024.0); }

/** Share of the input particles each rank owns under the partition `mins`, as max/avg and min/avg
 *  over ranks. Counted from every rank's own input against the splitters, so no tree data moves. */
static void imbalance(const sctl::Vector<sctl::Morton<kDim>>& mins, const std::vector<Real>& xs, const sctl::Comm& comm, double& hi, double& lo) {
  const Long np = comm.Size(), n = (Long)xs.size() / kDim;
  std::vector<sctl::MortonCode<kDim>> spl(np);
  for (Long r = 0; r < np; r++) spl[r] = mins[r].mid;
  std::vector<Long> cnt(np, 0);
  #pragma omp parallel
  { std::vector<Long> loc(np, 0);
    #pragma omp for schedule(static)
    for (Long i = 0; i < n; i++) {
      const sctl::MortonCode<kDim> c(&xs[i * kDim]);
      loc[std::max<Long>(0, std::upper_bound(spl.begin(), spl.end(), c) - spl.begin() - 1)]++;
    }
    #pragma omp critical
    for (Long r = 0; r < np; r++) cnt[r] += loc[r]; }
  std::vector<Long> tot(np);
  comm.Allreduce(sctl::Ptr2ConstItr<Long>(cnt.data(), np), sctl::Ptr2Itr<Long>(tot.data(), np), np, sctl::CommOp::SUM);
  Long mxv = tot[0], mnv = tot[0], sum = 0;
  for (Long r = 0; r < np; r++) { mxv = std::max(mxv, tot[r]); mnv = std::min(mnv, tot[r]); sum += tot[r]; }
  const double avg = (double)sum / np;
  hi = mxv / avg; lo = mnv / avg;
}

// Slots of the per-rep record: wall 0-4, mpi 5-9, compute 10-14, peak RSS 15-19 (one per stage:
// build, AddPart, AddData, GetData, Refine(unif)); imbalance 20-23 (max/avg, min/avg after the add,
// then after the uniform refine); 24 nodes after the uniform refine.
enum { kWall = 0, kMpi = 5, kShm = 10, kCompute = 15, kRss = 20, kImb = 25, kNodesUnif = 29, kSlots = 30 };

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    sctl::Comm comm = sctl::Comm::World();
    const Long np = comm.Size(), rank = comm.Rank();
    g_mpiprof = (mpiprof_fn)dlsym(RTLD_DEFAULT, "mpiprof_seconds");
    g_shmprof = (mpiprof_fn)dlsym(RTLD_DEFAULT, "shmprof_seconds");
    const Long N = (argc > 1 ? atol(argv[1]) : 10000000);   // per rank
    const Long M = (argc > 2 ? atol(argv[2]) : 100);
    const std::string mode = (argc > 3 ? argv[3] : "all");
    const int reps = (argc > 4 ? atoi(argv[4]) : 2);
    const Long Nu = (argc > 5 ? atol(argv[5]) : 1000000);  // uniform points, total over ranks
    const bool all = (mode == "all"), cpu = (mode == "cpu");

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
    gen_sphere(xs, N * rank, N * (rank + 1));
    gen_uniform(xu, Nu * rank / np, Nu * (rank + 1) / np);
    std::vector<Real> rho(N * 3);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < N * 3; i++) rho[i] = 0.5 + (Real)((i * 2654435761u) % 1000) / 1000.0;

    if (!rank) printf("\n# np=%ld  N=%ld/rank (%ld total)  M=%ld  dof=3  balance21=1  uniform refine: %ld pts total  %d threads/rank  %d timed reps\n",
                      (long)np, (long)N, (long)(N*np), (long)M, (long)Nu, omp_get_max_threads(), reps);
    if (!rank) printf("  %-18s %13s %10s %10s %10s %12s %12s %12s\n",
                      "variant", "build(sphere)", "AddPart", "AddData", "GetData", "Refine(unif)", "nodes", "nodes(unif)");

    const auto mx = [&comm](double t) { double g = 0; comm.Allreduce(sctl::Ptr2ConstItr<double>(&t,1), sctl::Ptr2Itr<double>(&g,1), 1, sctl::CommOp::MAX); return g; };
    const auto tick = [](){ cudaDeviceSynchronize(); return SCTL_GET_WTIME(); };
    // With GTPROF set the library prints its stage lines to stderr; a banner per bench stage groups them.
    const bool gtprof = (getenv("GTPROF") != nullptr);
    const auto banner = [&](const char* variant, const char* stage) {
      if (!gtprof) return;
      fflush(stdout); comm.Barrier();
      if (!rank) { fprintf(stderr, "  -- %s: %s --\n", variant, stage); fflush(stderr); }
    };

    const auto stage = [&](double* t, int i, auto&& fn) {  // wall, mpi and shared seconds of one API call, plus peak RSS after it
      const double a = tick(), ca = commsec(), sa = shmsec();
      fn();
      const double b = tick();
      t[kWall + i] = (b - a) * 1e3; t[kMpi + i] = (commsec() - ca) * 1e3; t[kShm + i] = (shmsec() - sa) * 1e3; t[kRss + i] = peakRssGB();
    };

    const auto row = [&](const char* name, auto&& run) {
      double best[kSlots]; for (int i = 0; i < kSlots; i++) best[i] = 1e30;
      double rss[5] = {0}, imb[4] = {0};
      long nodes = 0, nodes_unif = 0;
      for (int r = 0; r <= reps; r++) {
        double t[kSlots] = {0};
        comm.Barrier();
        nodes = run(t);
        for (int i = 0; i < 5; i++) t[kCompute + i] = t[kWall + i] - t[kMpi + i] - t[kShm + i];  // per-rank compute = wall - mpi - shared
        for (int i = 0; i < kRss; i++) { const double g = mx(t[i]); if (r && g < best[i]) best[i] = g; }
        for (int i = 0; i < 5; i++) rss[i] = std::max(rss[i], mx(t[kRss + i]));  // ru_maxrss only grows
        for (int i = 0; i < 4; i++) imb[i] = t[kImb + i];
        nodes_unif = (long)t[kNodesUnif];
      }
      long gn[2] = {nodes, nodes_unif}, g[2] = {0, 0};
      comm.Allreduce(sctl::Ptr2ConstItr<long>(gn, 2), sctl::Ptr2Itr<long>(g, 2), 2, sctl::CommOp::SUM);
      if (!rank) {
        printf("  %-18s %13.1f %10.1f %10.1f %10.1f %12.1f %12ld %12ld\n", name, best[0], best[1], best[2], best[3], best[4], g[0], g[1]);
        if (g_mpiprof) {
          printf("  %-18s %13.1f %10.1f %10.1f %10.1f %12.1f\n", "    - mpi",     best[kMpi], best[kMpi+1], best[kMpi+2], best[kMpi+3], best[kMpi+4]);
          printf("  %-18s %13.1f %10.1f %10.1f %10.1f %12.1f\n", "    - shared",  best[kShm], best[kShm+1], best[kShm+2], best[kShm+3], best[kShm+4]);
          printf("  %-18s %13.1f %10.1f %10.1f %10.1f %12.1f\n", "    - compute", best[kCompute], best[kCompute+1], best[kCompute+2], best[kCompute+3], best[kCompute+4]);
        }
        printf("  %-18s %13.2f %10.2f %10.2f %10.2f %12.2f\n", "    - peak RSS GB", rss[0], rss[1], rss[2], rss[3], rss[4]);
        printf("  %-18s particles/rank max/avg, min/avg: sphere partition %.2f/%.2f, uniform partition %.2f/%.2f\n", "    - imbalance", imb[0], imb[1], imb[2], imb[3]);
      }
    };

    if (all || cpu || mode == "s") row("sctl::PtTree", [&](double* t) {
      sctl::PtTree<Real, kDim> tr(comm);
      const char* const v = "sctl::PtTree";
      sctl::Vector<Real> c(N*kDim), d(N*3), u((Long)xu.size());
      for (Long i = 0; i < N*kDim; i++) c[i] = xs[i];
      for (Long i = 0; i < N*3; i++) d[i] = rho[i];
      for (Long i = 0; i < (Long)xu.size(); i++) u[i] = xu[i];
      banner(v, "build(sphere)");
      stage(t, 0, [&] { tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0); });
      stage(t, 1, [&] { tr.AddParticles("pt", c); });
      imbalance(tr.GetPartitionMID(), xs, comm, t[kImb+0], t[kImb+1]);
      stage(t, 2, [&] { tr.AddParticleData("rho", "pt", d); });
      sctl::Vector<Real> out;
      stage(t, 3, [&] { tr.GetParticleData(out, "rho"); });
      const long nodes = tr.GetNodeMID().Dim();
      banner(v, "Refine(unif)");
      stage(t, 4, [&] { tr.UpdateRefinement(u, M, true, sctl::Periodicity::NONE, 0); });
      imbalance(tr.GetPartitionMID(), xs, comm, t[kImb+2], t[kImb+3]);
      t[kNodesUnif] = (double)tr.GetNodeMID().Dim();
      return nodes;
    });

    if (all || cpu || mode == "c") row("GPUTree PtTree CPU", [&](double* t) {
      gpu_tree::PtTree<Real, kDim, gpu_tree::HostVector> tr(comm);
      const char* const v = "GPUTree PtTree CPU";
      gpu_tree::HostVector<Real> c(xs.begin(), xs.end()), d(rho.begin(), rho.end()), u(xu.begin(), xu.end());
      banner(v, "build(sphere)");
      stage(t, 0, [&] { tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0); });
      stage(t, 1, [&] { tr.AddParticles("pt", c); });
      imbalance(tr.GetPartitionMID(), xs, comm, t[kImb+0], t[kImb+1]);
      stage(t, 2, [&] { tr.AddParticleData("rho", "pt", d); });
      gpu_tree::HostVector<Real> out;
      stage(t, 3, [&] { tr.GetParticleData(out, "rho"); });
      const long nodes = tr.GetNodeMID().size();
      banner(v, "Refine(unif)");
      stage(t, 4, [&] { tr.UpdateRefinement(u, M, true, sctl::Periodicity::NONE, 0); });
      imbalance(tr.GetPartitionMID(), xs, comm, t[kImb+2], t[kImb+3]);
      t[kNodesUnif] = (double)tr.GetNodeMID().size();
      return nodes;
    });

    if (all || mode == "g") row("GPUTree PtTree GPU", [&](double* t) {
      gpu_tree::PtTree<Real, kDim, thrust::device_vector> tr(comm);
      const char* const v = "GPUTree PtTree GPU";
      thrust::device_vector<Real> c(xs.begin(), xs.end()), d(rho.begin(), rho.end()), u(xu.begin(), xu.end());
      banner(v, "build(sphere)");
      stage(t, 0, [&] { tr.UpdateRefinement(c, M, true, sctl::Periodicity::NONE, 0); });
      stage(t, 1, [&] { tr.AddParticles("pt", c); });
      imbalance(tr.GetPartitionMID(), xs, comm, t[kImb+0], t[kImb+1]);
      stage(t, 2, [&] { tr.AddParticleData("rho", "pt", d); });
      thrust::device_vector<Real> out;
      stage(t, 3, [&] { tr.GetParticleData(out, "rho"); });
      const long nodes = tr.GetNodeMID().size();
      banner(v, "Refine(unif)");
      stage(t, 4, [&] { tr.UpdateRefinement(u, M, true, sctl::Periodicity::NONE, 0); });
      imbalance(tr.GetPartitionMID(), xs, comm, t[kImb+2], t[kImb+3]);
      t[kNodesUnif] = (double)tr.GetNodeMID().size();
      return nodes;
    });
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
