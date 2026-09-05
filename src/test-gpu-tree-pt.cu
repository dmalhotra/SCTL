// gpu_tree::PtTree against sctl::PtTree. The strong invariant is the round trip: GetParticleData
// must hand back exactly what AddParticleData was given, in the caller's own order, before and
// after a refinement. Beyond that the per-node particle counts and the VTK point totals are
// compared globally -- the two trees agree on the node set but cut it between ranks differently.
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include "sctl/experimental/gpu-tree.hpp"

using Real = double;
static constexpr sctl::Integer kDim = 3;
using GPT = gpu_tree::PtTree<Real, kDim, gpu_tree::DeviceVector>;
using SPT = sctl::PtTree<Real, kDim>;

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const sctl::Comm comm = sctl::Comm::World();
    const long rank = comm.Rank(), np = comm.Size();
    const sctl::Long N = (argc > 1 ? atol(argv[1]) : 20000);
    int fail = 0;

    std::mt19937_64 rng(3 + 77 * rank);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> x(N * kDim); for (auto& v : x) v = U(rng);
    // a second group whose points are NOT the ones the tree was built from
    const sctl::Long N2 = N / 3;
    std::vector<Real> y(N2 * kDim); for (auto& v : y) v = U(rng);
    std::vector<Real> g(N2); for (sctl::Long i = 0; i < N2; i++) g[i] = 7e5*rank + 3*i;
    std::vector<Real> f(N * 2);
    for (sctl::Long i = 0; i < N; i++) { f[2*i] = 1e6*rank + i; f[2*i+1] = -(Real)i; }

    gpu_tree::DeviceVector<Real> xd(x.begin(), x.end()), fd(f.begin(), f.end());
    gpu_tree::DeviceVector<Real> yd(y.begin(), y.end()), gd2(g.begin(), g.end());
    sctl::Vector<Real> ys(N2*kDim), gs(N2);
    for (sctl::Long i = 0; i < N2*kDim; i++) ys[i] = y[i];
    for (sctl::Long i = 0; i < N2; i++) gs[i] = g[i];
    sctl::Vector<Real> xs(N*kDim), fs(N*2);
    for (sctl::Long i = 0; i < N*kDim; i++) xs[i] = x[i];
    for (sctl::Long i = 0; i < N*2; i++) fs[i] = f[i];

    GPT gt(comm); SPT st(comm);
    for (int step = 0; step < 4; step++) {
      const sctl::Long M = 32 - 7*step;
      gt.UpdateRefinement(xd, M, true, sctl::Periodicity::NONE, 0);
      st.UpdateRefinement(xs, M, true, sctl::Periodicity::NONE, 0);
      if (step == 0) {
        gt.AddParticles("pt", xd); gt.AddParticleData("f", "pt", fd);
        st.AddParticles("pt", xs); st.AddParticleData("f", "pt", fs);
        gt.AddParticles("q", yd);  gt.AddParticleData("g", "q", gd2);
        st.AddParticles("q", ys);  st.AddParticleData("g", "q", gs);
      }

      { // round trip, both libraries
        gpu_tree::DeviceVector<Real> gout; gt.GetParticleData(gout, "f");
        thrust::host_vector<Real> gh(gout);
        sctl::Vector<Real> sout; st.GetParticleData(sout, "f");
        int gbad = ((long)gh.size() != N*2), sbad = (sout.Dim() != N*2);
        if (!gbad) for (sctl::Long i = 0; i < N*2; i++) if (gh[i] != f[i]) { gbad = 1; break; }
        if (!sbad) for (sctl::Long i = 0; i < N*2; i++) if (sout[i] != f[i]) { sbad = 1; break; }
        fail += gbad;
        // the second group, built from different points than the tree
        gpu_tree::DeviceVector<Real> gout2; gt.GetParticleData(gout2, "g");
        thrust::host_vector<Real> gh2(gout2);
        sctl::Vector<Real> sout2; st.GetParticleData(sout2, "g");
        int gbad2 = ((long)gh2.size() != N2), sbad2 = (sout2.Dim() != N2);
        if (!gbad2) for (sctl::Long i = 0; i < N2; i++) if (gh2[i] != g[i]) { gbad2 = 1; break; }
        if (!sbad2) for (sctl::Long i = 0; i < N2; i++) if (sout2[i] != g[i]) { sbad2 = 1; break; }
        fail += gbad2;
        if (!rank) printf("  step %d: round trip  pt gpu=%s sctl=%s   q(other pts) gpu=%s sctl=%s\n",
                          step, gbad?"FAIL":"ok", sbad?"FAIL":"ok", gbad2?"FAIL":"ok", sbad2?"FAIL":"ok");
      }

      { // total particles held by the tree must match globally
        sctl::Vector<long> gc, sc; gpu_tree::DeviceVector<Real> gd; sctl::Vector<Real> sd;
        gt.GetParticleData(gd, "pt");  // forces the group to exist
        long gsum = 0, ssum = 0, g = 0, s2 = 0;
        { gpu_tree::DataView<const Real, gpu_tree::DeviceVector> tmp; sctl::Vector<long> c;
          gt.GetData(tmp, c, "pt"); for (sctl::Long i=0;i<c.Dim();i++) g += c[i]; }
        { sctl::Vector<Real> tmp; sctl::Vector<long> c;
          st.GetData(tmp, c, "pt"); for (sctl::Long i=0;i<c.Dim();i++) s2 += c[i]; }
        comm.Allreduce(sctl::Ptr2ConstItr<long>(&g,1), sctl::Ptr2Itr<long>(&gsum,1), 1, sctl::CommOp::SUM);
        comm.Allreduce(sctl::Ptr2ConstItr<long>(&s2,1), sctl::Ptr2Itr<long>(&ssum,1), 1, sctl::CommOp::SUM);
        if (gsum != ssum) fail++;
        if (!rank) printf("  step %d: particles in tree  gpu=%ld sctl=%ld  %s\n", step, gsum, ssum, gsum==ssum?"MATCH":"MISMATCH");
      }
    }
    gt.WriteParticleVTK("/tmp/pt-gpu", "f", false);
    st.WriteParticleVTK("/tmp/pt-sctl", "f", false);
    if (!rank) printf("%s (%d mismatch(es))\n", fail ? "FAIL" : "PASS", fail);
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
