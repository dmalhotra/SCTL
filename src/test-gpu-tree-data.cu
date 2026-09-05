// GPUTree's stateful data layer against sctl::Tree: attach per-node data, refine to a different
// tree, and require the migrated data to match sctl node-for-node.
//
// The two trees agree on the global node set but not on where they cut it between ranks, so every
// comparison here is global: each rank contributes only its owned nodes, the concatenation in rank
// order is the globally sorted list, and that is what is compared. Data values are a function of
// the node itself for the same reason -- a local index would depend on the partition.
#include <cstdio>
#include <cmath>
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
using GT = gpu_tree::GPUTree<Real, kDim, gpu_tree::DeviceVector>;
using GNode = sctl::Morton<kDim>;

// deterministic per-node value, independent of which rank holds the node
static Real node_value(const GNode& m, int step) {
  const auto c = m.template Coord<Real>();
  return 1e4 * step + 1e3 * m.Depth() + 137.0 * c[0] + 17.0 * c[1] + 2.0 * c[2];
}

template <class T> static sctl::Vector<T> gather_all(const sctl::Comm& comm, const sctl::Vector<T>& loc) {
  const long np = comm.Size();
  sctl::Vector<long> cnt(np), dsp(np);
  const long n = loc.Dim();
  comm.Allgather(sctl::Ptr2ConstItr<long>(&n, 1), 1, cnt.begin(), 1);
  long tot = 0;
  for (long i = 0; i < np; i++) { dsp[i] = tot; tot += cnt[i]; }
  sctl::Vector<T> out(tot);
  comm.Allgatherv(loc.begin(), n, out.begin(), cnt.begin(), dsp.begin());
  return out;
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const sctl::Comm comm = sctl::Comm::World();
    const long rank = comm.Rank(), np = comm.Size();
    const sctl::Long N = (argc > 1 ? atol(argv[1]) : 200000);
    int fail = 0;

    GT gt(comm);
    sctl::Tree<kDim> st(comm);

    for (int step = 0; step < 3; step++) {
      std::mt19937_64 rng(11 + step + 1000 * rank);
      std::uniform_real_distribution<Real> U(0, 1);
      const sctl::Long Nloc = N / np + (rank < N % np ? 1 : 0);
      std::vector<Real> x(Nloc * kDim);
      for (auto& v : x) v = U(rng);

      const sctl::Long M = 32 + 8 * step;
      gpu_tree::DeviceVector<Real> cd(x.begin(), x.end());
      sctl::Vector<Real> xs(Nloc * kDim);
      for (sctl::Long i = 0; i < Nloc * kDim; i++) xs[i] = x[i];

      gt.UpdateRefinement(cd, M, true, sctl::Periodicity::NONE, 0);
      st.UpdateRefinement(xs, M, true, sctl::Periodicity::NONE, 0);

      // ---- owned slices of each tree ----
      sctl::Long gb, ge; gt.GetOwnedRange(gb, ge);
      thrust::host_vector<GNode> gmid_h(gt.GetNodeMID());
      const auto& smid = st.GetNodeMID();
      const auto& sattr = st.GetNodeAttr();
      sctl::Long sb = 0, se = smid.Dim();
      while (sb < se && sattr[sb].Ghost) sb++;
      se = sb; while (se < smid.Dim() && !sattr[se].Ghost) se++;

      { // the global node sets must agree, even though the cuts do not
        sctl::Vector<GNode> g(ge - gb), s(se - sb);
        for (sctl::Long i = 0; i < ge - gb; i++) g[i] = gmid_h[gb + i];
        for (sctl::Long i = 0; i < se - sb; i++) s[i] = smid[sb + i];
        const auto ga = gather_all(comm, g), sa = gather_all(comm, s);
        int bad = (ga.Dim() != sa.Dim());
        if (!bad) for (sctl::Long i = 0; i < ga.Dim(); i++) if (!(ga[i] == sa[i])) { bad = 1; break; }
        if (bad) { if (!rank) printf("  step %d: GLOBAL NODE SET DIFFERS (%ld vs %ld)\n", step, (long)ga.Dim(), (long)sa.Dim()); fail++; }
      }

      if (step) { // what the previous step's data became, compared globally
        gpu_tree::DataView<const Real, gpu_tree::DeviceVector> gd; sctl::Vector<long> gc;
        sctl::Vector<Real> sd; sctl::Vector<long> sc;
        gt.GetData(gd, gc, "f");
        st.GetData(sd, sc, "f");
        thrust::host_vector<Real> gh(gd.begin(), gd.end());

        sctl::Vector<long> gco(ge - gb), sco(se - sb);
        for (sctl::Long i = 0; i < ge - gb; i++) gco[i] = gc[gb + i];
        for (sctl::Long i = 0; i < se - sb; i++) sco[i] = sc[sb + i];
        sctl::Long goff = 0, soff = 0;
        for (sctl::Long i = 0; i < gb; i++) goff += gc[i];
        for (sctl::Long i = 0; i < sb; i++) soff += sc[i];
        sctl::Long gn = 0, sn = 0;
        for (sctl::Long i = 0; i < gco.Dim(); i++) gn += gco[i];
        for (sctl::Long i = 0; i < sco.Dim(); i++) sn += sco[i];
        sctl::Vector<Real> gv(gn), sv(sn);
        for (sctl::Long i = 0; i < gn; i++) gv[i] = gh[goff + i];
        for (sctl::Long i = 0; i < sn; i++) sv[i] = sd[soff + i];

        const auto gca = gather_all(comm, gco), sca = gather_all(comm, sco);
        const auto gva = gather_all(comm, gv),  sva = gather_all(comm, sv);
        int bad_cnt = 0, bad_val = 0;
        if (gca.Dim() != sca.Dim() || gva.Dim() != sva.Dim()) { bad_cnt = bad_val = -1; fail++; }
        else {
          for (sctl::Long i = 0; i < gca.Dim(); i++) if (gca[i] != sca[i]) bad_cnt++;
          for (sctl::Long i = 0; i < gva.Dim(); i++) if (gva[i] != sva[i]) bad_val++;
          if (bad_cnt || bad_val) fail++;
        }
        if (!rank) printf("  step %d: migrated cnt mismatches=%d  value mismatches=%d  (global nodes=%ld items=%ld)\n",
                          step, bad_cnt, bad_val, (long)gca.Dim(), (long)gva.Dim());
        gt.DeleteData("f");
        st.DeleteData("f");
      }

      { // attach one value per node, a function of the node so both trees agree
        const sctl::Long Ng = (sctl::Long)gt.GetNodeMID().size(), Ns = smid.Dim();
        sctl::Vector<long> gcnt(Ng), scnt(Ns);
        std::vector<Real> gval(Ng); sctl::Vector<Real> sval(Ns);
        for (sctl::Long i = 0; i < Ng; i++) { gcnt[i] = 1; gval[i] = node_value(gmid_h[i], step); }
        for (sctl::Long i = 0; i < Ns; i++) { scnt[i] = 1; sval[i] = node_value(smid[i], step); }
        gpu_tree::DeviceVector<Real> vd(gval.begin(), gval.end());
        gt.AddData("f", vd, gcnt);
        st.AddData("f", sval, scnt);
      }
    }
    if (!rank) printf("%s (%d mismatch(es))\n", fail ? "FAIL" : "PASS", fail);
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
