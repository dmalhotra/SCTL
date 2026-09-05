// GPUTree::Broadcast / ReduceBroadcast against sctl::Tree.
//
// Broadcast's promise is self-checkable: after it, every node carrying data -- owned or ghost --
// must hold the value its owner published. Since the value is a function of the node, that can be
// verified without reference to sctl, on both libraries. ReduceBroadcast is then compared to sctl
// globally over owned nodes, the two trees cutting the node set differently.
#include <cstdio>
#include <random>
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

static Real node_value(const GNode& m) {
  const auto c = m.template Coord<Real>();
  return 1.0 + 1e3 * m.Depth() + 137.0 * c[0] + 17.0 * c[1] + 2.0 * c[2];
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const sctl::Comm comm = sctl::Comm::World();
    const long rank = comm.Rank(), np = comm.Size();
    const sctl::Long N = (argc > 1 ? atol(argv[1]) : 50000);
    int fail = 0;
    std::mt19937_64 rng(9 + 31 * rank);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> x(N * kDim); for (auto& v : x) v = U(rng);
    gpu_tree::DeviceVector<Real> cd(x.begin(), x.end());
    sctl::Vector<Real> xs(N * kDim); for (sctl::Long i = 0; i < N*kDim; i++) xs[i] = x[i];

    GT gt(comm); sctl::Tree<kDim> st(comm);
    gt.UpdateRefinement(cd, 32, true, sctl::Periodicity::NONE, 1);
    st.UpdateRefinement(xs, 32, true, sctl::Periodicity::NONE, 1);

    // one value per owned node, zero items on ghosts
    sctl::Long gb, ge; gt.GetOwnedRange(gb, ge);
    thrust::host_vector<GNode> gmid(gt.GetNodeMID());
    const sctl::Long Ng = (sctl::Long)gmid.size();
    sctl::Vector<long> gcnt(Ng); std::vector<Real> gval;
    for (sctl::Long i = 0; i < Ng; i++) { gcnt[i] = (i>=gb && i<ge) ? 1 : 0; if (gcnt[i]) gval.push_back(node_value(gmid[i])); }
    gpu_tree::DeviceVector<Real> gvd(gval.begin(), gval.end());
    gt.AddData("u", gvd, gcnt);
    gt.AddData("v", gvd, gcnt);

    const auto& smid = st.GetNodeMID(); const auto& sattr = st.GetNodeAttr();
    sctl::Vector<long> scnt(smid.Dim()); sctl::Vector<Real> sval;
    for (sctl::Long i = 0; i < smid.Dim(); i++) { scnt[i] = sattr[i].Ghost ? 0 : 1; if (scnt[i]) sval.PushBack(node_value(smid[i])); }
    st.AddData("u", sval, scnt);
    st.AddData("v", sval, scnt);

    // Both start from the same clean state -- ghosts carry no data -- so the reduce phase has
    // nothing to add and ReduceBroadcast must agree with Broadcast node for node.
    for (int mode = 0; mode < 2; mode++) {
      const char* what = mode ? "ReduceBroadcast" : "Broadcast";
      const char* nm = mode ? "v" : "u";
      if (mode) { gt.ReduceBroadcast<Real>(nm); st.ReduceBroadcast<Real>(nm); }
      else      { gt.Broadcast<Real>(nm);       st.Broadcast<Real>(nm); }

      gpu_tree::DeviceVector<Real> gd; sctl::Vector<long> gc; gt.GetData(gd, gc, nm);
      thrust::host_vector<Real> gh(gd);
      sctl::Vector<Real> sd; sctl::Vector<long> sc; st.GetData(sd, sc, nm);
      thrust::host_vector<GNode> gm(gt.GetNodeMID());

      long gfill = 0, sfill = 0, gbad = 0, sbad = 0, k = 0;
      for (sctl::Long i = 0; i < gc.Dim(); i++) { if (!gc[i]) continue; gfill++; if (gh[k] != node_value(gm[i])) gbad++; k++; }
      k = 0;
      for (sctl::Long i = 0; i < sc.Dim(); i++) { if (!sc[i]) continue; sfill++; if (sd[k] != node_value(smid[i])) sbad++; k++; }
      long gf = 0, sf = 0;
      comm.Allreduce(sctl::Ptr2ConstItr<long>(&gfill,1), sctl::Ptr2Itr<long>(&gf,1), 1, sctl::CommOp::SUM);
      comm.Allreduce(sctl::Ptr2ConstItr<long>(&sfill,1), sctl::Ptr2Itr<long>(&sf,1), 1, sctl::CommOp::SUM);
      fail += (gbad != 0);
      if (!rank) printf("  %-16s nodes carrying data: gpu=%ld sctl=%ld   wrong values: gpu=%ld sctl=%ld\n",
                        what, gf, sf, gbad, sbad);
    }
    if (!rank) printf("%s (%d mismatch(es))\n", fail ? "FAIL" : "PASS", fail);
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
