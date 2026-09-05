// GPUTree against sctl::Tree: the partition boundaries (GetPartitionMID), the per-node Leaf/Ghost
// flags (GetNodeAttr) and the node lists (GetNodeLists). Single rank, so the comparison is node-for-node.
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

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const sctl::Comm comm = sctl::Comm::Self();
    const sctl::Long N = (argc > 1 ? atol(argv[1]) : 200000), M = 32;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> x(N * kDim);
    for (auto& v : x) v = U(rng);

    int fail = 0;
    // every per-axis mask, so the partial ones (X|Z, ...) are covered as well as the named ones
    for (sctl::PeriodicityT m = 0; m < (1 << kDim); m++)
    for (auto per : {static_cast<sctl::Periodicity>(m)})
    for (int b21 = 0; b21 <= 1; b21++) {
      gpu_tree::DeviceVector<Real> cd(x.begin(), x.end());
      GT tr(comm);
      tr.UpdateRefinement(cd, M, b21, per, -1);
      const auto& lst_d = tr.GetNodeLists();
      const auto& part = tr.GetPartitionMID();
      thrust::host_vector<GT::NodeAttr> attr(tr.GetNodeAttr());
      thrust::host_vector<sctl::Long> lpar(lst_d.parent), lch(lst_d.child), lnbr(lst_d.nbr);
      thrust::host_vector<GNode> mid(tr.GetNodeMID());

      sctl::Vector<Real> xs(N * kDim);
      for (sctl::Long i = 0; i < N * kDim; i++) xs[i] = x[i];
      sctl::Tree<kDim> st(comm);
      st.UpdateRefinement(xs, M, b21, per, -1);
      const auto& smid = st.GetNodeMID();
      const auto& sattr = st.GetNodeAttr();
      const auto& spart = st.GetPartitionMID();
      const auto& slst = st.GetNodeLists();

      int bad_n = (mid.size() != (size_t)smid.Dim());
      int bad_leaf = 0, bad_ghost = 0, bad_par = 0, bad_ch = 0, bad_nbr = 0, bad_p2n = 0;
      if (!bad_n) for (size_t i = 0; i < mid.size(); i++) {
        if (attr[i].Leaf != sattr[i].Leaf) bad_leaf++;
        if (attr[i].Ghost != sattr[i].Ghost) bad_ghost++;
        if (lpar[i] != slst[i].parent) bad_par++;
        if ((mid[i].Depth() ? (sctl::Long)mid[i].Path2Node() : -1) != slst[i].p2n) bad_p2n++;
        for (int k = 0; k < (1 << kDim); k++) if (lch[i*(1<<kDim)+k] != slst[i].child[k]) bad_ch++;
        for (int k = 0; k < 27; k++) if (lnbr[i*27+k] != slst[i].nbr[k]) bad_nbr++;
      }
      char mask_name[4] = {'-', '-', '-', 0};
      for (sctl::Integer d = 0; d < kDim; d++) if (sctl::is_periodic(per, d)) mask_name[d] = "xyz"[d];
      const bool bad_part = !(spart.Dim() == 1 && part.Dim() == 1) || (part[0] < spart[0]) || (spart[0] < part[0]);
      printf("b21=%d per=%s nodes=%zu/%ld  leaf=%d ghost=%d p2n=%d parent=%d child=%d nbr=%d partition_%s\n",
             b21, mask_name, mid.size(), (long)smid.Dim(), bad_leaf, bad_ghost, bad_p2n, bad_par, bad_ch, bad_nbr,
             bad_part ? "MISMATCH" : "ok");
      fail += bad_n + bad_leaf + bad_ghost + bad_p2n + bad_par + bad_ch + bad_nbr + (bad_part ? 1 : 0);
    }
    printf("%s (%d mismatch(es))\n", fail ? "FAIL" : "PASS", fail);
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
