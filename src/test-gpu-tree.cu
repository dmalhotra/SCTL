/**
 * @file test-gpu-tree.cu
 * @brief gpu_tree against sctl on every backend.
 *
 * The same points go to both libraries. Node sets, counts and values are compared as the global
 * concatenation of each rank's owned nodes, since the two libraries cut the node set between ranks
 * differently; flags, lists and the partition compare node for node at one rank, where the cuts
 * coincide. Particle data must come back in the caller's order from both, bit for bit.
 *
 * Build (`make gpu`, or by hand with a CUDA-aware MPI):
 *
 *     nvcc -ccbin mpicxx -x cu -std=c++17 -O3 -arch=native -rdc=true --expt-relaxed-constexpr \
 *          -Xcompiler -fopenmp -DSCTL_HAVE_MPI -DSCTL_MAX_DEPTH=20 \
 *          -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -I include src/test-gpu-tree.cu -o bin/test-gpu-tree -ldl
 *
 * Run:
 *
 *     mpirun -np 4 bin/test-gpu-tree
 */
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
#include <thrust/copy.h>
#include <sctl/tree.hpp>
#include <sctl/tree.txx>
#include "sctl/experimental/gpu-tree.hpp"

using sctl::Integer;
using sctl::Long;
using sctl::Comm;

template <class Real, Integer DIM, template <class...> class DevVec> Long test_vs_sctl() {
  using NodeT = sctl::Morton<DIM>;
  using GT = gpu_tree::GPUTree<Real, DIM, DevVec>;
  using PT = gpu_tree::PtTree<Real, DIM, DevVec>;
  const Comm comm = Comm::World();
  const Long np = comm.Size(), rank = comm.Rank(), N = 20000;
  Long fails = 0;
  const auto check = [&comm, &fails, rank](const char* what, Long bad) {
    Long tot = 0;
    comm.Allreduce(sctl::Ptr2ConstItr<Long>(&bad, 1), sctl::Ptr2Itr<Long>(&tot, 1), 1, sctl::CommOp::SUM);
    fails += (tot != 0);
    if (!rank) printf("  %-72s %s\n", what, tot ? "FAIL" : "ok");
  };
  const auto to_host = [](const auto& d) {  // any backend container or view to an sctl::Vector
    sctl::Vector<std::remove_const_t<typename std::decay_t<decltype(d)>::value_type>> h((Long)d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
  };
  const auto gather = [&comm, np](const auto& loc) {  // concatenation over ranks
    using T = typename std::decay_t<decltype(loc)>::value_type;
    const Long n = loc.Dim();
    sctl::ScratchBuf<Long> cnt(np), dsp(np);
    comm.Allgather(sctl::Ptr2ConstItr<Long>(&n, 1), 1, cnt.begin(), 1);
    std::exclusive_scan(cnt.begin(), cnt.end(), dsp.begin(), Long(0));
    sctl::Vector<T> out(dsp[np - 1] + cnt[np - 1]);
    comm.Allgatherv(loc.begin(), n, out.begin(), cnt.begin(), dsp.begin());
    return out;
  };
  const auto equal = [](const auto& a, const auto& b) {
    if (a.Dim() != b.Dim()) return false;
    for (Long i = 0; i < a.Dim(); i++) if (!(a[i] == b[i])) return false;
    return true;
  };
  const auto owned = [](const sctl::Tree<DIM>& st, Long& b, Long& e) {  // sctl's owned range: its non-ghost nodes
    const auto& attr = st.GetNodeAttr();
    b = 0; while (b < attr.Dim() && attr[b].Ghost) b++;
    e = b; while (e < attr.Dim() && !attr[e].Ghost) e++;
  };
  const auto node_value = [](const NodeT& m, Real s) {  // a function of the node alone, so both libraries agree
    const auto c = m.template Coord<Real>();
    Real v = s + 1e3 * m.Depth();
    for (Integer k = 0; k < DIM; k++) v += (137.0 / (k + 1)) * c[k];
    return v;
  };

  std::mt19937_64 rng(7 + 1000 * rank);
  std::uniform_real_distribution<Real> U(0, 1);
  std::vector<Real> x(N * DIM), y(N * DIM);
  for (auto& v : x) v = U(rng);
  for (auto& v : y) { const Real u = U(rng); v = 0.5 + 0.45 * u * u * u; }  // clustered, for the refinements
  const DevVec<Real> xd(x.begin(), x.end()), yd(y.begin(), y.end());
  const sctl::Vector<Real> xs(N * DIM, sctl::Ptr2Itr<Real>(x.data(), N * DIM), false), ys(N * DIM, sctl::Ptr2Itr<Real>(y.data(), N * DIM), false);  // views of x, y

  { // node set, and at one rank flags, lists and partition, for every periodicity mask with and without 2:1 balance
    Long bad_set = 0, bad_local = 0;
    for (sctl::PeriodicityT m = 0; m < (1 << DIM); m++) for (Integer b21 = 0; b21 <= 1; b21++) {
      const auto per = static_cast<sctl::Periodicity>(m);
      const Integer halo = (b21 ? 1 : -1);
      GT gt(comm); gt.UpdateRefinement(xd, 32, b21, per, halo);
      sctl::Tree<DIM> st(comm); st.UpdateRefinement(xs, 32, b21, per, halo);
      Long gb, ge, sb, se;
      gt.GetOwnedRange(gb, ge); owned(st, sb, se);
      const sctl::Vector<NodeT> gmid = to_host(gt.GetNodeMID());
      const auto& smid = st.GetNodeMID();
      const sctl::Vector<NodeT> g(ge - gb, (sctl::Iterator<NodeT>)gmid.begin() + gb, false), s(se - sb, (sctl::Iterator<NodeT>)smid.begin() + sb, false);  // views of the owned slices
      bad_set += !equal(gather(g), gather(s));
      if (np == 1) {
        const sctl::Vector<typename GT::NodeAttr> attr = to_host(gt.GetNodeAttr());
        const auto& sattr = st.GetNodeAttr();
        const auto& L = gt.GetNodeLists();
        const sctl::Vector<Long> par = to_host(L.parent), ch = to_host(L.child), nb = to_host(L.nbr);
        const auto& sl = st.GetNodeLists();
        const auto& gp = gt.GetPartitionMID(); const auto& sp = st.GetPartitionMID();
        Long bad = !equal(gmid, smid) + (gp.Dim() != 1 || sp.Dim() != 1 || !(gp[0] == sp[0]));
        constexpr Integer nchild = 1 << DIM, nnbr = sctl::pow<DIM, Integer>(3);
        if (gmid.Dim() == smid.Dim()) for (Long i = 0; i < smid.Dim(); i++) {
          bad += (attr[i].Leaf != sattr[i].Leaf) + (attr[i].Ghost != sattr[i].Ghost) + (par[i] != sl[i].parent);
          for (Integer k = 0; k < nchild; k++) bad += (ch[i * nchild + k] != sl[i].child[k]);
          for (Integer k = 0; k < nnbr; k++) bad += (nb[i * nnbr + k] != sl[i].nbr[k]);
        }
        bad_local += bad;
      }
    }
    check("node set equals sctl::Tree's (every periodicity mask, balance on and off)", bad_set);
    if (np == 1) check("flags, node lists and partition equal sctl::Tree's node for node", bad_local);
  }

  { // named node data follows a refinement to other points, compared with sctl::Tree
    GT gt(comm); sctl::Tree<DIM> st(comm);
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(xs, 32, true, sctl::Periodicity::NONE, 0);
    { const sctl::Vector<NodeT> gmid = to_host(gt.GetNodeMID()); const auto& smid = st.GetNodeMID();
      sctl::ScratchBuf<Real> gv(gmid.Dim()); sctl::Vector<Real> sv(smid.Dim());
      sctl::Vector<Long> gc(gmid.Dim()), sc(smid.Dim());
      for (Long i = 0; i < gmid.Dim(); i++) { gc[i] = 1; gv[i] = node_value(gmid[i], 0); }
      for (Long i = 0; i < smid.Dim(); i++) { sc[i] = 1; sv[i] = node_value(smid[i], 0); }
      gt.AddData("f", DevVec<Real>(gv.begin(), gv.end()), gc);
      st.AddData("f", sv, sc); }
    gt.UpdateRefinement(yd, 40, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(ys, 40, true, sctl::Periodicity::NONE, 0);
    Long gb, ge, sb, se;
    gt.GetOwnedRange(gb, ge); owned(st, sb, se);
    gpu_tree::DataView<const Real, DevVec> gd; sctl::Vector<Long> gcnt; gt.GetData(gd, gcnt, "f");
    sctl::Vector<Real> sd; sctl::Vector<Long> scnt; st.GetData(sd, scnt, "f");
    const sctl::Vector<Real> gh = to_host(gd);
    const sctl::Vector<Long> gco(ge - gb, gcnt.begin() + gb, false), sco(se - sb, scnt.begin() + sb, false);  // views of the owned slices
    const Long goff = sctl::omp_par::reduce(gcnt.begin(), gb), soff = sctl::omp_par::reduce(scnt.begin(), sb);
    const Long gn = sctl::omp_par::reduce(gco.begin(), gco.Dim()), sn = sctl::omp_par::reduce(sco.begin(), sco.Dim());
    const sctl::Vector<Real> gv(gn, (sctl::Iterator<Real>)gh.begin() + goff, false), sv(sn, sd.begin() + soff, false);
    check("node data migrated by a refinement equals sctl::Tree's (counts)", !equal(gather(gco), gather(sco)));
    check("node data migrated by a refinement equals sctl::Tree's (values)", !equal(gather(gv), gather(sv)));
  }

  { // Broadcast and ReduceBroadcast on a halo-1 tree: every node carrying data ends with its owner's value
    GT gt(comm);
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 1);
    Long gb, ge;
    gt.GetOwnedRange(gb, ge);
    const sctl::Vector<NodeT> gmid = to_host(gt.GetNodeMID());
    sctl::Vector<Long> gc(gmid.Dim()); sctl::ScratchBuf<Real> gv(ge - gb);
    for (Long i = 0; i < gmid.Dim(); i++) { gc[i] = (i >= gb && i < ge); if (gc[i]) gv[i - gb] = node_value(gmid[i], 1); }
    const DevVec<Real> gvd(gv.begin(), gv.end());
    gt.AddData("u", gvd, gc); gt.AddData("v", gvd, gc);
    for (Integer mode = 0; mode < 2; mode++) {
      const char* nm = (mode ? "v" : "u");
      if (mode) gt.template ReduceBroadcast<Real>(nm); else gt.template Broadcast<Real>(nm);
      gpu_tree::DataView<const Real, DevVec> gd; sctl::Vector<Long> gcn; gt.GetData(gd, gcn, nm);
      const sctl::Vector<Real> gh = to_host(gd);
      const sctl::Vector<NodeT> gm = to_host(gt.GetNodeMID());  // ghosts were added
      Long bad = 0;  // which ghosts carry data depends on the cuts, so only the values are compared
      for (Long i = 0, k = 0; i < gcn.Dim(); i++) { if (gcn[i]) bad += (gh[k++] != node_value(gm[i], 1)); }
      check(mode ? "ReduceBroadcast: every node with data holds its owner's value" : "Broadcast: every node with data holds its owner's value", bad);
    }
  }

  { // particles: two groups, data filled through the view, data added after a repartition, re-cut of a re-cut
    PT gt(comm); sctl::PtTree<Real, DIM> st(comm);
    const Long N2 = N / 3;
    std::vector<Real> f(N * 2), g(N2), h(N);
    for (Long i = 0; i < N; i++) { f[2 * i] = 1e6 * rank + i; f[2 * i + 1] = -(Real)i; h[i] = 0.5 * i + rank; }
    for (Long i = 0; i < N2; i++) g[i] = 7e5 * rank + 3 * i;
    const DevVec<Real> fd(f.begin(), f.end()), gd(g.begin(), g.end()), hd(h.begin(), h.end()), y2d(y.begin(), y.begin() + N2 * DIM);
    const sctl::Vector<Real> fs(N * 2, sctl::Ptr2Itr<Real>(f.data(), N * 2), false), gs(N2, sctl::Ptr2Itr<Real>(g.data(), N2), false),
                             hs(N, sctl::Ptr2Itr<Real>(h.data(), N), false), y2s(N2 * DIM, sctl::Ptr2Itr<Real>(y.data(), N2 * DIM), false);  // views of f, g, h, y
    const auto round_trip = [&gt, &st, &to_host](const char* name, const std::vector<Real>& ref) {  // both libraries return the caller's array
      DevVec<Real> go; gt.GetParticleData(go, name);
      const sctl::Vector<Real> gh = to_host(go);
      sctl::Vector<Real> so; st.GetParticleData(so, name);
      Long bad = (gh.Dim() != (Long)ref.size()) + (so.Dim() != (Long)ref.size());
      if (!bad) for (Long i = 0; i < (Long)ref.size(); i++) bad += (gh[i] != ref[i]) + (so[i] != ref[i]);
      return bad;
    };
    const auto particle_total = [&gt, &st, &comm]() {  // particles held by the tree, summed over ranks, the same in both
      gpu_tree::DataView<const Real, DevVec> gv; sctl::Vector<Long> gc; gt.GetData(gv, gc, "pt");
      sctl::Vector<Real> sv; sctl::Vector<Long> sc; st.GetData(sv, sc, "pt");
      Long l[2] = {sctl::omp_par::reduce(gc.begin(), gc.Dim()), sctl::omp_par::reduce(sc.begin(), sc.Dim())}, t[2] = {0, 0};
      comm.Allreduce(sctl::Ptr2ConstItr<Long>(l, 2), sctl::Ptr2Itr<Long>(t, 2), 2, sctl::CommOp::SUM);
      return (Long)(t[0] != t[1]);
    };
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 0); st.UpdateRefinement(xs, 32, true, sctl::Periodicity::NONE, 0);
    gt.AddParticles("pt", xd); gt.AddParticleData("f", "pt", fd);
    st.AddParticles("pt", xs); st.AddParticleData("f", "pt", fs);
    gt.AddParticles("q", y2d); gt.AddParticleData("g", "q", gd);
    st.AddParticles("q", y2s); st.AddParticleData("g", "q", gs);
    gt.AddParticleData("u", "pt", 2); st.AddParticleData("u", "pt", 2);  // unwritten; filled below from f's storage, in node order
    { gpu_tree::DataView<const Real, DevVec> vf; gpu_tree::DataView<Real, DevVec> vu; sctl::Vector<Long> c1, c2;
      gt.GetData(vf, c1, "f"); gt.GetData(vu, c2, "u");
      thrust::copy(vf.begin(), vf.end(), vu.begin()); }
    { sctl::Vector<Real> vf, vu; sctl::Vector<Long> c1, c2;
      st.GetData(vf, c1, "f"); st.GetData(vu, c2, "u");
      std::copy(vf.begin(), vf.end(), vu.begin()); }
    check("particle data round-trips (two groups, one set filled through the view)", round_trip("f", f) + round_trip("g", g) + round_trip("u", f) + particle_total());
    gt.UpdateRefinement(yd, 25, true, sctl::Periodicity::NONE, 0); st.UpdateRefinement(ys, 25, true, sctl::Periodicity::NONE, 0);
    gt.AddParticleData("h", "pt", hd); st.AddParticleData("h", "pt", hs);  // added after the repartition: the forward scatter with its re-cut stage
    check("particle data round-trips after a repartition, including data added after it", round_trip("f", f) + round_trip("g", g) + round_trip("u", f) + round_trip("h", h) + particle_total());
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 0); st.UpdateRefinement(xs, 32, true, sctl::Periodicity::NONE, 0);
    check("particle data round-trips after a second repartition", round_trip("f", f) + round_trip("h", h) + particle_total());
    { const sctl::Vector<NodeT> m = to_host(gt.GetNodeMID()); Long bad = 0;
      for (Long i = 1; i < m.Dim(); i++) {
        const bool less = m[i - 1].mid < m[i].mid, same = !(m[i - 1].mid < m[i].mid) && !(m[i].mid < m[i - 1].mid);
        bad += !(less || (same && m[i - 1].Depth() <= m[i].Depth()));
      }
      check("node list is sorted by (code, depth)", bad); }
  }
  return fails;
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const bool root = (sctl::Comm::World().Rank() == 0);
    Long fails = 0;
    if (root) printf("HostVector backend\n");
    fails += test_vs_sctl<double, 3, gpu_tree::HostVector>();
    if (root) printf("DeviceVector backend\n");
    fails += test_vs_sctl<double, 3, gpu_tree::DeviceVector>();
    if (root) printf("std::vector backend\n");
    fails += test_vs_sctl<double, 3, std::vector>();
    SCTL_ASSERT_MSG(fails == 0, "test-gpu-tree: failures above");
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
