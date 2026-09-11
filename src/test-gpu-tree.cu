/**
 * @file test-gpu-tree.cu
 * @brief gpu_tree against sctl on every backend.
 *
 * The same points go to both libraries. Node sets, counts and values are compared as the global
 * concatenation of each rank's owned nodes, since the two libraries cut the node set between ranks
 * differently; flags, lists and the partition compare node for node at one rank, where the cuts
 * coincide. Particle data must come back in the caller's order from both, bit for bit. Broadcast
 * and ReduceBroadcast are checked against what the rank boundaries imply, which does not depend on
 * either library, since where the ranks divide is a choice each of them makes for itself.
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
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
#include <thrust/copy.h>
#include "sctl/tree.hpp"
#include "sctl/tree.txx"
#include "sctl/experimental/gpu-tree.hpp"

using sctl::Integer;
using sctl::Long;
using sctl::Comm;

template <class Real, Integer DIM, template <class...> class DevVec> Long test_vs_sctl() {
  using NodeT = sctl::Morton<DIM>;
  using GT = gpu_tree::GPUTree<Real, DIM, DevVec>;
  using PT = gpu_tree::PtTree<Real, DIM, DevVec>;
  const Comm& comm = Comm::World();
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
    b = 0;
    while (b < attr.Dim() && attr[b].Ghost) b++;
    e = b;
    while (e < attr.Dim() && !attr[e].Ghost) e++;
  };
  const auto node_value = [](const NodeT& m, Real s) {  // a function of the node alone, so both libraries agree
    const auto c = m.template Coord<Real>();
    Real v = s + 1e3 * m.Depth();
    for (Integer k = 0; k < DIM; k++) v += (137.0 / (k + 1)) * c[k];
    return v;
  };

  std::mt19937_64 rng(7 + 1000 * rank);
  std::uniform_real_distribution<Real> U(0, 1);
  sctl::Vector<Real> x(N * DIM), y(N * DIM);
  for (auto& v : x) v = U(rng);
  for (auto& v : y) {  // clustered, for the refinements
    const Real u = U(rng);
    v = 0.5 + 0.45 * u * u * u;
  }
  const DevVec<Real> xd(x.begin(), x.end()), yd(y.begin(), y.end());

  { // node set, and at one rank flags, lists and partition, for every periodicity mask with and without 2:1 balance
    Long bad_set = 0, bad_local = 0;
    for (sctl::PeriodicityT m = 0; m < (1 << DIM); m++) for (Integer b21 = 0; b21 <= 1; b21++) {
      const auto per = static_cast<sctl::Periodicity>(m);
      const Integer halo = (b21 ? 1 : -1);
      GT gt(comm);
      gt.UpdateRefinement(xd, 32, b21, per, halo);
      sctl::Tree<DIM> st(comm);
      st.UpdateRefinement(x, 32, b21, per, halo);
      Long gb, ge, sb, se;
      gt.GetOwnedRange(gb, ge);
      owned(st, sb, se);
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
        const auto& gp = gt.GetPartitionMID();
        const auto& sp = st.GetPartitionMID();
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

  { // Coincident codes: where more than M particles share one, the node sets are allowed to differ
    // (see the GPUTree doc comment), so check the property that does hold of both -- no leaf holds
    // more than M particles unless every one of them has the same code, when no split would help.
    const Long M = 32, sites = 64;
    sctl::Vector<Real> c(N * DIM);
    for (Long i = 0; i < N; i++) {  // N/sites particles per site, all exactly coincident
      std::mt19937_64 r(i / (N / sites));
      std::uniform_real_distribution<Real> Us(0, 1);
      for (Integer k = 0; k < DIM; k++) c[i * DIM + k] = Us(r);
    }
    sctl::Vector<sctl::MortonCode<DIM>> code(N);
    for (Long i = 0; i < N; i++) code[i] = sctl::MortonCode<DIM>(&c[i * DIM]);
    std::sort(code.begin(), code.end(), [](const sctl::MortonCode<DIM>& a, const sctl::MortonCode<DIM>& b) { return a < b; });
    const sctl::Vector<sctl::MortonCode<DIM>> all = gather(code);  // every rank's codes, for the counts

    const DevVec<Real> cd(c.begin(), c.end());
    GT gt(comm);
    gt.UpdateRefinement(cd, M, false, sctl::Periodicity::NONE, -1);
    const sctl::Vector<NodeT> gmid = to_host(gt.GetNodeMID());
    Long gb, ge;
    gt.GetOwnedRange(gb, ge);

    Long bad = 0;
    for (Long i = gb; i < ge; i++) {
      const bool leaf = (i + 1 < gmid.Dim()) ? !gmid[i].isAncestor(gmid[i + 1]) : true;
      if (!leaf) continue;
      const Long j0 = std::lower_bound(all.begin(), all.end(), gmid[i].mid) - all.begin();
      const Long j1 = std::lower_bound(all.begin(), all.end(), gmid[i].Next().mid) - all.begin();
      if (j1 - j0 <= M) continue;
      bad += (all[j0] < all[j1 - 1]) || (all[j1 - 1] < all[j0]);  // over M and splittable
    }
    check("a leaf over M particles holds one code, which no split separates", bad);
  }

  { // named node data follows a refinement to other points, compared with sctl::Tree
    GT gt(comm);
    sctl::Tree<DIM> st(comm);
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(x, 32, true, sctl::Periodicity::NONE, 0);
    {
      const sctl::Vector<NodeT> gmid = to_host(gt.GetNodeMID());
      const auto& smid = st.GetNodeMID();
      sctl::ScratchBuf<Real> gv(gmid.Dim());
      sctl::Vector<Real> sv(smid.Dim());
      sctl::Vector<Long> gc(gmid.Dim()), sc(smid.Dim());
      for (Long i = 0; i < gmid.Dim(); i++) {
        gc[i] = 1;
        gv[i] = node_value(gmid[i], 0);
      }
      for (Long i = 0; i < smid.Dim(); i++) {
        sc[i] = 1;
        sv[i] = node_value(smid[i], 0);
      }
      gt.AddData("f", DevVec<Real>(gv.begin(), gv.end()), gc);
      st.AddData("f", sv, sc);
    }
    gt.UpdateRefinement(yd, 40, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(y, 40, true, sctl::Periodicity::NONE, 0);
    Long gb, ge, sb, se;
    gt.GetOwnedRange(gb, ge);
    owned(st, sb, se);
    gpu_tree::DataView<const Real, DevVec> gd;
    sctl::Vector<Long> gcnt;
    gt.GetData(gd, gcnt, "f");
    sctl::Vector<Real> sd;
    sctl::Vector<Long> scnt;
    st.GetData(sd, scnt, "f");
    const sctl::Vector<Real> gh = to_host(gd);
    const sctl::Vector<Long> gco(ge - gb, gcnt.begin() + gb, false), sco(se - sb, scnt.begin() + sb, false);  // views of the owned slices
    const Long goff = sctl::omp_par::reduce(gcnt.begin(), gb), soff = sctl::omp_par::reduce(scnt.begin(), sb);
    const Long gn = sctl::omp_par::reduce(gco.begin(), gco.Dim()), sn = sctl::omp_par::reduce(sco.begin(), sco.Dim());
    const sctl::Vector<Real> gv(gn, (sctl::Iterator<Real>)gh.begin() + goff, false), sv(sn, sd.begin() + soff, false);
    check("node data migrated by a refinement equals sctl::Tree's (counts)", !equal(gather(gco), gather(sco)));
    check("node data migrated by a refinement equals sctl::Tree's (values)", !equal(gather(gv), gather(sv)));
  }

  { // Broadcast and ReduceBroadcast, with data on every node so that shared nodes really do reduce
    //
    // Data on owned nodes alone leaves no node carrying data on two ranks, so the reduce half sums
    // nothing. Here the ghosts get data too. Counts and values are functions of the node, not of
    // its index, so the two libraries agree on the layout despite cutting the node set differently.
    //
    // The expected values follow from the rank boundaries alone rather than from either library.
    // ReduceBroadcast sums onto the owner the partial values held by the ranks that share a node,
    // and the nodes a rank shares with earlier ranks are exactly the strict ancestors of its first
    // node. Every rank holds those, so an owned node comes back scaled by one plus the number of
    // other ranks whose first node it is an ancestor of. Which nodes those are depends on where
    // the ranks divide, and the two libraries choose different boundaries for the same node set,
    // so their reduced values differ while both are right. Only Broadcast, which leaves owned
    // values alone, may be compared between the two directly.
    constexpr Long dof = 3;
    const auto node_cnt = [](const NodeT& m) -> Long {
      return 1 + (Long)((m.mid.GetIntKey() >> 7) % 3);  // 1, 2 or 3 items on this node
    };
    const auto node_val = [](const NodeT& m, Long item, Long k) {
      const auto c = m.template Coord<Real>();
      Real v = 1 + 17 * item + 3 * k + 1e3 * m.Depth();
      for (Integer d = 0; d < DIM; d++) v += (137.0 / (d + 1)) * c[d];
      return v;
    };
    const auto ndiff = [](const auto& a, const auto& b) {
      if (a.Dim() != b.Dim()) return std::max(a.Dim(), b.Dim());
      Long n = 0;
      for (Long i = 0; i < a.Dim(); i++) n += !(a[i] == b[i]);
      return n;
    };
    // the values a rank puts on its own copy of each node; a ghost slot may instead be given a
    // marker that Broadcast has to replace, which node_val's positive values can never be
    const auto fill = [&node_cnt, &node_val](const sctl::Vector<NodeT>& mid, Long b, Long e, bool stale, sctl::Vector<Long>& cnt) {
      cnt.ReInit(mid.Dim());
      Long tot = 0;
      for (Long i = 0; i < mid.Dim(); i++) {
        cnt[i] = node_cnt(mid[i]);
        tot += cnt[i];
      }
      sctl::Vector<Real> val(tot * dof);
      for (Long i = 0, o = 0; i < mid.Dim(); i++) {
        const bool marked = stale && (i < b || i >= e);
        for (Long j = 0; j < cnt[i]; j++, o++) {
          for (Long k = 0; k < dof; k++) val[o * dof + k] = (marked ? -1 : node_val(mid[i], j, k));
        }
      }
      return val;
    };
    const auto expect = [&node_cnt, &node_val, &comm, np, rank](const sctl::Vector<NodeT>& owned_mid, bool reduce) {
      sctl::ScratchBuf<NodeT> mins(np);  // rank p's first owned node
      comm.Allgather((sctl::ConstIterator<NodeT>)owned_mid.begin(), 1, mins.begin(), 1);
      Long tot = 0;
      for (Long i = 0; i < owned_mid.Dim(); i++) tot += node_cnt(owned_mid[i]);
      sctl::Vector<Real> val(tot * dof);
      for (Long i = 0, o = 0; i < owned_mid.Dim(); i++) {
        Long mult = 1;
        if (reduce) {
          for (Long p = 0; p < np; p++) mult += (p != rank && owned_mid[i].isAncestor(mins[p]));
        }
        for (Long j = 0; j < node_cnt(owned_mid[i]); j++, o++) {
          for (Long k = 0; k < dof; k++) val[o * dof + k] = mult * node_val(owned_mid[i], j, k);
        }
      }
      return val;
    };
    // every node, owned and ghost, must hold the value its owner holds
    const auto owners_values = [&node_cnt, &node_val](const sctl::Vector<NodeT>& mid, const sctl::Vector<Long>& cnt, const sctl::Vector<Real>& val) {
      Long bad = 0, o = 0;
      for (Long i = 0; i < mid.Dim(); i++) {
        bad += (cnt[i] != node_cnt(mid[i]));
        for (Long j = 0; j < cnt[i]; j++, o++) {
          for (Long k = 0; k < dof; k++) bad += (val[o * dof + k] != node_val(mid[i], j, k));
        }
      }
      return bad;
    };
    const auto owned_vals_gpu = [&to_host](GT& t, const char* name, sctl::Vector<NodeT>& mid) {
      Long b, e;
      t.GetOwnedRange(b, e);
      gpu_tree::DataView<const Real, DevVec> d;
      sctl::Vector<Long> cnt;
      t.GetData(d, cnt, name);
      const sctl::Vector<Real> h = to_host(d);
      const sctl::Vector<NodeT> all = to_host(t.GetNodeMID());
      mid.ReInit(e - b);
      for (Long i = b; i < e; i++) mid[i - b] = all[i];
      const Long off = sctl::omp_par::reduce(cnt.begin(), b) * dof;
      const Long n = sctl::omp_par::reduce(cnt.begin() + b, e - b) * dof;
      sctl::Vector<Real> out(n);
      for (Long i = 0; i < n; i++) out[i] = h[off + i];
      return out;
    };
    const auto owned_vals_sctl = [&owned](sctl::Tree<DIM>& t, const char* name, sctl::Vector<NodeT>& mid) {
      Long b, e;
      owned(t, b, e);
      sctl::Vector<Real> d;
      sctl::Vector<Long> cnt;
      t.GetData(d, cnt, name);
      const auto& all = t.GetNodeMID();
      mid.ReInit(e - b);
      for (Long i = b; i < e; i++) mid[i - b] = all[i];
      const Long off = sctl::omp_par::reduce(cnt.begin(), b) * dof;
      const Long n = sctl::omp_par::reduce(cnt.begin() + b, e - b) * dof;
      sctl::Vector<Real> out(n);
      for (Long i = 0; i < n; i++) out[i] = d[off + i];
      return out;
    };

    for (Integer halo = 0; halo <= 1; halo++) {
      for (Integer mode = 0; mode < 2; mode++) {
        const char* op = (mode ? "ReduceBroadcast" : "Broadcast");
        GT gt(comm);
        sctl::Tree<DIM> st(comm);
        gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, halo);
        st.UpdateRefinement(x, 32, true, sctl::Periodicity::NONE, halo);
        {
          Long b, e;
          gt.GetOwnedRange(b, e);
          sctl::Vector<Long> cnt;
          const sctl::Vector<Real> v = fill(to_host(gt.GetNodeMID()), b, e, false, cnt);
          gt.AddData("d", DevVec<Real>(v.begin(), v.end()), cnt);
        }
        {
          Long b, e;
          owned(st, b, e);
          sctl::Vector<Long> cnt;
          sctl::Vector<Real> v = fill(st.GetNodeMID(), b, e, false, cnt);
          st.AddData("d", v, cnt);
        }
        if (mode) {
          gt.template ReduceBroadcast<Real>("d");
          st.template ReduceBroadcast<Real>("d");
        } else {
          gt.template Broadcast<Real>("d");
          st.template Broadcast<Real>("d");
        }
        char msg[160];
        sctl::Vector<NodeT> gmid, smid;
        const sctl::Vector<Real> g = owned_vals_gpu(gt, "d", gmid), s = owned_vals_sctl(st, "d", smid);
        std::snprintf(msg, sizeof msg, "%s, halo=%d: gpu_tree owned values are what the contract asks for", op, (int)halo);
        check(msg, ndiff(g, expect(gmid, mode)));
        std::snprintf(msg, sizeof msg, "%s, halo=%d: sctl::Tree owned values are what the contract asks for", op, (int)halo);
        check(msg, ndiff(s, expect(smid, mode)));
        if (!mode) {  // see above: only Broadcast's result is independent of where the ranks divide
          std::snprintf(msg, sizeof msg, "%s, halo=%d: the two libraries agree", op, (int)halo);
          check(msg, ndiff(gather(g), gather(s)));
        }

        // the ghosts already hold the owners' values and Broadcast does not accumulate
        gt.template Broadcast<Real>("d");
        st.template Broadcast<Real>("d");
        sctl::Vector<NodeT> gmid2, smid2;
        const sctl::Vector<Real> g2 = owned_vals_gpu(gt, "d", gmid2), s2 = owned_vals_sctl(st, "d", smid2);
        std::snprintf(msg, sizeof msg, "%s, halo=%d: a further Broadcast changes nothing", op, (int)halo);
        check(msg, ndiff(g, g2) + ndiff(s, s2));
      }
    }

    { // a ghost slot holding something stale must come back holding the owner's values
      GT gt(comm);
      sctl::Tree<DIM> st(comm);
      gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 1);
      st.UpdateRefinement(x, 32, true, sctl::Periodicity::NONE, 1);
      {
        Long b, e;
        gt.GetOwnedRange(b, e);
        sctl::Vector<Long> cnt;
        const sctl::Vector<Real> v = fill(to_host(gt.GetNodeMID()), b, e, true, cnt);
        gt.AddData("d", DevVec<Real>(v.begin(), v.end()), cnt);
        gt.template Broadcast<Real>("d");
        gpu_tree::DataView<const Real, DevVec> d;
        sctl::Vector<Long> cnt_out;
        gt.GetData(d, cnt_out, "d");
        check("Broadcast: gpu_tree replaces stale ghost values with the owner's", owners_values(to_host(gt.GetNodeMID()), cnt_out, to_host(d)));
      }
      {
        Long b, e;
        owned(st, b, e);
        sctl::Vector<Long> cnt;
        sctl::Vector<Real> v = fill(st.GetNodeMID(), b, e, true, cnt);
        st.AddData("d", v, cnt);
        st.template Broadcast<Real>("d");
        sctl::Vector<Real> d;
        sctl::Vector<Long> cnt_out;
        st.GetData(d, cnt_out, "d");
        check("Broadcast: sctl::Tree replaces stale ghost values with the owner's", owners_values(st.GetNodeMID(), cnt_out, d));
      }
    }
  }

  { // particles: two groups, data filled through the view, data added after a repartition, re-cut of a re-cut
    PT gt(comm);
    sctl::PtTree<Real, DIM> st(comm);
    const Long N2 = N / 3;
    sctl::Vector<Real> f(N * 2), g(N2), h(N);
    for (Long i = 0; i < N; i++) {
      f[2 * i] = 1e6 * rank + i;
      f[2 * i + 1] = -(Real)i;
      h[i] = 0.5 * i + rank;
    }
    for (Long i = 0; i < N2; i++) g[i] = 7e5 * rank + 3 * i;
    const DevVec<Real> fd(f.begin(), f.end()), gd(g.begin(), g.end()), hd(h.begin(), h.end()), y2d(y.begin(), y.begin() + N2 * DIM);
    const sctl::Vector<Real> y2s(N2 * DIM, y.begin(), false);  // y's first N2 particles
    const auto round_trip = [&gt, &st, &to_host](const char* name, const sctl::Vector<Real>& ref) {  // both libraries return the caller's array
      DevVec<Real> go;
      gt.GetParticleData(go, name);
      const sctl::Vector<Real> gh = to_host(go);
      sctl::Vector<Real> so;
      st.GetParticleData(so, name);
      Long bad = (gh.Dim() != ref.Dim()) + (so.Dim() != ref.Dim());
      if (!bad) for (Long i = 0; i < ref.Dim(); i++) bad += (gh[i] != ref[i]) + (so[i] != ref[i]);
      return bad;
    };
    const auto particle_total = [&gt, &st, &comm]() {  // particles held by the tree, summed over ranks, the same in both
      gpu_tree::DataView<const Real, DevVec> gv;
      sctl::Vector<Long> gc;
      gt.GetData(gv, gc, "pt");
      sctl::Vector<Real> sv;
      sctl::Vector<Long> sc;
      st.GetData(sv, sc, "pt");
      sctl::StaticArray<Long, 2> l{sctl::omp_par::reduce(gc.begin(), gc.Dim()), sctl::omp_par::reduce(sc.begin(), sc.Dim())}, t{0, 0};
      comm.Allreduce((sctl::ConstIterator<Long>)l, (sctl::Iterator<Long>)t, 2, sctl::CommOp::SUM);
      return (Long)(t[0] != t[1]);
    };
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(x, 32, true, sctl::Periodicity::NONE, 0);
    gt.AddParticles("pt", xd);
    gt.AddParticleData("f", "pt", fd);
    st.AddParticles("pt", x);
    st.AddParticleData("f", "pt", f);
    gt.AddParticles("q", y2d);
    gt.AddParticleData("g", "q", gd);
    st.AddParticles("q", y2s);
    st.AddParticleData("g", "q", g);
    // unwritten; filled below from f's storage, in node order
    gt.AddParticleData("u", "pt", 2);
    st.AddParticleData("u", "pt", 2);
    {
      gpu_tree::DataView<const Real, DevVec> vf;
      gpu_tree::DataView<Real, DevVec> vu;
      sctl::Vector<Long> c1, c2;
      gt.GetData(vf, c1, "f");
      gt.GetData(vu, c2, "u");
      thrust::copy(vf.begin(), vf.end(), vu.begin());
    }
    {
      sctl::Vector<Real> vf, vu;
      sctl::Vector<Long> c1, c2;
      st.GetData(vf, c1, "f");
      st.GetData(vu, c2, "u");
      std::copy(vf.begin(), vf.end(), vu.begin());
    }
    check("particle data round-trips (two groups, one set filled through the view)", round_trip("f", f) + round_trip("g", g) + round_trip("u", f) + particle_total());
    gt.UpdateRefinement(yd, 25, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(y, 25, true, sctl::Periodicity::NONE, 0);
    // added after the repartition: the forward scatter with its re-cut stage
    gt.AddParticleData("h", "pt", hd);
    st.AddParticleData("h", "pt", h);
    check("particle data round-trips after a repartition, including data added after it", round_trip("f", f) + round_trip("g", g) + round_trip("u", f) + round_trip("h", h) + particle_total());
    gt.UpdateRefinement(xd, 32, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(x, 32, true, sctl::Periodicity::NONE, 0);
    check("particle data round-trips after a second repartition", round_trip("f", f) + round_trip("h", h) + particle_total());
    // A Broadcast fills the group's ghost slots, so its counts stop being the owned-item counts. A
    // data set added after that is laid out against those counts, with its items in the owned
    // window; one added before keeps the owned-only layout. Both must round-trip, and still do once
    // a refinement has moved them.
    gt.template Broadcast<Real>("pt");
    st.template Broadcast<Real>("pt");
    gt.AddParticleData("b", "pt", hd);
    st.AddParticleData("b", "pt", h);
    check("particle data round-trips for a set added after a Broadcast filled the ghost slots",
          round_trip("b", h) + round_trip("f", f) + round_trip("h", h));
    gt.UpdateRefinement(yd, 25, true, sctl::Periodicity::NONE, 0);
    st.UpdateRefinement(y, 25, true, sctl::Periodicity::NONE, 0);
    check("that set round-trips after the next refinement",
          round_trip("b", h) + round_trip("f", f) + particle_total());
    {
      const sctl::Vector<NodeT> m = to_host(gt.GetNodeMID());
      Long bad = 0;
      for (Long i = 1; i < m.Dim(); i++) {
        const bool less = m[i - 1].mid < m[i].mid, same = !(m[i - 1].mid < m[i].mid) && !(m[i].mid < m[i - 1].mid);
        bad += !(less || (same && m[i - 1].Depth() <= m[i].Depth()));
      }
      check("node list is sorted by (code, depth)", bad);
    }
  }
  return fails;
}

/**
 * A tree whose walk produces more nodes than `buildTreeCpuChunked` reserves room for: clumps of
 * identical coordinates, more than `M` to a clump, which split at every level down to MAX_DEPTH
 * because duplicates never separate.
 *
 * Checks the node list is the complete linear tree it is meant to be -- sorted, its leaves covering
 * the domain end to end with no gap or overlap. A node lost, duplicated or written out of order past
 * the reserved room breaks that. Whether the nodes past the reservation stay in the pool scratch or
 * go to the heap depends on how much room the thread's chunk has; the tree is the same either way.
 */
template <class Real, Integer DIM, template <class...> class DevVec> Long test_walk_overflow(const char* what) {
  using NodeT = sctl::Morton<DIM>;
  const Comm& self = Comm::Self();
  const Long M = 512, per = M + 1, clumps = 400, N = clumps * per;

  std::vector<Real> X(N * DIM);
  for (Long g = 0; g < clumps; g++) {
    Real c[DIM];
    for (Integer d = 0; d < DIM; d++) c[d] = (Real)((g * 7919 + d * 104729) % 100003) / (Real)100003;
    for (Long j = 0; j < per; j++)
      for (Integer d = 0; d < DIM; d++) X[((g * per + j) * DIM) + d] = c[d];
  }
  DevVec<Real> Xd(X.size());
  thrust::copy(X.begin(), X.end(), Xd.begin());

  gpu_tree::GPUTree<Real, DIM, DevVec> tree(self);
  tree.UpdateRefinement(Xd, M, false, sctl::Periodicity::NONE, 0);

  const auto to_host = [](const auto& d) {
    sctl::Vector<std::remove_const_t<typename std::decay_t<decltype(d)>::value_type>> h((Long)d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
  };
  const auto mid = to_host(tree.GetNodeMID());
  const auto attr = to_host(tree.GetNodeAttr());

  // Morton equality takes the depth in too; here only where a box starts matters.
  const auto same_box = [](const NodeT& a, const NodeT& b) { return !(a.mid < b.mid) && !(b.mid < a.mid); };
  Long bad = 0, leaves = 0;
  NodeT reach = NodeT();  // how far the cover has got; leaves tile in order, each starting where the last ended
  for (Long i = 0; i < mid.Dim(); i++) {
    if (i && !(mid[i - 1] < mid[i])) bad++;
    if (!attr[i].Leaf) continue;
    if (!same_box(mid[i], reach)) bad++;
    reach = mid[i].Next();
    leaves++;
  }
  if (!same_box(reach, NodeT().Next())) bad++;
  if (leaves <= clumps) bad++;  // every clump had to be resolved into a box of its own
  printf("  %-72s %s\n", what, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
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
    if (root) {  // the host walk counts and writes in one pass, so it is the one that can exceed its estimate
      printf("walk overflow\n");
      fails += test_walk_overflow<double, 3, gpu_tree::HostVector>("HostVector: the cover is complete");
      fails += test_walk_overflow<double, 3, std::vector>("std::vector: the cover is complete");
    }
    SCTL_ASSERT_MSG(fails == 0, "test-gpu-tree: failures above");
  }
  sctl::Comm::MPI_Finalize();
  return 0;
}
