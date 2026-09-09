#ifndef _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_
#define _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_

// Included by gpu-tree.txx after its `detail` helpers (scratch_policy, local_sort_by_key,
// splitCounts, alltoallv, PersistentBuffer), which this file uses.

#include <utility>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

#include "sctl/experimental/sort-scatter.hpp"
#include "sctl/experimental/device_scratch.hpp"
#include "sctl/sort-scatter.txx"  // ensureRecut, recordRecut

namespace gpu_tree {

namespace detail_sortScatter {

// One thread per value, `n * dof` of them, so a key's block is read contiguously.
template <class T> struct GatherDofFunctor {
  const T* src;
  const Long* idx;
  T* dst;
  Long dof;
  SCTL_GPU_HD void operator()(Long e) const {
    const Long i = e / dof, k = e - i * dof;
    dst[e] = src[idx[i] * dof + k];
  }
};

/** `inv[m[i]] = i`. The only scattered write in the scheme, and it moves one index per key. */
struct InvertFunctor {
  const Long* m;
  Long* inv;
  SCTL_GPU_HD void operator()(Long i) const { inv[m[i]] = i; }
};

template <template <class...> class DevVec, class Policy>
void buildInverse(const Policy& pol, const DevVec<Long>& m, DevVec<Long>& inv, Long n) {
  inv.resize(n);
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n, InvertFunctor{
      thrust::raw_pointer_cast(m.data()), thrust::raw_pointer_cast(inv.data())});
}

using detail::resizeDiscard;  // sizing a retained buffer for output it is about to be given

/** One local stage: `dst[e] = src[map[e/dof]*dof + e%dof]` over `n*dof` values. */
template <template <class...> class DevVec, class T, class Policy>
void localMove(const Policy& pol, const T* src, T* dst, const DevVec<Long>& map, Long n, Long dof) {
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n * dof, GatherDofFunctor<T>{
      src, thrust::raw_pointer_cast(map.data()), dst, dof});
}

/** One exchange stage, between raw buffers, with the per-rank counts scaled by `dof`. */
template <template <class...> class DevVec, class T, class Policy>
void exchange(const Policy& pol, const T* src, T* dst,
              const sctl::Vector<Long>& scnt, const sctl::Vector<Long>& rcnt, Long dof, const Comm& comm) {
  const Long np = comm.Size();
  sctl::ScratchBuf<Long> sc(np), rc(np);
  for (Long r = 0; r < np; r++) {
    sc[r] = scnt[r] * dof;
    rc[r] = rcnt[r] * dof;
  }
  detail::alltoallv<DevVec>(pol, src, dst, sc, rc, (Long)sizeof(T), comm);
}

}  // namespace detail_sortScatter

template <class Key, template <class...> class DevVec>
void SortScatter<Key, DevVec>::Init(DevVec<Key> keys, const sctl::Vector<Key>& splitters) {
  const Long np = comm_.Size();
  SCTL_ASSERT_MSG(splitters.Dim() == np, "SortScatter::Init: one splitter per rank.");
  const auto pol = detail::scratch_policy<DevVec, Key>();
  const Long Nloc = (Long)keys.size();
  keys_ = std::move(keys);
  plan_ = detail_sortScatter::Plan<DevVec>{};
  plan_.Nloc = Nloc;

  { // stage 1: sort this rank's own keys, carrying their handed positions
    plan_.pre.resize(Nloc);
    thrust::sequence(pol, plan_.pre.begin(), plan_.pre.end(), Long(0));
    detail::local_sort_by_key(pol, keys_, plan_.pre, Nloc);
  }

  plan_.Nmid = Nloc;
  if (np > 1) {
    { // stage 2: each key to the rank owning its stretch; the sort left each destination's keys contiguous
      DeviceScratch<Key, DevVec> spl(np);
      thrust::copy(splitters.begin(), splitters.end(), spl.begin());
      plan_.scnt.ReInit(np);
      plan_.rcnt.ReInit(np);
      plan_.Nmid = detail::splitCounts(plan_.scnt.begin(), plan_.rcnt.begin(), keys_, Nloc, spl, comm_);
      DevVec<Key>& k2 = detail::PersistentBuffer<Key, DevVec, detail::Buf::PtSortK>();
      detail_sortScatter::resizeDiscard(k2, plan_.Nmid);
      detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(keys_.data()),
                                                 thrust::raw_pointer_cast(k2.data()), plan_.scnt, plan_.rcnt, Long(1), comm_);
      keys_.swap(k2);
    }
    { // stage 3: merge the arriving sorted runs into one block
      plan_.post.resize(plan_.Nmid);
      thrust::sequence(pol, plan_.post.begin(), plan_.post.end(), Long(0));
      detail::local_sort_by_key(pol, keys_, plan_.post, plan_.Nmid);
    }
  }
  plan_.Ntree = plan_.Nmid;
}

/** The keys are globally sorted, so this is a contiguous chunk move (no merge). */
template <class Key, template <class...> class DevVec>
void SortScatter<Key, DevVec>::Repartition(const sctl::Vector<Key>& splitters) {
  const Long np = comm_.Size();
  plan_.moved = false;
  if (np == 1) return;
  SCTL_ASSERT_MSG(splitters.Dim() == np, "SortScatter::Repartition: one splitter per rank.");
  const auto pol = detail::scratch_policy<DevVec, Key>();
  const Long N = (Long)keys_.size();

  DeviceScratch<Key, DevVec> spl(np);
  thrust::copy(splitters.begin(), splitters.end(), spl.begin());
  plan_.move_scnt.ReInit(np);
  plan_.move_rcnt.ReInit(np);
  const Long Nnew = detail::splitCounts(plan_.move_scnt.begin(), plan_.move_rcnt.begin(), keys_, N, spl, comm_);
  if (!sctl::sort_scatter_detail::recordRecut(plan_, N, Nnew, comm_)) return;
  DevVec<Key>& k2 = detail::PersistentBuffer<Key, DevVec, detail::Buf::PtSortK>();
  detail::resizeDiscard(k2, Nnew);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(keys_.data()),
                                             thrust::raw_pointer_cast(k2.data()), plan_.move_scnt, plan_.move_rcnt, Long(1), comm_);
  keys_.swap(k2);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::RepartitionData(DevVec<T>& data, Long dof, Long begin) const {
  const auto pol = detail::scratch_policy<DevVec, T>();
  // Values the keys held before the last Repartition; with nothing moved, what they hold now.
  const Long n = (plan_.moved ? plan_.move_n : plan_.Ntree) * dof;
  SCTL_ASSERT_MSG(begin >= 0 && begin + n <= (Long)data.size(),
                  "SortScatter::RepartitionData: data does not hold the previous SortedCount()*dof values at `begin`.");
  DevVec<T>& out = detail::PersistentBuffer<T, DevVec, detail::Buf::SwapOut>();
  if (!plan_.moved) {
    if (!begin && n == (Long)data.size()) return;  // already those values alone, in place
    using It = detail::ScratchIterator<T, DevVec>;  // nothing moved, but the surrounding values must go
    detail::resizeDiscard(out, n);
    T* const p = thrust::raw_pointer_cast(data.data());
    thrust::copy(pol, It(p + begin), It(p + begin + n), It(thrust::raw_pointer_cast(out.data())));
    data.swap(out);
    return;
  }
  detail::resizeDiscard(out, plan_.Ntree * dof);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(data.data()) + begin,
                                             thrust::raw_pointer_cast(out.data()), plan_.move_scnt, plan_.move_rcnt, dof, comm_);
  data.swap(out);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::ScatterForward(const T* src, T* dst, Long dof) const {
  const auto pol = detail::scratch_policy<DevVec, T>();
  if (comm_.Size() == 1) {  // stages 2-4 are absent, so the sort alone is the map
    detail_sortScatter::localMove<DevVec>(pol, src, dst, plan_.pre, plan_.Nloc, dof);
    return;
  }
  sctl::sort_scatter_detail::ensureRecut(plan_, comm_);
  DevVec<T>& a = detail::PersistentBuffer<T, DevVec, detail::Buf::PtSend>();
  DevVec<T>& b = detail::PersistentBuffer<T, DevVec, detail::Buf::PtRecv>();
  detail_sortScatter::resizeDiscard(a, plan_.Nloc * dof);
  detail_sortScatter::resizeDiscard(b, plan_.Nmid * dof);
  detail_sortScatter::localMove<DevVec>(pol, src, thrust::raw_pointer_cast(a.data()), plan_.pre, plan_.Nloc, dof);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(a.data()),
                                             thrust::raw_pointer_cast(b.data()), plan_.scnt, plan_.rcnt, dof, comm_);
  if (!plan_.recut) {
    detail_sortScatter::localMove<DevVec>(pol, thrust::raw_pointer_cast(b.data()), dst, plan_.post, plan_.Nmid, dof);
    return;
  }
  detail_sortScatter::resizeDiscard(a, plan_.Nmid * dof);
  detail_sortScatter::localMove<DevVec>(pol, thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(a.data()), plan_.post, plan_.Nmid, dof);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(a.data()), dst, plan_.rscnt, plan_.rrcnt, dof, comm_);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::ScatterReverse(const T* src, T* dst, Long dof) const {
  const auto pol = detail::scratch_policy<DevVec, T>();
  if (!plan_.inv) {  // inverses on the first move back: a caller that only moves data into sorted order never pays for them
    detail_sortScatter::buildInverse(pol, plan_.pre, plan_.pre_inv, plan_.Nloc);
    if ((Long)plan_.post.size()) detail_sortScatter::buildInverse(pol, plan_.post, plan_.post_inv, plan_.Nmid);
    plan_.inv = true;
  }
  if (comm_.Size() == 1) {
    detail_sortScatter::localMove<DevVec>(pol, src, dst, plan_.pre_inv, plan_.Nloc, dof);
    return;
  }
  // the stages of ScatterForward in reverse, each local one through its inverse
  sctl::sort_scatter_detail::ensureRecut(plan_, comm_);
  DevVec<T>& a = detail::PersistentBuffer<T, DevVec, detail::Buf::PtSend>();
  DevVec<T>& b = detail::PersistentBuffer<T, DevVec, detail::Buf::PtRecv>();
  const T* mid = src;  // the stage-3 output, whichever buffer holds it
  if (plan_.recut) {
    detail_sortScatter::resizeDiscard(b, plan_.Nmid * dof);
    detail_sortScatter::exchange<DevVec>(pol, src, thrust::raw_pointer_cast(b.data()), plan_.rrcnt, plan_.rscnt, dof, comm_);
    mid = thrust::raw_pointer_cast(b.data());
  }
  detail_sortScatter::resizeDiscard(a, plan_.Nmid * dof);
  detail_sortScatter::localMove<DevVec>(pol, mid, thrust::raw_pointer_cast(a.data()), plan_.post_inv, plan_.Nmid, dof);
  detail_sortScatter::resizeDiscard(b, plan_.Nloc * dof);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(a.data()),
                                             thrust::raw_pointer_cast(b.data()), plan_.rcnt, plan_.scnt, dof, comm_);
  detail_sortScatter::localMove<DevVec>(pol, thrust::raw_pointer_cast(b.data()), dst, plan_.pre_inv, plan_.Nloc, dof);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::ScatterForward(DevVec<T>& data, Long dof) const {
  SCTL_ASSERT_MSG((Long)data.size() == plan_.Nloc * dof, "SortScatter::ScatterForward: data holds LocalCount()*dof values.");
  DevVec<T>& out = detail::PersistentBuffer<T, DevVec, detail::Buf::SwapOut>();
  detail_sortScatter::resizeDiscard(out, plan_.Ntree * dof);
  ScatterForward(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(out.data()), dof);
  data.swap(out);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::ScatterReverse(DevVec<T>& data, Long dof) const {
  SCTL_ASSERT_MSG((Long)data.size() == plan_.Ntree * dof, "SortScatter::ScatterReverse: data holds SortedCount()*dof values.");
  DevVec<T>& out = detail::PersistentBuffer<T, DevVec, detail::Buf::SwapOut>();
  detail_sortScatter::resizeDiscard(out, plan_.Nloc * dof);
  ScatterReverse(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(out.data()), dof);
  data.swap(out);
}


// Keys with duplicates and one empty rank, cut two ways, round-tripped through every stage. The
// checks read backend memory on the host, since that is where the comparisons are.
template <class Key, template <class...> class DevVec> void SortScatter<Key, DevVec>::test() {
  const Comm comm = Comm::World();
  const Integer np = comm.Size(), rank = comm.Rank();
  const Long KMAX = Long(1) << 40, dof = 2;
  const Long N = (np > 2 && rank == np - 1 ? 0 : 100000);  // one empty rank when there are enough

  sctl::Vector<Key> keys(N);
  sctl::Vector<Long> payload(N * dof);
  { // payload row = (key, global index), so a moved row can be traced back
    Long gid0 = 0;
    comm.Scan(sctl::Ptr2ConstItr<Long>(&N, 1), sctl::Ptr2Itr<Long>(&gid0, 1), 1, sctl::CommOp::SUM);
    gid0 -= N;
    unsigned long long s = 0x9E3779B97F4A7C15ULL * (rank + 1);
    for (Long i = 0; i < N; i++) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      const Long k = (Long)((s >> 24) % (unsigned long long)KMAX) / (i % 7 == 0 ? 1024 : 1);  // some repeated keys
      keys[i] = (Key)k;
      payload[i * dof + 0] = k;
      payload[i * dof + 1] = gid0 + i;
    }
  }
  const auto splitters = [np, KMAX](Long shift) {  // even cut of the key range, shifted
    sctl::Vector<Key> spl(np);
    for (Integer r = 0; r < np; r++) spl[r] = (Key)(r * (KMAX / np) + (r ? shift : 0));
    return spl;
  };
  const auto toHost = [](const auto& d, Long n) {
    sctl::Vector<std::remove_const_t<typename std::decay_t<decltype(d)>::value_type>> h(n);
    if (n) detail::deviceToHost(d.data(), n, h.begin());
    return h;
  };
  const auto check = [&comm, &toHost, np, rank, N](const SortScatter& ss, const sctl::Vector<Key>& spl, const DevVec<Long>& sorted) {
    const Long n = ss.SortedCount();
    SCTL_ASSERT((Long)ss.SortedKeys().size() == n && (Long)sorted.size() == n * dof);
    const sctl::Vector<Key> k = toHost(ss.SortedKeys(), n);
    const sctl::Vector<Long> q = toHost(sorted, n * dof);
    Long bad = 0;
    for (Long i = 0; i < n; i++) {
      bad += (i && k[i] < k[i - 1]);                                                    // sorted
      bad += (rank && k[i] < spl[rank]) || (rank + 1 < np && !(k[i] < spl[rank + 1]));  // within my range
      bad += ((Long)k[i] != q[i * dof]);                                                // payload rode along
    }
    sctl::StaticArray<Long, 3> l{bad, n, N}, g;
    comm.Allreduce((sctl::ConstIterator<Long>)l, (sctl::Iterator<Long>)g, 3, sctl::CommOp::SUM);
    SCTL_ASSERT(g[0] == 0 && g[1] == g[2]);
  };
  const auto same = [&toHost](const DevVec<Long>& got, const sctl::Vector<Long>& want) {
    SCTL_ASSERT((Long)got.size() == want.Dim());
    const sctl::Vector<Long> h = toHost(got, want.Dim());
    for (Long i = 0; i < want.Dim(); i++) SCTL_ASSERT(h[i] == want[i]);
  };
  const auto roundTrip = [&check, &same, &payload, N](const SortScatter& ss, const sctl::Vector<Key>& spl) {
    DevVec<Long> q(payload.begin(), payload.end());
    ss.ScatterForward(q, dof);
    check(ss, spl, q);
    ss.ScatterReverse(q, dof);
    same(q, payload);
    { // the raw form, into caller-sized buffers
      const DevVec<Long> src(payload.begin(), payload.end());
      DevVec<Long> fwd(ss.SortedCount() * dof), back(N * dof);
      ss.template ScatterForward<Long>(thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(fwd.data()), dof);
      check(ss, spl, fwd);
      ss.template ScatterReverse<Long>(thrust::raw_pointer_cast(fwd.data()), thrust::raw_pointer_cast(back.data()), dof);
      same(back, payload);
    }
  };

  SortScatter ss(comm);
  const sctl::Vector<Key> splA = splitters(0), splB = splitters(KMAX / (3 * np));
  ss.Init(DevVec<Key>(keys.begin(), keys.end()), splA);
  SCTL_ASSERT(ss.LocalCount() == N);
  roundTrip(ss, splA);

  DevVec<Long> q(payload.begin(), payload.end());
  ss.ScatterForward(q, dof);  // in the first layout
  ss.Repartition(splB);       // re-cut
  ss.RepartitionData(q, dof); // follows the keys
  { // repartitioning the data must land where sorting into the new layout directly would
    DevVec<Long> q2(payload.begin(), payload.end());
    ss.ScatterForward(q2, dof);
    SCTL_ASSERT(q2.size() == q.size());
    same(q2, toHost(q, (Long)q.size()));
  }
  roundTrip(ss, splB);
  ss.Repartition(splA);  // re-cut of a re-cut
  roundTrip(ss, splA);
  ss.Repartition(splA);  // nothing moves
  roundTrip(ss, splA);
  { // a first splitter above some keys: the contract says it is not consulted, so rank 0 keeps
    // them. Dropping them instead shows up as a global count short of the input.
    sctl::Vector<Key> splC = splA;
    splC[0] = (Key)(np > 1 ? (Long)splA[1] / 2 : KMAX / 2);
    ss.Repartition(splC);
    roundTrip(ss, splC);
  }
  if (!rank) std::printf("gpu_tree::SortScatter::test passed on %d ranks\n", (int)np);
}

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_
