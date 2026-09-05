#ifndef _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_
#define _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_

// Included by gpu-tree.txx after its `detail` helpers (scratch_policy, local_sort_by_key,
// splitCounts, alltoallv, PersistentBuffer), which this file uses.

#include <algorithm>
#include <numeric>
#include <utility>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

#include "sctl/experimental/sort-scatter.hpp"
#include "sctl/experimental/device_scratch.hpp"

namespace gpu_tree {

namespace detail_sortScatter {

// One thread per value, not per key: consecutive threads then touch consecutive addresses within
// one key's block, where the per-key form has each thread issue its own strided access. At 100M
// keys and dof=3 that is 25.9 -> 19.7 ms for the gather and 56.3 -> 42.5 ms for the scatter. The
// divide and modulo by a runtime `dof` cost nothing measurable -- making it a compile-time constant
// changes neither timing.
//
// The thread count is the value count, `n * dof`, not `n`.
template <class T> struct GatherDofFunctor {
  const T* src; const Long* idx; T* dst; Long dof;
  SCTL_GPU_HD void operator()(Long e) const {
    const Long i = e / dof, k = e - i * dof;
    dst[e] = src[idx[i] * dof + k];
  }
};

/** `inv[m[i]] = i`. The only scattered write in the scheme, and it moves one index per key. */
template <Integer /*unused*/ D = 0> struct InvertFunctor {
  const Long* m; Long* inv;
  SCTL_GPU_HD void operator()(Long i) const { inv[m[i]] = i; }
};

template <template <class...> class DeviceVector, class Policy>
void buildInverse(const Policy& pol, const DeviceVector<Long>& m, DeviceVector<Long>& inv, Long n) {
  inv.resize(n);
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n, InvertFunctor<>{
      thrust::raw_pointer_cast(m.data()), thrust::raw_pointer_cast(inv.data())});
}

/** Inverses on the first move back: a caller that only moves data into sorted order never pays for them. */
template <template <class...> class DeviceVector, class Policy>
void ensureInverse(const Policy& pol, Plan<DeviceVector>& s) {
  if (s.inv) return;
  buildInverse(pol, s.pre, s.pre_inv, s.Nloc);
  if ((Long)s.post.size()) buildInverse(pol, s.post, s.post_inv, s.Nmid);
  s.inv = true;
}

/** Stage-4 counts from the stage-3 layout to the current one, on the first move after a repartition;
 *  no Alltoall, since every block size is allgathered. */
template <template <class...> class DeviceVector>
void ensureRecut(Plan<DeviceVector>& s, const Comm& comm) {
  if (!s.recut || s.recut_cnt) return;
  const Long np = comm.Size(), rank = comm.Rank();
  sctl::ScratchBuf<Long> mid(np), cur(np);
  comm.Allgather(sctl::Ptr2ConstItr<Long>(&s.Nmid, 1), 1, mid.begin(), 1);
  comm.Allgather(sctl::Ptr2ConstItr<Long>(&s.Ntree, 1), 1, cur.begin(), 1);
  sctl::ScratchBuf<Long> moff(np + 1), coff(np + 1);
  moff[0] = 0; std::inclusive_scan(mid.begin(), mid.end(), moff.begin() + 1);
  coff[0] = 0; std::inclusive_scan(cur.begin(), cur.end(), coff.begin() + 1);
  s.rscnt.ReInit(np); s.rrcnt.ReInit(np);
  for (Long q = 0; q < np; q++) {  // overlap of my stage-3 block with q's current block, and inverse
    s.rscnt[q] = std::max<Long>(0, std::min(moff[rank + 1], coff[q + 1]) - std::max(moff[rank], coff[q]));
    s.rrcnt[q] = std::max<Long>(0, std::min(moff[q + 1], coff[rank + 1]) - std::max(moff[q], coff[rank]));
  }
  s.recut_cnt = true;
}

/** One local stage: `dst[e] = src[map[e/dof]*dof + e%dof]` over `n*dof` values. */
template <template <class...> class DeviceVector, class T, class Policy>
void localMove(const Policy& pol, const T* src, T* dst, const DeviceVector<Long>& map, Long n, Long dof) {
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n * dof, GatherDofFunctor<T>{
      src, thrust::raw_pointer_cast(map.data()), dst, dof});
}

/** One exchange stage, between raw buffers, with the per-rank counts scaled by `dof`. */
template <template <class...> class DeviceVector, class T, class Policy>
void exchange(const Policy& pol, const T* src, Long nsrc, T* dst, Long ndst,
              const sctl::Vector<Long>& scnt, const sctl::Vector<Long>& rcnt, Long dof, const Comm& comm) {
  const Long np = comm.Size();
  if (np == 1) {
    using It = detail::ScratchIterator<T, DeviceVector>;
    thrust::copy(pol, It(const_cast<T*>(src)), It(const_cast<T*>(src)) + nsrc * dof, It(dst));
    return;
  }
  SCTL_UNUSED(ndst);
#ifdef SCTL_HAVE_MPI
  sctl::ScratchBuf<Long> sc(np), rc(np);
  for (Long r = 0; r < np; r++) { sc[r] = scnt[r] * dof; rc[r] = rcnt[r] * dof; }
  detail::alltoallv<DeviceVector>(pol, src, dst, sc, rc, (Long)sizeof(T), comm);
#endif
}

/**
 * Caller order -> sorted order, `dof` values per key. `src` holds `Nloc*dof` values, `dst` holds
 * `Ntree*dof`, and they must not overlap.
 */
template <class T, template <class...> class DeviceVector, class Policy>
void forward(const Policy& pol, const T* src, T* dst, Plan<DeviceVector>& s, Long dof, const Comm& comm) {
  if (comm.Size() == 1) {  // stages 2-4 are absent, so the sort alone is the map
    localMove<DeviceVector>(pol, src, dst, s.pre, s.Nloc, dof);
    return;
  }
  ensureRecut(s, comm);
  DeviceVector<T>& a = detail::PersistentBuffer<T, DeviceVector, detail::Buf::PtSend>();
  DeviceVector<T>& b = detail::PersistentBuffer<T, DeviceVector, detail::Buf::PtRecv>();
  a.resize(s.Nloc * dof);
  b.resize(s.Nmid * dof);
  localMove<DeviceVector>(pol, src, thrust::raw_pointer_cast(a.data()), s.pre, s.Nloc, dof);
  exchange<DeviceVector>(pol, thrust::raw_pointer_cast(a.data()), s.Nloc,
                         thrust::raw_pointer_cast(b.data()), s.Nmid, s.scnt, s.rcnt, dof, comm);
  if (!s.recut) {
    localMove<DeviceVector>(pol, thrust::raw_pointer_cast(b.data()), dst, s.post, s.Nmid, dof);
    return;
  }
  a.resize(s.Nmid * dof);
  localMove<DeviceVector>(pol, thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(a.data()), s.post, s.Nmid, dof);
  exchange<DeviceVector>(pol, thrust::raw_pointer_cast(a.data()), s.Nmid, dst, s.Ntree, s.rscnt, s.rrcnt, dof, comm);
}

/** Sorted order -> caller order: the same stages in reverse, each local one using its inverse. */
template <class T, template <class...> class DeviceVector, class Policy>
void reverse(const Policy& pol, const T* src, T* dst, Plan<DeviceVector>& s, Long dof, const Comm& comm) {
  ensureInverse(pol, s);
  if (comm.Size() == 1) {
    localMove<DeviceVector>(pol, src, dst, s.pre_inv, s.Nloc, dof);
    return;
  }
  ensureRecut(s, comm);
  DeviceVector<T>& a = detail::PersistentBuffer<T, DeviceVector, detail::Buf::PtSend>();
  DeviceVector<T>& b = detail::PersistentBuffer<T, DeviceVector, detail::Buf::PtRecv>();
  const T* mid = src;  // the stage-3 output, whichever buffer holds it
  if (s.recut) {
    b.resize(s.Nmid * dof);
    exchange<DeviceVector>(pol, src, s.Ntree, thrust::raw_pointer_cast(b.data()), s.Nmid, s.rrcnt, s.rscnt, dof, comm);
    mid = thrust::raw_pointer_cast(b.data());
  }
  a.resize(s.Nmid * dof);
  localMove<DeviceVector>(pol, mid, thrust::raw_pointer_cast(a.data()), s.post_inv, s.Nmid, dof);
  b.resize(s.Nloc * dof);
  exchange<DeviceVector>(pol, thrust::raw_pointer_cast(a.data()), s.Nmid,
                         thrust::raw_pointer_cast(b.data()), s.Nloc, s.rcnt, s.scnt, dof, comm);
  localMove<DeviceVector>(pol, thrust::raw_pointer_cast(b.data()), dst, s.pre_inv, s.Nloc, dof);
}

}  // namespace detail_sortScatter

template <class Key, template <class...> class DeviceVector>
void SortScatter<Key, DeviceVector>::Init(DeviceVector<Key> keys, const sctl::Vector<Key>& splitters) {
  const Long np = comm_.Size();
  SCTL_ASSERT_MSG(splitters.Dim() == np, "SortScatter::Init: one splitter per rank.");
  const auto pol = detail::scratch_policy<DeviceVector, Key>();
  const Long Nloc = (Long)keys.size();
  keys_ = std::move(keys);
  plan_ = detail_sortScatter::Plan<DeviceVector>{};
  plan_.Nloc = Nloc;

  { // stage 1: sort this rank's own keys, carrying their handed positions
    plan_.pre.resize(Nloc);
    thrust::sequence(pol, plan_.pre.begin(), plan_.pre.end(), Long(0));
    detail::local_sort_by_key(pol, keys_, plan_.pre, Nloc);
  }

  plan_.Nmid = Nloc;
  if (np > 1) {
    { // stage 2: each key to the rank owning its stretch; the sort left each destination's keys contiguous
      DeviceScratch<Key, DeviceVector> spl(np);
      thrust::copy(splitters.begin(), splitters.end(), spl.begin());
      sctl::ScratchBuf<Long> sc(np), rc(np);
      plan_.Nmid = detail::splitCounts(sc, rc, keys_, Nloc, spl, comm_);
      plan_.scnt.ReInit(np); plan_.rcnt.ReInit(np);
      for (Long r = 0; r < np; r++) { plan_.scnt[r] = sc[r]; plan_.rcnt[r] = rc[r]; }
      DeviceVector<Key>& k2 = detail::PersistentBuffer<Key, DeviceVector, detail::Buf::PtSortK>();
      k2.resize(plan_.Nmid);
      detail_sortScatter::exchange<DeviceVector>(pol, thrust::raw_pointer_cast(keys_.data()), Nloc,
                                                 thrust::raw_pointer_cast(k2.data()), plan_.Nmid, plan_.scnt, plan_.rcnt, Long(1), comm_);
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
template <class Key, template <class...> class DeviceVector>
void SortScatter<Key, DeviceVector>::Repartition(const sctl::Vector<Key>& splitters) {
  const Long np = comm_.Size();
  moved_ = false;
  if (np == 1) return;
  SCTL_ASSERT_MSG(splitters.Dim() == np, "SortScatter::Repartition: one splitter per rank.");
  const auto pol = detail::scratch_policy<DeviceVector, Key>();
  const Long N = (Long)keys_.size();

  DeviceScratch<Key, DeviceVector> spl(np);
  thrust::copy(splitters.begin(), splitters.end(), spl.begin());
  sctl::ScratchBuf<Long> sc(np), rc(np);
  const Long Nnew = detail::splitCounts(sc, rc, keys_, N, spl, comm_);
  { Long moved = N - sc[comm_.Rank()], tot = 0;  // skip when nothing crosses a rank boundary
    comm_.Allreduce(sctl::Ptr2ConstItr<Long>(&moved, 1), sctl::Ptr2Itr<Long>(&tot, 1), 1, sctl::CommOp::SUM);
    if (!tot) return; }
  { // move the keys to the new partition, keeping the counts for RepartitionData
    move_scnt_.ReInit(np); move_rcnt_.ReInit(np);
    for (Long r = 0; r < np; r++) { move_scnt_[r] = sc[r]; move_rcnt_[r] = rc[r]; }
    move_n_ = N;
    moved_ = true;
    DeviceVector<Key>& k2 = detail::PersistentBuffer<Key, DeviceVector, detail::Buf::PtSortK>();
    k2.resize(Nnew);
    detail_sortScatter::exchange<DeviceVector>(pol, thrust::raw_pointer_cast(keys_.data()), N,
                                               thrust::raw_pointer_cast(k2.data()), Nnew, move_scnt_, move_rcnt_, Long(1), comm_);
    keys_.swap(k2);
  }
  // stage 4 follows on the first move (ensureRecut), from the stage-3 layout: two re-cuts compose to one
  plan_.Ntree = Nnew;
  plan_.recut = true;
  plan_.recut_cnt = false;
}

template <class Key, template <class...> class DeviceVector> template <class T>
void SortScatter<Key, DeviceVector>::RepartitionData(DeviceVector<T>& data, Long dof) const {
  if (!moved_) return;
  SCTL_ASSERT_MSG((Long)data.size() == move_n_ * dof, "SortScatter::RepartitionData: data holds the previous SortedCount()*dof values.");
  DeviceVector<T> out(plan_.Ntree * dof);
  detail_sortScatter::exchange<DeviceVector>(detail::scratch_policy<DeviceVector, T>(), thrust::raw_pointer_cast(data.data()), move_n_,
                                             thrust::raw_pointer_cast(out.data()), plan_.Ntree, move_scnt_, move_rcnt_, dof, comm_);
  data.swap(out);
}

template <class Key, template <class...> class DeviceVector> template <class T>
void SortScatter<Key, DeviceVector>::ScatterForward(const T* src, T* dst, Long dof) const {
  detail_sortScatter::forward(detail::scratch_policy<DeviceVector, T>(), src, dst, plan_, dof, comm_);
}

template <class Key, template <class...> class DeviceVector> template <class T>
void SortScatter<Key, DeviceVector>::ScatterReverse(const T* src, T* dst, Long dof) const {
  detail_sortScatter::reverse(detail::scratch_policy<DeviceVector, T>(), src, dst, plan_, dof, comm_);
}

template <class Key, template <class...> class DeviceVector> template <class T>
void SortScatter<Key, DeviceVector>::ScatterForward(DeviceVector<T>& data, Long dof) const {
  SCTL_ASSERT_MSG((Long)data.size() == plan_.Nloc * dof, "SortScatter::ScatterForward: data holds LocalCount()*dof values.");
  DeviceVector<T> out(plan_.Ntree * dof);
  ScatterForward(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(out.data()), dof);
  data.swap(out);
}

template <class Key, template <class...> class DeviceVector> template <class T>
void SortScatter<Key, DeviceVector>::ScatterReverse(DeviceVector<T>& data, Long dof) const {
  SCTL_ASSERT_MSG((Long)data.size() == plan_.Ntree * dof, "SortScatter::ScatterReverse: data holds SortedCount()*dof values.");
  DeviceVector<T> out(plan_.Nloc * dof);
  ScatterReverse(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(out.data()), dof);
  data.swap(out);
}

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_
