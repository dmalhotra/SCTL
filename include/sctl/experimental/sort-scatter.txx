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
  const T* src; const Long* idx; T* dst; Long dof;
  SCTL_GPU_HD void operator()(Long e) const {
    const Long i = e / dof, k = e - i * dof;
    dst[e] = src[idx[i] * dof + k];
  }
};

/** `inv[m[i]] = i`. The only scattered write in the scheme, and it moves one index per key. */
struct InvertFunctor {
  const Long* m; Long* inv;
  SCTL_GPU_HD void operator()(Long i) const { inv[m[i]] = i; }
};

template <template <class...> class DevVec, class Policy>
void buildInverse(const Policy& pol, const DevVec<Long>& m, DevVec<Long>& inv, Long n) {
  inv.resize(n);
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n, InvertFunctor{
      thrust::raw_pointer_cast(m.data()), thrust::raw_pointer_cast(inv.data())});
}

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
#ifdef SCTL_HAVE_MPI
  sctl::ScratchBuf<Long> sc(np), rc(np);
  for (Long r = 0; r < np; r++) { sc[r] = scnt[r] * dof; rc[r] = rcnt[r] * dof; }
  detail::alltoallv<DevVec>(pol, src, dst, sc, rc, (Long)sizeof(T), comm);
#endif
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
      plan_.scnt.ReInit(np); plan_.rcnt.ReInit(np);
      plan_.Nmid = detail::splitCounts(plan_.scnt.begin(), plan_.rcnt.begin(), keys_, Nloc, spl, comm_);
      DevVec<Key>& k2 = detail::PersistentBuffer<Key, DevVec, detail::Buf::PtSortK>();
      k2.resize(plan_.Nmid);
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
  plan_.move_scnt.ReInit(np); plan_.move_rcnt.ReInit(np);
  const Long Nnew = detail::splitCounts(plan_.move_scnt.begin(), plan_.move_rcnt.begin(), keys_, N, spl, comm_);
  if (!sctl::sort_scatter_detail::recordRecut(plan_, N, Nnew, comm_)) return;
  DevVec<Key>& k2 = detail::PersistentBuffer<Key, DevVec, detail::Buf::PtSortK>();
  k2.resize(Nnew);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(keys_.data()),
                                             thrust::raw_pointer_cast(k2.data()), plan_.move_scnt, plan_.move_rcnt, Long(1), comm_);
  keys_.swap(k2);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::RepartitionData(DevVec<T>& data, Long dof) const {
  if (!plan_.moved) return;
  SCTL_ASSERT_MSG((Long)data.size() == plan_.move_n * dof, "SortScatter::RepartitionData: data holds the previous SortedCount()*dof values.");
  DevVec<T> out(plan_.Ntree * dof);
  detail_sortScatter::exchange<DevVec>(detail::scratch_policy<DevVec, T>(), thrust::raw_pointer_cast(data.data()),
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
  a.resize(plan_.Nloc * dof);
  b.resize(plan_.Nmid * dof);
  detail_sortScatter::localMove<DevVec>(pol, src, thrust::raw_pointer_cast(a.data()), plan_.pre, plan_.Nloc, dof);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(a.data()),
                                             thrust::raw_pointer_cast(b.data()), plan_.scnt, plan_.rcnt, dof, comm_);
  if (!plan_.recut) {
    detail_sortScatter::localMove<DevVec>(pol, thrust::raw_pointer_cast(b.data()), dst, plan_.post, plan_.Nmid, dof);
    return;
  }
  a.resize(plan_.Nmid * dof);
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
    b.resize(plan_.Nmid * dof);
    detail_sortScatter::exchange<DevVec>(pol, src, thrust::raw_pointer_cast(b.data()), plan_.rrcnt, plan_.rscnt, dof, comm_);
    mid = thrust::raw_pointer_cast(b.data());
  }
  a.resize(plan_.Nmid * dof);
  detail_sortScatter::localMove<DevVec>(pol, mid, thrust::raw_pointer_cast(a.data()), plan_.post_inv, plan_.Nmid, dof);
  b.resize(plan_.Nloc * dof);
  detail_sortScatter::exchange<DevVec>(pol, thrust::raw_pointer_cast(a.data()),
                                             thrust::raw_pointer_cast(b.data()), plan_.rcnt, plan_.scnt, dof, comm_);
  detail_sortScatter::localMove<DevVec>(pol, thrust::raw_pointer_cast(b.data()), dst, plan_.pre_inv, plan_.Nloc, dof);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::ScatterForward(DevVec<T>& data, Long dof) const {
  SCTL_ASSERT_MSG((Long)data.size() == plan_.Nloc * dof, "SortScatter::ScatterForward: data holds LocalCount()*dof values.");
  DevVec<T> out(plan_.Ntree * dof);
  ScatterForward(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(out.data()), dof);
  data.swap(out);
}

template <class Key, template <class...> class DevVec> template <class T>
void SortScatter<Key, DevVec>::ScatterReverse(DevVec<T>& data, Long dof) const {
  SCTL_ASSERT_MSG((Long)data.size() == plan_.Ntree * dof, "SortScatter::ScatterReverse: data holds SortedCount()*dof values.");
  DevVec<T> out(plan_.Nloc * dof);
  ScatterReverse(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(out.data()), dof);
  data.swap(out);
}

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_TXX_
