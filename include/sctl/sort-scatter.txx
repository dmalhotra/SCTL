#ifndef _SCTL_SORT_SCATTER_TXX_
#define _SCTL_SORT_SCATTER_TXX_

#include <algorithm>            // for lower_bound, min, max
#include <cstring>              // for memcpy
#include <functional>           // for less
#include <iostream>             // for cout (test)

#include "sctl/sort-scatter.hpp"
#include "sctl/common.hpp"        // for SCTL_ASSERT, Long, Integer
#include "sctl/comm.hpp"          // for Comm
#include "sctl/comm.txx"          // for Comm::Alltoallv, comm_detail::LocalSort
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator, Ptr2ConstItr
#include "sctl/iterator.txx"
#include "sctl/ompUtils.hpp"      // for omp_par::scan
#include "sctl/ompUtils.txx"
#include "sctl/scratch_pool.hpp"  // for ScratchBuf
#include "sctl/scratch_pool.txx"
#include "sctl/static-array.hpp"  // for StaticArray
#include "sctl/vector.hpp"        // for Vector
#include "sctl/vector.txx"

namespace sctl {

namespace sort_scatter_detail {

/** Sort `n` keys from `src` into `dst` (may alias), writing the source position of each into `idx`. */
template <class Key> void sortWithIndex(ConstIterator<Key> src, Iterator<Key> dst, Vector<Long>& idx, Long n) {
  ScratchBuf<comm_detail::SortPair<Key, Long>> in(n), out(n);
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < n; i++) {
    in[i].key = src[i];
    in[i].data = i;
  }
  comm_detail::LocalSort<comm_detail::SortPair<Key, Long>>(in.begin(), out.begin(), n, std::less<comm_detail::SortPair<Key, Long>>());
  if (idx.Dim() != n) idx.ReInit(n);
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < n; i++) {
    dst[i] = out[i].key;
    idx[i] = out[i].data;
  }
}

/** `scnt[r]`: the `n` sorted `keys` in `[splitters[r], splitters[r+1])` (ends open); `rcnt`: what
 *  comes back. Both sized np by the caller. Returns the receive total. */
template <class Key> Long splitCounts(Vector<Long>& scnt, Vector<Long>& rcnt, ConstIterator<Key> keys, Long n, const Vector<Key>& splitters, const Comm& comm) {
  const Integer np = comm.Size();
  ScratchBuf<Long> pos(np + 1);
  pos[0] = 0;
  pos[np] = n;
  #pragma omp parallel for schedule(static)
  for (Integer r = 1; r < np; r++) pos[r] = std::lower_bound(keys, keys + n, splitters[r]) - keys;
  for (Integer r = 0; r < np; r++) scnt[r] = pos[r + 1] - pos[r];
  comm.Alltoall<Long>(scnt.begin(), 1, rcnt.begin(), 1);
  Long nrecv = 0;
  for (Integer r = 0; r < np; r++) nrecv += rcnt[r];
  return nrecv;
}

/** `dst[i] = src[map[i]]` for `n` rows of `dof` values. */
template <class T> void localMove(ConstIterator<T> src, Iterator<T> dst, const Vector<Long>& map, Long n, Long dof) {
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < n; i++) std::memcpy(&dst[i * dof], &src[map[i] * dof], dof * sizeof(T));
}

/** One exchange stage, with the per-rank counts scaled by `dof`. */
template <class T> void exchange(ConstIterator<T> src, Iterator<T> dst, const Vector<Long>& scnt, const Vector<Long>& rcnt, Long dof, const Comm& comm) {
  const Integer np = comm.Size();
  ScratchBuf<Long> sc(np), rc(np), sd(np), rd(np);
  for (Integer r = 0; r < np; r++) {
    sc[r] = scnt[r] * dof;
    rc[r] = rcnt[r] * dof;
  }
  omp_par::scan(sc.begin(), sd.begin(), np, Long(0));
  omp_par::scan(rc.begin(), rd.begin(), np, Long(0));
  comm.Alltoallv<T>(src, sc.begin(), sd.begin(), dst, rc.begin(), rd.begin());
}

/** `inv[m[i]] = i`: the one scattered write in the scheme, one index per key. */
inline void buildInverse(const Vector<Long>& m, Vector<Long>& inv) {
  const Long n = m.Dim();
  if (inv.Dim() != n) inv.ReInit(n);
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < n; i++) inv[m[i]] = i;
}

/** Stage-4 counts from the stage-3 layout to the current one, on the first move after a repartition;
 *  no Alltoall, since every block size is allgathered. */
inline void ensureRecut(PlanBase& s, const Comm& comm) {
  if (!s.recut || s.recut_cnt) return;
  const Integer np = comm.Size(), rank = comm.Rank();
  ScratchBuf<Long> mid(np), cur(np);
  { // both block sizes in one gather
    ScratchBuf<Long> n(2 * np);
    const StaticArray<Long, 2> loc{s.Nmid, s.Ntree};
    comm.Allgather((ConstIterator<Long>)loc, 2, n.begin(), 2);
    for (Integer q = 0; q < np; q++) {
      mid[q] = n[2 * q];
      cur[q] = n[2 * q + 1];
    }
  }
  ScratchBuf<Long> moff(np + 1), coff(np + 1);
  omp_par::scan(mid.begin(), moff.begin(), np, Long(0));  // over exactly the np counts; the total is the entry past them
  omp_par::scan(cur.begin(), coff.begin(), np, Long(0));
  moff[np] = moff[np - 1] + mid[np - 1];
  coff[np] = coff[np - 1] + cur[np - 1];
  if (s.rscnt.Dim() != np) s.rscnt.ReInit(np);
  if (s.rrcnt.Dim() != np) s.rrcnt.ReInit(np);
  for (Integer q = 0; q < np; q++) {  // overlap of my stage-3 block with q's current block, and inverse
    s.rscnt[q] = std::max<Long>(0, std::min(moff[rank + 1], coff[q + 1]) - std::max(moff[rank], coff[q]));
    s.rrcnt[q] = std::max<Long>(0, std::min(moff[q + 1], coff[rank + 1]) - std::max(moff[q], coff[rank]));
  }
  s.recut_cnt = true;
}

/** A re-cut in which this rank keeps `nkeep` of its `n` keys and holds `Nnew` after: records
 *  stage 4 on `s`, to follow on the first move (ensureRecut) from the stage-3 layout, since two
 *  re-cuts compose to one. Returns whether any rank's keys move, which is also whether any rank's
 *  block size changes -- the cuts of a sorted block can only move together. */
inline bool recordRecut(PlanBase& s, Long n, Long nkeep, Long Nnew, const Comm& comm) {
  Long moved = n - nkeep, tot = 0;
  comm.Allreduce(Ptr2ConstItr<Long>(&moved, 1), Ptr2Itr<Long>(&tot, 1), 1, CommOp::SUM);
  if (!tot) return false;
  s.Ntree = Nnew;
  s.recut = true;
  s.recut_cnt = false;
  return true;
}

}  // namespace sort_scatter_detail

template <class Key> SortScatter<Key>::SortScatter(const Comm& comm) : comm_(comm) {}

template <class Key> void SortScatter<Key>::Init(const Vector<Key>& keys, const Vector<Key>& splitters) {
  const Integer np = comm_.Size();
  SCTL_ASSERT_MSG(splitters.Dim() == np, "SortScatter::Init: one splitter per rank.");
  const Long Nloc = keys.Dim();
  plan_ = sort_scatter_detail::Plan{};
  plan_.Nloc = Nloc;

  if (np == 1) {  // stage 1 alone: the sort is the result
    keys_.ReInit(Nloc);
    sort_scatter_detail::sortWithIndex<Key>(keys.begin(), keys_.begin(), plan_.pre, Nloc);
    plan_.Nmid = plan_.Ntree = Nloc;
    return;
  }

  ScratchBuf<Key> sorted(Nloc);
  sort_scatter_detail::sortWithIndex<Key>(keys.begin(), sorted.begin(), plan_.pre, Nloc);  // stage 1: this rank's own keys, carrying their handed positions

  { // stage 2: each key to the rank owning its stretch; the sort left each destination's keys contiguous
    plan_.scnt.ReInit(np);
    plan_.rcnt.ReInit(np);
    plan_.Nmid = sort_scatter_detail::splitCounts<Key>(plan_.scnt, plan_.rcnt, sorted.begin(), Nloc, splitters, comm_);
    keys_.ReInit(plan_.Nmid);
    sort_scatter_detail::exchange<Key>(sorted.begin(), keys_.begin(), plan_.scnt, plan_.rcnt, 1, comm_);
  }
  sort_scatter_detail::sortWithIndex<Key>(keys_.begin(), keys_.begin(), plan_.post, plan_.Nmid);  // stage 3: merge the arriving sorted runs
  plan_.Ntree = plan_.Nmid;
}

/** The keys are globally sorted, so this is a contiguous chunk move (no merge). */
template <class Key> void SortScatter<Key>::Repartition(const Vector<Key>& splitters) {
  const Integer np = comm_.Size();
  if (np == 1) return;
  SCTL_ASSERT_MSG(splitters.Dim() == np, "SortScatter::Repartition: one splitter per rank.");

  const Long n = keys_.Dim();
  Vector<Long> scnt(np), rcnt(np);
  const Long Nnew = sort_scatter_detail::splitCounts<Key>(scnt, rcnt, keys_.begin(), n, splitters, comm_);
  if (!sort_scatter_detail::recordRecut(plan_, n, scnt[comm_.Rank()], Nnew, comm_)) return;
  Vector<Key> recv(Nnew);
  sort_scatter_detail::exchange<Key>(keys_.begin(), recv.begin(), scnt, rcnt, 1, comm_);
  keys_.Swap(recv);
}

template <class Key> template <class T> void SortScatter<Key>::ScatterForward(ConstIterator<T> src, Iterator<T> dst, Long dof) const {
  if (comm_.Size() == 1) {  // stages 2-4 are absent, so the sort alone is the map
    sort_scatter_detail::localMove<T>(src, dst, plan_.pre, plan_.Nloc, dof);
    return;
  }
  sort_scatter_detail::ensureRecut(plan_, comm_);
  ScratchBuf<T> a(plan_.Nloc * dof), b(plan_.Nmid * dof);
  sort_scatter_detail::localMove<T>(src, a.begin(), plan_.pre, plan_.Nloc, dof);
  sort_scatter_detail::exchange<T>(a.begin(), b.begin(), plan_.scnt, plan_.rcnt, dof, comm_);
  if (!plan_.recut) {
    sort_scatter_detail::localMove<T>(b.begin(), dst, plan_.post, plan_.Nmid, dof);
    return;
  }
  ScratchBuf<T> c(plan_.Nmid * dof);
  sort_scatter_detail::localMove<T>(b.begin(), c.begin(), plan_.post, plan_.Nmid, dof);
  sort_scatter_detail::exchange<T>(c.begin(), dst, plan_.rscnt, plan_.rrcnt, dof, comm_);
}

template <class Key> template <class T> void SortScatter<Key>::ScatterReverse(ConstIterator<T> src, Iterator<T> dst, Long dof) const {
  if (!plan_.inv) {  // inverses on the first move back: a caller that only moves data into sorted order never pays for them
    sort_scatter_detail::buildInverse(plan_.pre, plan_.pre_inv);
    if (plan_.post.Dim()) sort_scatter_detail::buildInverse(plan_.post, plan_.post_inv);
    plan_.inv = true;
  }
  if (comm_.Size() == 1) {
    sort_scatter_detail::localMove<T>(src, dst, plan_.pre_inv, plan_.Nloc, dof);
    return;
  }
  // the stages of ScatterForward in reverse, each local one through its inverse
  sort_scatter_detail::ensureRecut(plan_, comm_);
  ScratchBuf<T> c(plan_.recut ? plan_.Nmid * dof : 0);
  ConstIterator<T> mid = src;  // the stage-3 output, whichever buffer holds it
  if (plan_.recut) {
    sort_scatter_detail::exchange<T>(src, c.begin(), plan_.rrcnt, plan_.rscnt, dof, comm_);
    mid = c.begin();
  }
  ScratchBuf<T> a(plan_.Nmid * dof), b(plan_.Nloc * dof);
  sort_scatter_detail::localMove<T>(mid, a.begin(), plan_.post_inv, plan_.Nmid, dof);
  sort_scatter_detail::exchange<T>(a.begin(), b.begin(), plan_.rcnt, plan_.scnt, dof, comm_);
  sort_scatter_detail::localMove<T>(b.begin(), dst, plan_.pre_inv, plan_.Nloc, dof);
}

template <class Key> template <class T> void SortScatter<Key>::ScatterForward(Vector<T>& data, Long dof) const {
  SCTL_ASSERT_MSG(data.Dim() == plan_.Nloc * dof, "SortScatter::ScatterForward: data holds LocalCount()*dof values.");
  Vector<T> out(plan_.Ntree * dof);
  ScatterForward<T>(data.begin(), out.begin(), dof);
  data.Swap(out);
}

template <class Key> template <class T> void SortScatter<Key>::ScatterReverse(Vector<T>& data, Long dof) const {
  SCTL_ASSERT_MSG(data.Dim() == plan_.Ntree * dof, "SortScatter::ScatterReverse: data holds SortedCount()*dof values.");
  Vector<T> out(plan_.Nloc * dof);
  ScatterReverse<T>(data.begin(), out.begin(), dof);
  data.Swap(out);
}

template <class Key> void SortScatter<Key>::test() {
  const Comm& comm = Comm::World();
  const Integer np = comm.Size(), rank = comm.Rank();
  const Long KMAX = Long(1) << 40, dof = 2;
  const Long N = (np > 2 && rank == np - 1 ? 0 : 100000);  // one empty rank when there are enough

  // keys: pseudo-random with duplicates; payload row = (key, global index)
  Vector<Key> keys(N);
  Vector<Long> payload(N * dof);
  {
    Long gid0 = 0;
    comm.Scan(Ptr2ConstItr<Long>(&N, 1), Ptr2Itr<Long>(&gid0, 1), 1, CommOp::SUM);
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
    Vector<Key> spl(np);
    for (Integer r = 0; r < np; r++) spl[r] = (Key)(r * (KMAX / np) + (r ? shift : 0));
    return spl;
  };
  const auto check = [&comm, np, rank, N, dof](const SortScatter& ss, const Vector<Key>& spl, const Vector<Long>& sorted_payload) {
    const Long n = ss.SortedCount();
    SCTL_ASSERT(ss.SortedKeys().Dim() == n && sorted_payload.Dim() == n * dof);
    Long bad = 0;
    for (Long i = 0; i < n; i++) {
      const Key k = ss.SortedKeys()[i];
      bad += (i && k < ss.SortedKeys()[i - 1]);                    // sorted
      bad += (rank && k < spl[rank]) || (rank + 1 < np && !(k < spl[rank + 1]));  // within my range
      bad += ((Long)k != sorted_payload[i * dof]);                 // payload rode along with its key
    }
    StaticArray<Long, 3> l{bad, n, N}, g;
    comm.Allreduce((ConstIterator<Long>)l, (Iterator<Long>)g, 3, CommOp::SUM);
    SCTL_ASSERT(g[0] == 0 && g[1] == g[2]);
  };
  const auto roundTrip = [&check, &payload, N, dof](const SortScatter& ss, const Vector<Key>& spl) {
    Vector<Long> q = payload;
    ss.ScatterForward(q, dof);
    check(ss, spl, q);
    ss.ScatterReverse(q, dof);
    SCTL_ASSERT(q.Dim() == payload.Dim());
    for (Long i = 0; i < q.Dim(); i++) SCTL_ASSERT(q[i] == payload[i]);
    { // the raw form, into caller-sized buffers
      Vector<Long> fwd(ss.SortedCount() * dof), back(N * dof);
      ss.ScatterForward<Long>(payload.begin(), fwd.begin(), dof);
      check(ss, spl, fwd);
      ss.ScatterReverse<Long>(fwd.begin(), back.begin(), dof);
      for (Long i = 0; i < back.Dim(); i++) SCTL_ASSERT(back[i] == payload[i]);
    }
  };

  SortScatter ss(comm);
  Vector<Key> splA = splitters(0), splB = splitters(KMAX / (3 * np));
  ss.Init(keys, splA);
  SCTL_ASSERT(ss.LocalCount() == N);
  roundTrip(ss, splA);
  Vector<Long> q = payload;
  ss.ScatterForward(q, dof);  // in the first layout
  ss.Repartition(splB);  // re-cut
  comm.PartitionN(q, ss.SortedCount());  // the payload follows the keys
  {
    Vector<Long> q2 = payload;
    ss.ScatterForward(q2, dof);
    SCTL_ASSERT(q2.Dim() == q.Dim());
    for (Long i = 0; i < q.Dim(); i++) SCTL_ASSERT(q2[i] == q[i]);
  }
  roundTrip(ss, splB);
  ss.Repartition(splA);  // re-cut of a re-cut
  roundTrip(ss, splA);
  ss.Repartition(splA);  // nothing moves
  roundTrip(ss, splA);
  if (!rank) std::cout << "SortScatter::test passed on " << np << " ranks\n";
}

}  // namespace sctl

#endif  // _SCTL_SORT_SCATTER_TXX_
