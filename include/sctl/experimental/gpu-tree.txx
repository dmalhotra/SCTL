// Template implementation of GPUTree from gpu-tree.hpp; the helpers live in per-stage detail_* namespaces.

#ifndef _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
#define _SCTL_EXPERIMENTAL_GPU_TREE_TXX_

#include <thrust/copy.h>
#include <thrust/merge.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#include <type_traits>
#include <utility>
#include <numeric>

#include "sctl/experimental/gpu-tree.hpp"
#include "sctl/experimental/device_scratch.hpp"
#include "sctl/comm.hpp"
#include "sctl/comm.txx"
#include "sctl/ompUtils.txx"
#include "sctl/profile.hpp"  // build stages as Profile blocks
#include "sctl/profile.txx"
#include "sctl/tree.hpp"   // tree_detail::Balance21, nbr_path_table
#include "sctl/vtudata.hpp"  // WriteTreeVTK
#include "sctl/vtudata.txx"
#include "sctl/tree.txx"
#include "sctl/scratch_pool.hpp"
#include "sctl/scratch_pool.txx"

namespace gpu_tree {

namespace detail {

// Execution policy for thrust calls on backend memory, with temporaries drawn from the scratch
// pool (thrust/cub otherwise cudaMalloc's them per call, which costs more than the work).
template <template <class...> class DevVec, class T> auto scratch_policy() {
#if defined(__CUDACC__) || defined(__HIPCC__)
  if constexpr (is_device_vector_v<DevVec<T>>) {
    static DeviceScratchAllocator<DevVec> alloc;
    return thrust::cuda::par(alloc);
  } else
#endif
  {
    return thrust::host;
  }
}

// Stage timing through sctl::Profile, compiled out with it. The device is drained before each mark
// so that a block's time is the stage's own.
template <template <class...> class DevVec> struct StageTimer {
  const Comm& comm;
  void tic(const char* name, Integer verbose = 1) const {
    sync();
    sctl::Profile::Tic(name, &comm, true, verbose);
  }
  void toc() const {
    sync();
    sctl::Profile::Toc();
  }
  static void sync() {
#if SCTL_PROFILE >= 0 && (defined(__CUDACC__) || defined(__HIPCC__))
    if constexpr (is_device_vector_v<DevVec<char>>) cudaDeviceSynchronize();
#endif
  }
};

// Which retained buffer a `PersistentBuffer` call means; no two uses may share a tag.
enum class Buf { PtMid, PtAlt, BcastOut, Closure, Frontier, ClosureRecv, GhostMerge, DataRecv,
                 PtSend, PtRecv, PtSortK, MigData, OldMid, SwapOut };

// Functor (not lambda) so nvcc captures it across thrust kernel boundaries.
template <class Real, Integer DIM> struct MakeMortonFunctor {
  const Real* coord_ptr;
  SCTL_GPU_HD MortonCode<DIM> operator()(Long i) const {
    return MortonCode<DIM>(coord_ptr + i * DIM);
  }
};

/** Element type and vector family of a container, for either a plain vector or a pool slice. */
template <class V> struct vec_family;
template <template <class...> class VV, class T, class... A> struct vec_family<VV<T, A...>> {
  using elem = T;
  template <class U> using to = VV<U>;
};
template <class T, template <class...> class VV> struct vec_family<DeviceScratch<T, VV>> {
  using elem = T;
  template <class U> using to = VV<U>;
};

template <class T> struct ToIntKeyFunctor {
  SCTL_GPU_HD std::uint64_t operator()(const T& x) const { return x.GetIntKey(); }
};
template <class T> struct FromIntKeyFunctor {
  SCTL_GPU_HD T operator()(std::uint64_t k) const { return T::FromIntKey(k); }
};

// Sort v[0,n), `pol` supplying the device path's temporaries. Device: a Morton code is a struct,
// so thrust picks a comparison sort for it; sorting its integer key reaches cub's radix sort
// instead and gives the identical order. Host: omp_par, since thrust's host backend is serial;
// merge_sort stops scaling past ~16 threads (bandwidth-bound), sample_sort doesn't, so pick by
// thread count.
template <class Policy, class Vec> void local_sort(const Policy& pol, Vec& v, Long n) {
  using T = typename vec_family<Vec>::elem;
  if constexpr (is_device_vector_v<Vec> && sctl::omp_par::is_radix_sortable<T>::value) {
    DeviceScratch<std::uint64_t, vec_family<Vec>::template to> k(n);
    thrust::transform(pol, v.begin(), v.begin() + n, k.begin(), ToIntKeyFunctor<T>{});
    thrust::sort(pol, k.begin(), k.begin() + n);
    thrust::transform(pol, k.begin(), k.begin() + n, v.begin(), FromIntKeyFunctor<T>{});
  } else if constexpr (is_device_vector_v<Vec>) {
    thrust::sort(pol, v.begin(), v.begin() + n);
  } else {
    T* const p = thrust::raw_pointer_cast(v.data());
    if constexpr (sctl::omp_par::is_radix_sortable<T>::value) sctl::omp_par::radix_sort(p, n, [](const T& x) { return x.GetIntKey(); });
    else if (sctl::omp_par_detail::PreferMergeSort(sizeof(T))) sctl::omp_par::merge_sort(p, p + n);
    else sctl::omp_par::sample_sort(p, p + n);
  }
}

// local_sort carrying a payload (the pre-sort index). Host path sorts packed pairs: omp_par has no
// by-key sort.
template <class Policy, class Vec, class IVec> void local_sort_by_key(const Policy& pol, Vec& keys, IVec& vals, Long n) {
  using T = typename vec_family<Vec>::elem;
  if constexpr (is_device_vector_v<Vec> && sctl::omp_par::is_radix_sortable<T>::value) {
    DeviceScratch<std::uint64_t, vec_family<Vec>::template to> k(n);
    thrust::transform(pol, keys.begin(), keys.begin() + n, k.begin(), ToIntKeyFunctor<T>{});
    thrust::sort_by_key(pol, k.begin(), k.begin() + n, vals.begin());
    thrust::transform(pol, k.begin(), k.begin() + n, keys.begin(), FromIntKeyFunctor<T>{});
  } else if constexpr (is_device_vector_v<Vec>) {
    thrust::sort_by_key(pol, keys.begin(), keys.begin() + n, vals.begin());
  } else {
    using ValT = typename vec_family<IVec>::elem;
    struct Pair {
      T key;
      ValT val;
      bool operator<(const Pair& o) const { return key < o.key; }
    };
    T* kp = thrust::raw_pointer_cast(keys.data());
    ValT* vp = thrust::raw_pointer_cast(vals.data());
    sctl::ScratchBuf<Pair> pairs(n);
    const auto pp = pairs.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n; i++) {
      pp[i].key = kp[i];
      pp[i].val = vp[i];
    }
    if constexpr (sctl::omp_par::is_radix_sortable<T>::value) {
      sctl::omp_par::radix_sort(pairs.begin(), n, [](const Pair& x) { return x.key.GetIntKey(); });
    } else if (sctl::omp_par_detail::PreferMergeSort(sizeof(Pair))) {
      sctl::omp_par::merge_sort(pairs.begin(), pairs.end());
    } else {
      sctl::omp_par::sample_sort(pairs.begin(), pairs.end());
    }
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n; i++) {
      kp[i] = pp[i].key;
      vp[i] = pp[i].val;
    }
  }
}

// Route a sorted array to its owners. Per-rank element counts (np each) from splitting `in[0,n)` at
// the device-resident `keys` (np keys, or np-1 splitters with the first block starting at 0), and
// the counts coming back; returns the number of elements to be received.
template <class T, template <class...> class DevVec, class Vec>
Long splitCounts(sctl::Iterator<Long> scnt, sctl::Iterator<Long> rcnt, const Vec& in, Long n,
                 const DeviceScratch<T, DevVec>& keys, const Comm& comm) {
  const Long np = comm.Size(), nkeys = keys.Dim();
  SCTL_ASSERT(nkeys == np || nkeys == np - 1);
  {
    sctl::ScratchBuf<Long> pos(np + 1);
    DeviceScratch<Long, DevVec> pos_d(nkeys);
    thrust::lower_bound(scratch_policy<DevVec, T>(), in.begin(), in.begin() + n, keys.begin(), keys.end(), pos_d.begin());
    thrust::copy(pos_d.begin(), pos_d.end(), pos.begin() + (np - nkeys));
    pos[0] = 0;  // `splitters[0]` is not consulted: rank 0 takes everything below the next splitter
    pos[np] = n;
    for (Long r = 0; r < np; r++) scnt[r] = pos[r + 1] - pos[r];
  }
  comm.Alltoall(scnt, 1, rcnt, 1);
  Long nrecv = 0;
  for (Long r = 0; r < np; r++) nrecv += rcnt[r];
  return nrecv;
}

/** Equivalence under `operator<`, for thrust's unique on the device path. */
template <class T> struct EquivPred {
  SCTL_GPU_HD bool operator()(const T& a, const T& b) const { return !(a < b) && !(b < a); }
};

/**
 * `thrust::unique` of a sorted range that stays inside it. thrust's OMP host backend flags run
 * heads through `zip(counting, first, first - 1)` and materializes the tuple before its index
 * check, so `first[-1]` is loaded (thrust/detail/range/head_flags.h); when the range begins
 * exactly at an mmap boundary -- a fresh pool chunk, or any large std::vector -- that load
 * faults. The device backend takes cub's paths and is unaffected, so it keeps thrust; the host
 * path uses sctl's parallel dedup, which reads only [first, last).
 */
template <class Policy, class Vec>
Long local_unique(const Policy& pol, Vec& v, Long n) {
  using T = typename vec_family<Vec>::elem;
  if constexpr (is_device_vector_v<Vec>) {
    return thrust::unique(pol, v.begin(), v.begin() + n, EquivPred<T>{}) - v.begin();
  } else {
    if (!n) return 0;
    T* p = thrust::raw_pointer_cast(v.data());
    sctl::ScratchBuf<T> tmp(n);
    const Long m = sctl::omp_par::dedup_sorted(p, tmp.begin(), n);
    sctl::omp_par::copy(tmp.begin(), tmp.begin() + m, p);
    return m;
  }
}

/** The copying form of `local_unique`; `in` may be a transform iterator. */
template <class Policy, class InIt, class Vec>
Long local_unique_copy(const Policy& pol, InIt in, Long n, Vec& out) {
  using T = typename vec_family<Vec>::elem;
  if constexpr (is_device_vector_v<Vec>) {
    return thrust::unique_copy(pol, in, in + n, out.begin(), EquivPred<T>{}) - out.begin();
  } else {
    return n ? sctl::omp_par::dedup_sorted(in, thrust::raw_pointer_cast(out.data()), n) : 0;
  }
}

// Alltoallv of esz-sized elements given per-rank element counts (displacements are their scans).
// Host buffers take Comm's exchange; device buffers go straight to MPI, which is CUDA-aware here.
inline void alltoallvHost(const void* sbuf, void* rbuf, const sctl::ScratchBuf<Long>& scnt, const sctl::ScratchBuf<Long>& rcnt, Long esz, const Comm& comm) {
  const Long np = comm.Size();
  sctl::ScratchBuf<Long> sb(np), rb(np), sd(np), rd(np);  // byte counts and offsets
  for (Long r = 0; r < np; r++) {
    sb[r] = scnt[r] * esz;
    rb[r] = rcnt[r] * esz;
  }
  std::exclusive_scan(sb.begin(), sb.end(), sd.begin(), Long(0));
  std::exclusive_scan(rb.begin(), rb.end(), rd.begin(), Long(0));
  const Long ns = sd[np - 1] + sb[np - 1], nr = rd[np - 1] + rb[np - 1];
  comm.Alltoallv(sctl::Ptr2ConstItr<char>(sbuf, ns), sb.begin(), sd.begin(), sctl::Ptr2Itr<char>(rbuf, nr), rb.begin(), rd.begin());
}

// The device path: the block a rank sends itself is copied on the device, since MPI streams it
// single-threaded at a few GB/s; the rest goes to MPI_Alltoallv, or pairwise once counts exceed int.
template <template <class...> class DevVec, class Policy>
void alltoallvDevice(const Policy& pol, const void* sbuf, void* rbuf, const sctl::ScratchBuf<Long>& scnt,
                     const sctl::ScratchBuf<Long>& rcnt, Long esz, const Comm& comm) {
#ifdef SCTL_HAVE_MPI
  const Long np = comm.Size(), rank = comm.Rank();
  sctl::ScratchBuf<Long> sd(np + 1), rd(np + 1);   // byte offsets, in Long
  const auto bytes = [esz](Long cnt) { return cnt * esz; };
  sd[0] = 0;
  rd[0] = 0;
  std::transform_inclusive_scan(scnt.begin(), scnt.end(), sd.begin() + 1, std::plus<Long>(), bytes);
  std::transform_inclusive_scan(rcnt.begin(), rcnt.end(), rd.begin() + 1, std::plus<Long>(), bytes);
  static const Long IMAX = 2147483647;
  SCTL_ASSERT(scnt[rank] == rcnt[rank]);
  if (const Long n = scnt[rank] * esz) {  // self block, copied on the device
    using It = ScratchIterator<char, DevVec>;
    const It s(const_cast<char*>((const char*)sbuf + sd[rank]));
    thrust::copy(pol, s, s + n, It((char*)rbuf + rd[rank]));
  }

  // MPI_Alltoallv's counts and displacements are `int`, and a dof=3 payload at 100M items per rank
  // is 2.4 GB. Counting in a larger unit than one byte buys that factor of headroom, so rather
  // than take `esz` -- which the byte-granular node-data migration passes as 1 -- find the largest
  // power of two that divides every count and displacement. The low set bit of their OR is exactly
  // that: if all are multiples of 2^k then none has a bit below k, so neither does the OR. At
  // dof=3 doubles the counts are multiples of 24, giving 8 and a 17 GB ceiling, which no caller
  // approaches. The limit is per count and per displacement, not on their total.
  Long g = sd[np - 1] | rd[np - 1];
  for (Long r = 0; r < np; r++) g |= (scnt[r] * esz) | (rcnt[r] * esz);
  const Long G = g ? std::min<Long>(g & -g, Long(1) << 20) : 1;
  bool fits = (sd[np - 1] / G <= IMAX && rd[np - 1] / G <= IMAX);
  for (Long r = 0; r < np && fits; r++) fits = (scnt[r] * esz / G <= IMAX && rcnt[r] * esz / G <= IMAX);
  if (fits) {
    sctl::ScratchBuf<int> sc(np), sdi(np), rc(np), rdi(np);
    for (Long r = 0; r < np; r++) {
      sc[r] = (int)(scnt[r] * esz / G);
      rc[r] = (int)(rcnt[r] * esz / G);
      sdi[r] = (int)(sd[r] / G);
      rdi[r] = (int)(rd[r] / G);
    }
    sc[rank] = 0;
    rc[rank] = 0;
    MPI_Datatype dt;
    MPI_Type_contiguous((int)G, MPI_BYTE, &dt);
    MPI_Type_commit(&dt);
    MPI_Alltoallv(sbuf, &sc[0], &sdi[0], dt, rbuf, &rc[0], &rdi[0], dt, comm.GetMPI_Comm());
    MPI_Type_free(&dt);
    return;
  }

  // Beyond that, exchange pairwise on a rotating schedule, one send and one receive outstanding:
  // the shape MPI's own large-message alltoallv uses. Offsets stay `Long` and become pointer
  // arithmetic, so only each message's count must fit an `int`.
  const char* sp = (const char*)sbuf;
  char* rp = (char*)rbuf;
  const Long CHUNK = Long(1) << 30;
  const auto nchunk = [CHUNK](Long n) { return (n + CHUNK - 1) / CHUNK; };
  for (Long k = 1; k < np; k++) {  // k = 0 is the self block
    const Long to = (rank + k) % np, from = (rank - k + np) % np;
    // The send and receive of one step are with *different* peers, so their chunk counts differ
    // and cannot share a lockstep loop -- doing that deadlocks, because whichever rank iterates
    // longer posts a zero-count receive that never gets a matching send. Post each direction's
    // chunks independently; a block's chunk count then matches at both ends, since both derive it
    // from the same size.
    const Long nreq = nchunk(sd[to + 1] - sd[to]) + nchunk(rd[from + 1] - rd[from]);
    sctl::ScratchBuf<MPI_Request> req(std::max<Long>(nreq, 1));
    Long m = 0;
    for (Long o = rd[from]; o < rd[from + 1]; o += CHUNK)
      MPI_Irecv(rp + o, (int)std::min<Long>(CHUNK, rd[from + 1] - o), MPI_BYTE, (int)from, 0, comm.GetMPI_Comm(), &req[m++]);
    for (Long o = sd[to]; o < sd[to + 1]; o += CHUNK)
      MPI_Isend(sp + o, (int)std::min<Long>(CHUNK, sd[to + 1] - o), MPI_BYTE, (int)to, 0, comm.GetMPI_Comm(), &req[m++]);
    if (m) MPI_Waitall((int)m, &req[0], MPI_STATUSES_IGNORE);
  }
#endif
}

template <template <class...> class DevVec, class Policy>
void alltoallv(const Policy& pol, const void* sbuf, void* rbuf, const sctl::ScratchBuf<Long>& scnt,
               const sctl::ScratchBuf<Long>& rcnt, Long esz, const Comm& comm) {
  if constexpr (is_device_vector_v<DevVec<char>>) alltoallvDevice<DevVec>(pol, sbuf, rbuf, scnt, rcnt, esz, comm);
  else alltoallvHost(sbuf, rbuf, scnt, rcnt, esz, comm);
}


// Exchange staged through pooled scratch: MPI sees the same registered addresses on every call, at
// the price of a copy in and a copy out.
template <class T, template <class...> class DevVec, class Policy>
void exchangePooled(const Policy& pol, const DevVec<T>& src, Long nsrc, DevVec<T>& dst, Long ndst,
                    const sctl::ScratchBuf<Long>& scnt, const sctl::ScratchBuf<Long>& rcnt, const Comm& comm) {
  DeviceScratch<T, DevVec> xs(nsrc), xr(ndst);
  thrust::copy(pol, src.begin(), src.begin() + nsrc, xs.begin());
  alltoallv<DevVec>(pol, thrust::raw_pointer_cast(xs.data()), thrust::raw_pointer_cast(xr.data()), scnt, rcnt, (Long)sizeof(T), comm);
  dst.resize(ndst);
  thrust::copy(pol, xr.begin(), xr.end(), dst.begin());
}

/** Offsets of `cnt[0,n)`: `dsp[0,n]`, with `dsp[n]` the total, which is returned. */
inline Long scanv(sctl::Iterator<Long> dsp, sctl::ConstIterator<Long> cnt, Long n) {
  dsp[0] = 0;
  std::inclusive_scan(cnt, cnt + n, dsp + 1);
  return dsp[n];
}

// `dof` of a data set as `sum(ndata) / sum(nitem)` over ranks, as in sctl::Tree: a rank holding no
// items has nothing to divide by.
inline Long globalDof(Long ndata, Long nitem, const Comm& comm) {
  sctl::StaticArray<Long, 2> Nl, Ng;
  Nl[0] = ndata;
  Nl[1] = nitem;
  comm.Allreduce((sctl::ConstIterator<Long>)Nl, (sctl::Iterator<Long>)Ng, 2, sctl::CommOp::SUM);
  const Long dof = Ng[0] / std::max<Long>(Ng[1], 1);
  SCTL_ASSERT(ndata == nitem * dof);
  SCTL_ASSERT(Ng[0] == Ng[1] * dof);
  return dof;
}

// Binary lower_bound over v[lo, hi); std::lower_bound is host-only.
template <class T> SCTL_GPU_HD Long lowerBound(const T* v, Long lo, Long hi, const T& key) {
  while (lo < hi) {
    const Long m = lo + (hi - lo) / 2;
    if (v[m] < key) lo = m + 1;
    else            hi = m;
  }
  return lo;
}

enum class WalkMode { Count, Write };

// DFS pre-order walk over a sorted anchor range, from start_node to end_target (exclusive); the
// anchors are the forced leaves, the walk fills the complete-tree nodes between them. Caller drives
// i over [0, n+1]: pair 0 emits start_node, the trailing pair (i==n) targets end_target.
template <Integer DIM, WalkMode MODE> struct AnchorWalkFunctor {
  const Morton<DIM>* anchors;
  Long n;
  Morton<DIM> start_node, end_target;
  const Long* offsets;  // WalkMode::Write only
  Morton<DIM>* out;     // WalkMode::Write only

  SCTL_GPU_HD Long operator()(Long i) const {
    using NodeT = Morton<DIM>;
    const bool is_tail = (i == n);
    const NodeT target = is_tail ? end_target : anchors[i];
    NodeT current = (i == 0) ? start_node : anchors[i - 1];
    Long count = 0;
    NodeT* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[i];
    if (i == 0 && start_node < end_target) {  // pair 0 emits start_node; nothing if the range is empty
      if constexpr (MODE == WalkMode::Write) w[count] = current;
      ++count;
    }
    while (current < target) {
      const bool descend = current.depth < MAX_DEPTH && current.isAncestor(target);
      current = descend ? current.DFD(static_cast<uint8_t>(current.depth + 1)) : current.Next();
      if (is_tail && !(current < target)) break;  // end_target is a boundary marker, not ours to emit
      if constexpr (MODE == WalkMode::Write) w[count] = current;
      ++count;
    }
    return count;
  }
};

// Exclusive scan of counts[0,n) into `offsets`, returning the total the scan already summed --
// reading back the last offset and count avoids a second pass over counts just to total them.
template <class Policy, template <class...> class DevVec>
Long scanCounts(const Policy& pol, const DeviceScratch<Long, DevVec>& counts, DeviceScratch<Long, DevVec>& offsets, Long n) {
  thrust::exclusive_scan(pol, counts.begin(), counts.begin() + n, offsets.begin(), Long(0));
  Long tail[2] = {0, 0};
  thrust::copy(offsets.begin() + (n - 1), offsets.begin() + n, &tail[0]);
  thrust::copy(counts.begin() + (n - 1), counts.begin() + n, &tail[1]);
  return tail[0] + tail[1];
}

// Turn a runtime periodicity mask into a template parameter: `f` is called with the matching mask as
// a `std::integral_constant`, so the kernel it launches carries one `Morton::NbrList` emitter rather
// than all of them. Every 2^DIM mask is enumerated, partial ones (X|Z, ...) included.
template <Integer DIM, class F, sctl::PeriodicityT MASK = 0>
void dispatchPeriodicity(sctl::Periodicity periodicity, const F& f) {
  constexpr sctl::Periodicity PER = static_cast<sctl::Periodicity>(MASK);
  if (periodicity == PER) {
    f(std::integral_constant<sctl::Periodicity, PER>{});
    return;
  }
  if constexpr (MASK + 1 < (1 << DIM)) dispatchPeriodicity<DIM, F, sctl::PeriodicityT(MASK + 1)>(periodicity, f);
  else SCTL_ASSERT_MSG(false, "dispatchPeriodicity: periodicity has bits outside DIM");
}

// The anchor walk in two passes, so a caller that knows (or can bound) the output size can write
// straight into its own buffer -- e.g. pooled scratch -- instead of having the walk allocate one.
// Count pass: fills `offsets` (exclusive scan of the per-pair node counts) and returns the total.
template <Integer DIM, template <class...> class DevVec>
Long anchorWalkCount(DeviceScratch<Long, DevVec>& offsets, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const Long n_pairs = n + 1;
  const auto pol = scratch_policy<DevVec, Morton<DIM>>();
  DeviceScratch<Long, DevVec> counts(n_pairs);
  const AnchorWalkFunctor<DIM, WalkMode::Count> fc{anchors_ptr, n, start_node, end_target, nullptr, nullptr};
  if constexpr (is_device_vector_v<DevVec<Morton<DIM>>>) {
    thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(n_pairs), counts.begin(), fc);
    return scanCounts(pol, counts, offsets, n_pairs);
  } else {  // thrust's host backend is serial, so drive the walk with OpenMP instead
    Long* const cnt = thrust::raw_pointer_cast(counts.data());
    Long* const off = thrust::raw_pointer_cast(offsets.data());
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n_pairs; i++) cnt[i] = fc(i);
    off[0] = 0;  // omp_par::scan is exclusive and takes off[0] as the (unwritten) seed
    sctl::omp_par::scan(cnt, off, n_pairs);
    return off[n_pairs - 1] + cnt[n_pairs - 1];
  }
}

// Write pass: emit the walk into `out`, which must hold the count from anchorWalkCount.
template <Integer DIM, template <class...> class DevVec>
void anchorWalkWrite(Morton<DIM>* out, const Long* offsets, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const Long n_pairs = n + 1;
  const auto pol = scratch_policy<DevVec, Morton<DIM>>();
  const AnchorWalkFunctor<DIM, WalkMode::Write> fw{anchors_ptr, n, start_node, end_target, offsets, out};
  if constexpr (is_device_vector_v<DevVec<Morton<DIM>>>) {
    thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n_pairs, fw);
  } else {
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n_pairs; i++) fw(i);
  }
}

// The complete preorder tree over [start_node, end_target) with `anchors` as its forced leaves.
template <Integer DIM, template <class...> class DevVec>
void treeFromAnchors(DevVec<Morton<DIM>>& tree, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  DeviceScratch<Long, DevVec> offsets(n + 1);
  const Long total = anchorWalkCount<DIM, DevVec>(offsets, anchors_ptr, n, start_node, end_target);
  tree.resize(total);
  anchorWalkWrite<DIM, DevVec>(thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(offsets.data()), anchors_ptr, n, start_node, end_target);
}

/** Redistribute the sorted `v` (`n` elements) so this rank ends up with `Ntgt`, order preserved: the
 *  device counterpart of `Comm::PartitionN`. Counts are known locally once every rank's size is; the
 *  result arrives in the retained `dst` and is swapped into `v`. */
template <class T, template <class...> class DevVec, class Policy>
void partitionN(const Policy& pol, DevVec<T>& v, Long n, Long Ntgt, const Comm& comm, DevVec<T>& dst) {
  const Long np = comm.Size(), rank = comm.Rank();
  if (np == 1) {
    v.resize(Ntgt);
    return;
  }
#ifdef SCTL_HAVE_MPI
  sctl::ScratchBuf<Long> cnt(np), off(np + 1), tgt(np), toff(np + 1);
  comm.Allgather(sctl::Ptr2ConstItr<Long>(&n, 1), 1, cnt.begin(), 1);
  comm.Allgather(sctl::Ptr2ConstItr<Long>(&Ntgt, 1), 1, tgt.begin(), 1);
  SCTL_ASSERT(scanv(off.begin(), cnt.begin(), np) == scanv(toff.begin(), tgt.begin(), np));
  { // nothing crosses a rank boundary: the layout already is the target
    bool same = true;
    for (Long q = 0; q <= np; q++) same = same && (off[q] == toff[q]);
    if (same) return;
  }

  sctl::ScratchBuf<Long> scnt(np), rcnt(np);
  for (Long q = 0; q < np; q++) {  // overlap of my current block with q's target block, and vice versa
    scnt[q] = std::max<Long>(0, std::min(off[rank + 1], toff[q + 1]) - std::max(off[rank], toff[q]));
    rcnt[q] = std::max<Long>(0, std::min(off[q + 1], toff[rank + 1]) - std::max(off[q], toff[rank]));
  }
  exchangePooled(pol, v, n, dst, Ntgt, scnt, rcnt, comm);
  v.swap(dst);
  v.resize(Ntgt);
#endif
}

}  // namespace detail

// Tree linearization from sorted Morton codes: single-rank build + the distributed walk stage.
namespace detail_build {
using detail::WalkMode;
using detail::lowerBound;

// A slice of at most M points spanning the whole domain is one leaf: the root. Returns false if the
// caller still has a tree to build.
template <Integer DIM, template <class...> class DevVec>
bool rootOnlyTree(DevVec<Morton<DIM>>& tree, Long N, Long M, const Morton<DIM>& start_bnd, const Morton<DIM>& end_bnd) {
  if (!(N <= M && start_bnd == Morton<DIM>{} && end_bnd == Morton<DIM>{}.Next())) return false;
  tree.resize(1);
  tree[0] = Morton<DIM>{};
  return true;
}

// Split-leaf of pair (pt[i], pt[i+M]): child of their common ancestor holding pt[i+M]. The M-gap
// forces the split: a depth-d box holding both endpoints has M+1 > M particles, so it refines.
template <Integer DIM> struct SplitLeafFunctor {
  const MortonCode<DIM>* pt;
  Long M;
  SCTL_GPU_HD Morton<DIM> operator()(Long i) const {
    uint8_t d = pt[i].CommonAncestor(pt[i + M]).depth;
    if (d < MAX_DEPTH) ++d;
    return pt[i + M].Ancestor(d);
  }
};

// GPU build of the slice [start_bnd, end_bnd) from sorted codes: anchors from the (pt[i], pt[i+M])
// pairs, then the walk between consecutive anchors.
template <Integer DIM, template <class...> class DevVec>
void buildTreeGpu(DevVec<Morton<DIM>>& tree, const DevVec<MortonCode<DIM>>& pt_mid, Long M, Long N, Long base, Morton<DIM> start_bnd, Morton<DIM> end_bnd) {
  using NodeMIDT = Morton<DIM>;

  if (rootOnlyTree<DIM, DevVec>(tree, N, M, start_bnd, end_bnd)) return;

  // Phase 1: anchors from the pairs within pt_mid[base, base+N), then clip to [start_bnd, end_bnd).
  // The boundary leaves need no points from outside: start_bnd is itself the anchor of the pair
  // straddling the lower boundary, and the walk stops at end_bnd.
  const Long N_pairs = std::max<Long>(N - M, 0);
  const auto pol = detail::scratch_policy<DevVec, NodeMIDT>();
  DeviceScratch<NodeMIDT, DevVec> anchors(N_pairs);
  SplitLeafFunctor<DIM> f{thrust::raw_pointer_cast(pt_mid.data()) + base, M};
  auto in = thrust::make_transform_iterator(thrust::counting_iterator<Long>(0), f);
  auto uniq_end = anchors.begin() + detail::local_unique_copy(pol, in, N_pairs, anchors);
  auto a_begin = thrust::lower_bound(pol, anchors.begin(), uniq_end, start_bnd);
  auto a_end   = thrust::lower_bound(pol, a_begin,         uniq_end, end_bnd);
  const NodeMIDT* anchors_ptr = thrust::raw_pointer_cast(anchors.data()) + (a_begin - anchors.begin());
  const Long n_anchors = a_end - a_begin;

  // Phase 2: linearize over the n_anchors+1 gaps (+1 = trailing gap to end_bnd).
  detail::treeFromAnchors<DIM, DevVec>(tree, anchors_ptr, n_anchors, start_bnd, end_bnd);
}


// Per-chunk anchor walk: walks pt_mid[begin_t, end_t) between chunk-boundary anchors
// (start_bnd / end_bnd at the ends; else the split-leaf of (pt[begin], pt[begin+M])).
template <Integer DIM, WalkMode MODE> struct ChunkedWalkFunctor {
  const MortonCode<DIM>* pt_mid;
  Long N, M, nthreads;
  const Long* offsets;  // WalkMode::Write only
  Morton<DIM>* out;     // WalkMode::Write only
  Morton<DIM> start_bnd;  // rank's lower boundary anchor (ROOT on the first rank)
  Morton<DIM> end_bnd;    // rank's upper boundary, exclusive (root.Next() on the last rank)

  SCTL_GPU_HD Long operator()(Long tid) const {
    using NodeT = Morton<DIM>;
    const SplitLeafFunctor<DIM> split{pt_mid, M};

    const Long  begin_t      = (N *  tid     ) / nthreads;
    const Long  end_t        = (N * (tid + 1)) / nthreads;
    const bool  is_last      = (tid == nthreads - 1);
    const NodeT start_anchor = (tid == 0) ? start_bnd : split(begin_t);
    const NodeT end_anchor   = (is_last)  ? end_bnd   : split(end_t);
    const Long  idx_start    = (tid == 0) ? 0 : lowerBound(pt_mid, begin_t, begin_t + M, start_anchor.mid);
    const Long  idx_end      = (is_last)  ? N : lowerBound(pt_mid, end_t,   end_t   + M, end_anchor.mid);

    Long count = 0;
    NodeT* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[tid];

    NodeT m0 = start_anchor;
    // Emit the complete-tree nodes from m0 up to `target`, leaving m0 there.
    const auto walk_to = [&m0, &count, w](const NodeT& target) {
      while (m0 != target) {
        if constexpr (MODE == WalkMode::Write) w[count] = m0;
        ++count;
        if (m0.isAncestor(target)) m0 = m0.DFD(static_cast<uint8_t>(m0.depth + 1));
        else                       m0 = m0.Next();
      }
    };
    Long pt_idx = idx_start;
    while (pt_idx < idx_end - M) {
      const NodeT m_ = split(pt_idx);
      if (m_ == m0) {  // > M coincident codes: their MAX_DEPTH box cannot split; skip past the run
        pt_idx = lowerBound(pt_mid, pt_idx, idx_end, m0.Next().mid);
        continue;
      }
      walk_to(m_);
      m0 = m_;
      pt_idx = lowerBound(pt_mid, pt_idx, pt_idx + M, m0.mid);
    }
    walk_to(end_anchor);  // tail to end_anchor / sentinel
    return count;
  }
};

// GPU port of buildTreeCpuChunked: chunked walk (count + exclusive_scan + write) via
// ChunkedWalkFunctor, no anchor materialization.
template <Integer DIM, template <class...> class DevVec>
void buildTreeGpuChunked(DevVec<Morton<DIM>>& tree, const DevVec<MortonCode<DIM>>& pt_mid, Long M, Long N, Long base, Morton<DIM> start_bnd, Morton<DIM> end_bnd) {  // walks pt_mid[base, base+N) (+M halo slack beyond)
  if (rootOnlyTree<DIM, DevVec>(tree, N, M, start_bnd, end_bnd)) return;

  const Long min_chunk = std::max<Long>(4 * M + 1, 64);
  const Long nthreads  = std::clamp<Long>(N / min_chunk, 1, 65536);

  const auto pol = detail::scratch_policy<DevVec, MortonCode<DIM>>();
  DeviceScratch<Long, DevVec> counts(nthreads), offsets(nthreads);
  ChunkedWalkFunctor<DIM, WalkMode::Count> fc{thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads, nullptr, nullptr, start_bnd, end_bnd};
  thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(nthreads), counts.begin(), fc);
  const Long total = detail::scanCounts(pol, counts, offsets, nthreads);

  tree.resize(total);
  ChunkedWalkFunctor<DIM, WalkMode::Write> fw{
      thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads, thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(tree.data()), start_bnd, end_bnd};
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), nthreads, fw);
}


// One Long per 64-byte cache line — avoids false sharing on per-thread counters.
struct alignas(64) PaddedLong {
  Long v;
  char pad[64 - sizeof(Long)];
};
static_assert(sizeof(PaddedLong) == 64);

// CPU build from sorted codes: each thread walks its pt_mid chunk into a per-thread (NUMA-local)
// ScratchBuf; after a barrier the counts are prefix-summed and slices copied into `tree`.
// Single-pass on purpose: a count-then-rewrite pass would re-read pt_mid cache-cold.
template <Integer DIM, template <class...> class DevVec>
void buildTreeCpuChunked(DevVec<Morton<DIM>>& tree, const DevVec<MortonCode<DIM>>& pt_mid, Long M, Long N, Long base, Morton<DIM> start_bnd, Morton<DIM> end_bnd) {  // walks pt_mid[base, base+N) (+M halo slack beyond)
  using NodeMIDT = Morton<DIM>;
  if (rootOnlyTree<DIM, DevVec>(tree, N, M, start_bnd, end_bnd)) return;

  // Cap threads so each chunk has well over M particles (so `begin + M` stays in-bounds).
  const Integer max_threads = SCTL_GET_MAX_THREADS();
  const Long min_chunk = std::max<Long>(4 * M + 1, 1024);
  const Integer nthreads = std::clamp<Integer>(N / min_chunk, 1, max_threads);

  // Upper bound: ~(MAX_DEPTH+1) nodes/leaf, chunk_size/M leaves/chunk, 4x slack.
  const Long chunk_size_max = (N + nthreads - 1) / nthreads;
  const Long max_emits = 4 * chunk_size_max * (MAX_DEPTH + 1) / std::max<Long>(1, M) + 4 * (MAX_DEPTH + 1) * (Long(1) << DIM) + 16;  // constant term: boundary anchors can sit at MAX_DEPTH

  sctl::ScratchBuf<PaddedLong> local_sizes(nthreads);  // padded: concurrent per-thread writes
  sctl::ScratchBuf<Long> offsets(nthreads);            // written once by `single`, read-only after
  sctl::ScratchBuf<Long> zero_offsets(nthreads);       // functor reads `offsets[tid] == 0`
  for (Integer t = 0; t < nthreads; ++t) {
    zero_offsets[t] = 0;
    local_sizes[t].v = 0;  // a smaller team than asked for would leave the rest unwritten
  }

  #pragma omp parallel num_threads(nthreads)
  {
    const Integer tid = SCTL_GET_THREAD_NUM();
    sctl::ScratchBuf<NodeMIDT> buf(max_emits);  // NUMA-local: first-touched on this thread's node
    const ChunkedWalkFunctor<DIM, WalkMode::Write> fw{thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads, &zero_offsets[0], &buf[0], start_bnd, end_bnd};
    const Long count = fw(tid);
    SCTL_ASSERT_MSG(count <= max_emits, "chunked walk: emitted more nodes than the bound allows");
    local_sizes[tid].v = count;

    #pragma omp barrier
    #pragma omp single
    {
      std::transform_exclusive_scan(local_sizes.begin(), local_sizes.end(), offsets.begin(), Long(0), std::plus<Long>(), [](const PaddedLong& s) { return s.v; });
      tree.resize(offsets[nthreads - 1] + local_sizes[nthreads - 1].v);
    }

    NodeMIDT* out_ptr = thrust::raw_pointer_cast(tree.data()) + offsets[tid];
    for (Long i = 0; i < count; ++i) out_ptr[i] = buf[i];
  }
}

}  // namespace detail_build

// Splitter selection for the distributed sort.
namespace detail_determineSplitters {
using detail::is_device_vector_v;

// Exact-rank splitters (np-1, into the caller's buffer) for the distributed sort, replicated on
// every rank: seed the cuts from per-rank data boundaries, then iterate probe -> gather candidates
// -> exact global ranks -> refine until each is within tol. Un-splittable cuts (target inside a
// duplicate run wider than tol) are frozen at the nearest achievable endpoint.
template <class Type, template <class...> class DevVec>
void determineSplitters(sctl::ScratchBuf<Type>& splitters, const DevVec<Type>& pt, const Comm& comm) {
  constexpr Integer MAXIT = 50;
  constexpr Integer budget = 16;
  constexpr Long MAXP = 2 * budget;  // probes per rank per round
  constexpr double tolfrac = 0.02;

  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long ns = np - 1;
  SCTL_ASSERT(splitters.Dim() == ns);
  if (!ns) return;

  const Long Nl = static_cast<Long>(pt.size());
  const Long Ng = [&Nl,&comm](){
    Long g;
    comm.Allreduce(sctl::Ptr2ConstItr<Long>(&Nl,1), sctl::Ptr2Itr<Long>(&g,1), 1, sctl::CommOp::SUM);
    return g;
  }();
  if (!Ng) return;

  const Long tol = std::max<Long>(1, Long(tolfrac * double(Ng) / double(np)));
  uint64_t rng = uint64_t(rank) * 0x9e3779b97f4a7c15ULL + 0x123456789abcdefULL;
  const auto next = [&rng]() {
    uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };

  const auto local_ranks = [&pt]
                           (sctl::Iterator<Long> r, sctl::ConstIterator<Type> q, const Long n, const bool upper = false) {  // batched binary search
    if (!n) return;
    if constexpr (is_device_vector_v<DevVec<Type>>) {  // device: thrust batched binary search
      DeviceScratch<Type, DevVec> q_d(n);
      DeviceScratch<Long, DevVec> r_d(n);
      const auto pol = detail::scratch_policy<DevVec, Type>();
      thrust::copy(q, q + n, q_d.begin());
      if (upper) thrust::upper_bound(pol, pt.begin(), pt.end(), q_d.begin(), q_d.end(), r_d.begin());
      else       thrust::lower_bound(pol, pt.begin(), pt.end(), q_d.begin(), q_d.end(), r_d.begin());
      thrust::copy(r_d.begin(), r_d.end(), r);
    } else {                                                 // host: omp binary search (thrust host backend is serial)
      const Type* pp = thrust::raw_pointer_cast(pt.data());
      const Long m = (Long)pt.size();
      #pragma omp parallel for schedule(static) if(n > 512)
      for (Long i = 0; i < n; i++) r[i] = (upper ? std::upper_bound(pp, pp + m, q[i]) : std::lower_bound(pp, pp + m, q[i])) - pp;
    }
  };

  sctl::ScratchBuf<Type> bracket(2*ns);
  sctl::ScratchBuf<Long> bracket_rl(2*ns), bracket_rg(2*ns);
  [&comm,np,ns,Nl,Ng,&pt, &bracket,&bracket_rl,&bracket_rg,&local_ranks]() { // Set bracket, bracket_rl, bracket_rg
    sctl::ScratchBuf<Long> rcnt(np), rdsp(np);
    const sctl::StaticArray<Long,1> scnt{Nl ? 2 : 0};
    comm.Allgather(scnt+0, 1, rcnt.begin(), 1);
    std::exclusive_scan(rcnt.begin(), rcnt.end(), rdsp.begin(), Long(0));

    const sctl::StaticArray<Type,2> sbuf{Nl?pt[0]:Type{}, Nl?pt[Nl-1]:Type{}};
    sctl::ScratchBuf<Type> bnd(rdsp[np-1] + rcnt[np-1]), bnd2(rdsp[np-1] + rcnt[np-1]);
    comm.Allgatherv(sbuf+0, scnt[0], bnd.begin(), rcnt.begin(), rdsp.begin());
    sctl::omp_par::merge_sort(bnd.begin(), bnd.end());
    const Long B = sctl::omp_par::dedup_sorted(bnd.begin(), bnd2.begin(), bnd.Dim());

    sctl::ScratchBuf<Long> lr_b(B), gr_b(B); // each boundary's local then exact global rank
    local_ranks(lr_b.begin(), bnd2.begin(), B);
    comm.Allreduce(lr_b.begin(), gr_b.begin(), B, sctl::CommOp::SUM);

    #pragma omp parallel for schedule(static) if(ns > 512)
    for (Long i = 0; i < ns; i++) {                              // straddling boundary pair for target rank t
      const Long t = (i + 1) * Ng / np;
      const Long up = std::min(B-1, std::max<Long>(1, std::lower_bound(gr_b.begin(), gr_b.begin()+B, t) - gr_b.begin()));
      const Long lo = std::max<Long>(0, up - 1);                 // lo>=0 even when B==1 (degenerate all-equal data)
      bracket[i*2+0] = bnd2[lo];
      bracket_rl[i*2+0] = lr_b[lo];
      bracket_rg[i*2+0] = gr_b[lo];
      bracket[i*2+1] = bnd2[up];
      bracket_rl[i*2+1] = lr_b[up];
      bracket_rg[i*2+1] = gr_b[up];
    }
  }();

  sctl::ScratchBuf<char> state(ns);
  enum CutState : char { ACTIVE = 0, DONE = 1, DONE_UPPER = 2 };
  std::fill(state.begin(), state.end(), (char)ACTIVE);


  const auto gather_pt = [&pt]
                         (sctl::Iterator<Type> out, sctl::ConstIterator<Long> idxs, Long n) {  // out[0,n) = pt[idxs[0,n)]: one gather (+D2H on device)
    if (!n) return;
    DeviceScratch<Long, DevVec> idx_d(n);
    DeviceScratch<Type, DevVec> gv_d(n);
    thrust::copy(idxs, idxs + n, idx_d.begin());
    thrust::gather(detail::scratch_policy<DevVec, Type>(), idx_d.begin(), idx_d.end(), pt.begin(), gv_d.begin());
    thrust::copy(gv_d.begin(), gv_d.end(), out);
  };

  // out: idxs (this rank's chosen local indices), local_cand (their point values); returns their
  // count, at most MAXP. budget is constexpr -> no capture.
  const auto probe = [ns,np,Ng,&state,&bracket,&bracket_rl,&bracket_rg,&next,&gather_pt]
                     (sctl::ScratchBuf<Long>& idxs, sctl::ScratchBuf<Type>& local_cand) -> Long {
    const double budget_ = [&state,ns]() {  // concentrate the round's budget onto the shrinking active set
      Long active_cnt = 0;
      for (Long i = 0; i < ns; i++) active_cnt += (state[i] ? 0 : 1);
      return double(budget) * double(ns) / std::max<Long>(1, active_cnt);
    }();

    Long m = 0;
    const Long start = Long(next() % (uint64_t)ns);
    for (Long j = 0; j < ns && m < MAXP; j++) {
      const Long i = (start + j) % ns;
      const Long rl0 = bracket_rl[i*2+0], rl1 = bracket_rl[i*2+1];
      const Long rg0 = bracket_rg[i*2+0], rg1 = bracket_rg[i*2+1];
      if (state[i] || rl1 == rl0) continue;

      const double share = double(rl1 - rl0) / double(rg1 - rg0);
      const double u = double(next() >> 11) * 0x1p-53; // uniform [0,1): top 53 bits scaled by 2^-53
      if (u >= std::min(1.0, double(budget_) * share)) continue;

      const Long opt_rank = (i + 1) * Ng / np;
      const Long idx = rl0 + (rl1 - rl0) * (opt_rank - rg0) / (rg1 - rg0); // interpolate to the target
      idxs[m++] = std::min(rl1 - 1, std::max(rl0, idx));
    }
    gather_pt(local_cand.begin(), idxs.begin(), m);
    return m;
  };

  // in: local_cand[0,mloc) (this rank's probes); out: cand (all ranks' probes ++ active brackets, sorted+deduped). ret: |cand|.
  const auto gather_candidates = [&comm,np,ns,&state,&bracket]
                                 (sctl::ScratchBuf<Type>& cand, const sctl::ScratchBuf<Type>& local_cand, const Long mloc) -> Long {
    sctl::ScratchBuf<Long> cntb(np), dspb(np);
    comm.Allgather(sctl::Ptr2ConstItr<Long>(&mloc,1), 1, cntb.begin(), 1);
    std::exclusive_scan(cntb.begin(), cntb.end(), dspb.begin(), Long(0));
    Long S = dspb[np-1] + cntb[np-1];

    sctl::ScratchBuf<Type> cand_raw(S + 2*ns);  // gather + sort here, then dedup out-of-place into cand
    comm.Allgatherv((mloc ? local_cand.begin() : sctl::NullIterator<Type>()), mloc,
                     cand_raw.begin(), cntb.begin(), dspb.begin());
    for (Long i = 0; i < ns; i++) if (!state[i]) { // append active brackets
      cand_raw[S++] = bracket[i*2+0];
      cand_raw[S++] = bracket[i*2+1];
    }

    SCTL_ASSERT(cand.Dim() >= S);
    sctl::omp_par::merge_sort(cand_raw.begin(), cand_raw.begin() + S);
    return sctl::omp_par::dedup_sorted(cand_raw.begin(), cand.begin(), S);
  };

  // in: cand, S; out: lr, gr sized S+ns. gr[0,S) = exact global ranks of cand;
  // gr[S+i] = global upper_bound rank of bracket[i*2+0] (end of blo's duplicate run), folded into the same Allreduce.
  const auto global_ranks = [&comm,ns,&local_ranks,&bracket]
                            (sctl::ScratchBuf<Long>& lr, sctl::ScratchBuf<Long>& gr, const sctl::ScratchBuf<Type>& cand, Long S) {
    SCTL_ASSERT(lr.Dim() >= S+ns);
    SCTL_ASSERT(gr.Dim() >= S+ns);

    sctl::ScratchBuf<Type> blo(ns);
    for (Long i = 0; i < ns; i++) blo[i] = bracket[i*2+0];
    local_ranks(lr.begin(), cand.begin(), S);
    local_ranks(lr.begin()+S, blo.begin(), ns, /*upper*/true);
    comm.Allreduce(lr.begin(), gr.begin(), S+ns, sctl::CommOp::SUM);
  };

  // in: cand, lr, gr (sized S+ns), S; out: splitters + updated bracket*/state. ret: whether any cut is still active.
  const auto refine = [/*out:*/ &splitters,&state,&bracket,&bracket_rl,&bracket_rg,  /*in:*/ ns,Ng,np,tol]
                      (const sctl::ScratchBuf<Type>& cand, const sctl::ScratchBuf<Long>& lr, const sctl::ScratchBuf<Long>& gr, Long S) -> bool {
    bool anyactive = false;
    #pragma omp parallel for schedule(static) reduction(||:anyactive) if(ns > 512)
    for (Long i = 0; i < ns; i++) { // refine each active bracket toward its target rank
      if (state[i]) continue;
      const Long t = (i + 1) * Ng / np;
      const Long up = std::lower_bound(gr.begin(), gr.begin()+S, t) - gr.begin(), lo = up - 1;

      const Long errlo = (lo >= 0) ? t - gr[lo] : t;
      const Long errup = (up < S) ? gr[up] - t : (Ng - t);
      splitters[i] = (errlo <= errup) ? cand[std::max<Long>(0,lo)] : cand[std::min<Long>(S-1,up)];
      if (lo < 0 || up >= S || std::min(errlo,errup) <= tol) {
        state[i] = DONE;
        continue;
      }

      if (gr[up] - gr[lo] < bracket_rg[i*2+1] - bracket_rg[i*2+0]) {  // bracket shrank: tighten toward target
        bracket[i*2+0]    = cand[lo];
        bracket[i*2+1]    = cand[up];
        bracket_rl[i*2+0] = lr[lo];
        bracket_rl[i*2+1] = lr[up];
        bracket_rg[i*2+0] = gr[lo];
        bracket_rg[i*2+1] = gr[up];
        anyactive = true;
        continue;
      }

      // no shrink: cand[lo]==blo. Exact un-splittable test on blo's duplicate run [L,U).
      const Long L = gr[lo], U = gr[S+i];  // L = global lower_bound(blo), U = global upper_bound(blo)
      if (U <= t) { // run ends before target -> undersampled this round, keep probing
        anyactive = true;
        continue;
      }
      // run straddles target: nearest endpoint is the best achievable.
      if (t - L <= U - t) {  // nearer endpoint L: splitter = blo (value in hand)
        splitters[i] = cand[lo];
        state[i] = DONE;
      }
      else                  state[i] = DONE_UPPER;                      // nearer endpoint U: splitter = successor of blo (resolved at loop end)
    }
    return anyactive;
  };

  sctl::ScratchBuf<Long> idxs(MAXP);
  sctl::ScratchBuf<Type> local_cand(MAXP), cand(np * MAXP + 2 * ns);  // cand: every rank's probes plus the active brackets
  for (Integer it = 0; it < MAXIT; it++) { // iterate: [ probe -> gather -> global-rank -> refine ] until every cut is within tol
    const Long mloc = probe(idxs, local_cand);
    const Long S = gather_candidates(cand, local_cand, mloc);
    sctl::ScratchBuf<Long> gr(S+ns), lr(S+ns);
    global_ranks(lr, gr, cand, S);
    if (!refine(cand, lr, gr, S)) break;
  }

  { // resolve upper-side frozen cuts: splitter = successor of blo (smallest global value > blo), one MIN-Allreduce
    Long m = 0;
    sctl::ScratchBuf<Long> up_cuts(ns);
    for (Long i = 0; i < ns; i++) if (state[i] == DONE_UPPER) up_cuts[m++] = i;
    if (m) {
      sctl::ScratchBuf<Long> uidx(m);
      { // uidx <-- local index of first point > blo
        sctl::ScratchBuf<Type> bvals(m);
        for (Long k = 0; k < m; k++) bvals[k] = bracket[up_cuts[k]*2+0];
        local_ranks(uidx.begin(), bvals.begin(), m, /*upper*/true);
      }

      sctl::ScratchBuf<Type> sloc(m);
      { // sloc <-- successor value pt[uidx] (or bhi where this rank has no point > blo)
        sctl::ScratchBuf<Long> gidx(m);
        for (Long k = 0; k < m; k++) gidx[k] = std::min(uidx[k], Nl-1);
        if (Nl > 0) gather_pt(sloc.begin(), gidx.begin(), m);
        for (Long k = 0; k < m; k++) sloc[k] = (Nl > 0 && uidx[k] < Nl) ? sloc[k] : bracket[up_cuts[k]*2+1];
      }

      sctl::ScratchBuf<Type> sglob(m);
      comm.Allreduce<sctl::CommOp::MIN>(sloc.begin(), sglob.begin(), m);
      for (Long k = 0; k < m; k++) splitters[up_cuts[k]] = sglob[k];
    }
  }
}

}  // namespace detail_determineSplitters

// 2:1 balance closure rule (leaf form of Tree::UpdateRefinement's touching-neighbor rule): every
// same-depth neighbor octant of a non-leaf node must exist, so that neighbor's parent must be
// non-leaf too. Everything below is shared by both balance schemes: the predicates, the boundary
// complete-tree fill the closure is seeded from.
namespace detail_balance21 {
using detail::AnchorWalkFunctor;
using detail::WalkMode;
using detail::scratch_policy;

// Longest possible fill (zero anchors): at most 2^DIM nodes per level.
template <Integer DIM> constexpr Long kCompleteTreeMax = (Long)MAX_DEPTH * ((Long)1 << DIM) + 1;

// Coarsest complete-tree nodes filling the Morton interval [start_node, end_target). The functor
// returns the node count it wrote.
template <Integer DIM, template <class...> class DevVec>
Long completeTree(Morton<DIM>* out, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const auto pol = scratch_policy<DevVec, Morton<DIM>>();
  DeviceScratch<Long, DevVec> off(1), cnt(1);
  thrust::fill(pol, off.begin(), off.end(), Long(0));
  const AnchorWalkFunctor<DIM, WalkMode::Write> fw{nullptr, 0, start_node, end_target, thrust::raw_pointer_cast(off.data()), out};
  thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(1), cnt.begin(), fw);
  Long n = 0;
  thrust::copy(cnt.begin(), cnt.end(), &n);
  SCTL_ASSERT_MSG(n <= kCompleteTreeMax<DIM>, "completeTree: output exceeded the fill bound.");
  return n;
}

// True for nodes that have children (the next node in walk order is a descendant).
template <Integer DIM> struct NonLeafPred {
  const Morton<DIM>* tree;
  Long Nn;
  Morton<DIM> next_first;  // walk-order successor of tree[Nn-1]
  SCTL_GPU_HD bool operator()(Long i) const {
    return tree[i].isAncestor((i + 1 < Nn) ? tree[i + 1] : next_first);
  }
};

template <Integer DIM> struct InvalidDepthPred {
  SCTL_GPU_HD bool operator()(const Morton<DIM>& m) const { return m.depth == Morton<DIM>::INVALID_DEPTH; }
};


// First child of a non-leaf node (same mid, one level deeper), or INVALID outside [lo,hi).
template <Integer DIM> struct FirstChildInSlice {
  Morton<DIM> lo, hi;
  SCTL_GPU_HD Morton<DIM> operator()(const Morton<DIM>& s) const {
    Morton<DIM> inv;
    inv.depth = Morton<DIM>::INVALID_DEPTH;
    if (s.Depth() >= MAX_DEPTH) return inv;
    const Morton<DIM> c(s.mid, (uint8_t)(s.Depth() + 1));
    if (c < lo || !(c < hi)) return inv;
    return c;
  }
};

/** `tree`, this rank's slice, completed to the whole domain by the ancestors of its two boundaries,
 *  sorted. `full` must hold `tree.size() + 2 * kCompleteTreeMax`; returns the count written. */
template <Integer DIM, template <class...> class DevVec, class Policy>
Long completeSlice(const Policy& pol, const DevVec<Morton<DIM>>& tree, const sctl::ScratchBuf<Morton<DIM>>& mins, Long rank, Long np, Morton<DIM> end_target, DeviceScratch<Morton<DIM>, DevVec>& full) {
  using NodeT = Morton<DIM>;
  constexpr Long BND = kCompleteTreeMax<DIM>;
  const Long Nn = (Long)tree.size();
  DeviceScratch<NodeT, DevVec> lf(rank > 0 ? BND : 0), rt(rank + 1 < np ? BND : 0);
  Long nl = 0, nr = 0;
  if (rank > 0) nl = completeTree<DIM, DevVec>(thrust::raw_pointer_cast(lf.data()), NodeT{}, mins[rank]);
  if (rank + 1 < np) nr = completeTree<DIM, DevVec>(thrust::raw_pointer_cast(rt.data()), end_target, NodeT{}.Next());
  thrust::copy(pol, lf.begin(), lf.begin() + nl, full.begin());
  thrust::copy(pol, tree.begin(), tree.end(), full.begin() + nl);
  thrust::copy(pol, rt.begin(), rt.begin() + nr, full.begin() + nl + Nn);
  return nl + Nn + nr;
}

/** Leaves of the balanced non-leaf set `S[0,n)`: the walk between the first children of consecutive
 *  non-leaf nodes, within [lo, hi). */
template <Integer DIM, template <class...> class DevVec, class Policy>
void leavesFromNonLeaf(const Policy& pol, DevVec<Morton<DIM>>& tree, const Morton<DIM>* S, Long n, Morton<DIM> lo, Morton<DIM> hi) {
  using It = detail::ScratchIterator<Morton<DIM>, DevVec>;
  DeviceScratch<Morton<DIM>, DevVec> anch(n);
  thrust::transform(pol, It(const_cast<Morton<DIM>*>(S)), It(const_cast<Morton<DIM>*>(S)) + n, anch.begin(), FirstChildInSlice<DIM>{lo, hi});
  const Long na = thrust::remove_if(pol, anch.begin(), anch.begin() + n, InvalidDepthPred<DIM>{}) - anch.begin();
  detail::treeFromAnchors<DIM>(tree, thrust::raw_pointer_cast(anch.data()), na, lo, hi);
}

}  // namespace detail_balance21



// Balance for the host backend: extract the non-leaf set, close it with sctl's Balance21 (OpenMP,
// includes its own redistribute), then rebuild the leaves. Faster than the thrust closure below on
// host vectors and far slower on the device, hence the split by backend in buildTreeDist.
namespace detail_balance21_host {
using detail_balance21::NonLeafPred;
using detail_balance21::kCompleteTreeMax;
using detail_balance21::completeSlice;
using detail_balance21::leavesFromNonLeaf;

template <Integer DIM, template <class...> class DevVec>
void balanceTreeDist(DevVec<Morton<DIM>>& tree, const sctl::ScratchBuf<Morton<DIM>>& mins, const Comm& comm, sctl::Periodicity periodicity) {
  using NodeT = Morton<DIM>;
  const auto pol = detail::scratch_policy<DevVec, NodeT>();
  const Long rank = comm.Rank();
  const Long np = comm.Size();
  const NodeT end_target = (rank + 1 < np) ? mins[rank + 1] : NodeT{}.Next();
  const Long Nn = (Long)tree.size();

  sctl::Vector<NodeT> S;  // passed straight to Balance21 (in/out)
  { // Balance21 builds a tree from the root, so every node's ancestors must be present: extend the
    // slice to the whole domain, then take its non-leaf nodes. thrust's host backend is serial, so
    // compact with OpenMP straight into S.
    DeviceScratch<NodeT, DevVec> full(Nn + 2 * kCompleteTreeMax<DIM>);
    const Long Nf = completeSlice<DIM, DevVec>(pol, tree, mins, rank, np, end_target, full);
    const NonLeafPred<DIM> is_nonleaf{thrust::raw_pointer_cast(full.data()), Nf, NodeT{}.Next()};
    const NodeT* const fp = thrust::raw_pointer_cast(full.data());
    const Integer nt = SCTL_GET_MAX_THREADS();
    sctl::ScratchBuf<Long> dsp(nt + 1);
    std::fill(dsp.begin(), dsp.end(), Long(0));  // a smaller team leaves its tail unwritten
    #pragma omp parallel num_threads(nt)
    {
      const Integer tid = SCTL_GET_THREAD_NUM();
      Long c = 0;
      for (Long i = Nf * tid / nt; i < Nf * (tid + 1) / nt; i++) c += is_nonleaf(i);
      dsp[tid + 1] = c;
    }
    std::inclusive_scan(dsp.begin() + 1, dsp.end(), dsp.begin() + 1);
    S.ReInit(dsp[nt]);
    #pragma omp parallel num_threads(nt)
    {
      const Integer tid = SCTL_GET_THREAD_NUM();
      Long o = dsp[tid];
      for (Long i = Nf * tid / nt; i < Nf * (tid + 1) / nt; i++) if (is_nonleaf(i)) S[o++] = fp[i];
    }
  }

  { // sctl's balance21 over the non-leaf set (host, OpenMP); it also redistributes by mins
    sctl::tree_detail::Balance21(S, mins.begin(), comm, periodicity);
  }

  leavesFromNonLeaf<DIM, DevVec>(pol, tree, &S[0], S.Dim(), mins[rank], end_target);  // S holds at least the root
}

}  // namespace detail_balance21_host

// Balance for the device backend: everything on the device. Local closure over the sorted non-leaf set
// (ClosureFrontier), then redistribute by `mins` and dedup (CUDA-aware MPI straight from device
// buffers), then rebuild the leaves. Nothing crosses PCIe.
namespace detail_balance21_gpu {
using detail::local_sort;

// Expand one frontier node: for each distinct parent-neighbor of its 3^DIM same-depth neighbors
// (the p2n map, in device scratch: <= 2^DIM of them), emit it if absent from the sorted non-leaf set S.
template <Integer DIM, sctl::Periodicity PER> struct ParentNbrSearch {
  static constexpr Integer MAX_CHILD = (1u << DIM);
  static constexpr Integer K = sctl::pow<DIM, Integer>(3);
  const Morton<DIM>* F;      // frontier nodes to expand
  const Morton<DIM>* S;      // sorted non-leaf set to search
  Long ns;
  const Integer* p_nbr_lst;  // distinct parent-neighbors per child slot (K stride)
  const Integer* p_nbr_cnt;
  Long base;
  Morton<DIM>* out;          // MAX_CHILD slots per frontier node

  SCTL_GPU_HD void operator()(Long t) const {
    using NodeT = Morton<DIM>;
    NodeT inv;
    inv.depth = NodeT::INVALID_DEPTH;
    NodeT* const w = out + (t - base) * MAX_CHILD;
    const NodeT s = F[t];
    const Integer d = s.Depth();
    Integer j = 0;
    if (d) {
      const NodeT p = s.Ancestor((uint8_t)(d - 1));
      const auto pnbrs = p.template NbrList<PER>((uint8_t)(d - 1));
      const Integer p2n = s.Path2Node();
      for (; j < p_nbr_cnt[p2n]; j++) {
        const NodeT q = pnbrs[p_nbr_lst[p2n * K + j]];
        if (q.Depth() == NodeT::INVALID_DEPTH) {
          w[j] = inv;
          continue;
        }
        const Long lo = detail::lowerBound(S, Long(0), ns, q);
        w[j] = (lo < ns && !(S[lo] < q) && !(q < S[lo])) ? inv : q;
      }
    }
    for (; j < MAX_CHILD; j++) w[j] = inv;
  }
};

// Expand only the nodes added in the previous round, looking their parent-neighbors up in the
// non-leaf set.
template <Integer DIM, template <class...> class DevVec>
void ClosureFrontier(DevVec<Morton<DIM>>& S, sctl::Periodicity periodicity) {
  using NodeT = Morton<DIM>;
  constexpr Integer K = sctl::pow<DIM, Integer>(3);
  constexpr Integer MAX_CHILD = (1u << DIM);
  const auto pol = detail::scratch_policy<DevVec, NodeT>();
  DeviceScratch<Integer, DevVec> pl_d(MAX_CHILD * K), pc_d(MAX_CHILD);
  { // distinct parent-neighbors per child slot, from sctl's nbr_path table
    sctl::ScratchBuf<Integer> pl(MAX_CHILD * K), pc(MAX_CHILD);
    for (Integer i = 0; i < MAX_CHILD * K; i++) pl[i] = 0;
    for (Integer i = 0; i < MAX_CHILD; i++) pc[i] = 0;
    const auto& tbl = sctl::tree_detail::nbr_path_table<DIM>();
    for (Integer i = 0; i < MAX_CHILD; i++) {
      for (Integer k = 0; k < K; k++) {
        const Integer v = tbl[i][k].p_nbr;
        bool seen = false;
        for (Integer j = 0; j < pc[i]; j++) seen |= (pl[i * K + j] == v);
        if (!seen) pl[i * K + pc[i]++] = v;
      }
    }
    thrust::copy(pl.begin(), pl.end(), pl_d.begin());
    thrust::copy(pc.begin(), pc.end(), pc_d.begin());
  }

  // Per-round buffers come from the scratch pool: `buf`, `add` (at most MAX_CHILD per frontier
  // node) and the merge target. `F` and `S` outlive a round, so they grow geometrically.
  const auto grow = [](DevVec<NodeT>& v, Long need) {
    if ((Long)v.size() < need) v.resize(std::max<Long>(need, 2 * (Long)v.size()));
  };
  DevVec<NodeT>& F = detail::PersistentBuffer<NodeT, DevVec, detail::Buf::Frontier>();
  F.resize(S.size());
  thrust::copy(pol, S.begin(), S.end(), F.begin());
  Long ns = (Long)S.size(), nf = ns;
  for (Integer round = 0; round < 4 * MAX_DEPTH; round++) {
    if (!nf) break;
    const Long chunk = std::min<Long>(nf, 4000000 / MAX_CHILD + 1);
    DeviceScratch<NodeT, DevVec> buf(chunk * MAX_CHILD), add(nf * MAX_CHILD);

    Long nadd = 0;
    for (Long c0 = 0; c0 < nf; c0 += chunk) {
      const Long nc = std::min<Long>(chunk, nf - c0);
      detail::dispatchPeriodicity<DIM>(periodicity, [&pol, c0, nc, &F, &S, ns, &pl_d, &pc_d, &buf](auto per_c) {
        thrust::for_each_n(pol, thrust::counting_iterator<Long>(c0), nc, ParentNbrSearch<DIM, decltype(per_c)::value>{
            thrust::raw_pointer_cast(F.data()), thrust::raw_pointer_cast(S.data()), ns,
            thrust::raw_pointer_cast(pl_d.data()), thrust::raw_pointer_cast(pc_d.data()), c0, thrust::raw_pointer_cast(buf.data())});
      });
      const Long nkeep = thrust::remove_if(pol, buf.begin(), buf.begin() + nc * MAX_CHILD, detail_balance21::InvalidDepthPred<DIM>{}) - buf.begin();
      thrust::copy(pol, buf.begin(), buf.begin() + nkeep, add.begin() + nadd);
      nadd += nkeep;
    }
    if (!nadd) break;
    local_sort(pol, add, nadd);
    nadd = detail::local_unique(pol, add, nadd);

    { // merge S and the new nodes, then hand the result back to S
      DeviceScratch<NodeT, DevVec> m(ns + nadd);
      thrust::merge(pol, S.begin(), S.begin() + ns, add.begin(), add.begin() + nadd, m.begin());
      grow(S, ns + nadd);
      thrust::copy(pol, m.begin(), m.end(), S.begin());
    }
    grow(F, nadd);  // next round expands only the new nodes
    thrust::copy(pol, add.begin(), add.begin() + nadd, F.begin());
    ns += nadd;
    nf = nadd;
  }
  S.resize(ns);
}


// Whole 2:1 balance on the device: extract the non-leaf set, close it, redistribute, rebuild leaves.
template <Integer DIM, template <class...> class DevVec>
void balanceTreeDist(DevVec<Morton<DIM>>& tree, const sctl::ScratchBuf<Morton<DIM>>& mins, const Comm& comm, sctl::Periodicity periodicity) {
  using NodeT = Morton<DIM>;
  const Long rank = comm.Rank(), np = comm.Size();
  const NodeT end_target = (rank + 1 < np) ? mins[rank + 1] : NodeT{}.Next();
  const Long Nn = (Long)tree.size();

  const auto pol = detail::scratch_policy<DevVec, NodeT>();
  DevVec<NodeT>& S = detail::PersistentBuffer<NodeT, DevVec, detail::Buf::Closure>();
  { // extend the slice to the whole domain, then take its non-leaf nodes
    DeviceScratch<NodeT, DevVec> full(Nn + 2 * detail_balance21::kCompleteTreeMax<DIM>);
    const Long Nf = detail_balance21::completeSlice<DIM, DevVec>(pol, tree, mins, rank, np, end_target, full);
    DeviceScratch<Long, DevVec> ix(Nf);
    const Long k = thrust::copy_if(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nf), ix.begin(),
                                   detail_balance21::NonLeafPred<DIM>{thrust::raw_pointer_cast(full.data()), Nf, NodeT{}.Next()}) - ix.begin();
    S.resize(k);
    thrust::gather(pol, ix.begin(), ix.begin() + k, full.begin(), S.begin());
  }
  ClosureFrontier<DIM>(S, periodicity);
  #ifdef SCTL_HAVE_MPI
  if (np > 1) { // redistribute so rank r keeps [mins[r], mins[r+1]); device buffers go straight into MPI (CUDA-aware)
    sctl::ScratchBuf<Long> scnt(np), rcnt(np);
    Long Nrecv = 0;
    { // S is sorted, so each rank's block is contiguous: split at the mins
      DeviceScratch<NodeT, DevVec> mins_d(np);
      thrust::copy(mins.begin(), mins.end(), mins_d.begin());
      Nrecv = detail::splitCounts(scnt.begin(), rcnt.begin(), S, (Long)S.size(), mins_d, comm);
    }
    DevVec<NodeT>& recv = detail::PersistentBuffer<NodeT, DevVec, detail::Buf::ClosureRecv>();
    detail::exchangePooled(pol, S, (Long)S.size(), recv, Nrecv, scnt, rcnt, comm);
    local_sort(pol, recv, Nrecv);  // np sorted runs -> one sorted block
    S.swap(recv);
  }
  #endif
  S.resize(detail::local_unique(pol, S, (Long)S.size()));

  detail_balance21::leavesFromNonLeaf<DIM, DevVec>(pol, tree, thrust::raw_pointer_cast(S.data()), (Long)S.size(), mins[rank], end_target);
}

}  // namespace detail_balance21_gpu

// Ghost-layer placeholders (distributed), mirroring Tree::UpdateRefinement's halo scheme: send
// each owned node to every rank whose owned interval meets its coarse neighborhood (NbrList at
// depth d0-halo_size; the self entry keeps the boundary-ancestor chain ghosted). Received ghosts
// are spliced around the owned slice with complete-tree fill -> full-domain tree, coarse outside the halo.
namespace detail_addGhostNodes {
using detail::WalkMode;
using detail::local_sort;
using detail::lowerBound;

template <Integer DIM> struct GhostPair {
  Long p;
  Morton<DIM> m;
  SCTL_GPU_HD bool operator<(const GhostPair& o) const { return p < o.p || (!(o.p < p) && m < o.m); }
};

template <Integer DIM, WalkMode MODE, sctl::Periodicity PER> struct GhostSendFunctor {
  static constexpr Integer K = sctl::pow<DIM, Integer>(3);
  const Morton<DIM>* tree;
  const Morton<DIM>* A;  // partition boundaries: A[r] is rank r's first node, lex order
  Morton<DIM> lo, hi;    // this rank's owned interval [lo, hi)
  Long np, rank;
  Integer halo;
  const Long* offsets;      // WalkMode::Write only
  GhostPair<DIM>* out;      // WalkMode::Write only

  SCTL_GPU_HD Long operator()(Long i) const {
    const Morton<DIM>& X = tree[i];
    const Integer lvl = (Integer(X.depth) > halo ? Integer(X.depth) - halo : 0);
    { // Every neighbor strictly inside [lo, hi) belongs to this rank, so nothing is sent and the
      // 3^DIM list need not be built. Strict on the low side: a neighbor starting exactly at `lo`
      // would make the lower_bound below return `rank`, and `p0 = lb - 1` would reach rank-1.
      Morton<DIM> nb0, nb1;
      X.NbrRange(nb0, nb1, uint8_t(lvl), PER);
      if (lo < nb0 && !(hi < nb1)) return 0;
    }
    const auto nl = X.template NbrList<PER>(uint8_t(lvl));
    Long count = 0;
    GhostPair<DIM>* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[i];
    for (Integer k = 0; k < K; k++) {
      const Morton<DIM>& m = nl[k];
      if (m.depth == Morton<DIM>::INVALID_DEPTH) continue;
      Long p0 = lowerBound(A, Long(0), np, m.DFD()) - 1;
      if (p0 < 0) p0 = 0;
      const Long p1 = lowerBound(A, Long(0), np, m.Next());
      for (Long p = p0; p < p1; p++) {
        if (p == rank) continue;
        if constexpr (MODE == WalkMode::Write) w[count] = GhostPair<DIM>{p, X};
        ++count;
      }
    }
    return count;
  }
};


template <Integer DIM> struct GhostPairToMid {
  SCTL_GPU_HD Morton<DIM> operator()(const GhostPair<DIM>& gp) const { return gp.m; }
};

// Splice ghost placeholders into `tree`; outputs the [begin, end) index range of the owned nodes
// within the updated list. halo_size < 0 exchanges no neighbor nodes but still splices the coarse
// complete-tree fill, so the list is full-domain on every rank (as in Tree::UpdateRefinement).
template <Integer DIM, template <class...> class DevVec>
void addGhostNodes(DevVec<Morton<DIM>>& tree, const sctl::ScratchBuf<Morton<DIM>>& mins, const Comm& comm, Integer halo_size, sctl::Periodicity periodicity, Long& owned_begin, Long& owned_end,
                   DevVec<Morton<DIM>>* user_mid = nullptr, sctl::Vector<Long>* user_cnt = nullptr) {
  using NodeT = Morton<DIM>;
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long Nn = static_cast<Long>(tree.size());
  owned_begin = 0;
  owned_end = Nn;
  if (np == 1) return;

  // `mins` is the partition: mins[r] is rank r's first node, in code and depth alike, so the
  // boundaries need no gathering here.
  const NodeT lo = mins[rank], hi = (rank + 1 < np) ? mins[rank + 1] : NodeT{}.Next();
  const auto pol = detail::scratch_policy<DevVec, NodeT>();
  DeviceScratch<NodeT, DevVec> mins_d(np);
  thrust::copy(mins.begin(), mins.end(), mins_d.begin());
  // halo_size < 0 exchanges no neighbor nodes, so the scan is skipped and `pairs` stays empty. The
  // count and write passes are separate blocks because `pairs` is a pool slice: its size has to be
  // known at construction, and the scan is what produces it.
  const Long Nscan = (halo_size >= 0 ? Nn : 0);
  DeviceScratch<Long, DevVec> offsets(Nscan);
  Long npairs_tot = 0;
  if (Nscan) { // how many (dest rank, node) pairs each owned node produces
    DeviceScratch<Long, DevVec> counts(Nn);  // released before `pairs` is taken, so the pool stays LIFO
    detail::dispatchPeriodicity<DIM>(periodicity, [&pol, &tree, &mins_d, lo, hi, np, rank, halo_size, Nn, &counts](auto per_c) {
      const GhostSendFunctor<DIM, WalkMode::Count, decltype(per_c)::value> fc{thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(mins_d.data()), lo, hi, np, rank, halo_size, nullptr, nullptr};
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), counts.begin(), fc);
    });
    npairs_tot = detail::scanCounts(pol, counts, offsets, Nn);
  }
  Long npairs = 0;
  DeviceScratch<GhostPair<DIM>, DevVec> pairs(npairs_tot);
  if (Nscan) { // emit the pairs, sort by (dest rank, node), drop duplicates
    detail::dispatchPeriodicity<DIM>(periodicity, [&pol, &tree, &mins_d, lo, hi, np, rank, halo_size, Nn, &offsets, &pairs](auto per_c) {
      const GhostSendFunctor<DIM, WalkMode::Write, decltype(per_c)::value> fw{thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(mins_d.data()), lo, hi, np, rank, halo_size,
                                                                              thrust::raw_pointer_cast(offsets.data()), thrust::raw_pointer_cast(pairs.data())};
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), thrust::make_discard_iterator(), fw);
    });
    local_sort(pol, pairs, npairs_tot);
    npairs = detail::local_unique(pol, pairs, npairs_tot);
  }

  Long Nrecv = 0;
  sctl::ScratchBuf<Long> scnt(np), rcnt(np);
  DeviceScratch<NodeT, DevVec> send_mid(npairs);
  { // per-destination segments and counts; pairs are sorted by (dest rank, node)
    sctl::ScratchBuf<GhostPair<DIM>> keys(np);
    for (Long r = 0; r < np; r++) keys[r] = GhostPair<DIM>{r, NodeT{}};
    DeviceScratch<GhostPair<DIM>, DevVec> keys_d(np);
    thrust::copy(keys.begin(), keys.end(), keys_d.begin());
    Nrecv = detail::splitCounts(scnt.begin(), rcnt.begin(), pairs, npairs, keys_d, comm);
    thrust::transform(pol, pairs.begin(), pairs.begin() + npairs, send_mid.begin(), GhostPairToMid<DIM>{});
  }
  if (user_mid) {  // the halo send list: which of my nodes each rank wants as a ghost
    user_mid->resize(npairs);
    thrust::copy(pol, send_mid.begin(), send_mid.begin() + npairs, user_mid->begin());
  }
  if (user_cnt) {  // the caller's vector may be a view, so keep it when it already fits
    if (user_cnt->Dim() != np) user_cnt->ReInit(np);
    sctl::omp_par::copy(scnt.begin(), scnt.begin() + np, user_cnt->begin());
  }

  DeviceScratch<NodeT, DevVec> ghost(Nrecv);
  detail::alltoallv<DevVec>(pol, thrust::raw_pointer_cast(send_mid.data()), thrust::raw_pointer_cast(ghost.data()), scnt, rcnt, sizeof(NodeT), comm);
  // sorted: each source's segment is sorted and source owned-intervals are ordered
  const Long Nsplit = thrust::lower_bound(pol, ghost.begin(), ghost.end(), mins[rank]) - ghost.begin();

  const NodeT* gp = thrust::raw_pointer_cast(ghost.data());
  DeviceScratch<Long, DevVec> off_l(rank > 0 ? Nsplit + 1 : 0), off_r(rank + 1 < np ? Nrecv - Nsplit + 1 : 0);
  const Long L = (rank > 0) ? detail::anchorWalkCount<DIM, DevVec>(off_l, gp, Nsplit, NodeT{}, mins[rank]) : 0;
  const Long R = (rank + 1 < np) ? detail::anchorWalkCount<DIM, DevVec>(off_r, gp + Nsplit, Nrecv - Nsplit, mins[rank + 1], NodeT{}.Next()) : 0;
  DeviceScratch<NodeT, DevVec> left(L), right(R);
  if (L) detail::anchorWalkWrite<DIM, DevVec>(thrust::raw_pointer_cast(left.data()), thrust::raw_pointer_cast(off_l.data()), gp, Nsplit, NodeT{}, mins[rank]);
  if (R) detail::anchorWalkWrite<DIM, DevVec>(thrust::raw_pointer_cast(right.data()), thrust::raw_pointer_cast(off_r.data()), gp + Nsplit, Nrecv - Nsplit, mins[rank + 1], NodeT{}.Next());

  // Swapped rather than assigned, so `tree` and this retained buffer trade storage each build
  // instead of one being freed and the other allocated.
  DevVec<NodeT>& merged = detail::PersistentBuffer<NodeT, DevVec, detail::Buf::GhostMerge>();
  merged.resize(L + Nn + R);
  thrust::copy(pol, left.begin(), left.end(), merged.begin());
  thrust::copy(pol, tree.begin(), tree.end(), merged.begin() + L);
  thrust::copy(pol, right.begin(), right.end(), merged.begin() + L + Nn);
  tree.swap(merged);
  owned_begin = L;
  owned_end = L + Nn;
}

}  // namespace detail_addGhostNodes

// Optional per-node flags, mirroring sctl::Tree::GetNodeAttr.
namespace detail_nodeAttr {

// Leaf: no child of this node follows it (sctl's test). Ghost: outside the owned index range.
template <Integer DIM, class AttrT> struct NodeAttrFunctor {
  const Morton<DIM>* tree;
  Long n, owned_begin, owned_end;
  SCTL_GPU_HD AttrT operator()(Long i) const {
    AttrT a{};
    a.Leaf = !(i + 1 < n && tree[i].isAncestor(tree[i + 1]));
    a.Ghost = (i < owned_begin || i >= owned_end);
    return a;
  }
};

}  // namespace detail_nodeAttr

// Optional per-node connectivity, mirroring sctl::Tree::GetNodeLists but structure-of-arrays.
namespace detail_nodeLists {

// Connectivity in four passes, written straight into the caller's arrays. Parent and child are
// kept in their own compact arrays -- 8 and 8*2^DIM bytes per node -- because the neighbor walk reads
// them repeatedly and the working set is what decides whether they stay cached.

// Pass 1: parent index, by one exact binary search per node.
template <Integer DIM> struct ParentPassFunctor {
  const Morton<DIM>* tree;
  Long n;
  Long* par;
  SCTL_GPU_HD Long find(const Morton<DIM>& key) const {
    const Long lo = detail::lowerBound(tree, Long(0), n, key);
    return (lo < n && !(key < tree[lo])) ? lo : -1;
  }
  SCTL_GPU_HD void operator()(Long i) const {
    const Integer d = tree[i].Depth();
    par[i] = d ? find(tree[i].Ancestor((uint8_t)(d - 1))) : -1;
  }
};

// Pass 2: each node writes itself into its parent's child slot. (parent, p2n) is unique: no atomics.
template <Integer DIM> struct ChildPassFunctor {
  static constexpr Integer MAX_CHILD = 1 << DIM;
  const Morton<DIM>* tree;
  const Long* par;
  Long* ch;
  SCTL_GPU_HD void operator()(Long i) const {
    const Long p = par[i];
    if (p >= 0) ch[p * MAX_CHILD + tree[i].Path2Node()] = i;
  }
};

// Pass 3: neighbors by descent from the root. The device runs it for every node; the host runs it
// for the root only and propagates level by level below. The one place periodicity is consulted.
template <Integer DIM, sctl::Periodicity PER> struct NbrDescentFunctor {
  static constexpr Integer MAX_CHILD = 1 << DIM;
  static constexpr Integer MAX_NBRS = sctl::pow<DIM, Integer>(3);
  const Morton<DIM>* tree;
  const Long* ch;
  Long* nbr;               // MAX_NBRS per node
  SCTL_GPU_HD void operator()(Long i) const {
    const Morton<DIM> X = tree[i];
    const Integer d = X.Depth();
    const auto nl = X.template NbrList<PER>((uint8_t)d);
    Long* const out = nbr + i * MAX_NBRS;
    for (Integer k = 0; k < MAX_NBRS; k++) {
      const Morton<DIM>& m = nl[k];
      if (m.depth == Morton<DIM>::INVALID_DEPTH) {
        out[k] = -1;
        continue;
      }
      Long cur = 0;  // the root is at index 0
      for (Integer l = 1; l <= d && cur >= 0; l++) cur = ch[cur * MAX_CHILD + m.Ancestor((uint8_t)l).Path2Node()];
      out[k] = cur;
    }
  }
};

// Pass 4, one depth at a time: a node's k-th neighbor is a child of its parent's k'-th neighbor,
// where k' and the child slot follow from the node's position bits within its parent plus the
// offset digits of k. Two array reads per entry, instead of a root-to-depth descent of dependent
// loads. Periodicity needs no dispatch here: a wrapped neighbor is reached through the parent's
// wrapped row, and an out-of-domain one inherits the -1 its parent's row already carries. Reads
// touch only depth-1 rows, writes only depth rows, so running over the whole array per level does
// not alias.
template <Integer DIM> struct NbrPropagateFunctor {
  static constexpr Integer MAX_CHILD = 1 << DIM;
  static constexpr Integer MAX_NBRS = sctl::pow<DIM, Integer>(3);
  const Morton<DIM>* tree;
  const Long* par;
  const Long* ch;
  Long* nbr;
  Integer depth;
  SCTL_GPU_HD void operator()(Long i) const {
    if ((Integer)tree[i].Depth() != depth) return;
    const Integer b = (Integer)tree[i].Path2Node();
    const Long* pr = nbr + par[i] * MAX_NBRS;
    Long* out = nbr + i * MAX_NBRS;
    for (Integer k = 0; k < MAX_NBRS; k++) {
      Integer kk = k, kp = 0, cj = 0, p3 = 1;
      for (Integer c = 0; c < DIM; c++) {
        const Integer t = ((b >> c) & 1) + kk % 3 - 1;  // this coord's move within the parent: -1..2
        kk /= 3;
        kp += ((t + 2) / 2) * p3;  // which of the parent's neighbors the target is a child of
        cj |= (t & 1) << c;        // and which child slot -- the AND is the periodic wrap
        p3 *= 3;
      }
      const Long pn = pr[kp];
      out[k] = (pn >= 0) ? ch[pn * MAX_CHILD + cj] : Long(-1);
    }
  }
};

}  // namespace detail_nodeLists

// Distributed build: device sample sort (radix -> exact-rank splitters -> Alltoallv -> re-sort),
// then a two-sided M-code halo and allgathered boundary anchors (mins). M is clamped to the
// smallest per-rank count. Concatenated over ranks, the output matches the single-rank build.

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::buildTreeDist(DevVec<Morton<DIM>>& tree, const DevVec<Real>& coord, Long M, const Comm& comm, bool balance21, sctl::Periodicity periodicity, Integer halo_size, Long* owned_range, Morton<DIM>* partition, DevVec<NodeAttr>* node_attr, NodeLists<DevVec>* node_lists, DevVec<Morton<DIM>>* user_mid, sctl::Vector<Long>* user_cnt) {

  using MortonT = MortonCode<DIM>;
  const auto pol = detail::scratch_policy<DevVec, MortonT>();
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long Nglob = [&coord, &comm]() {
    sctl::StaticArray<Long,2> N{(Long)coord.size()/DIM, 0};
    comm.Allreduce<sctl::CommOp::SUM>(N+0, N+1, 1);
    return N[1];
  }();
  sctl::ScratchBuf<Morton<DIM>> mins(np);  // the partition: each rank's first node
  const auto fillOutputs = [&tree, &mins, &pol, periodicity, owned_range, partition, node_attr, node_lists](Long owned_begin, Long owned_end) {
    if (owned_range) {
      owned_range[0] = owned_begin;
      owned_range[1] = owned_end;
    }
    if (partition) std::copy(mins.begin(), mins.end(), partition);
    if (node_attr) {
      const Long Nt = (Long)tree.size();
      node_attr->resize(Nt);
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nt),
                        node_attr->begin(),
                        detail_nodeAttr::NodeAttrFunctor<DIM, NodeAttr>{thrust::raw_pointer_cast(tree.data()), Nt, owned_begin, owned_end});
    }
    if (node_lists) {
      constexpr Integer MAX_CHILD = 1 << DIM, MAX_NBRS = sctl::pow<DIM, Integer>(3);
      const Long Nt = (Long)tree.size();
      const Morton<DIM>* const tp = thrust::raw_pointer_cast(tree.data());
      node_lists->parent.resize(Nt);
      node_lists->child.resize(Nt * MAX_CHILD);
      node_lists->nbr.resize(Nt * MAX_NBRS);
      Long* const pp = thrust::raw_pointer_cast(node_lists->parent.data());
      Long* const cp = thrust::raw_pointer_cast(node_lists->child.data());
      Long* const nbp = thrust::raw_pointer_cast(node_lists->nbr.data());
      thrust::fill(pol, node_lists->child.begin(), node_lists->child.end(), Long(-1));
      thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::ParentPassFunctor<DIM>{tp, Nt, pp});
      thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::ChildPassFunctor<DIM>{tp, pp, cp});
      if constexpr (detail::is_device_vector_v<DevVec<Long>>) {
        // One kernel of independent root-to-depth descents: the device hides their latency with
        // parallelism and the shallow levels stay in cache, where level-ordered launches idle
        // between levels.
        detail::dispatchPeriodicity<DIM>(periodicity, [&pol, Nt, tp, cp, nbp](auto per_c) {
          thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::NbrDescentFunctor<DIM, decltype(per_c)::value>{tp, cp, nbp});
        });
      } else {
        // The host is the opposite: the descents are dependent random loads, so seed the root's row
        // and propagate level by level instead.
        if (Nt) detail::dispatchPeriodicity<DIM>(periodicity, [&pol, tp, cp, nbp](auto per_c) {  // the root is always index 0
          thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Long(1), detail_nodeLists::NbrDescentFunctor<DIM, decltype(per_c)::value>{tp, cp, nbp});
        });
        for (Integer d = 1; d <= Morton<DIM>::MAX_DEPTH; d++)
          thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::NbrPropagateFunctor<DIM>{tp, pp, cp, nbp, d});
      }
    }
  };
  if (Nglob <= M) {  // all particles fit one leaf: root-only tree, held by rank 0
    tree.resize(rank == 0 ? 1 : 0);
    if (rank == 0) tree[0] = Morton<DIM>{};
    for (Long r = 0; r < np; r++) mins[r] = (r == 0 ? Morton<DIM>{} : Morton<DIM>{}.Next());
    if (user_mid) user_mid->resize(0);
    if (user_cnt) {
      if (user_cnt->Dim() != np) user_cnt->ReInit(np);
      user_cnt->SetZero();
    }
    fillOutputs(0, (Long)tree.size());
    return;
  }

  // Double-buffered: `pt_mid` is replaced three times below (sort, repartition, halo). Swapping with
  // a second retained buffer recycles the storage instead of freeing it and taking a fresh block.
  DevVec<MortonT>& pt_mid = detail::PersistentBuffer<MortonT, DevVec, detail::Buf::PtMid>();
  DevVec<MortonT>& alt = detail::PersistentBuffer<MortonT, DevVec, detail::Buf::PtAlt>();
  pt_mid.resize((Long)coord.size()/DIM);
  { // Encode coords -> Morton, then local sort (device radix / host omp_par).
    const Long Nloc = (Long)pt_mid.size();
    if constexpr (detail::is_device_vector_v<DevVec<Real>>) {
      detail::MakeMortonFunctor<Real, DIM> enc{thrust::raw_pointer_cast(coord.data())};
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nloc), pt_mid.begin(), enc);
    } else {
      const Real* cp = thrust::raw_pointer_cast(coord.data());
      MortonT* mp = thrust::raw_pointer_cast(pt_mid.data());
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < Nloc; ++i) mp[i] = MortonT(cp + i * DIM);
    }
    detail::local_sort(pol, pt_mid, Nloc);
  }

  #ifdef SCTL_HAVE_MPI
  if (np > 1) { // distributed sort
    sctl::ScratchBuf<MortonT> spl_h(np - 1);
    detail_determineSplitters::determineSplitters(spl_h, pt_mid, comm);
    DeviceScratch<MortonT, DevVec> spl_d(np - 1);
    thrust::copy(spl_h.begin(), spl_h.end(), spl_d.begin());

    sctl::ScratchBuf<Long> scnt(np), rcnt(np);
    const Long Nrecv = detail::splitCounts(scnt.begin(), rcnt.begin(), pt_mid, (Long)pt_mid.size(), spl_d, comm);

    detail::exchangePooled(pol, pt_mid, (Long)pt_mid.size(), alt, Nrecv, scnt, rcnt, comm);
    detail::local_sort(pol, alt, Nrecv);  // np sorted segments -> one sorted block
    pt_mid.swap(alt);
  }

  if (np > 1) { // M <- global_min(pt_mid.size(), M); repartition if necessary
    Long Nloc = (Long)pt_mid.size(), Nloc_min = 0;
    comm.Allreduce<sctl::CommOp::MIN>(sctl::Ptr2ConstItr<Long>(&Nloc, 1), sctl::Ptr2Itr<Long>(&Nloc_min, 1), 1);
    if (Nloc_min < M) {  // repartition to an even split; received segments concatenate in global-index order, so pt_mid stays sorted
      detail::partitionN(pol, pt_mid, Nloc, (rank + 1) * Nglob / np - rank * Nglob / np, comm, alt);
      Nloc_min = Nglob / np;  // smallest chunk of the even split
    }

    M = std::min<Long>(M, Nloc_min);
    if (M < 1) MPI_Abort(comm.GetMPI_Comm(), 1);
  }

  if (np > 1) { // halo: pt_mid <-- [M from left | pt_mid | M from right] (empty halo on domain-edge ranks)
    const Long recv0 = (rank > 0 ? M : 0);
    const Long recv1 = (rank < np - 1 ? M : 0);
    alt.resize(recv0 + pt_mid.size() + recv1);
    thrust::copy(pol, pt_mid.begin(), pt_mid.end(), alt.begin() + recv0);

    const int left  = (rank > 0      ? int(rank - 1) : MPI_PROC_NULL);
    const int right = (rank + 1 < np ? int(rank + 1) : MPI_PROC_NULL);
    MortonT* b = thrust::raw_pointer_cast(alt.data());
    const int mb = int(M * (Long)sizeof(MortonT));
    MPI_Sendrecv(b + recv0,                         mb, MPI_BYTE, left,  27, b + recv0 + pt_mid.size(), mb, MPI_BYTE, right, 27, comm.GetMPI_Comm(), MPI_STATUS_IGNORE);
    MPI_Sendrecv(b + recv0 + pt_mid.size() - recv1, mb, MPI_BYTE, right, 28, b,                         mb, MPI_BYTE, left,  28, comm.GetMPI_Comm(), MPI_STATUS_IGNORE);
    pt_mid.swap(alt);
  }
  #endif  // SCTL_HAVE_MPI

  { // build mins
    Morton<DIM> A{};
    if (rank > 0) {
      const MortonT ka = MortonT(pt_mid[0]);
      const MortonT kb = MortonT(pt_mid[M]);
      uint8_t d = ka.CommonAncestor(kb).depth;
      if (d < MAX_DEPTH) ++d;
      A = kb.Ancestor(d);
    }
    comm.Allgather(sctl::Ptr2ConstItr<Morton<DIM>>(&A, 1), 1, mins.begin(), 1);
  }

  { // build linear tree from pt_mid
    const Morton<DIM> end_bnd = (rank + 1 < np) ? mins[rank + 1] : Morton<DIM>{}.Next();
    const Long idx0 = thrust::lower_bound(pol, pt_mid.begin(), std::min(pt_mid.begin()+2*M, pt_mid.end()), mins[rank].mid) - pt_mid.begin();
    const Long idx1 = thrust::lower_bound(pol, std::max(pt_mid.begin(), pt_mid.end()-2*M), pt_mid.end(), end_bnd.mid) - pt_mid.begin();

    if constexpr (detail::is_device_vector_v<DevVec<Real>>) {
      // anchor build wins for small slices, chunked for large; the choice depends only on local size.
      constexpr Long kChunkedThreshold = 128 * 1024;
      if ((idx1 - idx0) * M < kChunkedThreshold) detail_build::buildTreeGpu<DIM>(tree, pt_mid, M, idx1 - idx0, idx0, mins[rank], end_bnd);
      else detail_build::buildTreeGpuChunked<DIM>(tree, pt_mid, M, idx1 - idx0, idx0, mins[rank], end_bnd);
    } else {
      detail_build::buildTreeCpuChunked<DIM>(tree, pt_mid, M, idx1 - idx0, idx0, mins[rank], end_bnd);
    }
  }

  if (balance21) {  // each backend's faster closure, see the two namespaces
    if constexpr (detail::is_device_vector_v<DevVec<Morton<DIM>>>) detail_balance21_gpu::balanceTreeDist<DIM>(tree, mins, comm, periodicity);
    else detail_balance21_host::balanceTreeDist<DIM>(tree, mins, comm, periodicity);
  }

  Long owned_begin = 0, owned_end = Long(tree.size());
  detail_addGhostNodes::addGhostNodes<DIM>(tree, mins, comm, halo_size, periodicity, owned_begin, owned_end, user_mid, user_cnt);

  fillOutputs(owned_begin, owned_end);
}

// Stateful interface: the tree, the partition and any named per-node data live in the object, so a
// rebuild can carry the data across. `buildTreeDist` above does the building; everything here is
// bookkeeping around it, mirroring sctl::Tree.

template <class Real, Integer DIM, template <class...> class DevVec>
GPUTree<Real, DIM, DevVec>::GPUTree(const Comm& comm) : comm_(comm) {
  // The coarsest uniform tree with at least one leaf per rank, as sctl::Tree builds it: one point
  // per cell of an n0^DIM grid, dealt out evenly, refined to one point per leaf. Every rank then
  // owns a non-empty range before any particle is added -- where a root-only tree would leave all
  // but the first owning nothing, so the first AddParticles would land the whole global particle
  // set on rank 0. An unrefined tree is still a valid one, so data can be attached immediately.
  const Long np = comm_.Size(), rank = comm_.Rank();
  Long n0 = 1;
  while (sctl::pow<DIM, Long>(n0) < np) n0++;
  const Long N = sctl::pow<DIM, Long>(n0), beg = N * rank / np, end = N * (rank + 1) / np;
  sctl::ScratchBuf<Real> h((end - beg) * DIM);
  for (Long i = beg; i < end; i++) {
    Long idx = i;
    for (Integer k = 0; k < DIM; k++) {
      h[(i - beg) * DIM + k] = (Real)(idx % n0) / (Real)n0;
      idx /= n0;
    }
  }
  DevVec<Real> coord(h.Dim());
  thrust::copy(h.begin(), h.end(), coord.begin());
  UpdateRefinement(coord);
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::UpdateRefinement(const DevVec<Real>& coord, Long M, bool balance21, sctl::Periodicity periodicity, Integer halo_size) {
  const Long np = comm_.Size(), rank = comm_.Rank();
  const auto pol = detail::scratch_policy<DevVec, Morton<DIM>>();

  // This rank's owned nodes before the rebuild, kept on the device for the search below; at
  // np > 1 they round-trip through the host once, for PartitionS.
  DevVec<Morton<DIM>>& old_mid = detail::PersistentBuffer<Morton<DIM>, DevVec, detail::Buf::OldMid>();
  const auto base_moved = [this](const std::string& name) { return !data_moved_by_derived_.count(name); };
  Long nbase = 0;  // data sets moved here
  for (const auto& kv : node_data_) nbase += base_moved(kv.first);
  const bool remap = nbase && mins_.Dim();
  old_mid.resize(remap ? owned_end_ - owned_begin_ : 0);
  if (remap) thrust::copy(pol, node_mid_.begin() + owned_begin_, node_mid_.begin() + owned_end_, old_mid.begin());
  const Long old_begin = owned_begin_, old_end = owned_end_;

  { // rebuild
    mins_.ReInit(np);
    Long owned[2] = {0, 0};
    buildTreeDist(node_mid_, coord, M, comm_, balance21, periodicity, halo_size, owned,
                  mins_.begin(), &node_attr_, &node_lists_, &user_mid_, &user_cnt_);
    owned_begin_ = owned[0];
    owned_end_ = owned[1];
    host_mid_stale_ = true;
  }
  if (!nbase) return;

  sctl::ScratchBuf<Long> range((Long)node_mid_.size() + 1);
  Long No = 0;
  { // move the old owned nodes to whoever owns their range now, then find the old nodes each new node absorbs
    if (np > 1) {  // PartitionS is a host operation, so the nodes round-trip for it alone
      sctl::Vector<Morton<DIM>> h((Long)old_mid.size());
      detail::deviceToHost(old_mid.data(), h.Dim(), h.begin());
      comm_.PartitionS(h, mins_[rank]);
      old_mid.resize(h.Dim());
      thrust::copy(h.begin(), h.end(), old_mid.begin());
    }
    No = (Long)old_mid.size();
    { // per new node, [range[i], range[i+1]) into old_mid: searched where the nodes live, so only range crosses the bus
      const Long Nn = (Long)node_mid_.size();
      range[Nn] = No;
      if constexpr (detail::is_device_vector_v<DevVec<Morton<DIM>>>) {
        DeviceScratch<Long, DevVec> r(Nn);
        thrust::lower_bound(pol, old_mid.begin(), old_mid.begin() + No, node_mid_.begin(), node_mid_.begin() + Nn, r.begin());
        detail::deviceToHost(r.data(), Nn, range.begin());
      } else {  // thrust's host backend is serial unless built for OMP, so parallelize it here
        const Morton<DIM>* o = thrust::raw_pointer_cast(old_mid.data());
        const Morton<DIM>* n = thrust::raw_pointer_cast(node_mid_.data());
        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Nn; i++) range[i] = std::lower_bound(o, o + No, n[i]) - o;
      }
    }
  }

  for (auto& kv : node_data_) {
    const std::string& name = kv.first;
    if (!base_moved(name)) continue;
    DevVec<char>& data = kv.second;
    sctl::Vector<Long>& cnt = node_cnt_[name];

    sctl::ScratchBuf<Long> dsp(cnt.Dim() + 1);
    const Long dof = detail::globalDof((Long)data.size(), detail::scanv(dsp.begin(), cnt.begin(), cnt.Dim()), comm_);
    const Long data_begin = dsp[old_begin], data_count = dsp[old_end] - dsp[old_begin];
    Long Ndata = 0;  // items after the aggregation
    { // counts follow the nodes, then aggregate onto the new nodes
      sctl::Vector<Long> cnt_tmp(old_end - old_begin, cnt.begin() + old_begin);  // a copy: PartitionN moves it
      if (np > 1) comm_.PartitionN(cnt_tmp, No);
      cnt.ReInit((Long)node_mid_.size());
      #pragma omp parallel for schedule(static) reduction(+ : Ndata)
      for (Long i = 0; i < (Long)node_mid_.size(); i++) {
        Long sum = 0;
        for (Long j = range[i]; j < range[i + 1]; j++) sum += cnt_tmp[j];
        cnt[i] = sum;
        Ndata += sum;
      }
    }
    { // the payload follows the same movement, on the device
      DevVec<char>& own = detail::PersistentBuffer<char, DevVec, detail::Buf::MigData>();
      own.resize(data_count * dof);
      thrust::copy(pol, data.begin() + data_begin * dof, data.begin() + (data_begin + data_count) * dof, own.begin());
      detail::partitionN(pol, own, data_count * dof, Ndata * dof, comm_, detail::PersistentBuffer<char, DevVec, detail::Buf::DataRecv>());
      data.swap(own);
    }
  }
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::AddData(const std::string& name, const DevVec<ValueType>& data, const sctl::Vector<Long>& cnt) {
  const Long nitem = sctl::omp_par::reduce(cnt.begin(), cnt.Dim());
  const Long dof = detail::globalDof((Long)data.size(), nitem, comm_);
  addData_(name, nitem * dof * (Long)sizeof(ValueType), cnt);
  DevVec<char>& dst = NodeData_(name);
  using It = detail::ScratchIterator<char, DevVec>;
  const It src(const_cast<char*>((const char*)thrust::raw_pointer_cast(data.data())));
  thrust::copy(src, src + (Long)dst.size(), dst.begin());
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::AddData(const std::string& name, Long dof, const sctl::Vector<Long>& cnt) {
  addData_(name, sctl::omp_par::reduce(cnt.begin(), cnt.Dim()) * dof * (Long)sizeof(ValueType), cnt);
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::addData_(const std::string& name, Long bytes, const sctl::Vector<Long>& cnt) {
  SCTL_ASSERT_MSG(node_data_.find(name) == node_data_.end(), "GPUTree::AddData: name already present.");
  SCTL_ASSERT(cnt.Dim() == (Long)node_mid_.size());
  node_data_[name].resize(bytes);
  node_cnt_[name] = cnt;
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::GetData(View<ValueType>& data, sctl::Vector<Long>& cnt, const std::string& name) {
  dataView(data, cnt, name);
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::GetData(View<const ValueType>& data, sctl::Vector<Long>& cnt, const std::string& name) const {
  dataView(data, cnt, name);
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class VT>
void GPUTree<Real, DIM, DevVec>::dataView(View<VT>& data, sctl::Vector<Long>& cnt, const std::string& name) const {
  const auto d = node_data_.find(name);
  const auto c = node_cnt_.find(name);
  SCTL_ASSERT_MSG(d != node_data_.end() && c != node_cnt_.end(), "GPUTree::GetData: unknown name.");
  SCTL_ASSERT(d->second.size() % sizeof(VT) == 0);
  data = View<VT>{(VT*)thrust::raw_pointer_cast(d->second.data()), (Long)d->second.size() / (Long)sizeof(VT)};
  cnt.ReInit(c->second.Dim(), (sctl::Iterator<Long>)c->second.begin(), false);
}


namespace detail_bcast {

/** Index of `m` in the sorted `nmid[0, Nn)`, or -1. */
template <Integer DIM> Long findNode(sctl::ConstIterator<Morton<DIM>> nmid, Long Nn, const Morton<DIM>& m) {
  const Long k = std::lower_bound(nmid, nmid + Nn, m) - nmid;
  return (k < Nn && nmid[k] == m) ? k : -1;
}

/** One whole block per call: `dst[dstoff[i]*w + j] = src[srcoff[i]*w + j]` for `j < len[i]*w`. */
template <class T> struct BlockCopyFunctor {
  const T* src;
  T* dst;
  const Long* srcoff;
  const Long* dstoff;
  const Long* len;
  Long w;
  SCTL_GPU_HD void operator()(Long i) const {
    const T* const s = src + srcoff[i] * w;
    T* const d = dst + dstoff[i] * w;
    for (Long j = 0; j < len[i] * w; j++) d[j] = s[j];
  }
};

/**
 * Copy `nb` variable-length blocks: `dst[dstoff[i]*w + j] = src[srcoff[i]*w + j]` for `j < len[i]*w`.
 * A scatter rather than a gather, so destination elements no block covers are left alone.
 *
 * One thread per block, and only the blocks' bounds ever reach the device. Naming each element's
 * source and destination outright would instead cost an eight-byte pair for every element moved,
 * which is many times the traffic of the copy it is arranging. Every caller passes node-granular
 * blocks, so `nb` runs with the node count and the blocks are individually small.
 */
template <template <class...> class DevVec, class Policy, class T>
void blockCopy(const Policy& pol, T* dst, const T* src,
               sctl::ConstIterator<Long> srcoff, sctl::ConstIterator<Long> dstoff, sctl::ConstIterator<Long> len, Long nb, Long w) {
  if (nb <= 0) return;
  if constexpr (!detail::is_device_vector_v<DevVec<char>>) {
    #pragma omp parallel for schedule(static) if (nb > 256)
    for (Long i = 0; i < nb; i++) std::memcpy(dst + dstoff[i] * w, src + srcoff[i] * w, len[i] * w * sizeof(T));
    return;
  }
  DeviceScratch<Long, DevVec> so(nb), doff(nb), l(nb);
  thrust::copy(srcoff, srcoff + nb, so.begin());
  thrust::copy(dstoff, dstoff + nb, doff.begin());
  thrust::copy(len, len + nb, l.begin());
  const Long* const so_p = thrust::raw_pointer_cast(so.data());
  const Long* const doff_p = thrust::raw_pointer_cast(doff.data());
  const Long* const l_p = thrust::raw_pointer_cast(l.data());
  // A byte at a time would leave most of the memory path idle, so regroup the copy into words
  // where the widths and both bases allow it. Callers pass byte buffers, hence `sizeof(T) == 1`.
  using Word = std::uint64_t;
  if constexpr (sizeof(T) == 1) {
    const bool aligned = !(w % (Long)sizeof(Word)) && !((std::uintptr_t)dst % sizeof(Word)) && !((std::uintptr_t)src % sizeof(Word));
    if (aligned) {
      thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), nb, BlockCopyFunctor<Word>{
          (const Word*)src, (Word*)dst, so_p, doff_p, l_p, w / (Long)sizeof(Word)});
      return;
    }
  }
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), nb, BlockCopyFunctor<T>{src, dst, so_p, doff_p, l_p, w});
}

/**
 * Send rank p the nodes `smid[sndsp[p], sndsp[p+1])` and, for each of them that this rank holds
 * (found in `nmid`), its `cnt` items of `w` bytes out of `data`; receive the same from every rank.
 * `consume` then sees the Nr received nodes with their item counts, item offsets and packed bytes.
 */
template <Integer DIM, template <class...> class DevVec, class Policy, class Consume>
void exchangeBlocks(const Policy& pol, const Comm& comm, sctl::ConstIterator<Morton<DIM>> smid, Long Ns, sctl::ConstIterator<Long> sncnt,
                    sctl::ConstIterator<Morton<DIM>> nmid, Long Nn, const sctl::Vector<Long>& cnt, sctl::ConstIterator<Long> dsp,
                    const DevVec<char>& data, Long w, Consume&& consume) {
  const Long np = comm.Size();
  sctl::ScratchBuf<Long> sndsp(np + 1), rncnt(np), rndsp(np + 1);
  detail::scanv(sndsp.begin(), sncnt, np);
  comm.Alltoall(sncnt, 1, rncnt.begin(), 1);
  const Long Nr = detail::scanv(rndsp.begin(), rncnt.begin(), np);
  sctl::ScratchBuf<Morton<DIM>> rmid(Nr);
  comm.Alltoallv(smid, sncnt, sndsp.begin(), rmid.begin(), rncnt.begin(), rndsp.begin());

  sctl::ScratchBuf<Long> sdcnt(Ns), rdcnt(Nr), soff(Ns);
  #pragma omp parallel for schedule(static) if (Ns > 256)
  for (Long i = 0; i < Ns; i++) {  // my nodes among the requests, and how many items each carries
    const Long k = findNode<DIM>(nmid, Nn, smid[i]);
    sdcnt[i] = (k >= 0 ? cnt[k] : 0);
    soff[i] = (k >= 0 ? dsp[k] : 0);
  }
  comm.Alltoallv(sdcnt.begin(), sncnt, sndsp.begin(), rdcnt.begin(), rncnt.begin(), rndsp.begin());

  sctl::ScratchBuf<Long> sddsp(Ns + 1), rddsp(Nr + 1);
  const Long Nsend = detail::scanv(sddsp.begin(), sdcnt.begin(), Ns);
  const Long Nrecv = detail::scanv(rddsp.begin(), rdcnt.begin(), Nr);
  DeviceScratch<char, DevVec> sbuf(Nsend * w), rbuf(Nrecv * w);
  blockCopy<DevVec>(pol, thrust::raw_pointer_cast(sbuf.data()), thrust::raw_pointer_cast(data.data()), soff.begin(), sddsp.begin(), sdcnt.begin(), Ns, w);
  sctl::ScratchBuf<Long> sbc(np), rbc(np);  // bytes per rank: the item offsets differenced at the rank boundaries
  for (Long p = 0; p < np; p++) {
    sbc[p] = (sddsp[sndsp[p + 1]] - sddsp[sndsp[p]]) * w;
    rbc[p] = (rddsp[rndsp[p + 1]] - rddsp[rndsp[p]]) * w;
  }
  detail::alltoallv<DevVec>(pol, thrust::raw_pointer_cast(sbuf.data()), thrust::raw_pointer_cast(rbuf.data()), sbc, rbc, Long(1), comm);
  consume(Nr, (sctl::ConstIterator<Morton<DIM>>)rmid.begin(), (sctl::ConstIterator<Long>)rdcnt.begin(), (sctl::ConstIterator<Long>)rddsp.begin(), (const char*)thrust::raw_pointer_cast(rbuf.data()));
}

}  // namespace detail_bcast

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::Broadcast(const std::string& name) {
  const Long np = comm_.Size();
  if (np == 1) return;
#ifdef SCTL_HAVE_MPI
  const detail::StageTimer<DevVec> prof{comm_};
  prof.tic("GPUTree::Broadcast", 6);
  const auto pol = detail::scratch_policy<DevVec, char>();
  DevVec<char>& data = NodeData_(name);
  sctl::Vector<Long>& cnt = NodeCnt_(name);
  const Long Nn = (Long)node_mid_.size();
  SCTL_ASSERT(cnt.Dim() == Nn);

  const sctl::ConstIterator<Morton<DIM>> nmid = hostNodeMID();
  sctl::ScratchBuf<Long> dsp(Nn + 1);
  const Long nitem = detail::scanv(dsp.begin(), cnt.begin(), Nn);
  const Long w = detail::globalDof((Long)data.size(), nitem, comm_);  // bytes per item; Broadcast only copies

  const Long Ns = (Long)user_mid_.size();  // the halo send list: which of my nodes each rank wants
  const Long ob = owned_begin_, oe = owned_end_;
  detail_bcast::exchangeBlocks<DIM, DevVec>(pol, comm_, hostUserMID(), Ns, user_cnt_.begin(), nmid, Nn, cnt, dsp.begin(), data, w,
      [&pol, &data, &cnt, nmid, &dsp, Nn, w, ob, oe](Long Nr, sctl::ConstIterator<Morton<DIM>> rmid, sctl::ConstIterator<Long> rdcnt, sctl::ConstIterator<Long> rddsp, const char* rbuf) {
        sctl::ScratchBuf<Long> ridx(Nr);
        #pragma omp parallel for schedule(static) if (Nr > 256)
        for (Long i = 0; i < Nr; i++) {
          ridx[i] = detail_bcast::findNode<DIM>(nmid, Nn, rmid[i]);
          SCTL_ASSERT(ridx[i] >= 0);
        }
        // where each arriving block lands, given the offsets the nodes will have
        const auto packRecv = [Nr, rdcnt, rddsp, &ridx](sctl::ConstIterator<Long> off, sctl::Iterator<Long> a, sctl::Iterator<Long> b, sctl::Iterator<Long> l) {
          Long m = 0;
          for (Long i = 0; i < Nr; i++) {
            if (!rdcnt[i]) continue;
            a[m] = rddsp[i];
            b[m] = off[ridx[i]];
            l[m] = rdcnt[i];
            m++;
          }
          return m;
        };
        { // Nothing moves when every ghost slot gets back the count it already had. Deciding that
          // costs a pass over what arrived and over the ghosts, never over the whole node list.
          Long changed = 0, arriving = 0, occupied = 0;
          #pragma omp parallel for schedule(static) reduction(+ : changed, arriving) if (Nr > 256)
          for (Long i = 0; i < Nr; i++) {
            changed += (rdcnt[i] != cnt[ridx[i]]);
            arriving += (rdcnt[i] != 0);
          }
          #pragma omp parallel for schedule(static) reduction(+ : occupied) if (ob > 256)
          for (Long i = 0; i < ob; i++) occupied += (cnt[i] != 0);  // the ghosts below my range
          #pragma omp parallel for schedule(static) reduction(+ : occupied) if (Nn - oe > 256)
          for (Long i = oe; i < Nn; i++) occupied += (cnt[i] != 0);  // and above it
          if (!changed && arriving == occupied) {
            sctl::ScratchBuf<Long> a(Nr), b(Nr), l(Nr);
            const Long m = packRecv(dsp.begin(), a.begin(), b.begin(), l.begin());
            detail_bcast::blockCopy<DevVec>(pol, (char*)thrust::raw_pointer_cast(data.data()), rbuf, a.begin(), b.begin(), l.begin(), m, w);
            return;
          }
        }
        // rebuild the array: only my own nodes keep what they held, every ghost comes from its owner
        sctl::Vector<Long> cnt_new(Nn);
        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Nn; i++) cnt_new[i] = (ob <= i && i < oe ? cnt[i] : 0);
        for (Long i = 0; i < Nr; i++) cnt_new[ridx[i]] = rdcnt[i];  // a ghost has one owner, so no two agree
        sctl::ScratchBuf<Long> dsp_new(Nn + 1);
        const Long nnew = detail::scanv(dsp_new.begin(), cnt_new.begin(), Nn);
        DevVec<char>& out = detail::PersistentBuffer<char, DevVec, detail::Buf::BcastOut>();
        out.resize(nnew * w);
        // my own blocks keep their contents, at their new offsets
        detail_bcast::blockCopy<DevVec>(pol, thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(data.data()), dsp.begin() + ob, dsp_new.begin() + ob, cnt.begin() + ob, oe - ob, w);
        { // every received block refills its node's slot, whatever that slot held before
          sctl::ScratchBuf<Long> a(Nr), b(Nr), l(Nr);
          const Long m = packRecv(dsp_new.begin(), a.begin(), b.begin(), l.begin());
          detail_bcast::blockCopy<DevVec>(pol, thrust::raw_pointer_cast(out.data()), rbuf, a.begin(), b.begin(), l.begin(), m, w);
        }
        data.swap(out);
        cnt.Swap(cnt_new);
      });
  prof.toc();
#endif
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::ReduceBroadcast(const std::string& name) {
  const Long np = comm_.Size(), rank = comm_.Rank();
  if (np == 1) return;
#ifdef SCTL_HAVE_MPI
  const detail::StageTimer<DevVec> prof{comm_};
  prof.tic("GPUTree::ReduceBroadcast", 6);
  const auto pol = detail::scratch_policy<DevVec, ValueType>();
  DevVec<char>& data = NodeData_(name);
  sctl::Vector<Long>& cnt = NodeCnt_(name);
  const Long Nn = (Long)node_mid_.size();
  const sctl::ConstIterator<Morton<DIM>> nmid = hostNodeMID();
  sctl::ScratchBuf<Long> dsp(Nn + 1);
  const Long nitem = detail::scanv(dsp.begin(), cnt.begin(), Nn);
  const Long dof = detail::globalDof((Long)data.size() / (Long)sizeof(ValueType), nitem, comm_);

  { // the ancestors of my first node are shared with earlier ranks; send them my partial values
    const Long Ns = mins_[rank].Depth();
    sctl::ScratchBuf<Morton<DIM>> smid(Ns);
    for (Long d = 0; d < Ns; d++) smid[d] = mins_[rank].Ancestor(d);
    sctl::ScratchBuf<Long> sncnt(np);
    for (Long p = 0; p < np; p++) {
      const Long a = std::lower_bound(smid.begin(), smid.begin() + Ns, mins_[p]) - smid.begin();
      const Long b = std::lower_bound(smid.begin(), smid.begin() + Ns, (p + 1 == np ? Morton<DIM>().Next() : mins_[p + 1])) - smid.begin();
      sncnt[p] = b - a;
    }
    detail_bcast::exchangeBlocks<DIM, DevVec>(pol, comm_, smid.begin(), Ns, sncnt.begin(), nmid, Nn, cnt, dsp.begin(), data, dof * (Long)sizeof(ValueType),
        [&pol, &data, &cnt, nmid, &dsp, Nn, dof](Long Nr, sctl::ConstIterator<Morton<DIM>> rmid, sctl::ConstIterator<Long> rdcnt, sctl::ConstIterator<Long> rddsp, const char* rbuf) {
          // add each received block into the node it belongs to
          ValueType* const d = (ValueType*)thrust::raw_pointer_cast(data.data());
          const ValueType* const r = (const ValueType*)rbuf;
          for (Long i = 0; i < Nr; i++) {
            if (!rdcnt[i]) continue;
            const Long idx = detail_bcast::findNode<DIM>(nmid, Nn, rmid[i]);
            if (idx < 0 || cnt[idx] != rdcnt[i]) continue;
            using It = detail::ScratchIterator<ValueType, DevVec>;
            thrust::transform(pol, It(d + dsp[idx] * dof), It(d + (dsp[idx] + cnt[idx]) * dof),
                              It(const_cast<ValueType*>(r + rddsp[i] * dof)), It(d + dsp[idx] * dof), thrust::plus<ValueType>());
          }
        });
  }
  Broadcast<ValueType>(name);
  prof.toc();
#endif
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::fillHostMID() const {
  if (!host_mid_stale_) return;
  if (node_mid_host_.Dim() != (Long)node_mid_.size()) node_mid_host_.ReInit((Long)node_mid_.size());
  detail::deviceToHost(node_mid_.data(), node_mid_host_.Dim(), node_mid_host_.begin());
  if (user_mid_host_.Dim() != (Long)user_mid_.size()) user_mid_host_.ReInit((Long)user_mid_.size());
  detail::deviceToHost(user_mid_.data(), user_mid_host_.Dim(), user_mid_host_.begin());
  host_mid_stale_ = false;
}

template <class Real, Integer DIM, template <class...> class DevVec>
sctl::ConstIterator<Morton<DIM>> GPUTree<Real, DIM, DevVec>::hostNodeMID() const {
  if constexpr (detail::is_device_vector_v<DevVec<char>>) {
    fillHostMID();
    return node_mid_host_.begin();
  } else {
    return sctl::Ptr2ConstItr<Morton<DIM>>(thrust::raw_pointer_cast(node_mid_.data()), (Long)node_mid_.size());
  }
}

template <class Real, Integer DIM, template <class...> class DevVec>
sctl::ConstIterator<Morton<DIM>> GPUTree<Real, DIM, DevVec>::hostUserMID() const {
  if constexpr (detail::is_device_vector_v<DevVec<char>>) {
    fillHostMID();
    return user_mid_host_.begin();
  } else {
    return sctl::Ptr2ConstItr<Morton<DIM>>(thrust::raw_pointer_cast(user_mid_.data()), (Long)user_mid_.size());
  }
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::WriteTreeVTK(std::string fname, bool show_ghost) const {
  using VTKReal = typename sctl::VTUData::VTKReal;
  sctl::VTUData vtu_data;
  if (DIM <= 3) {  // one cell per leaf, its 2^DIM corners as points
    static constexpr Integer Ncorner = (1u << DIM);
    const Long Nn = (Long)node_mid_.size();
    const sctl::ConstIterator<Morton<DIM>> mid = hostNodeMID();
    sctl::ScratchBuf<NodeAttr> attr(Nn);
    detail::deviceToHost(node_attr_.data(), Nn, attr.begin());

    sctl::Vector<VTKReal>& coord = vtu_data.coord;
    sctl::Vector<int32_t>& connect = vtu_data.connect;
    sctl::Vector<int32_t>& offset = vtu_data.offset;
    sctl::Vector<uint8_t>& types = vtu_data.types;

    sctl::StaticArray<VTKReal, DIM> c;
    Long point_cnt = coord.Dim() / 3;
    Long connect_cnt = connect.Dim();
    for (Long nid = 0; nid < Nn; nid++) {
      if (!show_ghost && attr[nid].Ghost) continue;
      if (!attr[nid].Leaf) continue;

      mid[nid].Coord((sctl::Iterator<VTKReal>)c);
      const VTKReal s = sctl::pow<VTKReal>(0.5, mid[nid].Depth());
      for (Integer j = 0; j < Ncorner; j++) {
        for (Integer i = 0; i < DIM; i++) coord.PushBack(c[i] + ((j & (1u << i)) ? 1 : 0) * s);
        for (Integer i = DIM; i < 3; i++) coord.PushBack(0);
        connect.PushBack(point_cnt);
        connect_cnt++;
        point_cnt++;
      }
      offset.PushBack(connect_cnt);
      if (DIM == 2) types.PushBack(8);
      else if (DIM == 3) types.PushBack(11);
      else types.PushBack(4);
    }
  }
  vtu_data.WriteVTK(fname, comm_);
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::DeleteData(const std::string& name) {
  SCTL_ASSERT_MSG(node_data_.find(name) != node_data_.end(), "GPUTree::DeleteData: unknown name.");
  node_data_.erase(name);
  node_cnt_.erase(name);
  data_moved_by_derived_.erase(name);
}

// Particle bookkeeping. The particles' codes, their order and the moves between caller and tree
// order live in a `SortScatter` per group (sort-scatter.hpp); the tree only maps them to nodes.
namespace detail_ptTree {

/**
 * A node's code, for comparing nodes against particles. Sound because a particle sits at MAX_DEPTH:
 * if its code differs from the node's the codes decide, and if it matches, the particle is the
 * deeper of the two and sorts after -- which is exactly where comparing codes alone places it.
 */
template <Integer DIM> struct NodeToCodeFunctor {
  SCTL_GPU_HD MortonCode<DIM> operator()(const Morton<DIM>& m) const { return m.mid; }
};

/** The partition as codes: `SortScatter`'s splitters. */
template <Integer DIM> void partitionCodes(sctl::Vector<MortonCode<DIM>>& codes, const sctl::Vector<Morton<DIM>>& mins) {
  if (codes.Dim() != mins.Dim()) codes.ReInit(mins.Dim());
  for (Long r = 0; r < mins.Dim(); r++) codes[r] = mins[r].mid;
}

}  // namespace detail_ptTree

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
PtTree<Real, DIM, DevVec, BaseTree>::PtTree(const Comm& comm) : BaseTree(comm) {
  detail_ptTree::partitionCodes<DIM>(partition_codes_, this->GetPartitionMID());
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::nodeCounts(const std::string& name, sctl::Vector<Long>& cnt) const {
  const auto pol = detail::scratch_policy<DevVec, MortonCode<DIM>>();
  const auto& node_mid = this->GetNodeMID();
  const auto& pm = groups_.find(name)->second.SortedKeys();
  const Long Nn = (Long)node_mid.size(), Npt = (Long)pm.size();
  // Difference on the device, so only the finished counts cross the bus.
  DeviceScratch<Long, DevVec> pos(Nn + 1);
  const auto nc0 = thrust::make_transform_iterator(node_mid.begin(), detail_ptTree::NodeToCodeFunctor<DIM>{});
  thrust::lower_bound(pol, pm.begin(), pm.begin() + Npt, nc0, nc0 + Nn, pos.begin());
  thrust::fill(pol, pos.begin() + Nn, pos.begin() + Nn + 1, Npt);
  DeviceScratch<Long, DevVec> d(Nn);
  thrust::transform(pol, pos.begin() + 1, pos.begin() + Nn + 1, pos.begin(), d.begin(), thrust::minus<Long>());
  if (cnt.Dim() != Nn) cnt.ReInit(Nn);
  detail::deviceToHost(d.data(), Nn, cnt.begin());
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::AddParticles(const std::string& name, const DevVec<Real>& coord) {
  SCTL_ASSERT_MSG(groups_.find(name) == groups_.end(), "PtTree::AddParticles: name already present.");
  const detail::StageTimer<DevVec> prof{this->GetComm()};
  prof.tic("PtTree::AddParticles", 6);
  const auto pol = detail::scratch_policy<DevVec, MortonCode<DIM>>();
  const Long Nloc = (Long)coord.size() / DIM;
  SCTL_ASSERT((Long)coord.size() == Nloc * DIM);

  // The key is the bare MortonCode, not the Morton: every particle sits at MAX_DEPTH, so the depth
  // would only double the key and the exchange volume (NodeToCodeFunctor: why codes alone order them).
  DevVec<MortonCode<DIM>> key(Nloc);
  thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nloc), key.begin(),
                    detail::MakeMortonFunctor<Real, DIM>{thrust::raw_pointer_cast(coord.data())});
  groups_.try_emplace(name, this->GetComm()).first->second.Init(std::move(key), partition_codes_);
  AddParticleData(name, name, coord);
  prof.toc();
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::AddParticleData(const std::string& data_name, const std::string& particle_name, const DevVec<Real>& data) {
  const auto it = groups_.find(particle_name);
  SCTL_ASSERT_MSG(it != groups_.end(), "PtTree::AddParticleData: unknown particle group.");
  const Long dof = detail::globalDof((Long)data.size(), it->second.LocalCount(), this->GetComm());
  AddParticleData(data_name, particle_name, dof);
  // the forward scatter reads the caller's array and writes the stored buffer, so neither end is copied
  it->second.ScatterForward((const Real*)thrust::raw_pointer_cast(data.data()), (Real*)thrust::raw_pointer_cast(this->NodeData_(data_name).data()), dof);
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::AddParticleData(const std::string& data_name, const std::string& particle_name, Long dof) {
  SCTL_ASSERT_MSG(groups_.find(particle_name) != groups_.end(), "PtTree::AddParticleData: unknown particle group.");
  SCTL_ASSERT_MSG(data_pt_name_.find(data_name) == data_pt_name_.end(), "PtTree::AddParticleData: data name already present.");
  sctl::Vector<Long> cnt;
  nodeCounts(particle_name, cnt);
  this->template AddData<Real>(data_name, dof, cnt);
  this->data_moved_by_derived_.insert(data_name);
  data_pt_name_[data_name] = particle_name;
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::GetParticleData(DevVec<Real>& data, const std::string& data_name) const {
  const auto it = data_pt_name_.find(data_name);
  SCTL_ASSERT_MSG(it != data_pt_name_.end(), "PtTree::GetParticleData: unknown data name.");
  const auto& g = groups_.find(it->second)->second;

  // the reverse scatter reads the stored buffer and writes the output, so the payload is touched once
  const DevVec<char>& raw = this->NodeData_(data_name);
  const Long dof = detail::globalDof((Long)raw.size() / (Long)sizeof(Real), g.SortedCount(), this->GetComm());
  data.resize(g.LocalCount() * dof);
  g.ScatterReverse((const Real*)thrust::raw_pointer_cast(raw.data()), (Real*)thrust::raw_pointer_cast(data.data()), dof);
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::DeleteParticleData(const std::string& data_name) {
  const auto it = data_pt_name_.find(data_name);
  SCTL_ASSERT_MSG(it != data_pt_name_.end(), "PtTree::DeleteParticleData: unknown data name.");
  const std::string particle_name = it->second;
  if (data_name == particle_name) {  // deleting the group takes every data set on it
    std::vector<std::string> lst;
    for (const auto& kv : data_pt_name_) if (kv.second == particle_name && kv.first != particle_name) lst.push_back(kv.first);
    for (const auto& x : lst) DeleteParticleData(x);
    groups_.erase(particle_name);
  }
  this->DeleteData(data_name);
  data_pt_name_.erase(data_name);
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::UpdateRefinement(const DevVec<Real>& coord, Long M, bool balance21, sctl::Periodicity periodicity, Integer halo_size) {
  const Comm& comm = this->GetComm();
  BaseTree::UpdateRefinement(coord, M, balance21, periodicity, halo_size);
  detail_ptTree::partitionCodes<DIM>(partition_codes_, this->GetPartitionMID());

  for (auto& kv : groups_) {  // payloads follow their keys' re-cut; per-node counts come from the particles
    const std::string& group = kv.first;
    kv.second.Repartition(partition_codes_);
    sctl::Vector<Long> cnt_new;
    nodeCounts(group, cnt_new);

    std::vector<std::string> names;
    for (const auto& p : data_pt_name_) if (p.second == group) names.push_back(p.first);
    for (const auto& name : names) {
      DevVec<char>& raw = this->NodeData_(name);
      sctl::Vector<Long>& cnt = this->NodeCnt_(name);
      const Long w = detail::globalDof((Long)raw.size(), sctl::omp_par::reduce(cnt.begin(), cnt.Dim()), comm);  // bytes per item
      kv.second.RepartitionData(raw, w);
      cnt = cnt_new;
    }
  }
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::WriteParticleVTK(std::string fname, std::string data_name, bool show_ghost) const {
  using VTKReal = typename sctl::VTUData::VTKReal;
  const auto it = data_pt_name_.find(data_name);
  SCTL_ASSERT_MSG(it != data_pt_name_.end(), "PtTree::WriteParticleVTK: unknown data name.");
  const std::string& particle_name = it->second;

  sctl::Vector<Long> pt_cnt, val_cnt;
  DataView<const Real, DevVec> pt_d, val_d;
  this->GetData(pt_d, pt_cnt, particle_name);
  this->GetData(val_d, val_cnt, data_name);
  sctl::ScratchBuf<Real> pt(pt_d.size()), val(val_d.size());
  detail::deviceToHost(pt_d.begin(), pt_d.size(), pt.begin());
  detail::deviceToHost(val_d.begin(), val_d.size(), val.begin());

  const Long Nn = (Long)this->GetNodeMID().size();
  sctl::ScratchBuf<typename BaseTree::NodeAttr> attr(Nn);
  detail::deviceToHost(this->GetNodeAttr().data(), Nn, attr.begin());
  const Long npt = sctl::omp_par::reduce(pt_cnt.begin(), pt_cnt.Dim());
  const Long vdof = (npt ? val.Dim() / npt : 0);

  sctl::VTUData vtu_data;
  Long pt_idx = 0;
  for (Long i = 0; i < Nn; i++) {
    const bool skip = (!show_ghost && attr[i].Ghost);
    for (Long j = 0; j < pt_cnt[i]; j++, pt_idx++) {
      if (skip) continue;
      for (Integer k = 0; k < DIM; k++) vtu_data.coord.PushBack((VTKReal)pt[pt_idx * DIM + k]);
      for (Integer k = DIM; k < 3; k++) vtu_data.coord.PushBack(0);
      for (Long k = 0; k < vdof; k++) vtu_data.value.PushBack((VTKReal)val[pt_idx * vdof + k]);
      vtu_data.connect.PushBack(vtu_data.offset.Dim());
      vtu_data.offset.PushBack(vtu_data.connect.Dim());
      vtu_data.types.PushBack(1);  // VTK_VERTEX
    }
  }
  vtu_data.WriteVTK(fname, this->GetComm());
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::test() {
  const Comm comm = Comm::World();
  const Long N = 100000;  // particles on this rank

  std::mt19937_64 rng(comm.Rank());
  std::uniform_real_distribution<Real> U(0, 1);
  std::vector<Real> x(N * DIM), f(N);
  for (auto& v : x) v = U(rng);
  for (Long i = 0; i < N; i++) f[i] = x[i * DIM];  // any per-particle value
  const DevVec<Real> xd(x.begin(), x.end()), fd(f.begin(), f.end());

  PtTree tree(comm);
  tree.UpdateRefinement(xd, 64, true);  // leaves hold at most 64 particles; 2:1 balanced
  tree.AddParticles("pt", xd);          // sorted into node order and distributed by the partition
  tree.AddParticleData("f", "pt", fd);  // follows the particles

  { // work on the stored data in place: the view aliases the tree's storage, in node order
    // cnt[i]: particles in node i
    DataView<Real, DevVec> v;
    sctl::Vector<Long> cnt;
    tree.GetData(v, cnt, "f");
    thrust::transform(v.begin(), v.end(), v.begin(), 2.0 * thrust::placeholders::_1);
  }

  DevVec<Real> out;
  tree.GetParticleData(out, "f");  // back in the caller's order: out[i] == 2 * f[i]
  std::vector<Real> h(out.size());
  thrust::copy(out.begin(), out.end(), h.begin());
  Long bad = 0;
  for (Long i = 0; i < N; i++) bad += (h[i] != 2 * f[i]);

  Long b, e;
  tree.GetOwnedRange(b, e);
  printf("rank %d: %ld particles, %ld nodes of which %ld owned, %ld values wrong\n",
         (int)comm.Rank(), (long)N, (long)tree.GetNodeMID().size(), (long)(e - b), (long)bad);
}

}  // namespace gpu_tree

#include "sctl/experimental/sort-scatter.txx"  // uses the detail helpers above

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
