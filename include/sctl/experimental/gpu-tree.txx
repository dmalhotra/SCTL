// Template implementation of GPUTree from gpu-tree.hpp + its internal detail_* helper namespaces:
// detail (shared helpers), detail_build, detail_determineSplitters, detail_balance21 (predicates
// shared by both balance schemes), detail_balance21_gpu (default) / detail_balance21_host
// (-DGT_BALANCE_HOST=1), detail_addGhostNodes.

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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <numeric>

#include "sctl/experimental/gpu-tree.hpp"
#include "sctl/experimental/device_scratch.hpp"
#include "sctl/comm.hpp"
#include "sctl/ompUtils.txx"
#include "sctl/tree.hpp"   // sctl::Tree::Balance21 (hybrid host balance)
#include "sctl/vtudata.hpp"  // WriteTreeVTK
#include "sctl/vtudata.txx"
#include "sctl/tree.txx"
#include "sctl/scratch_pool.hpp"
#include "sctl/scratch_pool.txx"

namespace gpu_tree {

namespace detail {

// Execution policy for thrust calls on backend memory, with temporaries drawn from the scratch
// pool (thrust/cub otherwise cudaMalloc's them per call, which costs more than the work).
template <template <class...> class DeviceVector, class T> auto scratch_policy() {
#if defined(__CUDACC__) || defined(__HIPCC__)
  if constexpr (is_device_vector_v<DeviceVector<T>>) {
    static DeviceScratchAllocator<DeviceVector> alloc;
    return thrust::cuda::par(alloc);
  } else
#endif
  {
    return thrust::host;
  }
}

// Which retained buffer a `PersistentBuffer` call means; no two uses may share a tag.
enum class Buf { PtMid, PtAlt, Closure, Frontier, ClosureRecv, GhostMerge, DataRecv,
                 PtSend, PtRecv, PtSortK, MigData, OldMid };

// Functor (not lambda) so nvcc captures it across thrust kernel boundaries.
template <class Real, Integer DIM> struct MakeMortonFunctor {
  const Real* coord_ptr;
  SCTL_GPU_HD MortonCode<DIM> operator()(Long i) const {
    return MortonCode<DIM>(coord_ptr + i * DIM);
  }
};

/** True for `sctl::MortonCode<DIM>`, whose ordering is its integer key's -- see `GetIntKey`. */
template <class T> struct is_morton_code : std::false_type {};
template <Integer D> struct is_morton_code<sctl::MortonCode<D>> : std::true_type {};

/** True when a sort of `T` can go through `GetIntKey` and reach a radix sort. Written as a
 *  specialization, not a `&&`: the second condition names a member only `MortonCode` has, and a
 *  variable template would instantiate it for every key type. */
template <class T, bool = is_morton_code<T>::value> struct radix_via_int_key : std::false_type {};
template <class T> struct radix_via_int_key<T, true> : std::integral_constant<bool, T::IntKeyIsExact> {};

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

// Sort v[0,n): radix on device, omp_par on host (thrust's host backend is serial). merge_sort
// stops scaling past ~16 threads (bandwidth-bound), sample_sort doesn't, so pick by thread count.
template <class Vec> void local_sort(Vec& v, Long n) {
  using T = typename vec_family<Vec>::elem;
  auto* p = thrust::raw_pointer_cast(v.data());
  if constexpr (is_device_vector_v<Vec>) {
    thrust::sort(v.begin(), v.begin() + n);
  } else if constexpr (radix_via_int_key<T>::value) {
    sctl::omp_par::radix_sort(p, n, [](const T& x) { return x.GetIntKey(); });
  } else {
    if (SCTL_GET_MAX_THREADS() <= 16) sctl::omp_par::merge_sort(p, p + n);
    else sctl::omp_par::sample_sort(p, p + n);
  }
}

// Same, with a caller-supplied execution policy (pooled temporaries) on the device path.
template <class Policy, class Vec> void local_sort(const Policy& pol, Vec& v, Long n) {
  using T = typename vec_family<Vec>::elem;
  // A Morton code is a struct, so thrust picks a comparison sort for it; sorting its integer key
  // reaches cub's radix sort instead and gives the identical order.
  if constexpr (is_device_vector_v<Vec> && radix_via_int_key<T>::value) {
    DeviceScratch<std::uint64_t, vec_family<Vec>::template to> k(n);
    thrust::transform(pol, v.begin(), v.begin() + n, k.begin(), ToIntKeyFunctor<T>{});
    thrust::sort(pol, k.begin(), k.begin() + n);
    thrust::transform(pol, k.begin(), k.begin() + n, v.begin(), FromIntKeyFunctor<T>{});
  } else if constexpr (is_device_vector_v<Vec>) {
    thrust::sort(pol, v.begin(), v.begin() + n);
  } else {
    local_sort(v, n);
  }
}

// local_sort carrying a payload (the pre-sort index). Host path sorts packed pairs: thrust's
// host backend is serial and omp_par has no by-key sort.
template <class Vec, class IVec> void local_sort_by_key(Vec& keys, IVec& vals, Long n);

// Same, with a caller-supplied execution policy (pooled temporaries) on the device path.
template <class Policy, class Vec, class IVec> void local_sort_by_key(const Policy& pol, Vec& keys, IVec& vals, Long n) {
  using T = typename vec_family<Vec>::elem;
  if constexpr (is_device_vector_v<Vec> && radix_via_int_key<T>::value) {
    DeviceScratch<std::uint64_t, vec_family<Vec>::template to> k(n);
    thrust::transform(pol, keys.begin(), keys.begin() + n, k.begin(), ToIntKeyFunctor<T>{});
    thrust::sort_by_key(pol, k.begin(), k.begin() + n, vals.begin());
    thrust::transform(pol, k.begin(), k.begin() + n, keys.begin(), FromIntKeyFunctor<T>{});
  } else if constexpr (is_device_vector_v<Vec>) {
    thrust::sort_by_key(pol, keys.begin(), keys.begin() + n, vals.begin());
  } else {
    local_sort_by_key(keys, vals, n);
  }
}

template <class Vec, class IVec> void local_sort_by_key(Vec& keys, IVec& vals, Long n) {
  if constexpr (is_device_vector_v<Vec>) {
    thrust::sort_by_key(keys.begin(), keys.begin() + n, vals.begin());
  } else {
    using KeyT = typename vec_family<Vec>::elem;
    using ValT = typename vec_family<IVec>::elem;
    struct Pair {
      KeyT key;
      ValT val;
      bool operator<(const Pair& o) const { return key < o.key; }
    };
    KeyT* kp = thrust::raw_pointer_cast(keys.data());
    ValT* vp = thrust::raw_pointer_cast(vals.data());
    sctl::ScratchBuf<Pair> pairs(n);
    const auto pp = pairs.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n; i++) { pp[i].key = kp[i]; pp[i].val = vp[i]; }
    if constexpr (radix_via_int_key<KeyT>::value) {
      sctl::omp_par::radix_sort(&pp[0], n, [](const Pair& x) { return x.key.GetIntKey(); });
    } else if (SCTL_GET_MAX_THREADS() <= 16) {
      sctl::omp_par::merge_sort(pairs.begin(), pairs.end());
    } else {
      sctl::omp_par::sample_sort(pairs.begin(), pairs.end());
    }
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n; i++) { kp[i] = pp[i].key; vp[i] = pp[i].val; }
  }
}

// Route a sorted array to its owners. Per-rank element counts from splitting `in[0,n)` at the
// device-resident `keys` (np keys, or np-1 splitters with the first block starting at 0), and the
// counts coming back; returns the number of elements to be received.
template <class T, template <class...> class DeviceVector, class Vec>
Long splitCounts(sctl::ScratchBuf<Long>& scnt, sctl::ScratchBuf<Long>& rcnt, const Vec& in, Long n,
                 const DeviceScratch<T, DeviceVector>& keys, const Comm& comm) {
  const Long np = comm.Size(), nkeys = keys.Dim();
  SCTL_ASSERT(nkeys == np || nkeys == np - 1);
  { sctl::ScratchBuf<Long> pos(np + 1);
    DeviceScratch<Long, DeviceVector> pos_d(nkeys);
    thrust::lower_bound(scratch_policy<DeviceVector, T>(), in.begin(), in.begin() + n, keys.begin(), keys.end(), pos_d.begin());
    thrust::copy(pos_d.begin(), pos_d.end(), pos.begin() + (np - nkeys));
    if (nkeys < np) pos[0] = 0;  // splitters: this rank keeps everything below the first one
    pos[np] = n;
    for (Long r = 0; r < np; r++) scnt[r] = pos[r + 1] - pos[r];
  }
  comm.Alltoall(scnt.begin(), 1, rcnt.begin(), 1);
  Long nrecv = 0;
  for (Long r = 0; r < np; r++) nrecv += rcnt[r];
  return nrecv;
}

// Alltoallv of esz-sized elements given per-rank element counts (displacements are their scans).
// Buffers may be device pointers -- they go straight to MPI, which is CUDA-aware here.
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
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < m; i++) p[i] = tmp[i];
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

inline void alltoallv(const void* sbuf, void* rbuf, const sctl::ScratchBuf<Long>& scnt,
                      const sctl::ScratchBuf<Long>& rcnt, Long esz, const Comm& comm) {
#ifdef SCTL_HAVE_MPI
  const Long np = comm.Size(), rank = comm.Rank();
  static const Long IMAX = 2147483647;
  sctl::ScratchBuf<Long> sd(np + 1), rd(np + 1);   // byte offsets, in Long
  sd[0] = 0; rd[0] = 0;
  for (Long r = 0; r < np; r++) { sd[r + 1] = sd[r] + scnt[r] * esz; rd[r + 1] = rd[r] + rcnt[r] * esz; }

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
      sc[r] = (int)(scnt[r] * esz / G); rc[r] = (int)(rcnt[r] * esz / G);
      sdi[r] = (int)(sd[r] / G);        rdi[r] = (int)(rd[r] / G);
    }
    MPI_Datatype dt;
    MPI_Type_contiguous((int)G, MPI_BYTE, &dt);
    MPI_Type_commit(&dt);
    MPI_Alltoallv(sbuf, &sc[0], &sdi[0], dt, rbuf, &rc[0], &rdi[0], dt, comm.GetMPI_Comm());
    MPI_Type_free(&dt);
    return;
  }

  // Beyond that, exchange pairwise on a rotating schedule, one send and one receive outstanding.
  // Posting every peer at once -- which this did first -- collapses at scale: at 16 ranks, 30
  // unscheduled 1 GB transfers per rank timed out after 40 minutes where the collective needs
  // seconds. Pairwise is the shape MPI's own large-message alltoallv uses. Offsets stay `Long` and
  // become pointer arithmetic, so only each message's count must fit an `int`.
  const char* sp = (const char*)sbuf;
  char* rp = (char*)rbuf;
  const Long CHUNK = Long(1) << 30;
  const auto nchunk = [CHUNK](Long n) { return (n + CHUNK - 1) / CHUNK; };
  for (Long k = 0; k < np; k++) {
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


// Exchange through pooled scratch. A buffer taken and released each build pays, on every call, for
// the peer mapping the transport rebuilds, the allocation, and the device-wide drain a release
// forces: 43.7 ms vs 5.8 ms for an 800 MB/rank exchange on 4 GPUs. The two copies cost far less.
template <class T, template <class...> class DeviceVector, class Policy>
void exchangePooled(const Policy& pol, const DeviceVector<T>& src, Long nsrc, DeviceVector<T>& dst, Long ndst,
                    const sctl::ScratchBuf<Long>& scnt, const sctl::ScratchBuf<Long>& rcnt, const Comm& comm) {
  DeviceScratch<T, DeviceVector> xs(nsrc), xr(ndst);
  thrust::copy(pol, src.begin(), src.begin() + nsrc, xs.begin());
  alltoallv(thrust::raw_pointer_cast(xs.data()), thrust::raw_pointer_cast(xr.data()), scnt, rcnt, (Long)sizeof(T), comm);
  dst.resize(ndst);
  thrust::copy(pol, xr.begin(), xr.end(), dst.begin());
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
  const Long* offsets;  // Write only
  Morton<DIM>* out;     // Write only

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

// The anchor walk in two passes, so a caller that knows (or can bound) the output size can write
// straight into its own buffer -- e.g. pooled scratch -- instead of having the walk allocate one.

// Exclusive scan of counts[0,n) into `offsets`, returning the total the scan already summed --
// reading back the last offset and count avoids a second pass over counts just to total them.
template <class Policy, template <class...> class DeviceVector>
Long scanCounts(const Policy& pol, const DeviceScratch<Long, DeviceVector>& counts, DeviceScratch<Long, DeviceVector>& offsets, Long n) {
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
  if (periodicity == PER) { f(std::integral_constant<sctl::Periodicity, PER>{}); return; }
  if constexpr (MASK + 1 < (1 << DIM)) dispatchPeriodicity<DIM, F, sctl::PeriodicityT(MASK + 1)>(periodicity, f);
}

// Count pass: fills `offsets` (exclusive scan of the per-pair node counts) and returns the total.
template <Integer DIM, template <class...> class DeviceVector>
Long anchorWalkCount(DeviceScratch<Long, DeviceVector>& offsets, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const Long n_pairs = n + 1;
  const auto pol = scratch_policy<DeviceVector, Morton<DIM>>();
  DeviceScratch<Long, DeviceVector> counts(n_pairs);
  const AnchorWalkFunctor<DIM, WalkMode::Count> fc{anchors_ptr, n, start_node, end_target, nullptr, nullptr};
  if constexpr (is_device_vector_v<DeviceVector<Morton<DIM>>>) {
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
template <Integer DIM, template <class...> class DeviceVector>
void anchorWalkWrite(Morton<DIM>* out, const Long* offsets, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const Long n_pairs = n + 1;
  const auto pol = scratch_policy<DeviceVector, Morton<DIM>>();
  const AnchorWalkFunctor<DIM, WalkMode::Write> fw{anchors_ptr, n, start_node, end_target, offsets, out};
  if constexpr (is_device_vector_v<DeviceVector<Morton<DIM>>>) {
    thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), n_pairs, fw);
  } else {
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < n_pairs; i++) fw(i);
  }
}

// The complete preorder tree over [start_node, end_target) with `anchors` as its forced leaves.
template <Integer DIM, template <class...> class DeviceVector>
void treeFromAnchors(DeviceVector<Morton<DIM>>& tree, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  DeviceScratch<Long, DeviceVector> offsets(n + 1);
  const Long total = anchorWalkCount<DIM, DeviceVector>(offsets, anchors_ptr, n, start_node, end_target);
  tree.resize(total);
  anchorWalkWrite<DIM, DeviceVector>(thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(offsets.data()), anchors_ptr, n, start_node, end_target);
}

}  // namespace detail

// Tree linearization from sorted Morton codes: single-rank build + the distributed walk stage.
namespace detail_build {
using detail::AnchorWalkFunctor;
using detail::WalkMode;
using detail::lowerBound;

// A slice of at most M points spanning the whole domain is one leaf: the root. Returns false if the
// caller still has a tree to build.
template <Integer DIM, template <class...> class DeviceVector>
bool rootOnlyTree(DeviceVector<Morton<DIM>>& tree, Long N, Long M, const Morton<DIM>& start_bnd, const Morton<DIM>& end_bnd) {
  if (!(N <= M && start_bnd == Morton<DIM>{} && end_bnd == Morton<DIM>{}.Next())) return false;
  tree.resize(1);
  tree[0] = Morton<DIM>{};
  return true;
}

// Split-leaf of pair (pt[i], pt[i+M]): child of their common ancestor holding pt[i+M]. The M-gap
// forces the split: a depth-d box holding both endpoints has M+1 > M particles, so it refines.
template <class Real, Integer DIM> struct SplitLeafFunctor {
  const MortonCode<DIM>* pt;
  Long M;
  SCTL_GPU_HD Morton<DIM> operator()(Long i) const {
    uint8_t d = pt[i].CommonAncestor(pt[i + M]).depth;
    if (d < MAX_DEPTH) ++d;
    return pt[i + M].Ancestor(d);
  }
};

// GPU build of the slice [start_bnd, end_bnd) from sorted codes: (1) SplitLeafFunctor over pairs
// (pt[i],pt[i+M]) -> anchors, deduped by unique_copy and clipped to the slice (fused via
// transform_iterator, no leaves in global memory); (2) linearize the gaps between consecutive
// anchors (count + exclusive_scan + write) via AnchorWalkFunctor.
template <class Real, Integer DIM, template <class...>
class DeviceVector> void buildTreeGpu(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M, Long N_owned = -1, Long base = 0, Morton<DIM> start_bnd = Morton<DIM>{}, Morton<DIM> end_bnd = Morton<DIM>{}.Next()) {
  using NodeMIDT = Morton<DIM>;

  const Long N = (N_owned < 0 ? static_cast<Long>(pt_mid.size()) : N_owned);
  if (rootOnlyTree<DIM, DeviceVector>(tree, N, M, start_bnd, end_bnd)) return;

  // Phase 1: anchors from the pairs within pt_mid[base, base+N), then clip to [start_bnd, end_bnd).
  // The boundary leaves need no points from outside: start_bnd is itself the anchor of the pair
  // straddling the lower boundary, and the walk stops at end_bnd.
  const Long N_pairs = std::max<Long>(N - M, 0);
  const auto pol = detail::scratch_policy<DeviceVector, NodeMIDT>();
  DeviceScratch<NodeMIDT, DeviceVector> anchors(N_pairs);
  SplitLeafFunctor<Real, DIM> f{thrust::raw_pointer_cast(pt_mid.data()) + base, M};
  auto in = thrust::make_transform_iterator(thrust::counting_iterator<Long>(0), f);
  auto uniq_end = anchors.begin() + detail::local_unique_copy(pol, in, N_pairs, anchors);
  auto a_begin = thrust::lower_bound(pol, anchors.begin(), uniq_end, start_bnd);
  auto a_end   = thrust::lower_bound(pol, a_begin,         uniq_end, end_bnd);
  const NodeMIDT* anchors_ptr = thrust::raw_pointer_cast(anchors.data()) + (a_begin - anchors.begin());
  const Long n_anchors = a_end - a_begin;

  // Phase 2: linearize over the n_anchors+1 gaps (+1 = trailing gap to end_bnd).
  detail::treeFromAnchors<DIM, DeviceVector>(tree, anchors_ptr, n_anchors, start_bnd, end_bnd);
}


// Per-chunk anchor walk: walks pt_mid[begin_t, end_t) between chunk-boundary anchors
// (ROOT / root.Next() at the ends; else the split-leaf of (pt[begin], pt[begin+M])).
template <class Real, Integer DIM, WalkMode MODE> struct ChunkedWalkFunctor {
  const MortonCode<DIM>* pt_mid;
  Long N, M, nthreads;
  const Long* offsets;  // Write only
  Morton<DIM>* out;     // Write only
  Morton<DIM> start_bnd;  // rank's lower boundary anchor (ROOT on the first rank)
  Morton<DIM> end_bnd;    // rank's upper boundary, exclusive (root.Next() on the last rank)

  SCTL_GPU_HD Long operator()(Long tid) const {
    using NodeT = Morton<DIM>;
    const SplitLeafFunctor<Real, DIM> split{pt_mid, M};

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
    const auto walk_to = [&](const NodeT& target) {
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
// ChunkedWalkFunctor, no anchor materialization; 64-particle min-chunk floor won everywhere.
template <class Real, Integer DIM, template <class...> class DeviceVector>
void buildTreeGpuChunked(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M, Long N_owned = -1, Long base = 0, Morton<DIM> start_bnd = Morton<DIM>{}, Morton<DIM> end_bnd = Morton<DIM>{}.Next()) {
  const Long N = (N_owned < 0 ? static_cast<Long>(pt_mid.size()) : N_owned);  // walk window is pt_mid[base, base+N) (+M halo slack beyond)
  if (rootOnlyTree<DIM, DeviceVector>(tree, N, M, start_bnd, end_bnd)) return;

  const Long min_chunk = std::max<Long>(4 * M + 1, 64);
  const Long nthreads  = std::clamp<Long>(N / min_chunk, 1, 65536);

  const auto pol = detail::scratch_policy<DeviceVector, MortonCode<DIM>>();
  DeviceScratch<Long, DeviceVector> counts(nthreads), offsets(nthreads);
  ChunkedWalkFunctor<Real, DIM, WalkMode::Count> fc{thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads, nullptr, nullptr, start_bnd, end_bnd};
  thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(nthreads), counts.begin(), fc);
  const Long total = detail::scanCounts(pol, counts, offsets, nthreads);

  tree.resize(total);
  ChunkedWalkFunctor<Real, DIM, WalkMode::Write> fw{
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
// Single-pass on purpose: a count-then-rewrite 2-pass was ~25% slower at N=100M,M=300 (the
// second walk re-reads pt_mid cache-cold).
template <class Real, Integer DIM, template <class...> class DeviceVector>
void buildTreeCpuChunked(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M, Long N_owned = -1, Long base = 0, Morton<DIM> start_bnd = Morton<DIM>{}, Morton<DIM> end_bnd = Morton<DIM>{}.Next()) {
  using NodeMIDT = Morton<DIM>;
  const Long N = (N_owned < 0 ? static_cast<Long>(pt_mid.size()) : N_owned);  // walk window is pt_mid[base, base+N) (+M halo slack beyond)
  if (rootOnlyTree<DIM, DeviceVector>(tree, N, M, start_bnd, end_bnd)) return;

  // Cap threads so each chunk has well over M particles (so `begin + M` stays in-bounds).
  const int max_threads = SCTL_GET_MAX_THREADS();
  const Long min_chunk = std::max<Long>(4 * M + 1, 1024);
  const int nthreads = std::clamp<int>(static_cast<int>(N / min_chunk), 1, max_threads);

  // Upper bound: ~(MAX_DEPTH+1) nodes/leaf, chunk_size/M leaves/chunk, 4x slack.
  const Long chunk_size_max = (N + nthreads - 1) / nthreads;
  const Long max_emits = 4 * chunk_size_max * (MAX_DEPTH + 1) / std::max<Long>(1, M) + 4 * (MAX_DEPTH + 1) * (Long(1) << DIM) + 16;  // constant term: boundary anchors can sit at MAX_DEPTH

  sctl::ScratchBuf<PaddedLong> local_sizes(nthreads);  // padded: concurrent per-thread writes
  sctl::ScratchBuf<Long> offsets(nthreads);            // written once by `single`, read-only after
  sctl::ScratchBuf<Long> zero_offsets(nthreads);       // functor reads `offsets[tid] == 0`
  for (int t = 0; t < nthreads; ++t) zero_offsets[t] = 0;

  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = SCTL_GET_THREAD_NUM();
    sctl::ScratchBuf<NodeMIDT> buf(max_emits);  // NUMA-local: first-touched on this thread's node
    const ChunkedWalkFunctor<Real, DIM, WalkMode::Write> fw{thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads, &zero_offsets[0], &buf[0], start_bnd, end_bnd};
    const Long count = fw(tid);
    local_sizes[tid].v = count;

    #pragma omp barrier
    #pragma omp single
    {
      Long total = 0;
      for (int s = 0; s < nthreads; ++s) {
        offsets[s] = total;
        total += local_sizes[s].v;
      }
      tree.resize(total);
    }

    NodeMIDT* out_ptr = thrust::raw_pointer_cast(tree.data()) + offsets[tid];
    for (Long i = 0; i < count; ++i) out_ptr[i] = buf[i];
  }
}

}  // namespace detail_build

// Splitter selection for the distributed sort.
namespace detail_determineSplitters {
using detail::is_device_vector_v;

// Exact-rank splitters for the distributed sort, replicated on every rank: seed np-1 cuts from
// per-rank data boundaries, then iterate probe -> gather candidates -> exact global ranks ->
// refine until each is within tol. Un-splittable cuts (target inside a duplicate run wider than
// tol) are frozen at the nearest achievable endpoint.
template <class Type, template <class...> class DeviceVector>
void determineSplitters(sctl::Vector<Type>& splitters, const DeviceVector<Type>& pt, const Comm& comm) {
  constexpr Integer MAXIT = 50;
  constexpr Integer budget = 16; // probes/round budget
  constexpr double tolfrac = 0.02; // 2% load-balance tolerance

  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long ns = np - 1;
  if (splitters.Dim() != ns) splitters.ReInit(ns);
  if (!ns) return;

  const Long Nl = static_cast<Long>(pt.size());
  const Long Ng = [&Nl,&comm](){
    Long g;
    comm.Allreduce(sctl::Ptr2ConstItr<Long>(&Nl,1), sctl::Ptr2Itr<Long>(&g,1), 1, sctl::CommOp::SUM);
    return g;
  }();
  if (!Ng) return;

  const Long tol = std::max<Long>(1, Long(tolfrac * double(Ng) / double(np)));   // load-balance tolerance
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
    if constexpr (is_device_vector_v<DeviceVector<Type>>) {  // device: thrust batched binary search
      DeviceScratch<Type, DeviceVector> q_d(n);
      DeviceScratch<Long, DeviceVector> r_d(n);
      const auto pol = detail::scratch_policy<DeviceVector, Type>();
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
    const Long B = sctl::omp_par::dedup_sorted(bnd.begin(), bnd2.begin(), bnd.Dim());  // out-of-place: bnd -> bnd2

    sctl::ScratchBuf<Long> lr_b(B), gr_b(B); // each boundary's local then exact global rank
    local_ranks(lr_b.begin(), bnd2.begin(), B);
    comm.Allreduce(lr_b.begin(), gr_b.begin(), B, sctl::CommOp::SUM);

    #pragma omp parallel for schedule(static) if(ns > 512)
    for (Long i = 0; i < ns; i++) {                              // straddling boundary pair for target rank t
      const Long t = (i + 1) * Ng / np;
      const Long up = std::min(B-1, std::max<Long>(1, std::lower_bound(gr_b.begin(), gr_b.begin()+B, t) - gr_b.begin()));
      const Long lo = std::max<Long>(0, up - 1);                 // lo>=0 even when B==1 (degenerate all-equal data)
      bracket[i*2+0] = bnd2[lo]; bracket_rl[i*2+0] = lr_b[lo]; bracket_rg[i*2+0] = gr_b[lo];
      bracket[i*2+1] = bnd2[up]; bracket_rl[i*2+1] = lr_b[up]; bracket_rg[i*2+1] = gr_b[up];
    }
  }();

  sctl::ScratchBuf<char> state(ns);
  enum CutState : char { ACTIVE = 0, DONE = 1, DONE_UPPER = 2 };
  std::fill(state.begin(), state.end(), (char)ACTIVE);


  const auto gather_pt = [&pt]
                         (sctl::Vector<Type>& out, const sctl::Vector<Long>& idxs) {  // out: pt[idxs]; in: idxs. one gather (+D2H on device)
    const Long n = idxs.Dim();
    if (out.Dim() != n) out.ReInit(n);
    if (!n) return;
    DeviceScratch<Long, DeviceVector> idx_d(n);
    DeviceScratch<Type, DeviceVector> gv_d(n);
    thrust::copy(idxs.begin(), idxs.end(), idx_d.begin());
    thrust::gather(detail::scratch_policy<DeviceVector, Type>(), idx_d.begin(), idx_d.end(), pt.begin(), gv_d.begin());
    thrust::copy(gv_d.begin(), gv_d.end(), out.begin());
  };

  // out: idxs (this rank's chosen local indices), local_cand (their point values). budget is constexpr -> no capture.
  const auto probe = [ns,np,Ng,&state,&bracket,&bracket_rl,&bracket_rg,&next,&gather_pt]
                     (sctl::Vector<Long>& idxs, sctl::Vector<Type>& local_cand) {
    const double budget_ = [&state,ns]() {  // concentrate the round's budget onto the shrinking active set
      Long active_cnt = 0;
      for (Long i = 0; i < ns; i++) active_cnt += (state[i] ? 0 : 1);
      return double(budget) * double(ns) / std::max<Long>(1, active_cnt);
    }();

    idxs.ReInit(0);
    const Long start = Long(next() % (uint64_t)ns);
    for (Long j = 0; j < ns && idxs.Dim() < 2*budget; j++) {
      const Long i = (start + j) % ns;
      const Long rl0 = bracket_rl[i*2+0], rl1 = bracket_rl[i*2+1];
      const Long rg0 = bracket_rg[i*2+0], rg1 = bracket_rg[i*2+1];
      if (state[i] || rl1 == rl0) continue;

      const double share = double(rl1 - rl0) / double(rg1 - rg0);
      const double u = double(next() >> 11) * 0x1p-53; // uniform [0,1): top 53 bits scaled by 2^-53
      if (u >= std::min(1.0, double(budget_) * share)) continue;

      const Long opt_rank = (i + 1) * Ng / np;
      const Long idx = rl0 + (rl1 - rl0) * (opt_rank - rg0) / (rg1 - rg0); // interpolate to the target
      idxs.PushBack(std::min(rl1 - 1, std::max(rl0, idx)));
    }
    gather_pt(local_cand, idxs);
  };

  // in: local_cand (this rank's probes); out: cand (all ranks' probes ++ active brackets, sorted+deduped). ret: |cand|.
  const auto gather_candidates = [&comm,np,ns,&state,&bracket]
                                 (sctl::Vector<Type>& cand, const sctl::Vector<Type>& local_cand) -> Long {
    const Long mloc = local_cand.Dim();
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

    cand.ReInit(S); // capacity for the dedup output (>= result)
    sctl::omp_par::merge_sort(cand_raw.begin(), cand_raw.begin() + S);
    return sctl::omp_par::dedup_sorted(cand_raw.begin(), cand.begin(), S);  // cand_raw -> cand[0..ret)
  };

  // in: cand, S; out: lr, gr sized S+ns. gr[0,S) = exact global ranks of cand;
  // gr[S+i] = global upper_bound rank of bracket[i*2+0] (end of blo's duplicate run), folded into the same Allreduce.
  const auto global_ranks = [&comm,ns,&local_ranks,&bracket]
                            (sctl::ScratchBuf<Long>& lr, sctl::ScratchBuf<Long>& gr, const sctl::Vector<Type>& cand, Long S) {
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
                      (const sctl::Vector<Type>& cand, const sctl::ScratchBuf<Long>& lr, const sctl::ScratchBuf<Long>& gr, Long S) -> bool {
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
        bracket[i*2+0]    = cand[lo]; bracket[i*2+1]    = cand[up];
        bracket_rl[i*2+0] = lr[lo];   bracket_rl[i*2+1] = lr[up];
        bracket_rg[i*2+0] = gr[lo];   bracket_rg[i*2+1] = gr[up];
        anyactive = true; continue;
      }

      // no shrink: cand[lo]==blo. Exact un-splittable test on blo's duplicate run [L,U).
      const Long L = gr[lo], U = gr[S+i];  // L = global lower_bound(blo), U = global upper_bound(blo)
      if (U <= t) { // run ends before target -> undersampled this round, keep probing
        anyactive = true;
        continue;
      }
      // run straddles target: nearest endpoint is the best achievable.
      if (t - L <= U - t) { splitters[i] = cand[lo]; state[i] = DONE; } // nearer endpoint L: splitter = blo (value in hand)
      else                  state[i] = DONE_UPPER;                      // nearer endpoint U: splitter = successor of blo (resolved at loop end)
    }
    return anyactive;
  };

  sctl::Vector<Long> idxs;
  sctl::Vector<Type> local_cand, cand;
  for (Integer it = 0; it < MAXIT; it++) { // iterate: [ probe -> gather -> global-rank -> refine ] until every cut is within tol
    probe(idxs, local_cand);
    const Long S = gather_candidates(cand, local_cand);
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
        sctl::Vector<Type> sloc_v(sloc);
        if (Nl > 0) gather_pt(sloc_v, sctl::Vector<Long>(gidx));
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
template <Integer DIM, template <class...> class DeviceVector>
Long completeTree(Morton<DIM>* out, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const auto pol = scratch_policy<DeviceVector, Morton<DIM>>();
  DeviceScratch<Long, DeviceVector> off(1), cnt(1);
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
  Morton<DIM> next_first;  // first node of the right neighbor rank (walk-order successor of tree[Nn-1])
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

}  // namespace detail_balance21



// Hybrid balance (-DGT_BALANCE_HOST=1): extract the non-leaf set on the device, close it with
// sctl's Balance21 on the host (OpenMP, includes its own redistribute), then rebuild the leaves on
// the device. The non-leaf set is ~1/2^DIM of the tree, so the PCIe transfer is small; it wins over
// the device closure only on small trees, where the device is launch-bound.
namespace detail_balance21_host {
using detail::treeFromAnchors;
using detail_balance21::NonLeafPred;
using detail_balance21::FirstChildInSlice;
using detail_balance21::kCompleteTreeMax;
using detail_balance21::completeTree;
using detail_balance21::InvalidDepthPred;

template <Integer DIM, template <class...> class DeviceVector>
void balanceTreeDist(DeviceVector<Morton<DIM>>& tree, const sctl::Vector<Morton<DIM>>& mins, const Comm& comm, sctl::Periodicity periodicity) {
  using NodeT = Morton<DIM>;
  const auto pol = detail::scratch_policy<DeviceVector, NodeT>();
  const Long rank = comm.Rank();
  const Long np = comm.Size();
  const NodeT end_target = (rank + 1 < np) ? mins[rank + 1] : NodeT{}.Next();
  const Long Nn = (Long)tree.size();

  sctl::Vector<NodeT> S;  // passed straight to Balance21 (in/out)
  { // Balance21 builds a tree from the root, so every node's ancestors must be present. Extend the
    // slice to the whole domain on the device -- walk ROOT -> mins[rank] and mins[rank+1] -> end --
    // then take the non-leaf nodes of that (the fill contributes only the boundary ancestors).
    constexpr Long BND = kCompleteTreeMax<DIM>;
    DeviceScratch<NodeT, DeviceVector> lf(rank > 0 ? BND : 0), rt(rank + 1 < np ? BND : 0);
    Long nl_ = 0, nr_ = 0;
    if (rank > 0) nl_ = completeTree<DIM, DeviceVector>(thrust::raw_pointer_cast(lf.data()), NodeT{}, mins[rank]);
    if (rank + 1 < np) nr_ = completeTree<DIM, DeviceVector>(thrust::raw_pointer_cast(rt.data()), end_target, NodeT{}.Next());

    const Long Nf = nl_ + Nn + nr_;
    DeviceScratch<NodeT, DeviceVector> full(Nf);
    thrust::copy(lf.begin(), lf.begin() + nl_, full.begin());
    thrust::copy(tree.begin(), tree.end(), full.begin() + nl_);
    thrust::copy(rt.begin(), rt.begin() + nr_, full.begin() + nl_ + Nn);

    const NonLeafPred<DIM> is_nonleaf{thrust::raw_pointer_cast(full.data()), Nf, NodeT{}.Next()};
    if constexpr (detail::is_device_vector_v<DeviceVector<NodeT>>) {
      DeviceScratch<Long, DeviceVector> nl(Nf);
      const Long Nnl = thrust::copy_if(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nf), nl.begin(), is_nonleaf) - nl.begin();
      DeviceScratch<NodeT, DeviceVector> nlv(Nnl);
      thrust::gather(pol, nl.begin(), nl.begin() + Nnl, full.begin(), nlv.begin());
      S.ReInit(Nnl);  // one bulk transfer: copying via device iterators element-by-element is slow
      thrust::copy(nlv.begin(), nlv.begin() + Nnl, S.begin());  // already sorted (full is)
    } else {  // thrust's host backend is serial, so compact with OpenMP straight into S
      const NodeT* const fp = thrust::raw_pointer_cast(full.data());
      const Integer nt = SCTL_GET_MAX_THREADS();
      sctl::ScratchBuf<Long> dsp(nt + 1);
      dsp[0] = 0;
      #pragma omp parallel num_threads(nt)
      { const Integer tid = SCTL_GET_THREAD_NUM();
        Long c = 0;
        for (Long i = Nf * tid / nt; i < Nf * (tid + 1) / nt; i++) c += is_nonleaf(i);
        dsp[tid + 1] = c;
      }
      std::inclusive_scan(dsp.begin() + 1, dsp.end(), dsp.begin() + 1);
      S.ReInit(dsp[nt]);
      #pragma omp parallel num_threads(nt)
      { const Integer tid = SCTL_GET_THREAD_NUM();
        Long o = dsp[tid];
        for (Long i = Nf * tid / nt; i < Nf * (tid + 1) / nt; i++) if (is_nonleaf(i)) S[o++] = fp[i];
      }
    }
  }

  { // sctl's balance21 over the non-leaf set (host, OpenMP); it also redistributes by mins
    sctl::tree_detail::Balance21<DIM>(S, mins, comm, periodicity);
  }

  { // balanced non-leaf set -> device; the leaves are built there
    DeviceVector<NodeT> S_d(S.Dim());
    thrust::copy(S.begin(), S.end(), S_d.begin());
    // anchors = first child of each non-leaf (as in Tree::UpdateRefinement's "add children of
    // parent_mid"); first_child keeps mid and adds a level, so the sequence stays sorted and the
    // walk between anchors emits the leaves.
    DeviceScratch<NodeT, DeviceVector> anch_d(S.Dim());
    thrust::transform(pol, S_d.begin(), S_d.end(), anch_d.begin(), FirstChildInSlice<DIM>{mins[rank], end_target});
    const Long na = thrust::remove_if(pol, anch_d.begin(), anch_d.end(), InvalidDepthPred<DIM>{}) - anch_d.begin();
    treeFromAnchors<DIM>(tree, thrust::raw_pointer_cast(anch_d.data()), na, mins[rank], end_target);
  }
}

}  // namespace detail_balance21_host

// Default balance: everything on the device. Local closure over the sorted non-leaf set
// (ClosureFrontier), then redistribute by `mins` and dedup (Stage2, CUDA-aware MPI straight from
// device buffers), then rebuild the leaves. Nothing crosses PCIe.
namespace detail_balance21_gpu {
using detail::local_sort;

// Expand one frontier node: for each distinct parent-neighbor of its 3^DIM same-depth neighbors
// (the p2n map: <= 2^DIM of them), emit it if absent from the sorted non-leaf set S. The map is a
// by-value member, so it rides in kernel parameters.
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
        if (q.Depth() == NodeT::INVALID_DEPTH) { w[j] = inv; continue; }
        const Long lo = detail::lowerBound(S, Long(0), ns, q);
        w[j] = (lo < ns && !(S[lo] < q) && !(q < S[lo])) ? inv : q;
      }
    }
    for (; j < MAX_CHILD; j++) w[j] = inv;
  }
};

// Expand only the nodes added in the previous round, looking their parent-neighbors up in the
// non-leaf set.
template <Integer DIM, template <class...> class DeviceVector>
void ClosureFrontier(DeviceVector<Morton<DIM>>& S, sctl::Periodicity periodicity) {
  using NodeT = Morton<DIM>;
  constexpr Integer K = sctl::pow<DIM, Integer>(3);
  constexpr Integer MAX_CHILD = (1u << DIM);
  const auto pol = detail::scratch_policy<DeviceVector, NodeT>();
  DeviceScratch<Integer, DeviceVector> pl_d(MAX_CHILD * K), pc_d(MAX_CHILD);
  { // distinct parent-neighbors per child slot, from sctl's nbr_path table
    std::array<Integer, MAX_CHILD * K> pl{};
    std::array<Integer, MAX_CHILD> pc{};
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
  const auto grow = [](DeviceVector<NodeT>& v, Long need) {
    if ((Long)v.size() < need) v.resize(std::max<Long>(need, 2 * (Long)v.size()));
  };
  DeviceVector<NodeT>& F = detail::PersistentBuffer<NodeT, DeviceVector, detail::Buf::Frontier>();
  F.resize(S.size());
  thrust::copy(pol, S.begin(), S.end(), F.begin());
  Long ns = (Long)S.size(), nf = ns;
  for (int round = 0; round < 4 * MAX_DEPTH; round++) {
    if (!nf) break;
    const Long chunk = std::min<Long>(nf, 4000000 / MAX_CHILD + 1);
    DeviceScratch<NodeT, DeviceVector> buf(chunk * MAX_CHILD), add(nf * MAX_CHILD);

    Long nadd = 0;
    for (Long c0 = 0; c0 < nf; c0 += chunk) {
      const Long nc = std::min<Long>(chunk, nf - c0);
      detail::dispatchPeriodicity<DIM>(periodicity, [&](auto per_c) {
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
      DeviceScratch<NodeT, DeviceVector> m(ns + nadd);
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


// Stage 2 on the device: redistribute the closed non-leaf set so rank r keeps [mins[r], mins[r+1]),
// then de-duplicate. Device buffers go straight into MPI (CUDA-aware), so nothing leaves the GPU.
template <Integer DIM, template <class...> class DeviceVector>
void Stage2(DeviceVector<Morton<DIM>>& S, const sctl::Vector<Morton<DIM>>& mins, const Comm& comm) {
  using NodeT = Morton<DIM>;
  const Long np = comm.Size();
  const auto pol = detail::scratch_policy<DeviceVector, NodeT>();
  const auto uniq = [&pol](DeviceVector<NodeT>& v) {
    v.resize(detail::local_unique(pol, v, (Long)v.size()));
  };
  if (np == 1) { uniq(S); return; }

#ifdef SCTL_HAVE_MPI
  sctl::ScratchBuf<Long> scnt(np), rcnt(np);
  Long Nrecv = 0;
  { // S is sorted, so each rank's block is contiguous: split at the mins
    DeviceScratch<NodeT, DeviceVector> mins_d(np);
    thrust::copy(mins.begin(), mins.begin() + np, mins_d.begin());
    Nrecv = detail::splitCounts(scnt, rcnt, S, (Long)S.size(), mins_d, comm);
  }

  DeviceVector<NodeT>& recv = detail::PersistentBuffer<NodeT, DeviceVector, detail::Buf::ClosureRecv>();
  detail::exchangePooled(pol, S, (Long)S.size(), recv, Nrecv, scnt, rcnt, comm);
  local_sort(pol, recv, Nrecv);  // np sorted runs -> one sorted block
  uniq(recv);
  S.swap(recv);
#endif
}

// Whole 2:1 balance on the device: extract the non-leaf set, close it, redistribute, rebuild leaves.
template <Integer DIM, template <class...> class DeviceVector>
void balanceTreeDist(DeviceVector<Morton<DIM>>& tree, const sctl::Vector<Morton<DIM>>& mins, const Comm& comm, sctl::Periodicity periodicity) {
  using NodeT = Morton<DIM>;
  const Long rank = comm.Rank(), np = comm.Size();
  const NodeT end_target = (rank + 1 < np) ? mins[rank + 1] : NodeT{}.Next();
  const Long Nn = (Long)tree.size();

  const auto pol = detail::scratch_policy<DeviceVector, NodeT>();
  DeviceVector<NodeT>& S = detail::PersistentBuffer<NodeT, DeviceVector, detail::Buf::Closure>();
  { // extend the slice to the whole domain, then take its non-leaf nodes
    constexpr Long BND = detail_balance21::kCompleteTreeMax<DIM>;
    DeviceScratch<NodeT, DeviceVector> lf(rank > 0 ? BND : 0), rt(rank + 1 < np ? BND : 0);
    Long nl_ = 0, nr_ = 0;
    if (rank > 0) nl_ = detail_balance21::completeTree<DIM, DeviceVector>(thrust::raw_pointer_cast(lf.data()), NodeT{}, mins[rank]);
    if (rank + 1 < np) nr_ = detail_balance21::completeTree<DIM, DeviceVector>(thrust::raw_pointer_cast(rt.data()), end_target, NodeT{}.Next());

    const Long Nf = nl_ + Nn + nr_;
    DeviceScratch<NodeT, DeviceVector> full(Nf);
    thrust::copy(pol, lf.begin(), lf.begin() + nl_, full.begin());
    thrust::copy(pol, tree.begin(), tree.end(), full.begin() + nl_);
    thrust::copy(pol, rt.begin(), rt.begin() + nr_, full.begin() + nl_ + Nn);

    DeviceScratch<Long, DeviceVector> ix(Nf);
    const Long k = thrust::copy_if(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nf), ix.begin(),
                                   detail_balance21::NonLeafPred<DIM>{thrust::raw_pointer_cast(full.data()), Nf, NodeT{}.Next()}) - ix.begin();
    S.resize(k);
    thrust::gather(pol, ix.begin(), ix.begin() + k, full.begin(), S.begin());
  }
  ClosureFrontier<DIM>(S, periodicity);
  Stage2<DIM>(S, mins, comm);

  { // leaves: the walk between the first children of consecutive non-leaf nodes
    DeviceScratch<NodeT, DeviceVector> anch(S.size());
    thrust::transform(pol, S.begin(), S.end(), anch.begin(), detail_balance21::FirstChildInSlice<DIM>{mins[rank], end_target});
    const Long na = thrust::remove_if(pol, anch.begin(), anch.end(), detail_balance21::InvalidDepthPred<DIM>{}) - anch.begin();
    detail::treeFromAnchors<DIM>(tree, thrust::raw_pointer_cast(anch.data()), na, mins[rank], end_target);
  }
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
  const Long* offsets;      // Write only
  GhostPair<DIM>* out;      // Write only

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
template <Integer DIM, template <class...> class DeviceVector>
void addGhostNodes(DeviceVector<Morton<DIM>>& tree, const sctl::ScratchBuf<Morton<DIM>>& mins, const Comm& comm, Integer halo_size, sctl::Periodicity periodicity, Long& owned_begin, Long& owned_end,
                   DeviceVector<Morton<DIM>>* user_mid = nullptr, sctl::Vector<Long>* user_cnt = nullptr) {
  using NodeT = Morton<DIM>;
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long Nn = static_cast<Long>(tree.size());
  owned_begin = 0; owned_end = Nn;
  if (np == 1) return;

  // `mins` is the partition: mins[r] is rank r's first node, in code and depth alike, so the
  // boundaries need no gathering here.
  const NodeT lo = mins[rank], hi = (rank + 1 < np) ? mins[rank + 1] : NodeT{}.Next();
  const auto pol = detail::scratch_policy<DeviceVector, NodeT>();
  DeviceScratch<NodeT, DeviceVector> mins_d(np);
  thrust::copy(mins.begin(), mins.end(), mins_d.begin());
  // halo_size < 0 exchanges no neighbor nodes, so the scan is skipped and `pairs` stays empty. The
  // count and write passes are separate blocks because `pairs` is a pool slice: its size has to be
  // known at construction, and the scan is what produces it.
  const Long Nscan = (halo_size >= 0 ? Nn : 0);
  DeviceScratch<Long, DeviceVector> offsets(Nscan);
  Long npairs_tot = 0;
  if (Nscan) { // how many (dest rank, node) pairs each owned node produces
    DeviceScratch<Long, DeviceVector> counts(Nn);  // released before `pairs` is taken, so the pool stays LIFO
    detail::dispatchPeriodicity<DIM>(periodicity, [&](auto per_c) {
      const GhostSendFunctor<DIM, WalkMode::Count, decltype(per_c)::value> fc{thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(mins_d.data()), lo, hi, np, rank, halo_size, nullptr, nullptr};
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), counts.begin(), fc);
    });
    npairs_tot = detail::scanCounts(pol, counts, offsets, Nn);
  }
  Long npairs = 0;
  DeviceScratch<GhostPair<DIM>, DeviceVector> pairs(npairs_tot);
  if (Nscan) { // emit the pairs, sort by (dest rank, node), drop duplicates
    detail::dispatchPeriodicity<DIM>(periodicity, [&](auto per_c) {
      const GhostSendFunctor<DIM, WalkMode::Write, decltype(per_c)::value> fw{thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(mins_d.data()), lo, hi, np, rank, halo_size,
                                                                              thrust::raw_pointer_cast(offsets.data()), thrust::raw_pointer_cast(pairs.data())};
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), thrust::make_discard_iterator(), fw);
    });
    local_sort(pol, pairs, npairs_tot);
    npairs = detail::local_unique(pol, pairs, npairs_tot);
  }

  Long Nrecv = 0;
  sctl::ScratchBuf<Long> scnt(np), rcnt(np);
  DeviceScratch<NodeT, DeviceVector> send_mid(npairs);
  { // per-destination segments and counts; pairs are sorted by (dest rank, node)
    sctl::ScratchBuf<GhostPair<DIM>> keys(np);
    for (Long r = 0; r < np; r++) keys[r] = GhostPair<DIM>{r, NodeT{}};
    DeviceScratch<GhostPair<DIM>, DeviceVector> keys_d(np);
    thrust::copy(keys.begin(), keys.end(), keys_d.begin());
    Nrecv = detail::splitCounts(scnt, rcnt, pairs, npairs, keys_d, comm);
    thrust::transform(pol, pairs.begin(), pairs.begin() + npairs, send_mid.begin(), GhostPairToMid<DIM>{});
  }
  if (user_mid) {  // the halo send list: which of my nodes each rank wants as a ghost
    user_mid->resize(npairs);
    thrust::copy(pol, send_mid.begin(), send_mid.begin() + npairs, user_mid->begin());
  }
  if (user_cnt) {
    user_cnt->ReInit(np);
    for (Long r = 0; r < np; r++) (*user_cnt)[r] = scnt[r];
  }

  DeviceScratch<NodeT, DeviceVector> ghost(Nrecv);
  detail::alltoallv(thrust::raw_pointer_cast(send_mid.data()), thrust::raw_pointer_cast(ghost.data()), scnt, rcnt, sizeof(NodeT), comm);
  // sorted: each source's segment is sorted and source owned-intervals are ordered
  const Long Nsplit = thrust::lower_bound(pol, ghost.begin(), ghost.end(), mins[rank]) - ghost.begin();

  const NodeT* gp = thrust::raw_pointer_cast(ghost.data());
  DeviceScratch<Long, DeviceVector> off_l(rank > 0 ? Nsplit + 1 : 0), off_r(rank + 1 < np ? Nrecv - Nsplit + 1 : 0);
  const Long L = (rank > 0) ? detail::anchorWalkCount<DIM, DeviceVector>(off_l, gp, Nsplit, NodeT{}, mins[rank]) : 0;
  const Long R = (rank + 1 < np) ? detail::anchorWalkCount<DIM, DeviceVector>(off_r, gp + Nsplit, Nrecv - Nsplit, mins[rank + 1], NodeT{}.Next()) : 0;
  DeviceScratch<NodeT, DeviceVector> left(L), right(R);
  if (L) detail::anchorWalkWrite<DIM, DeviceVector>(thrust::raw_pointer_cast(left.data()), thrust::raw_pointer_cast(off_l.data()), gp, Nsplit, NodeT{}, mins[rank]);
  if (R) detail::anchorWalkWrite<DIM, DeviceVector>(thrust::raw_pointer_cast(right.data()), thrust::raw_pointer_cast(off_r.data()), gp + Nsplit, Nrecv - Nsplit, mins[rank + 1], NodeT{}.Next());

  // Swapped rather than assigned, so `tree` and this retained buffer trade storage each build
  // instead of one being freed and the other allocated.
  DeviceVector<NodeT>& merged = detail::PersistentBuffer<NodeT, DeviceVector, detail::Buf::GhostMerge>();
  merged.resize(L + Nn + R);
  thrust::copy(pol, left.begin(), left.end(), merged.begin());
  thrust::copy(pol, tree.begin(), tree.end(), merged.begin() + L);
  thrust::copy(pol, right.begin(), right.end(), merged.begin() + L + Nn);
  tree.swap(merged);
  owned_begin = L; owned_end = L + Nn;
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

// Connectivity in three passes, written straight into the caller's arrays. Parent and child are
// kept in their own compact arrays -- 8 and 64 bytes per node -- because the neighbor walk reads
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

// Pass 3, root only: its row seeds the propagation below. Walking from the root costs nothing for
// the root itself, and this keeps the one place periodicity must be consulted explicitly.
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
      if (m.depth == Morton<DIM>::INVALID_DEPTH) { out[k] = -1; continue; }
      Long cur = 0;  // the root is at index 0
      for (Integer l = 1; l <= d && cur >= 0; l++) cur = ch[cur * MAX_CHILD + m.Ancestor((uint8_t)l).Path2Node()];
      out[k] = cur;
    }
  }
};

// Pass 4, one depth at a time: a node's k-th neighbor is a child of its parent's k'-th neighbor,
// where k' and the child slot follow from the node's position bits within its parent plus the
// offset digits of k. Two array reads per entry, against a root-to-depth descent of dependent
// loads per entry: the host lists block went 758 -> 171 ms at 9.3M nodes, uninitialized resizes
// included. Periodicity needs no dispatch here: a
// wrapped neighbor is reached through the parent's wrapped row, and an out-of-domain one inherits
// the -1 its parent's row already carries. Reads touch only depth-1 rows, writes only depth rows,
// so running over the whole array per level does not alias.
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
// smallest per-rank count. Concatenated over ranks, the output matches single-rank buildTree.

template <class Real, Integer DIM, template <class...> class DevVec> template <template <class...> class DeviceVector>
void GPUTree<Real, DIM, DevVec>::buildTreeDist(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M, const Comm& comm, bool balance21, sctl::Periodicity periodicity, Integer halo_size, Long* owned_range, detail::no_deduce_t<DeviceVector<Long>>* sort_scatter_index, Morton<DIM>* partition, detail::no_deduce_t<DeviceVector<NodeAttr>>* node_attr, detail::no_deduce_t<NodeLists<DeviceVector>>* node_lists, detail::no_deduce_t<DeviceVector<Morton<DIM>>>* user_mid, sctl::Vector<Long>* user_cnt) {
  // Env-gated per-stage profiler (sync + barrier so each delta is the true stage wall time).
  const bool gtprof = (getenv("GTPROF") != nullptr);
  double t_last = 0;
  const auto mark = [&](const char* name) {
    if (!gtprof) return;
#if defined(__CUDACC__) || defined(__HIPCC__)
    cudaDeviceSynchronize();
#endif
    comm.Barrier();
    const double t = SCTL_GET_WTIME();
    if (comm.Rank() == 0 && name) fprintf(stderr, "  %-24s %8.2f ms\n", name, (t - t_last) * 1e3);
    t_last = t;
  };
  mark(nullptr);

  using MortonT = MortonCode<DIM>;
  const auto pol = detail::scratch_policy<DeviceVector, MortonT>();
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long Nglob = [&coord, &comm]() {
    sctl::StaticArray<Long,2> N{(Long)coord.size()/DIM, 0};
    comm.Allreduce<sctl::CommOp::SUM>(N+0, N+1, 1);
    return N[1];
  }();
  if (Nglob <= M) {  // all particles fit one leaf: root-only tree, held by rank 0
    tree.resize(rank == 0 ? 1 : 0);
    if (rank == 0) tree[0] = Morton<DIM>{};
    return;
  }

  // sort_scatter_index[i] = global pre-sort index of the particle at owned sorted position i,
  // carried through both redistributions below. Global, so concatenated over ranks it is a
  // permutation of [0, Nglob) matching the single-rank order.
  DeviceVector<Long> idx; // TODO: is scatter index handled efficiently?
  // Double-buffered: `pt_mid` is replaced three times below (sort, repartition, halo). Swapping with
  // a second retained buffer recycles the storage instead of freeing it and taking a fresh block.
  DeviceVector<MortonT>& pt_mid = detail::PersistentBuffer<MortonT, DeviceVector, detail::Buf::PtMid>();
  DeviceVector<MortonT>& alt = detail::PersistentBuffer<MortonT, DeviceVector, detail::Buf::PtAlt>();
  pt_mid.resize((Long)coord.size()/DIM);
  { // Encode coords -> Morton, then local sort (device radix / host omp_par).
    const Long Nloc = (Long)pt_mid.size();
    if constexpr (detail::is_device_vector_v<DeviceVector<Real>>) {
      detail::MakeMortonFunctor<Real, DIM> enc{thrust::raw_pointer_cast(coord.data())};
      thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nloc), pt_mid.begin(), enc);
    } else {
      const Real* cp = thrust::raw_pointer_cast(coord.data());
      MortonT* mp = thrust::raw_pointer_cast(pt_mid.data());
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < Nloc; ++i) mp[i] = MortonT(cp + i * DIM);
    }
    if (sort_scatter_index) {
      Long goff = 0;  // global index of this rank's first input particle
      comm.Scan<sctl::CommOp::SUM>(sctl::Ptr2ConstItr<Long>(&Nloc, 1), sctl::Ptr2Itr<Long>(&goff, 1), 1);
      goff -= Nloc;
      idx.resize(Nloc);
      thrust::sequence(pol, idx.begin(), idx.end(), goff);
      detail::local_sort_by_key(pol, pt_mid, idx, Nloc);
    } else {
      detail::local_sort(pol, pt_mid, Nloc);
    }
  }

  #ifdef SCTL_HAVE_MPI
  if (np > 1) { // distributed sort
    sctl::ScratchBuf<MortonT> spl_h_buf(np - 1);
    sctl::Vector<MortonT> spl_h(spl_h_buf);  // determineSplitters takes a Vector
    detail_determineSplitters::determineSplitters(spl_h, pt_mid, comm);
    DeviceScratch<MortonT, DeviceVector> spl_d(np - 1);
    thrust::copy(spl_h.begin(), spl_h.end(), spl_d.begin());

    sctl::ScratchBuf<Long> scnt(np), rcnt(np);
    const Long Nrecv = detail::splitCounts(scnt, rcnt, pt_mid, (Long)pt_mid.size(), spl_d, comm);

    detail::exchangePooled(pol, pt_mid, (Long)pt_mid.size(), alt, Nrecv, scnt, rcnt, comm);
    if (sort_scatter_index) {  // the index rides along on the same partition
      DeviceVector<Long> ibuf;
      detail::exchangePooled(pol, idx, (Long)idx.size(), ibuf, Nrecv, scnt, rcnt, comm);
      idx = std::move(ibuf);
      detail::local_sort_by_key(pol, alt, idx, Nrecv);  // np sorted segments -> one sorted block
    } else {
      detail::local_sort(pol, alt, Nrecv);
    }
    pt_mid.swap(alt);
  }
  mark("encode+sort+splitters");

  if (np > 1) { // M <- global_min(pt_mid.size(), M); repartition if necessary
    Long Nloc = (Long)pt_mid.size(), Nloc_min = 0;
    comm.Allreduce<sctl::CommOp::MIN>(sctl::Ptr2ConstItr<Long>(&Nloc, 1), sctl::Ptr2Itr<Long>(&Nloc_min, 1), 1);
    if (Nloc_min < M) {  // repartition
      sctl::ScratchBuf<Long> cnts(np), off(np + 1);
      comm.Allgather(sctl::Ptr2ConstItr<Long>(&Nloc, 1), 1, cnts.begin(), 1);

      off[0] = 0;  // global start offset of each rank's current block
      std::inclusive_scan(cnts.begin(), cnts.end(), off.begin() + 1);
      const Long my_lo = off[rank];
      const Long my_hi = off[rank + 1];
      const Long tgt_lo = rank * Nglob / np;
      const Long tgt_hi = (rank + 1) * Nglob / np;

      // every rank's block is known, so both send and recv counts are computed locally (no Alltoall)
      sctl::ScratchBuf<Long> scnt(np), rcnt(np);
      for (Long q = 0; q < np; q++) {
        const Long q_lo = q * Nglob / np;
        const Long q_hi = (q + 1) * Nglob / np;
        scnt[q] = std::max<Long>(0, std::min(my_hi, q_hi) - std::max(my_lo, q_lo));
        rcnt[q] = std::max<Long>(0, std::min(off[q + 1], tgt_hi) - std::max(off[q], tgt_lo));
      }

      detail::exchangePooled(pol, pt_mid, (Long)pt_mid.size(), alt, tgt_hi - tgt_lo, scnt, rcnt, comm);
      pt_mid.swap(alt);  // received segments concatenate in global-index order -> already sorted
      if (sort_scatter_index) {
        DeviceVector<Long> itmp;
        detail::exchangePooled(pol, idx, (Long)idx.size(), itmp, tgt_hi - tgt_lo, scnt, rcnt, comm);
        idx = std::move(itmp);
      }
      Nloc_min = Nglob / np;  // even split: smallest chunk is floor(Nglob/np)
    }

    M = std::min<Long>(M, Nloc_min);
    if (M < 1) MPI_Abort(comm.GetMPI_Comm(), 1);
  }
  mark("rebalance");

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

  sctl::ScratchBuf<Morton<DIM>> mins(np);
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
  mark("halo+mins");

  { // build linear tree from pt_mid
    const Morton<DIM> end_bnd = (rank + 1 < np) ? mins[rank + 1] : Morton<DIM>{}.Next();
    const Long idx0 = thrust::lower_bound(pol, pt_mid.begin(), std::min(pt_mid.begin()+2*M, pt_mid.end()), mins[rank].mid) - pt_mid.begin();
    const Long idx1 = thrust::lower_bound(pol, std::max(pt_mid.begin(), pt_mid.end()-2*M), pt_mid.end(), end_bnd.mid) - pt_mid.begin();

    if constexpr (detail::is_device_vector_v<DeviceVector<Real>>) {
      // anchor build wins for small slices, chunked for large; the choice depends only on local size.
      constexpr Long kChunkedThreshold = 128 * 1024;
      if ((idx1 - idx0) * M < kChunkedThreshold) detail_build::buildTreeGpu<Real, DIM>(tree, pt_mid, M, idx1 - idx0, idx0, mins[rank], end_bnd);
      else detail_build::buildTreeGpuChunked<Real, DIM>(tree, pt_mid, M, idx1 - idx0, idx0, mins[rank], end_bnd);
    } else {
      detail_build::buildTreeCpuChunked<Real, DIM>(tree, pt_mid, M, idx1 - idx0, idx0, mins[rank], end_bnd);
    }
  }
  mark("walk (linearize)");

#if defined(GT_BALANCE_HOST) && GT_BALANCE_HOST  // opt-in: closure on the host (see detail_balance21_host)
  if (balance21) detail_balance21_host::balanceTreeDist<DIM>(tree, sctl::Vector<Morton<DIM>>(mins), comm, periodicity);
#else
  if (balance21) detail_balance21_gpu::balanceTreeDist<DIM>(tree, sctl::Vector<Morton<DIM>>(mins), comm, periodicity);
#endif
  mark("balance21");

  Long owned_begin = 0, owned_end = Long(tree.size());
  detail_addGhostNodes::addGhostNodes<DIM>(tree, mins, comm, halo_size, periodicity, owned_begin, owned_end, user_mid, user_cnt);
  mark("ghost");

  if (sort_scatter_index) *sort_scatter_index = std::move(idx);

  if (owned_range) {
    owned_range[0] = owned_begin;
    owned_range[1] = owned_end;
  }
  if (partition) for (Long r = 0; r < np; r++) partition[r] = mins[r];
  if (node_attr) {
    const Long Nt = (Long)tree.size();
    node_attr->resize(Nt);
    thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nt),
                      node_attr->begin(),
                      detail_nodeAttr::NodeAttrFunctor<DIM, NodeAttr>{thrust::raw_pointer_cast(tree.data()), Nt, owned_begin, owned_end});
  }
  if (node_lists) {
    static constexpr Integer MAX_CHILD = 1 << DIM, MAX_NBRS = sctl::pow<DIM, Integer>(3);
    const Long Nt = (Long)tree.size();
    const Morton<DIM>* const tp = thrust::raw_pointer_cast(tree.data());
    node_lists->parent.resize(Nt);
    node_lists->child.resize(Nt * MAX_CHILD);
    node_lists->nbr.resize(Nt * MAX_NBRS);
    Long* const pp = thrust::raw_pointer_cast(node_lists->parent.data());
    Long* const cp = thrust::raw_pointer_cast(node_lists->child.data());
    Long* const np_ = thrust::raw_pointer_cast(node_lists->nbr.data());
    thrust::fill(pol, node_lists->child.begin(), node_lists->child.end(), Long(-1));
    thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::ParentPassFunctor<DIM>{tp, Nt, pp});
    thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::ChildPassFunctor<DIM>{tp, pp, cp});
    if constexpr (detail::is_device_vector_v<DeviceVector<Long>>) {
      // One kernel of independent root-to-depth descents: the device hides their latency with
      // parallelism and the shallow levels stay in cache, where twenty level-ordered launches
      // cost it ~8 ms in idle sweeps.
      detail::dispatchPeriodicity<DIM>(periodicity, [&](auto per_c) {
        thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::NbrDescentFunctor<DIM, decltype(per_c)::value>{tp, cp, np_});
      });
    } else {
      // The host is the opposite: the descents are dependent random loads (758 ms at 9.3M nodes),
      // so seed the root's row and propagate level by level instead.
      detail::dispatchPeriodicity<DIM>(periodicity, [&](auto per_c) {  // the root is always index 0
        thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Long(1), detail_nodeLists::NbrDescentFunctor<DIM, decltype(per_c)::value>{tp, cp, np_});
      });
      for (Integer d = 1; d <= Morton<DIM>::MAX_DEPTH; d++)
        thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), Nt, detail_nodeLists::NbrPropagateFunctor<DIM>{tp, pp, cp, np_, d});
    }
  }
}

// Stateful interface: the tree, the partition and any named per-node data live in the object, so a
// rebuild can carry the data across. `buildTreeDist` above does the building; everything here is
// bookkeeping around it, mirroring sctl::Tree.
namespace detail_treeData {

/** Redistribute a sorted device vector so this rank ends up with `Ntgt` elements, order preserved
 *  (device counterpart of `Comm::PartitionN`). Counts are known locally once every rank's size is. */
template <class T, template <class...> class DeviceVector, class Policy>
void partitionN(const Policy& pol, DeviceVector<T>& v, Long n, Long Ntgt, const Comm& comm) {
  const Long np = comm.Size(), rank = comm.Rank();
  if (np == 1) { v.resize(Ntgt); return; }
#ifdef SCTL_HAVE_MPI
  sctl::ScratchBuf<Long> cnt(np), off(np + 1), tgt(np), toff(np + 1);
  comm.Allgather(sctl::Ptr2ConstItr<Long>(&n, 1), 1, cnt.begin(), 1);
  comm.Allgather(sctl::Ptr2ConstItr<Long>(&Ntgt, 1), 1, tgt.begin(), 1);
  off[0] = 0; std::inclusive_scan(cnt.begin(), cnt.end(), off.begin() + 1);
  toff[0] = 0; std::inclusive_scan(tgt.begin(), tgt.end(), toff.begin() + 1);
  SCTL_ASSERT(off[np] == toff[np]);

  sctl::ScratchBuf<Long> scnt(np), rcnt(np);
  for (Long q = 0; q < np; q++) {  // overlap of my current block with q's target block, and vice versa
    scnt[q] = std::max<Long>(0, std::min(off[rank + 1], toff[q + 1]) - std::max(off[rank], toff[q]));
    rcnt[q] = std::max<Long>(0, std::min(off[q + 1], toff[rank + 1]) - std::max(off[q], toff[rank]));
  }
  DeviceVector<T>& dst = detail::PersistentBuffer<T, DeviceVector, detail::Buf::DataRecv>();
  detail::exchangePooled(pol, v, n, dst, Ntgt, scnt, rcnt, comm);
  v.swap(dst);
  v.resize(Ntgt);
#endif
}

}  // namespace detail_treeData

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
  sctl::Vector<Real> h((end - beg) * DIM);
  for (Long i = beg; i < end; i++) {
    Long idx = i;
    for (Integer k = 0; k < DIM; k++) { h[(i - beg) * DIM + k] = (Real)(idx % n0) / (Real)n0; idx /= n0; }
  }
  DevVec<Real> coord(h.Dim());
  thrust::copy(h.begin(), h.end(), coord.begin());
  UpdateRefinement(coord);
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::remapRanges(sctl::Vector<Long>& range, const DevVec<Morton<DIM>>& old_mid, const DevVec<Morton<DIM>>& new_mid) {
  const Long Nn = (Long)new_mid.size(), No = (Long)old_mid.size();
  range.ReInit(Nn + 1);
  range[Nn] = No;
  if constexpr (detail::is_device_vector_v<DevVec<Morton<DIM>>>) {
    const auto pol = detail::scratch_policy<DevVec, Morton<DIM>>();
    DeviceScratch<Long, DevVec> r(Nn);
    thrust::lower_bound(pol, old_mid.begin(), old_mid.begin() + No, new_mid.begin(), new_mid.begin() + Nn, r.begin());
    detail::deviceToHost(r.data(), Nn, range.begin());
  } else {  // thrust's host backend is serial unless built for OMP, so parallelize it here
    const Morton<DIM>* o = thrust::raw_pointer_cast(old_mid.data());
    const Morton<DIM>* n = thrust::raw_pointer_cast(new_mid.data());
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < Nn; i++) range[i] = std::lower_bound(o, o + No, n[i]) - o;
  }
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::UpdateRefinement(const DevVec<Real>& coord, Long M, bool balance21, sctl::Periodicity periodicity, Integer halo_size) {
  const Long np = comm_.Size(), rank = comm_.Rank();
  const auto pol = detail::scratch_policy<DevVec, Morton<DIM>>();

  // This rank's owned nodes before the rebuild, kept on the device: the search that consumes them
  // runs there too, so only its result has to cross the bus.
  DevVec<Morton<DIM>>& old_mid = detail::PersistentBuffer<Morton<DIM>, DevVec, detail::Buf::OldMid>();
  const bool remap = !node_data_.empty() && mins_.Dim();
  old_mid.resize(remap ? owned_end_ - owned_begin_ : 0);
  if (remap) thrust::copy(pol, node_mid_.begin() + owned_begin_, node_mid_.begin() + owned_end_, old_mid.begin());
  const Long old_begin = owned_begin_, old_end = owned_end_;

  { // rebuild
    mins_.ReInit(np);
    Long owned[2] = {0, 0};
    buildTreeDist(node_mid_, coord, M, comm_, balance21, periodicity, halo_size, owned,
                  (DevVec<Long>*)nullptr, mins_.begin(), &node_attr_, &node_lists_, &user_mid_, &user_cnt_);
    owned_begin_ = owned[0];
    owned_end_ = owned[1];
  }
  if (node_data_.empty()) return;

  // Remap: move the old owned nodes to whoever owns their range now, then find, for each new node,
  // the old nodes it absorbs. A new node's count is their counts summed.
  sctl::Vector<Long> range;
  Long No = 0;
  {
    if (np > 1) {  // PartitionS is a host operation, so the nodes round-trip for it alone
      sctl::Vector<Morton<DIM>> h((Long)old_mid.size());
      detail::deviceToHost(old_mid.data(), h.Dim(), h.begin());
      comm_.PartitionS(h, mins_[rank]);
      old_mid.resize(h.Dim());
      thrust::copy(h.begin(), h.end(), old_mid.begin());
    }
    No = (Long)old_mid.size();
    remapRanges(range, old_mid, node_mid_);
  }

  for (auto& kv : node_data_) {
    const std::string& name = kv.first;
    DevVec<char>& data = kv.second;
    sctl::Vector<Long>& cnt = node_cnt_[name];

    const Long dof = [this, &data, &cnt]() {
      sctl::StaticArray<Long, 2> Nl, Ng;
      Nl[0] = (Long)data.size();
      Nl[1] = sctl::omp_par::reduce(cnt.begin(), cnt.Dim());
      comm_.Allreduce((sctl::ConstIterator<Long>)Nl, (sctl::Iterator<Long>)Ng, 2, sctl::CommOp::SUM);
      const Long d = Ng[0] / std::max<Long>(Ng[1], 1);
      SCTL_ASSERT(Nl[0] == Nl[1] * d);
      return d;
    }();

    const Long data_begin = sctl::omp_par::reduce(cnt.begin(), old_begin);
    const Long data_count = sctl::omp_par::reduce(cnt.begin() + old_begin, old_end - old_begin);
    { // counts follow the nodes, then aggregate onto the new nodes
      sctl::Vector<Long> cnt_own(old_end - old_begin, cnt.begin() + old_begin, false);
      sctl::Vector<Long> cnt_tmp(cnt_own);
      if (np > 1) comm_.PartitionN(cnt_tmp, No);
      cnt.ReInit((Long)node_mid_.size());
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < (Long)node_mid_.size(); i++) {
        Long sum = 0;
        for (Long j = range[i]; j < range[i + 1]; j++) sum += cnt_tmp[j];
        cnt[i] = sum;
      }
    }
    { // the payload follows the same movement, on the device
      const Long Ndata = sctl::omp_par::reduce(cnt.begin(), cnt.Dim()) * dof;
      DevVec<char>& own = detail::PersistentBuffer<char, DevVec, detail::Buf::MigData>();
      own.resize(data_count * dof);
      thrust::copy(pol, data.begin() + data_begin * dof, data.begin() + (data_begin + data_count) * dof, own.begin());
      detail_treeData::partitionN(pol, own, data_count * dof, Ndata, comm_);
      data.swap(own);
    }
  }
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::AddData(const std::string& name, const DevVec<ValueType>& data, const sctl::Vector<Long>& cnt) {
  SCTL_ASSERT_MSG(node_data_.find(name) == node_data_.end(), "GPUTree::AddData: name already present.");
  { // dof must be uniform across ranks, as in sctl::Tree
    sctl::StaticArray<Long, 2> Nl, Ng;
    Nl[0] = (Long)data.size();
    Nl[1] = sctl::omp_par::reduce(cnt.begin(), cnt.Dim());
    comm_.Allreduce((sctl::ConstIterator<Long>)Nl, (sctl::Iterator<Long>)Ng, 2, sctl::CommOp::SUM);
    const Long dof = Ng[0] / std::max<Long>(Ng[1], 1);
    SCTL_ASSERT(Nl[0] == Nl[1] * dof);
    if (dof) SCTL_ASSERT(cnt.Dim() == (Long)node_mid_.size());
  }
  DevVec<char>& dst = node_data_[name];
  dst.resize((Long)data.size() * (Long)sizeof(ValueType));
  thrust::copy(thrust::device_pointer_cast((const char*)thrust::raw_pointer_cast(data.data())),
               thrust::device_pointer_cast((const char*)thrust::raw_pointer_cast(data.data())) + dst.size(), dst.begin());
  node_cnt_[name] = cnt;
}

template <class Real, Integer DIM, template <class...> class DevVec>
DevVec<char>& GPUTree<Real, DIM, DevVec>::AddDataUninit_(const std::string& name, const sctl::Vector<Long>& cnt, Long item_bytes) {
  SCTL_ASSERT_MSG(node_data_.find(name) == node_data_.end(), "GPUTree::AddDataUninit_: name already present.");
  if (item_bytes) SCTL_ASSERT(cnt.Dim() == (Long)node_mid_.size());
  DevVec<char>& dst = node_data_[name];
  dst.resize(sctl::omp_par::reduce(cnt.begin(), cnt.Dim()) * item_bytes);
  node_cnt_[name] = cnt;
  return dst;
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::GetData(DevVec<ValueType>& data, sctl::Vector<Long>& cnt, const std::string& name) const {
  const auto d = node_data_.find(name);
  const auto c = node_cnt_.find(name);
  SCTL_ASSERT_MSG(d != node_data_.end() && c != node_cnt_.end(), "GPUTree::GetData: unknown name.");
  SCTL_ASSERT(d->second.size() % sizeof(ValueType) == 0);
  data.resize((Long)d->second.size() / (Long)sizeof(ValueType));
  thrust::copy(d->second.begin(), d->second.end(),
               thrust::device_pointer_cast((char*)thrust::raw_pointer_cast(data.data())));
  cnt = c->second;
}


namespace detail_bcast {

/** dst[di[k]] = src[si[k]] -- touches only the elements the blocks actually cover. */
template <class T> struct BlockCopyFunctor {
  const T* src; T* dst; const Long* si; const Long* di;
  SCTL_GPU_HD void operator()(Long k) const { dst[di[k]] = src[si[k]]; }
};

/**
 * Copy variable-length blocks: `dst[dstoff[i]*w + j] = src[srcoff[i]*w + j]` for `j < len[i]*w`.
 * Expressed as a scatter rather than a gather, so destination elements no block covers are left
 * alone -- a gather over the whole destination would read an index for them that was never set.
 */
template <class T, template <class...> class DeviceVector, class Policy>
void blockCopy(const Policy& pol, DeviceVector<T>& dst, const DeviceVector<T>& src,
               const sctl::Vector<Long>& srcoff, const sctl::Vector<Long>& dstoff, const sctl::Vector<Long>& len, Long w) {
  const Long nb = len.Dim();
  sctl::Vector<Long> off(nb + 1);
  off[0] = 0;
  for (Long i = 0; i < nb; i++) off[i + 1] = off[i] + len[i] * w;
  const Long tot = off[nb];
  if (!tot) return;
  sctl::Vector<Long> si(tot), di(tot);
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < nb; i++) {
    for (Long j = 0; j < len[i] * w; j++) { si[off[i] + j] = srcoff[i] * w + j; di[off[i] + j] = dstoff[i] * w + j; }
  }
  DeviceScratch<Long, DeviceVector> si_d(tot), di_d(tot);
  thrust::copy(si.begin(), si.end(), si_d.begin());
  thrust::copy(di.begin(), di.end(), di_d.begin());
  thrust::for_each_n(pol, thrust::counting_iterator<Long>(0), tot, BlockCopyFunctor<T>{
      thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(dst.data()),
      thrust::raw_pointer_cast(si_d.data()), thrust::raw_pointer_cast(di_d.data())});
}

inline void scanv(sctl::Vector<Long>& dsp, const sctl::Vector<Long>& cnt) {
  dsp.ReInit(cnt.Dim());
  Long t = 0;
  for (Long i = 0; i < cnt.Dim(); i++) { dsp[i] = t; t += cnt[i]; }
}

}  // namespace detail_bcast

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::Broadcast(const std::string& name) {
  const Long np = comm_.Size();
  if (np == 1) return;
#ifdef SCTL_HAVE_MPI
  const auto pol = detail::scratch_policy<DevVec, char>();
  DevVec<char>& data = node_data_[name];
  sctl::Vector<Long>& cnt = node_cnt_[name];
  const Long Nn = (Long)node_mid_.size();
  SCTL_ASSERT(cnt.Dim() == Nn);

  sctl::Vector<Morton<DIM>> nmid(Nn);
  thrust::copy(node_mid_.begin(), node_mid_.end(), nmid.begin());
  sctl::Vector<Long> dsp; detail_bcast::scanv(dsp, cnt);
  Long nitem = 0; for (Long i = 0; i < Nn; i++) nitem += cnt[i];
  const Long w = (nitem ? (Long)data.size() / nitem : 0);  // bytes per item; Broadcast only copies

  // which of my nodes each rank wants, and how many items each carries
  const Long Ns = (Long)user_mid_.size();
  sctl::Vector<Morton<DIM>> smid(Ns);
  thrust::copy(user_mid_.begin(), user_mid_.end(), smid.begin());
  sctl::Vector<Long> sncnt(user_cnt_), sndsp, rncnt(np), rndsp;
  detail_bcast::scanv(sndsp, sncnt);
  comm_.Alltoall(sncnt.begin(), 1, rncnt.begin(), 1);
  detail_bcast::scanv(rndsp, rncnt);
  const Long Nr = rndsp[np - 1] + rncnt[np - 1];
  sctl::Vector<Morton<DIM>> rmid(Nr);
  comm_.Alltoallv(smid.begin(), sncnt.begin(), sndsp.begin(), rmid.begin(), rncnt.begin(), rndsp.begin());

  sctl::Vector<Long> sdcnt(Ns), rdcnt(Nr), sidx(Ns);
  for (Long i = 0; i < Ns; i++) {
    sidx[i] = std::lower_bound(nmid.begin(), nmid.begin() + Nn, smid[i]) - nmid.begin();
    SCTL_ASSERT(sidx[i] < Nn && nmid[sidx[i]] == smid[i]);
    sdcnt[i] = cnt[sidx[i]];
  }
  comm_.Alltoallv(sdcnt.begin(), sncnt.begin(), sndsp.begin(), rdcnt.begin(), rncnt.begin(), rndsp.begin());

  { // pack, exchange, then rebuild the array with the ghost slots filled
    sctl::Vector<Long> sddsp, rddsp;
    detail_bcast::scanv(sddsp, sdcnt);
    detail_bcast::scanv(rddsp, rdcnt);
    const Long Nsend = (Ns ? sddsp[Ns - 1] + sdcnt[Ns - 1] : 0);
    const Long Nrecv = (Nr ? rddsp[Nr - 1] + rdcnt[Nr - 1] : 0);

    sctl::Vector<Long> soff(Ns);
    for (Long i = 0; i < Ns; i++) soff[i] = dsp[sidx[i]];
    DevVec<char> sbuf(Nsend * w);
    detail_bcast::blockCopy(pol, sbuf, data, soff, sddsp, sdcnt, w);

    sctl::ScratchBuf<Long> sbc(np), rbc(np);
    for (Long p = 0; p < np; p++) {
      Long a = 0, b = 0;
      for (Long i = 0; i < sncnt[p]; i++) a += sdcnt[sndsp[p] + i];
      for (Long i = 0; i < rncnt[p]; i++) b += rdcnt[rndsp[p] + i];
      sbc[p] = a * w; rbc[p] = b * w;
    }
    DevVec<char> rbuf(Nrecv * w);
    detail::alltoallv(thrust::raw_pointer_cast(sbuf.data()), thrust::raw_pointer_cast(rbuf.data()), sbc, rbc, Long(1), comm_);

    sctl::Vector<Long> cnt_new(cnt), ridx(Nr);
    for (Long i = 0; i < Nr; i++) {
      ridx[i] = std::lower_bound(nmid.begin(), nmid.begin() + Nn, rmid[i]) - nmid.begin();
      SCTL_ASSERT(ridx[i] < Nn && nmid[ridx[i]] == rmid[i]);
      if (!cnt_new[ridx[i]]) cnt_new[ridx[i]] = rdcnt[i];
    }
    sctl::Vector<Long> dsp_new; detail_bcast::scanv(dsp_new, cnt_new);
    Long nnew = 0; for (Long i = 0; i < Nn; i++) nnew += cnt_new[i];

    DevVec<char> out(nnew * w);
    { // my own blocks keep their contents, at their new offsets
      sctl::Vector<Long> a(Nn), b(Nn), l(Nn);
      for (Long i = 0; i < Nn; i++) { a[i] = dsp[i]; b[i] = dsp_new[i]; l[i] = cnt[i]; }
      detail_bcast::blockCopy(pol, out, data, a, b, l, w);
    }
    { // received blocks land in the slots that were empty
      sctl::Vector<Long> a, b, l;
      for (Long i = 0; i < Nr; i++) if (!cnt[ridx[i]] && rdcnt[i]) { a.PushBack(rddsp[i]); b.PushBack(dsp_new[ridx[i]]); l.PushBack(rdcnt[i]); }
      detail_bcast::blockCopy(pol, out, rbuf, a, b, l, w);
    }
    data.swap(out);
    cnt = cnt_new;
  }
#endif
}

template <class Real, Integer DIM, template <class...> class DevVec>
template <class ValueType>
void GPUTree<Real, DIM, DevVec>::ReduceBroadcast(const std::string& name) {
  const Long np = comm_.Size(), rank = comm_.Rank();
  if (np == 1) return;
#ifdef SCTL_HAVE_MPI
  const auto pol = detail::scratch_policy<DevVec, ValueType>();
  DevVec<char>& data = node_data_[name];
  sctl::Vector<Long>& cnt = node_cnt_[name];
  const Long Nn = (Long)node_mid_.size();
  sctl::Vector<Morton<DIM>> nmid(Nn);
  thrust::copy(node_mid_.begin(), node_mid_.end(), nmid.begin());
  sctl::Vector<Long> dsp; detail_bcast::scanv(dsp, cnt);
  Long nitem = 0; for (Long i = 0; i < Nn; i++) nitem += cnt[i];
  const Long dof = (nitem ? (Long)data.size() / (Long)sizeof(ValueType) / nitem : 0);

  { // the ancestors of my first node are shared with earlier ranks; send them my partial values
    sctl::Vector<Morton<DIM>> smid;
    for (Integer d = 0; d < mins_[rank].Depth(); d++) smid.PushBack(mins_[rank].Ancestor(d));
    const Long Ns = smid.Dim();
    sctl::Vector<Long> sncnt(np), sndsp, rncnt(np), rndsp;
    for (Long p = 0; p < np; p++) {
      const Long a = std::lower_bound(smid.begin(), smid.begin() + Ns, mins_[p]) - smid.begin();
      const Long b = std::lower_bound(smid.begin(), smid.begin() + Ns, (p + 1 == np ? Morton<DIM>().Next() : mins_[p + 1])) - smid.begin();
      sncnt[p] = b - a;
    }
    detail_bcast::scanv(sndsp, sncnt);
    comm_.Alltoall(sncnt.begin(), 1, rncnt.begin(), 1);
    detail_bcast::scanv(rndsp, rncnt);
    const Long Nr = rndsp[np - 1] + rncnt[np - 1];
    sctl::Vector<Morton<DIM>> rmid(Nr);
    comm_.Alltoallv(smid.begin(), sncnt.begin(), sndsp.begin(), rmid.begin(), rncnt.begin(), rndsp.begin());

    sctl::Vector<Long> sdcnt(Ns), rdcnt(Nr), sidx(Ns);
    for (Long i = 0; i < Ns; i++) {
      sidx[i] = std::lower_bound(nmid.begin(), nmid.begin() + Nn, smid[i]) - nmid.begin();
      sdcnt[i] = (sidx[i] < Nn && nmid[sidx[i]] == smid[i]) ? cnt[sidx[i]] : 0;
    }
    comm_.Alltoallv(sdcnt.begin(), sncnt.begin(), sndsp.begin(), rdcnt.begin(), rncnt.begin(), rndsp.begin());

    sctl::Vector<Long> sddsp, rddsp;
    detail_bcast::scanv(sddsp, sdcnt);
    detail_bcast::scanv(rddsp, rdcnt);
    const Long Nsend = (Ns ? sddsp[Ns - 1] + sdcnt[Ns - 1] : 0);
    const Long Nrecv = (Nr ? rddsp[Nr - 1] + rdcnt[Nr - 1] : 0);

    sctl::Vector<Long> soff(Ns);
    for (Long i = 0; i < Ns; i++) soff[i] = (sdcnt[i] ? dsp[sidx[i]] : 0);
    DevVec<char> sbuf(Nsend * dof * (Long)sizeof(ValueType));
    detail_bcast::blockCopy(pol, sbuf, data, soff, sddsp, sdcnt, dof * (Long)sizeof(ValueType));

    sctl::ScratchBuf<Long> sbc(np), rbc(np);
    for (Long p = 0; p < np; p++) {
      Long a = 0, b = 0;
      for (Long i = 0; i < sncnt[p]; i++) a += sdcnt[sndsp[p] + i];
      for (Long i = 0; i < rncnt[p]; i++) b += rdcnt[rndsp[p] + i];
      sbc[p] = a * dof * (Long)sizeof(ValueType); rbc[p] = b * dof * (Long)sizeof(ValueType);
    }
    DevVec<char> rbuf(Nrecv * dof * (Long)sizeof(ValueType));
    detail::alltoallv(thrust::raw_pointer_cast(sbuf.data()), thrust::raw_pointer_cast(rbuf.data()), sbc, rbc, Long(1), comm_);

    { // add each received block into the node it belongs to
      ValueType* const d = (ValueType*)thrust::raw_pointer_cast(data.data());
      const ValueType* const r = (const ValueType*)thrust::raw_pointer_cast(rbuf.data());
      for (Long i = 0; i < Nr; i++) {
        if (!rdcnt[i]) continue;
        const Long idx = std::lower_bound(nmid.begin(), nmid.begin() + Nn, rmid[i]) - nmid.begin();
        if (idx >= Nn || !(nmid[idx] == rmid[i]) || cnt[idx] != rdcnt[i]) continue;
        thrust::transform(pol, thrust::device_pointer_cast(d + dsp[idx] * dof),
                          thrust::device_pointer_cast(d + (dsp[idx] + cnt[idx]) * dof),
                          thrust::device_pointer_cast(r + rddsp[i] * dof),
                          thrust::device_pointer_cast(d + dsp[idx] * dof), thrust::plus<ValueType>());
      }
    }
  }
  Broadcast<ValueType>(name);
#endif
}

template <class Real, Integer DIM, template <class...> class DevVec>
void GPUTree<Real, DIM, DevVec>::WriteTreeVTK(std::string fname, bool show_ghost) const {
  using VTKReal = typename sctl::VTUData::VTKReal;
  sctl::VTUData vtu_data;
  if (DIM <= 3) {  // one cell per leaf, its 2^DIM corners as points
    static constexpr Integer Ncorner = (1u << DIM);
    const Long Nn = (Long)node_mid_.size();
    sctl::Vector<Morton<DIM>> mid(Nn);
    sctl::Vector<NodeAttr> attr(Nn);
    thrust::copy(node_mid_.begin(), node_mid_.end(), mid.begin());
    thrust::copy(node_attr_.begin(), node_attr_.end(), attr.begin());

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

}  // namespace detail_ptTree

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
PtTree<Real, DIM, DevVec, BaseTree>::PtTree(const Comm& comm) : BaseTree(comm) {}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
Long PtTree<Real, DIM, DevVec, BaseTree>::globalDof(Long ndata, Long nitem) const {
  sctl::StaticArray<Long, 2> Nl, Ng;
  Nl[0] = ndata; Nl[1] = nitem;
  this->GetComm().Allreduce((sctl::ConstIterator<Long>)Nl, (sctl::Iterator<Long>)Ng, 2, sctl::CommOp::SUM);
  const Long dof = Ng[0] / std::max<Long>(Ng[1], 1);
  SCTL_ASSERT(Nl[0] == Nl[1] * dof);
  return dof;
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
sctl::Vector<MortonCode<DIM>> PtTree<Real, DIM, DevVec, BaseTree>::partitionCodes() const {
  const auto& mins = this->GetPartitionMID();
  sctl::Vector<MortonCode<DIM>> codes(mins.Dim());
  for (Long r = 0; r < mins.Dim(); r++) codes[r] = mins[r].mid;
  return codes;
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::nodeCounts(const std::string& name, sctl::Vector<Long>& cnt) const {
  const auto pol = detail::scratch_policy<DevVec, MortonCode<DIM>>();
  const auto& node_mid = this->GetNodeMID();
  const auto& pm = groups_.find(name)->second.SortedKeys();
  const Long Nn = (Long)node_mid.size(), Npt = (Long)pm.size();
  // The search is cheap; differencing its result on the host was not. Difference on the device and
  // move the finished counts across once.
  DeviceScratch<Long, DevVec> pos(Nn + 1);
  const auto nc0 = thrust::make_transform_iterator(node_mid.begin(), detail_ptTree::NodeToCodeFunctor<DIM>{});
  thrust::lower_bound(pol, pm.begin(), pm.begin() + Npt, nc0, nc0 + Nn, pos.begin());
  thrust::fill(pol, pos.begin() + Nn, pos.begin() + Nn + 1, Npt);
  DeviceScratch<Long, DevVec> d(Nn);
  thrust::transform(pol, pos.begin() + 1, pos.begin() + Nn + 1, pos.begin(), d.begin(), thrust::minus<Long>());
  cnt.ReInit(Nn);
  detail::deviceToHost(d.data(), Nn, cnt.begin());
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::AddParticles(const std::string& name, const DevVec<Real>& coord) {
  SCTL_ASSERT_MSG(groups_.find(name) == groups_.end(), "PtTree::AddParticles: name already present.");
  const auto pol = detail::scratch_policy<DevVec, MortonCode<DIM>>();
  const Long Nloc = (Long)coord.size() / DIM;
  SCTL_ASSERT((Long)coord.size() == Nloc * DIM);

  // Sort the bare MortonCode, not the Morton. Every particle sits at MAX_DEPTH, so the depth field
  // is a constant: carrying it doubles the key to 16 bytes, costs a struct comparison instead of a
  // 64-bit one, and moves twice the bytes through the exchange. Comparing codes gives the same
  // order and the same splits -- a particle sharing a node's code sorts after it either way, since
  // its depth is greater -- and nothing downstream needs the depth back.
  DevVec<MortonCode<DIM>> key(Nloc);
  thrust::transform(pol, thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nloc), key.begin(),
                    detail::MakeMortonFunctor<Real, DIM>{thrust::raw_pointer_cast(coord.data())});
  groups_.try_emplace(name, this->GetComm()).first->second.Init(std::move(key), partitionCodes());
  AddParticleData(name, name, coord);
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::AddParticleData(const std::string& data_name, const std::string& particle_name, const DevVec<Real>& data) {
  const auto it = groups_.find(particle_name);
  SCTL_ASSERT_MSG(it != groups_.end(), "PtTree::AddParticleData: unknown particle group.");
  SCTL_ASSERT_MSG(data_pt_name_.find(data_name) == data_pt_name_.end(), "PtTree::AddParticleData: data name already present.");
  const auto& g = it->second;
  const Long dof = globalDof((Long)data.size(), g.LocalCount());

  sctl::Vector<Long> cnt;
  nodeCounts(particle_name, cnt);
  // the forward scatter reads the caller's array and writes the stored buffer, so neither end is copied
  DevVec<char>& raw = this->AddDataUninit_(data_name, cnt, dof * (Long)sizeof(Real));
  g.ScatterForward((const Real*)thrust::raw_pointer_cast(data.data()), (Real*)thrust::raw_pointer_cast(raw.data()), dof);
  data_pt_name_[data_name] = particle_name;
}

template <class Real, Integer DIM, template <class...> class DevVec, class BaseTree>
void PtTree<Real, DIM, DevVec, BaseTree>::GetParticleData(DevVec<Real>& data, const std::string& data_name) const {
  const auto it = data_pt_name_.find(data_name);
  SCTL_ASSERT_MSG(it != data_pt_name_.end(), "PtTree::GetParticleData: unknown data name.");
  const auto& g = groups_.find(it->second)->second;

  // the reverse scatter reads the stored buffer and writes the output, so the payload is touched once
  const DevVec<char>& raw = this->NodeData_(data_name);
  const Long dof = globalDof((Long)raw.size() / (Long)sizeof(Real), g.SortedCount());
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
  const auto pol = detail::scratch_policy<DevVec, char>();
  BaseTree::UpdateRefinement(coord, M, balance21, periodicity, halo_size);

  const sctl::Vector<MortonCode<DIM>> codes = partitionCodes();
  for (auto& kv : groups_) {
    const std::string& group = kv.first;
    kv.second.Repartition(codes);  // slide the codes onto the new partition; the scatter maps follow

    // The base moved each payload by node: after a split, every item of an old node lands on the
    // first new node inside it, which can sit on a different rank than the particle's own Morton
    // says. So the payload gets a second, particle-aware repartition -- the same correction sctl
    // applies -- and the per-node counts are recomputed from the particles themselves.
    sctl::Vector<Long> cnt_new;
    nodeCounts(group, cnt_new);
    const Long Nnew = kv.second.SortedCount();
    Long ob = 0, oe = 0;
    this->GetOwnedRange(ob, oe);

    std::vector<std::string> names;
    for (const auto& p : data_pt_name_) if (p.second == group) names.push_back(p.first);
    for (const auto& name : names) {
      // in place: one staging buffer for the exchange, then swap. Going through GetData/AddData
      // would copy the payload three times over.
      DevVec<char>& raw = this->NodeData_(name);
      sctl::Vector<Long>& cnt = this->NodeCnt_(name);
      Long tot = 0, beg = 0, own = 0;
      for (Long i = 0; i < cnt.Dim(); i++) { if (i < ob) beg += cnt[i]; if (i >= ob && i < oe) own += cnt[i]; tot += cnt[i]; }
      // bytes per item, reduced globally: a rank that currently holds no particles has nothing to
      // divide by, and a local 0 would shrink its share of the repartition to nothing
      Long w = 0;
      { sctl::StaticArray<Long, 2> Nl, Ng;
        Nl[0] = (Long)raw.size(); Nl[1] = tot;
        comm.Allreduce((sctl::ConstIterator<Long>)Nl, (sctl::Iterator<Long>)Ng, 2, sctl::CommOp::SUM);
        w = Ng[0] / std::max<Long>(Ng[1], 1); }
      DevVec<char> slice(own * w);
      if (own * w) thrust::copy(pol, raw.begin() + beg * w, raw.begin() + (beg + own) * w, slice.begin());
      detail_treeData::partitionN(pol, slice, own * w, Nnew * w, comm);
      raw.swap(slice);
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
  DevVec<Real> pt_d, val_d;
  this->GetData(pt_d, pt_cnt, particle_name);
  this->GetData(val_d, val_cnt, data_name);
  sctl::Vector<Real> pt((Long)pt_d.size()), val((Long)val_d.size());
  thrust::copy(pt_d.begin(), pt_d.end(), pt.begin());
  thrust::copy(val_d.begin(), val_d.end(), val.begin());

  const Long Nn = (Long)this->GetNodeMID().size();
  sctl::Vector<typename BaseTree::NodeAttr> attr(Nn);
  thrust::copy(this->GetNodeAttr().begin(), this->GetNodeAttr().end(), attr.begin());
  Long npt = 0;
  for (Long i = 0; i < pt_cnt.Dim(); i++) npt += pt_cnt[i];
  const Long vdof = (npt ? (Long)val.Dim() / npt : 0);

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

}  // namespace gpu_tree

#include "sctl/experimental/sort-scatter.txx"  // uses the detail helpers above

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
