// Template implementation of GPUTree from gpu-tree.hpp + internal `detail::` helpers.

#ifndef _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
#define _SCTL_EXPERIMENTAL_GPU_TREE_TXX_

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "sctl/experimental/gpu-tree.hpp"
#include "sctl/ompUtils.txx"
#include "sctl/scratch_pool.hpp"
#include "sctl/scratch_pool.txx"

namespace gpu_tree {

namespace detail {

// True iff `Vec::data()` returns a thrust::device_ptr (i.e. Vec is GPU-resident).
template <class T> struct is_device_ptr                        : std::false_type {};
template <class T> struct is_device_ptr<thrust::device_ptr<T>> : std::true_type  {};

template <class Vec>
inline constexpr bool is_device_vector_v = is_device_ptr<typename std::decay<decltype(std::declval<Vec>().data())>::type>::value;

// Functor (not lambda) so nvcc captures it across thrust kernel boundaries.
template <class Real, Integer DIM> struct MakeMortonFunctor {
  const Real* coord_ptr;
  SCTL_GPU_HD MortonCode<DIM> operator()(Long i) const {
    return MortonCode<DIM>(coord_ptr + i * DIM);
  }
};

// Split-leaf of pair (pt[i], pt[i+M]): child of their common ancestor at depth d_common+1
// containing pt[i+M]. Pair distance M forces the split — if both endpoints sit in the same
// depth-d box, it contains M+1 > M particles and must refine.
template <class Real, Integer DIM> struct SplitLeafFunctor {
  const MortonCode<DIM>* pt;
  Long M;
  SCTL_GPU_HD Morton<DIM> operator()(Long i) const {
    uint8_t d = pt[i].CommonAncestor(pt[i + M]).depth;
    if (d < MAX_DEPTH) ++d;
    return pt[i + M].Ancestor(d);
  }
};

enum class WalkMode { Count, Write };

// DFS pre-order walk between consecutive anchors (ROOT -> anchors[0] for pair 0). Caller
// drives i over [0, n+1); i == n is a synthetic trailing pair with target = root.Next(),
// so the walk emits any leaves between anchors[n-1] and morton-end (matches sctl::Tree).
template <class Real, Integer DIM, WalkMode MODE> struct LinearizeWalkFunctor {
  const Morton<DIM>* anchors;
  Long n;
  const Long* offsets;  // Write only
  Morton<DIM>* out;     // Write only

  SCTL_GPU_HD Long operator()(Long i) const {
    using NodeT = Morton<DIM>;
    const bool is_tail = (i == n);
    const NodeT target = is_tail ? NodeT{}.Next() : anchors[i];
    NodeT current = (i == 0) ? NodeT{} : anchors[i - 1];

    Long count = 0;
    NodeT* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[i];

    if (i == 0) {  // pair 0 emits ROOT first
      if constexpr (MODE == WalkMode::Write) w[count] = current;
      ++count;
    }
    while (current < target) {
      const bool descend = current.depth < MAX_DEPTH && current.isAncestor(target);
      current = descend ? current.DFD(static_cast<uint8_t>(current.depth + 1)) : current.Next();
      if (is_tail && !(current < target)) break;  // sentinel is a marker, not a real node
      if constexpr (MODE == WalkMode::Write) w[count] = current;
      ++count;
    }
    return count;
  }
};

// Per-chunk variant of LinearizeWalkFunctor: walks pt_mid[begin_t, end_t) between synthetic
// chunk-boundary anchors (ROOT at tid=0, root.Next() at the last chunk; otherwise the split-
// leaf formula on (pt[begin], pt[begin+M])).
template <class Real, Integer DIM, WalkMode MODE> struct ChunkedWalkFunctor {
  const MortonCode<DIM>* pt_mid;
  Long N, M, nthreads;
  const Long* offsets;  // Write only
  Morton<DIM>* out;     // Write only

  // Binary lower_bound in [lo, hi); std::lower_bound is host-only.
  SCTL_GPU_HD Long lower_bound_window(Long lo, Long hi, const MortonCode<DIM>& key) const {
    while (lo < hi) {
      const Long mid = lo + (hi - lo) / 2;
      if (pt_mid[mid] < key) lo = mid + 1;
      else                   hi = mid;
    }
    return lo;
  }

  SCTL_GPU_HD Long operator()(Long tid) const {
    using NodeT = Morton<DIM>;
    const SplitLeafFunctor<Real, DIM> split{pt_mid, M};

    const Long  begin_t      = (N *  tid     ) / nthreads;
    const Long  end_t        = (N * (tid + 1)) / nthreads;
    const bool  is_last      = (tid == nthreads - 1);
    const NodeT start_anchor = (tid == 0) ? NodeT{}        : split(begin_t);
    const NodeT end_anchor   = is_last    ? NodeT{}.Next() : split(end_t);
    const Long  idx_start    = (tid == 0) ? 0 : lower_bound_window(begin_t, begin_t + M, start_anchor.mid);
    const Long  idx_end      = is_last    ? N : lower_bound_window(end_t,   end_t   + M, end_anchor.mid);

    Long count = 0;
    NodeT* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[tid];

    NodeT m0 = start_anchor;
    Long pt_idx = idx_start;
    while (pt_idx < idx_end - M) {
      const NodeT m_ = split(pt_idx);
      while (m0 != m_) {
        if constexpr (MODE == WalkMode::Write) w[count] = m0;
        ++count;
        if (m0.isAncestor(m_)) m0 = m0.DFD(static_cast<uint8_t>(m0.depth + 1));
        else                   m0 = m0.Next();
      }
      m0 = m_;
      pt_idx = lower_bound_window(pt_idx, pt_idx + M, m0.mid);
    }
    while (m0 != end_anchor) {  // tail to end_anchor / sentinel
      if constexpr (MODE == WalkMode::Write) w[count] = m0;
      ++count;
      if (m0.isAncestor(end_anchor)) m0 = m0.DFD(static_cast<uint8_t>(m0.depth + 1));
      else                           m0 = m0.Next();
    }
    return count;
  }
};

// One Long per 64-byte cache line — avoids false sharing on per-thread counters.
struct alignas(64) PaddedLong {
  Long v;
  char pad[64 - sizeof(Long)];
};
static_assert(sizeof(PaddedLong) == 64);

// CPU build from sorted Morton codes via chunked sctl-style parallel walk: each thread walks
// pt_mid[begin_t, end_t) between synthetic chunk-boundary anchors, emitting to a NUMA-local
// per-thread ScratchBuf via ChunkedWalkFunctor<Write>. After a barrier the per-thread counts
// are prefix-summed and slices copied into `tree`. Output matches `sctl::Tree`.
//
// A 2-pass variant (count, then walk again writing directly into tree) was ~25% slower at
// N=100M, M=300 — the second walk re-reads pt_mid (~10–20 MB per chunk) cache-cold.
template <class Real, Integer DIM, template <class...> class DeviceVector>
void buildTreeCpuChunked(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M) {
  using NodeMIDT = Morton<DIM>;
  const Long N = static_cast<Long>(pt_mid.size());
  if (N <= M) {  // root-only tree
    tree.resize(1);
    tree[0] = NodeMIDT{};
    return;
  }

  // Cap threads so each chunk has well over M particles (so `begin + M` stays in-bounds).
  const int max_threads = SCTL_GET_MAX_THREADS();
  const Long min_chunk = std::max<Long>(4 * M + 1, 1024);
  const int nthreads = std::clamp<int>(static_cast<int>(N / min_chunk), 1, max_threads);

  // Upper bound: ~(MAX_DEPTH+1) nodes/leaf, chunk_size/M leaves/chunk, 4x slack.
  const Long chunk_size_max = (N + nthreads - 1) / nthreads;
  const Long max_emits = 4 * chunk_size_max * (MAX_DEPTH + 1) / std::max<Long>(1, M) + 4 * MAX_DEPTH + 16;

  sctl::ScratchBuf<PaddedLong> local_sizes(nthreads);  // padded: concurrent per-thread writes
  sctl::ScratchBuf<Long> offsets(nthreads);            // written once by `single`, read-only after
  sctl::ScratchBuf<Long> zero_offsets(nthreads);       // functor reads `offsets[tid] == 0`
  for (int t = 0; t < nthreads; ++t) zero_offsets[t] = 0;

  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = SCTL_GET_THREAD_NUM();
    sctl::ScratchBuf<NodeMIDT> buf(max_emits);  // NUMA-local: first-touched on this thread's node
    const ChunkedWalkFunctor<Real, DIM, WalkMode::Write> fw{thrust::raw_pointer_cast(pt_mid.data()), N, M, nthreads, &zero_offsets[0], &buf[0]};
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

// GPU build from sorted Morton codes:
//   Phase 1 — anchors: SplitLeafFunctor over each pair (pt[i], pt[i+M]) emits a per-pair
//     anchor; the sequence is sorted by construction, so thrust::unique_copy dedupes.
//     Fused via transform_iterator to keep per-pair leaves out of global memory.
//   Phase 2 — linearize: 2-pass count + exclusive_scan + write via LinearizeWalkFunctor over
//     n+1 pairs. Pair n walks anchors[n-1] -> root.Next() so leaves between the last anchor
//     and morton-end are emitted (matches sctl::Tree exactly).
template <class Real, Integer DIM, template <class...>
class DeviceVector> void buildTreeGpu(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M) {
  using NodeMIDT = Morton<DIM>;

  const Long N = static_cast<Long>(pt_mid.size());
  if (N <= M) {  // root-only tree
    tree.resize(1);
    tree[0] = NodeMIDT{};
    return;
  }
  const Long N_pairs = N - M;

  // Phase 1: per-pair anchors + dedupe.
  DeviceVector<NodeMIDT> anchors(N_pairs);
  SplitLeafFunctor<Real, DIM> f{thrust::raw_pointer_cast(pt_mid.data()), M};
  auto in     = thrust::make_transform_iterator(thrust::counting_iterator<Long>(0),       f);
  auto in_end = thrust::make_transform_iterator(thrust::counting_iterator<Long>(N_pairs), f);
  auto new_end = thrust::unique_copy(in, in_end, anchors.begin());
  const Long n_anchors = new_end - anchors.begin();

  // Phase 2: linearize over n_anchors+1 pairs (+1 = trailing pair to root.Next()).
  const Long n_pairs = n_anchors + 1;
  DeviceVector<Long> counts(n_pairs), offsets(n_pairs);
  LinearizeWalkFunctor<Real, DIM, WalkMode::Count> fc{thrust::raw_pointer_cast(anchors.data()), n_anchors, nullptr, nullptr};
  thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(n_pairs), counts.begin(), fc);
  thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), Long(0));
  const Long total = thrust::reduce(counts.begin(), counts.end(), Long(0));

  tree.resize(total);
  LinearizeWalkFunctor<Real, DIM, WalkMode::Write> fw{
      thrust::raw_pointer_cast(anchors.data()), n_anchors,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(tree.data())};
  thrust::for_each_n(thrust::counting_iterator<Long>(0), n_pairs, fw);
}

// GPU port of buildTreeCpuChunked: single-pass chunked walk (no anchor materialization).
// 2-pass count + exclusive_scan + write via ChunkedWalkFunctor over `nthreads` chunks.
// 64-particle min_chunk floor dominates everywhere measured.
template <class Real, Integer DIM, template <class...> class DeviceVector>
void buildTreeGpuChunked(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M) {
  const Long N = static_cast<Long>(pt_mid.size());
  if (N <= M) {  // root-only tree
    tree.resize(1);
    tree[0] = Morton<DIM>{};
    return;
  }

  const Long min_chunk = std::max<Long>(4 * M + 1, 64);
  const Long nthreads  = std::clamp<Long>(N / min_chunk, 1, 65536);

  DeviceVector<Long> counts(nthreads), offsets(nthreads);
  ChunkedWalkFunctor<Real, DIM, WalkMode::Count> fc{thrust::raw_pointer_cast(pt_mid.data()), N, M, nthreads, nullptr, nullptr};
  thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(nthreads), counts.begin(), fc);
  thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), Long(0));
  const Long total = thrust::reduce(counts.begin(), counts.end(), Long(0));

  tree.resize(total);
  ChunkedWalkFunctor<Real, DIM, WalkMode::Write> fw{
      thrust::raw_pointer_cast(pt_mid.data()), N, M, nthreads,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(tree.data())};
  thrust::for_each_n(thrust::counting_iterator<Long>(0), nthreads, fw);
}

}  // namespace detail

template <class Real, Integer DIM> template <template <class...> class DeviceVector>
void GPUTree<Real, DIM>::buildTree(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M, DeviceVector<Long>* sort_scatter_index) {
  using MortonT = MortonCode<DIM>;
  constexpr bool on_device = detail::is_device_vector_v<DeviceVector<Real>>;

  const Long N = static_cast<Long>(coord.size()) / DIM;

  DeviceVector<MortonT> pt_mid(N);
  if constexpr (on_device) {
    detail::MakeMortonFunctor<Real, DIM> f{thrust::raw_pointer_cast(coord.data())};
    thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(N), pt_mid.begin(), f);
    if (sort_scatter_index) {
      sort_scatter_index->resize(N);
      thrust::sequence(sort_scatter_index->begin(), sort_scatter_index->end());
      thrust::sort_by_key(pt_mid.begin(), pt_mid.end(), sort_scatter_index->begin());
    } else {
      thrust::sort(pt_mid.begin(), pt_mid.end());
    }
  } else {
    // thrust/OMP was ~3x slower than omp_par::merge_sort on this workload.
    const Real* cp = thrust::raw_pointer_cast(coord.data());
    MortonT*    mp = thrust::raw_pointer_cast(pt_mid.data());
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < N; ++i) mp[i] = MortonT(cp + i * DIM);

    if (sort_scatter_index) {
      sort_scatter_index->resize(N);
      Long* ip = thrust::raw_pointer_cast(sort_scatter_index->data());
      struct Pair {
        MortonT key;
        Long data;
        bool operator<(const Pair& o) const { return key < o.key; }
      };
      sctl::ScratchBuf<Pair> pairs(N);  // avoids ~16*N bytes of per-call heap alloc
      auto pp = pairs.begin();
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < N; ++i) {
        pp[i].key = mp[i];
        pp[i].data = i;
      }
      sctl::omp_par::merge_sort(pairs.begin(), pairs.end());
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < N; ++i) {
        mp[i] = pp[i].key;
        ip[i] = pp[i].data;
      }
    } else {
      sctl::omp_par::merge_sort(mp, mp + N);
    }
  }

  buildTreeFromSortedMorton(tree, pt_mid, M);
}

template <class Real, Integer DIM> template <template <class...> class DeviceVector>
void GPUTree<Real, DIM>::buildTreeFromSortedMorton(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M) {
  using MortonT = MortonCode<DIM>;
  constexpr bool on_device = detail::is_device_vector_v<DeviceVector<MortonT>>;

  if constexpr (on_device) {
    // Chunked walk dominates except at very small N with fine M (N <= ~100K, M=1) where
    // pair-based wins. N*M >= 128K is the empirical crossover.
    constexpr Long kChunkedThreshold = 128 * 1024;
    const Long N = static_cast<Long>(pt_mid.size());
    if (N * M >= kChunkedThreshold) {
      detail::buildTreeGpuChunked<Real, DIM>(tree, pt_mid, M);
    } else {
      detail::buildTreeGpu<Real, DIM>(tree, pt_mid, M);
    }
  } else {
    detail::buildTreeCpuChunked<Real, DIM>(tree, pt_mid, M);
  }
}

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
