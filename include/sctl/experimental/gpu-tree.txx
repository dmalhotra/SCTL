// Template implementation of GPUTree from gpu-tree.hpp + internal `detail::` helpers.

#ifndef _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
#define _SCTL_EXPERIMENTAL_GPU_TREE_TXX_

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "sctl/experimental/gpu-tree.hpp"
#include "sctl/comm.hpp"
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

// Sort v[0, n): thrust radix sort on device, omp_par parallel merge sort on host (thrust's
// host backend is single-threaded). Mirrors buildTree's host/device split so the CPU path
// of the distributed build stays parallel.
template <class Vec> void local_sort(Vec& v, Long n) {
  if constexpr (is_device_vector_v<Vec>) {
    thrust::sort(v.begin(), v.begin() + n);
  } else {
    auto* p = thrust::raw_pointer_cast(v.data());
    sctl::omp_par::merge_sort(p, p + n);
  }
}

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
  Morton<DIM> start_bnd;  // rank's lower boundary anchor (ROOT on the first rank)
  Morton<DIM> end_bnd;    // rank's upper boundary, exclusive (root.Next() on the last rank)

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
    const NodeT start_anchor = (tid == 0) ? start_bnd : split(begin_t);
    const NodeT end_anchor   = (is_last)  ? end_bnd   : split(end_t);
    const Long  idx_start    = (tid == 0) ? 0 : lower_bound_window(begin_t, begin_t + M, start_anchor.mid);
    const Long  idx_end      = (is_last)  ? N : lower_bound_window(end_t,   end_t   + M, end_anchor.mid);

    Long count = 0;
    NodeT* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[tid];

    NodeT m0 = start_anchor;
    Long pt_idx = idx_start;
    while (pt_idx < idx_end - M) {
      const NodeT m_ = split(pt_idx);
      if (m_ == m0) {  // > M coincident codes: their MAX_DEPTH box cannot split; skip past the run
        pt_idx = lower_bound_window(pt_idx, idx_end, m0.Next().mid);
        continue;
      }
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
void buildTreeCpuChunked(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M, Long N_owned = -1, Long base = 0, Morton<DIM> start_bnd = Morton<DIM>{}, Morton<DIM> end_bnd = Morton<DIM>{}.Next()) {
  using NodeMIDT = Morton<DIM>;
  const Long N = (N_owned < 0 ? static_cast<Long>(pt_mid.size()) : N_owned);  // walk window is pt_mid[base, base+N) (+M halo slack beyond)
  if (N <= M && start_bnd == NodeMIDT{} && end_bnd == NodeMIDT{}.Next()) {  // root-only tree
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
void buildTreeGpuChunked(DeviceVector<Morton<DIM>>& tree, const DeviceVector<MortonCode<DIM>>& pt_mid, Long M, Long N_owned = -1, Long base = 0, Morton<DIM> start_bnd = Morton<DIM>{}, Morton<DIM> end_bnd = Morton<DIM>{}.Next()) {
  const Long N = (N_owned < 0 ? static_cast<Long>(pt_mid.size()) : N_owned);  // walk window is pt_mid[base, base+N) (+M halo slack beyond)
  if (N <= M && start_bnd == Morton<DIM>{} && end_bnd == Morton<DIM>{}.Next()) {  // root-only tree
    tree.resize(1);
    tree[0] = Morton<DIM>{};
    return;
  }

  const Long min_chunk = std::max<Long>(4 * M + 1, 64);
  const Long nthreads  = std::clamp<Long>(N / min_chunk, 1, 65536);

  DeviceVector<Long> counts(nthreads), offsets(nthreads);
  ChunkedWalkFunctor<Real, DIM, WalkMode::Count> fc{thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads, nullptr, nullptr, start_bnd, end_bnd};
  thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(nthreads), counts.begin(), fc);
  thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), Long(0));
  const Long total = thrust::reduce(counts.begin(), counts.end(), Long(0));

  tree.resize(total);
  ChunkedWalkFunctor<Real, DIM, WalkMode::Write> fw{
      thrust::raw_pointer_cast(pt_mid.data()) + base, N, M, nthreads,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(tree.data()), start_bnd, end_bnd};
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


#ifdef SCTL_HAVE_MPI
namespace detail {

// Splitter selection for the distributed sort: port of Comm::DetermineSplitter's iterative
// exact-rank histogramming (see comm.txx for the algorithm commentary). Host-side scalar
// logic; the device only answers batched lower_bound rank queries against the locally
// sorted codes. Returns all np-1 cuts (state is replicated on every rank).
template <Integer DIM, template <class...> class DeviceVector>
std::vector<MortonCode<DIM>> determineSplitters(const DeviceVector<MortonCode<DIM>>& pt, Long totSize, const Comm& comm) {
  using MortonT = MortonCode<DIM>;
  const MPI_Comm& mpi_comm = comm.GetMPI_Comm();
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long ns = np - 1, nloc = static_cast<Long>(pt.size());
  std::vector<MortonT> best(ns);
  if (!ns) return best;

  const auto local_ranks = [&pt](const std::vector<MortonT>& q) {  // batched lower_bound against the sorted local codes
    DeviceVector<MortonT> q_d(q.begin(), q.end());
    DeviceVector<Long> r_d(q.size());
    thrust::lower_bound(pt.begin(), pt.end(), q_d.begin(), q_d.end(), r_d.begin());
    std::vector<Long> r(q.size());
    thrust::copy(r_d.begin(), r_d.end(), r.begin());
    return r;
  };

  MortonT gmin{}, gmax{};
  { // global min/max code (bracket anchors), skipping empty ranks
    const long long nloc_ll = nloc;
    const MortonT l0 = nloc ? MortonT(pt[0]) : MortonT{};
    const MortonT l1 = nloc ? MortonT(pt[nloc - 1]) : MortonT{};
    std::vector<long long> cnt(np);
    std::vector<MortonT> firsts(np), lasts(np);
    MPI_Allgather(&nloc_ll, 1, MPI_LONG_LONG, cnt.data(), 1, MPI_LONG_LONG, mpi_comm);
    MPI_Allgather(&l0, sizeof(MortonT), MPI_BYTE, firsts.data(), sizeof(MortonT), MPI_BYTE, mpi_comm);
    MPI_Allgather(&l1, sizeof(MortonT), MPI_BYTE, lasts.data(), sizeof(MortonT), MPI_BYTE, mpi_comm);
    bool found = false;
    for (Long i = 0; i < np; i++) {
      if (!cnt[i]) continue;
      if (!found) { gmin = firsts[i]; gmax = lasts[i]; found = true; }
      else {
        if (firsts[i] < gmin) gmin = firsts[i];
        if (gmax < lasts[i]) gmax = lasts[i];
      }
    }
  }

  const Long tol = std::max<Long>(1, Long(0.02 * double(totSize) / double(np)));  // 2% load-balance tolerance
  const Long budget = 32;                                                         // per-process probes/round
  constexpr int MAXIT = 50;

  uint64_t rng = uint64_t(rank) * 0x9e3779b97f4a7c15ULL + 0x123456789abcdefULL;  // splitmix64; only affects probe choice
  const auto next = [&rng]() {
    uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };

  std::vector<Long> targ(ns), rlo(ns), rhi(ns);  // per-cut bracket state (values + global ranks)
  std::vector<MortonT> blo(ns), bhi(ns);
  std::vector<char> done(ns), use_rand(ns);
  for (Long c = 0; c < ns; c++) {
    targ[c] = (c + 1) * totSize / np;
    blo[c] = gmin; bhi[c] = gmax;
    rlo[c] = 0; rhi[c] = totSize;
    done[c] = 0; use_rand[c] = 0;
    best[c] = gmax;
  }

  for (int it = 0; it < MAXIT; it++) {
    Long nun = 0;
    for (Long c = 0; c < ns; c++) if (!done[c]) nun++;

    std::vector<MortonT> myc;
    { // probe: contribute to <= budget active cuts (random subset), interpolated or random fall-back
      std::vector<Long> sel;
      for (Long c = 0; c < ns; c++) {
        if (done[c]) continue;
        if (nun > budget && (next() % (uint64_t)nun) >= (uint64_t)budget) continue;
        sel.push_back(c);
      }
      std::vector<MortonT> q(2 * sel.size());
      for (std::size_t i = 0; i < sel.size(); i++) { q[2 * i] = blo[sel[i]]; q[2 * i + 1] = bhi[sel[i]]; }
      const std::vector<Long> ab = sel.empty() ? std::vector<Long>{} : local_ranks(q);
      for (std::size_t i = 0; i < sel.size(); i++) {
        const Long c = sel[i], a = ab[2 * i], b = ab[2 * i + 1];
        if (b <= a) continue;
        Long idx;
        if (use_rand[c]) idx = a + Long(next() % uint64_t(b - a));
        else {
          double f = double(targ[c] - rlo[c]) / double(std::max<Long>(1, rhi[c] - rlo[c]));
          f = std::min(1.0, std::max(0.0, f));
          idx = a + Long(f * double(b - a));
        }
        idx = std::min(b - 1, std::max(a, idx));
        myc.push_back(MortonT(pt[idx]));  // per-element D2H, <= budget per round
      }
    }

    Long S = 0;
    std::vector<MortonT> comb;
    { // assemble sorted candidate set: gathered probes ++ anchors ++ active brackets
      const int mloc = int(myc.size());
      std::vector<int> cntb(np), dspb(np);
      MPI_Allgather(&mloc, 1, MPI_INT, cntb.data(), 1, MPI_INT, mpi_comm);
      int tot = 0;
      for (Long r = 0; r < np; r++) { dspb[r] = tot * int(sizeof(MortonT)); tot += cntb[r]; cntb[r] *= int(sizeof(MortonT)); }
      comb.resize(tot + 2 * ns + 2);
      MPI_Allgatherv(mloc ? myc.data() : nullptr, mloc * int(sizeof(MortonT)), MPI_BYTE, comb.data(), cntb.data(), dspb.data(), MPI_BYTE, mpi_comm);
      S = tot;
      comb[S++] = gmin;
      comb[S++] = gmax;
      for (Long c = 0; c < ns; c++) if (!done[c]) { comb[S++] = blo[c]; comb[S++] = bhi[c]; }
      std::sort(comb.begin(), comb.begin() + S);
      S = std::unique(comb.begin(), comb.begin() + S, [](const MortonT& x, const MortonT& y) { return !(x < y) && !(y < x); }) - comb.begin();
      comb.resize(S);
    }

    std::vector<Long> gr(S);
    { // exact global rank of every candidate
      const std::vector<Long> lr = local_ranks(comb);
      MPI_Allreduce(lr.data(), gr.data(), int(S), MPI_LONG_LONG, MPI_SUM, mpi_comm);
    }

    bool anyactive = false;
    for (Long c = 0; c < ns; c++) {  // assign nearest candidate; refine bracket or freeze
      if (done[c]) continue;
      const Long t = targ[c];
      const Long up = std::lower_bound(gr.begin(), gr.end(), t) - gr.begin(), lo = up - 1;
      const Long errlo = (lo >= 0) ? t - gr[lo] : t, errup = (up < S) ? gr[up] - t : (totSize - t);
      best[c] = (errlo <= errup) ? comb[std::max<Long>(0, lo)] : comb[std::min<Long>(S - 1, up)];
      if (lo < 0 || up >= S || std::min(errlo, errup) <= tol || (gr[up] - gr[lo]) <= tol) { done[c] = 1; continue; }
      const Long oldw = rhi[c] - rlo[c], neww = gr[up] - gr[lo];
      if (neww >= oldw) { done[c] = 1; continue; }  // un-splittable (duplicate mass)
      use_rand[c] = (neww > oldw / 2);
      blo[c] = comb[lo]; rlo[c] = gr[lo];
      bhi[c] = comb[up]; rhi[c] = gr[up];
      anyactive = true;
    }
    if (!anyactive) break;
  }
  return best;
}

// --- 2:1 balance refinement helpers ---------------------------------------------------
// Closure rule (leaf form, equivalent to Tree::UpdateRefinement's touching-parent-neighbor
// rule on a complete tree): every same-depth neighbor octant of an internal node must exist
// as a node. Unsatisfied neighbor octants ("requirements") are routed to their owning rank
// and inserted as leaves; iterate to global fixpoint. A requirement never straddles a rank
// boundary: boundary anchors remain nodes in every refinement, so a straddling octant is an
// ancestor of one and is always already satisfied.

template <Integer DIM> struct NodeEqPred {
  SCTL_GPU_HD bool operator()(const Morton<DIM>& a, const Morton<DIM>& b) const { return !(a < b) && !(b < a); }
};

template <Integer DIM> struct InvalidDepthPred {
  SCTL_GPU_HD bool operator()(const Morton<DIM>& m) const { return m.depth == Morton<DIM>::INVALID_DEPTH; }
};

struct NonZeroCharPred {
  SCTL_GPU_HD bool operator()(char c) const { return c != 0; }
};

// Slot j = node i * 3^DIM + k: neighbor k of internal node i (INVALID for leaves, clipped
// neighbors, and the self slot).
template <Integer DIM> struct BalanceReqFunctor {
  static constexpr Integer K = sctl::pow<DIM, Integer>(3);
  const Morton<DIM>* tree;
  Long Nn;
  Morton<DIM> next_first;  // first node of the right neighbor rank (walk-order successor of tree[Nn-1])

  SCTL_GPU_HD Morton<DIM> operator()(Long j) const {
    const Long i = j / K;
    const Integer k = Integer(j % K);
    Morton<DIM> inv;
    inv.depth = Morton<DIM>::INVALID_DEPTH;
    if (k == (K - 1) / 2) return inv;  // self
    const Morton<DIM>& X = tree[i];
    const Morton<DIM>& nxt = (i + 1 < Nn) ? tree[i + 1] : next_first;
    if (!X.isAncestor(nxt)) return inv;  // leaf
    const Morton<DIM> q = X.NbrList(X.depth, sctl::Periodicity::NONE)[k];
    if (q.depth == Morton<DIM>::INVALID_DEPTH) return inv;
    { // emit only unsatisfied requirements (keeps the compacted set tiny)
      Long lo = 0, hi = Nn;
      while (lo < hi) {
        const Long m = lo + (hi - lo) / 2;
        if (tree[m] < q) lo = m + 1;
        else             hi = m;
      }
      if (lo < Nn && !(tree[lo].mid < q.mid) && !(q.mid < tree[lo].mid)) return inv;
    }
    return q;
  }
};

template <Integer DIM> struct LeafFlagFunctor {
  const Morton<DIM>* tree;
  Long Nn;
  Morton<DIM> next_first;
  SCTL_GPU_HD char operator()(Long i) const {
    const Morton<DIM>& nxt = (i + 1 < Nn) ? tree[i + 1] : next_first;
    return tree[i].isAncestor(nxt) ? 0 : 1;
  }
};

// Requirement q is satisfied iff a node at q's corner with depth >= q.depth exists (in a
// complete tree that is exactly "q exists"). Lex lower_bound: same-mid hit implies depth >=.
template <Integer DIM> struct SatisfiedPred {
  const Morton<DIM>* tree;
  Long Nn;
  SCTL_GPU_HD bool operator()(const Morton<DIM>& q) const {
    Long lo = 0, hi = Nn;
    while (lo < hi) {
      const Long m = lo + (hi - lo) / 2;
      if (tree[m] < q) lo = m + 1;
      else             hi = m;
    }
    if (lo == Nn) return false;
    return !(tree[lo].mid < q.mid) && !(q.mid < tree[lo].mid);
  }
};

// Linearize: drop duplicates and any octant with a finer octant inside it (in lex order an
// ancestor's immediate successor is always one of its descendants).
template <Integer DIM> struct KeepFinestFunctor {
  const Morton<DIM>* x;
  Long n;
  SCTL_GPU_HD char operator()(Long i) const {
    if (i + 1 < n) {
      const Morton<DIM>& a = x[i], &b = x[i + 1];
      if (!(a < b) && !(b < a)) return 0;
      if (a.isAncestor(b)) return 0;
    }
    return 1;
  }
};

// LinearizeWalkFunctor variant for a rank's slice: pair 0 starts at start_node (emitted);
// the trailing pair walks to end_target EXCLUSIVE (the next rank's first node, or the
// morton-end sentinel on the last rank).
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
    if (i == 0) {
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

// Rebuild a rank's complete preorder slice from a sorted, linearized leaf/anchor range.
template <Integer DIM, template <class...> class DeviceVector>
void rebuildFromAnchors(DeviceVector<Morton<DIM>>& tree, const Morton<DIM>* anchors_ptr, Long n, const Morton<DIM>& start_node, const Morton<DIM>& end_target) {
  const Long n_pairs = n + 1;
  DeviceVector<Long> counts(n_pairs), offsets(n_pairs);
  const AnchorWalkFunctor<DIM, WalkMode::Count> fc{anchors_ptr, n, start_node, end_target, nullptr, nullptr};
  thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(n_pairs), counts.begin(), fc);
  thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), Long(0));
  const Long total = thrust::reduce(counts.begin(), counts.end(), Long(0));

  tree.resize(total);
  const AnchorWalkFunctor<DIM, WalkMode::Write> fw{
      anchors_ptr, n, start_node, end_target,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(tree.data())};
  thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(n_pairs), counts.begin(), fw);  // transform (not for_each_n) keeps host containers on the host backend
}

template <Integer DIM, template <class...> class DeviceVector>
void balanceTreeDist(DeviceVector<Morton<DIM>>& tree, const Comm& comm) {
  using NodeT = Morton<DIM>;
  const MPI_Comm& mpi_comm = comm.GetMPI_Comm();
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  constexpr Integer K = sctl::pow<DIM, Integer>(3);

  const Long Nn0 = static_cast<Long>(tree.size());
  std::vector<NodeT> A(np);  // fixed rank boundaries: first node of each rank (nodes in every refinement)
  {
    NodeT a0{}; if (Nn0) a0 = NodeT(tree[0]);
    MPI_Allgather(&a0, sizeof(NodeT), MPI_BYTE, A.data(), sizeof(NodeT), MPI_BYTE, mpi_comm);
    std::vector<long long> nn(np); const long long my_nn = Nn0;
    MPI_Allgather(&my_nn, 1, MPI_LONG_LONG, nn.data(), 1, MPI_LONG_LONG, mpi_comm);
    for (Long r = np - 1; r >= 0; r--) if (!nn[r]) A[r] = (r + 1 < np) ? A[r + 1] : NodeT{}.Next();  // empty rank: zero-width interval
  }
  const NodeT end_target = (rank + 1 < np) ? A[rank + 1] : NodeT{}.Next();
  const NodeT next_first = end_target;  // walk-order successor of the local last node = next non-empty rank's first node (fixed across rounds)
  DeviceVector<NodeT> Akey_d;
  { // routing keys (A[r].mid, depth 0): lex lower_bound gives each rank's segment start
    std::vector<NodeT> keys(A);
    for (NodeT& k : keys) k.depth = 0;
    Akey_d = DeviceVector<NodeT>(keys.begin(), keys.end());
  }

  constexpr int MAXROUNDS = 4 * MAX_DEPTH;
  for (int round = 0; round < MAXROUNDS; round++) {
    const Long Nn = static_cast<Long>(tree.size());

    Long nreq = 0;
    DeviceVector<NodeT> req;
    { // unsatisfied requirements from internal nodes (chunked: the K-slot expansion is transient)
      const BalanceReqFunctor<DIM> fg{thrust::raw_pointer_cast(tree.data()), Nn, next_first};
      const Long chunk = std::min<Long>(Nn, 4000000);
      DeviceVector<NodeT> buf(chunk * K);
      for (Long c0 = 0; c0 < Nn; c0 += chunk) {
        const Long nc = std::min<Long>(chunk, Nn - c0);
        thrust::transform(thrust::counting_iterator<Long>(c0 * K), thrust::counting_iterator<Long>((c0 + nc) * K), buf.begin(), fg);
        const Long nkeep = thrust::remove_if(buf.begin(), buf.begin() + nc * K, InvalidDepthPred<DIM>{}) - buf.begin();
        req.resize(nreq + nkeep);
        thrust::copy(buf.begin(), buf.begin() + nkeep, req.begin() + nreq);
        nreq += nkeep;
      }
      local_sort(req, nreq);
      nreq = thrust::unique(req.begin(), req.begin() + nreq, NodeEqPred<DIM>{}) - req.begin();
    }

    Long Nrecv = 0;
    std::vector<int> scnt(np), sdsp(np), rcnt(np), rdsp(np);
    { // send counts: reqs are mid-sorted, so per-owner segments are contiguous
      DeviceVector<Long> pos_d(np);
      thrust::lower_bound(req.begin(), req.begin() + nreq, Akey_d.begin(), Akey_d.end(), pos_d.begin());
      std::vector<Long> pos(np + 1);
      for (Long r = 0; r < np; r++) pos[r] = pos_d[r];
      pos[np] = nreq;
      for (Long r = 0; r < np; r++) {
        scnt[r] = int((pos[r + 1] - pos[r]) * (Long)sizeof(NodeT));
        sdsp[r] = int(pos[r] * (Long)sizeof(NodeT));
      }
      MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, mpi_comm);
      for (Long r = 0; r < np; r++) { rdsp[r] = int(Nrecv * (Long)sizeof(NodeT)); Nrecv += rcnt[r] / (Long)sizeof(NodeT); }
    }

    Long nsurv = 0;
    DeviceVector<NodeT> rreq(Nrecv);
    { // route to owners; re-filter against the owner's slice
      MPI_Alltoallv(thrust::raw_pointer_cast(req.data()), scnt.data(), sdsp.data(), MPI_BYTE,
                    thrust::raw_pointer_cast(rreq.data()), rcnt.data(), rdsp.data(), MPI_BYTE, mpi_comm);
      local_sort(rreq, (Long)rreq.size());
      auto rr = thrust::unique(rreq.begin(), rreq.end(), NodeEqPred<DIM>{});
      rr = thrust::remove_if(rreq.begin(), rr, SatisfiedPred<DIM>{thrust::raw_pointer_cast(tree.data()), Nn});
      nsurv = rr - rreq.begin();
    }

    long long nsurv_ll = nsurv, glob_ll = 0;
    MPI_Allreduce(&nsurv_ll, &glob_ll, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);
    if (!glob_ll) break;      // global fixpoint
    if (!nsurv) continue;     // only remote ranks changed this round

    { // new anchor set = linearize(current leaves ++ surviving requirements); rebuild slice
      DeviceVector<char> lf(Nn);
      thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), lf.begin(), LeafFlagFunctor<DIM>{thrust::raw_pointer_cast(tree.data()), Nn, next_first});
      const Long nleaf = thrust::count(lf.begin(), lf.end(), (char)1);
      DeviceVector<NodeT> anch(nleaf + nsurv);
      thrust::copy_if(tree.begin(), tree.end(), lf.begin(), anch.begin(), NonZeroCharPred{});
      thrust::copy(rreq.begin(), rreq.begin() + nsurv, anch.begin() + nleaf);
      local_sort(anch, (Long)anch.size());

      DeviceVector<char> keep(anch.size());
      thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Long(anch.size())), keep.begin(), KeepFinestFunctor<DIM>{thrust::raw_pointer_cast(anch.data()), Long(anch.size())});
      const Long nkeep = thrust::count(keep.begin(), keep.end(), (char)1);
      DeviceVector<NodeT> anch2(nkeep);
      thrust::copy_if(anch.begin(), anch.end(), keep.begin(), anch2.begin(), NonZeroCharPred{});

      rebuildFromAnchors<DIM>(tree, thrust::raw_pointer_cast(anch2.data()), Long(anch2.size()), A[rank], end_target);
    }
  }
}

// --- ghost node placeholders ----------------------------------------------------------
// Mirror of Tree::UpdateRefinement's halo scheme, with the rank's true first node as the
// partition key: an owned node is sent to every rank whose owned interval intersects the
// node's coarse neighborhood (NbrList at depth d0-halo_size, self entry included -- the
// self entry is what guarantees the boundary-ancestor chain is always ghosted). Received
// ghosts are spliced around the owned slice with complete-tree placeholder fill, giving a
// full-domain complete linear tree that is coarse outside the halo.

template <Integer DIM> struct GhostPair {
  Long p;
  Morton<DIM> m;
  SCTL_GPU_HD bool operator<(const GhostPair& o) const { return p < o.p || (!(o.p < p) && m < o.m); }
};

template <Integer DIM> struct GhostPairEqPred {
  SCTL_GPU_HD bool operator()(const GhostPair<DIM>& a, const GhostPair<DIM>& b) const { return !(a < b) && !(b < a); }
};

template <Integer DIM> struct GhostPairToMid {
  SCTL_GPU_HD Morton<DIM> operator()(const GhostPair<DIM>& gp) const { return gp.m; }
};

template <Integer DIM, WalkMode MODE> struct GhostSendFunctor {
  static constexpr Integer K = sctl::pow<DIM, Integer>(3);
  const Morton<DIM>* tree;
  const Morton<DIM>* A;  // rank boundaries (first node of each rank), lex order
  Long np, rank;
  Integer halo;
  const Long* offsets;      // Write only
  GhostPair<DIM>* out;      // Write only

  SCTL_GPU_HD Long lb(const Morton<DIM>& key) const {
    Long lo = 0, hi = np;
    while (lo < hi) {
      const Long m = lo + (hi - lo) / 2;
      if (A[m] < key) lo = m + 1;
      else            hi = m;
    }
    return lo;
  }

  SCTL_GPU_HD Long operator()(Long i) const {
    const Morton<DIM>& X = tree[i];
    const Integer lvl = (Integer(X.depth) > halo ? Integer(X.depth) - halo : 0);
    const auto nl = X.NbrList(uint8_t(lvl), sctl::Periodicity::NONE);
    Long count = 0;
    GhostPair<DIM>* w = nullptr;
    if constexpr (MODE == WalkMode::Write) w = out + offsets[i];
    for (Integer k = 0; k < K; k++) {
      const Morton<DIM>& m = nl[k];
      if (m.depth == Morton<DIM>::INVALID_DEPTH) continue;
      Long p0 = lb(m.DFD()) - 1;
      if (p0 < 0) p0 = 0;
      const Long p1 = lb(m.Next());
      for (Long p = p0; p < p1; p++) {
        if (p == rank) continue;
        if constexpr (MODE == WalkMode::Write) w[count] = GhostPair<DIM>{p, X};
        ++count;
      }
    }
    return count;
  }
};

// Splice ghost placeholders into `tree`; outputs the [begin, end) index range of the owned
// nodes within the updated list.
template <Integer DIM, template <class...> class DeviceVector>
void addGhostNodes(DeviceVector<Morton<DIM>>& tree, const Comm& comm, Integer halo_size, Long& owned_begin, Long& owned_end) {
  using NodeT = Morton<DIM>;
  const MPI_Comm& mpi_comm = comm.GetMPI_Comm();
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long Nn = static_cast<Long>(tree.size());
  owned_begin = 0; owned_end = Nn;
  if (np == 1) return;

  DeviceVector<NodeT> A_d;
  std::vector<NodeT> A(np);
  { // rank boundaries; empty ranks inherit the next non-empty rank's first node (zero-width interval)
    NodeT a0{}; if (Nn) a0 = NodeT(tree[0]);
    MPI_Allgather(&a0, sizeof(NodeT), MPI_BYTE, A.data(), sizeof(NodeT), MPI_BYTE, mpi_comm);
    std::vector<long long> nn(np); const long long my_nn = Nn;
    MPI_Allgather(&my_nn, 1, MPI_LONG_LONG, nn.data(), 1, MPI_LONG_LONG, mpi_comm);
    for (Long r = np - 1; r >= 0; r--) if (!nn[r]) A[r] = (r + 1 < np) ? A[r + 1] : NodeT{}.Next();
    A_d = DeviceVector<NodeT>(A.begin(), A.end());
  }

  Long npairs = 0;
  DeviceVector<GhostPair<DIM>> pairs;
  { // (dest rank, node) pairs, deduped
    DeviceVector<Long> counts(Nn), offsets(Nn);
    const GhostSendFunctor<DIM, WalkMode::Count> fc{thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(A_d.data()), np, rank, halo_size, nullptr, nullptr};
    thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), counts.begin(), fc);
    thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), Long(0));
    pairs.resize(thrust::reduce(counts.begin(), counts.end(), Long(0)));
    const GhostSendFunctor<DIM, WalkMode::Write> fw{thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(A_d.data()), np, rank, halo_size,
                                                    thrust::raw_pointer_cast(offsets.data()), thrust::raw_pointer_cast(pairs.data())};
    thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nn), counts.begin(), fw);
    local_sort(pairs, (Long)pairs.size());
    npairs = thrust::unique(pairs.begin(), pairs.end(), GhostPairEqPred<DIM>{}) - pairs.begin();
  }

  Long Nrecv = 0;
  std::vector<int> scnt(np), sdsp(np), rcnt(np), rdsp(np);
  DeviceVector<NodeT> send_mid(npairs);
  { // per-destination segments and counts
    std::vector<GhostPair<DIM>> keys(np);
    for (Long r = 0; r < np; r++) keys[r] = GhostPair<DIM>{r, NodeT{}};
    const DeviceVector<GhostPair<DIM>> keys_d(keys.begin(), keys.end());
    DeviceVector<Long> pos_d(np);
    thrust::lower_bound(pairs.begin(), pairs.begin() + npairs, keys_d.begin(), keys_d.end(), pos_d.begin());
    std::vector<Long> pos(np + 1);
    for (Long r = 0; r < np; r++) pos[r] = pos_d[r];
    pos[np] = npairs;
    for (Long r = 0; r < np; r++) {
      scnt[r] = int((pos[r + 1] - pos[r]) * (Long)sizeof(NodeT));
      sdsp[r] = int(pos[r] * (Long)sizeof(NodeT));
    }
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, mpi_comm);
    for (Long r = 0; r < np; r++) { rdsp[r] = int(Nrecv * (Long)sizeof(NodeT)); Nrecv += rcnt[r] / (Long)sizeof(NodeT); }
    thrust::transform(pairs.begin(), pairs.begin() + npairs, send_mid.begin(), GhostPairToMid<DIM>{});
  }

  DeviceVector<NodeT> ghost(Nrecv);
  MPI_Alltoallv(thrust::raw_pointer_cast(send_mid.data()), scnt.data(), sdsp.data(), MPI_BYTE,
                thrust::raw_pointer_cast(ghost.data()), rcnt.data(), rdsp.data(), MPI_BYTE, mpi_comm);
  // sorted: each source's segment is sorted and source owned-intervals are ordered
  const Long Nsplit = thrust::lower_bound(ghost.begin(), ghost.end(), A[rank]) - ghost.begin();

  DeviceVector<NodeT> left, right;
  if (rank > 0) rebuildFromAnchors<DIM>(left, thrust::raw_pointer_cast(ghost.data()), Nsplit, NodeT{}, A[rank]);
  if (rank + 1 < np) rebuildFromAnchors<DIM>(right, thrust::raw_pointer_cast(ghost.data()) + Nsplit, Nrecv - Nsplit, A[rank + 1], NodeT{}.Next());

  const Long L = static_cast<Long>(left.size()), R = static_cast<Long>(right.size());
  DeviceVector<NodeT> merged(L + Nn + R);
  thrust::copy(left.begin(), left.end(), merged.begin());
  thrust::copy(tree.begin(), tree.end(), merged.begin() + L);
  thrust::copy(right.begin(), right.end(), merged.begin() + L + Nn);
  tree.swap(merged);
  owned_begin = L; owned_end = L + Nn;
}

}  // namespace detail

// Distributed build: device-buffer sample sort (local radix sort -> iterative exact-rank
// splitters -> one Alltoallv straight out of the sorted device array -> radix re-sort)
// followed by a two-sided M-code halo and boundary anchors from the pair straddling each
// rank boundary (allgathered mins, as in Tree::UpdateRefinement). M is clamped to the
// smallest per-rank count, so ranks with as little as one code work. Concatenated output
// over ranks matches the single-rank buildTree exactly.
template <class Real, Integer DIM> template <template <class...> class DeviceVector>
void GPUTree<Real, DIM>::buildTreeDist(DeviceVector<Morton<DIM>>& tree, const DeviceVector<Real>& coord, Long M, const Comm& comm, bool balance21, Integer halo_size, Long* owned_range) {
  using MortonT = MortonCode<DIM>;
  const MPI_Comm& mpi_comm = comm.GetMPI_Comm();
  const Long rank = comm.Rank();
  const Long np = comm.Size();

  const Long Nloc = static_cast<Long>(coord.size()) / DIM;
  if (np == 1) {
    buildTree(tree, coord, M);
    if (balance21) detail::balanceTreeDist<DIM>(tree, comm);
    if (owned_range) { owned_range[0] = 0; owned_range[1] = Long(tree.size()); }
    return;
  }

  // Env-gated per-stage profiler (sync + barrier so each delta is the true stage wall time).
  const bool gtprof = (getenv("GTPROF") != nullptr);
  double t_last = 0;
  const auto mark = [&](const char* name) {
    if (!gtprof) return;
    cudaDeviceSynchronize();
    comm.Barrier();
    const double t = MPI_Wtime();
    if (rank == 0 && name) fprintf(stderr, "  %-24s %8.2f ms\n", name, (t - t_last) * 1e3);
    t_last = t;
  };
  mark(nullptr);

  DeviceVector<MortonT> pt(Nloc);
  { // Encode + local sort (device radix).
    detail::MakeMortonFunctor<Real, DIM> enc{thrust::raw_pointer_cast(coord.data())};
    thrust::transform(thrust::counting_iterator<Long>(0), thrust::counting_iterator<Long>(Nloc), pt.begin(), enc);
    detail::local_sort(pt, (Long)pt.size());
  }

  long long nloc_ll = Nloc, nglob_ll = 0;
  MPI_Allreduce(&nloc_ll, &nglob_ll, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);
  if (nglob_ll <= (long long)M) {  // root-only global tree, held by rank 0
    tree.resize(rank == 0 ? 1 : 0);
    if (rank == 0) tree[0] = Morton<DIM>{};
    return;
  }

  // Distributed sort in two passes (mirrors Comm::SampleSort). Pass 1: value-based splitters
  // make the data globally sorted across ranks (rank r's block < rank r+1's), but possibly
  // imbalanced on coincident runs. Pass 2: an index rebalance -- now valid because the data
  // is globally sorted -- evens the per-rank counts, splitting coincident runs by index so no
  // rank is starved. Only after (2) does M <- global_min behave (shrinking M only for tiny N).
  const Long Nglob = Long(nglob_ll);

  Long Nmid = 0;
  DeviceVector<MortonT> mid;
  { // Pass 1: value splitters -> globally-sorted (imbalanced) block `mid`
    const std::vector<MortonT> spl_h = detail::determineSplitters<DIM>(pt, Nglob, comm);
    DeviceVector<MortonT> spl_d(spl_h.begin(), spl_h.end());
    DeviceVector<Long> pos_d(np - 1);
    thrust::lower_bound(pt.begin(), pt.end(), spl_d.begin(), spl_d.end(), pos_d.begin());
    std::vector<Long> pos(np + 1);
    pos[0] = 0; pos[np] = Nloc;
    for (Long r = 0; r + 1 < np; r++) pos[r + 1] = pos_d[r];
    std::vector<int> scnt(np), sdsp(np), rcnt(np), rdsp(np);
    for (Long r = 0; r < np; r++) { scnt[r] = int((pos[r+1]-pos[r])*(Long)sizeof(MortonT)); sdsp[r] = int(pos[r]*(Long)sizeof(MortonT)); }
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, mpi_comm);
    for (Long r = 0; r < np; r++) { rdsp[r] = int(Nmid*(Long)sizeof(MortonT)); Nmid += rcnt[r]/(Long)sizeof(MortonT); }
    mid.resize(Nmid);
    MPI_Alltoallv(thrust::raw_pointer_cast(pt.data()), scnt.data(), sdsp.data(), MPI_BYTE,
                  thrust::raw_pointer_cast(mid.data()), rcnt.data(), rdsp.data(), MPI_BYTE, mpi_comm);
    detail::local_sort(mid, (Long)mid.size());  // np sorted segments -> one sorted block
  }
  mark("encode+sort+splitters");

  // Pass 2: index rebalance -> rank r gets global indices [r*Nglob/np, (r+1)*Nglob/np).
  // Skipped when pass 1 already balanced (value splitters only starve a rank on heavy
  // coincident runs), keeping the common case a single redistribution.
  long long minmid_ll = Nmid, minmid = 0;
  MPI_Allreduce(&minmid_ll, &minmid, 1, MPI_LONG_LONG, MPI_MIN, mpi_comm);

  Long Nrecv = 0;
  DeviceVector<MortonT> owned;
  if (minmid >= M) {  // balanced: own pass-1 result directly
    Nrecv = Nmid;
    owned = std::move(mid);
  } else {            // rebalance by index (mid is globally sorted, so this splits coincident runs)
    long long nmid_ll = Nmid, goff_ll = 0;
    MPI_Exscan(&nmid_ll, &goff_ll, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);
    if (rank == 0) goff_ll = 0;  // MPI leaves the root's recvbuf undefined
    const Long goff = Long(goff_ll);
    std::vector<int> scnt(np), sdsp(np), rcnt(np), rdsp(np);
    {
      std::vector<Long> pos(np + 1);
      for (Long r = 0; r <= np; r++) { const Long gcut = r * Nglob / np; pos[r] = std::min<Long>(Nmid, std::max<Long>(0, gcut - goff)); }
      for (Long r = 0; r < np; r++) { scnt[r] = int((pos[r+1]-pos[r])*(Long)sizeof(MortonT)); sdsp[r] = int(pos[r]*(Long)sizeof(MortonT)); }
    }
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, mpi_comm);
    for (Long r = 0; r < np; r++) { rdsp[r] = int(Nrecv * (Long)sizeof(MortonT)); Nrecv += rcnt[r] / (Long)sizeof(MortonT); }
    owned.resize(Nrecv);
    MPI_Alltoallv(thrust::raw_pointer_cast(mid.data()), scnt.data(), sdsp.data(), MPI_BYTE,
                  thrust::raw_pointer_cast(owned.data()), rcnt.data(), rdsp.data(), MPI_BYTE, mpi_comm);
    detail::local_sort(owned, (Long)owned.size());  // globally-sorted segments; re-sort is ~free
  }

  // M <- global_min(Nrecv, M): parity with UpdateRefinement's clamp; abort only if a rank is empty.
  {
    long long m_ll = std::min<long long>(Nrecv, (long long)M), mg_ll = 0;
    MPI_Allreduce(&m_ll, &mg_ll, 1, MPI_LONG_LONG, MPI_MIN, mpi_comm);
    M = Long(mg_ll);
    if (M < 1) MPI_Abort(mpi_comm, 1);
  }

  mark("rebalance");
  // buf = [left halo M | owned Nrecv | right halo M].
  const Long recv0 = (rank > 0 ? M : 0), recv1 = (rank + 1 < np ? M : 0);
  DeviceVector<MortonT> buf(M + Nrecv + M);
  thrust::copy(owned.begin(), owned.begin() + Nrecv, buf.begin() + M);

  { // two-sided halo: M codes from each neighbor (device-to-device)
    const int left  = (rank > 0      ? int(rank - 1) : MPI_PROC_NULL);
    const int right = (rank + 1 < np ? int(rank + 1) : MPI_PROC_NULL);
    MortonT* b = thrust::raw_pointer_cast(buf.data());
    const int mb = int(M * (Long)sizeof(MortonT));
    MPI_Sendrecv(b + M,     mb, MPI_BYTE, left,  27, b + M + Nrecv, mb, MPI_BYTE, right, 27, mpi_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(b + Nrecv, mb, MPI_BYTE, right, 28, b,             mb, MPI_BYTE, left,  28, mpi_comm, MPI_STATUS_IGNORE);
  }

  // mins[r]: split-leaf of the pair straddling the rank boundary (buf[0], buf[M]); allgathered
  // so both sides cut at the same key.
  std::vector<Morton<DIM>> mins(np);
  {
    Morton<DIM> A{};
    if (rank > 0) {
      const MortonT ka = MortonT(buf[0]), kb = MortonT(buf[M]);
      uint8_t d = ka.CommonAncestor(kb).depth;
      if (d < MAX_DEPTH) ++d;
      A = kb.Ancestor(d);
    }
    MPI_Allgather(&A, sizeof(Morton<DIM>), MPI_BYTE, mins.data(), sizeof(Morton<DIM>), MPI_BYTE, mpi_comm);
  }
  const Morton<DIM> end_bnd = (rank + 1 < np) ? mins[rank + 1] : Morton<DIM>{}.Next();
  mark("halo+mins");

  // own the codes in [mins[rank].mid, mins[rank+1].mid)
  Long idx0 = 0, idx1 = 0;
  {
    const auto lo = buf.begin() + (M - recv0), hi = buf.begin() + (M + Nrecv + recv1);
    idx0 = thrust::lower_bound(lo, hi, mins[rank].mid) - buf.begin();
    idx1 = (rank + 1 < np) ? (thrust::lower_bound(lo, hi, mins[rank + 1].mid) - buf.begin()) : (M + Nrecv);
  }

  constexpr bool on_device = detail::is_device_vector_v<DeviceVector<Real>>;
  if constexpr (on_device) {
    detail::buildTreeGpuChunked<Real, DIM>(tree, buf, M, idx1 - idx0, idx0, mins[rank], end_bnd);
  } else {
    detail::buildTreeCpuChunked<Real, DIM>(tree, buf, M, idx1 - idx0, idx0, mins[rank], end_bnd);
  }
  mark("walk (linearize)");
  if (balance21) detail::balanceTreeDist<DIM>(tree, comm);
  mark("balance21");

  Long owned_begin = 0, owned_end = Long(tree.size());
  if (halo_size >= 0) detail::addGhostNodes<DIM>(tree, comm, halo_size, owned_begin, owned_end);
  mark("ghost");
  if (owned_range) { owned_range[0] = owned_begin; owned_range[1] = owned_end; }
}
#endif  // SCTL_HAVE_MPI

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_GPU_TREE_TXX_
