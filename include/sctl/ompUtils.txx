#ifndef _SCTL_OMPUTILS_TXX_
#define _SCTL_OMPUTILS_TXX_

#include <algorithm>          // for lower_bound, sort, merge, copy
#include <cstring>            // for memcpy
#include <functional>         // for less
#include <iterator>           // for iterator_traits
#include <type_traits>        // for is_trivially_copyable

#include "sctl/common.hpp"        // for Integer, Long, SCTL_UNUSED, sctl
#include "sctl/ompUtils.hpp"      // for merge_sort, merge, reduce, scan
#include "sctl/iterator.hpp"      // for Iterator
#include "sctl/iterator.txx"      // for Ptr2Itr
#include "sctl/scratch_pool.hpp"  // for ScratchBuf
#include "sctl/scratch_pool.txx"

namespace sctl {

namespace omp_par_detail {

  inline Integer PickThreads(Long nbytes, Integer requested) {
    constexpr Long kFullThreadsBytes      = 2L * 1024L * 1024L;
    if (requested > 0) return requested;
    if (requested == 0) return 1;
    if (nbytes < kFullThreadsBytes) return 1;
    if (SCTL_IN_PARALLEL()) return 1;
    return (Integer)SCTL_GET_MAX_THREADS();
  }
}

template <class OutputIt, class InputIt> inline void omp_par::memcpy(OutputIt dst, InputIt src, Long n, Integer nthreads) {
  using T = typename std::iterator_traits<OutputIt>::value_type;
  using src_value_t = typename std::iterator_traits<InputIt>::value_type;
  static_assert(std::is_same<T, typename std::remove_cv<src_value_t>::type>::value,
                "omp_par::memcpy: source and destination value types must match");
  static_assert(std::is_base_of<std::random_access_iterator_tag, typename std::iterator_traits<OutputIt>::iterator_category>::value &&
                std::is_base_of<std::random_access_iterator_tag, typename std::iterator_traits<InputIt>::iterator_category>::value,
                "omp_par::memcpy: iterators must be random-access over contiguous storage");
  static_assert(std::is_trivially_copyable<T>::value,
                "omp_par::memcpy: T must be trivially copyable; use omp_par::copy for arbitrary types");
  if (n <= 0) return;
  if ((const void*)&dst[0] == (const void*)&src[0]) return;

  const Long nbytes = n * (Long)sizeof(T);
  const Integer nt = omp_par_detail::PickThreads(nbytes, nthreads);

  if (nt <= 1) {
    std::memcpy((void*)&dst[0], (const void*)&src[0], (size_t)nbytes);
    return;
  }

  #pragma omp parallel num_threads(nt)
  {
    const Integer tid = (Integer)SCTL_GET_THREAD_NUM();
    const Integer p   = (Integer)SCTL_GET_NUM_THREADS();
    const Long s = ((Long)tid * n) / p;
    const Long e = ((Long)(tid + 1) * n) / p;
    if (e > s) std::memcpy((void*)&dst[s], (const void*)&src[s], (size_t)((e - s) * (Long)sizeof(T)));
  }
}

template <class InputIt, class OutputIt> inline OutputIt omp_par::copy(InputIt first, InputIt last, OutputIt dst, Integer nthreads) {
  using val_t  = typename std::iterator_traits<InputIt>::value_type;
  using diff_t = typename std::iterator_traits<InputIt>::difference_type;

  const diff_t n = last - first;
  if (n <= 0) return dst;

  const Long nbytes = (Long)n * (Long)sizeof(val_t);
  const Integer nt = omp_par_detail::PickThreads(nbytes, nthreads);

  if (nt <= 1) return std::copy(first, last, dst);

  #pragma omp parallel num_threads(nt)
  {
    const Integer tid = (Integer)SCTL_GET_THREAD_NUM();
    const Integer p   = (Integer)SCTL_GET_NUM_THREADS();
    const diff_t s = (diff_t)(((Long)tid * (Long)n) / p);
    const diff_t e = (diff_t)(((Long)(tid + 1) * (Long)n) / p);
    if (e > s) std::copy(first + s, first + e, dst + s);
  }
  return dst + n;
}

template <class ConstIter, class Iter, class Int, class StrictWeakOrdering> inline void omp_par::merge(ConstIter A_, ConstIter A_last, ConstIter B_, ConstIter B_last, Iter C_, Int p, StrictWeakOrdering comp) {
  typedef typename std::iterator_traits<Iter>::difference_type _DiffType;
  typedef typename std::iterator_traits<Iter>::value_type _ValType;

  _DiffType N1 = A_last - A_;
  _DiffType N2 = B_last - B_;
  if (N1 == 0 && N2 == 0) return;
  if (N1 == 0 || N2 == 0) {
    ConstIter A = (N1 == 0 ? B_ : A_);
    _DiffType N = (N1 == 0 ? N2 : N1);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p; i++) {
      _DiffType indx1 = (i * N) / p;
      _DiffType indx2 = ((i + 1) * N) / p;
      if (indx2 - indx1 > 0) std::copy(A+indx1, A+indx2, C_+indx1);
      //if (indx2 - indx1 > 0) memcpy(&C_[indx1], &A[indx1], (indx2 - indx1) * sizeof(_ValType));
    }
    return;
  }

  // Split both arrays ( A and B ) into n equal parts.
  // Find the position of each split in the final merged array.
  int n = 10;
  ScratchBuf<_ValType>  split(p * n * 2);
  ScratchBuf<_DiffType> split_size(p * n * 2);
#pragma omp parallel for
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < n; j++) {
      int indx = i * n + j;
      _DiffType indx1 = (indx * N1) / (p * n);
      split[indx] = A_[indx1];
      split_size[indx] = indx1 + (std::lower_bound(B_, B_last, split[indx], comp) - B_);

      indx1 = (indx * N2) / (p * n);
      indx += p * n;
      split[indx] = B_[indx1];
      split_size[indx] = indx1 + (std::lower_bound(A_, A_last, split[indx], comp) - A_);
    }
  }

  // Find the closest split position for each thread that will
  // divide the final array equally between the threads.
  ScratchBuf<_DiffType> split_indx_A(p + 1);
  ScratchBuf<_DiffType> split_indx_B(p + 1);
  split_indx_A[0] = 0;
  split_indx_B[0] = 0;
  split_indx_A[p] = N1;
  split_indx_B[p] = N2;
#pragma omp parallel for schedule(static)
  for (int i = 1; i < p; i++) {
    _DiffType req_size = (i * (N1 + N2)) / p;

    int j = std::lower_bound(split_size.begin(), split_size.begin() + p * n, req_size, std::less<_DiffType>()) - split_size.begin();
    if (j >= p * n) j = p * n - 1;
    _ValType split1 = split[j];
    _DiffType split_size1 = split_size[j];

    j = (std::lower_bound(split_size.begin() + p * n, split_size.begin() + p * n * 2, req_size, std::less<_DiffType>()) - (split_size.begin() + p * n)) + p * n;
    if (j >= 2 * p * n) j = 2 * p * n - 1;
    if (abs(split_size[j] - req_size) < abs(split_size1 - req_size)) {
      split1 = split[j];
      split_size1 = split_size[j];
    }

    split_indx_A[i] = std::lower_bound(A_, A_last, split1, comp) - A_;
    split_indx_B[i] = std::lower_bound(B_, B_last, split1, comp) - B_;
  }

// Merge for each thread independently.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; i++) {
    Iter C = C_ + split_indx_A[i] + split_indx_B[i];
    std::merge(A_ + split_indx_A[i], A_ + split_indx_A[i + 1], B_ + split_indx_B[i], B_ + split_indx_B[i + 1], C, comp);
  }
}

template <class T, class StrictWeakOrdering> inline void omp_par::merge_sort(T A, T A_last, StrictWeakOrdering comp) {
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;

  int p = SCTL_GET_MAX_THREADS();
  _DiffType N = A_last - A;
  if (N < 2 * p) {
    std::sort(A, A_last, comp);
    return;
  }
  SCTL_UNUSED(A[0]);
  SCTL_UNUSED(A[N - 1]);

  // Split the array A into p equal parts.
  ScratchBuf<_DiffType> split(p + 1);
  split[p] = N;
#pragma omp parallel for schedule(static)
  for (int id = 0; id < p; id++) {
    split[id] = (id * N) / p;
  }

// Sort each part independently.
#pragma omp parallel for schedule(static)
  for (int id = 0; id < p; id++) {
    std::sort(A + split[id], A + split[id + 1], comp);
  }

  // Merge two parts at a time.
  ScratchBuf<_ValType> B(N);
  Iterator<_ValType> A_ = Ptr2Itr<_ValType>(&A[0], N);
  Iterator<_ValType> B_ = B.begin();
  for (int j = 1; j < p; j = j * 2) {
    for (int i = 0; i < p; i = i + 2 * j) {
      if (i + j < p) {
        omp_par::merge(A_ + split[i], A_ + split[i + j], A_ + split[i + j], A_ + split[(i + 2 * j <= p ? i + 2 * j : p)], B_ + split[i], p, comp);
      } else {
#pragma omp parallel for
        for (int k = split[i]; k < split[p]; k++) B_[k] = A_[k];
      }
    }
    Iterator<_ValType> tmp_swap = A_;
    A_ = B_;
    B_ = tmp_swap;
  }

  // The final result should be in A.
  if (A_ != Ptr2Itr<_ValType>(&A[0], N)) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] = A_[i];
  }
}

template <class T> inline void omp_par::merge_sort(T A, T A_last) {
  typedef typename std::iterator_traits<T>::value_type _ValType;
  omp_par::merge_sort(A, A_last, std::less<_ValType>());
}

template <class ConstIter, class Iter, class StrictWeakOrdering> inline void omp_par::sample_sort(ConstIter A, Iter B, Long N, StrictWeakOrdering comp) {
  typedef typename std::iterator_traits<Iter>::value_type _ValType;
  const Integer nt = SCTL_GET_MAX_THREADS();
  const bool inplace = (N > 0) && ((const void*)&A[0] == (const void*)&B[0]);  // output aliases input

  if (SCTL_IN_PARALLEL() || nt < 2 || N < 64 * nt) {  // serial std::sort when nested or too small to parallelize
    if (!inplace) for (Long i = 0; i < N; i++) B[i] = A[i];
    std::sort(B, B + N, comp);
    return;
  }
  if (inplace) {  // can't scatter onto the input: sort into scratch then copy back
    ScratchBuf<_ValType> tmp_buf(N);
    omp_par::sample_sort(A, tmp_buf.begin(), N, comp);
    Iterator<_ValType> tmp = tmp_buf.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < N; i++) B[i] = tmp[i];
    return;
  }

  // Adaptive bucket count: aim for >=~1024 elem/bucket, between nt and 16*nt buckets.
  const Long nbuck = std::max<Long>(nt, std::min<Long>(16 * nt, N / 1024));

  // 1. Pick nbuck-1 splitters by sorting a regularly-strided oversample.
  const Long Ns = std::min<Long>(nbuck * 8, N);
  ScratchBuf<_ValType> samp_buf(Ns), split_buf(nbuck - 1);
  Iterator<_ValType> samp = samp_buf.begin(), split = split_buf.begin();
  for (Long i = 0; i < Ns; i++) samp[i] = A[i * N / Ns];
  std::sort(samp, samp + Ns, comp);
  for (Long k = 0; k < nbuck - 1; k++) split[k] = samp[(k + 1) * Ns / nbuck];
  auto bucket = [&split, &comp, nbuck](const _ValType& x) { return std::upper_bound(split, split + (nbuck - 1), x, comp) - split; };

  // 2. Per-thread histogram of bucket counts over contiguous chunks.
  ScratchBuf<Long> hist_buf(nt * nbuck), chunk_buf(nt + 1);
  Iterator<Long> hist = hist_buf.begin(), chunk = chunk_buf.begin();
  for (Long i = 0; i < nt * nbuck; i++) hist[i] = 0;
  for (Integer t = 0; t <= nt; t++) chunk[t] = (Long)t * N / nt;
  #pragma omp parallel num_threads(nt)
  { const Integer t = SCTL_GET_THREAD_NUM();
    Iterator<Long> h = hist + t * nbuck;
    for (Long i = chunk[t]; i < chunk[t + 1]; i++) h[bucket(A[i])]++;
  }

  // 3. Bucket offsets in B and per-(thread,bucket) write positions.
  ScratchBuf<Long> bdsp_buf(nbuck + 1), tdsp_buf(nt * nbuck);
  Iterator<Long> bdsp = bdsp_buf.begin(), tdsp = tdsp_buf.begin();
  bdsp[0] = 0;
  for (Long b = 0; b < nbuck; b++) {
    Long o = bdsp[b];
    for (Integer t = 0; t < nt; t++) { tdsp[t * nbuck + b] = o; o += hist[t * nbuck + b]; }
    bdsp[b + 1] = o;
  }

  // 4. Scatter A into B grouped by bucket.
  #pragma omp parallel num_threads(nt)
  { const Integer t = SCTL_GET_THREAD_NUM();
    ScratchBuf<Long> o_buf(nbuck); Iterator<Long> o = o_buf.begin();
    for (Long b = 0; b < nbuck; b++) o[b] = tdsp[t * nbuck + b];
    for (Long i = chunk[t]; i < chunk[t + 1]; i++) { Long b = bucket(A[i]); B[o[b]++] = A[i]; }
  }

  // 5. Sort each bucket independently (dynamic for load balance).
  #pragma omp parallel for schedule(dynamic) num_threads(nt)
  for (Long b = 0; b < nbuck; b++) std::sort(B + bdsp[b], B + bdsp[b + 1], comp);
}

template <class T, class StrictWeakOrdering> inline void omp_par::sample_sort(T A, T A_last, StrictWeakOrdering comp) {
  omp_par::sample_sort(A, A, A_last - A, comp);  // A==B: in-place path is auto-detected
}

template <class T> inline void omp_par::sample_sort(T A, T A_last) {
  typedef typename std::iterator_traits<T>::value_type _ValType;
  omp_par::sample_sort(A, A_last, std::less<_ValType>());
}

template <class ConstIter, class Iter, class StrictWeakOrdering> inline void omp_par::multiway_merge(ConstIter runs, ConstIterator<Long> run_dsp, Long nruns, Iter out, StrictWeakOrdering comp) {
  typedef typename std::iterator_traits<Iter>::value_type _ValType;
  const Integer nt = SCTL_GET_MAX_THREADS();
  const Long N = run_dsp[nruns];

  // Split the merged output into nt contiguous chunks via sampled splitters (regular-strided
  // oversample of the runs); for nt==1 this is empty -> a single serial heap merge.
  ScratchBuf<_ValType> tsplit_buf(nt > 1 ? nt - 1 : 0);
  Iterator<_ValType> tsplit = tsplit_buf.begin();
  if (nt > 1 && N > 0) {
    const Long osamp = 16, Ns = std::min<Long>((Long)nt * osamp, N);
    ScratchBuf<_ValType> samp(Ns);
    for (Long i = 0; i < Ns; i++) samp[i] = runs[i * N / Ns];
    std::sort(samp.begin(), samp.begin() + Ns, comp);
    for (Integer k = 0; k < nt - 1; k++) tsplit[k] = samp[std::min<Long>(Ns - 1, (Long)(k + 1) * Ns / nt)];
  }

  #pragma omp parallel num_threads(nt)
  { const Integer t = SCTL_GET_THREAD_NUM();
    ScratchBuf<Long> pos_buf(nruns), end_buf(nruns);
    Iterator<Long> pos = pos_buf.begin(), end = end_buf.begin();
    Long out_off = 0;  // this thread's sub-range of each run; output offset = #elements before its chunk
    for (Long d = 0; d < nruns; d++) {
      ConstIter rb = runs + run_dsp[d], re = runs + run_dsp[d + 1];
      pos[d] = (t == 0)      ? run_dsp[d]     : std::lower_bound(rb, re, tsplit[t - 1], comp) - runs;
      end[d] = (t == nt - 1) ? run_dsp[d + 1] : std::lower_bound(rb, re, tsplit[t],     comp) - runs;
      out_off += pos[d] - run_dsp[d];
    }
    auto hcmp = [&runs, &pos, &comp](Long a, Long b) { return comp(runs[pos[b]], runs[pos[a]]); };  // inverted -> min
    ScratchBuf<Long> heap_buf(nruns); Iterator<Long> heap = heap_buf.begin(); Long hn = 0;
    for (Long d = 0; d < nruns; d++) if (pos[d] < end[d]) heap[hn++] = d;
    std::make_heap(heap, heap + hn, hcmp);
    Long o = out_off;
    while (hn > 0) {
      std::pop_heap(heap, heap + hn, hcmp);
      const Long d = heap[--hn];
      out[o++] = runs[pos[d]++];
      if (pos[d] < end[d]) { heap[hn++] = d; std::push_heap(heap, heap + hn, hcmp); }
    }
  }
}

template <class ConstIter, class Int> typename std::iterator_traits<ConstIter>::value_type omp_par::reduce(ConstIter A, Int cnt) {
  typedef typename std::iterator_traits<ConstIter>::value_type ValueType;
  ValueType sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (Int i = 0; i < cnt; i++) sum += A[i];
  return sum;
}

template <class ConstIter, class Iter, class Int> void omp_par::scan(ConstIter A, Iter B, Int cnt) {
  typedef typename std::iterator_traits<Iter>::value_type ValueType;

  Integer p = SCTL_GET_MAX_THREADS();
  if (cnt < (Int)100 * p) {
    for (Int i = 1; i < cnt; i++) B[i] = B[i - 1] + A[i - 1];
    return;
  }

  Int step_size = cnt / p;

#pragma omp parallel for schedule(static)
  for (Integer i = 0; i < p; i++) {
    Int start = i * step_size;
    Int end = start + step_size;
    if (i == p - 1) end = cnt;
    if (i != 0) B[start] = 0;
    for (Int j = (Int)start + 1; j < (Int)end; j++) B[j] = B[j - 1] + A[j - 1];
  }

  ScratchBuf<ValueType> sum(p);
  sum[0] = 0;
  for (Integer i = 1; i < p; i++) sum[i] = sum[i - 1] + B[i * step_size - 1] + A[i * step_size - 1];

#pragma omp parallel for schedule(static)
  for (Integer i = 1; i < p; i++) {
    Int start = i * step_size;
    Int end = start + step_size;
    if (i == p - 1) end = cnt;
    ValueType sum_ = sum[i];
    for (Int j = (Int)start; j < (Int)end; j++) B[j] += sum_;
  }
}

}  // end namespace

#endif // _SCTL_OMPUTILS_TXX_
