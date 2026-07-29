#ifndef _SCTL_OMPUTILS_HPP_
#define _SCTL_OMPUTILS_HPP_

#include <iterator>         // for iterator_traits

#include "sctl/common.hpp"   // for sctl
#include "sctl/iterator.hpp" // for Iterator, ConstIterator

namespace sctl {
namespace omp_par {

/**
 * Parallel bytewise copy over generic random-access iterators (raw pointers
 * or sctl `Iterator` / `ConstIterator`, which are bounds-checked in MEMDEBUG).
 * Byte-wise (memcpy) semantics require contiguous, trivially-copyable storage;
 * value types of both iterators must match and be trivially copyable (asserted
 * at compile time — contiguity itself cannot be checked before C++20 and is
 * the caller's responsibility). For element-wise copy through arbitrary
 * iterators, use `omp_par::copy` (or plain `std::copy`).
 *
 * The thread count is chosen by an empirical heuristic when `nthreads < 0`:
 *   - bytes < 2 MB         → serial (`std::memcpy`)
 *   - bytes >= 2 MB          → full `omp_get_max_threads()`
 * Pass `nthreads > 0` to force a specific thread count; pass `1` to force serial.
 * If called from inside an `omp parallel` region the heuristic forces serial.
 *
 * @tparam OutputIt random-access iterator over contiguous storage.
 * @tparam InputIt  random-access iterator over contiguous storage.
 * @param[out] dst destination iterator.
 * @param[in]  src source iterator.
 * @param[in]  n   number of elements (NOT bytes).
 * @param[in]  nthreads explicit thread count, or -1 for the heuristic.
 */
template <class OutputIt, class InputIt> void memcpy(OutputIt dst, InputIt src, Long n, Integer nthreads = -1);

/**
 * Parallel element-wise copy over random-access iterators. Each chunk is
 * dispatched to `std::copy`, so user `operator=` is invoked normally (safe for
 * non-trivially-copyable T). Thread-count heuristic is identical to
 * `omp_par::memcpy`, using `sizeof(value_type) * (last - first)` as the byte
 * budget.
 *
 * @tparam InputIt  random-access input iterator.
 * @tparam OutputIt random-access output iterator.
 * @param[in]  first range begin.
 * @param[in]  last  range end (one past last element).
 * @param[out] dst   destination range begin.
 * @param[in]  nthreads explicit thread count, or -1 for the heuristic.
 * @return iterator past the last element written: `dst + (last - first)`.
 */
template <class InputIt, class OutputIt> OutputIt copy(InputIt first, InputIt last, OutputIt dst, Integer nthreads = -1);

/**
 * Merges two sorted ranges into a single sorted range.
 *
 * @tparam ConstIter Iterator type for the input ranges.
 * @tparam Iter Iterator type for the output range.
 * @tparam Int Integer type for indexing.
 * @tparam StrictWeakOrdering Functor type for comparing elements.
 *
 * @param A_ Beginning iterator of the first range.
 * @param A_last Ending iterator of the first range.
 * @param B_ Beginning iterator of the second range.
 * @param B_last Ending iterator of the second range.
 * @param C_ Beginning iterator of the output range.
 * @param p Integer indicating the parallelism level.
 * @param comp Functor for comparing elements.
 */
template <class ConstIter, class Iter, class Int, class StrictWeakOrdering> void merge(ConstIter A_, ConstIter A_last, ConstIter B_, ConstIter B_last, Iter C_, Int p, StrictWeakOrdering comp);

/**
 * Performs merge sort on a range using a custom comparison function.
 *
 * @tparam T Iterator type for the input range.
 * @tparam StrictWeakOrdering Functor type for comparing elements.
 *
 * @param A Beginning iterator of the range.
 * @param A_last Ending iterator of the range.
 * @param comp Functor for comparing elements.
 */
template <class T, class StrictWeakOrdering> void merge_sort(T A, T A_last, StrictWeakOrdering comp);

/**
 * Performs merge sort on a range.
 *
 * @tparam T Iterator type for the input range.
 *
 * @param A Beginning iterator of the range.
 * @param A_last Ending iterator of the range.
 */
template <class T> void merge_sort(T A, T A_last);

/**
 * Parallel sample sort. Writes the sorted output to a separate buffer B (out-of-place).
 * Scales better than merge_sort for large records by bounding data movement (one
 * bucket scatter + per-bucket std::sort) rather than O(log p) full-array merge passes.
 *
 * @param A Beginning iterator of the (unmodified) input range.
 * @param B Beginning iterator of the output range (size N).
 * @param N Number of elements.
 * @param comp Functor for comparing elements.
 */
template <class ConstIter, class Iter, class StrictWeakOrdering> void sample_sort(ConstIter A, Iter B, Long N, StrictWeakOrdering comp);

/**
 * In-place parallel sample sort (sorts the range [A, A_last) in place using an
 * internal scratch buffer). Drop-in replacement for merge_sort.
 *
 * @param A Beginning iterator of the range.
 * @param A_last Ending iterator of the range.
 * @param comp Functor for comparing elements.
 */
template <class T, class StrictWeakOrdering> void sample_sort(T A, T A_last, StrictWeakOrdering comp);

/**
 * In-place parallel sample sort using the default (operator<) ordering.
 *
 * @param A Beginning iterator of the range.
 * @param A_last Ending iterator of the range.
 */
template <class T> void sample_sort(T A, T A_last);

/**
 * Parallel multiway merge of `nruns` sorted runs into a single sorted output. The runs are the
 * contiguous segments [run_dsp[d], run_dsp[d+1]) of `runs` (run_dsp has nruns+1 entries, with
 * run_dsp[0] the start offset and run_dsp[nruns] the total count); `out` (that many elements)
 * receives their merge. Parallelized by splitting the output into per-thread contiguous chunks
 * via sampled splitters, each thread heap-merging its chunk's sub-runs -- single pass, no extra
 * copies, exploits the pre-sortedness (O((N/p)*log nruns) vs O((N/p)*log(N/p)) for a re-sort).
 *
 * @param runs Beginning iterator of the buffer holding the concatenated sorted runs.
 * @param run_dsp Run boundary offsets into `runs` (nruns+1 entries, ascending).
 * @param nruns Number of runs.
 * @param out Beginning iterator of the output range (size run_dsp[nruns]).
 * @param comp Functor for comparing elements.
 */
template <class ConstIter, class Iter, class StrictWeakOrdering> void multiway_merge(ConstIter runs, ConstIterator<Long> run_dsp, Long nruns, Iter out, StrictWeakOrdering comp);

/**
 * Reduces the elements in a range to a single value.
 *
 * @tparam ConstIter Iterator type for the input range.
 * @tparam Int Integer type for indexing.
 *
 * @param A Beginning iterator of the range.
 * @param cnt Number of elements in the range.
 * @return The reduced value.
 */
template <class ConstIter, class Int> typename std::iterator_traits<ConstIter>::value_type reduce(ConstIter A, Int cnt);

/**
 * Parallel **exclusive** prefix sum with caller-supplied seed.
 *
 * Computes
 *
 *     B[0] is left untouched (must be initialised by the caller),
 *     B[i] = B[0] + A[0] + A[1] + ... + A[i-1]   for i = 1, ..., cnt-1.
 *
 * Equivalently, `B[i]` is the sum of the first `i` elements of `A` shifted by
 * the caller-supplied initial value at `B[0]`. With the conventional seed
 * `B[0] = 0`, this is a standard exclusive prefix sum (e.g. converting a
 * count array into a displacement array). The function **does not write**
 * `B[0]`, so leaving it uninitialised is a bug.
 *
 * `A[cnt-1]` is not read; only `A[0..cnt-2]` participate. Aliasing `A` and
 * `B` is not supported.
 *
 * @tparam ConstIter Iterator type for the input range.
 * @tparam Iter Iterator type for the output range.
 * @tparam Int Integer type for indexing.
 *
 * @param[in] A Beginning iterator of the input range (length `cnt`).
 * @param[in,out] B Beginning iterator of the output range (length `cnt`); `B[0]` is read as the seed and must be initialised before the call; `B[1..cnt-1]` are written.
 * @param[in] cnt Number of elements in each range.
 */
template <class ConstIter, class Iter, class Int> void scan(ConstIter A, Iter B, Int cnt);

}  // end namespace omp_par
}  // end namespace sctl

#endif // _SCTL_OMPUTILS_HPP_
