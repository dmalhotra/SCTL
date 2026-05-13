#ifndef _SCTL_OMPUTILS_HPP_
#define _SCTL_OMPUTILS_HPP_

#include <iterator>         // for iterator_traits

#include "sctl/common.hpp"  // for sctl

namespace sctl {
namespace omp_par {

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
