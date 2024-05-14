#ifndef _SCTL_OMP_UTILS_H_
#define _SCTL_OMP_UTILS_H_

#include <sctl/common.hpp>

#include <iterator>

namespace SCTL_NAMESPACE {
namespace omp_par {

/**
 * @brief Merges two sorted ranges into a single sorted range.
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
 * @brief Performs merge sort on a range using a custom comparison function.
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
 * @brief Performs merge sort on a range.
 *
 * @tparam T Iterator type for the input range.
 *
 * @param A Beginning iterator of the range.
 * @param A_last Ending iterator of the range.
 */
template <class T> void merge_sort(T A, T A_last);

/**
 * @brief Reduces the elements in a range to a single value.
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
 * @brief Performs a parallel prefix sum (scan) operation on a range.
 *
 * @tparam ConstIter Iterator type for the input range.
 * @tparam Iter Iterator type for the output range.
 * @tparam Int Integer type for indexing.
 *
 * @param A Beginning iterator of the input range.
 * @param B Beginning iterator of the output range.
 * @param cnt Number of elements in the range.
 */
template <class ConstIter, class Iter, class Int> void scan(ConstIter A, Iter B, Int cnt);

}  // end namespace omp_par
}  // end namespace SCTL_NAMESPACE

#include SCTL_INCLUDE(ompUtils.txx)

#endif  //_SCTL_OMP_UTILS_H_
