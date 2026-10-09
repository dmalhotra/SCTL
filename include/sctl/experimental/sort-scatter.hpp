/**
 * @file sort-scatter.hpp
 * SortScatter with the keys and data in host or device memory (experimental).
 */

#ifndef _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
#define _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_

#include "sctl/comm.hpp"
#include "sctl/experimental/gpu-vector.hpp"
#include "sctl/vector.hpp"
#include "sctl/sort-scatter.hpp"  // for sort_scatter_detail::PlanBase

namespace gpu_tree {

using sctl::Long;
using sctl::Comm;

namespace detail_sortScatter {

/** sctl::sort_scatter_detail::PlanBase plus the permutations of the local sorts, in DevVec memory. */
template <template <class...> class DevVec> struct Plan : sctl::sort_scatter_detail::PlanBase {
  DevVec<Long> pre, pre_inv;    ///< permutation of the local sort before the Alltoallv, and its inverse
  DevVec<Long> post, post_inv;  ///< permutation of the local sort after the Alltoallv, and its inverse
};

}  // namespace detail_sortScatter

/**
 * Sorts keys across all ranks and stores the permutation, so that other data can be rearranged the same way
 * (ScatterForward) or back (ScatterReverse). Init, Repartition and the scatters are collective.
 *
 * @tparam Key type of the keys; trivially copyable, with operator<, GetIntKey() and FromIntKey() (e.g. MortonCode).
 * @tparam DevVec container for the keys and data: HostVector or DeviceVector.
 */
template <class Key, template <class...> class DevVec = HostVector>
class SortScatter {
 public:
  explicit SortScatter(const Comm& comm = Comm::Self()) : comm_(comm) {}

  /**
   * Sorts the keys across all ranks.
   *
   * @param[in] keys keys on this rank, in any order; a DevVec<Key> converts to the view.
   * @param[in] splitter this rank gets the keys >= *splitter and < the next rank's *splitter (ignored on rank 0). If
   *                     null on all ranks, every rank gets about the same number of keys.
   */
  void Init(DataView<const Key, DevVec> keys, const Key* splitter = nullptr);

  /**
   * Sorts the keys across all ranks of comm.
   *
   * @param[in] keys keys on this rank, in any order; a DevVec<Key> converts to the view.
   * @param[in] splitter this rank gets the keys >= *splitter and < the next rank's *splitter (ignored on rank 0). If
   *                     null on all ranks, every rank gets about the same number of keys.
   * @param[in] comm communicator to use from now on.
   */
  void Init(DataView<const Key, DevVec> keys, const Key* splitter, const Comm& comm);

  /**
   * Moves the sorted keys between ranks according to new splitters. Data already in the order of SortedKeys() can be
   * moved along with detail::partitionN.
   *
   * @param[in] splitter this rank gets the keys >= *splitter and < the next rank's *splitter (ignored on rank 0). If
   *                     null on all ranks, every rank gets about the same number of keys.
   */
  void Repartition(const Key* splitter = nullptr);

  const DevVec<Key>& SortedKeys() const { return keys_; }  ///< the sorted keys on this rank
  Long LocalCount() const { return plan_.Nloc; }                 ///< number of keys passed to Init on this rank
  Long SortedCount() const { return plan_.Ntree; }               ///< number of keys in SortedKeys()
  const Comm& GetComm() const { return comm_; }

  /**
   * Rearranges data from the order of the keys passed to Init to the order of SortedKeys().
   *
   * @param[in,out] data LocalCount()*dof values on input, SortedCount()*dof values on output. Its storage is swapped
   *                     with an internal buffer.
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is data.size()/LocalCount() (taken from
   *                a rank that has keys).
   */
  template <class T> void ScatterForward(DevVec<T>& data, Long dof = -1) const;

  /**
   * Rearranges data from the order of SortedKeys() back to the order of the keys passed to Init.
   *
   * @param[in,out] data SortedCount()*dof values on input, LocalCount()*dof values on output. Its storage is swapped
   *                     with an internal buffer.
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is data.size()/SortedCount() (taken
   *                from a rank that has keys).
   */
  template <class T> void ScatterReverse(DevVec<T>& data, Long dof = -1) const;

  /**
   * Rearranges src from the order of the keys passed to Init to the order of SortedKeys(), writing the result to dst.
   *
   * @param[out] dst SortedCount()*dof values; resized only if its size differs. Must not overlap src.
   * @param[in] src LocalCount()*dof values.
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is src.size()/LocalCount() (taken from
   *                a rank that has keys).
   */
  template <class T> void ScatterForward(DevVec<T>& dst, const DevVec<T>& src, Long dof = -1) const;

  /**
   * Rearranges src from the order of SortedKeys() back to the order of the keys passed to Init, writing the result to
   * dst.
   *
   * @param[out] dst LocalCount()*dof values; resized only if its size differs. Must not overlap src.
   * @param[in] src SortedCount()*dof values.
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is src.size()/SortedCount() (taken
   *                from a rank that has keys).
   */
  template <class T> void ScatterReverse(DevVec<T>& dst, const DevVec<T>& src, Long dof = -1) const;

  /**
   * Rearranges src from the order of the keys passed to Init to the order of SortedKeys(), writing the result to dst.
   *
   * @param[out] dst SortedCount()*dof values in DevVec memory. Must not overlap src.
   * @param[in] src LocalCount()*dof values in DevVec memory.
   * @param[in] dof number of values per key, the same on all ranks.
   */
  template <class T> void ScatterForward(T* dst, const T* src, Long dof) const;

  /**
   * Rearranges src from the order of SortedKeys() back to the order of the keys passed to Init, writing the result to
   * dst.
   *
   * @param[out] dst LocalCount()*dof values in DevVec memory. Must not overlap src.
   * @param[in] src SortedCount()*dof values in DevVec memory.
   * @param[in] dof number of values per key, the same on all ranks.
   */
  template <class T> void ScatterReverse(T* dst, const T* src, Long dof) const;

  /** Test on Comm::World(); Key must be constructible from Long. */
  static void test();

 private:
  Comm comm_;
  DevVec<Key> keys_;
  mutable detail_sortScatter::Plan<DevVec> plan_;  ///< some entries are computed on first use, hence mutable
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
