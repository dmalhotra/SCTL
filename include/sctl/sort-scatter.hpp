#ifndef _SCTL_SORT_SCATTER_HPP_
#define _SCTL_SORT_SCATTER_HPP_

#include "sctl/common.hpp"    // for Long, sctl
#include "sctl/comm.hpp"      // for Comm
#include "sctl/comm.txx"      // for Comm::Self
#include "sctl/iterator.hpp"  // for Iterator, ConstIterator
#include "sctl/vector.hpp"    // for Vector

namespace sctl {

namespace sort_scatter_detail {

/**
 * Counts and flags of a SortScatter: how many keys this rank has at each step, and the Alltoallv counts. The
 * permutations of the local sorts are in the derived struct.
 */
struct PlanBase {
  Long Nloc = 0;   ///< number of keys passed to Init on this rank
  Long Nmid = 0;   ///< number of keys on this rank after the Alltoallv in Init
  Long Ntree = 0;  ///< number of keys on this rank now (changed by Repartition)

  Vector<Long> scnt, rcnt;      ///< send and receive counts of the Alltoallv in Init

  bool recut = false;           ///< the layout has changed since Init
  bool recut_cnt = false;       ///< rscnt and rrcnt are up to date
  Vector<Long> rscnt, rrcnt;    ///< send and receive counts to go from the layout after Init to the current one

  bool inv = false;             ///< the inverse permutations are up to date

  /** Resets the counts and flags; the vectors are left as they are. */
  void Reset();
};

/** PlanBase plus the permutations of the local sorts, in host memory. */
struct Plan : PlanBase {
  Vector<Long> pre, pre_inv;    ///< permutation of the local sort before the Alltoallv, and its inverse
  Vector<Long> post, post_inv;  ///< permutation of the local sort after the Alltoallv, and its inverse
};

}  // namespace sort_scatter_detail

/**
 * Sorts keys across all ranks and stores the permutation, so that other data can be rearranged the same way
 * (ScatterForward) or back (ScatterReverse). Init, Repartition and the scatters are collective.
 *
 * @tparam Key type of the keys; trivially copyable, with operator<.
 */
template <class Key> class SortScatter {
 public:
  explicit SortScatter(const Comm& comm = Comm::Self());

  /**
   * Sorts the keys across all ranks.
   *
   * @param[in] keys keys on this rank, in any order.
   * @param[in] splitter this rank gets the keys >= *splitter and < the next rank's *splitter (ignored on rank 0). If
   *                     null on all ranks, every rank gets about the same number of keys.
   */
  void Init(const Vector<Key>& keys, const Key* splitter = nullptr);

  /**
   * Sorts the keys across all ranks of comm.
   *
   * @param[in] keys keys on this rank, in any order.
   * @param[in] splitter this rank gets the keys >= *splitter and < the next rank's *splitter (ignored on rank 0). If
   *                     null on all ranks, every rank gets about the same number of keys.
   * @param[in] comm communicator to use from now on.
   */
  void Init(const Vector<Key>& keys, const Key* splitter, const Comm& comm);

  /**
   * Moves the sorted keys between ranks according to new splitters. Data already in the order of SortedKeys() can be
   * moved along with Comm::PartitionN(data, SortedCount()).
   *
   * @param[in] splitter this rank gets the keys >= *splitter and < the next rank's *splitter (ignored on rank 0). If
   *                     null on all ranks, every rank gets about the same number of keys.
   */
  void Repartition(const Key* splitter = nullptr);

  const Vector<Key>& SortedKeys() const { return keys_; }  ///< the sorted keys on this rank
  Long LocalCount() const { return plan_.Nloc; }           ///< number of keys passed to Init on this rank
  Long SortedCount() const { return plan_.Ntree; }         ///< number of keys in SortedKeys()
  const Comm& GetComm() const { return comm_; }

  /**
   * Rearranges data from the order of the keys passed to Init to the order of SortedKeys().
   *
   * @param[in,out] data LocalCount()*dof values on input, SortedCount()*dof values on output. Replaced by a new
   *                     Vector, so it must not be a fixed-size Vector (as a view of a ScratchBuf is by default).
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is data.Dim()/LocalCount() (taken from a
   *                rank that has keys).
   */
  template <class T> void ScatterForward(Vector<T>& data, Long dof = -1) const;

  /**
   * Rearranges data from the order of SortedKeys() back to the order of the keys passed to Init.
   *
   * @param[in,out] data SortedCount()*dof values on input, LocalCount()*dof values on output. Replaced by a new
   *                     Vector, so it must not be a fixed-size Vector (as a view of a ScratchBuf is by default).
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is data.Dim()/SortedCount() (taken from
   *                a rank that has keys).
   */
  template <class T> void ScatterReverse(Vector<T>& data, Long dof = -1) const;

  /**
   * Rearranges src from the order of the keys passed to Init to the order of SortedKeys(), writing the result to dst.
   *
   * @param[out] dst SortedCount()*dof values; resized only if its size differs. Must not overlap src.
   * @param[in] src LocalCount()*dof values.
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is src.Dim()/LocalCount() (taken from a
   *                rank that has keys).
   */
  template <class T> void ScatterForward(Vector<T>& dst, const Vector<T>& src, Long dof = -1) const;

  /**
   * Rearranges src from the order of SortedKeys() back to the order of the keys passed to Init, writing the result to
   * dst.
   *
   * @param[out] dst LocalCount()*dof values; resized only if its size differs. Must not overlap src.
   * @param[in] src SortedCount()*dof values.
   * @param[in] dof number of values per key, the same on all ranks. If -1, it is src.Dim()/SortedCount() (taken from
   *                a rank that has keys).
   */
  template <class T> void ScatterReverse(Vector<T>& dst, const Vector<T>& src, Long dof = -1) const;

  /**
   * Rearranges src from the order of the keys passed to Init to the order of SortedKeys(), writing the result to dst.
   *
   * @param[out] dst iterator to SortedCount()*dof values. Must not overlap src.
   * @param[in] src iterator to LocalCount()*dof values.
   * @param[in] dof number of values per key, the same on all ranks.
   */
  template <class DIter, class SIter> void ScatterForward(DIter dst, SIter src, Long dof) const;

  /**
   * Rearranges src from the order of SortedKeys() back to the order of the keys passed to Init, writing the result to
   * dst.
   *
   * @param[out] dst iterator to LocalCount()*dof values. Must not overlap src.
   * @param[in] src iterator to SortedCount()*dof values.
   * @param[in] dof number of values per key, the same on all ranks.
   */
  template <class DIter, class SIter> void ScatterReverse(DIter dst, SIter src, Long dof) const;

  /** Test on Comm::World(); Key must be constructible from Long. */
  static void test();

 private:
  Comm comm_;
  Vector<Key> keys_;
  mutable sort_scatter_detail::Plan plan_;  ///< some entries are computed on first use, hence mutable
};

}  // namespace sctl

#endif  // _SCTL_SORT_SCATTER_HPP_
