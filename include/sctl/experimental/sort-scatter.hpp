/**
 * @file sort-scatter.hpp
 * SortScatter: keys sorted globally and cut at rank splitters, with the permutation to and from the
 * order the caller handed them in (experimental).
 */

#ifndef _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
#define _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_

#include "sctl/comm.hpp"
#include "sctl/experimental/gpu-vector.hpp"
#include "sctl/vector.hpp"
#include "sctl/sort-scatter.hpp"  // the host-side Plan and its helpers

namespace gpu_tree {

using sctl::Long;
using sctl::Comm;

namespace detail_sortScatter {

/** `sctl::sort_scatter_detail::PlanBase` -- the stages, their counts and flags -- with the local maps in backend memory. */
template <template <class...> class DevVec> struct Plan : sctl::sort_scatter_detail::PlanBase {
  DevVec<Long> pre, pre_inv;    ///< Nloc: stage 1 and its inverse
  DevVec<Long> post, post_inv;  ///< Nmid: stage 3 and its inverse; unused at np=1
};

}  // namespace detail_sortScatter

/**
 * Keys sorted globally and cut at rank splitters, with the permutation to and from the caller's
 * order: the backend-side `Comm::SortScatterIndex` plus `ScatterForward`/`ScatterReverse` as an
 * object. Those rebuild the exchange from a global index on every move; here the sort's stages are
 * recorded once (`detail_sortScatter::Plan`) and replayed, the payload never leaves backend memory,
 * and the sorted keys are kept for reuse.
 *
 * @tparam Key Trivially copyable, ordered by `operator<`, with `GetIntKey`/`FromIntKey` for the
 *             radix path (as `MortonCode` has).
 * @tparam DevVec Backend container: `HostVector` or `gpu_tree::DeviceVector`.
 *
 * Every member that moves keys or data is collective.
 */
template <class Key, template <class...> class DevVec = HostVector>
class SortScatter {
 public:
  explicit SortScatter(const Comm& comm = Comm::Self()) : comm_(comm) {}

  /**
   * Sort `keys` (caller order) into the global order cut at `splitters`: np entries, `splitters[r]`
   * the first key of rank r's range; `splitters[0]` is not consulted. Pass `std::move(keys)` to
   * avoid the copy.
   */
  void Init(DevVec<Key> keys, const sctl::Vector<Key>& splitters);

  /** Move the sorted keys to the partition given by new `splitters`; the operators follow. */
  void Repartition(const sctl::Vector<Key>& splitters);

  /**
   * Move `data`, `dof` values per key in the layout before the last `Repartition` (its previous
   * `SortedCount()`), to the current layout, in place. A no-op when that `Repartition` moved nothing.
   */
  template <class T> void RepartitionData(DevVec<T>& data, Long dof) const;

  const DevVec<Key>& SortedKeys() const { return keys_; }  ///< this rank's stretch of the global order
  Long LocalCount() const { return plan_.Nloc; }                 ///< keys the caller handed in
  Long SortedCount() const { return plan_.Ntree; }               ///< keys held now
  const Comm& GetComm() const { return comm_; }

  /** Caller order -> sorted order, `dof` values per key (agreed across ranks): `LocalCount()*dof` values in, `SortedCount()*dof` out. */
  template <class T> void ScatterForward(DevVec<T>& data, Long dof) const;

  /** Sorted order -> caller order: the inverse of `ScatterForward`. */
  template <class T> void ScatterReverse(DevVec<T>& data, Long dof) const;

  /** Same between caller-sized raw buffers: `src` `LocalCount()*dof` values, `dst` `SortedCount()*dof`, no overlap. */
  template <class T> void ScatterForward(const T* src, T* dst, Long dof) const;

  /** `src` `SortedCount()*dof` values, `dst` `LocalCount()*dof`, no overlap. */
  template <class T> void ScatterReverse(const T* src, T* dst, Long dof) const;

 private:
  Comm comm_;
  DevVec<Key> keys_;
  mutable detail_sortScatter::Plan<DevVec> plan_;  ///< inverses and stage-4 counts are built on first use
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
