/**
 * @file sort-scatter.hpp
 * SortScatter: keys sorted globally and cut at rank splitters, with the permutation to and from the
 * order the caller handed them in (experimental).
 */

#ifndef _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
#define _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_

#include <vector>

#include "sctl/comm.hpp"
#include "sctl/vector.hpp"

namespace gpu_tree {

using sctl::Long;
using sctl::Comm;

namespace detail_sortScatter {

/**
 * The sorted order as the stages that produced it, replayed to move a payload either way:
 *
 *   1. local sort of this rank's keys      (`pre`)
 *   2. exchange to the owning ranks        (`scnt`/`rcnt`)
 *   3. local merge of the arriving runs    (`post`)
 *   4. re-cut to the current partition     (`rscnt`/`rrcnt`; only after a repartition)
 *
 * Counts alone describe both exchanges: stage 1 leaves each destination's keys contiguous, and a
 * re-cut of a sorted block is contiguous too. Stage 4 is always recomputed from the stage-3 layout,
 * since two re-cuts compose to one. Each local map is kept with its inverse so both directions are
 * gathers (a scattered write costs twice a scattered read: 42.5 vs 19.7 ms at 100M keys, dof=3);
 * the inverses are built on the first move back.
 */
template <template <class...> class DeviceVector> struct Plan {
  Long Nloc = 0;   ///< keys this rank was handed
  Long Nmid = 0;   ///< keys held after stages 1-3
  Long Ntree = 0;  ///< keys held now; differs from `Nmid` once the partition has changed

  DeviceVector<Long> pre, pre_inv;    ///< Nloc: stage 1 and its inverse
  sctl::Vector<Long> scnt, rcnt;      ///< stage 2; unused at np=1
  DeviceVector<Long> post, post_inv;  ///< Nmid: stage 3 and its inverse; unused at np=1

  bool recut = false;                 ///< stage 4 present
  bool recut_cnt = false;             ///< stage-4 counts computed; done on the first move after a repartition
  sctl::Vector<Long> rscnt, rrcnt;    ///< stage 4

  bool inv = false;                   ///< the inverses exist; built on the first move back
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
 * @tparam DeviceVector Backend container: `HostVector` or `thrust::device_vector`.
 *
 * Every member that moves keys or data is collective.
 */
template <class Key, template <class...> class DeviceVector = std::vector>
class SortScatter {
 public:
  explicit SortScatter(const Comm& comm = Comm::Self()) : comm_(comm) {}

  /**
   * Sort `keys` (caller order) into the global order cut at `splitters`: np entries, `splitters[r]`
   * the first key of rank r's range; `splitters[0]` is not consulted. Pass `std::move(keys)` to
   * avoid the copy.
   */
  void Init(DeviceVector<Key> keys, const sctl::Vector<Key>& splitters);

  /** Move the sorted keys to the partition given by new `splitters`; the operators follow. */
  void Repartition(const sctl::Vector<Key>& splitters);

  const DeviceVector<Key>& SortedKeys() const { return keys_; }  ///< this rank's stretch of the global order
  Long LocalCount() const { return plan_.Nloc; }                 ///< keys the caller handed in
  Long SortedCount() const { return plan_.Ntree; }               ///< keys held now
  const Comm& GetComm() const { return comm_; }

  /** Caller order -> sorted order, `dof` values per key (agreed across ranks): `LocalCount()*dof` values in, `SortedCount()*dof` out. */
  template <class T> void ScatterForward(DeviceVector<T>& data, Long dof) const;

  /** Sorted order -> caller order: the inverse of `ScatterForward`. */
  template <class T> void ScatterReverse(DeviceVector<T>& data, Long dof) const;

  /** Same between caller-sized raw buffers: `src` `LocalCount()*dof` values, `dst` `SortedCount()*dof`, no overlap. */
  template <class T> void ScatterForward(const T* src, T* dst, Long dof) const;

  /** `src` `SortedCount()*dof` values, `dst` `LocalCount()*dof`, no overlap. */
  template <class T> void ScatterReverse(const T* src, T* dst, Long dof) const;

 private:
  Comm comm_;
  DeviceVector<Key> keys_;
  mutable detail_sortScatter::Plan<DeviceVector> plan_;  ///< inverses and stage-4 counts are built on first use
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
