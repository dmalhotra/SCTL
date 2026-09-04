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
 * The sorted order recorded as the stages that produced it, so a payload can be moved either way
 * without deriving anything:
 *
 *   1. a local rearrangement -- the sort of this rank's own keys
 *   2. one exchange           -- each key to the rank owning its stretch of the order
 *   3. a local rearrangement -- merging the arriving sorted runs into one block
 *   4. one exchange           -- present only once the partition has changed
 *
 * Stage 4 composes with itself: re-cutting an already sorted block twice is again one re-cut, so
 * it is recomputed from the fixed stage-3 layout rather than appended to.
 *
 * Neither exchange needs a map. Stage 1 leaves each destination's keys contiguous, and a re-cut of
 * a sorted block likewise, so the per-rank counts alone describe both moves; the payload reuses
 * those counts scaled by `dof`, and the reverse direction just swaps them.
 *
 * Each local stage is kept twice, as the map and its inverse, so both directions are gathers. A
 * scattered write costs about twice a scattered read -- it lands on part of a memory chunk and
 * forces a fetch, merge and store -- which at dof=3 and 100M keys is 42.5 ms against 19.7. The
 * inverses cost one scattered write of one index per key, paid once, on the first move back.
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
 * Keys sorted globally and cut at rank splitters, with the permutation to and from the order the
 * caller handed them in: the backend-side counterpart of `Comm::SortScatterIndex` with
 * `ScatterForward`/`ScatterReverse`. sctl keeps a global index per key and rebuilds the exchange on
 * every move; here the sort is recorded as the stages that produced it (`detail_sortScatter::Plan`)
 * and replayed, so a move is two local permutations and one exchange described by per-rank counts,
 * and the payload never leaves the backend memory.
 *
 * @tparam Key Needs `operator<`, trivial copyability, and `GetIntKey`/`FromIntKey` for the radix
 *             sort path (as `MortonCode` has).
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
   * the first key of rank r's range (`splitters[0]` is not consulted; rank 0 starts at the smallest
   * key). Replaces any earlier contents. Pass `std::move(keys)` to avoid the copy.
   */
  void Init(DeviceVector<Key> keys, const sctl::Vector<Key>& splitters);

  /** Move the sorted keys to the partition given by new `splitters`; the operators follow. */
  void Repartition(const sctl::Vector<Key>& splitters);

  const DeviceVector<Key>& SortedKeys() const { return keys_; }  ///< this rank's stretch of the global order
  Long LocalCount() const { return plan_.Nloc; }                 ///< keys the caller handed in
  Long SortedCount() const { return plan_.Ntree; }               ///< keys held now
  const Comm& GetComm() const { return comm_; }

  /**
   * Caller order -> sorted order, `dof` values per key. `data` holds `LocalCount()*dof` values on
   * entry and `SortedCount()*dof` on return. `dof` must agree across ranks.
   */
  template <class T> void ScatterForward(DeviceVector<T>& data, Long dof) const;

  /** Sorted order -> caller order: the inverse of `ScatterForward`. */
  template <class T> void ScatterReverse(DeviceVector<T>& data, Long dof) const;

  /**
   * Same, between raw buffers the caller has sized, so a payload can move straight into or out of
   * type-erased storage: `src` holds `LocalCount()*dof` values, `dst` `SortedCount()*dof`, no overlap.
   */
  template <class T> void ScatterForward(const T* src, T* dst, Long dof) const;

  /** `src` holds `SortedCount()*dof` values, `dst` `LocalCount()*dof`, no overlap. */
  template <class T> void ScatterReverse(const T* src, T* dst, Long dof) const;

 private:
  Comm comm_;
  DeviceVector<Key> keys_;
  mutable detail_sortScatter::Plan<DeviceVector> plan_;  ///< the inverses and stage-4 counts are built on first use
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_SORT_SCATTER_HPP_
