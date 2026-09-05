#ifndef _SCTL_SORT_SCATTER_HPP_
#define _SCTL_SORT_SCATTER_HPP_

#include "sctl/common.hpp"    // for Long, Integer, sctl
#include "sctl/comm.hpp"      // for Comm
#include "sctl/comm.txx"      // for Comm::Self
#include "sctl/iterator.hpp"  // for Iterator, ConstIterator
#include "sctl/vector.hpp"    // for Vector

namespace sctl {

namespace sort_scatter_detail {

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
 * gathers; the inverses are built on the first move back.
 */
struct Plan {
  Long Nloc = 0;   ///< keys this rank was handed
  Long Nmid = 0;   ///< keys held after stages 1-3
  Long Ntree = 0;  ///< keys held now; differs from `Nmid` once the partition has changed

  Vector<Long> pre, pre_inv;    ///< Nloc: stage 1 and its inverse
  Vector<Long> scnt, rcnt;      ///< stage 2; unused at np=1
  Vector<Long> post, post_inv;  ///< Nmid: stage 3 and its inverse; unused at np=1

  bool recut = false;           ///< stage 4 present
  bool recut_cnt = false;       ///< stage-4 counts computed; done on the first move after a repartition
  Vector<Long> rscnt, rrcnt;    ///< stage 4

  bool inv = false;             ///< the inverses exist; built on the first move back
};

}  // namespace sort_scatter_detail

/**
 * Keys sorted globally and cut at rank splitters, with the permutation to and from the caller's
 * order: `Comm::SortScatterIndex` plus `ScatterForward`/`ScatterReverse` as an object. Those rebuild
 * the exchange from a global index on every move; here the sort's stages are recorded once
 * (`sort_scatter_detail::Plan`) and replayed, and the sorted keys are kept for reuse.
 *
 * @tparam Key Trivially copyable, ordered by `operator<`; radix-sorted when
 *             `omp_par::is_radix_sortable<Key>` holds (as for `MortonCode`).
 *
 * Every member that moves keys or data is collective.
 */
template <class Key> class SortScatter {
 public:
  explicit SortScatter(const Comm& comm = Comm::Self());

  /**
   * Sort `keys` (caller order) into the global order cut at `splitters`: np entries, `splitters[r]`
   * the first key of rank r's range; `splitters[0]` is not consulted.
   */
  void Init(const Vector<Key>& keys, const Vector<Key>& splitters);

  /** Move the sorted keys to the partition given by new `splitters`; the operators follow. */
  void Repartition(const Vector<Key>& splitters);

  /**
   * Move `data`, `dof` values per key in the layout before the last `Repartition` (its previous
   * `SortedCount()`), to the current layout, in place. A no-op when that `Repartition` moved nothing.
   */
  template <class T> void RepartitionData(Vector<T>& data, Long dof) const;

  const Vector<Key>& SortedKeys() const { return keys_; }  ///< this rank's stretch of the global order
  Long LocalCount() const { return plan_.Nloc; }           ///< keys the caller handed in
  Long SortedCount() const { return plan_.Ntree; }         ///< keys held now
  const Comm& GetComm() const { return comm_; }

  /** Caller order -> sorted order, `dof` values per key (agreed across ranks): `LocalCount()*dof` values in, `SortedCount()*dof` out. */
  template <class T> void ScatterForward(Vector<T>& data, Long dof) const;

  /** Sorted order -> caller order: the inverse of `ScatterForward`. */
  template <class T> void ScatterReverse(Vector<T>& data, Long dof) const;

  /** Same between caller-sized buffers: `src` `LocalCount()*dof` values, `dst` `SortedCount()*dof`, no overlap. */
  template <class T> void ScatterForward(ConstIterator<T> src, Iterator<T> dst, Long dof) const;

  /** `src` `SortedCount()*dof` values, `dst` `LocalCount()*dof`, no overlap. */
  template <class T> void ScatterReverse(ConstIterator<T> src, Iterator<T> dst, Long dof) const;

  /** Round trips through Init, Repartition and both scatters on Comm::World(); Key constructible from Long. */
  static void test();

 private:
  Comm comm_;
  Vector<Key> keys_;
  mutable sort_scatter_detail::Plan plan_;  ///< inverses and stage-4 counts are built on first use
  Vector<Long> move_scnt_, move_rcnt_;      ///< the last Repartition's move, previous layout -> current
  Long move_n_ = 0;                         ///< keys held before it
  bool moved_ = false;                      ///< whether it moved keys; RepartitionData is a no-op otherwise
};

}  // namespace sctl

#endif  // _SCTL_SORT_SCATTER_HPP_
