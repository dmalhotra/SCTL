#ifndef _SCTL_COMM_HPP_
#define _SCTL_COMM_HPP_

#include <algorithm>          // for lower_bound, sort
#include <functional>         // for less
#include <map>                // for multimap
#include <memory>             // for shared_ptr
#include <utility>            // for move
#include <vector>             // for vector

#include "sctl/common.hpp"    // for Long, Integer, sctl
#include "sctl/iterator.hpp"  // for ConstIterator, Iterator

#ifdef SCTL_HAVE_MPI
#include <mpi.h>
#include <stack>            // for stack
#endif
#ifdef SCTL_HAVE_PETSC
#include <petscsys.h>
#endif

namespace sctl {

template <class ValueType> class Vector;

/**
 * Operation types for Allreduce and Scan collective operations.
 */
enum class CommOp {
  SUM,
  MIN,
  MAX
};

/**
 * Object oriented wrapper to MPI. It uses MPI when compiled with `mpicxx` and the macro `SCTL_HAVE_MPI`
 * is defined, otherwise, it defaults to the *self* communicator.
 */
class Comm {

 public:

  /**
   * Initialize MPI, and ask the kernel once whether this process may read the memory of the ranks
   * sharing its node. That answer is a property of the process, so settling it here spares every
   * communicator a probe of its own -- and spares any routine that consults it from being
   * collective. A program that brings up MPI without this leaves the direct path off and sends
   * everything through MPI.
   */
  static void MPI_Init(int* argc, char*** argv);

  /**
   * Finalize MPI.
   */
  static void MPI_Finalize();

  /**
   * Default constructor, initializes to the *self* communicator.
   */
  Comm();

#ifdef SCTL_HAVE_MPI
  /**
   * Convert MPI_Comm to Comm.
   */
  explicit Comm(const MPI_Comm mpi_comm) : impl_(std::make_shared<Impl>()) { impl_->Init(mpi_comm); }
#endif

  /**
   * Copy constructor. Cheap reference-share of the underlying `Impl`
   * (shared_ptr increment) — no `MPI_Comm_dup`. The original
   * `MPI_Comm` is released only when the last copy is destroyed.
   */
  Comm(const Comm& c);

  /**
   * Move constructor. Transfers the `Impl` pointer; leaves `c` empty so
   * its destructor is a no-op.
   */
  Comm(Comm&& c) noexcept;

  /**
   * The *self* communicator, built on first use and shared thereafter.
   *
   * Cached because each call would otherwise duplicate a communicator, which is a finite resource,
   * and because this is the default argument of most classes here. Released by `MPI_Finalize`, so
   * nothing may hold the reference past that -- as nothing holding a `Comm` could anyway.
   */
  [[nodiscard]] static const Comm& Self();

  /**
   * The *world* communicator, built on first use and shared thereafter. As `Self()`.
   */
  [[nodiscard]] static const Comm& World();

  /**
   * Copy assignment. Reference-shares `c`'s underlying `Impl`. Releases
   * the prior `Impl` (which frees its `MPI_Comm` if this was the last ref).
   */
  Comm& operator=(const Comm& c);

  /**
   * Move assignment. Transfers `c`'s `Impl` pointer; `c` is left empty.
   */
  Comm& operator=(Comm&& c) noexcept;

  /**
   * Destructor.
   */
  ~Comm();

#ifdef SCTL_HAVE_MPI
  /**
   * Convert to MPI_Comm.
   */
  [[nodiscard]] const MPI_Comm& GetMPI_Comm() const noexcept { return impl_->mpi_comm_; }

  /**
   * The MPI datatype for `Type`: `sizeof(Type)` contiguous bytes, built on first use and freed at
   * finalize, so a caller counting in elements rather than bytes does not build one per call.
   *
   * The same handle the collectives here use. Exposed for code that reaches MPI directly with a
   * buffer this class does not own, as `gpu_tree`'s device exchanges do.
   */
  template <class Type> [[nodiscard]] static MPI_Datatype MPIDatatype();
#endif

  /**
   * Split communicator.
   *
   * @param[in] clr identify different communicator groups.
   */
  [[nodiscard]] Comm Split(Integer clr) const;

  /**
   * @return rank of the current process.
   */
  [[nodiscard]] Integer Rank() const noexcept;

  /**
   * @return size of this communicator.
   */
  [[nodiscard]] Integer Size() const noexcept;

  /**
   * Whether `rank` is a rank of this communicator running on this node -- this rank included, for
   * which it is trivially true. Not the same question as `rank == Rank()`: with ranks 0 to 3 on one
   * node, rank 0 sees `SameNode(2)` as true.
   *
   * Established when the communicator is built, so this is a local lookup. Reports the real
   * topology whatever the direct-read build flags say -- those govern reading a peer's memory, not
   * who shares the node. Without MPI there is one rank, itself.
   *
   * @param[in] rank A rank of this communicator.
   */
  [[nodiscard]] bool SameNode(Integer rank) const;

  /**
   * Synchronize all processes.
   */
  void Barrier() const;

  /**
   * Opaque, move-only handle for an outstanding non-blocking communication
   * request returned by Isend(), Irecv(), or Ialltoallv_sparse(). The handle
   * owns a pooled MPI_Request slot inside the `Comm`; that slot is released
   * when the handle is passed to Wait() exactly once.
   *
   * Lifetime contract:
   *   - The handle must be passed to Wait() (via `std::move`) before it goes
   *     out of scope. Dropping a non-empty Request would orphan the
   *     underlying MPI_Request; debug builds trap in the destructor.
   *   - After Wait() consumes a handle, the handle is empty and must not be
   *     reused. Empty handles (default-constructed or moved-from) are safe
   *     to destroy and may also be passed to Wait() as a no-op.
   *   - The handle is non-copyable; ownership transfers via move only.
   */
  class Request {
   public:
    Request() = default;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    Request& operator=(Request&& other) noexcept {
      SCTL_ASSERT_MSG(!ptr_, "Comm::Request: overwriting a non-empty request leaks the pending MPI_Request; Wait() must be called first");
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
      return *this;
    }
    ~Request() {
      SCTL_ASSERT_MSG(!ptr_, "Comm::Request destroyed without Wait(); the underlying MPI_Request is leaked");
    }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

   private:
    friend class Comm;
    explicit Request(void* p) noexcept : ptr_(p) {}
    void* release_() noexcept { void* p = ptr_; ptr_ = nullptr; return p; }
    void* ptr_ = nullptr;
  };

  /**
   * Non-blocking send.
   *
   * @tparam SType type of the send-data.
   *
   * @param[in] sbuf const-iterator to the send buffer.
   *
   * @param[in] scount number of elements to send.
   *
   * @param[in] dest the rank of the destination process.
   *
   * @param[in] tag identifier tag to be matched at receive.
   *
   * @return a Request handle. Must be passed to Wait() (via `std::move`)
   *         before going out of scope; otherwise the underlying MPI_Request
   *         is leaked. The return value is `[[nodiscard]]` — discarding it
   *         is a programmer error.
   */
  template <class SType> [[nodiscard]] Request Isend(ConstIterator<SType> sbuf, Long scount, Integer dest, Integer tag = 0) const;

  /**
   * Non-blocking synchronous send. Semantically equivalent to `Isend` except
   * that completion (via `Wait`) is delayed until the matching receive has
   * been posted at the destination — i.e. always uses the rendezvous protocol,
   * never the eager fast-path. Intended for the post-Irecvs-then-post-sends
   * pattern, where this synchronization is already satisfied by construction
   * and Issend's only practical effect is to bound unexpected-message-buffer
   * pressure on the receiver and to surface mismatched-receive bugs as
   * immediate deadlocks rather than silent buffering at scale.
   *
   * @tparam SType type of the send-data.
   *
   * @param[in] sbuf const-iterator to the send buffer.
   *
   * @param[in] scount number of elements to send.
   *
   * @param[in] dest the rank of the destination process.
   *
   * @param[in] tag identifier tag to be matched at receive.
   *
   * @return a Request handle. Same lifetime contract as Isend(): must be
   *         passed to Wait() before going out of scope.
   */
  template <class SType> [[nodiscard]] Request Issend(ConstIterator<SType> sbuf, Long scount, Integer dest, Integer tag = 0) const;

  /**
   * Blocking send, matched by `Recv` at the destination. Where the destination is a rank on this
   * node and the kernel permits it, the destination reads this buffer instead of being sent a copy
   * of it, and this returns once it has done so; otherwise this is `Issend` followed by `Wait`.
   *
   * Deadlocks if two ranks both send to each other before either receives, as `MPI_Send` does for
   * a message too large to buffer.
   *
   * @tparam SType type of the send-data.
   *
   * @param[in] sbuf const-iterator to the send buffer.
   *
   * @param[in] scount number of elements to send.
   *
   * @param[in] dest the rank of the destination process.
   *
   * @param[in] tag identifier tag to be matched at receive.
   */
  template <class SType> void Send(ConstIterator<SType> sbuf, Long scount, Integer dest, Integer tag = 0) const;

  /**
   * Blocking receive, matched by `Send` at the source. Where the source is a rank on this node and
   * the kernel permits it, this reads the source's send buffer instead of receiving a copy of it;
   * otherwise this is `Irecv` followed by `Wait`. As with MPI, `rcount` is an upper bound: what
   * arrives is what the source sent.
   *
   * @tparam RType type of the receive-data.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] rcount number of elements the buffer holds.
   *
   * @param[in] source the rank of the source process.
   *
   * @param[in] tag identifier tag to be matched by the corresponding Send.
   */
  template <class RType> void Recv(Iterator<RType> rbuf, Long rcount, Integer source, Integer tag = 0) const;

  /**
   * Non-blocking receive.
   *
   * @tparam RType type of the receive-data.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] rcount number of elements to receive.
   *
   * @param[in] source the rank of the source process.
   *
   * @param[in] tag identifier tag to be matched by the corresponding Isend.
   *
   * @return a Request handle. Same lifetime contract as Isend().
   */
  template <class RType> [[nodiscard]] Request Irecv(Iterator<RType> rbuf, Long rcount, Integer source, Integer tag = 0) const;

  /**
   * Wait for a non-blocking send or receive. Consumes the handle by value;
   * after the call, the moved-from variable in the caller is empty.
   *
   * @param[in] req Request handle returned by Isend(), Irecv(), or
   *                Ialltoallv_sparse(). May be empty (no-op).
   */
  void Wait(Request req) const;

  /**
   * Broadcast to all processes in the communicator.
   *
   * @tparam Type type of the data.
   *
   * @param[in,out] buff send-buffer on the sending process, or the receive buffer on the receiving process.
   *
   * @param[in] count number of elements in the message.
   *
   * @param[in] root rank of the sending process.
   */
  template <class Type> void Bcast(Iterator<Type> buf, Long count, Integer root) const;

  /**
   * Gather and concatenate equal size messages from all processes in the communicator.
   *
   * @tparam SType type of the send-data.
   * @tparam RType type of the receive-data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[in] scount number of elements in the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] rcount number of elements in the receive buffer. The total number of elements in the receive buffer
   * should be `rcount * Size()`.
   */
  template <class SType, class RType> void Allgather(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, Long rcount) const;

  /**
   * Gather and concatenate messages of different lengths from all processes in the communicator.
   *
   * @tparam SType type of the send-data.
   * @tparam RType type of the receive-data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[in] scount number of elements in the send buffer on each process.
   *
   * @param[out] rbuf iterator to the receive buffer where the gathered data is stored.
   *
   * @param[in] rcounts iterator to the number of elements to receive from each process.
   *
   * @param[in] rdispls iterator to the displacements in the receive buffer where the data from each process is stored.
   */
  template <class SType, class RType> void Allgatherv(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const;

  /**
   * Perform all-to-all operation for equal size messages.
   *
   * @tparam SType type of the send-data.
   * @tparam RType type of the receive-data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[in] scount number of elements in each send message. Size of send-buffer must be `scount * Size()`.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] rcount number of elements in each receive message. Size of receive-buffer must be `rcount * Size()`.
   */
  template <class SType, class RType> void Alltoall(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, Long rcount) const;

  /**
   * Sparse all-to-all communication. Collective. The payload of blocks that stay on the node never
   * goes through MPI: the self block is copied, and node peers' blocks are read straight out of
   * their send buffers (Linux; falls back to MPI where the kernel forbids it). The request covers
   * the rest.
   *
   * @tparam BlockingDirect Allow the node-local blocks to be read out of their peers' send buffers.
   * That cannot be done without synchronizing the node before returning -- no rank may leave while
   * a peer is still reading its send buffer -- so it costs this routine the non-blocking behaviour
   * its name promises, and it is off by default. Ask for it only where the request is waited on
   * straight away. It is a template parameter because every rank must make the same choice: as a
   * runtime argument a rank could work one out for itself, and a node whose ranks disagreed would
   * have some of them synchronizing it and the others leaving. `Alltoallv` blocks anyway and so
   * reads node peers without being asked.
   *
   * @note Collective. With `BlockingDirect`, also not fully non-blocking: the node-local part of
   * the exchange is complete when the call returns and only the rest is left for `Wait`.
   *
   * @note Reading a peer's memory needs ptrace permission, which `Comm::MPI_Init` asks the kernel
   * about once for the process. By default nothing is done to obtain it: where the kernel already
   * permits the reads (yama `ptrace_scope` 0, as on a node a job owns) the direct path is used,
   * and where it does not the probe fails and everything goes through MPI -- as it also does for a
   * program that brings up MPI without `Comm::MPI_Init`. Building with `-DSCTL_COMM_PTRACER` lets
   * the ranks widen it for themselves with `PR_SET_PTRACER_ANY`, which opens their address space
   * to every process of the same user for the rest of their lifetime -- do not do that on a shared
   * node. `-DSCTL_COMM_NO_DIRECT` turns the direct path off outright. A read the kernel refuses
   * after the probe has passed sends that one exchange through MPI instead; it is not remembered,
   * since a permission that changed under a running job is not something to carry a flag for.
   *
   * @tparam SType type of the send-data.
   * @tparam RType type of the receive-data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[in] scounts iterator to the number of elements to send to each process.
   *
   * @param[in] sdispls iterator to the displacements in the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] rcounts iterator to the number of elements to receive from each process.
   *
   * @param[in] rdispls iterator to the displacements in the receive buffer.
   *
   * @param[in] tag identifier tag to be matched by all processes in the communicator.
   *
   * @return a Request handle. Same lifetime contract as Isend(): must be
   *         passed to Wait() before destruction.
   */
  template <bool BlockingDirect = false, class SType, class RType> [[nodiscard]] Request Ialltoallv_sparse(ConstIterator<SType> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls, Integer tag = 0) const;

  /**
   * All-to-all communication with varying send and receive counts and displacements.
   *
   * @tparam Type type of the data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[in] scounts iterator to the number of elements to send to each process.
   *
   * @param[in] sdispls iterator to the displacements in the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] rcounts iterator to the number of elements to receive from each process.
   *
   * @param[in] rdispls iterator to the displacements in the receive buffer.
   */
  template <class Type> void Alltoallv(ConstIterator<Type> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<Type> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const;

  /**
   * All-to-all communication via recursive divide-and-conquer (bitonic
   * split-exchange), ported from pvfmm's `par::Mpi_Alltoallv_dense`.
   *
   * Recursively halves the rank group and performs a single `MPI_Sendrecv`
   * payload exchange between matched halves at each level, rearranging the
   * combined buffer in-rank between iterations. O(log p) iterations,
   * O(log p) intermediate buffers, O(log p · N_local) byte movement.
   *
   * Functionally equivalent to `Alltoallv`, but uses the hand-rolled
   * algorithm rather than the vendor `MPI_Alltoallv`. Useful when the
   * vendor implementation is known to be inferior at the relevant
   * (p, message_size) shape — typically a niche case.
   *
   * @tparam Type trivially-copyable payload type.
   *
   * @param[in]  sbuf    iterator to the send buffer.
   * @param[in]  scounts per-rank send counts (length `Size()`).
   * @param[in]  sdispls per-rank send-buffer displacements (length `Size()`).
   * @param[out] rbuf    iterator to the receive buffer.
   * @param[in]  rcounts per-rank receive counts (length `Size()`).
   * @param[in]  rdispls per-rank receive-buffer displacements (length `Size()`).
   */
  template <class Type> void Alltoallv_dense(ConstIterator<Type> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<Type> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const;

  /**
   * Perform an all-reduce operation.
   *
   * @tparam Type type of the data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] count number of elements.
   *
   * @param[in] op reduction operation.
   */
  template <class Type> void Allreduce(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, CommOp op) const;

  /**
   * All-reduce with the reduction op fixed at compile time. Unlike the runtime-`op` overload, only
   * the selected op's reduction is instantiated, so it works for types that define just the needed
   * comparison (e.g. a type with only `operator<` for `CommOp::MIN`).
   *
   * @tparam op reduction operation (compile-time).
   * @tparam Type type of the data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] count number of elements.
   */
  template <CommOp op, class Type> void Allreduce(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count) const;

  /**
   * Perform a scan operation.
   *
   * @tparam Type type of the data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] count number of elements.
   *
   * @param[in] op scan operation.
   */
  template <class Type> void Scan(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, CommOp op) const;

  /**
   * Scan with the reduction op fixed at compile time. Unlike the runtime-`op` overload, only the
   * selected op's reduction is instantiated, so it works for types that define just the needed
   * comparison (e.g. a type with only `operator<` for `CommOp::MIN`).
   *
   * @tparam op scan operation (compile-time).
   * @tparam Type type of the data.
   *
   * @param[in] sbuf iterator to the send buffer.
   *
   * @param[out] rbuf iterator to the receive buffer.
   *
   * @param[in] count number of elements.
   */
  template <CommOp op, class Type> void Scan(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count) const;

  /**
   * Perform a weighted partitioning of a vector.
   *
   * @tparam Type type of the vector elements.
   *
   * @param[in,out] nodeList vector to partition.
   *
   * @param[in] wts_ optional weights for weighted partitioning.
   */
  template <class Type> void PartitionW(Vector<Type>& nodeList, const Vector<Long>* wts_ = nullptr) const;

  /**
   * Partition a vector given the number of local elements after partitioning.
   *
   * @tparam Type type of the vector elements.
   *
   * @param[in,out] v vector to partition.
   *
   * @param[in] N number of local elements after partitioning.
   */
  template <class Type> void PartitionN(Vector<Type>& v, Long N) const;

  /**
   * Perform a partitioning of a vector with a splitter element.
   *
   * @tparam Type type of the vector elements.
   * @tparam Compare comparison function type.
   *
   * @param[in,out] nodeList vector to partition.
   *
   * @param[in] splitter element to partition around.
   *
   * @param[in] comp comparison function for elements.
   */
  template <class Type, class Compare> void PartitionS(Vector<Type>& nodeList, const Type& splitter, Compare comp) const;

  /**
   * Perform a partitioning of a vector with a splitter element using the default comparison function.
   *
   * @tparam Type type of the vector elements.
   *
   * @param[in,out] nodeList vector to partition.
   *
   * @param[in] splitter element to partition around.
   */
  template <class Type> void PartitionS(Vector<Type>& nodeList, const Type& splitter) const {
    PartitionS(nodeList, splitter, std::less<Type>());
  }

  /**
   * Sorts the elements of an array using HyperQuickSort algorithm with a custom comparison function.
   *
   * @tparam Type type of the elements in the array.
   * @tparam Compare comparison function type.
   *
   * @param[in] arr input array to be sorted.
   *
   * @param[out] SortedElem sorted array.
   *
   * @param[in] comp comparison function for elements.
   */
  template <class Type, class Compare> void HyperQuickSort(const Vector<Type>& arr, Vector<Type>& SortedElem, Compare comp, bool partition = true) const;

  /**
   * Sorts the elements of an array using HyperQuickSort algorithm with the default comparison function.
   *
   * @tparam Type type of the elements in the array.
   *
   * @param[in] arr input array to be sorted.
   *
   * @param[out] SortedElem sorted array.
   */
  template <class Type> void HyperQuickSort(const Vector<Type>& arr, Vector<Type>& SortedElem) const {
    HyperQuickSort(arr, SortedElem, std::less<Type>());
  }

  /**
   * Sorts the elements of a distributed array using a single-pass sample sort (regular
   * sampling -> one Alltoallv -> k-way merge). Scales better than HyperQuickSort for large
   * arrays by avoiding its O(log p) merge/comm-split rounds.
   *
   * @tparam Type type of the elements in the array.
   * @tparam Compare comparison functor type.
   *
   * @param[in] arr input array to be sorted.
   * @param[out] SortedElem sorted array.
   * @param[in] comp comparison function for elements.
   * @param[in] partition if true (default) the output is equally partitioned across ranks
   *   (via PartitionW), matching HyperQuickSort; if false the output is globally sorted but
   *   only approximately balanced (skip when a PartitionS/repartition follows).
   */
  template <class Type, class Compare> void SampleSort(const Vector<Type>& arr, Vector<Type>& SortedElem, Compare comp, bool partition = true) const;

  /**
   * Sorts the elements of a distributed array using a single-pass sample sort with the
   * default comparison function.
   */
  template <class Type> void SampleSort(const Vector<Type>& arr, Vector<Type>& SortedElem) const {
    SampleSort(arr, SortedElem, std::less<Type>());
  }

  /**
   * Distributed sort using a caller-provided per-rank splitter (like PartitionS): each rank
   * supplies its own lower boundary, gathered internally, and rank r ends up with the
   * globally-sorted elements in [splitter_r, splitter_{r+1}). Skips splitter selection and
   * PartitionW, sorting and partitioning in one pass when the boundaries are already known.
   *
   * @tparam Type type of the elements in the array.
   * @tparam Compare comparison functor type.
   *
   * @param[in] arr input array to be sorted.
   * @param[out] SortedElem sorted, splitter-partitioned array.
   * @param[in] splitter this rank's lower boundary key (splitters must be ascending by rank).
   * @param[in] comp comparison function for elements.
   */
  template <class Type, class Compare> void SampleSort(const Vector<Type>& arr, Vector<Type>& SortedElem, const Type& splitter, Compare comp) const;

  /**
   * Distributed sort using a per-rank splitter and the default comparison function.
   */
  template <class Type> void SampleSort(const Vector<Type>& arr, Vector<Type>& SortedElem, const Type& splitter) const {
    SampleSort(arr, SortedElem, splitter, std::less<Type>());
  }

  /**
   * Generates scatter indices corresponding to a sorted array.
   *
   * @tparam Type type of the elements in the array.
   *
   * @param[in] key array of keys to be sorted.
   *
   * @param[out] scatter_index array of indices giving the original global-position of each element in the sorted array.
   *
   * @param[in] split_key optional key to determine the partitioning of the sorted array between processes.
   */
  template <class Type> void SortScatterIndex(const Vector<Type>& key, Vector<Long>& scatter_index, const Type* split_key = nullptr) const;

  /**
   * Scatter data elements forward (i.e. sorted to unsorted order) using the provided scatter index.
   *
   * @tparam Type type of the data elements.
   *
   * @param[in,out] data_ data elements to be scattered.
   *
   * @param[in] scatter_index array of indices giving the original global-position of each element in the sorted array.
   */
  template <class Type> void ScatterForward(Vector<Type>& data_, const Vector<Long>& scatter_index) const;

  /**
   * Scatter data elements in reverse (i.e. unsorted to sorted order) using the provided scatter index.
   *
   * @tparam Type type of the data elements.
   *
   * @param[in,out] data_ data elements to be scattered.
   *
   * @param[in] scatter_index array of indices giving the original global-position of each element in the sorted array.
   *
   * @param[in] loc_size_ number of local element after rearrangement.
   */
  template <class Type> void ScatterReverse(Vector<Type>& data_, const Vector<Long>& scatter_index_, Long loc_size_ = 0) const;

 private:

  /**
   * Core of SampleSort: given a locally-sorted array `loc` and this rank's lower boundary
   * `splitter` (one value per rank, gathered internally so the split is always consistent),
   * redistribute (one Alltoallv) and parallel-merge so rank r ends up with the globally-sorted
   * elements in [splitter_r, splitter_{r+1}).
   */
  template <class Type, class Compare> void DistributeAndMerge(const Vector<Type>& loc, const Type& splitter, Vector<Type>& SortedElem, Compare comp) const;

  /**
   * Determine this process's lower-boundary splitter for a load-balanced distributed sort via
   * iterative exact-rank histogramming. Generic (uses only `comp` and actual elements), O(npes)
   * communication per round; the resulting balance is independent of the data distribution.
   *
   * @param[in] loc locally-sorted elements on this process.
   * @param[in] totSize total element count across all processes.
   * @param[in] comp comparison functor.
   * @return the value at global rank Rank()*totSize/npes (rank 0's return value is unused).
   */
  template <class Type, class Compare> Type DetermineSplitter(const Vector<Type>& loc, Long totSize, Compare comp) const;

#ifdef SCTL_HAVE_MPI
  /**
   * Internal reference-counted state for the duplicated `MPI_Comm` and
   * its per-comm `MPI_Request` pool. Held via `shared_ptr<Impl>` so that
   * `Comm` copy/assignment are O(1) and the underlying `MPI_Comm_free`
   * fires exactly once when the last `Comm` referencing it is destroyed.
   */
  struct Impl {
    int mpi_rank_;
    int mpi_size_;
    int mpi_tag_ub_;
    MPI_Comm mpi_comm_;
    mutable std::stack<void*> req;

    // The ranks of this communicator sharing this node, whose memory this rank may be able to read
    // directly (Linux process_vm_readv) instead of receiving through MPI. Built with the
    // communicator, so one made from an MPI_Comm and one made by Split each get their own.
    //
    // Held as the node's comm ranks in ascending order and searched, rather than as a table indexed
    // by comm rank: that table would be one entry per rank of the communicator -- 7.6 MB each at a
    // million ranks -- to carry node_size useful entries.
    MPI_Comm node_comm_ = MPI_COMM_NULL;
    std::vector<int> node_rank_;         ///< comm ranks on this node, ascending
    std::vector<int> node_pid_;          ///< their pids, in the same order

    // Whether node peers' memory may be read on this communicator. Whether the kernel permits it is
    // a separate question from who is on the node, and one this communicator does not ask:
    // `Comm::MPI_Init` settles it once for the process, over the whole node. Written by `InitNode`
    // and never again, so every rank of a node holds the same value and point-to-point code may
    // read it as freely as a collective.
    bool direct_ = false;

    /** Position of `rank` in `node_rank_`, or -1 when `rank` is not a rank on this node. */
    Integer NodeIdx(Integer rank) const {
      const auto i = std::lower_bound(node_rank_.begin(), node_rank_.end(), (int)rank);
      if (i == node_rank_.end() || *i != (int)rank) return -1;
      return (Integer)(i - node_rank_.begin());
    }

    Impl();
    ~Impl();

    Impl(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl& operator=(Impl&&) = delete;

    /**
     * Initialize the impl by duplicating the given MPI_Comm.
     * Collective on the input communicator.
     */
    void Init(MPI_Comm mpi_comm);

    /** Find the ranks sharing this node, and take the process's direct-read answer. Called by
     *  `Init`; collective on the communicator. */
    void InitNode();
  };

  template <class Type> static MPI_Op GetMPIOp(CommOp op);
  template <CommOp op, class Type> static MPI_Op GetMPIOp();
  template <class Type> void AllreduceImpl(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, MPI_Op mpi_op) const;
  template <class Type> void ScanImpl(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, MPI_Op mpi_op) const;
  static void RegisterDatatype(MPI_Datatype datatype);
  static void RegisterOp(MPI_Op op);
  static void FreeRegisteredHandles();
  static std::vector<MPI_Datatype>& DatatypeRegistry();
  static std::vector<MPI_Op>& OpRegistry();

#ifdef SCTL_HAVE_MPI
  /**
   * Read every node peer's block for this rank straight out of that peer's send buffer.
   * `scounts`/`sdispls` locate this rank's block for each peer, so each peer can be told where to
   * read from; `rcounts`/`rdispls` where each peer's block for this rank goes.
   *
   * Blocking and collective on the node: it returns only once every peer has finished reading this
   * rank's send buffer, which is what makes the reads safe. Callers must have faulted in the
   * receive blocks already -- doing so afterwards would overwrite what was read.
   *
   * @return false if any read on this node was refused, in which case nothing was delivered and the
   * direct path is given up for the rest of the run.
   */
  template <class SType, class RType>
  bool ReadNodeBlocks(ConstIterator<SType> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls,
                      Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const;
#endif

  Vector<MPI_Request>& NewReq(Long request_count) const;

  void DelReq(Vector<MPI_Request>* req_ptr) const;

  std::shared_ptr<Impl> impl_;

  template <class Type> class CommDatatype;

#else
  mutable std::multimap<Integer, ConstIterator<char>> send_req;
  mutable std::multimap<Integer, Iterator<char>> recv_req;
#endif
};

}  // end namespace

#endif // _SCTL_COMM_HPP_
