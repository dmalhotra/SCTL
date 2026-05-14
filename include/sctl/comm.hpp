#ifndef _SCTL_COMM_HPP_
#define _SCTL_COMM_HPP_

#include <functional>         // for less
#include <map>                // for multimap
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
   * Initialize MPI.
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
  explicit Comm(const MPI_Comm mpi_comm) { Init(mpi_comm); }
#endif

  /**
   * Duplicate communicator. The copy calls `MPI_Comm_dup` to obtain an
   * independent handle.
   */
  Comm(const Comm& c);

  /**
   * Move constructor. Steals `c`'s `MPI_Comm` handle and pending-request
   * pool; leaves `c` holding `MPI_COMM_NULL` and an empty pool, so its
   * destructor is a no-op.
   */
  Comm(Comm&& c) noexcept;

  /**
   * *self* communicator.
   */
  [[nodiscard]] static Comm Self();

  /**
   * *world* communicator.
   */
  [[nodiscard]] static Comm World();

  /**
   * Duplicate communicator (copy assignment, via `MPI_Comm_dup`).
   */
  Comm& operator=(const Comm& c);

  /**
   * Move assignment. Releases any handle/pool currently held by `*this`,
   * then steals `c`'s; `c` is left holding `MPI_COMM_NULL`.
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
  [[nodiscard]] const MPI_Comm& GetMPI_Comm() const noexcept { return mpi_comm_; }
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
   * Sparse all-to-all communication.
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
  template <class SType, class RType> [[nodiscard]] Request Ialltoallv_sparse(ConstIterator<SType> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls, Integer tag = 0) const;

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
  template <class Type, class Compare> void HyperQuickSort(const Vector<Type>& arr, Vector<Type>& SortedElem, Compare comp) const;

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
   * Structure to hold a pair of elements for sorting.
   */
  template <typename A, typename B> struct SortPair {
    int operator<(const SortPair<A, B>& p1) const { return key < p1.key; }
    A key;
    B data;
  };

#ifdef SCTL_HAVE_MPI
  /**
   * Initialize the communicator.
   *
   * @param[in] mpi_comm MPI communicator.
   */
  void Init(const MPI_Comm mpi_comm);

  template <class Type> static MPI_Op GetMPIOp(CommOp op);
  static void RegisterDatatype(MPI_Datatype datatype);
  static void RegisterOp(MPI_Op op);
  static void FreeRegisteredHandles();
  static std::vector<MPI_Datatype>& DatatypeRegistry();
  static std::vector<MPI_Op>& OpRegistry();

  Vector<MPI_Request>& NewReq(Long request_count) const;

  void DelReq(Vector<MPI_Request>* req_ptr) const;

  mutable std::stack<void*> req;

  int mpi_rank_;
  int mpi_size_;
  int mpi_tag_ub_;
  MPI_Comm mpi_comm_;

  template <class Type> class CommDatatype;

#else
  mutable std::multimap<Integer, ConstIterator<char>> send_req;
  mutable std::multimap<Integer, Iterator<char>> recv_req;
#endif
};

}  // end namespace

#endif // _SCTL_COMM_HPP_
