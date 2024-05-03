#ifndef _SCTL_COMM_HPP_
#define _SCTL_COMM_HPP_

#include <sctl/common.hpp>

#include <map>
#include <stack>
#ifdef SCTL_HAVE_MPI
#include <mpi.h>
#endif
#ifdef SCTL_HAVE_PETSC
#include <petscsys.h>
#endif

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;

/**
 * Object oriented wrapper to MPI. It uses MPI when SCTL_HAVE_MPI is defined,
 * otherwise, it defaults to the *self* communicator.
 */
class Comm {

 public:

  /**
   * Operation types for Allreduce and Scan collective operations.
   */
  enum class CommOp {
    SUM,
    MIN,
    MAX
  };

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
   * Duplicate communicator.
   */
  Comm(const Comm& c);

  /**
   * *self* communicator.
   */
  static Comm Self();

  /**
   * *world* communicator.
   */
  static Comm World();

  /**
   * Duplicate communicator.
   */
  Comm& operator=(const Comm& c);

  /**
   * Destructor.
   */
  ~Comm();

#ifdef SCTL_HAVE_MPI
  /**
   * Convert to MPI_Comm.
   */
  const MPI_Comm& GetMPI_Comm() const { return mpi_comm_; }
#endif

  /**
   * Split communicator.
   *
   * @param[in] clr indentify different communicator groups.
   */
  Comm Split(Integer clr) const;

  /**
   * @return rank of the current process.
   */
  Integer Rank() const;

  /**
   * @return size of this communicator.
   */
  Integer Size() const;

  /**
   * Synchronize all processes.
   */
  void Barrier() const;

  /**
   * Non-blocking send.
   *
   * @tparam SType type of the send-data.
   *
   * @param[in] sbuf const-iterator to the send bufffer.
   *
   * @param[in] scount number of elements to send.
   *
   * @param[in] dest the rank of the destination process.
   *
   * @param[in] tag identifier tag to be matched at receive.
   *
   * @return a context pointer for the request.
   */
  template <class SType> void* Isend(ConstIterator<SType> sbuf, Long scount, Integer dest, Integer tag = 0) const;

  /**
   * Non-blocking receive.
   *
   * @tparam RType type of the receive-data.
   *
   * @param[out] rbuf iterator to the receive bufffer.
   *
   * @param[in] rcount number of elements to receive.
   *
   * @param[in] source the rank of the source process.
   *
   * @param[in] tag identifier tag to be matched by the corresponding Isend.
   *
   * @return a context pointer for the request.
   */
  template <class RType> void* Irecv(Iterator<RType> rbuf, Long rcount, Integer source, Integer tag = 0) const;

  /**
   * Wait for a non-blocking send or receive.
   *
   * @param[in] req_ptr context pointer for the request.
   */
  void Wait(void* req_ptr) const;

  /**
   * Broadcast to all processed in the communicator.
   *
   * @tparam Type type of the data.
   *
   * \param[in,out] buff send-buffer on the sending process, or the receive buffer on the receiving process.
   *
   * \param[in] count number of elements in the mesage.
   *
   * \param[in] root rank of the sending process.
   */
  template <class Type> void Bcast(Iterator<Type> buf, Long count, Long root) const;

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
   * @return a context pointer for the request.
   */
  template <class SType, class RType> void* Ialltoallv_sparse(ConstIterator<SType> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls, Integer tag = 0) const;

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
  template <class Type> void Scan(ConstIterator<Type> sbuf, Iterator<Type> rbuf, int count, CommOp op) const;

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
   * @param[in] split_key optional key to determine the paritioning of the sorted array between processes.
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

  Vector<MPI_Request>* NewReq() const;

  void DelReq(Vector<MPI_Request>* req_ptr) const;

  mutable std::stack<void*> req;

  int mpi_rank_;
  int mpi_size_;
  MPI_Comm mpi_comm_;

  template <class Type> class CommDatatype;

#else
  mutable std::multimap<Integer, ConstIterator<char>> send_req;
  mutable std::multimap<Integer, Iterator<char>> recv_req;
#endif
};

}  // end namespace

#include SCTL_INCLUDE(comm.txx)

#endif  //_SCTL_COMM_HPP_
