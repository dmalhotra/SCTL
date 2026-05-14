#ifndef _SCTL_COMM_TXX_
#define _SCTL_COMM_TXX_

#include <algorithm>              // for lower_bound, max, min, sort, upper_...
#include <cassert>                // for assert
#include <functional>             // for less
#include <limits>                 // for numeric_limits
#include <map>                    // for multimap, __map_iterator, operator==
#include <type_traits>            // for is_trivially_copyable
#include <utility>                // for pair
#include <vector>                 // for vector

#include "sctl/common.hpp"        // for Long, Integer, SCTL_ASSERT, SCTL_UN...
#include "sctl/comm.hpp"          // for Comm, CommOp
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for Iterator::Iterator<ValueType>, Iter...
#include "sctl/ompUtils.txx"      // for scan, merge_sort
#include "sctl/static-array.hpp"  // for StaticArray
#include "sctl/static-array.txx"  // for StaticArray::operator[], StaticArra...
#include "sctl/vector.hpp"        // for Vector
#include "sctl/vector.txx"        // for Vector::operator[], Vector::begin

namespace sctl {

namespace comm_detail {

template <class IteratorType> inline void TouchBuffer(IteratorType buf, Long count) {
  if (!count) return;
  SCTL_UNUSED(buf[0]        );
  SCTL_UNUSED(buf[count - 1]);
}

}  // namespace comm_detail

#ifdef SCTL_HAVE_MPI
static_assert(MPI_VERSION >= 3, "SCTL requires MPI_VERSION >= 3 when SCTL_HAVE_MPI is enabled");
namespace comm_detail {

#ifdef SCTL_COMM_MAX_CHUNKS
static_assert(SCTL_COMM_MAX_CHUNKS > 0, "SCTL_COMM_MAX_CHUNKS must be positive");
static_assert(SCTL_COMM_MAX_CHUNKS <= std::numeric_limits<int>::max(), "SCTL_COMM_MAX_CHUNKS must fit in int");
#endif
#ifdef SCTL_MPI_COUNT_LIMIT
static_assert(SCTL_MPI_COUNT_LIMIT > 0, "SCTL_MPI_COUNT_LIMIT must be positive");
static_assert(SCTL_MPI_COUNT_LIMIT <= std::numeric_limits<int>::max(), "SCTL_MPI_COUNT_LIMIT must fit in int");
#endif

constexpr Long MPIIntLimit() {
#ifdef SCTL_MPI_COUNT_LIMIT
  return static_cast<Long>(SCTL_MPI_COUNT_LIMIT);
#else
  return static_cast<Long>(std::numeric_limits<int>::max());
#endif
}

constexpr Long MPIIntMax() {
  return static_cast<Long>(std::numeric_limits<int>::max());
}

constexpr bool MPIHasLargeCount() {
#if MPI_VERSION >= 4
  return true;
#else
  return false;
#endif
}

constexpr Long MPIMaxChunks() {
#ifdef SCTL_COMM_MAX_CHUNKS
  return static_cast<Long>(SCTL_COMM_MAX_CHUNKS);
#else
  return 1000;
#endif
}

inline bool MPIIsActive() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) return false;
  int finalized = 0;
  MPI_Finalized(&finalized);
  return !finalized;
}

inline void WarnIfMPIInactive(const char* op_name) {
  if (!MPIIsActive()) SCTL_WARN(std::string("MPI operation called while MPI is inactive: ") + op_name);
}

constexpr Long GCD(Long a, Long b) {
  return b ? GCD(b, a % b) : a;
}

constexpr bool MPIFitsInt(Long value) {
  return value >= 0 && value <= MPIIntMax();
}

inline int MPIAsInt(Long value) {
  SCTL_ASSERT(MPIFitsInt(value));
  return static_cast<int>(value);
}

constexpr bool MPIFitsCount(Long value) {
  return value >= 0 && value <= MPIIntLimit();
}

inline int MPIAsCount(Long value) {
  SCTL_ASSERT(MPIFitsCount(value));
  return static_cast<int>(value);
}

inline MPI_Count MPIAsCountLarge(Long value) {
  SCTL_ASSERT(value >= 0);
  return static_cast<MPI_Count>(value);
}

inline MPI_Aint MPIAsAint(Long value) {
  SCTL_ASSERT(value >= 0);
  return static_cast<MPI_Aint>(value);
}

constexpr Long MPINumChunks(Long value) {
  return value ? (value - 1) / MPIIntLimit() + 1 : 0;
}

constexpr Long MPIChunkTagBaseUnchecked(Long user_tag) {
  return user_tag * MPIMaxChunks();
}

constexpr Long MPIChunkTagUnchecked(Long user_tag, Long chunk_idx) {
  return MPIChunkTagBaseUnchecked(user_tag) + chunk_idx;
}

inline void TrackPointToPoint(Long request_count, Long total_bytes) {
  Profile::IncrementCounter(ProfileCounter::PROF_MPI_COUNT, request_count);
  Profile::IncrementCounter(ProfileCounter::PROF_MPI_BYTES, total_bytes);
}

inline void TrackCollective(Long collective_count, Long total_bytes) {
  Profile::IncrementCounter(ProfileCounter::PROF_MPI_COLLECTIVE_COUNT, collective_count);
  Profile::IncrementCounter(ProfileCounter::PROF_MPI_COLLECTIVE_BYTES, total_bytes);
}

inline Long MPIChunkTagBase(Long user_tag) {
  SCTL_ASSERT(user_tag >= 0);
  SCTL_ASSERT(user_tag <= MPIIntMax() / MPIMaxChunks());
  return MPIChunkTagBaseUnchecked(user_tag);
}

inline int MPIChunkTag(Long user_tag, Long chunk_idx) {
  SCTL_ASSERT(chunk_idx >= 0);
  SCTL_ASSERT(chunk_idx < MPIMaxChunks());
  return MPIAsInt(MPIChunkTagUnchecked(user_tag, chunk_idx));
}

inline void AssertChunkedTagRange(Long user_tag, Long chunk_count, int mpi_tag_ub) {
  SCTL_ASSERT(user_tag >= 0);
  SCTL_ASSERT(chunk_count <= MPIMaxChunks());
  SCTL_ASSERT(MPIChunkTagBase(user_tag) + MPIMaxChunks() - 1 <= static_cast<Long>(mpi_tag_ub));
}

inline void MPIWaitAllBatched(MPI_Request* request, Long request_count) {
  for (Long offset = 0; offset < request_count; offset += MPIIntMax()) {
    const Long batch = std::min<Long>(request_count - offset, MPIIntMax());
    const int err = MPI_Waitall(MPIAsInt(batch), request + offset, MPI_STATUSES_IGNORE);
    SCTL_ASSERT(err == MPI_SUCCESS);
  }
}

}  // namespace comm_detail

/**
 * An abstract class used for communicating messages using user-defined
 * datatypes. The user must implement the static member function "value()" that
 * returns the MPI_Datatype corresponding to this user-defined datatype.
 * \author Hari Sundar, hsundar@gmail.com
 */
template <class Type> class Comm::CommDatatype {
 public:
  static MPI_Datatype value() {
    static MPI_Datatype datatype = [](){
      MPI_Datatype datatype;
      MPI_Type_contiguous(sizeof(Type), MPI_BYTE, &datatype);
      MPI_Type_commit(&datatype);
      Comm::RegisterDatatype(datatype);
      return datatype;
    }();
    return datatype;
  }

  static MPI_Op sum() {
    static MPI_Op myop = [](){
      MPI_Op myop;
      int commune = 1;
      MPI_Op_create(sum_fn, commune, &myop);
      Comm::RegisterOp(myop);
      return myop;
    }();
    return myop;
  }

  static MPI_Op min() {
    static MPI_Op myop = [](){
      MPI_Op myop;
      int commune = 1;
      MPI_Op_create(min_fn, commune, &myop);
      Comm::RegisterOp(myop);
      return myop;
    }();
    return myop;
  }

  static MPI_Op max() {
    static MPI_Op myop = [](){
      MPI_Op myop;
      int commune = 1;
      MPI_Op_create(max_fn, commune, &myop);
      Comm::RegisterOp(myop);
      return myop;
    }();
    return myop;
  }

 private:
  static void sum_fn(void* a_, void* b_, int* len_, MPI_Datatype* datatype) {
    Type* a = (Type*)a_;
    Type* b = (Type*)b_;
    int len = *len_;
    for (int i = 0; i < len; i++) {
      b[i] = a[i] + b[i];
    }
  }

  static void min_fn(void* a_, void* b_, int* len_, MPI_Datatype* datatype) {
    Type* a = (Type*)a_;
    Type* b = (Type*)b_;
    int len = *len_;
    for (int i = 0; i < len; i++) {
      if (a[i] < b[i]) b[i] = a[i];
    }
  }

  static void max_fn(void* a_, void* b_, int* len_, MPI_Datatype* datatype) {
    Type* a = (Type*)a_;
    Type* b = (Type*)b_;
    int len = *len_;
    for (int i = 0; i < len; i++) {
      if (a[i] > b[i]) b[i] = a[i];
    }
  }
};
#endif

inline void Comm::MPI_Init(int* argc, char*** argv) {
#ifdef SCTL_HAVE_PETSC
  PetscInitialize(argc, argv, NULL, NULL);
#elif defined(SCTL_HAVE_MPI)
  int provided;
  ::MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided < MPI_THREAD_SERIALIZED) SCTL_WARN("MPI implementation does not support MPI_THREAD_SERIALIZED.");
#endif
}

inline void Comm::MPI_Finalize() {
#ifdef SCTL_HAVE_MPI
  if (comm_detail::MPIIsActive()) FreeRegisteredHandles();
#endif
#ifdef SCTL_HAVE_PETSC
  PetscFinalize();
#elif defined(SCTL_HAVE_MPI)
  ::MPI_Finalize();
#endif
}


inline Comm::Comm() {
#ifdef SCTL_HAVE_MPI
  Init(MPI_COMM_SELF);
#endif
}

inline Comm::Comm(const Comm& c) {
#ifdef SCTL_HAVE_MPI
  Init(c.mpi_comm_);
#endif
}

inline Comm::Comm(Comm&& c) noexcept {
#ifdef SCTL_HAVE_MPI
  mpi_rank_ = c.mpi_rank_;
  mpi_size_ = c.mpi_size_;
  mpi_tag_ub_ = c.mpi_tag_ub_;
  mpi_comm_ = c.mpi_comm_;
  req = std::move(c.req);

  c.mpi_comm_ = MPI_COMM_NULL;
  c.mpi_rank_ = 0;
  c.mpi_size_ = 1;
  c.mpi_tag_ub_ = std::numeric_limits<int>::max();
#else
  send_req = std::move(c.send_req);
  recv_req = std::move(c.recv_req);
#endif
}

inline Comm Comm::Self() {
#ifdef SCTL_HAVE_MPI
  return Comm(MPI_COMM_SELF);
#else
  return Comm();
#endif
}

inline Comm Comm::World() {
#ifdef SCTL_HAVE_MPI
  return Comm(MPI_COMM_WORLD);
#else
  return Comm();
#endif
}

inline Comm& Comm::operator=(const Comm& c) {
#ifdef SCTL_HAVE_MPI
  if (this == &c) return *this;
  if (comm_detail::MPIIsActive()) {
    #pragma omp critical(SCTL_COMM_DUP)
    if (mpi_comm_ != MPI_COMM_NULL) MPI_Comm_free(&mpi_comm_);
    Init(c.mpi_comm_);
  } else {
    SCTL_WARN("Comm::operator= called while MPI is inactive; resetting to MPI_COMM_NULL.");
    mpi_rank_ = 0;
    mpi_size_ = 1;
    mpi_tag_ub_ = std::numeric_limits<int>::max();
    mpi_comm_ = MPI_COMM_NULL;
  }
#endif
  return *this;
}

inline Comm& Comm::operator=(Comm&& c) noexcept {
  if (this == &c) return *this;
#ifdef SCTL_HAVE_MPI
  #pragma omp critical(SCTL_COMM_REQ)
  while (!req.empty()) {
    delete (Vector<MPI_Request>*)req.top();
    req.pop();
  }
  if (comm_detail::MPIIsActive() && mpi_comm_ != MPI_COMM_NULL) {
    #pragma omp critical(SCTL_COMM_DUP)
    MPI_Comm_free(&mpi_comm_);
  }
  mpi_rank_ = c.mpi_rank_;
  mpi_size_ = c.mpi_size_;
  mpi_tag_ub_ = c.mpi_tag_ub_;
  mpi_comm_ = c.mpi_comm_;
  req = std::move(c.req);

  c.mpi_comm_ = MPI_COMM_NULL;
  c.mpi_rank_ = 0;
  c.mpi_size_ = 1;
  c.mpi_tag_ub_ = std::numeric_limits<int>::max();
#else
  send_req = std::move(c.send_req);
  recv_req = std::move(c.recv_req);
#endif
  return *this;
}

inline Comm::~Comm() {
#ifdef SCTL_HAVE_MPI
  #pragma omp critical(SCTL_COMM_REQ)
  while (!req.empty()) {
    delete (Vector<MPI_Request>*)req.top();
    req.pop();
  }
  if (comm_detail::MPIIsActive()) {
    #pragma omp critical(SCTL_COMM_DUP)
    if (mpi_comm_ != MPI_COMM_NULL) MPI_Comm_free(&mpi_comm_);
  }
#endif
}

inline Comm Comm::Split(Integer clr) const {
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Split");
  MPI_Comm new_comm;
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_split(mpi_comm_, clr, mpi_rank_, &new_comm);
  Comm c(new_comm);
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_free(&new_comm);
  return c;
#else
  return Comm();
#endif
}

inline Integer Comm::Rank() const noexcept {
#ifdef SCTL_HAVE_MPI
  return mpi_rank_;
#else
  return 0;
#endif
}

inline Integer Comm::Size() const noexcept {
#ifdef SCTL_HAVE_MPI
  return mpi_size_;
#else
  return 1;
#endif
}

inline void Comm::Barrier() const {
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Barrier");
  MPI_Barrier(mpi_comm_);
#endif
}

template <class SType> Comm::Request Comm::Isend(ConstIterator<SType> sbuf, Long scount, Integer dest, Integer tag) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!scount) return Request();
  comm_detail::WarnIfMPIInactive("Comm::Isend");
#if MPI_VERSION >= 4
  Vector<MPI_Request>& request = NewReq(1);
  comm_detail::TouchBuffer(sbuf, scount);
  comm_detail::TrackPointToPoint(1, scount * sizeof(SType));
  MPI_Isend_c(&sbuf[0], comm_detail::MPIAsCountLarge(scount), CommDatatype<SType>::value(), dest, tag, mpi_comm_, &request[0]);
  return Request(&request);
#else
  const Long request_count = comm_detail::MPINumChunks(scount);
  comm_detail::AssertChunkedTagRange(tag, request_count, mpi_tag_ub_);
  Vector<MPI_Request>& request = NewReq(request_count);
  comm_detail::TouchBuffer(sbuf, scount);
  comm_detail::TrackPointToPoint(request_count, scount * sizeof(SType));
  Long offset = 0;
  for (Long i = 0; i < request_count; i++) {
    const Long chunk = std::min<Long>(scount - offset, comm_detail::MPIIntLimit());
    MPI_Isend(&sbuf[offset], comm_detail::MPIAsCount(chunk), CommDatatype<SType>::value(), dest, comm_detail::MPIChunkTag(tag, i), mpi_comm_, &request[i]);
    offset += chunk;
  }
  return Request(&request);
#endif
#else
  auto it = recv_req.find(tag);
  if (it == recv_req.end()) {
    send_req.insert(std::pair<Integer, ConstIterator<char>>(tag, (ConstIterator<char>)sbuf));
  } else {
    memcopy(it->second, (ConstIterator<char>)sbuf, scount * sizeof(SType));
    recv_req.erase(it);
  }
  return Request();
#endif
}

template <class RType> Comm::Request Comm::Irecv(Iterator<RType> rbuf, Long rcount, Integer source, Integer tag) const {
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!rcount) return Request();
  comm_detail::WarnIfMPIInactive("Comm::Irecv");
#if MPI_VERSION >= 4
  Vector<MPI_Request>& request = NewReq(1);
  comm_detail::TouchBuffer(rbuf, rcount);
  comm_detail::TrackPointToPoint(1, rcount * sizeof(RType));
  MPI_Irecv_c(&rbuf[0], comm_detail::MPIAsCountLarge(rcount), CommDatatype<RType>::value(), source, tag, mpi_comm_, &request[0]);
  return Request(&request);
#else
  const Long request_count = comm_detail::MPINumChunks(rcount);
  comm_detail::AssertChunkedTagRange(tag, request_count, mpi_tag_ub_);
  Vector<MPI_Request>& request = NewReq(request_count);
  comm_detail::TouchBuffer(rbuf, rcount);
  comm_detail::TrackPointToPoint(request_count, rcount * sizeof(RType));
  Long offset = 0;
  for (Long i = 0; i < request_count; i++) {
    const Long chunk = std::min<Long>(rcount - offset, comm_detail::MPIIntLimit());
    MPI_Irecv(&rbuf[offset], comm_detail::MPIAsCount(chunk), CommDatatype<RType>::value(), source, comm_detail::MPIChunkTag(tag, i), mpi_comm_, &request[i]);
    offset += chunk;
  }
  return Request(&request);
#endif
#else
  auto it = send_req.find(tag);
  if (it == send_req.end()) {
    recv_req.insert(std::pair<Integer, Iterator<char>>(tag, (Iterator<char>)rbuf));
  } else {
    memcopy((Iterator<char>)rbuf, it->second, rcount * sizeof(RType));
    send_req.erase(it);
  }
  return Request();
#endif
}

inline void Comm::Wait(Request req) const {
  void* req_ptr = req.release_();
#ifdef SCTL_HAVE_MPI
  if (req_ptr == nullptr) return;
  comm_detail::WarnIfMPIInactive("Comm::Wait");
  Vector<MPI_Request>& request = *(Vector<MPI_Request>*)req_ptr;
  if (request.Dim()) {
    comm_detail::MPIWaitAllBatched(&request[0], request.Dim());
  }
  DelReq(&request);
#else
  SCTL_UNUSED(req_ptr);
#endif
}

template <class Type> void Comm::Bcast(Iterator<Type> buf, Long count, Integer root) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!count) return;
  comm_detail::WarnIfMPIInactive("Comm::Bcast");
  comm_detail::TouchBuffer(buf, count);
#if MPI_VERSION >= 4
  comm_detail::TrackCollective(1, count * sizeof(Type));
  MPI_Bcast_c(&buf[0], comm_detail::MPIAsCountLarge(count), CommDatatype<Type>::value(), root, mpi_comm_);
#else
  comm_detail::TrackCollective(comm_detail::MPINumChunks(count), count * sizeof(Type));
  for (Long offset = 0; offset < count; offset += comm_detail::MPIIntLimit()) {
    const Long chunk = std::min<Long>(count - offset, comm_detail::MPIIntLimit());
    MPI_Bcast(&buf[offset], comm_detail::MPIAsCount(chunk), CommDatatype<Type>::value(), root, mpi_comm_);
  }
#endif
#endif
}

template <class SType, class RType> void Comm::Allgather(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, Long rcount) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
  comm_detail::TouchBuffer(sbuf, scount);
  comm_detail::TouchBuffer(rbuf, rcount * Size());
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Allgather");
  comm_detail::TrackCollective(1, scount * sizeof(SType) + rcount * sizeof(RType));
#if MPI_VERSION >= 4
  MPI_Allgather_c((scount ? &sbuf[0] : nullptr), comm_detail::MPIAsCountLarge(scount), CommDatatype<SType>::value(), (rcount ? &rbuf[0] : nullptr), comm_detail::MPIAsCountLarge(rcount), CommDatatype<RType>::value(), mpi_comm_);
#else
  if (comm_detail::MPIFitsCount(scount) && comm_detail::MPIFitsCount(rcount)) {
    MPI_Allgather((scount ? &sbuf[0] : nullptr), comm_detail::MPIAsCount(scount), CommDatatype<SType>::value(), (rcount ? &rbuf[0] : nullptr), comm_detail::MPIAsCount(rcount), CommDatatype<RType>::value(), mpi_comm_);
  } else {
    SCTL_ASSERT(scount * sizeof(SType) == rcount * sizeof(RType));
    Vector<Long> rcounts_(mpi_size_), rdispls_(mpi_size_);
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < mpi_size_; i++) {
      rcounts_[i] = rcount;
      rdispls_[i] = i * rcount;
    }
    Allgatherv(sbuf, scount, rbuf, rcounts_.begin(), rdispls_.begin());
  }
#endif
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, scount * sizeof(SType));
#endif
}

template <class SType, class RType> void Comm::Allgatherv(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Allgatherv");
  Long rcount_sum = 0, recv_span = 0;
  #pragma omp parallel for schedule(static) reduction(+ : rcount_sum) reduction(max : recv_span)
  for (Integer i = 0; i < mpi_size_; i++) {
    SCTL_ASSERT(rcounts[i] >= 0);
    SCTL_ASSERT(rdispls[i] >= 0);
    rcount_sum += rcounts[i];
    if (rcounts[i]) {
      SCTL_UNUSED(rbuf[rdispls[i]]);
      SCTL_UNUSED(rbuf[rdispls[i] + rcounts[i] - 1]);
      recv_span = std::max<Long>(recv_span, rdispls[i] + rcounts[i]);
    }
  }
  comm_detail::TouchBuffer(sbuf, scount);
  if (!rcount_sum) return;

  comm_detail::TrackCollective(1, scount * sizeof(SType) + rcount_sum * sizeof(RType));
#if MPI_VERSION >= 4
  Vector<MPI_Count> rcounts_(mpi_size_);
  Vector<MPI_Aint> rdispls_(mpi_size_);
  #pragma omp parallel for schedule(static)
  for (Integer i = 0; i < mpi_size_; i++) {
    rcounts_[i] = comm_detail::MPIAsCountLarge(rcounts[i]);
    rdispls_[i] = comm_detail::MPIAsAint(rdispls[i]);
  }
  MPI_Allgatherv_c((scount ? &sbuf[0] : nullptr), comm_detail::MPIAsCountLarge(scount), CommDatatype<SType>::value(), (recv_span ? &rbuf[0] : nullptr), &rcounts_.begin()[0], &rdispls_.begin()[0], CommDatatype<RType>::value(), mpi_comm_);
  return;
#else
  bool fits_typed = comm_detail::MPIFitsCount(scount) && comm_detail::MPIFitsCount(recv_span);
  if (fits_typed) {  // Keep the original typed collective path when the full placed receive range fits in int.
    Vector<int> rcounts_(mpi_size_), rdispls_(mpi_size_);
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < mpi_size_; i++) {
      rcounts_[i] = comm_detail::MPIAsCount(rcounts[i]);
      rdispls_[i] = comm_detail::MPIAsCount(rdispls[i]);
    }
    MPI_Allgatherv((scount ? &sbuf[0] : nullptr), comm_detail::MPIAsCount(scount), CommDatatype<SType>::value(), (recv_span ? &rbuf[0] : nullptr), &rcounts_.begin()[0], &rdispls_.begin()[0], CommDatatype<RType>::value(), mpi_comm_);
    return;
  }

  const Long unit_size = comm_detail::GCD(sizeof(SType), sizeof(RType));
  SCTL_ASSERT(unit_size > 0);
  SCTL_ASSERT(sizeof(SType) % unit_size == 0);
  SCTL_ASSERT(sizeof(RType) % unit_size == 0);
  const Long send_units_scale = static_cast<Long>(sizeof(SType)) / unit_size;
  const Long recv_units_scale = static_cast<Long>(sizeof(RType)) / unit_size;

  const Long send_units = scount * send_units_scale;

  // Express both sides in a shared unit so mixed send/recv types can still use MPI_Allgatherv.
  Vector<Long> recv_counts_units(mpi_size_), recv_displs_units(mpi_size_);
  #pragma omp parallel for schedule(static)
  for (Integer i = 0; i < mpi_size_; i++) {
    recv_counts_units[i] = rcounts[i] * recv_units_scale;
    recv_displs_units[i] = rdispls[i] * recv_units_scale;
  }
  SCTL_ASSERT(send_units == recv_counts_units[mpi_rank_]);

  MPI_Datatype unit_type = MPI_BYTE;
  bool free_unit_type = false;
  if (unit_size > 1) {
    MPI_Type_contiguous(comm_detail::MPIAsInt(unit_size), MPI_BYTE, &unit_type);
    MPI_Type_commit(&unit_type);
    free_unit_type = true;
  }

  ConstIterator<char> sbuf_bytes = (ConstIterator<char>)sbuf;
  Iterator<char> rbuf_bytes = (Iterator<char>)rbuf;
  { // Batch over sorted placed receive blocks to avoid scanning empty windows in sparse layouts.
    const Long window_limit = comm_detail::MPIIntLimit();

    std::vector<std::pair<Long,Integer>> recv_order;
    recv_order.reserve(mpi_size_);
    for (Integer i = 0; i < mpi_size_; i++) {
      if (recv_counts_units[i]) recv_order.push_back(std::make_pair(recv_displs_units[i], i));
    }
    omp_par::merge_sort(recv_order.begin(), recv_order.end());
    const Long recv_order_size = static_cast<Long>(recv_order.size());
    SCTL_ASSERT(recv_order_size);
    for (Long i = 1; i < recv_order_size; i++) {
      const Integer prev_pid = recv_order[i - 1].second;
      const Integer curr_pid = recv_order[i].second;
      SCTL_ASSERT(recv_displs_units[prev_pid] + recv_counts_units[prev_pid] <= recv_displs_units[curr_pid]);
    }

    Vector<int> rcounts_(mpi_size_), rdispls_(mpi_size_);
    rcounts_ = 0;
    rdispls_ = 0;

    Long idx_begin = 0, window_begin = recv_order[0].first, num_windows = 0;
    while (idx_begin < recv_order_size) {
      Long scount_ = 0, sdispl_ = 0;
      Long idx_end = idx_begin, window_end = window_begin;
      for (Long j = idx_begin; j < recv_order_size; j++) {
        const Integer pid = recv_order[j].second;
        if (recv_displs_units[pid] < window_begin + window_limit) {
          const Long msg_begin = std::max<Long>(window_begin, recv_displs_units[pid]);
          const Long msg_end   = std::min<Long>(window_begin + window_limit, recv_displs_units[pid] + recv_counts_units[pid]);
          rdispls_[pid] = msg_begin - window_begin;
          rcounts_[pid] = msg_end - msg_begin;

          if (mpi_rank_ == pid) {
            scount_ = msg_end - msg_begin;
            sdispl_ = msg_begin - recv_displs_units[pid];
          }

          window_end = msg_end;
          idx_end = j;
        } else break;
      }

      MPI_Allgatherv((scount_ ? &sbuf_bytes[sdispl_ * unit_size] : nullptr), comm_detail::MPIAsCount(scount_), unit_type, &rbuf_bytes[window_begin * unit_size], &rcounts_.begin()[0], &rdispls_.begin()[0], unit_type, mpi_comm_);
      num_windows++;

      for (Integer j = idx_begin; j <= idx_end; j++) {
        const Integer pid = recv_order[j].second;
        rdispls_[pid] = 0;
        rcounts_[pid] = 0;
      }
      { // Set idx_begin, window_begin for next iteration
        const Integer pid = recv_order[idx_end].second;
        if (window_end < recv_displs_units[pid] + recv_counts_units[pid]) {
          idx_begin = idx_end;
          window_begin = window_end;
        } else {
          idx_begin = idx_end + 1;
          if (idx_begin < recv_order_size) window_begin = recv_order[idx_begin].first;
        }
      }
    }
    comm_detail::TrackCollective(num_windows - 1, 0);
  }

  if (free_unit_type) MPI_Type_free(&unit_type);
#endif
#else
  memcopy((Iterator<char>)(rbuf + rdispls[0]), (ConstIterator<char>)sbuf, scount * sizeof(SType));
#endif
}

template <class SType, class RType> void Comm::Alltoall(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, Long rcount) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Alltoall");
  if (scount) {
    SCTL_UNUSED(sbuf[0]                  );
    SCTL_UNUSED(sbuf[scount * Size() - 1]);
  }
  if (rcount) {
    SCTL_UNUSED(rbuf[0]                  );
    SCTL_UNUSED(rbuf[rcount * Size() - 1]);
  }
  comm_detail::TrackCollective(1, scount * sizeof(SType) + rcount * sizeof(RType));
#if MPI_VERSION >= 4
  MPI_Alltoall_c((scount ? &sbuf[0] : nullptr), comm_detail::MPIAsCountLarge(scount), CommDatatype<SType>::value(), (rcount ? &rbuf[0] : nullptr), comm_detail::MPIAsCountLarge(rcount), CommDatatype<RType>::value(), mpi_comm_);
#else
  if (comm_detail::MPIFitsCount(scount) && comm_detail::MPIFitsCount(rcount)) {
    MPI_Alltoall((scount ? &sbuf[0] : nullptr), comm_detail::MPIAsCount(scount), CommDatatype<SType>::value(), (rcount ? &rbuf[0] : nullptr), comm_detail::MPIAsCount(rcount), CommDatatype<RType>::value(), mpi_comm_);
  } else {
    SCTL_ASSERT(scount * sizeof(SType) == rcount * sizeof(RType));
    Vector<Long> scounts(mpi_size_), sdispls(mpi_size_), rcounts(mpi_size_), rdispls(mpi_size_);
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < mpi_size_; i++) {
      scounts[i] = scount;
      sdispls[i] = i * scount;
      rcounts[i] = rcount;
      rdispls[i] = i * rcount;
    }
    auto mpi_req = Ialltoallv_sparse(sbuf, scounts.begin(), sdispls.begin(), rbuf, rcounts.begin(), rdispls.begin(), 0);
    Wait(std::move(mpi_req));
  }
#endif
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, scount * sizeof(SType));
#endif
}

template <class SType, class RType> Comm::Request Comm::Ialltoallv_sparse(ConstIterator<SType> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls, Integer tag) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Ialltoallv_sparse");
#if MPI_VERSION >= 4
  Long request_count = 0;
  Long total_bytes = 0;
  // MPI-4 large-count point-to-point can exchange each peer payload directly in bytes.
  for (Integer i = 0; i < mpi_size_; i++) {
    const Long recv_bytes = rcounts[i] * sizeof(RType);
    const Long send_bytes = scounts[i] * sizeof(SType);
    request_count += (recv_bytes != 0);
    request_count += (send_bytes != 0);
    total_bytes += recv_bytes + send_bytes;
  }
  if (!request_count) return Request();
  Vector<MPI_Request>& request = NewReq(request_count);
  Long request_iter = 0;

  comm_detail::TrackPointToPoint(request_count, total_bytes);

  for (Integer i = 0; i < mpi_size_; i++) {
    const Long recv_bytes = rcounts[i] * sizeof(RType);
    if (recv_bytes) {
      Iterator<char> recv_buf = (Iterator<char>)(rbuf + rdispls[i]);
      SCTL_UNUSED(recv_buf[0]             );
      SCTL_UNUSED(recv_buf[recv_bytes - 1]);
      MPI_Irecv_c(&recv_buf[0], comm_detail::MPIAsCountLarge(recv_bytes), MPI_BYTE, i, tag, mpi_comm_, &request[request_iter]);
      request_iter++;
    }
  }
  for (Integer i = 0; i < mpi_size_; i++) {
    const Long send_bytes = scounts[i] * sizeof(SType);
    if (send_bytes) {
      ConstIterator<char> send_buf = (ConstIterator<char>)(sbuf + sdispls[i]);
      SCTL_UNUSED(send_buf[0]             );
      SCTL_UNUSED(send_buf[send_bytes - 1]);
      MPI_Isend_c(&send_buf[0], comm_detail::MPIAsCountLarge(send_bytes), MPI_BYTE, i, tag, mpi_comm_, &request[request_iter]);
      request_iter++;
    }
  }
  return Request(&request);
#else
  Long request_count = 0;
  Long max_chunk_count = 0;
  Long total_bytes = 0;
  // Older MPI implementations need large peer messages to be split into int-sized byte chunks.
  for (Integer i = 0; i < mpi_size_; i++) {
    const Long recv_bytes = rcounts[i] * sizeof(RType);
    const Long send_bytes = scounts[i] * sizeof(SType);
    request_count += comm_detail::MPINumChunks(recv_bytes);
    request_count += comm_detail::MPINumChunks(send_bytes);
    max_chunk_count = std::max<Long>(max_chunk_count, comm_detail::MPINumChunks(recv_bytes));
    max_chunk_count = std::max<Long>(max_chunk_count, comm_detail::MPINumChunks(send_bytes));
    total_bytes += recv_bytes + send_bytes;
  }
  if (!request_count) return Request();
  comm_detail::AssertChunkedTagRange(tag, max_chunk_count, mpi_tag_ub_);
  Vector<MPI_Request>& request = NewReq(request_count);
  Long request_iter = 0;

  comm_detail::TrackPointToPoint(request_count, total_bytes);

  for (Integer i = 0; i < mpi_size_; i++) {
    const Long recv_bytes = rcounts[i] * sizeof(RType);
    if (recv_bytes) {
      Iterator<char> recv_buf = (Iterator<char>)(rbuf + rdispls[i]);
      SCTL_UNUSED(recv_buf[0]             );
      SCTL_UNUSED(recv_buf[recv_bytes - 1]);
      Long offset = 0;
      for (Long j = 0; j < comm_detail::MPINumChunks(recv_bytes); j++) {
        const Long chunk = std::min<Long>(recv_bytes - offset, comm_detail::MPIIntLimit());
        MPI_Irecv(&recv_buf[offset], comm_detail::MPIAsCount(chunk), MPI_BYTE, i, comm_detail::MPIChunkTag(tag, j), mpi_comm_, &request[request_iter]);
        request_iter++;
        offset += chunk;
      }
    }
  }
  for (Integer i = 0; i < mpi_size_; i++) {
    const Long send_bytes = scounts[i] * sizeof(SType);
    if (send_bytes) {
      ConstIterator<char> send_buf = (ConstIterator<char>)(sbuf + sdispls[i]);
      SCTL_UNUSED(send_buf[0]             );
      SCTL_UNUSED(send_buf[send_bytes - 1]);
      Long offset = 0;
      for (Long j = 0; j < comm_detail::MPINumChunks(send_bytes); j++) {
        const Long chunk = std::min<Long>(send_bytes - offset, comm_detail::MPIIntLimit());
        MPI_Isend(&send_buf[offset], comm_detail::MPIAsCount(chunk), MPI_BYTE, i, comm_detail::MPIChunkTag(tag, j), mpi_comm_, &request[request_iter]);
        request_iter++;
        offset += chunk;
      }
    }
  }
  return Request(&request);
#endif
#else
  memcopy((Iterator<char>)(rbuf + rdispls[0]), (ConstIterator<char>)(sbuf + sdispls[0]), scounts[0] * sizeof(SType));
  return Request();
#endif
}

template <class Type> void Comm::Alltoallv(ConstIterator<Type> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<Type> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  comm_detail::WarnIfMPIInactive("Comm::Alltoallv");
#if MPI_VERSION >= 4
  {
    // MPI-4 handles large counts and displacements directly through the _c binding.
    Vector<MPI_Count> scnt(mpi_size_), rcnt(mpi_size_);
    Vector<MPI_Aint> sdsp(mpi_size_), rdsp(mpi_size_);
    Long stotal = 0, rtotal = 0;
    #pragma omp parallel for schedule(static) reduction(+ : stotal, rtotal)
    for (Integer i = 0; i < mpi_size_; i++) {
      scnt[i] = comm_detail::MPIAsCountLarge(scounts[i]);
      sdsp[i] = comm_detail::MPIAsAint(sdispls[i]);
      rcnt[i] = comm_detail::MPIAsCountLarge(rcounts[i]);
      rdsp[i] = comm_detail::MPIAsAint(rdispls[i]);
      stotal += scounts[i];
      rtotal += rcounts[i];
    }
    comm_detail::TrackCollective(1, stotal * sizeof(Type) + rtotal * sizeof(Type));
    MPI_Alltoallv_c((stotal ? &sbuf[0] : nullptr), &scnt[0], &sdsp[0], CommDatatype<Type>::value(), (rtotal ? &rbuf[0] : nullptr), &rcnt[0], &rdsp[0], CommDatatype<Type>::value(), mpi_comm_);
    return;
  }
#else
  bool fits_int = true;
  for (Integer i = 0; i < mpi_size_; i++) {
    fits_int = fits_int && comm_detail::MPIFitsCount(scounts[i]) && comm_detail::MPIFitsCount(sdispls[i]) && comm_detail::MPIFitsCount(rcounts[i]) && comm_detail::MPIFitsCount(rdispls[i]);
  }
  if (!fits_int) {  // Fall back to sparse point-to-point exchange once any count or displacement exceeds int.
    auto mpi_req = Ialltoallv_sparse(sbuf, scounts, sdispls, rbuf, rcounts, rdispls, 0);
    Wait(std::move(mpi_req));
    return;
  }

  {  // Use Alltoallv_sparse of average connectivity<64
    Long connectivity = 0, glb_connectivity = 0;
    #pragma omp parallel for schedule(static) reduction(+ : connectivity)
    for (Integer i = 0; i < mpi_size_; i++) {
      if (rcounts[i]) connectivity++;
    }
    Allreduce(Ptr2ConstItr<Long>(&connectivity, 1), Ptr2Itr<Long>(&glb_connectivity, 1), 1, CommOp::SUM);
    if (glb_connectivity < 64 * Size()) {
      auto mpi_req = Ialltoallv_sparse(sbuf, scounts, sdispls, rbuf, rcounts, rdispls, 0);
      Wait(std::move(mpi_req));
      { // Verify
        #ifdef SCTL_MEMDEBUG
        for (long i = 0; i < mpi_size_-1; i++) {
          SCTL_ASSERT(sdispls[i+1]-sdispls[i] == scounts[i]);
          SCTL_ASSERT(rdispls[i+1]-rdispls[i] == rcounts[i]);
        }
        SCTL_ASSERT(sdispls[0] == 0);
        SCTL_ASSERT(rdispls[0] == 0);

        const Long Nsend = sdispls[mpi_size_-1] + scounts[mpi_size_-1];
        Vector<Type> sbuf_verify(Nsend);
        mpi_req = Ialltoallv_sparse(rbuf, rcounts, rdispls, sbuf_verify.begin(), scounts, sdispls, 1);
        Wait(std::move(mpi_req));

        for (long p = 0; p < mpi_size_; p++) {
          for (long j = 0; j < scounts[p]*(long)sizeof(Type); j++) {
            long i = sdispls[p]*(long)sizeof(Type) + j;
            if (((char*)&sbuf_verify[0])[i] != ((char*)&sbuf[0])[i]) {
              std::cout<<Rank()<<' '<<p<<'\n';
            }
            SCTL_ASSERT(((char*)&sbuf_verify[0])[i] == ((char*)&sbuf[0])[i]);
          }
        }
        Barrier();
        #endif
      }
      return;
    }
  }

  {  // Use vendor MPI_Alltoallv
    //#ifndef ALLTOALLV_FIX
    Vector<int> scnt, sdsp, rcnt, rdsp;
    scnt.ReInit(mpi_size_);
    sdsp.ReInit(mpi_size_);
    rcnt.ReInit(mpi_size_);
    rdsp.ReInit(mpi_size_);
    Long stotal = 0, rtotal = 0;
    #pragma omp parallel for schedule(static) reduction(+ : stotal, rtotal)
    for (Integer i = 0; i < mpi_size_; i++) {
      scnt[i] = scounts[i];
      sdsp[i] = sdispls[i];
      rcnt[i] = rcounts[i];
      rdsp[i] = rdispls[i];
      stotal += scounts[i];
      rtotal += rcounts[i];
    }

    comm_detail::TrackCollective(1, stotal * sizeof(Type) + rtotal * sizeof(Type));
    MPI_Alltoallv((stotal ? &sbuf[0] : nullptr), &scnt[0], &sdsp[0], CommDatatype<Type>::value(), (rtotal ? &rbuf[0] : nullptr), &rcnt[0], &rdsp[0], CommDatatype<Type>::value(), mpi_comm_);
    return;
    //#endif
  }

// TODO: implement hypercube scheme
#endif
#else
  memcopy((Iterator<char>)(rbuf + rdispls[0]), (ConstIterator<char>)(sbuf + sdispls[0]), scounts[0] * sizeof(Type));
#endif
}

template <class Type> void Comm::Allreduce(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, CommOp op) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!count) return;
  comm_detail::WarnIfMPIInactive("Comm::Allreduce");
  const MPI_Op mpi_op = GetMPIOp<Type>(op);
  comm_detail::TouchBuffer(sbuf, count);
  comm_detail::TouchBuffer(rbuf, count);
#if MPI_VERSION >= 4
  comm_detail::TrackCollective(1, count * sizeof(Type));
  MPI_Allreduce_c(&sbuf[0], &rbuf[0], comm_detail::MPIAsCountLarge(count), CommDatatype<Type>::value(), mpi_op, mpi_comm_);
#else
  comm_detail::TrackCollective(comm_detail::MPINumChunks(count), count * sizeof(Type));
  for (Long offset = 0; offset < count; offset += comm_detail::MPIIntLimit()) {
    const Long chunk = std::min<Long>(count - offset, comm_detail::MPIIntLimit());
    MPI_Allreduce(&sbuf[offset], &rbuf[offset], comm_detail::MPIAsCount(chunk), CommDatatype<Type>::value(), mpi_op, mpi_comm_);
  }
#endif
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, count * sizeof(Type));
#endif
}

template <class Type> void Comm::Scan(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, CommOp op) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!count) return;
  comm_detail::WarnIfMPIInactive("Comm::Scan");
  const MPI_Op mpi_op = GetMPIOp<Type>(op);
  comm_detail::TouchBuffer(sbuf, count);
  comm_detail::TouchBuffer(rbuf, count);
#if MPI_VERSION >= 4
  comm_detail::TrackCollective(1, count * sizeof(Type));
  MPI_Scan_c(&sbuf[0], &rbuf[0], comm_detail::MPIAsCountLarge(count), CommDatatype<Type>::value(), mpi_op, mpi_comm_);
#else
  comm_detail::TrackCollective(comm_detail::MPINumChunks(count), count * sizeof(Type));
  for (Long offset = 0; offset < count; offset += comm_detail::MPIIntLimit()) {
    const Long chunk = std::min<Long>(count - offset, comm_detail::MPIIntLimit());
    MPI_Scan(&sbuf[offset], &rbuf[offset], comm_detail::MPIAsCount(chunk), CommDatatype<Type>::value(), mpi_op, mpi_comm_);
  }
#endif
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, count * sizeof(Type));
#endif
}

template <class Type> void Comm::PartitionW(Vector<Type>& nodeList, const Vector<Long>* wts_) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  Integer npes = Size();
  if (npes == 1) return;
  Long nlSize = nodeList.Dim();

  Vector<Long> wts;
  Long localWt = 0;
  if (wts_ == nullptr) {  // Construct arrays of wts.
    wts.ReInit(nlSize);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < nlSize; i++) {
      wts[i] = 1;
    }
    localWt = nlSize;
  } else {
    wts.ReInit(nlSize, (Iterator<Long>)wts_->begin(), false);
    #pragma omp parallel for reduction(+ : localWt)
    for (Long i = 0; i < nlSize; i++) {
      localWt += wts[i];
    }
  }

  Long off1 = 0, off2 = 0, totalWt = 0;
  {  // compute the total weight of the problem ...
    Allreduce<Long>(Ptr2ConstItr<Long>(&localWt, 1), Ptr2Itr<Long>(&totalWt, 1), 1, CommOp::SUM);
    Scan<Long>(Ptr2ConstItr<Long>(&localWt, 1), Ptr2Itr<Long>(&off2, 1), 1, CommOp::SUM);
    off1 = off2 - localWt;
  }

  Vector<Long> lscn;
  if (nlSize) {  // perform a local scan on the weights first ...
    lscn.ReInit(nlSize);
    lscn[0] = off1;
    omp_par::scan(wts.begin(), lscn.begin(), nlSize);
  }

  Vector<Long> sendSz, recvSz, sendOff, recvOff;
  sendSz.ReInit(npes);
  recvSz.ReInit(npes);
  sendOff.ReInit(npes);
  recvOff.ReInit(npes);
  sendSz.SetZero();

  if (nlSize > 0 && totalWt > 0) {  // Compute sendSz
    Long pid1 = (off1 * npes) / totalWt;
    Long pid2 = ((off2 + 1) * npes) / totalWt + 1;
    assert((totalWt * pid2) / npes >= off2);
    pid1 = (pid1 < 0 ? 0 : pid1);
    pid2 = (pid2 > npes ? npes : pid2);
    #pragma omp parallel for schedule(static)
    for (Integer i = pid1; i < pid2; i++) {
      Long wt1 = (totalWt * (i)) / npes;
      Long wt2 = (totalWt * (i + 1)) / npes;
      Long start = std::lower_bound(lscn.begin(), lscn.begin() + nlSize, wt1, std::less<Long>()) - lscn.begin();
      Long end = std::lower_bound(lscn.begin(), lscn.begin() + nlSize, wt2, std::less<Long>()) - lscn.begin();
      if (i == 0) start = 0;
      if (i == npes - 1) end = nlSize;
      sendSz[i] = end - start;
    }
  } else {
    sendSz[0] = nlSize;
  }

  // Exchange sendSz, recvSz
  Alltoall<Long>(sendSz.begin(), 1, recvSz.begin(), 1);

  {  // Compute sendOff, recvOff
    sendOff[0] = 0;
    omp_par::scan(sendSz.begin(), sendOff.begin(), npes);
    recvOff[0] = 0;
    omp_par::scan(recvSz.begin(), recvOff.begin(), npes);
    assert(sendOff[npes - 1] + sendSz[npes - 1] == nlSize);
  }

  // perform All2All  ...
  Vector<Type> newNodes;
  newNodes.ReInit(recvSz[npes - 1] + recvOff[npes - 1]);
  auto mpi_req = Ialltoallv_sparse<Type>(nodeList.begin(), sendSz.begin(), sendOff.begin(), newNodes.begin(), recvSz.begin(), recvOff.begin());
  Wait(std::move(mpi_req));

  // reset the pointer ...
  nodeList.Swap(newNodes);
}

template <class Type> void Comm::PartitionN(Vector<Type>& v, Long N) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  Integer rank = Rank();
  Integer np = Size();
  if (np == 1) return;

  Vector<Long> v_cnt(np), v_dsp(np + 1);
  Vector<Long> N_cnt(np), N_dsp(np + 1);
  {  // Set v_cnt, v_dsp
    v_dsp[0] = 0;
    Long cnt = v.Dim();
    Allgather(Ptr2ConstItr<Long>(&cnt, 1), 1, v_cnt.begin(), 1);
    omp_par::scan(v_cnt.begin(), v_dsp.begin(), np);
    v_dsp[np] = v_cnt[np - 1] + v_dsp[np - 1];
  }
  {  // Set N_cnt, N_dsp
    N_dsp[0] = 0;
    Long cnt = N;
    Allgather(Ptr2ConstItr<Long>(&cnt, 1), 1, N_cnt.begin(), 1);
    omp_par::scan(N_cnt.begin(), N_dsp.begin(), np);
    N_dsp[np] = N_cnt[np - 1] + N_dsp[np - 1];
  }
  {  // Adjust for dof
    Long dof = (N_dsp[np] ? v_dsp[np] / N_dsp[np] : 0);
    assert(dof * N_dsp[np] == v_dsp[np]);
    if (dof == 0) return;

    if (dof != 1) {
      #pragma omp parallel for schedule(static)
      for (Integer i = 0; i < np; i++) N_cnt[i] *= dof;
      #pragma omp parallel for schedule(static)
      for (Integer i = 0; i <= np; i++) N_dsp[i] *= dof;
    }
  }

  Vector<Type> v_(N_cnt[rank]);
  {  // Set v_
    Vector<Long> scnt(np), sdsp(np);
    Vector<Long> rcnt(np), rdsp(np);
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < np; i++) {
      {  // Set scnt
        Long n0 = N_dsp[i + 0];
        Long n1 = N_dsp[i + 1];
        if (n0 < v_dsp[rank + 0]) n0 = v_dsp[rank + 0];
        if (n1 < v_dsp[rank + 0]) n1 = v_dsp[rank + 0];
        if (n0 > v_dsp[rank + 1]) n0 = v_dsp[rank + 1];
        if (n1 > v_dsp[rank + 1]) n1 = v_dsp[rank + 1];
        scnt[i] = n1 - n0;
      }
      {  // Set rcnt
        Long n0 = v_dsp[i + 0];
        Long n1 = v_dsp[i + 1];
        if (n0 < N_dsp[rank + 0]) n0 = N_dsp[rank + 0];
        if (n1 < N_dsp[rank + 0]) n1 = N_dsp[rank + 0];
        if (n0 > N_dsp[rank + 1]) n0 = N_dsp[rank + 1];
        if (n1 > N_dsp[rank + 1]) n1 = N_dsp[rank + 1];
        rcnt[i] = n1 - n0;
      }
    }
    sdsp[0] = 0;
    omp_par::scan(scnt.begin(), sdsp.begin(), np);
    rdsp[0] = 0;
    omp_par::scan(rcnt.begin(), rdsp.begin(), np);

    auto mpi_request = Ialltoallv_sparse(v.begin(), scnt.begin(), sdsp.begin(), v_.begin(), rcnt.begin(), rdsp.begin());
    Wait(std::move(mpi_request));
  }
  v.Swap(v_);
}

template <class Type, class Compare> void Comm::PartitionS(Vector<Type>& nodeList, const Type& splitter, Compare comp) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  Integer npes = Size();
  if (npes == 1) return;

  Vector<Type> mins(npes);
  Allgather(Ptr2ConstItr<Type>(&splitter, 1), 1, mins.begin(), 1);

  Vector<Long> scnt(npes), sdsp(npes);
  Vector<Long> rcnt(npes), rdsp(npes);
  {  // Compute scnt, sdsp
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      sdsp[i] = std::lower_bound(nodeList.begin(), nodeList.begin() + nodeList.Dim(), mins[i], comp) - nodeList.begin();
    }
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes - 1; i++) {
      scnt[i] = sdsp[i + 1] - sdsp[i];
    }
    scnt[npes - 1] = nodeList.Dim() - sdsp[npes - 1];
  }
  {  // Compute rcnt, rdsp
    rdsp[0] = 0;
    Alltoall(scnt.begin(), 1, rcnt.begin(), 1);
    omp_par::scan(rcnt.begin(), rdsp.begin(), npes);
  }
  {  // Redistribute nodeList
    Vector<Type> nodeList_(rdsp[npes - 1] + rcnt[npes - 1]);
    auto mpi_request = Ialltoallv_sparse(nodeList.begin(), scnt.begin(), sdsp.begin(), nodeList_.begin(), rcnt.begin(), rdsp.begin());
    Wait(std::move(mpi_request));
    nodeList.Swap(nodeList_);
  }
}

template <class Type> void Comm::SortScatterIndex(const Vector<Type>& key, Vector<Long>& scatter_index, const Type* split_key_) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  typedef SortPair<Type, Long> Pair_t;
  Integer npes = Size();

  Vector<Pair_t> parray(key.Dim());
  {  // Build global index.
    Long glb_dsp = 0;
    Long loc_size = key.Dim();
    Scan(Ptr2ConstItr<Long>(&loc_size, 1), Ptr2Itr<Long>(&glb_dsp, 1), 1, CommOp::SUM);
    glb_dsp -= loc_size;
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < loc_size; i++) {
      parray[i].key = key[i];
      parray[i].data = glb_dsp + i;
    }
  }

  Vector<Pair_t> psorted;
  HyperQuickSort(parray, psorted);

  if (npes > 1 && split_key_ != nullptr) {  // Partition data
    Vector<Type> split_key(npes);
    Allgather(Ptr2ConstItr<Type>(split_key_, 1), 1, split_key.begin(), 1);

    Vector<Long> sendSz(npes);
    Vector<Long> recvSz(npes);
    Vector<Long> sendOff(npes);
    Vector<Long> recvOff(npes);
    Long nlSize = psorted.Dim();
    sendSz.SetZero();

    if (nlSize > 0) {  // Compute sendSz
      // Determine processor range.
      Long pid1 = std::lower_bound(split_key.begin(), split_key.begin() + npes, psorted[0].key) - split_key.begin() - 1;
      Long pid2 = std::upper_bound(split_key.begin(), split_key.begin() + npes, psorted[nlSize - 1].key) - split_key.begin() + 1;
      pid1 = (pid1 < 0 ? 0 : pid1);
      pid2 = (pid2 > npes ? npes : pid2);

      #pragma omp parallel for schedule(static)
      for (Integer i = pid1; i < pid2; i++) {
        Pair_t p1;
        p1.key = split_key[i];
        Pair_t p2;
        p2.key = split_key[i + 1 < npes ? i + 1 : i];
        Long start = std::lower_bound(psorted.begin(), psorted.begin() + nlSize, p1, std::less<Pair_t>()) - psorted.begin();
        Long end = std::lower_bound(psorted.begin(), psorted.begin() + nlSize, p2, std::less<Pair_t>()) - psorted.begin();
        if (i == 0) start = 0;
        if (i == npes - 1) end = nlSize;
        sendSz[i] = end - start;
      }
    }

    // Exchange sendSz, recvSz
    Alltoall<Long>(sendSz.begin(), 1, recvSz.begin(), 1);

    // compute offsets ...
    {  // Compute sendOff, recvOff
      sendOff[0] = 0;
      omp_par::scan(sendSz.begin(), sendOff.begin(), npes);
      recvOff[0] = 0;
      omp_par::scan(recvSz.begin(), recvOff.begin(), npes);
      assert(sendOff[npes - 1] + sendSz[npes - 1] == nlSize);
    }

    // perform All2All  ...
    Vector<Pair_t> newNodes(recvSz[npes - 1] + recvOff[npes - 1]);
    auto mpi_req = Ialltoallv_sparse<Pair_t>(psorted.begin(), sendSz.begin(), sendOff.begin(), newNodes.begin(), recvSz.begin(), recvOff.begin());
    Wait(std::move(mpi_req));

    // reset the pointer ...
    psorted.Swap(newNodes);
  }

  scatter_index.ReInit(psorted.Dim());
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < psorted.Dim(); i++) {
    scatter_index[i] = psorted[i].data;
  }
}

template <class Type> void Comm::ScatterForward(Vector<Type>& data_, const Vector<Long>& scatter_index) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  typedef SortPair<Long, Long> Pair_t;
  Integer npes = Size(), rank = Rank();

  Long data_dim = 0;
  Long send_size = 0;
  Long recv_size = 0;
  {  // Set data_dim, send_size, recv_size
    recv_size = scatter_index.Dim();
    StaticArray<Long, 2> glb_size;
    StaticArray<Long, 2> loc_size;
    loc_size[0] = data_.Dim();
    loc_size[1] = recv_size;
    Allreduce<Long>(loc_size, glb_size, 2, CommOp::SUM);
    if (glb_size[0] == 0 || glb_size[1] == 0) return;  // Nothing to be done.
    data_dim = glb_size[0] / glb_size[1];
    SCTL_ASSERT(glb_size[0] == data_dim * glb_size[1]);
    send_size = data_.Dim() / data_dim;
  }

  if (npes == 1) {  // Scatter directly
    Vector<Type> data;
    data.ReInit(recv_size * data_dim);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = scatter_index[i] * data_dim;
      Long trg_indx = i * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = data_[src_indx + j];
    }
    data_.Swap(data);
    return;
  }

  Vector<Long> glb_scan;
  {  // Global scan of data size.
    glb_scan.ReInit(npes);
    Long glb_rank = 0;
    Scan(Ptr2ConstItr<Long>(&send_size, 1), Ptr2Itr<Long>(&glb_rank, 1), 1, CommOp::SUM);
    glb_rank -= send_size;
    Allgather(Ptr2ConstItr<Long>(&glb_rank, 1), 1, glb_scan.begin(), 1);
  }

  Vector<Pair_t> psorted;
  {  // Sort scatter_index.
    psorted.ReInit(recv_size);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      psorted[i].key = scatter_index[i];
      psorted[i].data = i;
    }
    omp_par::merge_sort(psorted.begin(), psorted.begin() + recv_size);
  }

  Vector<Long> recv_indx(recv_size);
  Vector<Long> send_indx(send_size);
  Vector<Long> sendSz(npes);
  Vector<Long> sendOff(npes);
  Vector<Long> recvSz(npes);
  Vector<Long> recvOff(npes);
  {  // Exchange send, recv indices.
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      recv_indx[i] = psorted[i].key;
    }

    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      Long start = std::lower_bound(recv_indx.begin(), recv_indx.begin() + recv_size, glb_scan[i]) - recv_indx.begin();
      Long end = (i + 1 < npes ? std::lower_bound(recv_indx.begin(), recv_indx.begin() + recv_size, glb_scan[i + 1]) - recv_indx.begin() : recv_size);
      recvSz[i] = end - start;
      recvOff[i] = start;
    }

    Alltoall(recvSz.begin(), 1, sendSz.begin(), 1);
    sendOff[0] = 0;
    omp_par::scan(sendSz.begin(), sendOff.begin(), npes);
    assert(sendOff[npes - 1] + sendSz[npes - 1] == send_size);

    Alltoallv(recv_indx.begin(), recvSz.begin(), recvOff.begin(), send_indx.begin(), sendSz.begin(), sendOff.begin());
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      assert(send_indx[i] >= glb_scan[rank]);
      send_indx[i] -= glb_scan[rank];
      assert(send_indx[i] < send_size);
    }
  }

  Vector<Type> send_buff;
  {  // Prepare send buffer
    send_buff.ReInit(send_size * data_dim);
    ConstIterator<Type> data = data_.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      Long src_indx = send_indx[i] * data_dim;
      Long trg_indx = i * data_dim;
      for (Long j = 0; j < data_dim; j++) send_buff[trg_indx + j] = data[src_indx + j];
    }
  }

  Vector<Type> recv_buff;
  {  // All2Allv
    recv_buff.ReInit(recv_size * data_dim);
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      sendSz[i] *= data_dim;
      sendOff[i] *= data_dim;
      recvSz[i] *= data_dim;
      recvOff[i] *= data_dim;
    }
    Alltoallv(send_buff.begin(), sendSz.begin(), sendOff.begin(), recv_buff.begin(), recvSz.begin(), recvOff.begin());
  }

  {  // Build output data.
    data_.ReInit(recv_size * data_dim);
    Iterator<Type> data = data_.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = i * data_dim;
      Long trg_indx = psorted[i].data * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = recv_buff[src_indx + j];
    }
  }
}

template <class Type> void Comm::ScatterReverse(Vector<Type>& data_, const Vector<Long>& scatter_index_, Long loc_size_) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  typedef SortPair<Long, Long> Pair_t;
  Integer npes = Size(), rank = Rank();

  Long data_dim = 0;
  Long send_size = 0;
  Long recv_size = 0;
  {  // Set data_dim, send_size, recv_size
    recv_size = loc_size_;
    StaticArray<Long, 3> glb_size;
    StaticArray<Long, 3> loc_size;
    loc_size[0] = data_.Dim();
    loc_size[1] = scatter_index_.Dim();
    loc_size[2] = recv_size;
    Allreduce<Long>(loc_size, glb_size, 3, CommOp::SUM);
    if (glb_size[0] == 0 || glb_size[1] == 0) return;  // Nothing to be done.

    SCTL_ASSERT(glb_size[0] % glb_size[1] == 0);
    data_dim = glb_size[0] / glb_size[1];

    SCTL_ASSERT(loc_size[0] % data_dim == 0);
    send_size = loc_size[0] / data_dim;

    if (glb_size[2] == 0) { // partition uniformly
      recv_size = (((rank + 1) * glb_size[1]) / npes) - ((rank * glb_size[1]) / npes);
    } else {
      SCTL_ASSERT(glb_size[2] % glb_size[1] == 0);
      const Long dof = glb_size[2] / glb_size[1];
      SCTL_ASSERT(loc_size[2] % dof == 0);
      recv_size = loc_size[2] / dof;
    }
  }

  if (npes == 1) {  // Scatter directly
    Vector<Type> data;
    data.ReInit(recv_size * data_dim);
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = i * data_dim;
      Long trg_indx = scatter_index_[i] * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = data_[src_indx + j];
    }
    data_.Swap(data);
    return;
  }

  Vector<Long> scatter_index;
  {
    StaticArray<Long, 2> glb_rank;
    StaticArray<Long, 3> glb_size;
    StaticArray<Long, 2> loc_size;
    loc_size[0] = data_.Dim() / data_dim;
    loc_size[1] = scatter_index_.Dim();
    Scan<Long>(loc_size, glb_rank, 2, CommOp::SUM);
    Allreduce<Long>(loc_size, glb_size, 2, CommOp::SUM);
    SCTL_ASSERT(glb_size[0] == glb_size[1]);
    glb_rank[0] -= loc_size[0];
    glb_rank[1] -= loc_size[1];

    Vector<Long> glb_scan0(npes + 1);
    Vector<Long> glb_scan1(npes + 1);
    Allgather<Long>(glb_rank + 0, 1, glb_scan0.begin(), 1);
    Allgather<Long>(glb_rank + 1, 1, glb_scan1.begin(), 1);
    glb_scan0[npes] = glb_size[0];
    glb_scan1[npes] = glb_size[1];

    if (loc_size[0] != loc_size[1] || glb_rank[0] != glb_rank[1]) {  // Repartition scatter_index
      scatter_index.ReInit(loc_size[0]);

      Vector<Long> send_dsp(npes + 1);
      Vector<Long> recv_dsp(npes + 1);
      #pragma omp parallel for schedule(static)
      for (Integer i = 0; i <= npes; i++) {
        send_dsp[i] = std::min(std::max(glb_scan0[i], glb_rank[1]), glb_rank[1] + loc_size[1]) - glb_rank[1];
        recv_dsp[i] = std::min(std::max(glb_scan1[i], glb_rank[0]), glb_rank[0] + loc_size[0]) - glb_rank[0];
      }

      // Long commCnt=0;
      Vector<Long> send_cnt(npes + 0);
      Vector<Long> recv_cnt(npes + 0);
      #pragma omp parallel for schedule(static)  // reduction(+:commCnt)
      for (Integer i = 0; i < npes; i++) {
        send_cnt[i] = send_dsp[i + 1] - send_dsp[i];
        recv_cnt[i] = recv_dsp[i + 1] - recv_dsp[i];
        // if(send_cnt[i] && i!=rank) commCnt++;
        // if(recv_cnt[i] && i!=rank) commCnt++;
      }

      auto mpi_req = Ialltoallv_sparse<Long>(scatter_index_.begin(), send_cnt.begin(), send_dsp.begin(), scatter_index.begin(), recv_cnt.begin(), recv_dsp.begin(), 0);
      Wait(std::move(mpi_req));
    } else {
      scatter_index.ReInit(scatter_index_.Dim(), (Iterator<Long>)scatter_index_.begin(), false);
    }
  }

  Vector<Long> glb_scan(npes);
  {  // Global data size.
    Long glb_rank = 0;
    Scan(Ptr2ConstItr<Long>(&recv_size, 1), Ptr2Itr<Long>(&glb_rank, 1), 1, CommOp::SUM);
    glb_rank -= recv_size;
    Allgather(Ptr2ConstItr<Long>(&glb_rank, 1), 1, glb_scan.begin(), 1);
  }

  Vector<Pair_t> psorted(send_size);
  {  // Sort scatter_index.
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      psorted[i].key = scatter_index[i];
      psorted[i].data = i;
    }
    omp_par::merge_sort(psorted.begin(), psorted.begin() + send_size);
  }

  Vector<Long> recv_indx(recv_size);
  Vector<Long> send_indx(send_size);
  Vector<Long> sendSz(npes);
  Vector<Long> sendOff(npes);
  Vector<Long> recvSz(npes);
  Vector<Long> recvOff(npes);
  {  // Exchange send, recv indices.
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      send_indx[i] = psorted[i].key;
    }

    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      Long start = std::lower_bound(send_indx.begin(), send_indx.begin() + send_size, glb_scan[i]) - send_indx.begin();
      Long end = (i + 1 < npes ? std::lower_bound(send_indx.begin(), send_indx.begin() + send_size, glb_scan[i + 1]) - send_indx.begin() : send_size);
      sendSz[i] = end - start;
      sendOff[i] = start;
    }

    Alltoall(sendSz.begin(), 1, recvSz.begin(), 1);
    recvOff[0] = 0;
    omp_par::scan(recvSz.begin(), recvOff.begin(), npes);
    assert(recvOff[npes - 1] + recvSz[npes - 1] == recv_size);

    Alltoallv(send_indx.begin(), sendSz.begin(), sendOff.begin(), recv_indx.begin(), recvSz.begin(), recvOff.begin());
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      assert(recv_indx[i] >= glb_scan[rank]);
      recv_indx[i] -= glb_scan[rank];
      assert(recv_indx[i] < recv_size);
    }
  }

  Vector<Type> send_buff;
  {  // Prepare send buffer
    send_buff.ReInit(send_size * data_dim);
    ConstIterator<Type> data = data_.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      Long src_indx = psorted[i].data * data_dim;
      Long trg_indx = i * data_dim;
      for (Long j = 0; j < data_dim; j++) send_buff[trg_indx + j] = data[src_indx + j];
    }
  }

  Vector<Type> recv_buff;
  {  // All2Allv
    recv_buff.ReInit(recv_size * data_dim);
    #pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      sendSz[i] *= data_dim;
      sendOff[i] *= data_dim;
      recvSz[i] *= data_dim;
      recvOff[i] *= data_dim;
    }
    Alltoallv(send_buff.begin(), sendSz.begin(), sendOff.begin(), recv_buff.begin(), recvSz.begin(), recvOff.begin());
  }

  {  // Build output data.
    data_.ReInit(recv_size * data_dim);
    Iterator<Type> data = data_.begin();
    #pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = i * data_dim;
      Long trg_indx = recv_indx[i] * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = recv_buff[src_indx + j];
    }
  }
}

#ifdef SCTL_HAVE_MPI
template <class Type> inline MPI_Op Comm::GetMPIOp(CommOp op) {
  switch (op) {
    case CommOp::SUM:
      return CommDatatype<Type>::sum();
    case CommOp::MIN:
      return CommDatatype<Type>::min();
    case CommOp::MAX:
      return CommDatatype<Type>::max();
    default:
      return MPI_OP_NULL;
  }
}

inline void Comm::RegisterDatatype(MPI_Datatype datatype) {
  #pragma omp critical(SCTL_COMM_HANDLE_REG)
  {
    DatatypeRegistry().push_back(datatype);
  }
}

inline void Comm::RegisterOp(MPI_Op op) {
  #pragma omp critical(SCTL_COMM_HANDLE_REG)
  {
    OpRegistry().push_back(op);
  }
}

inline void Comm::FreeRegisteredHandles() {
  #pragma omp critical(SCTL_COMM_HANDLE_REG)
  {
    std::vector<MPI_Op>& op_registry = OpRegistry();
    std::vector<MPI_Datatype>& datatype_registry = DatatypeRegistry();
    for (std::size_t i = 0; i < op_registry.size(); i++) {
      if (op_registry[i] != MPI_OP_NULL) {
        MPI_Op_free(&op_registry[i]);
      }
    }
    op_registry.clear();
    for (std::size_t i = 0; i < datatype_registry.size(); i++) {
      if (datatype_registry[i] != MPI_DATATYPE_NULL) {
        MPI_Type_free(&datatype_registry[i]);
      }
    }
    datatype_registry.clear();
  }
}

inline std::vector<MPI_Datatype>& Comm::DatatypeRegistry() {
  static std::vector<MPI_Datatype> registry;
  return registry;
}

inline std::vector<MPI_Op>& Comm::OpRegistry() {
  static std::vector<MPI_Op> registry;
  return registry;
}

inline Vector<MPI_Request>& Comm::NewReq(Long request_count) const {
  Vector<MPI_Request>* request;
  #pragma omp critical(SCTL_COMM_REQ)
  {
    if (req.empty()) req.push(new Vector<MPI_Request>);
    request = (Vector<MPI_Request>*)req.top();
    req.pop();
  }
  request->ReInit(request_count);
  return *request;
}

inline void Comm::Init(const MPI_Comm mpi_comm) {
  comm_detail::WarnIfMPIInactive("Comm::Init");
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_dup(mpi_comm, &mpi_comm_);
  MPI_Comm_rank(mpi_comm_, &mpi_rank_);
  MPI_Comm_size(mpi_comm_, &mpi_size_);
  int flag = 0;
  int* tag_ub_ptr = nullptr;
  MPI_Comm_get_attr(mpi_comm_, MPI_TAG_UB, &tag_ub_ptr, &flag);
  mpi_tag_ub_ = (flag && tag_ub_ptr) ? *tag_ub_ptr : std::numeric_limits<int>::max();
}

inline void Comm::DelReq(Vector<MPI_Request>* req_ptr) const {
  #pragma omp critical(SCTL_COMM_REQ)
  if (req_ptr) req.push(req_ptr);
}

#define SCTL_HS_MPIDATATYPE(CTYPE, MPITYPE)              \
  template <> class Comm::CommDatatype<CTYPE> {     \
   public:                                          \
    static inline MPI_Datatype value() { return MPITYPE; } \
    static inline MPI_Op sum() { return MPI_SUM; }         \
    static inline MPI_Op min() { return MPI_MIN; }         \
    static inline MPI_Op max() { return MPI_MAX; }         \
  }

SCTL_HS_MPIDATATYPE(short, MPI_SHORT);
SCTL_HS_MPIDATATYPE(int, MPI_INT);
SCTL_HS_MPIDATATYPE(long, MPI_LONG);
SCTL_HS_MPIDATATYPE(unsigned short, MPI_UNSIGNED_SHORT);
SCTL_HS_MPIDATATYPE(unsigned int, MPI_UNSIGNED);
SCTL_HS_MPIDATATYPE(unsigned long, MPI_UNSIGNED_LONG);
SCTL_HS_MPIDATATYPE(float, MPI_FLOAT);
SCTL_HS_MPIDATATYPE(double, MPI_DOUBLE);
SCTL_HS_MPIDATATYPE(long double, MPI_LONG_DOUBLE);
SCTL_HS_MPIDATATYPE(long long, MPI_LONG_LONG_INT);
SCTL_HS_MPIDATATYPE(char, MPI_CHAR);
SCTL_HS_MPIDATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
#undef SCTL_HS_MPIDATATYPE
#endif

template <class Type, class Compare> void Comm::HyperQuickSort(const Vector<Type>& arr_, Vector<Type>& SortedElem, Compare comp) const {  // O( ((N/p)+log(p))*(log(N/p)+log(p)) )
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  Integer npes, myrank, omp_p;
  {  // Get comm size and rank.
    npes = Size();
    myrank = Rank();
    omp_p = omp_get_max_threads();
  }
  srand(myrank);

  Long totSize;
  {                 // Local and global sizes. O(log p)
    Long nelem = arr_.Dim();
    Allreduce<Long>(Ptr2ConstItr<Long>(&nelem, 1), Ptr2Itr<Long>(&totSize, 1), 1, CommOp::SUM);
  }

  if (npes == 1) {  // SortedElem <--- local_sort(arr_)
    SortedElem = arr_;
    omp_par::merge_sort(SortedElem.begin(), SortedElem.end(), comp);
    return;
  }

  Vector<Type> arr;
  {  // arr <-- local_sort(arr_)
    arr = arr_;
    omp_par::merge_sort(arr.begin(), arr.end(), comp);
  }

  Vector<Type> nbuff, nbuff_ext, rbuff, rbuff_ext;  // Allocate memory.
  MPI_Comm comm = mpi_comm_;                        // Copy comm
  bool free_comm = false;                           // Flag to free comm.

  // Binary split and merge in each iteration.
  while (npes > 1 && totSize > 0) {  // O(log p) iterations.
    Type split_key;
    Long totSize_new;
    {  // Determine split_key. O( log(N/p) + log(p) )
      Integer glb_splt_count;
      Vector<Type> glb_splitters;
      {  // Take random splitters. glb_splt_count = const = 100~1000
        Integer splt_count;
        Long nelem = arr.Dim();
        {  // Set splt_coun. O( 1 ) -- Let p * splt_count = t
          splt_count = (100 * nelem) / totSize;
          if (npes > 100) splt_count = (drand48() * totSize) < (100 * nelem) ? 1 : 0;
          if (splt_count > nelem) splt_count = nelem;
          MPI_Allreduce  (&splt_count, &glb_splt_count, 1, CommDatatype<Integer>::value(), CommDatatype<Integer>::sum(), comm);
          if (!glb_splt_count) splt_count = std::min<Long>(1, nelem);
          MPI_Allreduce  (&splt_count, &glb_splt_count, 1, CommDatatype<Integer>::value(), CommDatatype<Integer>::sum(), comm);
          SCTL_ASSERT(glb_splt_count);
        }

        Vector<Type> splitters(splt_count);
        for (Integer i = 0; i < splt_count; i++) {
          splitters[i] = arr[rand() % nelem];
        }

        Vector<Integer> glb_splt_cnts(npes), glb_splt_disp(npes);
        {  // Set glb_splt_cnts, glb_splt_disp
          MPI_Allgather(&splt_count, 1, CommDatatype<Integer>::value(), &glb_splt_cnts[0], 1, CommDatatype<Integer>::value(), comm);
          glb_splt_disp[0] = 0;
          omp_par::scan(glb_splt_cnts.begin(), glb_splt_disp.begin(), npes);
          SCTL_ASSERT(glb_splt_count == glb_splt_cnts[npes - 1] + glb_splt_disp[npes - 1]);
        }

        {  // Gather all splitters. O( log(p) )
          glb_splitters.ReInit(glb_splt_count);
          Vector<int> glb_splt_cnts_(npes), glb_splt_disp_(npes);
          for (Integer i = 0; i < npes; i++) {
            glb_splt_cnts_[i] = glb_splt_cnts[i];
            glb_splt_disp_[i] = glb_splt_disp[i];
          }
          MPI_Allgatherv((splt_count ? &splitters[0] : nullptr), splt_count, CommDatatype<Type>::value(), &glb_splitters[0], &glb_splt_cnts_[0], &glb_splt_disp_[0], CommDatatype<Type>::value(), comm);
        }
      }

      // Determine split key. O( log(N/p) + log(p) )
      Vector<Long> lrank(glb_splt_count);
      {  // Compute local rank
        #pragma omp parallel for schedule(static)
        for (Integer i = 0; i < glb_splt_count; i++) {
          lrank[i] = std::lower_bound(arr.begin(), arr.end(), glb_splitters[i], comp) - arr.begin();
        }
      }

      Vector<Long> grank(glb_splt_count);
      {  // Compute global rank
        MPI_Allreduce(&lrank[0], &grank[0], glb_splt_count, CommDatatype<Long>::value(), CommDatatype<Long>::sum(), comm);
      }

      {  // Determine split_key, totSize_new
        Integer splitter_idx = 0;
        for (Integer i = 0; i < glb_splt_count; i++) {
          if (labs(grank[i] - totSize / 2) < labs(grank[splitter_idx] - totSize / 2)) {
            splitter_idx = i;
          }
        }
        split_key = glb_splitters[splitter_idx];

        if (myrank <= (npes - 1) / 2)
          totSize_new = grank[splitter_idx];
        else
          totSize_new = totSize - grank[splitter_idx];

        // double err=(((double)grank[splitter_idx])/(totSize/2))-1.0;
        // if(fabs<double>(err)<0.01 || npes<=16) break;
        // else if(!myrank) std::cout<<err<<'\n';
      }
    }

    Integer split_id = (npes - 1) / 2;
    {  // Split problem into two. O( N/p )
      Integer partner;
      {  // Set partner
        partner = myrank + (split_id+1) * (myrank<=split_id ? 1 : -1);
        if (partner >= npes) partner = npes - 1;
        assert(partner >= 0);
      }
      bool extra_partner = (npes % 2 == 1 && myrank == npes - 1);

      Long ssize = 0, lsize = 0;
      ConstIterator<Type> sbuff, lbuff;
      {  // Set ssize, lsize, sbuff, lbuff
        Long split_indx = std::lower_bound(arr.begin(), arr.end(), split_key, comp) - arr.begin();
        ssize = (myrank > split_id ? split_indx : arr.Dim() - split_indx);
        sbuff = (myrank > split_id ? arr.begin() : arr.begin() + split_indx);
        lsize = (myrank <= split_id ? split_indx : arr.Dim() - split_indx);
        lbuff = (myrank <= split_id ? arr.begin() : arr.begin() + split_indx);
      }

      Long rsize = 0, ext_rsize = 0;
      {  // Get rsize, ext_rsize
        Long ext_ssize = 0;
        MPI_Status status;
        MPI_Sendrecv(&ssize, 1, CommDatatype<Long>::value(), partner, 0, &rsize, 1, CommDatatype<Long>::value(), partner, 0, comm, &status);
        if (extra_partner) MPI_Sendrecv(&ext_ssize, 1, CommDatatype<Long>::value(), split_id, 0, &ext_rsize, 1, CommDatatype<Long>::value(), split_id, 0, comm, &status);
      }

      {  // Exchange data.
        rbuff.ReInit(rsize);
        rbuff_ext.ReInit(ext_rsize);
        MPI_Status status;
        const Long peer_chunk_count = std::max<Long>(comm_detail::MPINumChunks(ssize), comm_detail::MPINumChunks(rsize));
        SCTL_ASSERT(peer_chunk_count == 0 || peer_chunk_count - 1 <= static_cast<Long>(mpi_tag_ub_));
        Long soff = 0, roff = 0;
        for (Long chunk_idx = 0; chunk_idx < peer_chunk_count; chunk_idx++) {
          const Long send_chunk = std::min<Long>(ssize - soff, comm_detail::MPIIntLimit());
          const Long recv_chunk = std::min<Long>(rsize - roff, comm_detail::MPIIntLimit());
          MPI_Sendrecv((send_chunk ? &sbuff[soff] : nullptr), comm_detail::MPIAsCount(send_chunk), CommDatatype<Type>::value(), partner, comm_detail::MPIAsInt(chunk_idx), (recv_chunk ? &rbuff[roff] : nullptr), comm_detail::MPIAsCount(recv_chunk), CommDatatype<Type>::value(), partner, comm_detail::MPIAsInt(chunk_idx), comm, &status);
          soff += send_chunk;
          roff += recv_chunk;
        }
        if (extra_partner) {
          const Long extra_chunk_count = comm_detail::MPINumChunks(ext_rsize);
          SCTL_ASSERT(extra_chunk_count == 0 || extra_chunk_count - 1 <= static_cast<Long>(mpi_tag_ub_));
          Long roff_ext = 0;
          for (Long chunk_idx = 0; chunk_idx < extra_chunk_count; chunk_idx++) {
            const Long recv_chunk = std::min<Long>(ext_rsize - roff_ext, comm_detail::MPIIntLimit());
            MPI_Sendrecv(nullptr, 0, CommDatatype<Type>::value(), split_id, comm_detail::MPIAsInt(chunk_idx), (recv_chunk ? &rbuff_ext[roff_ext] : nullptr), comm_detail::MPIAsCount(recv_chunk), CommDatatype<Type>::value(), split_id, comm_detail::MPIAsInt(chunk_idx), comm, &status);
            roff_ext += recv_chunk;
          }
        }
      }

      {  // nbuff <-- merge(lbuff, rbuff, rbuff_ext)
        nbuff.ReInit(lsize + rsize);
        omp_par::merge<ConstIterator<Type>>(lbuff, (lbuff + lsize), rbuff.begin(), rbuff.begin() + rsize, nbuff.begin(), omp_p, comp);
        if (ext_rsize > 0) {
          if (nbuff.Dim() > 0) {
            nbuff_ext.ReInit(lsize + rsize + ext_rsize);
            omp_par::merge(nbuff.begin(), nbuff.begin() + (lsize + rsize), rbuff_ext.begin(), rbuff_ext.begin() + ext_rsize, nbuff_ext.begin(), omp_p, comp);
            nbuff.Swap(nbuff_ext);
            nbuff_ext.ReInit(0);
          } else {
            nbuff.Swap(rbuff_ext);
          }
        }
      }

      // Copy new data.
      totSize = totSize_new;
      arr.Swap(nbuff);
    }

    {  // Split comm.  O( log(p) ) ??
      MPI_Comm scomm;
      #pragma omp critical(SCTL_COMM_DUP)
      MPI_Comm_split(comm, myrank <= split_id, myrank, &scomm);
      #pragma omp critical(SCTL_COMM_DUP)
      if (free_comm) MPI_Comm_free(&comm);
      comm = scomm;
      free_comm = true;

      npes = (myrank <= split_id ? split_id + 1 : npes - split_id - 1);
      myrank = (myrank <= split_id ? myrank : myrank - split_id - 1);
    }
  }
  #pragma omp critical(SCTL_COMM_DUP)
  if (free_comm) MPI_Comm_free(&comm);

  SortedElem = arr;
  PartitionW<Type>(SortedElem);
#else
  SortedElem = arr_;
  std::sort(SortedElem.begin(), SortedElem.begin() + SortedElem.Dim(), comp);
#endif
}

}  // end namespace

#endif // _SCTL_COMM_TXX_
