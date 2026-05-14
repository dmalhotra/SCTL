#ifndef _SCTL_SCRATCH_POOL_TXX_
#define _SCTL_SCRATCH_POOL_TXX_

#include <omp.h>              // for omp_get_thread_num, omp_get_max_threads
#include <algorithm>          // for max
#include <new>                // for placement new
#include <type_traits>        // for is_trivially_default_constructible, is_trivially_destructible

#include "sctl/scratch_pool.hpp"
#include "sctl/common.hpp"    // for SCTL_ASSERT, SCTL_MEM_ALIGN
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"  // for Ptr2Itr, Ptr2ConstItr
#include "sctl/mem_mgr.hpp"
#include "sctl/mem_mgr.txx"   // for aligned_new, aligned_delete
#include "sctl/vector.hpp"

namespace sctl {

inline ScratchPool::ScratchPool() : thread_pools_(SCTL_SCRATCH_POOL_MAX_THREADS, nullptr) {
  // Touch the global MemoryManager so it is constructed before us. Static
  // destruction is reverse of construction, so this guarantees our dtor's
  // calls to `aligned_delete` still see a live MemoryManager.
  (void)MemoryManager::glbMemMgr();
}

inline ScratchPool::~ScratchPool() {
  for (Chunk* head : thread_pools_) {
    if (head == nullptr) continue;
    SCTL_ASSERT(head->prev == nullptr);
    SCTL_ASSERT(head->live_count == 0);
    aligned_delete(head->base);
    delete head;
  }
}

inline ScratchPool& ScratchPool::Instance() {
  static ScratchPool inst;
  return inst;
}

inline Integer ScratchPool::ResolveTid(Integer thread_id) const {
  if (thread_id < 0) {
    const Integer level = omp_get_level();
    Integer id = 0;
    for (Integer l = 1; l <= level; l++) {
      id *= omp_get_team_size(l);
      id += omp_get_ancestor_thread_num(l);
    }
    return id;
  }
  return thread_id;
}

inline void ScratchPool::EnsureCapacity(Integer tid) {
  // `thread_pools_` is fixed-size and pre-allocated at construction;
  // dynamically resizing it would race with concurrent allocators holding
  // `Chunk*&` references into the vector. Bump SCTL_SCRATCH_POOL_MAX_THREADS
  // if you need more slots.
  SCTL_ASSERT_MSG(tid < SCTL_SCRATCH_POOL_MAX_THREADS, "ScratchPool: thread ID " << tid << " exceeds pool capacity " << SCTL_SCRATCH_POOL_MAX_THREADS);
}

inline void ScratchPool::AllocBytes(Integer tid, Long bytes, Chunk*& out_chunk, Long& out_offset) {
  EnsureCapacity(tid);
  Chunk*& head = thread_pools_[tid];

  // In MEMDEBUG builds, reserve a redzone trailer after each allocation
  // and stamp it with a sentinel; verified on free to detect off-the-end
  // writes. Release builds pack allocations tight (no overhead).
#ifdef SCTL_MEMDEBUG
  constexpr Long redzone = MemoryManager::end_padding;
#else
  constexpr Long redzone = 0;
#endif
  const Long bytes_total = bytes + redzone;

  // Lazy-init the first chunk on this thread (NUMA first-touch).
  if (head == nullptr) {
    Long cap = (Long)SCTL_SCRATCH_POOL_INIT_BYTES;
    while (cap < bytes_total) cap *= 2;
    head = new Chunk{aligned_new<char>(cap), cap, 0, 0, nullptr};
  }

  // Align top up to SCTL_MEM_ALIGN.
  constexpr Long mask = (Long)SCTL_MEM_ALIGN - 1;
  Long aligned_top = (head->top + mask) & ~mask;

  if (aligned_top + bytes_total > head->capacity) {
    // Grow: new chunk 2x, or larger if the request demands it.
    Long new_cap = head->capacity * 2;
    while (new_cap < bytes_total) new_cap *= 2;
    Chunk* new_chunk = new Chunk{aligned_new<char>(new_cap), new_cap, 0, 0, head};
    head = new_chunk;
    aligned_top = 0;
  }

  out_chunk = head;
  out_offset = aligned_top;
  head->top = aligned_top + bytes_total;
  head->live_count++;

#ifdef SCTL_MEMDEBUG
  // Stamp the redzone with the sentinel; verified on free.
  for (Long i = 0; i < redzone; ++i) {
    head->base[aligned_top + bytes + i] = MemoryManager::init_mem_val;
  }
#endif
}

inline void ScratchPool::FreeBytes(Integer tid, Chunk* chunk, Long byte_offset, Long bytes) {
  Chunk*& head = thread_pools_[tid];
  SCTL_ASSERT(chunk->live_count > 0);

#ifdef SCTL_MEMDEBUG
  // Verify the redzone trailer is intact.
  for (Long i = 0; i < MemoryManager::end_padding; ++i) {
    SCTL_ASSERT_MSG(chunk->base[byte_offset + bytes + i] == MemoryManager::init_mem_val,
                    "ScratchBuf: out-of-bounds write past buffer end detected.");
  }
#else
  SCTL_UNUSED(bytes);
#endif

  --chunk->live_count;

  if (chunk == head) {
    // Pop the stack. LIFO is enforced by RAII.
#ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(byte_offset <= chunk->top);
#endif
    chunk->top = byte_offset;
    return;
  }

  // Free non-head chunks that have drained.
  if (chunk != head && chunk->live_count == 0) {
    // Splice `chunk` out of the prev chain.
    SCTL_ASSERT(head->prev == chunk);
    head->prev = chunk->prev;
    aligned_delete(chunk->base);
    delete chunk;
  }
}

inline Long ScratchPool::DebugChunkCount(Integer thread_id) const {
  Integer tid = (thread_id < 0) ? (Integer)omp_get_thread_num() : thread_id;
  if (tid >= (Integer)thread_pools_.size()) return 0;
  Long n = 0;
  for (Chunk* c = thread_pools_[tid]; c; c = c->prev) ++n;
  return n;
}

inline Long ScratchPool::DebugLiveCount(Integer thread_id) const {
  Integer tid = (thread_id < 0) ? (Integer)omp_get_thread_num() : thread_id;
  if (tid >= (Integer)thread_pools_.size()) return 0;
  Long n = 0;
  for (Chunk* c = thread_pools_[tid]; c; c = c->prev) n += c->live_count;
  return n;
}

template <class T>
inline ScratchBuf<T>::ScratchBuf(Long count)
  : ScratchBuf(count, -1, ScratchPool::Instance()) {}

template <class T>
inline ScratchBuf<T>::ScratchBuf(Long count, Integer thread_id)
  : ScratchBuf(count, thread_id, ScratchPool::Instance()) {}

template <class T>
inline ScratchBuf<T>::ScratchBuf(Long count, Integer thread_id, ScratchPool& pool)
  : pool_(&pool),
    tid_(pool.ResolveTid(thread_id)),
    chunk_(nullptr),
    byte_offset_(0),
    count_(count) {
  pool_->AllocBytes(tid_, count * (Long)sizeof(T), chunk_, byte_offset_);
  // For trivial types we leave the bytes uninitialized — the scratch buffer
  // is expected to be written before it is read. Non-trivial types get their
  // default constructor called.
  if constexpr (!std::is_trivially_default_constructible<T>::value) {
    T* arr = reinterpret_cast<T*>(&chunk_->base[byte_offset_]);
    for (Long i = 0; i < count; ++i) new (arr + i) T();
  }
}

template <class T>
inline ScratchBuf<T>::~ScratchBuf() {
  if constexpr (!std::is_trivially_destructible<T>::value) {
    T* arr = reinterpret_cast<T*>(&chunk_->base[byte_offset_]);
    for (Long i = count_ - 1; i >= 0; --i) arr[i].~T();
  }
  pool_->FreeBytes(tid_, chunk_, byte_offset_, count_ * (Long)sizeof(T));
}

template <class T>
inline Iterator<T> ScratchBuf<T>::begin() {
  return Ptr2Itr<T>(&chunk_->base[byte_offset_], count_);
}

template <class T>
inline Iterator<T> ScratchBuf<T>::end() {
  return begin() + count_;
}

template <class T>
inline ConstIterator<T> ScratchBuf<T>::begin() const {
  return Ptr2ConstItr<T>(&chunk_->base[byte_offset_], count_);
}

template <class T>
inline ConstIterator<T> ScratchBuf<T>::end() const {
  return begin() + count_;
}

template <class T>
inline Long ScratchBuf<T>::Dim() const { return count_; }

template <class T>
inline T& ScratchBuf<T>::operator[](Long i) {
  return reinterpret_cast<T*>(&chunk_->base[byte_offset_])[i];
}

template <class T>
inline const T& ScratchBuf<T>::operator[](Long i) const {
  return reinterpret_cast<const T*>(&chunk_->base[byte_offset_])[i];
}

template <class T>
inline Vector<T> ScratchBuf<T>::AsVector() {
  return Vector<T>(count_, begin(), false);
}

}  // namespace sctl

#endif  // _SCTL_SCRATCH_POOL_TXX_
