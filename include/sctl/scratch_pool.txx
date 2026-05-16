#ifndef _SCTL_SCRATCH_POOL_TXX_
#define _SCTL_SCRATCH_POOL_TXX_

#include <algorithm>          // for max
#include <cstdlib>            // for std::aligned_alloc, std::free
#include <new>                // for placement new
#include <type_traits>        // for is_trivially_default_constructible, is_trivially_destructible

#include "sctl/scratch_pool.hpp"
#include "sctl/common.hpp"    // for SCTL_ASSERT, SCTL_MEM_ALIGN
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"  // for Iterator arithmetic, Ptr2Itr
#include "sctl/mem_mgr.hpp"   // for MemoryManager::{end_padding,init_mem_val} (constexpr)
#include "sctl/vector.hpp"

namespace sctl {

namespace internal {

// Newer chunks push at head via `prev`; allocations bump `top` within
// [base, end). `alignas(SCTL_MEM_ALIGN)` keeps the four hot fields on one
// cache line (otherwise ~25% of chunks would straddle two). Alignment of
// `top` is preserved by induction: `base` is aligned, every alloc consumes
// a multiple of SCTL_MEM_ALIGN (see AllocBytes).
inline ScratchChunk::ScratchChunk(Iterator<char> base, Iterator<char> top, Iterator<char> end, ScratchChunk* prev)
  : base(base), top(top), end(end), prev(prev)
#ifdef SCTL_MEMDEBUG
  , live_count(0)
#endif
{}

}  // namespace internal

inline ScratchPool::ScratchPool() = default;

inline ScratchPool::~ScratchPool() {
  if (head_ == nullptr) return;
#ifdef SCTL_MEMDEBUG
  SCTL_ASSERT(head_->live_count == 0);
#endif
  // LIFO discipline drains all non-head chunks, so only head remains.
  SCTL_ASSERT(head_->prev == nullptr);
  std::free(&head_->base[0]);
  delete head_;
}

inline ScratchPool& ScratchPool::Instance() {
  thread_local ScratchPool inst;
  return inst;
}

// `[[gnu::always_inline]]` overrides GCC's cost model so the hot path
// inlines into callers (+8 cycles per call otherwise).
//
// Alignment trick: round the allocation *size* up to SCTL_MEM_ALIGN instead
// of aligning the *start* pointer. Since `base` is aligned and every alloc
// consumes a multiple of SCTL_MEM_ALIGN, `top` stays aligned by induction —
// no per-call `(top + mask) & ~mask`.
[[gnu::always_inline]] inline void ScratchPool::AllocBytes(Long bytes, Chunk*& out_chunk, Iterator<char>& out_data) {
#ifdef SCTL_MEMDEBUG
  SCTL_ASSERT(bytes >= 0);
  constexpr Long redzone = MemoryManager::end_padding;
#else
  constexpr Long redzone = 0;
#endif

  constexpr Long align_mask = (Long)SCTL_MEM_ALIGN - 1;
  const Long bytes_padded = (bytes + redzone + align_mask) & ~align_mask;
  // Short-circuit on the lazy-init case (head_ == nullptr): avoids null
  // pointer arithmetic (UB / -fsanitize=pointer-subtract,pointer-compare).
  // Otherwise `top` and `end` point into the same chunk so the subtract
  // and comparison are well-defined.
  Iterator<char> alloc_start;
  if (__builtin_expect(head_ == nullptr || bytes_padded > head_->end - head_->top, 0)) {
    // Overflow path. Also taken on the first allocation (head_ == nullptr).
    // `std::aligned_alloc` (not `aligned_new`) so libc returns virtual pages
    // that fault in on the writing thread — NUMA-local first-touch.
    const Long prev_cap = (head_ == nullptr) ? 0 : (head_->end - head_->base);
    Long new_cap = std::max<Long>((Long)SCTL_SCRATCH_POOL_INIT_BYTES, prev_cap * 2);
    while (new_cap < bytes_padded) new_cap *= 2;
    new_cap = (new_cap + align_mask) & ~align_mask;  // aligned_alloc requires size % alignment == 0
    void* raw = std::aligned_alloc(SCTL_MEM_ALIGN, new_cap);
    SCTL_ASSERT_MSG(raw != nullptr, "ScratchPool: chunk allocation failed.");
    Iterator<char> new_base = Ptr2Itr<char>(static_cast<char*>(raw), new_cap);
    head_ = new Chunk(new_base, new_base, new_base + new_cap, head_);
    alloc_start = new_base;
  } else {
    alloc_start = head_->top;
  }
  out_chunk = head_;
  out_data = alloc_start;
  head_->top = alloc_start + bytes_padded;

#ifdef SCTL_MEMDEBUG
  head_->live_count++;
  SCTL_ASSERT_MSG(((alloc_start - head_->base) & (Long)align_mask) == 0,
                  "ScratchPool: alignment invariant violated.");
  // Stamp the redzone trailer; verified in FreeBytes.
  Iterator<char> redzone_start = alloc_start + bytes;
  for (Long i = 0; i < redzone; ++i) redzone_start[i] = MemoryManager::init_mem_val;
#endif
}

[[gnu::always_inline]] inline void ScratchPool::FreeBytes(Chunk* chunk, Iterator<char> data, Long bytes) {
#ifdef SCTL_MEMDEBUG
  SCTL_ASSERT(chunk->live_count > 0);
  constexpr Long redzone = MemoryManager::end_padding;
  // Check redzone trailer for out-of-bounds writes.
  Iterator<char> redzone_start = data + bytes;
  for (Long i = 0; i < redzone; ++i) {
    SCTL_ASSERT_MSG((char)redzone_start[i] == MemoryManager::init_mem_val,
                    "ScratchBuf: out-of-bounds write past buffer end detected.");
  }
  --chunk->live_count;
#else
  SCTL_UNUSED(bytes);
#endif

  if (__builtin_expect(chunk == head_, 1)) {
#ifdef SCTL_MEMDEBUG
    // Strict LIFO: this alloc's end exactly matches the chunk's top (no
    // inter-allocation padding thanks to size-rounding in AllocBytes).
    constexpr Long align_mask = (Long)SCTL_MEM_ALIGN - 1;
    const Long bytes_padded = (bytes + redzone + align_mask) & ~align_mask;
    SCTL_ASSERT_MSG(data + bytes_padded == chunk->top,
                    "ScratchBuf: LIFO violation (free out of order).");
#endif
    chunk->top = data;
    return;
  }

  // Non-head chunk. Under LIFO, the only allocation in a non-head chunk that
  // can be freed is its first one (at `base`) — everything after it has
  // already been freed. Splice the chunk out.
  if (data == chunk->base) {
#ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(chunk->live_count == 0);
    SCTL_ASSERT(head_->prev == chunk);
#endif
    head_->prev = chunk->prev;
    std::free(&chunk->base[0]);
    delete chunk;
  }
}

inline Long ScratchPool::DebugChunkCount() const {
  Long n = 0;
  for (const Chunk* c = head_; c != nullptr; c = c->prev) ++n;
  return n;
}

inline Long ScratchPool::DebugLiveCount() const {
#ifdef SCTL_MEMDEBUG
  Long n = 0;
  for (const Chunk* c = head_; c != nullptr; c = c->prev) n += c->live_count;
  return n;
#else
  // No per-chunk counter in release; report 0 only when known-empty.
  if (head_ == nullptr) return 0;
  return (head_->top == head_->base) ? 0 : -1;
#endif
}

// ---------------------------------------------------------------------------
// ScratchBuf
// ---------------------------------------------------------------------------

template <class T>
[[gnu::always_inline]] inline ScratchBuf<T>::ScratchBuf(Long count)
  : ScratchBuf(count, ScratchPool::Instance()) {}

template <class T>
[[gnu::always_inline]] inline ScratchBuf<T>::ScratchBuf(Long count, ScratchPool& pool)
  : pool_(&pool), count_(count) {
  Iterator<char> raw;
  pool_->AllocBytes(count * (Long)sizeof(T), chunk_, raw);
  data_ = Iterator<T>(raw);
  if constexpr (!std::is_trivially_default_constructible<T>::value) {
    T* elem = &data_[0];
    for (Long i = 0; i < count; ++i) new (elem + i) T();
  }
}

template <class T>
[[gnu::always_inline]] inline ScratchBuf<T>::~ScratchBuf() {
  if constexpr (!std::is_trivially_destructible<T>::value) {
    T* elem = &data_[0];
    for (Long i = count_ - 1; i >= 0; --i) elem[i].~T();
  }
  pool_->FreeBytes(chunk_, Iterator<char>(data_), count_ * (Long)sizeof(T));
}

template <class T>
inline Iterator<T> ScratchBuf<T>::begin() { return data_; }

template <class T>
inline Iterator<T> ScratchBuf<T>::end() { return data_ + count_; }

template <class T>
inline ConstIterator<T> ScratchBuf<T>::begin() const { return data_; }

template <class T>
inline ConstIterator<T> ScratchBuf<T>::end() const { return data_ + count_; }

template <class T>
inline Long ScratchBuf<T>::Dim() const { return count_; }

template <class T>
inline T& ScratchBuf<T>::operator[](Long i) { return data_[i]; }

template <class T>
inline const T& ScratchBuf<T>::operator[](Long i) const { return data_[i]; }

// Defined here (not vector.txx): vector.hpp forward-declares ScratchBuf, but
// constructing the view needs ScratchBuf complete. The `disable_reinit`
// default is declared in vector.hpp.
template <class T>
inline Vector<T>::Vector(ScratchBuf<T>& buf, bool disable_reinit)
  : Vector(buf.Dim(), buf.begin(), /*own_data=*/false, disable_reinit) {}

}  // namespace sctl

#endif  // _SCTL_SCRATCH_POOL_TXX_
