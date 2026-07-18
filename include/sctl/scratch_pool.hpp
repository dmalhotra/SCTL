#ifndef _SCTL_SCRATCH_POOL_HPP_
#define _SCTL_SCRATCH_POOL_HPP_

#include <cstddef>            // for size_t

#include "sctl/common.hpp"    // for Long, Integer, sctl
#include "sctl/iterator.hpp"  // for Iterator, ConstIterator

// Size of the first chunk a thread allocates from its pool. Subsequent chunks
// double in size on overflow (or grow further if a single request demands).
// Override at compile time with -DSCTL_SCRATCH_POOL_INIT_BYTES=N.
#ifndef SCTL_SCRATCH_POOL_INIT_BYTES
#define SCTL_SCRATCH_POOL_INIT_BYTES (1LL * 1024 * 1024)
#endif

namespace sctl {

template <class ValueType> class Vector;

class ScratchPool;

namespace internal {

// One chunk in a ScratchPool's linked list of memory blocks. See the txx for
// the design rationale (alignment, size-rounding invariant, etc.).
struct alignas(SCTL_MEM_ALIGN) ScratchChunk {
  Iterator<char> base;
  Iterator<char> top;
  Iterator<char> end;
  ScratchChunk*  prev;
#ifdef SCTL_MEMDEBUG
  Long           live_count;
#endif

  ScratchChunk(Iterator<char> base, Iterator<char> top, Iterator<char> end, ScratchChunk* prev);
};

}  // namespace internal

/**
 * RAII handle to a stack-allocated scratch buffer carved out of a `ScratchPool`.
 *
 *     {
 *       ScratchBuf<double> buf(N);
 *       // ... use buf[i], buf.begin(), buf.end(), buf.Dim() ...
 *     }                                  // buf is freed here
 *
 * Rules:
 *   - **LIFO only.** Within a pool, ScratchBufs must be destroyed in the
 *     reverse order of construction. Lexical scoping makes this automatic.
 *   - **Stack-only.** Non-copyable, non-movable, never heap-allocated.
 *   - **`alignof(T) <= SCTL_MEM_ALIGN`** (enforced at compile time).
 *
 * Trivial element types are left uninitialized at construction; non-trivial
 * types are default-constructed and destroyed in reverse on free.
 */
template <class T> class ScratchBuf {
  static_assert(alignof(T) <= (std::size_t)SCTL_MEM_ALIGN,
                "ScratchBuf<T>: alignof(T) exceeds SCTL_MEM_ALIGN; pool cannot honor the requested alignment.");
 public:
  /** Allocate `count` T's from the calling thread's pool (`ScratchPool::Instance()`). */
  explicit ScratchBuf(Long count);

  /**
   * Allocate from a user-supplied pool instead of the thread-local default.
   * Provided for tests and isolation. The user pool is NOT thread-safe.
   */
  ScratchBuf(Long count, ScratchPool& pool);

  ScratchBuf() = delete;
  ScratchBuf(const ScratchBuf&) = delete;
  ScratchBuf& operator=(const ScratchBuf&) = delete;
  ScratchBuf(ScratchBuf&&) = delete;
  ScratchBuf& operator=(ScratchBuf&&) = delete;
  static void* operator new(std::size_t)   = delete;
  static void* operator new[](std::size_t) = delete;
  static void  operator delete(void*)      = delete;
  static void  operator delete[](void*)    = delete;

  ~ScratchBuf();

  Iterator<T>      begin();
  Iterator<T>      end();
  ConstIterator<T> begin() const;
  ConstIterator<T> end()   const;
  Long             Dim() const;
  T&               operator[](Long i);
  const T&         operator[](Long i) const;

 private:
  ScratchPool*            pool_;
  internal::ScratchChunk* chunk_;
  Iterator<T>             data_;
  Long                    count_;
};

/**
 * Per-thread stack allocator backing `ScratchBuf`. Allocation is a pointer
 * bump within a per-thread chunk; lock-free, NUMA-local first-touch.
 *
 * Use `ScratchPool::Instance()` to get the calling thread's pool. The pool
 * persists for the thread's lifetime, including across OpenMP parallel
 * regions (so the warmed chunk stays hot for the next region). Inside a
 * parallel region `Instance()` returns a separate per-thread pool, so
 * team-thread-0 does not share pages with the serial (master) pool.
 *
 * A user-constructed pool is allowed (for tests and isolation) but is NOT
 * thread-safe — concurrent allocators must use distinct pools or
 * `Instance()`.
 */
class ScratchPool {
 public:
  ScratchPool();
  ~ScratchPool();
  ScratchPool(const ScratchPool&) = delete;
  ScratchPool& operator=(const ScratchPool&) = delete;

  /** Returns the calling thread's pool (thread_local storage). */
  static ScratchPool& Instance();

  /** Diagnostic: number of chunks currently held. Mainly for tests. */
  Long DebugChunkCount() const;

  /**
   * Diagnostic: number of live allocations. Exact under SCTL_MEMDEBUG;
   * release builds return 0 when known-empty, -1 otherwise.
   */
  Long DebugLiveCount() const;

 private:
  template <class U> friend class ScratchBuf;

  using Chunk = internal::ScratchChunk;

  void AllocBytes(Long bytes, Chunk*& out_chunk, Iterator<char>& out_data);
  void FreeBytes(Chunk* chunk, Iterator<char> data, Long bytes);

  Chunk* head_{nullptr};   // eagerly allocated by the ctor; never null after construction
};

}  // namespace sctl

#endif  // _SCTL_SCRATCH_POOL_HPP_
