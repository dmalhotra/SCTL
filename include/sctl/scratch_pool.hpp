// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _SCTL_SCRATCH_POOL_HPP_
#define _SCTL_SCRATCH_POOL_HPP_

#include <cstddef>            // for size_t
#include <deque>              // the holders made incomplete for ScratchBuf below
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>

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
 *       // ... use buf[i], buf.begin(), buf.end(), buf.size() ...
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
  Long             size() const;
  T&               operator[](Long i);
  const T&         operator[](Long i) const;

  /**
   * Ask this buffer's pool to be ready to hand out `count` elements, so the chunk that needs is
   * taken now rather than during the work that wants it. This buffer is unchanged -- it is the pool
   * that is being asked.
   *
   * The question is what the pool can serve as it stands, buffers already live included, so asking
   * from a full chunk does take a new one even where the chunk would be large enough empty. That is
   * the case it is for: a caller that has just filled its chunk and wants the next round not to.
   */
  void Reserve(Long count);

  /**
   * Grow to `count` elements, when this is the top-most buffer of its pool: the last one made of
   * those still live.
   *
   * In place while this buffer's chunk has room. Otherwise the buffer moves to a larger chunk, taken
   * by the pool as for a buffer made that does not fit (the chunks double), with its elements copied
   * there if `copy`, and unspecified if not; the pool keeps that chunk, so a buffer of the next
   * round grows in place. A move invalidates the iterators of the buffer and its views (Vector): read
   * begin() again.
   *
   * The return value is the size now held: `count`, or the old size when another buffer of the pool
   * lives above this one (until that one is freed), or for a shrink, which is refused so that a
   * caller cannot hand back memory a neighbour would then overlap.
   */
  Long RequestResize(Long count, bool copy = true);

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
 * regions (so the warmed chunk stays hot for the next region); when the
 * thread ends, its chunk goes to the next thread made. Inside a
 * parallel region `Instance()` returns a separate per-thread pool, so
 * team-thread-0 does not share pages with the serial (master) pool.
 *
 * A user-constructed pool is allowed (for tests and isolation) but is NOT
 * thread-safe — concurrent allocators must use distinct pools or
 * `Instance()`.
 */
class ScratchPool {
 public:
  /**
   * Notified of a chunk after it is allocated and again before it is released, so a pool can hold
   * memory the plain allocator cannot give it. `gpu_tree`'s staging pool registers its chunks with
   * the CUDA driver this way, which keeps that knowledge out of core sctl.
   */
  using ChunkHook = void (*)(void* base, Long bytes);

  ScratchPool();

  /** A pool whose chunks are passed to `on_new` once allocated and to `on_free` before release. */
  ScratchPool(ChunkHook on_new, ChunkHook on_free);

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

  struct Recycled {};
  /** The pool of a thread, for `Instance()`: its chunk is one that a thread which ended left, if any, and it leaves
   *  its chunk the same way. */
  explicit ScratchPool(Recycled);

  /** A slice of `bytes`; a new chunk, taken when the head cannot hold it, has at least `min_cap` bytes. */
  void AllocBytes(Long bytes, Chunk*& out_chunk, Iterator<char>& out_data, Long min_cap = 0);
  void FreeBytes(Chunk* chunk, Iterator<char> data, Long bytes);
  void ReleaseChunk(Chunk* chunk);

  /** Take a chunk that can hold `bytes` if the current one cannot, and keep it. See
   *  `ScratchBuf::Reserve`, which is how callers reach this. */
  void Reserve(Long bytes);

  /** Largest size the slice at `data` could grow to, or 0 when it is not the head chunk's top-most
   *  slice. Counts only room the chunk already has. */
  Long ResizableBytes(Chunk* chunk, Iterator<char> data, Long bytes) const;

  /** Whether the slice at `data` is the top-most live slice of the pool: the top of the head chunk,
   *  or of the chunk below an empty head. */
  bool IsTopMost(Chunk* chunk, Iterator<char> data, Long bytes) const;

  /** Move the top-most slice (`IsTopMost`) to a new one of `new_bytes`, copying its `bytes` if
   *  `copy`, and free the old one: `chunk` and `data` are then those of the new slice. */
  void MoveBytes(Chunk*& chunk, Iterator<char>& data, Long bytes, Long new_bytes, bool copy);

  /** Move `top` to fit `new_bytes`, which `ResizableBytes` must have allowed. */
  void CommitResize(Chunk* chunk, Iterator<char> data, Long bytes, Long new_bytes);

  /** What a slice of `bytes` consumes: the size rounded up to `SCTL_MEM_ALIGN` (plus the debug
   *  redzone), and never zero, so that `top == base` means the chunk holds no live buffer. */
  static Long PaddedBytes(Long bytes);

  Chunk* head_{nullptr};   // eagerly allocated by the ctor; never null after construction
  ChunkHook on_new_{nullptr};
  ChunkHook on_free_{nullptr};
  bool recycled_{false};   // made by Recycled: its chunk outlives it
  Integer slot_{-1};       // the team position of the thread that made it, or -1 outside parallel regions
};

}  // namespace sctl

namespace std {
// Incomplete on purpose: a `ScratchBuf` must be a named local, released in stack order.
template <class T> class optional<sctl::ScratchBuf<T>>;
template <class T> class shared_ptr<sctl::ScratchBuf<T>>;
template <class T, class A> class list<sctl::ScratchBuf<T>, A>;
template <class T, class A> class forward_list<sctl::ScratchBuf<T>, A>;
template <class T, class A> class deque<sctl::ScratchBuf<T>, A>;
template <class K, class T, class C, class A> class map<K, sctl::ScratchBuf<T>, C, A>;
template <class K, class T, class H, class E, class A> class unordered_map<K, sctl::ScratchBuf<T>, H, E, A>;
}  // namespace std

#endif  // _SCTL_SCRATCH_POOL_HPP_
