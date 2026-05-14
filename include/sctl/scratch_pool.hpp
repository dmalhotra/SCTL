#ifndef _SCTL_SCRATCH_POOL_HPP_
#define _SCTL_SCRATCH_POOL_HPP_

#include <cstddef>            // for size_t
#include <vector>             // for vector

#include "sctl/common.hpp"    // for Long, Integer, sctl
#include "sctl/iterator.hpp"  // for Iterator, ConstIterator

// Maximum number of OpenMP threads (or flattened nested-team IDs) the pool
// supports. The pool pre-allocates one head-pointer slot per slot at
// construction, costing `SCTL_SCRATCH_POOL_MAX_THREADS * sizeof(void*)` bytes
// per pool instance (~32 KB at the default). Override at compile time with
// `-DSCTL_SCRATCH_POOL_MAX_THREADS=N` if you need more, or to shrink the
// per-instance overhead when stamping out many pools.
#ifndef SCTL_SCRATCH_POOL_MAX_THREADS
#define SCTL_SCRATCH_POOL_MAX_THREADS 4096
#endif

#ifndef SCTL_SCRATCH_POOL_INIT_BYTES
#define SCTL_SCRATCH_POOL_INIT_BYTES (1LL * 1024 * 1024)
#endif

namespace sctl {

template <class ValueType> class Vector;

class ScratchPool;

namespace internal {
struct ScratchChunk {
  Iterator<char> base;
  Long capacity;
  Long top;
  Long live_count;
  ScratchChunk* prev;
};
}  // namespace internal

/**
 * RAII handle to a scratch allocation. Construction is only possible as a
 * prvalue returned by `ScratchPool::Alloc<T>(...)`. The allocation lives
 * exactly as long as the local variable holding the handle; the destructor
 * pops the stack and runs T's destructor on each element.
 *
 * Non-copyable, non-movable, no heap allocation.
 */
template <class T> class ScratchBuf {
  static_assert(alignof(T) <= (std::size_t)SCTL_MEM_ALIGN,
                "ScratchBuf<T>: alignof(T) exceeds SCTL_MEM_ALIGN; pool cannot honor the requested alignment.");
 public:
  /**
   * Allocate `count` default-constructed T's from the current thread's
   * pool. Uses `omp_get_thread_num()` to choose the pool.
   */
  explicit ScratchBuf(Long count);

  /**
   * Allocate from pool `thread_id` (rather than the calling thread's).
   * Useful when one thread prepares scratch space on behalf of another.
   */
  ScratchBuf(Long count, Integer thread_id);

  /**
   * Allocate from a user-supplied pool instead of the global default.
   * Provided mainly for tests and for callers that want isolation from
   * the global pool.
   */
  ScratchBuf(Long count, Integer thread_id, ScratchPool& pool);

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

  /**
   * Convenience: returns a non-owning Vector view of the buffer.
   * Equivalent to `Vector<T>(buf.Dim(), buf.begin(), false)`.
   */
  Vector<T> AsVector();

 private:
  ScratchPool* pool_;
  Integer tid_;
  internal::ScratchChunk* chunk_;
  Long byte_offset_;
  Long count_;
};

/**
 * Per-OpenMP-thread stack allocator. Each thread has an independent chain
 * of memory chunks; allocation is a pointer bump within the newest chunk,
 * and overflow allocates a new chunk twice the previous size. Older
 * chunks are freed automatically once all their allocations have been
 * released; the largest (newest) chunk is retained as cached working set.
 *
 * Lock-free on the hot allocation path. The only lock is on the rare
 * cold-path resize of the per-thread vector when `omp_get_max_threads()`
 * grows.
 */
class ScratchPool {
 public:
  ScratchPool();
  ~ScratchPool();
  ScratchPool(const ScratchPool&) = delete;
  ScratchPool& operator=(const ScratchPool&) = delete;

  /**
   * Global default instance. Used implicitly by `ScratchBuf<T>(n)` and
   * `ScratchBuf<T>(n, tid)` — most code should not need to touch this
   * directly.
   */
  static ScratchPool& Instance();

  /**
   * Diagnostic accessor: number of chunks currently held by thread `tid`.
   * Mainly for tests.
   */
  Long DebugChunkCount(Integer thread_id = -1) const;

  /**
   * Diagnostic accessor: total live allocations across all chunks for
   * thread `tid`. Mainly for tests.
   */
  Long DebugLiveCount(Integer thread_id = -1) const;

 private:
  template <class U> friend class ScratchBuf;

  using Chunk = internal::ScratchChunk;

  void AllocBytes(Integer tid, Long bytes, Chunk*& out_chunk, Long& out_offset);
  void FreeBytes(Integer tid, Chunk* chunk, Long byte_offset, Long bytes);

  Integer ResolveTid(Integer thread_id) const;
  void    EnsureCapacity(Integer tid);

  std::vector<Chunk*> thread_pools_;  // index = OpenMP thread id; pointer to head chunk
};

}  // namespace sctl

#endif  // _SCTL_SCRATCH_POOL_HPP_
