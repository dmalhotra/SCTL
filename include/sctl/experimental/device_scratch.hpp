// Pooled scratch memory for backend (device or host) temporaries: DeviceScratch is the
// thrust-iterator analogue of sctl::ScratchBuf, and DeviceScratchAllocator routes thrust's own
// temporary storage (cub scan/sort/select scratch) to the same pool.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_

#include <thrust/device_ptr.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>  // the host-register and copy calls below; not otherwise declared
#elif defined(__CUDACC__)
#include <cuda_runtime.h>     // the host-register and copy calls below; not otherwise declared
#endif

#include <cstddef>
#include <deque>  // the holders made incomplete for DeviceScratch below
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "sctl/common.hpp"
#include "sctl/scratch_pool.hpp"  // for ScratchPool

// First chunk size; later chunks double (or grow to fit a single request).
#ifndef SCTL_DEVICE_SCRATCH_INIT_BYTES
#define SCTL_DEVICE_SCRATCH_INIT_BYTES (4LL * 1024 * 1024)
#endif

namespace gpu_tree {

using sctl::Long;

namespace detail {

// True iff `Vec::data()` returns a thrust::device_ptr (i.e. Vec is GPU-resident).
template <class T> struct is_device_ptr                        : std::false_type {};
template <class T> struct is_device_ptr<thrust::device_ptr<T>> : std::true_type  {};

template <class Vec>
inline constexpr bool is_device_vector_v = is_device_ptr<typename std::decay<decltype(std::declval<Vec>().data())>::type>::value;

// Iterator thrust dispatches on: device_ptr<T> for the device backend, plain T* for the host one.
template <class T, template <class...> class DevVec>
using ScratchIterator = std::conditional_t<is_device_vector_v<DevVec<T>>, thrust::device_ptr<T>, T*>;

/**
 * Copy `n` elements of device storage into a host buffer, staged through a retained pinned buffer.
 *
 * A caller-owned destination is pageable and usually freshly allocated, which costs twice over:
 * the driver cannot DMA into it, and it faults in a page at a time inside the driver's copy. Taking
 * the DMA into pinned memory, then filling the destination with the host threads, avoids both. The
 * staging buffer comes from `pinnedStagingPool()` and is live only within the call.
 *
 * On host backends `src` is already a host pointer and this is a plain copy.
 */
template <class SrcPtr, class DstPtr> void deviceToHost(SrcPtr src, Long n, DstPtr dst);

/**
 * Page-locked host staging memory, as one byte-addressed arena shared by every element type rather
 * than a buffer per type. Chunks are registered with the driver so a DMA can write into them, and
 * registration follows the pages being faulted in: registering them cold instead costs several
 * times as much and places the whole chunk on the faulting thread's NUMA node.
 *
 * The retained chunk grows to the largest request and is never shrunk, so one big request leaves
 * that much host memory page-locked for the process.
 *
 * Not thread-safe, like any pool outside `ScratchPool::Instance()`: take from it outside parallel
 * regions, one buffer at a time, and release the buffer before returning to the caller.
 */
sctl::ScratchPool& pinnedStagingPool();

}  // namespace detail

/**
 * Bump allocator for backend scratch memory, one instance per backend (`Instance()`).
 *
 * Allocation is a pointer bump inside a chunk; on overflow a new chunk is added (doubling) and
 * older chunks stay live, so outstanding pointers remain valid. The head chunk is retained however
 * empty it gets -- that is the point, since releasing backend memory costs ~1 ms per large block.
 * An older chunk is released once it empties, so after a few rounds of use the pool converges on
 * one chunk holding the high-water mark.
 *
 * Not thread-safe: one pool serves the thread issuing the backend calls.
 */
template <template <class...> class DevVec> class DeviceScratchPool {
 public:
  // Chunk bases and allocation sizes are both rounded to this, so every allocation starts aligned for any
  // type the pool hands out. The device's blocks arrive 256-aligned already; the host's give only 16.
  static constexpr Long ALIGN = SCTL_MEM_ALIGN;

  /** The pool for this backend. */
  static DeviceScratchPool& Instance();

  /** A pool of one's own. For tests and isolation; not thread-safe, like `Instance()`. */
  DeviceScratchPool() = default;

  ~DeviceScratchPool();

  DeviceScratchPool(const DeviceScratchPool&) = delete;
  DeviceScratchPool& operator=(const DeviceScratchPool&) = delete;

  /** Diagnostic: number of chunks currently held. Mainly for tests. */
  Long DebugChunkCount() const;

  /**
   * Diagnostic: number of live allocations. Exact under SCTL_MEMDEBUG;
   * release builds return 0 when known-empty, -1 otherwise.
   */
  Long DebugLiveCount() const;

 private:
  template <class, template <class...> class> friend class DeviceScratch;
  template <template <class...> class> friend class DeviceScratchAllocator;

  /** One block of backend memory: allocations are taken from it in order and freed in reverse order. */
  struct alignas(SCTL_MEM_ALIGN) Chunk {
    DevVec<char>* buf;  // released with the chunk; at exit only on host backends
    char* base;
    char* top;
    char* end;
    Chunk* prev;
    Long live_count;  // maintained under SCTL_MEMDEBUG only, as ScratchPool does
  };

  /** Allocate `bytes` from the pool; returns the chunk the allocation came from and its pointer. */
  std::pair<Chunk*, char*> AllocBytes(Long bytes);

  /** Free an allocation (LIFO: it must be the last one taken from `chunk`). */
  void FreeBytes(Chunk* chunk, char* p, Long bytes);

  /** Free the allocation and release the chunk if that emptied it. */
  void Rewind(Chunk* chunk, char* p);

  /** Bytes of trailer past each allocation: nonzero only under SCTL_MEMDEBUG on a host backend. */
  static constexpr Long Redzone();

  /** Stamp the trailer past an allocation, so `CheckRedzone` can tell it was written over. */
  static void StampRedzone(char* p, Long bytes);

  /** Verify the trailer stamped by `AllocBytes`. */
  static void CheckRedzone(const char* p, Long bytes);

  /** What an allocation of `bytes` consumes: rounded up to `ALIGN`, and never zero, so that
   *  `top == base` means the chunk holds no live allocation. */
  static Long PaddedBytes(Long bytes);

  /** Same, with the owning chunk located by the LIFO invariant. */
  void FreeBytes(char* p, Long bytes);

  void NewChunk(Long need);

  Chunk* head_{nullptr};  ///< first chunk taken on first use: a static must not reach CUDA during startup
};

/**
 * RAII handle to a scratch buffer allocated from `DeviceScratchPool<DevVec>`.
 *
 *     {
 *       DeviceScratch<Morton<3>, DevVec> buf(n);
 *       thrust::for_each(buf.begin(), buf.end(), ...);   // thrust iterators
 *     }                                                  // freed here
 *
 * Same rules as sctl::ScratchBuf: LIFO destruction (lexical scoping gives it), stack-only,
 * non-copyable, non-movable. Contents are uninitialized -- the element type must be trivial,
 * since no constructor can run on backend memory.
 */
template <class T, template <class...> class DevVec> class DeviceScratch {
  using Pool = DeviceScratchPool<DevVec>;
  static_assert(std::is_trivially_copyable<T>::value, "DeviceScratch<T>: T must be trivially copyable.");
  static_assert(alignof(T) <= (std::size_t)Pool::ALIGN, "DeviceScratch<T>: alignof(T) exceeds the pool alignment.");

 public:
  using iterator = detail::ScratchIterator<T, DevVec>;

  /** Allocate `count` T's from this backend's pool. */
  explicit DeviceScratch(Long count);

  /** Allocate from a caller-supplied pool instead. For tests and isolation. */
  DeviceScratch(Long count, Pool& pool);

  ~DeviceScratch();

  DeviceScratch() = delete;
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;
  DeviceScratch(DeviceScratch&&) = delete;
  DeviceScratch& operator=(DeviceScratch&&) = delete;
  static void* operator new(std::size_t) = delete;
  static void* operator new[](std::size_t) = delete;
  static void operator delete(void*) = delete;
  static void operator delete[](void*) = delete;

  iterator begin() const;
  iterator end() const;
  iterator data() const;  // thrust convention: use raw_pointer_cast for functors
  Long Dim() const;

 private:
  Pool* pool_;
  typename Pool::Chunk* chunk_;
  T* data_;
  Long count_;
};

/**
 * Allocator adaptor handing thrust's temporary storage to the same pool, e.g.
 * `thrust::cuda::par(alloc)`. Thrust frees its temporaries in LIFO order, so it fits the pool.
 */
template <template <class...> class DevVec> class DeviceScratchAllocator {
 public:
  using value_type = char;

  char* allocate(std::ptrdiff_t n);
  void deallocate(char* p, std::size_t n);
};

}  // namespace gpu_tree

namespace std {
// Incomplete on purpose: a `DeviceScratch` must be a named local, released in stack order.
template <class T, template <class...> class V> class optional<gpu_tree::DeviceScratch<T, V>>;
template <class T, template <class...> class V> class shared_ptr<gpu_tree::DeviceScratch<T, V>>;
template <class T, template <class...> class V, class A> class list<gpu_tree::DeviceScratch<T, V>, A>;
template <class T, template <class...> class V, class A> class forward_list<gpu_tree::DeviceScratch<T, V>, A>;
template <class T, template <class...> class V, class A> class deque<gpu_tree::DeviceScratch<T, V>, A>;
template <class K, class T, template <class...> class V, class C, class A> class map<K, gpu_tree::DeviceScratch<T, V>, C, A>;
template <class K, class T, template <class...> class V, class H, class E, class A> class unordered_map<K, gpu_tree::DeviceScratch<T, V>, H, E, A>;
}  // namespace std

#include "sctl/experimental/device_scratch.txx"

#endif  // _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_
