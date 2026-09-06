// Pooled scratch memory for backend (device or host) temporaries: DeviceScratch is the
// thrust-iterator analogue of sctl::ScratchBuf, and DeviceScratchAllocator routes thrust's own
// temporary storage (cub scan/sort/select scratch) to the same pool.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_

#include <thrust/device_ptr.h>

#include <cstddef>
#include <type_traits>
#include <utility>

#include "sctl/common.hpp"

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
 * Long-lived working storage for an array that is rebuilt on every call. Unlike `DeviceScratch`
 * (fixed size, LIFO, released at scope exit) this grows on demand and is retained for the process,
 * so an array that ends up the same size each call stops allocating after the first. Swap into it
 * instead of assigning a fresh vector and the storage is recycled rather than freed and retaken --
 * which matters because a release is an allocator round trip and drains the device.
 *
 * `Tag` separates independent arrays so two of them do not share one buffer. Retention is deliberate,
 * as in `DeviceScratchPool`. Like the pool, the buffers are process-wide: a caller that runs two
 * builds concurrently in one process must not share a tag between them.
 */
template <class T, template <class...> class DevVec, auto Tag> DevVec<T>& PersistentBuffer();

/**
 * Copy `n` elements of device storage into a host buffer, staged through a retained pinned buffer.
 *
 * A caller-owned destination is pageable and usually freshly allocated, which costs twice over:
 * the driver cannot DMA into it, and it faults in a page at a time inside the driver's copy. Taking
 * the DMA into one pinned buffer that is kept and regrown, then filling the destination with the
 * host threads, avoids both. No `Tag`: the staging buffer is live only within the call, so all
 * uses of a given `T` share one.
 *
 * On host backends `src` is already a host pointer and this is a plain copy.
 */
template <class SrcPtr, class T> void deviceToHost(SrcPtr src, Long n, T* dst);

}  // namespace detail

/**
 * Bump allocator backing `DeviceScratch`, one instance per backend (`Instance()`).
 *
 * Allocation is a pointer bump inside a chunk; on overflow a new chunk is added (doubling) and
 * older chunks stay live, so outstanding pointers remain valid. Chunks are **never** released:
 * retaining them is the point -- `cudaMalloc`/`cudaFree` cost ~1 ms per large block and dominate
 * the tree build otherwise. Peak reservation is bounded by 2x the high-water mark.
 *
 * Not thread-safe: one pool serves the thread issuing the backend calls.
 */
template <template <class...> class DevVec> class DeviceScratchPool {
 public:
  static constexpr Long ALIGN = 256;  // cudaMalloc's guarantee; rounding sizes keeps `top` aligned

  /** The pool for this backend. */
  static DeviceScratchPool& Instance();

  DeviceScratchPool(const DeviceScratchPool&) = delete;
  DeviceScratchPool& operator=(const DeviceScratchPool&) = delete;

 private:
  template <class, template <class...> class> friend class DeviceScratch;
  template <template <class...> class> friend class DeviceScratchAllocator;

  /** One chunk of the pool; `DeviceScratch` holds the chunk its slice came from. */
  struct Chunk {
    DevVec<char>* buf;  // leaked by design: freeing device memory at exit races CUDA teardown
    char* base;
    char* top;
    char* end;
    Chunk* prev;
  };

  /** Carve `bytes` off the pool; returns the owning chunk and the slice. */
  std::pair<Chunk*, char*> AllocBytes(Long bytes);

  /** Return a slice (LIFO: it must be the last one taken from `chunk`). */
  void FreeBytes(Chunk* chunk, char* p, Long bytes);

  /** Same, with the owning chunk located by the LIFO invariant. */
  void FreeBytes(char* p, Long bytes);

  DeviceScratchPool() = default;
  void NewChunk(Long need);

  Chunk* head_{nullptr};
};

/**
 * RAII handle to a scratch buffer carved out of `DeviceScratchPool<DevVec>`.
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

  /** Allocate from a user-supplied pool instead (tests, isolation). */
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

#include "sctl/experimental/device_scratch.txx"

#endif  // _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_
