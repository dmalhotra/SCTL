// Pooled scratch memory for backend (device or host) temporaries: DeviceScratch is the
// thrust-iterator analogue of sctl::ScratchBuf, and DeviceScratchAllocator routes thrust's own
// temporary storage (cub scan/sort/select scratch) to the same pool.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_

#include <thrust/device_ptr.h>

#include <algorithm>
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
template <class T, template <class...> class DeviceVector>
using ScratchIterator = std::conditional_t<is_device_vector_v<DeviceVector<T>>, thrust::device_ptr<T>, T*>;

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
template <template <class...> class DeviceVector> class DeviceScratchPool {
 public:
  static constexpr Long ALIGN = 256;  // cudaMalloc's guarantee; rounding sizes keeps `top` aligned

  static DeviceScratchPool& Instance() {
    static DeviceScratchPool pool;
    return pool;
  }

  DeviceScratchPool(const DeviceScratchPool&) = delete;
  DeviceScratchPool& operator=(const DeviceScratchPool&) = delete;

  struct Chunk {
    DeviceVector<char>* buf;  // leaked by design: freeing device memory at exit races CUDA teardown
    char* base;
    char* top;
    char* end;
    Chunk* prev;
  };

  std::pair<Chunk*, char*> AllocBytes(Long bytes) {
    const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
    if (head_ == nullptr || need > head_->end - head_->top) NewChunk(need);
    char* const p = head_->top;
    head_->top += need;
    high_water_ = std::max(high_water_, Live());
    return {head_, p};
  }

  void FreeBytes(Chunk* chunk, char* p, Long bytes) {
    const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
    SCTL_ASSERT_MSG(chunk->top == p + need, "DeviceScratch: LIFO violation (free out of order).");
    chunk->top = p;
  }

  void FreeBytes(char* p, Long bytes) {  // owning chunk located by the LIFO invariant
    const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
    for (Chunk* c = head_; c; c = c->prev) {
      if (c->top == p + need) { c->top = p; return; }
    }
    SCTL_ASSERT_MSG(false, "DeviceScratch: LIFO violation (free out of order).");
  }

  Long Reserved() const {  // bytes held by the pool
    Long n = 0;
    for (const Chunk* c = head_; c; c = c->prev) n += c->end - c->base;
    return n;
  }
  Long Live() const {  // bytes currently handed out
    Long n = 0;
    for (const Chunk* c = head_; c; c = c->prev) n += c->top - c->base;
    return n;
  }
  Long HighWater() const { return high_water_; }
  Long ChunkCount() const {
    Long n = 0;
    for (const Chunk* c = head_; c; c = c->prev) ++n;
    return n;
  }

 private:
  DeviceScratchPool() = default;

  void NewChunk(Long need) {
    const Long prev_cap = head_ ? head_->end - head_->base : 0;
    Long cap = std::max<Long>((Long)SCTL_DEVICE_SCRATCH_INIT_BYTES, prev_cap * 2);
    while (cap < need) cap *= 2;
    auto* buf = new DeviceVector<char>(cap);
    char* const base = thrust::raw_pointer_cast(buf->data());
    head_ = new Chunk{buf, base, base, base + cap, head_};
  }

  Chunk* head_{nullptr};
  Long high_water_{0};
};

/**
 * RAII handle to a scratch buffer carved out of `DeviceScratchPool<DeviceVector>`.
 *
 *     {
 *       DeviceScratch<Morton<3>, DeviceVector> buf(n);
 *       thrust::for_each(buf.begin(), buf.end(), ...);   // thrust iterators
 *     }                                                  // freed here
 *
 * Same rules as sctl::ScratchBuf: LIFO destruction (lexical scoping gives it), stack-only,
 * non-copyable, non-movable. Contents are uninitialized -- the element type must be trivial,
 * since no constructor can run on backend memory.
 */
template <class T, template <class...> class DeviceVector> class DeviceScratch {
  using Pool = DeviceScratchPool<DeviceVector>;
  static_assert(std::is_trivially_copyable<T>::value, "DeviceScratch<T>: T must be trivially copyable.");
  static_assert(alignof(T) <= (std::size_t)Pool::ALIGN, "DeviceScratch<T>: alignof(T) exceeds the pool alignment.");

 public:
  using iterator = detail::ScratchIterator<T, DeviceVector>;

  explicit DeviceScratch(Long count) : DeviceScratch(count, Pool::Instance()) {}

  DeviceScratch(Long count, Pool& pool) : pool_(&pool), count_(count) {
    SCTL_ASSERT(count >= 0);
    const auto slot = pool.AllocBytes(count * (Long)sizeof(T));
    chunk_ = slot.first;
    data_ = reinterpret_cast<T*>(slot.second);
  }

  ~DeviceScratch() { pool_->FreeBytes(chunk_, reinterpret_cast<char*>(data_), count_ * (Long)sizeof(T)); }

  DeviceScratch() = delete;
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;
  DeviceScratch(DeviceScratch&&) = delete;
  DeviceScratch& operator=(DeviceScratch&&) = delete;
  static void* operator new(std::size_t) = delete;
  static void* operator new[](std::size_t) = delete;
  static void operator delete(void*) = delete;
  static void operator delete[](void*) = delete;

  iterator begin() const { return iterator(data_); }
  iterator end() const { return iterator(data_ + count_); }
  iterator data() const { return iterator(data_); }  // thrust convention: use raw_pointer_cast for functors
  Long Dim() const { return count_; }

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
template <template <class...> class DeviceVector> class DeviceScratchAllocator {
 public:
  using value_type = char;

  char* allocate(std::ptrdiff_t n) {
    const auto slot = DeviceScratchPool<DeviceVector>::Instance().AllocBytes((Long)n);
    return slot.second;
  }
  void deallocate(char* p, std::size_t n) {
    DeviceScratchPool<DeviceVector>::Instance().FreeBytes(p, (Long)n);
  }
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_HPP_
