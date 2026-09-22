// Implementation of the gpu_runtime helpers, DeviceScratchPool / DeviceScratch / DeviceScratchAllocator
// from device_scratch.hpp.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "sctl/experimental/device_scratch.hpp"
#include "sctl/iterator.txx"      // for Ptr2Itr
#include "sctl/mem_mgr.txx"       // for advise_huge_pages, MemoryManager
#include "sctl/ompUtils.hpp"      // for omp_par::copy, omp_par::prefault
#include "sctl/ompUtils.txx"
#include "sctl/scratch_pool.txx"  // for ScratchBuf

namespace gpu_tree {

namespace detail {

/**
 * The device-runtime calls made here, over whichever runtime is compiling this file. CUDA and HIP
 * spell them the same way under different prefixes, so only the names differ. A build with neither
 * cannot produce a device pointer, so `CopyToHost` is unreachable there and the rest have nothing
 * to do -- which is what a host backend wants.
 *
 * `__HIPCC__` is tested first: on the NVIDIA platform hipcc defines both, and there the hip names
 * are the cuda ones.
 */
namespace gpu_runtime {

/** Page-lock `bytes` at `p` so the driver can copy out of it directly. False if the runtime refused. */
inline bool HostRegister(void* p, std::size_t bytes) {
#if defined(__HIPCC__)
  return hipHostRegister(p, bytes, hipHostRegisterDefault) == hipSuccess;
#elif defined(__CUDACC__)
  return cudaHostRegister(p, bytes, cudaHostRegisterDefault) == cudaSuccess;
#else
  (void)p;
  (void)bytes;
  return true;  // nothing to pin
#endif
}

/** Undo `HostRegister`. Runs at exit, where the runtime may already be gone, so the result is of no use. */
inline void HostUnregister(void* p) {
#if defined(__HIPCC__)
  hipHostUnregister(p);
#elif defined(__CUDACC__)
  cudaHostUnregister(p);
#else
  (void)p;
#endif
}

/** Copy `bytes` of device memory at `src` to host memory at `dst`. False if the runtime refused. */
inline bool CopyToHost(void* dst, const void* src, std::size_t bytes) {
#if defined(__HIPCC__)
  return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost) == hipSuccess;
#elif defined(__CUDACC__)
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
#else
  (void)dst;
  (void)src;
  (void)bytes;
  return false;  // no runtime to copy with, and no device pointer to copy from
#endif
}

/** Fill `bytes` of device memory at `p` with `value`. False if the runtime refused. */
inline bool MemsetDevice(void* p, int value, std::size_t bytes) {
#if defined(__HIPCC__)
  return hipMemset(p, value, bytes) == hipSuccess;
#elif defined(__CUDACC__)
  return cudaMemset(p, value, bytes) == cudaSuccess;
#else
  (void)p;
  (void)value;
  (void)bytes;
  return false;  // no runtime, and no device pointer to fill
#endif
}

/** Wait for the device to finish what it has been given. */
inline void DeviceSynchronize() {
#if defined(__HIPCC__)
  hipDeviceSynchronize();
#elif defined(__CUDACC__)
  cudaDeviceSynchronize();
#endif
}

}  // namespace gpu_runtime

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
template <class T, template <class...> class DevVec, auto Tag>
inline DevVec<T>& PersistentBuffer() {
  if constexpr (is_device_vector_v<DevVec<T>>) {
    static DevVec<T>* buf = new DevVec<T>();  // never destroyed, for the reason in ~DeviceScratchPool
    return *buf;
  } else {  // host storage: that reason does not apply, and a leak checker would report it
    static DevVec<T> buf;
    return buf;
  }
}

/**
 * Size a buffer for output it is about to be given in full. `resize` alone preserves the contents,
 * which are dead in that case, and copies them when the buffer has to grow -- a device-to-device
 * copy of the whole buffer on the CUDA backend.
 */
template <class T, template <class...> class DevVec> inline void resizeDiscard(DevVec<T>& v, Long n) {
  v.clear();
  v.resize(n);
}

inline sctl::ScratchPool& pinnedStagingPool() {
  static sctl::ScratchPool pool(
      [](void* base, Long bytes) {  // fault the chunk in first: registering cold memory is far dearer
        sctl::omp_par::prefault(sctl::Ptr2Itr<char>((char*)base, bytes), bytes);
        const bool ok = gpu_runtime::HostRegister(base, (std::size_t)bytes);
        SCTL_ASSERT_MSG(ok, "pinnedStagingPool: the device runtime refused to page-lock a staging chunk.");
      },
      [](void* base, Long) { gpu_runtime::HostUnregister(base); });  // at exit the runtime may already be gone
  return pool;
}

/** `dst` takes a pointer or an sctl iterator: with SCTL_MEMDEBUG the containers hand back the latter. */
template <class SrcPtr, class DstPtr> inline void deviceToHost(SrcPtr src, Long n, DstPtr dst) {
  using T = typename std::remove_cv<typename std::remove_reference<decltype(*dst)>::type>::type;
  if (!n) return;
  if constexpr (is_device_ptr<SrcPtr>::value) {
    sctl::ScratchBuf<T> stage(n, pinnedStagingPool());
    const bool ok = gpu_runtime::CopyToHost(&stage[0], thrust::raw_pointer_cast(src), n * sizeof(T));
    SCTL_ASSERT_MSG(ok, "deviceToHost: the device runtime refused the copy.");
    sctl::omp_par::copy(stage.begin(), stage.end(), dst);
  } else {
    sctl::omp_par::copy(src, src + n, dst);
  }
}

}  // namespace detail

template <template <class...> class DevVec>
inline DeviceScratchPool<DevVec>& DeviceScratchPool<DevVec>::Instance() {
  static DeviceScratchPool pool;
  return pool;
}

// The chunk list is host memory and always freed. Its backing storage is freed only on the host
// backends: this runs during static destruction, where the CUDA runtime may already have unloaded
// and thrust's deallocate would then throw out of a destructor.
template <template <class...> class DevVec>
inline DeviceScratchPool<DevVec>::~DeviceScratchPool() {
#ifdef SCTL_MEMDEBUG
  SCTL_ASSERT_MSG(DebugLiveCount() == 0, "~DeviceScratchPool: the pool still holds live slices.");
#endif
  for (Chunk* c = head_; c != nullptr;) {
    Chunk* const prev = c->prev;
    if constexpr (!detail::is_device_vector_v<DevVec<char>>) delete c->buf;
    delete c;
    c = prev;
  }
  head_ = nullptr;
}

template <template <class...> class DevVec>
inline Long DeviceScratchPool<DevVec>::DebugChunkCount() const {
  Long n = 0;
  for (const Chunk* c = head_; c; c = c->prev) n++;
  return n;
}

template <template <class...> class DevVec>
inline Long DeviceScratchPool<DevVec>::DebugLiveCount() const {
#ifdef SCTL_MEMDEBUG
  Long n = 0;
  for (const Chunk* c = head_; c; c = c->prev) n += c->live_count;
  return n;
#else
  return (head_ == nullptr || head_->top == head_->base) ? 0 : -1;
#endif
}

// A redzone past each slice, stamped on allocation and checked on free. Debug builds only: on a
// device backend each stamp and each check is a round trip, which is what the pool exists to avoid.
template <template <class...> class DevVec>
constexpr Long DeviceScratchPool<DevVec>::Redzone() {
#ifdef SCTL_MEMDEBUG
  return sctl::MemoryManager::end_padding;
#else
  return 0;
#endif
}

template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::StampRedzone(char* p, Long bytes) {
  if constexpr (Redzone() > 0) {
    if constexpr (detail::is_device_vector_v<DevVec<char>>) {
      const bool ok = detail::gpu_runtime::MemsetDevice(p + bytes, sctl::MemoryManager::init_mem_val, (std::size_t)Redzone());
      SCTL_ASSERT_MSG(ok, "DeviceScratchPool: the device runtime refused to stamp a redzone.");
    } else {
      for (Long i = 0; i < Redzone(); i++) p[bytes + i] = sctl::MemoryManager::init_mem_val;
    }
  }
}

template <template <class...> class DevVec>
[[gnu::always_inline]] inline Long DeviceScratchPool<DevVec>::PaddedBytes(Long bytes) {
  return std::max<Long>(ALIGN, (bytes + Redzone() + ALIGN - 1) & ~(ALIGN - 1));
}

template <template <class...> class DevVec>
[[gnu::always_inline]] inline std::pair<typename DeviceScratchPool<DevVec>::Chunk*, char*> DeviceScratchPool<DevVec>::AllocBytes(Long bytes) {
  const Long need = PaddedBytes(bytes);
  if (head_ == nullptr || need > head_->end - head_->top) NewChunk(need);
  char* const p = head_->top;
  head_->top += need;
#ifdef SCTL_MEMDEBUG
  head_->live_count++;
  SCTL_ASSERT_MSG(((p - head_->base) & (ALIGN - 1)) == 0, "DeviceScratchPool: alignment invariant violated.");
  StampRedzone(p, bytes);
#endif
  return {head_, p};
}

// Rewind the slice, and hand an emptied chunk back unless it is the head, which is kept for the
// next build. Only `head_->prev` can empty first, since its slices were taken before the head's and
// LIFO frees them after, so the splice stays local. Releasing here is safe -- unlike at exit, the
// backend is still up.
template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::Rewind(Chunk* chunk, char* p) {
  chunk->top = p;
  if (chunk == head_ || chunk->top != chunk->base) return;
  SCTL_ASSERT(head_->prev == chunk);
  head_->prev = chunk->prev;
  delete chunk->buf;
  delete chunk;
}

template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::CheckRedzone(const char* p, Long bytes) {
  if constexpr (Redzone() > 0) {
    char trailer[Redzone()];
    const char* seen = p + bytes;
    if constexpr (detail::is_device_vector_v<DevVec<char>>) {
      const bool ok = detail::gpu_runtime::CopyToHost(trailer, p + bytes, (std::size_t)Redzone());
      SCTL_ASSERT_MSG(ok, "DeviceScratchPool: the device runtime refused to read a redzone back.");
      seen = trailer;
    }
    for (Long i = 0; i < Redzone(); i++) {
      SCTL_ASSERT_MSG(seen[i] == sctl::MemoryManager::init_mem_val,
                      "DeviceScratch: out-of-bounds write past buffer end detected.");
    }
  }
}

template <template <class...> class DevVec>
[[gnu::always_inline]] inline void DeviceScratchPool<DevVec>::FreeBytes(Chunk* chunk, char* p, Long bytes) {
  const Long need = PaddedBytes(bytes);
#ifdef SCTL_MEMDEBUG
  SCTL_ASSERT_MSG(chunk->top == p + need, "DeviceScratch: LIFO violation (free out of order).");
  CheckRedzone(p, bytes);
  SCTL_ASSERT(chunk->live_count > 0);
  chunk->live_count--;
#endif
  Rewind(chunk, p);
}

template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::FreeBytes(char* p, Long bytes) {
  const Long need = PaddedBytes(bytes);
  for (Chunk* c = head_; c; c = c->prev) {
    if (c->base <= p && p < c->end && c->top == p + need) {
#ifdef SCTL_MEMDEBUG
      CheckRedzone(p, bytes);
      SCTL_ASSERT(c->live_count > 0);
      c->live_count--;
#endif
      Rewind(c, p);
      return;
    }
  }
#ifdef SCTL_MEMDEBUG
  SCTL_ASSERT_MSG(false, "DeviceScratch: LIFO violation (free out of order).");
#endif
}

// Chunks double so the pool converges after a few builds; `need` wins when a single request is
// larger. The previous head stays in the list -- slices already handed out point into it.
template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::NewChunk(Long need) {
  const Long prev_cap = head_ ? head_->end - head_->base : 0;
  Long cap = std::max<Long>((Long)SCTL_DEVICE_SCRATCH_INIT_BYTES, prev_cap * 2);
  while (cap < need) cap *= 2;
  auto* buf = new DevVec<char>(cap + ALIGN - 1);  // room to align the base; the host backend gives only 16
  char* const raw = thrust::raw_pointer_cast(buf->data());
  if constexpr (!detail::is_device_vector_v<DevVec<char>>) {
    sctl::advise_huge_pages(raw, cap + ALIGN - 1);
    sctl::omp_par::prefault(sctl::Ptr2Itr<char>(raw, cap + ALIGN - 1), cap + ALIGN - 1);
  }
  char* const base = raw + ((ALIGN - (Long)((std::uintptr_t)raw & (ALIGN - 1))) & (ALIGN - 1));
  SCTL_ASSERT((std::uintptr_t)base % (std::uintptr_t)ALIGN == 0);
  if (head_ != nullptr && head_->top == head_->base) {  // outgrown and holding nothing: let it go
    Chunk* const prev = head_->prev;                    // else it is buried, and Rewind below
    delete head_->buf;                                  // could never reach it again
    delete head_;
    head_ = prev;
  }
  head_ = new Chunk{buf, base, base, base + cap, head_, 0};
}

template <class T, template <class...> class DevVec>
inline DeviceScratch<T, DevVec>::DeviceScratch(Long count) : pool_(&Pool::Instance()), count_(count) {
  SCTL_ASSERT(count >= 0);
  const auto slot = pool_->AllocBytes(count * (Long)sizeof(T));
  chunk_ = slot.first;
  data_ = reinterpret_cast<T*>(slot.second);
}

template <class T, template <class...> class DevVec>
inline DeviceScratch<T, DevVec>::DeviceScratch(Long count, Pool& pool) : pool_(&pool), count_(count) {
  SCTL_ASSERT(count >= 0);
  const auto slot = pool_->AllocBytes(count * (Long)sizeof(T));
  chunk_ = slot.first;
  data_ = reinterpret_cast<T*>(slot.second);
}

template <class T, template <class...> class DevVec>
inline Long DeviceScratch<T, DevVec>::Dim() const { return count_; }

template <class T, template <class...> class DevVec>
inline DeviceScratch<T, DevVec>::~DeviceScratch() {
  pool_->FreeBytes(chunk_, reinterpret_cast<char*>(data_), count_ * (Long)sizeof(T));
}

template <class T, template <class...> class DevVec>
inline typename DeviceScratch<T, DevVec>::iterator DeviceScratch<T, DevVec>::begin() const { return iterator(data_); }

template <class T, template <class...> class DevVec>
inline typename DeviceScratch<T, DevVec>::iterator DeviceScratch<T, DevVec>::end() const { return iterator(data_ + count_); }

template <class T, template <class...> class DevVec>
inline typename DeviceScratch<T, DevVec>::iterator DeviceScratch<T, DevVec>::data() const { return iterator(data_); }


template <template <class...> class DevVec>
inline char* DeviceScratchAllocator<DevVec>::allocate(std::ptrdiff_t n) {
  return DeviceScratchPool<DevVec>::Instance().AllocBytes((Long)n).second;
}

template <template <class...> class DevVec>
inline void DeviceScratchAllocator<DevVec>::deallocate(char* p, std::size_t n) {
  DeviceScratchPool<DevVec>::Instance().FreeBytes(p, (Long)n);
}

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_
