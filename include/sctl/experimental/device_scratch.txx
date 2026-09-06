// Template implementation of DeviceScratchPool / DeviceScratch / DeviceScratchAllocator
// from device_scratch.hpp.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_

#include <algorithm>

#include "sctl/experimental/device_scratch.hpp"

namespace gpu_tree {

namespace detail {

template <class T, template <class...> class DevVec, auto Tag>
inline DevVec<T>& PersistentBuffer() {
  static DevVec<T> buf;
  return buf;
}

template <class SrcPtr, class T> inline void deviceToHost(SrcPtr src, Long n, T* dst) {
  if (!n) return;
  const T* p = nullptr;  // host-side source of the fill: the staging buffer, or `src` itself
  if constexpr (is_device_ptr<SrcPtr>::value) {
    static T* stage = nullptr;
    static Long cap = 0;
    if (n > cap) {
      if (stage) cudaFreeHost(stage);
      cap = std::max<Long>(2 * cap, n);
      SCTL_ASSERT(cudaMallocHost((void**)&stage, cap * sizeof(T)) == cudaSuccess);
    }
    SCTL_ASSERT(cudaMemcpy(stage, thrust::raw_pointer_cast(src), n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    p = stage;
  } else {
    p = src;
  }
  #pragma omp parallel for schedule(static)
  for (Long i = 0; i < n; i++) dst[i] = p[i];
}

}  // namespace detail

template <template <class...> class DevVec>
inline DeviceScratchPool<DevVec>& DeviceScratchPool<DevVec>::Instance() {
  static DeviceScratchPool pool;
  return pool;
}

template <template <class...> class DevVec>
inline std::pair<typename DeviceScratchPool<DevVec>::Chunk*, char*> DeviceScratchPool<DevVec>::AllocBytes(Long bytes) {
  const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
  if (head_ == nullptr || need > head_->end - head_->top) NewChunk(need);
  char* const p = head_->top;
  head_->top += need;
  return {head_, p};
}

template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::FreeBytes(Chunk* chunk, char* p, Long bytes) {
  const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
  SCTL_ASSERT_MSG(chunk->top == p + need, "DeviceScratch: LIFO violation (free out of order).");
  chunk->top = p;
}

template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::FreeBytes(char* p, Long bytes) {
  const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
  for (Chunk* c = head_; c; c = c->prev) {
    if (c->top == p + need) { c->top = p; return; }
  }
  SCTL_ASSERT_MSG(false, "DeviceScratch: LIFO violation (free out of order).");
}

// Chunks double so the pool converges after a few builds; `need` wins when a single request is
// larger. The previous head stays in the list -- slices already handed out point into it.
template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::NewChunk(Long need) {
  const Long prev_cap = head_ ? head_->end - head_->base : 0;
  Long cap = std::max<Long>((Long)SCTL_DEVICE_SCRATCH_INIT_BYTES, prev_cap * 2);
  while (cap < need) cap *= 2;
  auto* buf = new DevVec<char>(cap);
  char* const base = thrust::raw_pointer_cast(buf->data());
  head_ = new Chunk{buf, base, base, base + cap, head_};
}

template <class T, template <class...> class DevVec>
inline DeviceScratch<T, DevVec>::DeviceScratch(Long count) : pool_(&Pool::Instance()), count_(count) {
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
