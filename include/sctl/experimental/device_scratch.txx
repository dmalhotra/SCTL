// Template implementation of DeviceScratchPool / DeviceScratch / DeviceScratchAllocator
// from device_scratch.hpp.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_

#include <algorithm>

#include "sctl/experimental/device_scratch.hpp"

namespace gpu_tree {

namespace detail {

template <class T, template <class...> class DeviceVector, auto Tag>
inline DeviceVector<T>& PersistentBuffer() {
  static DeviceVector<T> buf;
  return buf;
}

}  // namespace detail

template <template <class...> class DeviceVector>
inline DeviceScratchPool<DeviceVector>& DeviceScratchPool<DeviceVector>::Instance() {
  static DeviceScratchPool pool;
  return pool;
}

template <template <class...> class DeviceVector>
inline std::pair<typename DeviceScratchPool<DeviceVector>::Chunk*, char*> DeviceScratchPool<DeviceVector>::AllocBytes(Long bytes) {
  const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
  if (head_ == nullptr || need > head_->end - head_->top) NewChunk(need);
  char* const p = head_->top;
  head_->top += need;
  return {head_, p};
}

template <template <class...> class DeviceVector>
inline void DeviceScratchPool<DeviceVector>::FreeBytes(Chunk* chunk, char* p, Long bytes) {
  const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
  SCTL_ASSERT_MSG(chunk->top == p + need, "DeviceScratch: LIFO violation (free out of order).");
  chunk->top = p;
}

template <template <class...> class DeviceVector>
inline void DeviceScratchPool<DeviceVector>::FreeBytes(char* p, Long bytes) {
  const Long need = (bytes + ALIGN - 1) & ~(ALIGN - 1);
  for (Chunk* c = head_; c; c = c->prev) {
    if (c->top == p + need) { c->top = p; return; }
  }
  SCTL_ASSERT_MSG(false, "DeviceScratch: LIFO violation (free out of order).");
}

// Chunks double so the pool converges after a few builds; `need` wins when a single request is
// larger. The previous head stays in the list -- slices already handed out point into it.
template <template <class...> class DeviceVector>
inline void DeviceScratchPool<DeviceVector>::NewChunk(Long need) {
  const Long prev_cap = head_ ? head_->end - head_->base : 0;
  Long cap = std::max<Long>((Long)SCTL_DEVICE_SCRATCH_INIT_BYTES, prev_cap * 2);
  while (cap < need) cap *= 2;
  auto* buf = new DeviceVector<char>(cap);
  char* const base = thrust::raw_pointer_cast(buf->data());
  head_ = new Chunk{buf, base, base, base + cap, head_};
}

template <class T, template <class...> class DeviceVector>
inline DeviceScratch<T, DeviceVector>::DeviceScratch(Long count) : DeviceScratch(count, Pool::Instance()) {}

template <class T, template <class...> class DeviceVector>
inline DeviceScratch<T, DeviceVector>::DeviceScratch(Long count, Pool& pool) : pool_(&pool), count_(count) {
  SCTL_ASSERT(count >= 0);
  const auto slot = pool.AllocBytes(count * (Long)sizeof(T));
  chunk_ = slot.first;
  data_ = reinterpret_cast<T*>(slot.second);
}

template <class T, template <class...> class DeviceVector>
inline DeviceScratch<T, DeviceVector>::~DeviceScratch() {
  pool_->FreeBytes(chunk_, reinterpret_cast<char*>(data_), count_ * (Long)sizeof(T));
}

template <class T, template <class...> class DeviceVector>
inline typename DeviceScratch<T, DeviceVector>::iterator DeviceScratch<T, DeviceVector>::begin() const { return iterator(data_); }

template <class T, template <class...> class DeviceVector>
inline typename DeviceScratch<T, DeviceVector>::iterator DeviceScratch<T, DeviceVector>::end() const { return iterator(data_ + count_); }

template <class T, template <class...> class DeviceVector>
inline typename DeviceScratch<T, DeviceVector>::iterator DeviceScratch<T, DeviceVector>::data() const { return iterator(data_); }

template <class T, template <class...> class DeviceVector>
inline Long DeviceScratch<T, DeviceVector>::Dim() const { return count_; }

template <template <class...> class DeviceVector>
inline char* DeviceScratchAllocator<DeviceVector>::allocate(std::ptrdiff_t n) {
  return DeviceScratchPool<DeviceVector>::Instance().AllocBytes((Long)n).second;
}

template <template <class...> class DeviceVector>
inline void DeviceScratchAllocator<DeviceVector>::deallocate(char* p, std::size_t n) {
  DeviceScratchPool<DeviceVector>::Instance().FreeBytes(p, (Long)n);
}

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_
