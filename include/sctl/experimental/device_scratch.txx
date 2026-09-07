// Template implementation of DeviceScratchPool / DeviceScratch / DeviceScratchAllocator
// from device_scratch.hpp.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_
#define _SCTL_EXPERIMENTAL_DEVICE_SCRATCH_TXX_

#include <algorithm>
#include <cstdint>

#include "sctl/experimental/device_scratch.hpp"
#include "sctl/iterator.txx"      // for Ptr2Itr
#include "sctl/ompUtils.hpp"      // for omp_par::copy, omp_par::prefault
#include "sctl/ompUtils.txx"
#include "sctl/scratch_pool.txx"  // for ScratchBuf

namespace gpu_tree {

namespace detail {

template <class T, template <class...> class DevVec, auto Tag>
inline DevVec<T>& PersistentBuffer() {
  static DevVec<T>* buf = new DevVec<T>();  // never destroyed, for the reason in ~DeviceScratchPool
  return *buf;
}

inline sctl::ScratchPool& pinnedStagingPool() {
  static sctl::ScratchPool pool(
      [](void* base, Long bytes) {  // fault the chunk in first: registering cold memory is far dearer
        sctl::omp_par::prefault(sctl::Ptr2Itr<char>((char*)base, bytes), bytes);
        SCTL_ASSERT(cudaHostRegister(base, (std::size_t)bytes, cudaHostRegisterDefault) == cudaSuccess);
      },
      [](void* base, Long) { cudaHostUnregister(base); });  // at exit the runtime may already be gone
  return pool;
}

template <class SrcPtr, class T> inline void deviceToHost(SrcPtr src, Long n, T* dst) {
  if (!n) return;
  if constexpr (is_device_ptr<SrcPtr>::value) {
    sctl::ScratchBuf<T> stage(n, pinnedStagingPool());
    SCTL_ASSERT(cudaMemcpy(&stage[0], thrust::raw_pointer_cast(src), n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
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
  for (Chunk* c = head_; c != nullptr;) {
    Chunk* const prev = c->prev;
    if constexpr (!detail::is_device_vector_v<DevVec<char>>) delete c->buf;
    delete c;
    c = prev;
  }
  head_ = nullptr;
}

template <template <class...> class DevVec>
inline Long DeviceScratchPool<DevVec>::PaddedBytes(Long bytes) {
  return std::max<Long>(ALIGN, (bytes + ALIGN - 1) & ~(ALIGN - 1));
}

template <template <class...> class DevVec>
inline std::pair<typename DeviceScratchPool<DevVec>::Chunk*, char*> DeviceScratchPool<DevVec>::AllocBytes(Long bytes) {
  const Long need = PaddedBytes(bytes);
  if (head_ == nullptr || need > head_->end - head_->top) NewChunk(need);
  char* const p = head_->top;
  head_->top += need;
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
inline void DeviceScratchPool<DevVec>::FreeBytes(Chunk* chunk, char* p, Long bytes) {
  const Long need = PaddedBytes(bytes);
  SCTL_ASSERT_MSG(chunk->top == p + need, "DeviceScratch: LIFO violation (free out of order).");
  Rewind(chunk, p);
}

template <template <class...> class DevVec>
inline void DeviceScratchPool<DevVec>::FreeBytes(char* p, Long bytes) {
  const Long need = PaddedBytes(bytes);
  for (Chunk* c = head_; c; c = c->prev) {
    if (c->base <= p && p < c->end && c->top == p + need) {
      Rewind(c, p);
      return;
    }
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
  auto* buf = new DevVec<char>(cap + ALIGN - 1);  // room to align the base; the host backend gives only 16
  char* const raw = thrust::raw_pointer_cast(buf->data());
  char* const base = raw + ((ALIGN - (Long)((std::uintptr_t)raw & (ALIGN - 1))) & (ALIGN - 1));
  if (head_ != nullptr && head_->top == head_->base) {  // outgrown and holding nothing: let it go
    Chunk* const prev = head_->prev;                    // else it is buried, and Rewind below
    delete head_->buf;                                  // could never reach it again
    delete head_;
    head_ = prev;
  }
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
