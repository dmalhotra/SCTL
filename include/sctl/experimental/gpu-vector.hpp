// Backend containers for gpu_tree: HostVector and DeviceVector, whose resize leaves trivial elements
// uninitialized, and DataView, a non-owning view of backend memory.

#ifndef _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_
#define _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "sctl/common.hpp"
#include "sctl/morton.hpp"  // SCTL_GPU_HD
#include "sctl/experimental/device_scratch.hpp"  // is_device_vector_v

namespace gpu_tree {

namespace detail {
/// `std::allocator` whose default construction is a no-op, so `std::vector::resize` leaves trivial elements uninitialized.
template <class T> struct DefaultInitAllocator : std::allocator<T> {
  template <class U> struct rebind { using other = DefaultInitAllocator<U>; };
  template <class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) { ::new (static_cast<void*>(p)) U; }
  template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
/// `thrust::device_allocator` whose construct is a no-op (thrust's uninitialized_vector idiom), allocating stream-ordered from the device's memory pool (`gpu_runtime::DeviceMalloc`).
template <class T> struct DeviceUninitAllocator : thrust::device_allocator<T> {
  using pointer = thrust::device_ptr<T>;
  using size_type = std::size_t;
  template <class U> struct rebind { using other = DeviceUninitAllocator<U>; };
  SCTL_GPU_HD void construct(T*) {}
  pointer allocate(size_type n) { return pointer(static_cast<T*>(gpu_runtime::DeviceMalloc(n * sizeof(T)))); }
  void deallocate(pointer p, size_type) { gpu_runtime::DeviceFree(thrust::raw_pointer_cast(p)); }
};
}  // namespace detail

/**
 * Host backend container: `std::vector` whose `resize` leaves new elements uninitialized.
 *
 * @tparam T Element type.
 */
template <class T> class HostVector : public std::vector<T, detail::DefaultInitAllocator<T>> {
 public:
  using std::vector<T, detail::DefaultInitAllocator<T>>::vector;
};

/**
 * Device backend container: `thrust::device_vector` whose `resize` leaves new elements
 * uninitialized, allocated stream-ordered from the device's memory pool.
 *
 * @tparam T Element type; trivially copyable, since no constructor runs on device memory.
 */
template <class T> class DeviceVector : public thrust::device_vector<T, detail::DeviceUninitAllocator<T>> {
 public:
  using thrust::device_vector<T, detail::DeviceUninitAllocator<T>>::device_vector;
};

/**
 * Non-owning view of `n` values at `ptr` in backend memory. Valid until the storage it points into
 * is reallocated or freed.
 *
 * @tparam T Element type, `const` for a read-only view.
 * @tparam DevVec The backend's container template; selects the iterator type.
 */
template <class T, template <class...> class DevVec> struct DataView {
  using value_type = T;
  using iterator = std::conditional_t<detail::is_device_vector_v<DevVec<char>>, thrust::device_ptr<T>, T*>;  // probed on DevVec<char>: T may be const, which no container holds
  T* ptr = nullptr;
  sctl::Long n = 0;
  DataView() = default;
  DataView(T* p, sctl::Long count) : ptr(p), n(count) {}
  /// The whole of `v`: any container or view whose `data()` gives a pointer convertible to `T*`. Valid while `v` is not reallocated.
  template <class V, class = std::enable_if_t<std::is_convertible<decltype(thrust::raw_pointer_cast(std::declval<V&>().data())), T*>::value>>
  DataView(V& v) : ptr(thrust::raw_pointer_cast(v.data())), n((sctl::Long)v.size()) {}
  T* data() const { return ptr; }
  sctl::Long size() const { return n; }
  iterator begin() const { return iterator(ptr); }
  iterator end() const { return iterator(ptr) + n; }
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_
