// Backend containers for gpu_tree: HostVector and DeviceVector, whose resize leaves trivial elements
// uninitialized, and DataView, the non-owning view GetData hands out.

#ifndef _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_
#define _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_

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
/// `thrust::device_allocator` whose construct is a no-op (thrust's uninitialized_vector idiom).
template <class T> struct DeviceUninitAllocator : thrust::device_allocator<T> {
  template <class U> struct rebind { using other = DeviceUninitAllocator<U>; };
  SCTL_GPU_HD void construct(T*) {}
};
}  // namespace detail

/// Host backend: `std::vector` that leaves new elements uninitialized, as `sctl::Vector` does. A class rather than an alias, so it deduces as the tree's container template parameter.
template <class T> class HostVector : public std::vector<T, detail::DefaultInitAllocator<T>> {
 public:
  using std::vector<T, detail::DefaultInitAllocator<T>>::vector;
};

/// Device backend: `thrust::device_vector` without value-initialization on resize.
template <class T> class DeviceVector : public thrust::device_vector<T, detail::DeviceUninitAllocator<T>> {
 public:
  using thrust::device_vector<T, detail::DeviceUninitAllocator<T>>::device_vector;
};

/// Non-owning view of a data set: `n` values at `ptr` in node order, with the iterator thrust dispatches on for the backend. Valid until the set is reallocated.
template <class T, template <class...> class DevVec> struct DataView {
  using value_type = T;
  using iterator = std::conditional_t<detail::is_device_vector_v<DevVec<char>>, thrust::device_ptr<T>, T*>;
  T* ptr = nullptr;
  sctl::Long n = 0;
  T* data() const { return ptr; }
  sctl::Long size() const { return n; }
  iterator begin() const { return iterator(ptr); }
  iterator end() const { return iterator(ptr) + n; }
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_
