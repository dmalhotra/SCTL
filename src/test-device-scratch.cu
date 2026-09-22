// Unit tests for gpu_tree::DeviceScratchPool / DeviceScratch, the device-side twin of
// sctl::ScratchPool / sctl::ScratchBuf that src/test-scratch-pool.cpp covers.
//
// Everything here drives a caller-supplied pool rather than Instance(), so the tests neither see
// nor disturb the state the tree build leaves behind. The pool is exercised on both backends: the
// host one, where the allocations are ordinary memory this file can read and write, and the device one,
// where only the pool's bookkeeping is checked from here.

#include <cstdio>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include "sctl/experimental/device_scratch.hpp"
#include "sctl/experimental/device_scratch.txx"

using sctl::Long;

template <class... A> using HostVec = std::vector<A...>;
template <class... A> using DeviceVec = thrust::device_vector<A...>;

static Long checks = 0, failures = 0;

static void check(const char* what, bool bad) {
  checks++;
  if (bad) {
    failures++;
    std::printf("  FAIL %s\n", what);
  }
}

/** The bookkeeping, which is the same whichever backend holds the bytes. */
template <template <class...> class DevVec> static void test_pool(const char* backend) {
  std::printf("DeviceScratchPool<%s> :\n", backend);
  using Pool = gpu_tree::DeviceScratchPool<DevVec>;
  {  // a fresh pool holds nothing
    Pool pool;
    check("fresh pool holds no chunk", pool.DebugChunkCount() != 0);
    check("fresh pool holds no allocation", pool.DebugLiveCount() != 0);
  }
  {  // one allocation, then drained
    Pool pool;
    {
      gpu_tree::DeviceScratch<double, DevVec> a(1024, pool);
      check("Dim reports the request", a.size() != 1024);
      check("a allocation takes a chunk", pool.DebugChunkCount() < 1);
    }
    check("the allocation is given back", pool.DebugLiveCount() != 0);
  }
  {  // nested allocations come back in reverse order
    Pool pool;
    gpu_tree::DeviceScratch<char, DevVec> a(64, pool);
    {
      gpu_tree::DeviceScratch<char, DevVec> b(64, pool);
      gpu_tree::DeviceScratch<char, DevVec> c(64, pool);
      check("three allocations are distinct", thrust::raw_pointer_cast(b.data()) == thrust::raw_pointer_cast(c.data()));
    }
    check("the outer allocation is still live", pool.DebugLiveCount() == 0);
  }
  {  // a request larger than the current chunk takes another, and the pool still drains
    Pool pool;
    {
      gpu_tree::DeviceScratch<char, DevVec> a(1024, pool);
      gpu_tree::DeviceScratch<char, DevVec> big(64 << 20, pool);
      check("an oversize request is served", big.size() != (64 << 20));
      check("it took a second chunk", pool.DebugChunkCount() < 2);
    }
    check("both allocations are given back", pool.DebugLiveCount() != 0);
    check("the head chunk is retained", pool.DebugChunkCount() != 1);
  }
  {  // a zero-length allocation is legal and consumes a slot, so an empty chunk stays distinguishable
    Pool pool;
    gpu_tree::DeviceScratch<double, DevVec> z(0, pool);
    check("a zero-length allocation has size 0", z.size() != 0);
  }
  {  // allocations are aligned for any type the pool hands out
    Pool pool;
    gpu_tree::DeviceScratch<char, DevVec> a(1, pool);
    gpu_tree::DeviceScratch<double, DevVec> b(1, pool);
    const auto addr = (std::uintptr_t)thrust::raw_pointer_cast(b.data());
    check("a allocation is aligned to SCTL_MEM_ALIGN", addr % (std::uintptr_t)Pool::ALIGN != 0);
  }
}

/** Reading and writing the bytes, which only the host backend can do from here. */
static void test_host_storage() {
  std::printf("DeviceScratch<host> storage :\n");
  gpu_tree::DeviceScratchPool<HostVec> pool;
  {
    gpu_tree::DeviceScratch<double, HostVec> a(1000, pool);
    for (Long i = 0; i < a.size(); i++) a.data()[i] = (double)i;
    bool bad = false;
    for (Long i = 0; i < a.size(); i++) bad = bad || (a.data()[i] != (double)i);
    check("a allocation round-trips its values", bad);
    check("begin/end span the allocation", a.end() - a.begin() != a.size());
  }
  {  // a second allocation must not overlap the first
    gpu_tree::DeviceScratch<char, HostVec> a(128, pool);
    gpu_tree::DeviceScratch<char, HostVec> b(128, pool);
    for (Long i = 0; i < 128; i++) a.data()[i] = 'a';
    for (Long i = 0; i < 128; i++) b.data()[i] = 'b';
    bool bad = false;
    for (Long i = 0; i < 128; i++) bad = bad || (a.data()[i] != 'a');
    check("a live neighbour is not overwritten", bad);
  }
  {  // storage is reused once released, which is the point of the pool
    const void* first = nullptr;
    {
      gpu_tree::DeviceScratch<char, HostVec> a(256, pool);
      first = thrust::raw_pointer_cast(a.data());
    }
    gpu_tree::DeviceScratch<char, HostVec> b(256, pool);
    check("a released allocation is handed out again", thrust::raw_pointer_cast(b.data()) != first);
  }
}

int main() {
  test_pool<HostVec>("host");
  test_pool<DeviceVec>("device");
  test_host_storage();
  std::printf("%s (%ld check(s), %ld failed)\n", failures ? "FAILED" : "PASS", (long)checks, (long)failures);
  return failures ? 1 : 0;
}
