// A thrust-free interface to the device-backed gpu_tree, for callers that are not compiled by
// nvcc. The templates are instantiated once in src/device-tree.cu for Real in {float, double} and
// DIM in {2, 3}; a caller includes this header, links the archive, and needs only libcudart.
//
// Buffers cross the interface as pointer and length, tagged with the memory space they live in.
// Everything the tree hands back is a device pointer.
//
// The caller and the archive must agree on SCTL_HAVE_MPI, which changes the layout of Comm, and
// on the size of Long.

#ifndef _SCTL_EXPERIMENTAL_DEVICE_TREE_HPP_
#define _SCTL_EXPERIMENTAL_DEVICE_TREE_HPP_

#include <string>
#include <type_traits>

#include "sctl/common.hpp"  // for Long, Integer, Periodicity
#include "sctl/comm.hpp"    // for Comm
#include "sctl/comm.txx"
#include "sctl/morton.hpp"  // for Morton
#include "sctl/morton.txx"
#include "sctl/vector.hpp"  // for Vector
#include "sctl/vector.txx"

namespace sctl {
namespace device_tree {

/** Which memory a caller's buffer lives in. */
enum class MemSpace {
  Host,
  Device
};

/** Non-owning run of `n` values in device memory. */
template <class T> struct Span {
  T* ptr = nullptr;
  Long n = 0;

  T* data() const
  {
    return ptr;
  }

  Long Dim() const
  {
    return n;
  }
};

/** Layout-compatible stand-in for the tree's node attributes, which is a nested type of a template this header cannot name. */
struct NodeAttr {
  unsigned char Leaf : 1,
                Ghost : 1;
};

/** Node index lists, one array per kind; `-1` means absent. Indices are into the node list. */
struct NodeLists {
  Span<const Long> parent;
  Span<const Long> child;
  Span<const Long> nbr;
};

/** Payload types the precompiled library instantiates node data for. */
enum class Payload {
  Real,
  Char,
  Long
};

/** Maps a payload type to its tag; a type with no tag does not compile. */
template <class Real, class T> struct PayloadOf;
template <class Real> struct PayloadOf<Real, Real> { static constexpr Payload value = Payload::Real; };
template <class Real> struct PayloadOf<Real, char> { static constexpr Payload value = Payload::Char; };
template <class Real> struct PayloadOf<Real, long> { static constexpr Payload value = Payload::Long; };

/**
 * Point tree whose nodes, topology and payloads live on the device.
 *
 * Mirrors `gpu_tree::PtTree<Real, DIM, DeviceVector>`; see that class for the semantics of each
 * operation. Views returned here stay valid until the data set is reallocated, which
 * `UpdateRefinement`, `Broadcast`, `ReduceBroadcast` and the delete operations all do.
 */
template <class Real, Integer DIM> class PtTree {
 public:
  explicit PtTree(const Comm& comm = Comm::Self());
  ~PtTree();

  PtTree(PtTree&& other) noexcept;
  PtTree& operator=(PtTree&& other) noexcept;
  PtTree(const PtTree&) = delete;
  PtTree& operator=(const PtTree&) = delete;

  /** `coord` holds `n` points, `DIM` values each. */
  void UpdateRefinement(const Real* coord, Long n, MemSpace space, Long M = 1, bool balance21 = false, Periodicity periodicity = Periodicity::NONE, Integer halo_size = -1);

  void AddParticles(const std::string& name, const Real* coord, Long n, MemSpace space);

  /** `n` is the element count, `dof * Nlocal[particle_name]`. */
  void AddParticleData(const std::string& data_name, const std::string& particle_name, const Real* data, Long n, MemSpace space);

  /** `dof` unwritten values per particle, to be filled through the view `Data` returns. */
  void AddParticleData(const std::string& data_name, const std::string& particle_name, Long dof);

  /** Elements `GetParticleData` will write, so a caller can size its buffer. */
  Long ParticleDataDim(const std::string& data_name) const;

  /** Writes `n` values in the caller's particle order; `n` must be `ParticleDataDim(data_name)`. */
  void GetParticleData(const std::string& data_name, Real* out, Long n, MemSpace space) const;

  void DeleteParticleData(const std::string& data_name);

  /** `cnt[i] * dof` unwritten elements for node i. */
  void AddData(const std::string& name, Payload payload, Long dof, const Vector<Long>& cnt);

  /** View of a data set's storage, with the per-node element counts. */
  template <class T> Span<T> Data(const std::string& name, Vector<Long>& cnt)
  {
    static_assert(!std::is_const<T>::value, "device_tree::PtTree::Data: T must not be const.");
    Long n = 0;
    void* p = dataPtr(name, PayloadOf<Real, T>::value, n, cnt);
    return Span<T>{static_cast<T*>(p), n};
  }

  void Broadcast(const std::string& name);

  /** Sums onto the owner what the ranks sharing a node hold, then broadcasts the result. Real payloads only. */
  void ReduceBroadcast(const std::string& name);

  Span<const Morton<DIM>> NodeMID() const;
  Span<const NodeAttr> NodeAttrs() const;
  NodeLists Lists() const;
  void OwnedRange(Long& begin, Long& end) const;

 private:
  void* dataPtr(const std::string& name, Payload payload, Long& n, Vector<Long>& cnt);

  struct Impl;
  Impl* p_ = nullptr;
};

}  // namespace device_tree
}  // namespace sctl

#endif  // _SCTL_EXPERIMENTAL_DEVICE_TREE_HPP_
