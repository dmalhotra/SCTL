// The one translation unit that nvcc compiles: holds gpu_tree and thrust, and defines the
// thrust-free interface declared in sctl/experimental/device-tree.hpp. The explicit
// instantiations at the bottom are what the archive ships.

#include "sctl/tree.hpp"
#include "sctl/tree.txx"
#include "sctl/experimental/gpu-tree.hpp"
#include "sctl/experimental/device-tree.hpp"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace sctl {
namespace device_tree {

namespace detail_deviceTree {

  /** The caller's buffer as a device-side range, copying in from the host when that is where it is. */
  template <class T> gpu_tree::DeviceVector<T> toDevice(const T* src, Long n, MemSpace space)
  {
    if (space == MemSpace::Host)
    {
      return gpu_tree::DeviceVector<T>(src, src + n);
    }
    const thrust::device_ptr<const T> p(src);
    return gpu_tree::DeviceVector<T>(p, p + n);
  }

  /** The reverse: out of the tree's own buffer into wherever the caller's buffer lives. */
  template <class T> void fromDevice(const gpu_tree::DeviceVector<T>& src, T* dst, MemSpace space)
  {
    if (space == MemSpace::Host)
    {
      thrust::copy(src.begin(), src.end(), dst);
      return;
    }
    thrust::copy(src.begin(), src.end(), thrust::device_ptr<T>(dst));
  }

  template <class T> T* raw(gpu_tree::DeviceVector<T>& v)
  {
    return thrust::raw_pointer_cast(v.data());
  }

  template <class T> const T* raw(const gpu_tree::DeviceVector<T>& v)
  {
    return thrust::raw_pointer_cast(v.data());
  }

}  // namespace detail_deviceTree

template <class Real, Integer DIM> struct PtTree<Real, DIM>::Impl {
  using Tree = gpu_tree::PtTree<Real, DIM, gpu_tree::DeviceVector>;

  explicit Impl(const Comm& comm) : tree(comm) {}

  Tree tree;
};

template <class Real, Integer DIM> PtTree<Real, DIM>::PtTree(const Comm& comm) : p_(new Impl(comm)) {}

template <class Real, Integer DIM> PtTree<Real, DIM>::~PtTree()
{
  delete p_;
}

template <class Real, Integer DIM> PtTree<Real, DIM>::PtTree(PtTree&& other) noexcept : p_(other.p_)
{
  other.p_ = nullptr;
}

template <class Real, Integer DIM> PtTree<Real, DIM>& PtTree<Real, DIM>::operator=(PtTree&& other) noexcept
{
  if (this != &other)
  {
    delete p_;
    p_ = other.p_;
    other.p_ = nullptr;
  }
  return *this;
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::UpdateRefinement(const Real* coord, Long n, MemSpace space, Long M, bool balance21, Periodicity periodicity, Integer halo_size)
{
  const auto x = detail_deviceTree::toDevice(coord, n * DIM, space);
  p_->tree.UpdateRefinement(x, M, balance21, periodicity, halo_size);
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::AddParticles(const std::string& name, const Real* coord, Long n, MemSpace space)
{
  const auto x = detail_deviceTree::toDevice(coord, n * DIM, space);
  p_->tree.AddParticles(name, x);
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::AddParticleData(const std::string& data_name, const std::string& particle_name, const Real* data, Long n, MemSpace space)
{
  const auto v = detail_deviceTree::toDevice(data, n, space);
  p_->tree.AddParticleData(data_name, particle_name, v);
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::AddParticleData(const std::string& data_name, const std::string& particle_name, Long dof)
{
  p_->tree.AddParticleData(data_name, particle_name, dof);
}

template <class Real, Integer DIM> Long PtTree<Real, DIM>::ParticleDataDim(const std::string& data_name) const
{
  gpu_tree::DeviceVector<Real> v;
  p_->tree.GetParticleData(v, data_name);
  return (Long)v.size();
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::GetParticleData(const std::string& data_name, Real* out, Long n, MemSpace space) const
{
  gpu_tree::DeviceVector<Real> v;
  p_->tree.GetParticleData(v, data_name);
  SCTL_ASSERT_MSG((Long)v.size() == n, "device_tree::PtTree::GetParticleData: n does not match ParticleDataDim.");
  detail_deviceTree::fromDevice(v, out, space);
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::DeleteParticleData(const std::string& data_name)
{
  p_->tree.DeleteParticleData(data_name);
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::AddData(const std::string& name, Payload payload, Long dof, const Vector<Long>& cnt)
{
  switch (payload)
  {
    case Payload::Real:
      p_->tree.template AddData<Real>(name, dof, cnt);
      return;
    case Payload::Char:
      p_->tree.template AddData<char>(name, dof, cnt);
      return;
    case Payload::Long:
      p_->tree.template AddData<long>(name, dof, cnt);
      return;
  }
  SCTL_ASSERT_MSG(false, "device_tree::PtTree::AddData: unknown payload type.");
}

template <class Real, Integer DIM> void* PtTree<Real, DIM>::dataPtr(const std::string& name, Payload payload, Long& n, Vector<Long>& cnt)
{
  switch (payload)
  {
    case Payload::Real:
    {
      typename Impl::Tree::template View<Real> v;
      p_->tree.GetData(v, cnt, name);
      n = v.size();
      return v.ptr;
    }
    case Payload::Char:
    {
      typename Impl::Tree::template View<char> v;
      p_->tree.GetData(v, cnt, name);
      n = v.size();
      return v.ptr;
    }
    case Payload::Long:
    {
      typename Impl::Tree::template View<long> v;
      p_->tree.GetData(v, cnt, name);
      n = v.size();
      return v.ptr;
    }
  }
  SCTL_ASSERT_MSG(false, "device_tree::PtTree::Data: unknown payload type.");
  n = 0;
  return nullptr;
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::Broadcast(const std::string& name)
{
  p_->tree.Broadcast(name);
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::ReduceBroadcast(const std::string& name)
{
  p_->tree.template ReduceBroadcast<Real>(name);
}

template <class Real, Integer DIM> Span<const Morton<DIM>> PtTree<Real, DIM>::NodeMID() const
{
  const auto& v = p_->tree.GetNodeMID();
  return Span<const Morton<DIM>>{detail_deviceTree::raw(v), (Long)v.size()};
}

template <class Real, Integer DIM> Span<const NodeAttr> PtTree<Real, DIM>::NodeAttrs() const
{
  using TreeAttr = typename Impl::Tree::NodeAttr;
  static_assert(sizeof(TreeAttr) == sizeof(NodeAttr), "device_tree::NodeAttr must match the tree's node attributes.");
  const auto& v = p_->tree.GetNodeAttr();
  return Span<const NodeAttr>{reinterpret_cast<const NodeAttr*>(detail_deviceTree::raw(v)), (Long)v.size()};
}

template <class Real, Integer DIM> NodeLists PtTree<Real, DIM>::Lists() const
{
  const auto& l = p_->tree.GetNodeLists();
  NodeLists out;
  out.parent = Span<const Long>{detail_deviceTree::raw(l.parent), (Long)l.parent.size()};
  out.child = Span<const Long>{detail_deviceTree::raw(l.child), (Long)l.child.size()};
  out.nbr = Span<const Long>{detail_deviceTree::raw(l.nbr), (Long)l.nbr.size()};
  return out;
}

template <class Real, Integer DIM> void PtTree<Real, DIM>::OwnedRange(Long& begin, Long& end) const
{
  p_->tree.GetOwnedRange(begin, end);
}

template class PtTree<float, 2>;
template class PtTree<float, 3>;
template class PtTree<double, 2>;
template class PtTree<double, 3>;

}  // namespace device_tree
}  // namespace sctl
