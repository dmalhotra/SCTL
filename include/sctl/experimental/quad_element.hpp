#ifndef _SCTL_QUAD_ELEMENT_HPP_
#define _SCTL_QUAD_ELEMENT_HPP_

#include <string>
#include <utility>
#include <vector>
#include <sctl.hpp>

namespace sctl {

  class VTUData;
  template <class ValueType> class Matrix;

  namespace detail_quadelem {
    template <class Real> struct Access;
  }

  template <class Real> class QuadElemList : public ElementListBase<Real> {

      friend struct detail_quadelem::Access<Real>;

    public:
      QuadElemList() {}

      template <class ValueType> QuadElemList(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      template <class ValueType> void Init(Integer order, const Vector<ValueType>& coord, const Comm& comm = Comm::Self());

      virtual ~QuadElemList() {}

      Long Size() const override;

      Integer Order() const;

      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx, const Vector<Real>* origin = nullptr) const;

      void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const override;

      void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const override;

      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      enum class QuadScheme { TensorProduct, Duffy, Hedgehog };

      void SetQuadScheme(QuadScheme s) {
        scheme_ = s;
      }

      static const Vector<Real>& ParamNodes(const Integer Order);

      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      template <class ValueType> void Read(const std::string& fname, const Comm& comm = Comm::Self());

      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F = Vector<Real>(), const Long elem_idx = -1) const;

      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      template <class ValueType> void Copy(QuadElemList<ValueType>& elem_lst) const;

      template<typename> friend class QuadElemList;

      template<typename> friend struct QuadElemTestAccess;

    private:

      Long nelem = 0;
      Integer order = 0;
      Vector<Real> coord;
      Vector<Real> dcoord_du, dcoord_dv;
      Vector<Real> X_node, Xn_node;
      Vector<Long> node_cnt;
      QuadScheme scheme_ = QuadScheme::TensorProduct;
  };

}

#endif
