#ifndef _SCTL_KERNEL_FUNCTIONS_HPP_
#define _SCTL_KERNEL_FUNCTIONS_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(generic-kernel.hpp)
#include SCTL_INCLUDE(vec.hpp)

namespace SCTL_NAMESPACE {

  namespace kernel_impl {

    struct Laplace3D_FxU {
      static const std::string& Name() {
        static const std::string name = "Laplace3D-FxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 6;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        u[0][0] = rinv;
      }
    };

    struct Laplace3D_DxU {
      static const std::string& Name() {
        static const std::string name = "Laplace3D-DxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 14;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
        VecType rinv3 = rinv * rinv * rinv;
        u[0][0] = rdotn * rinv3;
      }
    };

    struct Laplace3D_FxdU {
      static const std::string& Name() {
        static const std::string name = "Laplace3D-FxdU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 11;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return -1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][3], const VecType (&r)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        u[0][0] = r[0] * rinv3;
        u[0][1] = r[1] * rinv3;
        u[0][2] = r[2] * rinv3;
      }
    };

    struct Stokes3D_FxU {
      static const std::string& Name() {
        static const std::string name = "Stokes3D-FxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 23;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (8 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv*rinv*rinv;
        for (Integer i = 0; i < 3; i++) {
          for (Integer j = 0; j < 3; j++) {
            u[i][j] = (i==j ? rinv : VecType::Zero()) + r[i]*r[j]*rinv3;
          }
        }
      }
    };

    struct Stokes3D_DxU {
      static const std::string& Name() {
        static const std::string name = "Stokes3D-DxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 26;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 3 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv*rinv;
        VecType rinv5 = rinv2*rinv2*rinv;
        VecType rdotn_rinv5 = (r[0]*n[0] + r[1]*n[1] + r[2]*n[2])*rinv5;
        for (Integer i = 0; i < 3; i++) {
          for (Integer j = 0; j < 3; j++) {
            u[i][j] = r[i]*r[j]*rdotn_rinv5;
          }
        }
      }
    };

    struct Stokes3D_FxT {
      static const std::string& Name() {
        static const std::string name = "Stokes3D-FxT";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 39;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return -3 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][9], const VecType (&r)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv*rinv;
        VecType rinv5 = rinv2*rinv2*rinv;
        for (Integer i = 0; i < 3; i++) {
          for (Integer j = 0; j < 3; j++) {
            for (Integer k = 0; k < 3; k++) {
              u[i][j*3+k] = r[i]*r[j]*r[k]*rinv5;
            }
          }
        }
      }
    };

    struct Stokes3D_FSxU {
      static const std::string& Name() {
        static const std::string name = "Stokes3D-FSxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 26;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (8 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[4][3], const VecType (&r)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv*rinv*rinv;
        for (Integer i = 0; i < 3; i++) {
          for (Integer j = 0; j < 3; j++) {
            u[i][j] = (i==j ? rinv : VecType::Zero()) + r[i]*r[j]*rinv3;
          }
        }
        for (Integer j = 0; j < 3; j++) {
          u[3][j] = r[j]*rinv3;
        }
      }
    };

    struct Stokes3D_FxUP {
      static const std::string& Name() {
        static const std::string name = "Stokes3D-FxUP";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 26;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (8 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][4], const VecType (&r)[3], const void* ctx_ptr) {
        VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv*rinv*rinv;
        for (Integer i = 0; i < 3; i++) {
          for (Integer j = 0; j < 3; j++) {
            u[i][j] = (i==j ? rinv : VecType::Zero()) + r[i]*r[j]*rinv3;
          }
        }
        for (Integer i = 0; i < 3; i++) {
          u[i][3] = r[i]*rinv3;
        }
      }
    };

  }  // namespace kernel_impl

  // Notation:
  // F = single-layer source
  // D = double-layer source
  // U = potential
  // dU = grad potential
  struct Laplace3D_FxU : public GenericKernel<kernel_impl::Laplace3D_FxU> {};
  struct Laplace3D_DxU : public GenericKernel<kernel_impl::Laplace3D_DxU> {};
  struct Laplace3D_FxdU : public GenericKernel<kernel_impl::Laplace3D_FxdU>{};
  struct Stokes3D_FxU : public GenericKernel<kernel_impl::Stokes3D_FxU> {};
  struct Stokes3D_DxU : public GenericKernel<kernel_impl::Stokes3D_DxU> {};
  struct Stokes3D_FxT : public GenericKernel<kernel_impl::Stokes3D_FxT> {}; // single-layer source ---> traction-tensor
  struct Stokes3D_FSxU : public GenericKernel<kernel_impl::Stokes3D_FSxU> {}; // single-layer + source/sink ---> velocity (required for FMM translations involving double-layer - M2M, M2L, M2T)
  struct Stokes3D_FxUP : public GenericKernel<kernel_impl::Stokes3D_FxUP> {}; // single-layer source ---> velocity + pressure

}  // end namespace

#endif  //_SCTL_KERNEL_FUNCTIONS_HPP_
