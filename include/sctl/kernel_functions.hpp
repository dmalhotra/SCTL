#ifndef _SCTL_KERNEL_FUNCTIONS_HPP_
#define _SCTL_KERNEL_FUNCTIONS_HPP_

#include <string>                   // for basic_string, string

#include "sctl/common.hpp"          // for Integer, sctl
#include "sctl/generic-kernel.hpp"  // for GenericKernel
#include "sctl/math_utils.hpp"      // for const_pi
#include "sctl/vec.txx"             // for operator*, operator+, Vec::Zero

namespace sctl {

  namespace kernel_impl {

    /**
     * A micro-kernel provides uKerMatrix, which fills the KDIM0 x KDIM1 kernel
     * matrix for one source-target displacement. It may additionally provide an
     * optional fused apply,
     *
     *   static constexpr bool FUSED_APPLY = true;
     *   template <Integer digits, Integer DOF, class VecType>
     *   static void uKerApply(VecType* v, const VecType (&r)[DIM], const VecType* n, const VecType* f, const void* ctx_ptr);
     *
     * which accumulates the same unscaled values that applying uKerMatrix to the
     * densities f would produce, but without materializing the matrix. This pays
     * off when the matrix has structural zeros or repeated entries (antisymmetric
     * or complex-valued kernels), where a generic apply spends multiply-adds on
     * entries it cannot see are redundant. Callers that do not use the hook get
     * identical results from uKerMatrix.
     */

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

      // Volume potential for uniform unit charge; used in periodic-FMM BC correction.
      template <class Real> static void VolPoten(Matrix<Real>& U, const Vector<Real>& X) {
        const Long N = X.Dim() / 3;
        SCTL_ASSERT(X.Dim() == N * 3);
        if (U.Dim(0) != 1 || U.Dim(1) != N) U.ReInit(1, N);
        for (Long i = 0; i < N; i++) {
          const Real x = X[i*3+0], y = X[i*3+1], z = X[i*3+2];
          U[0][i] = -(x*x + y*y + z*z) / 6;
        }
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

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv3 = rinv * rinv * rinv;
        for (Integer k = 0; k < DOF; k++) {
          const VecType f_rinv3 = f[k] * rinv3;
          for (Integer i = 0; i < 3; i++) v[k*3+i] += r[i] * f_rinv3;
        }
      }

      // Volume potential for uniform unit charge; used in periodic-FMM BC correction.
      template <class Real> static void VolPoten(Matrix<Real>& U, const Vector<Real>& X) {
        const Long N = X.Dim() / 3;
        SCTL_ASSERT(X.Dim() == N * 3);
        if (U.Dim(0) != 1 || U.Dim(1) != N * 3) U.ReInit(1, N * 3);
        for (Long i = 0; i < N; i++) {
          U[0][i*3+0] = -X[i*3+0] / 3;
          U[0][i*3+1] = -X[i*3+1] / 3;
          U[0][i*3+2] = -X[i*3+2] / 3;
        }
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

      // Volume potential for uniform unit force; used in periodic-FMM BC correction.
      template <class Real> static void VolPoten(Matrix<Real>& U, const Vector<Real>& X) {
        const Long N = X.Dim() / 3;
        SCTL_ASSERT(X.Dim() == N * 3);
        if (U.Dim(0) != 3 || U.Dim(1) != N * 3) U.ReInit(3, N * 3);
        for (Long i = 0; i < N; i++) {
          const Real x = X[i*3 + 0];
          const Real y = X[i*3 + 1];
          const Real z = X[i*3 + 2];
          const Real rx_2 = y*y + z*z;
          const Real ry_2 = x*x + z*z;
          const Real rz_2 = x*x + y*y;
          U[0][i*3+0] = -rx_2/4; U[0][i*3+1] =       0; U[0][i*3+2] =       0;
          U[1][i*3+0] =       0; U[1][i*3+1] = -ry_2/4; U[1][i*3+2] =       0;
          U[2][i*3+0] =       0; U[2][i*3+1] =       0; U[2][i*3+2] = -rz_2/4;
        }
      }

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv3 = rinv*rinv*rinv;
        for (Integer k = 0; k < DOF; k++) { // identity + rank-one
          const VecType* f_ = f + k*3;
          VecType* v_ = v + k*3;
          const VecType fdotr_rinv3 = (f_[0]*r[0] + f_[1]*r[1] + f_[2]*r[2]) * rinv3;
          for (Integer i = 0; i < 3; i++) v_[i] += f_[i]*rinv + r[i]*fdotr_rinv3;
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

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        const VecType rinv5 = rinv2 * rinv2 * rinv;
        const VecType rdotn_rinv5 = (r[0]*n[0] + r[1]*n[1] + r[2]*n[2]) * rinv5;
        for (Integer k = 0; k < DOF; k++) { // the KDIM0 x KDIM1 block is rank-one
          const VecType* f_ = f + k*3;
          VecType* v_ = v + k*3;
          const VecType fdotr_rdotn_rinv5 = (f_[0]*r[0] + f_[1]*r[1] + f_[2]*r[2]) * rdotn_rinv5;
          for (Integer i = 0; i < 3; i++) v_[i] += r[i] * fdotr_rdotn_rinv5;
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

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv*rinv;
        const VecType rinv5 = rinv2*rinv2*rinv;
        for (Integer k = 0; k < DOF; k++) { // rank-one in the source index
          const VecType* f_ = f + k*3;
          VecType* v_ = v + k*9;
          const VecType fdotr_rinv5 = (f_[0]*r[0] + f_[1]*r[1] + f_[2]*r[2]) * rinv5;
          for (Integer i = 0; i < 3; i++) {
            const VecType ri = r[i]*fdotr_rinv5;
            for (Integer j = 0; j < 3; j++) v_[i*3+j] += ri*r[j];
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

      // Volume potential for uniform unit source/sink; used in periodic-FMM BC correction.
      template <class Real> static void VolPoten(Matrix<Real>& U, const Vector<Real>& X) {
        const Long N = X.Dim() / 3;
        SCTL_ASSERT(X.Dim() == N * 3);
        if (U.Dim(0) != 4 || U.Dim(1) != N * 3) U.ReInit(4, N * 3);
        for (Long i = 0; i < N; i++) {
          const Real x = X[i*3 + 0];
          const Real y = X[i*3 + 1];
          const Real z = X[i*3 + 2];
          const Real rx_2 = y*y + z*z;
          const Real ry_2 = x*x + z*z;
          const Real rz_2 = x*x + y*y;
          U[0][i*3+0] = -rx_2/4; U[0][i*3+1] =       0; U[0][i*3+2] =       0;
          U[1][i*3+0] =       0; U[1][i*3+1] = -ry_2/4; U[1][i*3+2] =       0;
          U[2][i*3+0] =       0; U[2][i*3+1] =       0; U[2][i*3+2] = -rz_2/4;
          U[3][i*3+0] = x/6;
          U[3][i*3+1] = y/6;
          U[3][i*3+2] = z/6;
        }
      }

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv3 = rinv*rinv*rinv;
        for (Integer k = 0; k < DOF; k++) { // single-layer and source/sink share one dot product
          const VecType* f_ = f + k*4;
          VecType* v_ = v + k*3;
          const VecType t = (f_[0]*r[0] + f_[1]*r[1] + f_[2]*r[2] + f_[3]) * rinv3;
          for (Integer i = 0; i < 3; i++) v_[i] += f_[i]*rinv + r[i]*t;
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

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv3 = rinv*rinv*rinv;
        for (Integer k = 0; k < DOF; k++) { // velocity and pressure share one dot product
          const VecType* f_ = f + k*3;
          VecType* v_ = v + k*4;
          const VecType fdotr_rinv3 = (f_[0]*r[0] + f_[1]*r[1] + f_[2]*r[2]) * rinv3;
          for (Integer i = 0; i < 3; i++) v_[i] += f_[i]*rinv + r[i]*fdotr_rinv3;
          v_[3] += fdotr_rinv3;
        }
      }

    };

    struct Laplace3D_Fxd2U {
      static const std::string& Name() {
        static const std::string name = "Laplace3D-Fxd2U";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 25;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][9], const VecType (&r)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        const VecType rinv3 = rinv * rinv2;
        const VecType t5 = VecType((Real)3) * rinv3 * rinv2;
        for (Integer i = 0; i < 3; i++) {
          for (Integer j = 0; j < 3; j++) {
            u[0][i*3+j] = t5 * r[i] * r[j] - (i==j ? rinv3 : VecType::Zero());
          }
        }
      }
    };

    struct Laplace3D_DxdU {
      static const std::string& Name() {
        static const std::string name = "Laplace3D-DxdU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 22;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      // gradient of Laplace3D_DxU
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        const VecType rinv3 = rinv * rinv2;
        const VecType rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
        const VecType t5 = VecType((Real)3) * rdotn * rinv3 * rinv2;
        for (Integer i = 0; i < 3; i++) u[0][i] = n[i] * rinv3 - r[i] * t5;
      }
    };

    struct BiotSavart3D_FxU {
      static const std::string& Name() {
        static const std::string name = "BiotSavart3D-FxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 15;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv3 = rinv * rinv * rinv;
        u[0][0] = VecType::Zero(); u[0][1] = -r[2]*rinv3;     u[0][2] =  r[1]*rinv3;
        u[1][0] =  r[2]*rinv3;     u[1][1] = VecType::Zero(); u[1][2] = -r[0]*rinv3;
        u[2][0] = -r[1]*rinv3;     u[2][1] =  r[0]*rinv3;     u[2][2] = VecType::Zero();
      }

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv3 = rinv * rinv * rinv;
        for (Integer k = 0; k < DOF; k++) { // v <-- f x r / r^3
          const VecType* f_ = f + k*3;
          VecType* v_ = v + k*3;
          v_[0] += (f_[1]*r[2] - r[1]*f_[2]) * rinv3;
          v_[1] += (f_[2]*r[0] - r[2]*f_[0]) * rinv3;
          v_[2] += (f_[0]*r[1] - r[0]*f_[1]) * rinv3;
        }
      }
    };

    struct BiotSavart3D_FxdU {
      static const std::string& Name() {
        static const std::string name = "BiotSavart3D-FxdU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 51;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return -1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][9], const VecType (&r)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        const VecType rinv3 = rinv * rinv2;
        const VecType t5 = VecType((Real)3) * rinv3 * rinv2;
        const VecType Z = VecType::Zero();

        u[0][0] =                    Z; u[1][0] =         t5*r[2]*r[0]; u[2][0] =        -t5*r[1]*r[0];
        u[0][1] =                    Z; u[1][1] =         t5*r[2]*r[1]; u[2][1] =  rinv3 -t5*r[1]*r[1];
        u[0][2] =                    Z; u[1][2] = -rinv3 +t5*r[2]*r[2]; u[2][2] =        -t5*r[1]*r[2];

        u[0][3] =        -t5*r[2]*r[0]; u[1][3] =                    Z; u[2][3] = -rinv3 +t5*r[0]*r[0];
        u[0][4] =        -t5*r[2]*r[1]; u[1][4] =                    Z; u[2][4] =         t5*r[0]*r[1];
        u[0][5] =  rinv3 -t5*r[2]*r[2]; u[1][5] =                    Z; u[2][5] =         t5*r[0]*r[2];

        u[0][6] =         t5*r[1]*r[0]; u[1][6] =  rinv3 -t5*r[0]*r[0]; u[2][6] =                    Z;
        u[0][7] = -rinv3 +t5*r[1]*r[1]; u[1][7] =        -t5*r[0]*r[1]; u[2][7] =                    Z;
        u[0][8] =         t5*r[1]*r[2]; u[1][8] =        -t5*r[0]*r[2]; u[2][8] =                    Z;
      }
    };

    // The Helmholtz kernels read the wavenumber from ctx_ptr, which must point to
    // a value of the same real type used for evaluation. Complex quantities are
    // stored as adjacent real/imaginary pairs, so a complex kernel with KDIM0 x
    // KDIM1 complex entries appears here as 2*KDIM0 x 2*KDIM1 real ones, of which
    // only half are distinct -- hence the fused applies.

    struct Helmholtz3D_FxU {
      static const std::string& Name() {
        static const std::string name = "Helmholtz3D-FxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 20;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[2][2], const VecType (&r)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType sin_rmu, cos_rmu;
        sincos(sin_rmu, cos_rmu, r2 * rinv * mu);
        u[0][0] = cos_rmu * rinv; u[0][1] = sin_rmu * rinv;
        u[1][0] = -u[0][1];       u[1][1] = u[0][0];
      }

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType sin_rmu, cos_rmu;
        sincos(sin_rmu, cos_rmu, r2 * rinv * mu);
        const VecType G0 = cos_rmu * rinv, G1 = sin_rmu * rinv;
        for (Integer k = 0; k < DOF; k++) {
          v[k*2+0] += f[k*2+0]*G0 - f[k*2+1]*G1;
          v[k*2+1] += f[k*2+0]*G1 + f[k*2+1]*G0;
        }
      }
    };

    struct Helmholtz3D_DxU {
      static const std::string& Name() {
        static const std::string name = "Helmholtz3D-DxU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 29;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[2][2], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        const VecType ndotr = n[0]*r[0] + n[1]*r[1] + n[2]*r[2];
        VecType sin_rmu, cos_rmu;
        sincos(sin_rmu, cos_rmu, r2 * rinv * mu);
        u[0][0] = (-mu*sin_rmu - cos_rmu * rinv) * rinv2 * ndotr;
        u[0][1] = ( mu*cos_rmu - sin_rmu * rinv) * rinv2 * ndotr;
        u[1][0] = -u[0][1];  u[1][1] = u[0][0];
      }

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        const VecType ndotr = n[0]*r[0] + n[1]*r[1] + n[2]*r[2];
        VecType sin_rmu, cos_rmu;
        sincos(sin_rmu, cos_rmu, r2 * rinv * mu);
        const VecType G0 = (-mu*sin_rmu - cos_rmu * rinv) * rinv2 * ndotr;
        const VecType G1 = ( mu*cos_rmu - sin_rmu * rinv) * rinv2 * ndotr;
        for (Integer k = 0; k < DOF; k++) {
          v[k*2+0] += f[k*2+0]*G0 - f[k*2+1]*G1;
          v[k*2+1] += f[k*2+0]*G1 + f[k*2+1]*G0;
        }
      }
    };

    struct Helmholtz3D_FxdU {
      static const std::string& Name() {
        static const std::string name = "Helmholtz3D-FxdU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 35;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[2][6], const VecType (&r)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        VecType sin_rmu, cos_rmu;
        sincos(sin_rmu, cos_rmu, r2 * rinv * mu);
        const VecType G0 = (-mu*sin_rmu - cos_rmu * rinv) * rinv2;
        const VecType G1 = ( mu*cos_rmu - sin_rmu * rinv) * rinv2;
        for (Integer i = 0; i < 3; i++) {
          u[0][i*2+0] = G0 * r[i];      u[0][i*2+1] = G1 * r[i];
          u[1][i*2+0] = -u[0][i*2+1];   u[1][i*2+1] = u[0][i*2+0];
        }
      }

      static constexpr bool FUSED_APPLY = true;
      template <Integer digits, Integer DOF, class VecType> static void uKerApply(VecType* v, const VecType (&r)[3], const VecType* n, const VecType* f, const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;
        VecType sin_rmu, cos_rmu;
        sincos(sin_rmu, cos_rmu, r2 * rinv * mu);
        const VecType G0 = (-mu*sin_rmu - cos_rmu * rinv) * rinv2;
        const VecType G1 = ( mu*cos_rmu - sin_rmu * rinv) * rinv2;
        for (Integer k = 0; k < DOF; k++) {
          const VecType fG0 = f[k*2+0]*G0 - f[k*2+1]*G1;
          const VecType fG1 = f[k*2+0]*G1 + f[k*2+1]*G0;
          for (Integer i = 0; i < 3; i++) {
            v[k*6+i*2+0] += fG0 * r[i];
            v[k*6+i*2+1] += fG1 * r[i];
          }
        }
      }
    };

    /**
     * Gradient of the difference between the Helmholtz and Laplace single-layer
     * kernels. The (cos(r*mu) - 1) factor is evaluated through a half-angle
     * identity so it stays accurate as r*mu -> 0, where the two kernels cancel.
     */
    struct HelmholtzDiff3D_FxdU {
      static const std::string& Name() {
        static const std::string name = "HelmholtzDiff3D-FxdU";
        return name;
      }
      static constexpr Integer FLOPS() {
        return 41;
      }
      template <class Real> static constexpr Real uKerScaleFactor() {
        return 1 / (4 * const_pi<Real>());
      }
      template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[2][6], const VecType (&r)[3], const void* ctx_ptr) {
        using Real = typename VecType::ScalarType;
        const VecType mu(*(const Real*)ctx_ptr);
        const VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        const VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        const VecType rinv2 = rinv * rinv;

        VecType sin_h, cos_h;
        sincos(sin_h, cos_h, r2 * rinv * mu * VecType((Real)0.5));
        const VecType two((Real)2);
        const VecType sin_rmu = two * sin_h * cos_h;
        const VecType cos_rmu_m1 = -two * sin_h * sin_h;
        const VecType cos_rmu = VecType((Real)1) + cos_rmu_m1;

        const VecType G0 = (-mu*sin_rmu - cos_rmu_m1 * rinv) * rinv2;
        const VecType G1 = ( mu*cos_rmu - sin_rmu    * rinv) * rinv2;
        for (Integer i = 0; i < 3; i++) {
          u[0][i*2+0] = G0 * r[i];      u[0][i*2+1] = G1 * r[i];
          u[1][i*2+0] = -u[0][i*2+1];   u[1][i*2+1] = u[0][i*2+0];
        }
      }
    };

  }  // namespace kernel_impl

  // Notation:
  // F = single-layer source
  // D = double-layer source
  // U = potential
  // dU = grad potential
  // d2U = Hessian of potential
  using Laplace3D_FxU = GenericKernel<kernel_impl::Laplace3D_FxU>;
  using Laplace3D_DxU = GenericKernel<kernel_impl::Laplace3D_DxU>;
  using Laplace3D_FxdU = GenericKernel<kernel_impl::Laplace3D_FxdU>;
  using Laplace3D_Fxd2U = GenericKernel<kernel_impl::Laplace3D_Fxd2U>;
  using Laplace3D_DxdU = GenericKernel<kernel_impl::Laplace3D_DxdU>;
  using BiotSavart3D_FxU = GenericKernel<kernel_impl::BiotSavart3D_FxU>;
  using BiotSavart3D_FxdU = GenericKernel<kernel_impl::BiotSavart3D_FxdU>;
  using Helmholtz3D_FxU = GenericKernel<kernel_impl::Helmholtz3D_FxU>; // ctx_ptr = pointer to the wavenumber
  using Helmholtz3D_DxU = GenericKernel<kernel_impl::Helmholtz3D_DxU>;
  using Helmholtz3D_FxdU = GenericKernel<kernel_impl::Helmholtz3D_FxdU>;
  using HelmholtzDiff3D_FxdU = GenericKernel<kernel_impl::HelmholtzDiff3D_FxdU>;
  using Stokes3D_FxU = GenericKernel<kernel_impl::Stokes3D_FxU>;
  using Stokes3D_DxU = GenericKernel<kernel_impl::Stokes3D_DxU>;
  using Stokes3D_FxT = GenericKernel<kernel_impl::Stokes3D_FxT>; // single-layer source ---> traction-tensor
  using Stokes3D_FSxU = GenericKernel<kernel_impl::Stokes3D_FSxU>; // single-layer + source/sink ---> velocity (required for FMM translations involving double-layer - M2M, M2L, M2T)
  using Stokes3D_FxUP = GenericKernel<kernel_impl::Stokes3D_FxUP>; // single-layer source ---> velocity + pressure

}  // end namespace

#endif // _SCTL_KERNEL_FUNCTIONS_HPP_
