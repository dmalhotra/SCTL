.. _tutorial-kernels:

Writing Custom Kernel Objects
=============================

In scientific computing, particularly in integral equation methods and fast multipole methods, kernel functions play a crucial role.
This tutorial will guide you through writing custom PDE kernels using the `GenericKernel` class template.
Predefined kernel functions for Laplace and Stokes in 3D can be found in :ref:`kernel_functions.hpp <kernel_functions_hpp>`.
The API for the `GenericKernel` class is documented in :ref:`generic-kernel.hpp <generic-kernel_hpp>`.

1. **Define the Micro-Kernel**

   The micro-kernel struct must include the following member functions:

   - **Name**: Returns the kernel's unique name.
   - **FLOPS**: Returns the number of floating-point operations for each scalar kernel evaluation.
   - **uKerScaleFactor**: Returns the scaling factor for the kernel.
   - **uKerMatrix**: Computes the kernel matrix given a distance vector (and optionally a normal vector for double-layer kernels).

   **Example: Laplace single-layer micro-kernel**:

   .. code-block:: cpp

       struct Laplace3D_SL_uKer {
         static const std::string& Name() {
           static const std::string name = "Laplace3D-SL";
           return name;
         }

         static constexpr Integer FLOPS() {
           return 6;
         }

         template <class Real> static constexpr Real uKerScaleFactor() {
           return 1 / (4 * const_pi<Real>());
         }

         template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const void* ctx_ptr) {
           VecType r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
           VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
           u[0][0] = rinv;
         }
       };

   **Example: Laplace double-layer micro-kernel**:

   .. code-block:: cpp

       struct Laplace3D_DL_uKer {
         static const std::string& Name() {
           static const std::string name = "Laplace3D-DL";
           return name;
         }

         static constexpr Integer FLOPS() {
           return 14;
         }

         template <class Real> static constexpr Real uKerScaleFactor() {
           return 1 / (4 * const_pi<Real>());
         }

         template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
           VecType r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
           VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
           VecType rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
           VecType rinv3 = rinv * rinv * rinv;
           u[0][0] = rdotn * rinv3;
         }
       };

   The `uKerMatrix` routine for the double-layer kernel has an additional argument for the normal vector at the source particle.
   The kernel matrix ``u[SRC_DOF][TRG_DOF]``, the distance vector ``r[DIM]``, and the normal vector ``n[DIM]`` are arrays of SIMD (Single Instruction, Multiple Data) vector type (:ref:`Vec\<Real,DIM\> <vec_hpp>`) for enhanced performance.
   The dimensions ``SRC_DOF`` and ``TRG_DOF`` are the dimensions of the source density and the target potential; ``DIM`` is the dimension of the coordinate space.
   In the above example, for 3D Laplace, ``DIM=3`` and both the density and the potential are scalars (``SRC_DOF=1`` and ``TRG_DOF=1``).

2. **Define the Kernel Object**

   The micro-kernel struct is passed as a template parameter to the `GenericKernel` class to define the new kernel object:

   .. code-block:: cpp

       using Laplace3D_SL = GenericKernel<Laplace3D_SL_uKer>;
       using Laplace3D_DL = GenericKernel<Laplace3D_DL_uKer>;


