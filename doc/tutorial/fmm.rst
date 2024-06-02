.. _tutorial-fmm:

Using the ParticleFMM Class
===========================

The `ParticleFMM` class provides an efficient way to evaluate potentials from particle sources using `PVFMM <http://pvfmm.org>`_.
A broad overview on how to use this class is presented here.
The complete API can be found in :ref:`fmm-wrapper.hpp <fmm-wrapper_hpp>`.

Overview
--------

The `ParticleFMM` class allows for distributed memory parallelism and supports custom kernel functions for multipole and local translations. Here's a brief overview of its main functionalities:

1. **Constructor**:

   - ``ParticleFMM(comm = Comm::Self())``: Initializes the FMM object with an optional communicator for parallelism.


2. **Setting Parameters**:

   - ``SetComm(comm)``: Sets the communicator for parallelism.

   - ``SetAccuracy(digits)``: Sets the accuracy of the FMM evaluation in terms of the number of digits.

3. **Setting Kernel Functions**:

   - ``SetKernels(ker_m2m, ker_m2l, ker_l2l)``: Sets the kernels for multipole-to-multipole, multipole-to-local, and local-to-local translations.

   - ``AddSrc(name, ker_s2m, ker_s2l)``: Adds a source type with kernels for source-to-multipole and source-to-local translations.

   - ``AddTrg(name, ker_m2t, ker_l2t)``: Adds a target type with kernels for multipole-to-target and local-to-target translations.

   - ``SetKernelS2T(src_name, trg_name, ker_s2t)``: Sets the kernel for source-to-target translations.

4. **Managing Source and Target Types**:

   - ``DeleteSrc(name)``: Deletes a source type.

   - ``DeleteTrg(name)``: Deletes a target type.

5. **Setting Coordinates and Densities**:

   - ``SetSrcCoord(name, src_coord, src_normal = Vector<Real>())``: Sets the coordinates for a source type.

   - ``SetSrcDensity(name, src_density)``: Sets the densities for a source type.

   - ``SetTrgCoord(name, trg_coord)``: Sets the coordinates for a target type.

6. **Evaluating Potentials**:

   - ``Eval(U, trg_name) const``: Evaluates the potential for a target type using FMM.

   - ``EvalDirect(U, trg_name) const``: Evaluates the potential for a target type using direct evaluation.

Example Usage
-------------

1. **Creating the FMM Object**

   First, create an instance of the `ParticleFMM` class. Optionally, you can provide a communicator for parallel processing.

   .. code-block:: cpp

      sctl::ParticleFMM<double, 3> fmm(comm);

2. **Setting Accuracy**

   Set the desired accuracy for the FMM evaluation.

   .. code-block:: cpp

      fmm.SetAccuracy(10);

3. **Setting Kernel Functions**

   Define and set the kernel functions for multipole and local translations.

   .. code-block:: cpp

      Stokes3D_FSxU kernel_m2l;
      Stokes3D_FxU kernel_sl;
      Stokes3D_DxU kernel_dl;

      fmm.SetKernels(kernel_m2l, kernel_m2l, kernel_sl);
      fmm.AddTrg("Velocity", kernel_m2l, kernel_sl);
      fmm.AddSrc("DoubleLayer", kernel_dl, kernel_dl);
      fmm.SetKernelS2T("DoubleLayer", "Velocity", kernel_dl);

4. **Setting Particle Data**

   Set the coordinates and densities for the source and target particles.

   .. code-block:: cpp

      fmm.SetTrgCoord("Velocity", trg_coord);
      fmm.SetSrcCoord("DoubleLayer", dl_coord, dl_norml);
      fmm.SetSrcDensity("DoubleLayer", dl_den);

5. **Evaluating Potentials**

   Evaluate the potential using FMM.

   .. code-block:: cpp

      Vector<double> Ufmm, Uref;
      fmm.Eval(Ufmm, "Velocity");  // FMM evaluation

