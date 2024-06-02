.. _fmm-wrapper_hpp:

fmm-wrapper.hpp
===============

This header file provides a wrapper class for PVFMM for distributed memory particle N-body computations.
When PVFMM is not available, computes interactions directly.

Classes and Types
-----------------

.. doxygenclass:: sctl::ParticleFMM
..   :members:
..

    **Constructor**:

    - ``ParticleFMM(const Comm& comm = Comm::Self())``: Constructor.

    **Methods**:

    - ``void SetComm(const Comm& comm)``: Sets communicator for distributed memory parallelism.
    - ``void SetAccuracy(Integer digits)``: Sets FMM accuracy.
    - ``SetKernels(ker_m2m, ker_m2l, ker_l2l)``: Sets FMM kernels.
    - ``AddSrc(src_name, ker_s2m, ker_s2l)``: Adds a source type.
    - ``AddTrg(trg_name, ker_m2t, ker_l2t)``: Adds a target type.
    - ``SetKernelS2T(src_name, trg_name, ker_s2t)``: Sets kernel function for source-to-target interactions.
    - ``DeleteSrc(src_name)``: Deletes a source type.
    - ``DeleteTrg(trg_name)``: Deletes a target type.
    - ``SetSrcCoord(src_name, src_coord, src_normal = Vector<Real>())``: Sets coordinates for a source type.
    - ``SetSrcDensity(name, src_density)``: Sets densities for a source type.
    - ``SetTrgCoord(trg_name, trg_coord)``: Sets coordinates for a target type.
    - ``Eval(U, trg_name) const``: Evaluates the potential for a target type using FMM.
    - ``EvalDirect(U, trg_name) const``: Evaluates the potential for a target type using direct evaluation.

    **Usage guide**: :ref:`Using ParticleFMM class <tutorial-fmm>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/fmm-wrapper.hpp
   :language: c++
