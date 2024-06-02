.. _vec_hpp:

vec.hpp
=======

This header provides the Vec class for working with SIMD vectors.

Classes and Types
-----------------

.. doxygenclass:: sctl::Vec
..

   **Types**:

   - ``ScalarType``: Type alias for the scalar type of the vector elements.

   - ``VData``: Type alias for the internal data representation of the vector.

   - ``MaskType``: Type alias for the mask type associated with the vector.

   **Methods**:

   - ``Size``: Get the size of the vector.

   - ``Zero``: Create a vector initialized with all elements set to zero.

   - ``Load1``: Load a scalar value into all elements of the vector.

   - ``Load``: Load a vector of scalar values from unaligned memory.

   - ``LoadAligned``: Load a vector of scalar values from aligned memory.

   - ``Store``: Stores the vector data into unaligned memory.

   - ``StoreAligned``: Stores the vector data into aligned memory.
   
   - ``operator=``: Copy assignment operator.
   
   - ``operator[]``: Accesses the element at the specified index.
   
   - ``insert(i, value)``: Insert a value at the specified index in the vector.
   
   - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with another Vec.
   
   - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with a scalar value.
   
   - ``operator&``, ``operator^``, ``operator|``, ``AndNot``: Bitwise operations.
   
   - ``operator<<``, ``operator>>``: Bitwise shift operations.
   
   - ``max``, ``min``: Element-wise maximum and minimum.
   
   - ``approx_rsqrt``, ``approx_sqrt``, ``approx_exp``, ``sincos``, ``approx_sincos``: Mathematical functions.
   
   - ``get``: Retrieves the vector data.

   - ``set``: Sets the vector data.

   **Usage guide**: :ref:`Using Vec class <tutorial-vec>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/vec.hpp
   :language: cpp

