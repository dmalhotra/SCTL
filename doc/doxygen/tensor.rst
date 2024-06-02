.. _tensor_hpp:

tensor.hpp
==========

This header file provides the `Tensor` class.

Classes and Types
-----------------

.. doxygenclass:: sctl::Tensor
..   :members:
..

    **Constructor**:

    - ``Tensor()``: Constructs an empty tensor.

    - ``Tensor(src_iter)``: Constructs a tensor from an iterator.

    - ``Tensor(M)``: Copy constructor.

    - ``Tensor(v)``: Constructs a tensor with all elements initialized to a specific value.

    **Methods**:

    - ``operator=``: Copy assignment operator.

    - ``operator=``: Assignment operator setting all elements to a scalar value.

    - ``Order()``: Get the order of the tensor.

    - ``Size()``: Get the total number of elements in the tensor.

    - ``Dim<k>()``: Get the size of a specific dimension of the tensor.

    - ``begin()``: Get an iterator to the beginning of the tensor.

    - ``end()``: Get an iterator to the end of the tensor.

    - ``operator()``: Access a specific element of the tensor.

    - ``RotateLeft() const``: Rotate tensor dimensions to the left.

    - ``RotateRight() const``: Rotate tensor dimensions to the right.

    - ``operator+()``, ``operator-()``: Unary operators.

    - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Element-wise arithmetic operations with a scalar.

    - ``operator+``, ``operator-``: Element-wise arithmetic operations with another tensor with the same dimensions.

    - ``operator*``: Matrix multiplication of two second order tensors.

    **Usage guide**: :ref:`Using Tensor class <tutorial-tensor>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/tensor.hpp
   :language: c++

.. .. doxygenclass:: sctl::Tensor
..    :members:
..    :undoc-members:
..    :outline:

