.. _permutation_hpp:

permutation.hpp
===============

This header file provides the `Permutation` class.

Classes and Types
-----------------

.. doxygenclass:: sctl::Permutation
..   :members:
..

    **Attributes**:

    - ``perm`` (Vector<Long>): The permutation vector representing the indices of the elements in the original order.

    - ``scal`` (Vector<ValueType>): The scaling vector representing the scaling factors applied to each element.

    **Constructor**:

    - ``Permutation()``

    - ``Permutation(size)``: Constructs a permutation operator of size `size`.

    **Methods**:

    - ``RandPerm(size)``: Generates a random permutation operator of size `size`.

    - ``GetMatrix() const``: Retrieves the permutation operator as a regular matrix.

    - ``Dim() const``: Returns the dimension of the permutation operator.

    - ``Transpose()``: Computes the transpose of the permutation operator.

    - ``operator*=(s)``,  ``operator/=(s)``: In-place arithmetic operations with a scalar.

    - ``operator*``: Multiplication of a scalar with each element of the permutation operator.

    - ``operator*``: Multiplication of a matrix by the permutation operator.

    - ``operator*``: Multiplication of two permutation matrices.

    - ``operator<<``: Stream insertion operator for printing the permutation operator.

    **Usage guide**: :ref:`Using Permutation class <tutorial-permutation>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/permutation.hpp
   :language: c++

.. .. doxygenclass:: sctl::Permutation
..    :members:
..    :undoc-members:
..    :outline:

