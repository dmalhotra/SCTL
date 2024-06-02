.. _matrix_hpp:

matrix.hpp
==========

This header file provides the `Matrix` class.

Classes and Types
-----------------

.. doxygenclass:: sctl::Matrix
..   :members:
..

    **Constructor**:

    - ``Matrix()``: Constructs an empty matrix.

    - ``Matrix(dim1, dim2, data_, own_data_)``: Constructor to create a matrix with specified dimensions and optional initial data.

    - ``Matrix(M)``: Copy constructor.

    **Methods**:

    - ``Swap(M)``: Swaps the contents of two matrices.

    - ``ReInit(dim1, dim2, data_, own_data_)``: Reinitializes the matrix with new dimensions and optional initial data.

    - ``Write<Type>(fname)``: Writes the matrix to a file with specified type.

    - ``Read<Type>(fname)``: Reads the matrix data from a file with specified type.

    - ``Dim(i)``: Returns the size of the matrix along the specified dimension.

    - ``SetZero()``: Sets all elements of the matrix to zero.

    - ``begin()``: Returns an iterator to the beginning of the matrix.

    - ``end()``: Returns an iterator to the end of the matrix.

    - ``RowPerm(P)``: Permutes the rows of the matrix according to the given permutation.

    - ``ColPerm(P)``: Permutes the columns of the matrix according to the given permutation.

    - ``Transpose()``: Computes the transpose of the matrix.

    - ``SVD(tU, tS, tVT)``: Computes the Singular Value Decomposition (SVD) of the matrix. Original matrix is destroyed.

    - ``pinv(eps)``: Computes the Moore-Penrose pseudo-inverse of the matrix. Original matrix is destroyed.

    - ``operator=``, ``operator+=``, ``operator-=``: In-place arithmetic operations with another matrix.

    - ``operator+``, ``operator-``: Arithmetic operations with another matrix.

    - ``operator*``: Multiplies this matrix with another matrix.

    - ``GEMM``: Computes matrix-matrix multiplication.

    - ``operator=``, ``operator+=``, ``operator-=``, ``operator*=``, ``operator/=``: In-place arithmetic operations with a scalar.

    - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with a scalar.
    
    **Usage guide**: :ref:`Using Matrix class <tutorial-matrix>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/matrix.hpp
   :language: c++

.. .. doxygenclass:: sctl::Matrix
..    :members:
..    :undoc-members:
..    :outline:

