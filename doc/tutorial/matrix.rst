.. _tutorial-matrix:

Using the Matrix class
======================

The `Matrix` class is a template class designed to represent and manipulate matrices.
It provides various operations for matrix-matrix and matrix-scalar computations, as well as utilities for input/output, transposition, singular value decomposition (SVD), and pseudo-inverse computation.


This tutorial provided a basic overview of how to use the Matrix class.
For more advanced usage and additional features, please refer to the Matrix class API in :ref:`matrix.hpp <matrix_hpp>`.

.. :ref:`Matrix class documentation <matrix-dox>`.

1. **Instantiate the Class**: Instantiate an object of the `Matrix` class with the desired template parameter (`ValueType`).

   .. code-block:: cpp

      Matrix<double> mat;

2. **Initialize the Matrix**: Initialize the matrix with its dimensions and optionally with data.

   .. code-block:: cpp

      Matrix<double> mat(3, 3);  // Creates a 3x3 matrix filled with default-initialized values.

3. **Perform Matrix Operations**:
   You can perform various matrix operations such as addition, subtraction, multiplication, and element-wise operations.

   .. code-block:: cpp

      Matrix<double> A(3, 3);
      Matrix<double> B(3, 3);
      Matrix<double> C = A + B;  // Addition
      Matrix<double> D = A * B;  // Matrix multiplication
      Matrix<double> E = A * 2.0;  // Scalar multiplication

4. **Access Matrix Elements**:
   You can access individual elements of the matrix using the `operator()` or `operator[]`.

   .. code-block:: cpp

      double val = mat(0, 0);  // Access element at row 0, column 0
      mat(1, 2) = 10.0;  // Assign a new value to the element at row 1, column 2

5. **Transpose the Matrix**:
   Transpose the matrix using the `Transpose` method.

   .. code-block:: cpp

      Matrix<double> transposed = mat.Transpose();

