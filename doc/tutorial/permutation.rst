.. _tutorial-permutation:

Using the Permutation class
===========================

This tutorial provides an overview of the Permutation class and demonstrates how to create, manipulate, and perform operations with permutation objects.
For more advanced usage and additional features, please refer to the :ref:`Permutation class documentation <permutation-dox>`.

Creating Permutation Matrices
------------------------------

You can create a permutation matrix using one of the available constructors. For example:

.. code-block:: cpp

   Permutation<int> perm;  // Creates an empty permutation matrix
   Permutation<double> randPerm = Permutation<double>::RandPerm(5);  // Creates a random permutation matrix of size 5

Accessing Permutation Matrix Properties
----------------------------------------

The Permutation class provides methods to access properties of permutation matrices, such as dimension, permutation vector, and scaling vector:

.. code-block:: cpp

   Long dimension = randPerm.Dim();  // Get the dimension of the permutation matrix
   Vector<Long> permutationVector = randPerm.perm;  // Get the permutation vector
   Vector<double> scalingVector = randPerm.scal;  // Get the scaling vector

Performing Operations
----------------------

You can perform various operations with permutation matrices, including multiplication, division, and transposition:

.. code-block:: cpp

   randPerm *= 2.0;  // Multiply the permutation matrix by a scalar value
   randPerm.Transpose();  // Compute the transpose of the permutation matrix

Using Operators
---------------

The Permutation class overloads operators to facilitate operations with permutation matrices:

.. code-block:: cpp

   Permutation<double> result = randPerm * anotherPermutationMatrix;  // Multiply two permutation matrices
   Matrix<double> transformedMatrix = randPerm * someMatrix;  // Multiply a matrix by the permutation matrix

Outputting Permutation Matrices
-------------------------------

You can output permutation operator to the console or a file using the stream insertion operator `<<`:

.. code-block:: cpp

   std::cout << randPerm << std::endl;  // Output the permutation operator to the console

