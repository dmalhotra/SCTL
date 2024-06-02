.. _tutorial-permutation:

Using the Permutation class
===========================

This tutorial provides an overview of the Permutation class and demonstrates how to create, manipulate, and perform operations with permutation objects.
For more advanced usage and additional features, please refer to the Permutation class API in :ref:`permutation.hpp <permutation_hpp>`.

.. :ref:`Permutation class documentation <permutation-dox>`.

1. **Creating Permutation Operator**: You can create a permutation operator using one of the available constructors. For example:
   
   .. code-block:: cpp
   
      Permutation<int> perm;  // Creates an empty permutation operator
      Permutation<double> randPerm = Permutation<double>::RandPerm(5);  // Creates a random permutation operator of size 5
   
2. **Accessing Permutation Operator Properties**: The Permutation class provides methods to access properties of permutation operators, such as dimension, permutation vector, and scaling vector:
   
   .. code-block:: cpp
   
      Long dimension = randPerm.Dim();  // Get the dimension of the permutation operator
      Vector<Long> permutationVector = randPerm.perm;  // Get the permutation vector
      Vector<double> scalingVector = randPerm.scal;  // Get the scaling vector
   
3. **Performing Operations**: You can perform various operations with permutation operator, including scalar multiplication, and transposition:
   
   .. code-block:: cpp
   
      randPerm *= 2.0;  // Multiply the permutation operator by a scalar value
      randPerm.Transpose();  // Compute the transpose of the permutation operator
   
4. **Using Operators**: The Permutation class overloads operators to facilitate operations with matrices and other permutation objects:
   
   .. code-block:: cpp
   
      Permutation<double> result = randPerm * anotherPermutationOperator;  // Multiply two permutation operators
      Matrix<double> transformedMatrix = randPerm * someMatrix;  // Multiply a matrix by the permutation operator
   
