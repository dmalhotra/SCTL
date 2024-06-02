.. _tutorial-vector:

Using the Vector class
===========================

The Vector class is a dynamically resizable array with various utility
functions. It provides functionalities similar to std::vector with additional
features and optimizations.

This tutorial provided a basic overview of how to use the Vector class.
For more advanced usage and additional features, please refer to the Vector class API in :ref:`vector.hpp <vector_hpp>`.

.. :ref:`Vector class documentation <vector-dox>`.

1. **Creating a Vector**: To create a Vector object, you can use one of the following constructors:

   - **Default Constructor**: Creates an empty vector.

      .. code-block:: cpp

          sctl::Vector<double> vec1;

   - **Constructor with Dimension**: Creates a vector of a specified dimension.

      .. code-block:: cpp

          sctl::Vector<double> vec2(10);

   - **Constructor with Initializer List**: Creates a vector initialized with values from an initializer list.

      .. code-block:: cpp

          sctl::Vector<double> vec3 = {1.0, 2.0, 3.0};

2. **Accessing Elements**: You can access elements of the vector using the subscript operator []:

   .. code-block:: cpp

       double elem = vec3[1];  // Accesses the second element (index 1) of vec3

3. **Vector Operations**: The Vector class supports various vector operations such as addition, subtraction, multiplication, division, and element-wise operations.

   .. code-block:: cpp

       sctl::Vector<double> result_add = vec1 + vec2;
       sctl::Vector<double> result_sub = vec1 - vec2;
       sctl::Vector<double> result_mul = vec1 * vec2;
       sctl::Vector<double> result_div = vec1 / vec2;

   The Vector class also supports element-wise addition, subtraction, multiplication, and division with scalars:

   .. code-block:: cpp

       sctl::Vector<double> result = vec1 + 5.0;  // Adds 5.0 to each element of vec1

4. **Iterating Over Elements**: You can iterate over the elements of a vector using iterators:

   .. code-block:: cpp

       for (auto it = vec1.begin(); it != vec1.end(); ++it) {
           // Access *it
       }

   Alternatively, you can use range-based for loop:

   .. code-block:: cpp

       for (const auto& elem : vec1) {
           // Access elem
       }


