.. _tutorial-vector:

Using the Vector class
===========================

This tutorial provided a basic overview of how to use the Vector class.
For more advanced usage and additional features, please refer to the :ref:`Vector class documentation <vector-dox>`.

The Vector class is a dynamically resizable array with various utility
functions. It provides functionalities similar to std::vector with additional
features and optimizations.

..  Including the Header
..  ---------------------
..  To use the Vector class in your code, include the `sctl/vector.hpp` header file:
.. 
.. .. code-block:: cpp
.. 
..     #include <sctl/vector.hpp>

Creating a Vector
-----------------
To create a Vector object, you can use one of the following constructors:

1. Default Constructor:
   Creates an empty vector.

    .. code-block:: cpp

        sctl::Vector<double> vec1;

2. Constructor with Dimension:
   Creates a vector of a specified dimension.

    .. code-block:: cpp

        sctl::Vector<double> vec2(10);

3. Constructor with Initializer List:
   Creates a vector initialized with values from an initializer list.

    .. code-block:: cpp

        sctl::Vector<double> vec3 = {1.0, 2.0, 3.0};

Accessing Elements
-------------------
You can access elements of the vector using the subscript operator []:

.. code-block:: cpp

    double elem = vec3[1];  // Accesses the second element (index 1) of vec3

Vector Operations
-----------------
The Vector class supports various vector operations such as addition, subtraction, multiplication, division, and element-wise operations.

1. Addition:

.. code-block:: cpp

    sctl::Vector<double> result = vec1 + vec2;

2. Subtraction:

.. code-block:: cpp

    sctl::Vector<double> result = vec1 - vec2;

3. Multiplication:

.. code-block:: cpp

    sctl::Vector<double> result = vec1 * vec2;

4. Division:

.. code-block:: cpp

    sctl::Vector<double> result = vec1 / vec2;

5. Element-wise Operations:
   The Vector class also supports element-wise addition, subtraction, multiplication, and division with scalars:

.. code-block:: cpp

    sctl::Vector<double> result = vec1 + 5.0;  // Adds 5.0 to each element of vec1

Iterating Over Elements
-----------------------
You can iterate over the elements of a vector using iterators:

.. code-block:: cpp

    for (auto it = vec1.begin(); it != vec1.end(); ++it) {
        // Access *it
    }

Alternatively, you can use range-based for loop:

.. code-block:: cpp

    for (const auto& elem : vec1) {
        // Access elem
    }


