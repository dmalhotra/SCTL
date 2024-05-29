.. _tutorial-vec:

Using the Vec class
===================

This tutorial provided a basic overview of how to use the Vec class.
For more advanced usage and additional features, please refer to the Vector class API in :ref:`vec.hpp <vec_hpp>`.

The `Vec` class enables efficient parallelization
of computations on multiple data elements simultaneously using SIMD (Single
Instruction, Multiple Data) instructions. This tutorial will guide you through
the usage of the `Vec` class, including its initialization, arithmetic operations,
comparison operators, and other functionalities.

Initializing Vectors
---------------------

You can initialize vectors using various constructors provided by the `Vec` class.
Here are some common initialization methods:

- **Zero-Initialized Vector**:

  .. code-block:: cpp

      Vec<double> zeroVec = Vec<double>::Zero();

- **Initializing with a Scalar Value**:

  .. code-block:: cpp

      Vec<double> scalarVec(5.0); // All elements initialized with 5.0

- **Initializing with Multiple Scalar Values**:

  .. code-block:: cpp

      Vec<double> multiVec(1.0, 2.0, 3.0, 4.0); // Elements: {1.0, 2.0, 3.0, 4.0}

- **Loading from Memory**:

  .. code-block:: cpp

      double data[4] = {1.0, 2.0, 3.0, 4.0};
      Vec<double> loadedVec = Vec<double>::Load(data); // Load from unaligned memory

  .. code-block:: cpp

      double alignedData[4] __attribute__((aligned(64))) = {1.0, 2.0, 3.0, 4.0};
      Vec<double> alignedVec = Vec<double>::LoadAligned(alignedData); // Load from aligned memory

Accessing Vector Elements
--------------------------

You can access individual elements of the vector using the subscript operator `[]`:

  .. code-block:: cpp

      Vec<double> vec(1.0, 2.0, 3.0);
      double secondElement = vec[1]; // Accessing the second element (index 1)

Arithmetic Operations
----------------------

The `Vec` class supports various arithmetic operations such as addition, subtraction,
multiplication, and division:

  .. code-block:: cpp

      Vec<double> a(1.0, 2.0, 3.0, 4.0);
      Vec<double> b(5.0, 6.0, 7.0, 8.0);

      Vec<double> sum = a + b; // Element-wise addition
      Vec<double> difference = a - b; // Element-wise subtraction
      Vec<double> product = a * b; // Element-wise multiplication
      Vec<double> quotient = a / b; // Element-wise division

Comparison Operations
----------------------

You can compare vectors using comparison operators, which return a mask indicating
the comparison result. The mask can then be used in other operations.

  .. code-block:: cpp

      Vec<double> a(1.0, 2.0, 3.0, 4.0);
      Vec<double> b(5.0, 2.0, 7.0, -1.0);

      auto lessThanMask = a < b; // Element-wise less-than comparison
      auto greaterThanMask = a > b; // Element-wise greater-than comparison

Other Operations
-----------------

- **Store to Memory**:

  .. code-block:: cpp

      double result[4];
      sum.Store(result); // Store vector data to unaligned memory

  .. code-block:: cpp

      double alignedResult[4] __attribute__((aligned(64)));
      sum.StoreAligned(alignedResult); // Store vector data to aligned memory

- **Other Mathematical Functions**:

  Additional mathematical functions such as square root (`approx_sqrt`), reciprocal
  square root (`approx_rsqrt`), sine and cosine (`sincos`), and exponential (`exp`)
  are provided.

- **Printing Vectors**:

  Vectors can be printed using the `operator<<`:

  .. code-block:: cpp

      Vec<double,4> vec(1.0, 2.0, 3.0, 4.0);
      std::cout << vec << std::endl; // Output: 1, 2, 3, 4

