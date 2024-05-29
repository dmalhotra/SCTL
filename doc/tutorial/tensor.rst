.. _tutorial-tensor:

Using the Tensor class
======================

The `Tensor` class is a template class serving as a container for storing multidimensional arrays of data with various operations defined on them.
It is a statically sized, meaning that the dimensions are known at compile time.
The template parameter determine the element type, allowing for flexibility in the type of data stored in the tensor.

Key Features:
    1. **Multidimensional Representation**: Supports tensors of any number of dimensions, allowing users to work with data structures ranging from simple vectors to complex multi-dimensional arrays.
    
    2. **Extensive Operations**: Provides a wide range of operations, including element-wise arithmetic operations (addition, subtraction, multiplication, division), matrix multiplication (for 2D tensors), rotation (generalization of transpose) operation, and more.
    
    4. **Compile-Time Dimension Handling**: Dimensions of the tensor can be queried at compile-time, enabling optimizations and compile-time error checking.

The following is a general overview on using the Tensor class.
For more advanced usage and additional features, please refer to the Tensor class API in :ref:`tensor.hpp <tensor_hpp>`.

.. :ref:`Tensor class documentation <tensor-dox>`.

1. **Creating a Tensor**:

.. code-block:: cpp

    // Define a 2x3 tensor of doubles
    Tensor<double, true, 2, 3> tensor;

2. **Initializing Tensor Elements**: The data is stored in row-major ordering.

.. code-block:: cpp

    // Initialize tensor elements with random values
    for (auto& x : tensor)
        x = drand48();

3. **Accessing Tensor Elements**:

.. code-block:: cpp

    // Accessing individual elements of the tensor
    double element = tensor(0, 1); // Accessing element at index (0, 1)

4. **Performing Operations**:

.. code-block:: cpp

    // Perform element-wise multiplication by a scalar
    Tensor<double, true, 2, 3> scaled_tensor = tensor * 10;

    // Perform rotation of the tensor (generalization of transpose operation)
    Tensor<double, true, 3, 2> rotated_tensor = tensor.RotateLeft();

    // Perform matrix multiplication (only for second-order tensors)
    Tensor<double, true, 2, 2> resultMatrix = tensor * rotated_tensor;

5. **Iterating Over Tensor Elements**:

.. code-block:: cpp

    // Iterate over tensor elements
    for (auto it = tensor.begin(); it != tensor.end(); ++it) {
        // Access and process each element
    }

6. **Querying Tensor Properties**:

.. code-block:: cpp

    // Get tensor properties
    long order = tensor.Order(); // number of dimensions
    long size = tensor.Size(); // total number of elements
    long dimension0 = tensor.template Dim<0>();
    long dimension1 = tensor.template Dim<1>();

