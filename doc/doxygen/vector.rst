.. _vector_hpp:

vector.hpp
==========

This header file provides the `Vector` class and associated functions for managing vectors of elements.

Classes and Types
-----------------

.. doxygenclass:: sctl::Vector
..   :members:
..

    **Constructor**:

    - ``Vector()``: Default constructor.

    - ``Vector(dim, data, own_data)``: Constructor with specified dimension, data pointer, and ownership flag.

    - ``Vector(const Vector& V)``: Copy constructor.

    - ``Vector(const std::vector<ValueType>& V)``: Constructor from `std::vector`.

    - ``Vector(std::initializer_list<ValueType> V)``: Constructor from initializer list.

    **Methods**:

    - ``Swap(v1)``: Swaps the contents of two vectors.

    - ``ReInit(dim, data, own_data)``: Reinitializes the vector with specified dimension, data pointer, and ownership flag.

    - ``Write(fname)``: Writes the vector to a file.

    - ``Read(fname)``: Reads the vector from a file.

    - ``Dim()``: Returns the dimension of the vector.

    - ``SetZero()``: Sets all elements of the vector to zero.

    - ``begin()``: Returns an iterator to the beginning of the vector.

    - ``end()``: Returns an iterator to the end of the vector.

    - ``PushBack(x)``: Appends an element to the end of the vector.

    - ``operator[]``: Accesses elements of the vector by index.

    - ``operator=``: Assigns the vector from another vector or `std::vector`.

    - ``operator+=``, ``operator-=``, ``operator*=``, ``operator/=``: In-place arithmetic operations with another vector.

    - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with another vector.

    - ``operator=``, ``operator+=``, ``operator-=``, ``operator*=``, ``operator/=``: In-place arithmetic operations with a scalar.

    - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with a scalar.
    
    - ``operator-``: Unary negation.

    - ``operator<<``: Output stream operator for writing a vector to an output stream.

    **Usage guide**: :ref:`Using Vector class <tutorial-vector>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/vector.hpp
   :language: c++

.. .. doxygenclass:: sctl::Vector
..    :members:

..   :members-only:


..   :undoc-members:




.. .. role:: cppcode(code)
..    :language: c++
.. 
.. :cppcode:`class Vector<T>`
.. ==========================

