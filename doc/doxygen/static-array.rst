.. _static-array_hpp:

static-array.hpp
================

In debug-build (when ``SCTL_MEMDEBUG`` is defined), this header file provides the `StaticArray` class, which represents a fixed-size array with checks for memory errors such as out-of-bounds accesses.
It is intended to be used as regular C-style fixed-size array (such as :code:`double X[10]`) and in regular build `StaticArray` is an alias for C-style array.

Classes and Types
-----------------

.. doxygenclass:: sctl::StaticArray
..   :members:
..

    **Methods**:

    - ``StaticArray()``: Default constructor.
    - ``StaticArray(const StaticArray&)``: Copy constructor.
    - ``StaticArray& operator=(const StaticArray&)``: Copy assignment operator.
    - ``StaticArray(std::initializer_list<ValueType> arr_)``: Constructor with initializer list.
    - ``~StaticArray()``: Destructor.
    - ``const ValueType& operator*() const``: Dereference operator for const access.
    - ``ValueType& operator*()``: Dereference operator for non-const access.
    - ``const ValueType* operator->() const``: Member access operator for const access.
    - ``ValueType* operator->()``: Member access operator for non-const access.
    - ``const ValueType& operator[](difference_type off) const``: Subscript operator for const access.
    - ``ValueType& operator[](difference_type off)``: Subscript operator for non-const access.
    - ``operator ConstIterator<ValueType>() const``: Conversion to const iterator.
    - ``operator Iterator<ValueType>()``: Conversion to non-const iterator.
    - ``ConstIterator<ValueType> operator+(difference_type i) const``: Addition with a difference type for const iterator.
    - ``Iterator<ValueType> operator+(difference_type i)``: Addition with a difference type for non-const iterator.
    - ``ConstIterator<ValueType> operator-(difference_type i) const``: Subtraction with a difference type for const iterator.
    - ``Iterator<ValueType> operator-(difference_type i)``: Subtraction with a difference type for non-const iterator.
    - ``difference_type operator-(const ConstIterator<ValueType>& I) const``: Subtraction between iterators.
    - ``bool operator==(const ConstIterator<ValueType>& I) const``: Equality comparison operator.
    - ``bool operator!=(const ConstIterator<ValueType>& I) const``: Inequality comparison operator.
    - ``bool operator< (const ConstIterator<ValueType>& I) const``: Less than comparison operator.
    - ``bool operator<=(const ConstIterator<ValueType>& I) const``: Less than or equal to comparison operator.
    - ``bool operator> (const ConstIterator<ValueType>& I) const``: Greater than comparison operator.
    - ``bool operator>=(const ConstIterator<ValueType>& I) const``: Greater than or equal to comparison operator.

..    **Usage guide**: :ref:`Using StaticArray <tutorial-staticarray>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/static-array.hpp
   :language: c++

