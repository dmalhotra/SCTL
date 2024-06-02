.. _iterator_hpp:

iterator.hpp
============

In debug-build (when ``SCTL_MEMDEBUG`` is defined), this header file provides the `ConstIterator` and `Iterator` classes, which offer iterator functionalities with additional memory debugging capabilities.
In a regular build, these types are aliases for regular pointers.

Classes and Types
-----------------

.. doxygenclass:: sctl::ConstIterator
..   :members:
..

    **Methods**:

    - ``ConstIterator()``: Default constructor.
    - ``ConstIterator(const ConstIterator&)``: Copy constructor.
    - ``ConstIterator& operator=(const ConstIterator&)``: Copy assignment operator.
    - ``explicit ConstIterator(pointer base_, difference_type len_, bool dynamic_alloc = false)``: Constructor with base pointer and length.
    - ``template <class AnotherType> explicit ConstIterator(const ConstIterator<AnotherType>& I)``: Template copy constructor.
    - ``reference operator*() const``: Dereference operator.
    - ``pointer operator->() const``: Member access operator.
    - ``reference operator[](difference_type off) const``: Subscript operator.
    - ``ConstIterator& operator++()``: Pre-increment operator.
    - ``ConstIterator operator++(int)``: Post-increment operator.
    - ``ConstIterator& operator--()``: Pre-decrement operator.
    - ``ConstIterator operator--(int)``: Post-decrement operator.
    - ``ConstIterator& operator+=(difference_type i)``: Addition assignment operator.
    - ``ConstIterator operator+(difference_type i) const``: Addition operator.
    - ``template <class T> friend ConstIterator<T> operator+(typename ConstIterator<T>::difference_type i, const ConstIterator<T>& right)``: Addition operator for constant iterator.
    - ``ConstIterator& operator-=(difference_type i)``: Subtraction assignment operator.
    - ``ConstIterator operator-(difference_type i) const``: Subtraction operator.
    - ``difference_type operator-(const ConstIterator& I) const``: Difference operator.
    - ``bool operator==(const ConstIterator& I) const``: Equality comparison operator.
    - ``bool operator!=(const ConstIterator& I) const``: Inequality comparison operator.
    - ``bool operator<(const ConstIterator& I) const``: Less-than comparison operator.
    - ``bool operator<=(const ConstIterator& I) const``: Less-than-or-equal comparison operator.
    - ``bool operator>(const ConstIterator& I) const``: Greater-than comparison operator.
    - ``bool operator>=(const ConstIterator& I) const``: Greater-than-or-equal comparison operator.
    - ``friend std::ostream& operator<<(std::ostream& out, const ConstIterator& I)``: Output stream operator.

|

.. doxygenclass:: sctl::Iterator
..   :members:
..

    **Methods**:

    - ``Iterator()``: Default constructor.
    - ``Iterator(const Iterator&)``: Copy constructor.
    - ``Iterator& operator=(const Iterator&)``: Copy assignment operator.
    - ``explicit Iterator(pointer base_, difference_type len_, bool dynamic_alloc = false)``: Constructor with base pointer and length.
    - ``template <class AnotherType> explicit Iterator(const ConstIterator<AnotherType>& I)``: Template copy constructor.
    - ``reference operator*() const``: Dereference operator.
    - ``pointer operator->() const``: Member access operator.
    - ``reference operator[](difference_type off) const``: Subscript operator.
    - ``Iterator& operator++()``: Pre-increment operator.
    - ``Iterator operator++(int)``: Post-increment operator.
    - ``Iterator& operator--()``: Pre-decrement operator.
    - ``Iterator operator--(int)``: Post-decrement operator.
    - ``Iterator& operator+=(difference_type i)``: Addition assignment operator.
    - ``Iterator operator+(difference_type i) const``: Addition operator.
    - ``template <class T> friend Iterator<T> operator+(typename Iterator<T>::difference_type i, const Iterator<T>& right)``: Addition operator for iterator.
    - ``Iterator& operator-=(difference_type i)``: Subtraction assignment operator.
    - ``Iterator operator-(difference_type i) const``: Subtraction operator.
    - ``difference_type operator-(const ConstIterator<ValueType>& I) const``: Difference operator.

Functions
---------

.. doxygenfunction:: sctl::NullIterator

|

.. doxygenfunction:: sctl::Ptr2Itr

|

.. doxygenfunction:: sctl::Ptr2ConstItr

|

.. doxygenfunction:: sctl::memcopy

|

.. doxygenfunction:: sctl::memset

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/iterator.hpp
   :language: c++
