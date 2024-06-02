.. _complex_hpp:

complex.hpp
===========

This header file provides the `Complex` template class for representing and performing operations on complex numbers.
The class offers functionalities for arithmetic operations including addition, subtraction, multiplication, and division.

Classes and Types
-----------------

.. doxygenclass:: sctl::Complex
..   :members:
..

    **Constructor**:

    - ``Complex(ValueType r = 0, ValueType i = 0)``: Constructs a complex number with specified real and imaginary parts, defaulting to zero.

    **Methods**:

    - ``operator-``: Unary negation operator.
    - ``conj``: Returns the conjugate of the complex number.
    - ``operator==``, ``operator!=``: Equality and inequality comparison operators.
    - ``operator+=``, ``operator-=``, ``operator*=``, ``operator/=``: In-place arithmetic operations with another complex number.
    - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with another complex number or a scalar.
    - ``operator<<(output, V)``: Output stream operator for complex numbers.

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/complex.hpp
   :language: c++

