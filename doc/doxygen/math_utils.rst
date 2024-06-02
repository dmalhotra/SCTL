.. _math_utils_hpp:

math_utils.hpp
==============

This header file provides various mathematical utilities, including functions for fundamental mathematical operations, and constants.
It also provides the `QuadReal` class for quadruple-precision floating-point numbers when
``SCTL_QUAD_T`` is defined (e.g. with GCC we can use the compiler flag ``-DSCTL_QUAD_T=__float128``).

Classes and Types
-----------------

.. doxygenclass:: sctl::QuadReal
..   :members:
..

    **Constructor**:

    - ``QuadReal()``: Default constructor.
    - ``QuadReal(const QuadReal& v)``: Copy constructor.
    - ``template <class ValueType> QuadReal(ValueType v)``: Constructor with explicit conversion from another type.

    **Methods**:

    - ``operator=``: Copy assignment operator.
    - ``operator ValueType() const``: Explicit conversion operator to another type.
    - ``operator+=``, ``operator-=``, ``operator*=``, ``operator/=``: In-place arithmetic operations with another `QuadReal`.
    - ``operator+``, ``operator-``, ``operator*``, ``operator/``: Arithmetic operations with another `QuadReal`.
    - ``operator-``: Unary negation operator.
    - ``operator<``, ``operator>``, ``operator!=``, ``operator==``, ``operator<=``, ``operator>=``: Comparison operators.
    - ``friend operator+``, ``friend operator-``, ``friend operator*``, ``friend operator/``: Arithmetic operations with `QuadRealType`.
    - ``friend operator<``, ``friend operator>``, ``friend operator!=``, ``friend operator==``, ``friend operator<=``, ``friend operator>=``: Comparison operators with `QuadRealType`.
    - ``friend trunc(const QuadReal)``: Truncates a `QuadReal`.

|

Functions
---------

**Mathematical Constants**:

- ``template <class Real> constexpr Real const_pi()``: Returns the mathematical constant pi.
- ``template <class Real> constexpr Real const_e()``: Returns the mathematical constant e.

**Basic Operations**:

- ``template <class Real> Real fabs(const Real a)``: Returns the absolute value of the input.
- ``template <class Real> Real trunc(const Real a)``: Truncates the input real number to the nearest integer towards zero.
- ``template <class Real> Real round(const Real a)``: Rounds the input real number to the nearest integer.
- ``template <class Real> Real floor(const Real a)``: Rounds the input real number down to the nearest integer.
- ``template <class Real> Real ceil(const Real a)``: Rounds the input real number up to the nearest integer.

**Trigonometric Functions**:

- ``template <class Real> Real sin(const Real a)``: Computes the sine of the input angle.
- ``template <class Real> Real cos(const Real a)``: Computes the cosine of the input angle.
- ``template <class Real> Real tan(const Real a)``: Computes the tangent of the input angle.
- ``template <class Real> Real asin(const Real a)``: Computes the arcsine of the input value.
- ``template <class Real> Real acos(const Real a)``: Computes the arccosine of the input value.
- ``template <class Real> Real atan(const Real a)``: Computes the arctangent of the input value.
- ``template <class Real> Real atan2(const Real a, const Real b)``: Computes the arctangent of the ratio of two input values.

**Exponential Functions**:

- ``template <class Real> Real exp(const Real a)``: Computes the exponential function of the input value.
- ``template <class Real> Real log(const Real a)``: Computes the natural logarithm of the input value.
- ``template <class Real> Real log2(const Real a)``: Computes the base-2 logarithm of the input value.

**Power Functions**:

- ``template <class Real> Real sqrt(const Real a)``: Computes the square root of the input real number.
- ``template <class Real, class ExpType> Real pow(const Real b, const ExpType e)``: Computes the power of a base raised to an exponent.
- ``template <Long e, class ValueType> constexpr ValueType pow(ValueType b)``: Computes the power of a base raised to a compile-time constant exponent.

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/math_utils.hpp
   :language: c++
