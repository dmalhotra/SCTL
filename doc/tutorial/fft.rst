.. _tutorial-fft:

Using the FFT Class
===================

The `FFT` class is a wrapper for FFTW, a widely used library for computing Fast Fourier Transforms (FFTs). It provides a straightforward interface to set up and execute FFT transforms. This tutorial will guide you through the basic usage of the `FFT` class. For more detailed information, refer to the API documentation in :ref:`fft_wrapper.hpp <fft_wrapper_hpp>`.

Prerequisites
-------------

1. **Install FFTW**:
   Ensure that you have the FFTW library installed and available on your system. You may need to add additional compiler flags to specify the location of the FFTW library and headers.

2. **Enable FFTW Support in SCTL**:
   Configure SCTL with FFTW support by defining the appropriate flags (e.g., ``SCTL_HAVE_FFTW`` for double-precision).

   .. note:: If FFTW is not configured as above, then SCTL will fallback to direct Fourier transform computation; however, this will be slow.

Basic Usage
-----------

1. **Instantiate the FFT Object**

   Create an instance of the `FFT` class:

   .. code-block:: cpp

      sctl::FFT<double> fft;

2. **Set Up the FFT**

   Configure the FFT object with the desired transformation type, number of transforms, and dimensions of the input data:

   .. code-block:: cpp

      sctl::Vector<sctl::Long> fft_dim;
      fft_dim.PushBack(2);
      fft_dim.PushBack(5);
      fft_dim.PushBack(3);
      sctl::Long howmany = 3;

      fft.Setup(sctl::FFT_Type::R2C, howmany, fft_dim);

   The `FFT` class supports the following types of FFT transformations:

   - Real-to-Complex (R2C)
   - Complex-to-Complex (C2C)
   - Complex-to-Complex Inverse (C2C_INV)
   - Complex-to-Real (C2R)

..

3. **Execute the FFT**

   Prepare the input data and execute the FFT:

   .. code-block:: cpp

      sctl::Vector<double> input(fft.Dim(0)), output;
      for (auto& x : input) x = drand48(); // Example: fill with random data

      fft.Execute(input, output);

4. **Perform Inverse FFT**

   If you need to perform an inverse FFT, set up another FFT object for the inverse transformation:

   .. code-block:: cpp

      sctl::FFT<double> fft_inv;
      fft_inv.Setup(sctl::FFT_Type::C2R, howmany, fft_dim);

      sctl::Vector<double> inverse_output;
      fft_inv.Execute(output, inverse_output);

Complete Example
~~~~~~~~~~~~~~~~

Below is a complete example demonstrating the setup and execution of both forward and inverse FFT transformations:

.. code-block:: cpp

   #include "sctl.hpp"
   using namespace sctl;

   int main() {

       Vector<Long> fft_dim;
       fft_dim.PushBack(2);
       fft_dim.PushBack(5);
       fft_dim.PushBack(3);
       Long howmany = 3;

       FFT<double> fft;
       fft.Setup(FFT_Type::R2C, howmany, fft_dim);

       Vector<double> input(fft.Dim(0)), output;
       for (auto& x : input) x = drand48(); // Example: fill with random data

       fft.Execute(input, output);

       FFT<double> fft_inv;
       fft_inv.Setup(FFT_Type::C2R, howmany, fft_dim);

       Vector<double> inverse_output;
       fft_inv.Execute(output, inverse_output);

       auto inf_norm = [](const Vector<double>& v) {
           double max_val = 0;
           for (const auto& x : v) max_val = std::max<double>(max_val, std::fabs(x));
           return max_val;
       };

       double error = inf_norm(inverse_output - input);
       std::cout << "Error: " << error << std::endl;

       return 0;
   }

