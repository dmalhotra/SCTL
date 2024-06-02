.. _tutorial-interp-quadrule:

Using the InterpQuadRule Class
==============================

The `InterpQuadRule` class is designed to build generalized Chebyshev quadrature rules.
This tutorial will guide you through the basic usage of the `InterpQuadRule` class.
For more advanced usage and additional features, please refer to the class API in :ref:`quadrule.hpp <quadrule_hpp>`.

The following example demonstrates how to use the `InterpQuadRule` class to build a quadrature rule for a set of integrand functions.

1. **Define the Integrand Functions**

    The integrand functions can be defined within a lambda function. In this example, the integrands include a polynomial part \( p(x) \) and a logarithmic part \( q(x) \log(x) \):

    .. code-block:: c++

            Integer order = 16;
            auto integrands = [order](const Vector<double>& nds) {
                const Long N = nds.Dim();
                Matrix<double> M(N, order);
                for (Long j = 0; j < N; j++) {
                    for (Long i = 0; i < order/2; i++) { // p(x)
                        M[j][i] = pow<double>(nds[j], i);
                    }
                    for (Long i = order/2; i < order; i++) { // q(x) log(x)
                        M[j][i] = pow<double>(nds[j], i-order/2) * log<double>(nds[j]);
                    }
                }
                return M;
            };

2. **Build the Quadrature Rule**

    Use the `Build` method of the `InterpQuadRule` class to compute the quadrature nodes and weights.
    The parameters include the lambda function for integrands, the interval `[0.0, 1.0]`, and other optional parameters for accuracy and order:

    .. code-block:: c++

            Vector<double> nds, wts;
            InterpQuadRule::Build(nds, wts, integrands, 0.0, 1.0, 1e-16, 0, 1e-4, 1, false);

**Complete Example:**

Below is the complete example, combining all steps:

.. code-block:: c++

    void build_log_singular_quadrature(Vector<double>& nds, Vector<double>& wts, const Integer order) {
        auto integrands = [order](const Vector<double>& nds) { // p(x) + q(x) log(x)
            const Long N = nds.Dim();
            Matrix<double> M(N, order);
            for (Long j = 0; j < N; j++) {
                for (Long i = 0; i < order/2; i++) { // p(x)
                    M[j][i] = pow<double>(nds[j],i);
                }
                for (Long i = order/2; i < order; i++) { // q(x) log(x)
                    M[j][i] = pow<double>(nds[j],i-order/2) * log<double>(nds[j]);
                }
            }
            return M;
        };

        InterpQuadRule::Build(nds, wts, integrands, 0.0, 1.0, 1e-16, 0, 1e-4, 1, false);
    }

