..
   Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _stacktrace_h:

stacktrace.h
============

This header defines a utility for generating stack trace if the program crashes.
It can be enabled by defining ``SCTL_SIG_HANDLER`` in code or using the compiler flag ``-DSCTL_SIG_HANDLER``.

.. .. raw:: html
.. 
..    <div style="border-top: 3px solid"></div>
..    <br>
.. 
.. .. literalinclude:: ../../include/sctl/stacktrace.h
..    :language: c++
.. 
