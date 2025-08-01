.. Copyright (C) 2025, Bernard Roesler. All rights reserved.
   Part of the scikit-sparse project.
   See pyproject.toml for full author list and LICENSE.txt for license details.
   SPDX-License-Identifier: BSD-2-Clause

Column Approximate Minimum Degree (COLAMD) Ordering
===================================================

.. module:: sksparse.colamd
   :synopsis: Column Approximate Minimum Degree (COLAMD) Ordering

.. versionadded:: 0.5.0

The :mod:`sksparse.colamd` module provides efficient an implementation of the
`Column Approximate Minimum Degree (COLAMD)
<https://dl.acm.org/doi/abs/10.1145/1024074.1024079>`_
ordering algorithm for sparse matrices.

It exposes the main functions of the `COLAMD package
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/COLAMD>`_, which
computes a column ordering :math:`Q` of a sparse matrix that minimizes the
fill-in of the Cholesky decomposition of :math:`(AQ)^{\top}(AQ)`. The
:func:`.colamd` function is appropriate for use with non-symmetric and
non-square matrices, for LU factorization, QR factorization, and other
decompositions that require a column ordering.

This module also provides a symmetric variant, :func:`.symamd`, which computes a
permutation `P` of a symmetric matrix `A` such that the Cholesky factorization
of :math:`PAP^{\\top}` has less fill-in and requires fewer floating point
operations than `A`. This function assumes that its input is symmetric.

The :func:`.colamd` and :func:`.symamd` functions accept both real and complex
matrices, in any format supported by :mod:`scipy.sparse` (CSC format is most
efficient).


Quickstart
----------

If :math:`A` is a sparse matrix, then the
following code computes the COLAMD ordering of :math:`A`:

.. code:: python

    from sksparse.colamd import colamd
    A = ...  # some sparse matrix
    q = colamd(A)
    AQ = A[:, q]  # permute the columns of A

to give the permuted matrix :math:`AQ`, where :math:`Q` is the permutation
matrix corresponding to the ordering :math:`q`.

We can then continue from above to compute the LU decompositions of the
original and permuted matrix, and compare the number of non-zeros in each:

.. code:: python

    from scipy.sparse.linalg import splu
    lu = splu(A)
    luq = splu(A[:, q])
    L, U = lu.L, lu.U
    Lq, Uq = luq.L, luq.U
    print("Number of non-zeros in L + U:  ", (L + U).nnz)
    print("Number of non-zeros in Lq + Uq:", (Lq + Uq).nnz)

The number of non-zeros in the LU factorization of the permuted matrix
should be less than or equal to the number of non-zeros in the LU
factorization of the original matrix, but this is not guaranteed.


Top-level Functions
-------------------

The main functions this module provides are :func:`colamd` and :func:`symamd`.

.. autofunction:: colamd

.. autofunction:: symamd


:class:`COLAMDStats` Objects
----------------------------

An :class:`COLAMDStats` object is a dataclass returned by the :func:`colamd`
function when the ``return_info`` parameter is set to ``True``. It contains
information about the ordering, including the return status.

.. autoclass:: COLAMDStats

Typically, the :class:`COLAMDStats` object is unnecessary, and you can
just use the permutation vector returned by :func:`colamd`.


.. Convenience Methods
.. -------------------

.. The COLAMD package also provides a convenience function to get the default
.. control parameters from the COLAMD package:

.. .. autofunction:: amd_default_control


Error Handling
--------------

Errors raised by the COLAMD package are converted into Python exceptions. The
following exceptions are available:

.. autoclass:: COLAMDError
   :show-inheritance:

.. autoclass:: COLAMDValueError
   :show-inheritance:

.. autoclass:: COLAMDMemoryError
   :show-inheritance:

.. autoclass:: COLAMDInternalError
   :show-inheritance:


Example
-------

This figure shows the effect of COLAMD ordering that reduces the fill-in of the
Cholesky factorization of a sparse matrix.

.. figure:: examples/colamd_example.svg
   :alt: COLAMD Example
   :align: center
   :width: 90%

   The number of non-zeros in the LU factorization of the original matrix
   (left) and the permuted matrix (right) using COLAMD ordering.

The source code for this example is:

.. literalinclude:: examples/colamd_example.py
   :language: python
