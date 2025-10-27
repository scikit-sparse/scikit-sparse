.. Copyright (C) 2025, Bernard Roesler. All rights reserved.
   Part of the scikit-sparse project.
   See pyproject.toml for full author list and LICENSE.txt for license details.
   SPDX-License-Identifier: BSD-2-Clause

==========================================================================================
Constrained Column Approximate Minimum Degree (CCOLAMD) Ordering (:mod:`sksparse.ccolamd`)
==========================================================================================

.. currentmodule:: sksparse.ccolamd


The :mod:`sksparse.ccolamd` module provides efficient an implementation of the
`Column Approximate Minimum Degree (CCOLAMD) <colamd_paper_>`_ ordering
algorithm for sparse matrices.

It exposes the main functions of the `CCOLAMD package <ccolamd_github_>`_,
which computes a column ordering :math:`Q` of a sparse matrix that minimizes
the fill-in of the Cholesky decomposition of :math:`(AQ)^{\top}(AQ)`. The
:func:`.ccolamd` function is appropriate for use with non-symmetric and
non-square matrices, for LU factorization, QR factorization, and other
decompositions that require a column ordering.

This module also provides a symmetric variant, :func:`.csymamd`, which computes
a permutation `P` of a symmetric matrix `A` such that the Cholesky
factorization of :math:`PAP^{\top}` has less fill-in and requires fewer
floating point operations than `A`. This function assumes that its input is
symmetric.

The :func:`.ccolamd` and :func:`.csymamd` functions accept both real and
complex matrices, in any format supported by :mod:`scipy.sparse` (CSC format is
most efficient).

These function are identical to that of the `COLAMD package <colamd_github_>`_
(:mod:`sksparse.colamd`), except that they allow the user to specify a set of
constraints on the ordering of the matrix.

.. _colamd_paper: https://dl.acm.org/doi/abs/10.1145/1024074.1024079
.. _colamd_github: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/COLAMD
.. _ccolamd_github: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CCOLAMD


Quickstart
----------

If :math:`A` is a sparse matrix, then the
following code computes the CCOLAMD ordering of :math:`A`:

.. code:: python

    from sksparse.ccolamd import ccolamd
    A = ...  # some sparse matrix
    # Set some constraints, e.g., to fix the first K rows and columns
    N = A.shape[0]
    K = N // 2                     # number of constrained variables
    C = np.full(N, K)              # none are constrained (all == K)
    C[:K] = np.arange(K)           # first K variables are constrained
    q = ccolamd(A, constraints=C)
    AQ = A[:, q]                   # permute the columns of A

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


:class:`CCOLAMDStats` Objects
-----------------------------

An :class:`CCOLAMDStats` object is a dataclass returned by the :func:`ccolamd`
function when the ``return_info`` parameter is set to ``True``. It contains
information about the ordering, including the return status.

Typically, the :class:`CCOLAMDStats` object is unnecessary, and you can
just use the permutation vector returned by :func:`ccolamd`.


Convenience Methods
-------------------

The CCOLAMD package also provides a convenience function,
:func:`ccolamd_get_defaults` to get the default control parameters from the
CCOLAMD package. Most users will not need to use this function, as the default
control parameters are used automatically by :func:`ccolamd`.


Error Handling
--------------

Errors raised by the CCOLAMD package are converted into Python exceptions. See
the :ref:`ccolamd-exceptions` for details.
