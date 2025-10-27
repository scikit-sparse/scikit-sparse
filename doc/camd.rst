.. Copyright (C) 2025, Bernard Roesler. All rights reserved.
   Part of the scikit-sparse project.
   See pyproject.toml for full author list and LICENSE.txt for license details.
   SPDX-License-Identifier: BSD-2-Clause

Constrained Approximate Minimum Degree (CAMD) Ordering (:mod:`sksparse.camd`)
=============================================================================

.. module:: sksparse.camd
   :synopsis: Constrained Approximate Minimum Degree (CAMD) Ordering

.. versionadded:: 0.5.0

Overview
--------

This module provides efficient implementations of the `Approximate Minimum
Degree (AMD) <https://epubs.siam.org/doi/abs/10.1137/S0895479894278952>`_
ordering algorithm for sparse, square matrices.

It exposes the main function of the `CAMD package
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CAMD>`_, which
computes a symmetric ordering of a sparse matrix that minimizes the fill-in of
the Cholesky decomposition. The CAMD function accepts both real and complex
matrices, in any format supported by :mod:`scipy.sparse` (CSC format is most
efficient).

This function is identical to that of the `AMD package
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/AMD>`_
(:mod:`sksparse.amd`), except that it allows the user to specify a set of
constraints on the ordering of the matrix.

Quickstart
----------

If :math:`A` is a sparse, square matrix, then the
following code computes the CAMD ordering of :math:`A`:

.. code:: python

    from sksparse.camd import camd
    A = ...  # some sparse matrix
    # Set some constraints, e.g., to fix the first two rows and columns
    C = np.ones(A.shape[0], dtype=int)
    C[:2] = 0
    p = camd(A, constraints=C)
    PAPT = A[p][:, p]

to give the permuted matrix :math:`PAP^T`, where :math:`P` is the permutation
matrix corresponding to the ordering :math:`p`. If :math:`A` is not symmetric,
then this is the same as CAMD computes the ordering of the symbolically
symmetric matrix :math:`A + A^T`.

We can then continue from above to compute the Cholesky decompositions of the
original and permuted matrix, and compare the number of non-zeros in each:

.. code:: python

    from sksparse.cholmod import cholesky
    A_factor = cholesky(A)
    PAPT_factor = cholesky(PAPT)
    L = A_factor.L()
    Lp = PAPT_factor.L()
    print("Number of non-zeros in L: ", L.nnz)
    print("Number of non-zeros in Lp:", Lp.nnz)

The number of non-zeros in the Cholesky factorization of the permuted matrix
should be less than or equal to the number of non-zeros in the Cholesky
factorization of the original matrix, but this is not guaranteed.


Top-level Function
-------------------

The main function this module provides is :func:`camd`.

.. autofunction:: camd


:class:`CAMDInfo` Objects
-------------------------

An :class:`CAMDInfo` object is a dataclass returned by the :func:`camd` function
when the `return_info` parameter is set to `True`. It contains information
about the CAMD ordering, including the return status, the number of non-zeros in
the Cholesky factorization, and others.

.. autoclass:: CAMDInfo

We can use :class:`CAMDInfo` objects to compare the number of non-zeros in the
Cholesky factorization of the original matrix, without computing it directly:

.. code:: python

    from sksparse.camd import camd
    from sksparse.cholmod import camd
    A = ...  # some sparse matrix
    N = A.shape[0]
    p, info = camd(A, return_info=True)
    PAPT_factor = cholesky(A[p][:, p])
    print("nnz in L of A:   ", info.Lnz + N)
    print("nnz in L of PAPT:", PAPT_factor.L().nnz)

Typically, however, the :class:`CAMDInfo` object is unnecessary, and you can
just use the permutation vector returned by :func:`camd`.


Convenience Methods
-------------------

The CAMD package also provides a convenience function to get the default
control parameters from the CAMD package:

.. autofunction:: camd_default_control


Error Handling
--------------

Errors raised by the CAMD package are converted into Python exceptions. The
following exceptions are available:

.. class:: CAMDError

    A base class for all exceptions raised by the CAMD package.

.. class:: CAMDMemoryError

    Raised when the CAMD package runs out of memory during the ordering process.

.. class:: CAMDInvalidMatrixError

    Raised when the input matrix is not valid for the CAMD ordering algorithm.
    This error is only raised if the internal format of the input matrix is
    corrupted in some way. The Python wrapper that :mod:`sksparse.camd` provides
    converts its input to :class:`scipy.sparse.csc_array`, so this error should
    not occur in practice.


References
----------
* Amestoy, P. R., Davis, T. A., & Duff, I. S. (1996). *An approximate minimum
  degree ordering algorithm*. SIAM Journal on Matrix Analysis and Applications,
  17(4), 886-905. <https://epubs.siam.org/doi/abs/10.1137/S0895479894278952>.
