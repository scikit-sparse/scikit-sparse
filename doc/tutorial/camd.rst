.. Copyright (C) 2025, Bernard Roesler. All rights reserved.
   Part of the scikit-sparse project.
   See pyproject.toml for full author list and LICENSE.txt for license details.
   SPDX-License-Identifier: BSD-2-Clause

=============================================================================
Constrained Approximate Minimum Degree (CAMD) Ordering (:mod:`sksparse.camd`)
=============================================================================

.. currentmodule:: sksparse.camd


The :mod:`sksparse.camd` module provides an interface to the `Approximate
Minimum Degree (AMD) <amd_paper_>`_ ordering algorithm for sparse, square
matrices.

It exposes the main function of the `CAMD package
<camd_github_>`_, which
computes a symmetric ordering of a sparse matrix that minimizes the fill-in of
the Cholesky decomposition. The CAMD function accepts both real and complex
matrices, in any format supported by :mod:`scipy.sparse` (CSC format is most
efficient).

This function is identical to that of the `AMD package
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/AMD>`_
(:mod:`sksparse.amd`), except that it allows the user to specify a set of
constraints on the ordering of the matrix.

.. _amd_paper: https://epubs.siam.org/doi/abs/10.1137/S0895479894278952
.. _camd_github: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CAMD


Quickstart
----------

If :math:`A` is a sparse, square matrix, then the following code computes the
CAMD ordering of :math:`A`:

.. code:: python

    from sksparse.camd import camd
    A = ...  # some sparse matrix
    # Set some constraints, e.g., to fix the first K rows and columns
    N = A.shape[0]
    K = N // 2                    # number of constrained variables
    C = np.full(N, K)             # none are constrained (all == K)
    C[:K] = np.arange(K)          # first K variables are constrained
    p = camd(A, constraints=C)
    PAPT = A[p][:, p]

to give the permuted matrix :math:`PAP^T`, where :math:`P` is the permutation
matrix corresponding to the ordering :math:`p`. If :math:`A` is not symmetric,
then this is the same as CAMD computes the ordering of the symbolically
symmetric matrix :math:`A + A^T`.

We can then continue from above to compute the Cholesky decompositions of the
original and permuted matrix using :func:`~sksparse.cholmod.cholesky`, and
compare the number of non-zeros in each:

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


Example
-------

To see the effects of CAMD ordering, we can load a sparse matrix from the
`SuiteSparse Matrix Collection <SSMC_>`_ and compute its CAMD ordering.

.. _SSMC: https://sparse.tamu.edu

.. literalinclude:: examples/camd_example.py
   :language: python

The figure shows the effect of CAMD ordering that reduces the fill-in of the
Cholesky factorization of a sparse matrix, while constraining some of the rows.

.. figure:: examples/camd_example.svg
   :alt: CAMD Example
   :align: center
   :width: 90%

   The number of non-zeros in the Cholesky factorization of the original matrix
   (left) and the permuted matrix (right) using CAMD ordering.


:class:`CAMDInfo` Objects
-------------------------

An :class:`CAMDInfo` object is a dataclass returned by the :func:`camd`
function when the ``return_info`` parameter is set to ``True``. It contains
information about the CAMD ordering, including the return status, the number of
non-zeros in the Cholesky factorization, and others.

We can use :class:`CAMDInfo` objects to compare the number of non-zeros in the
Cholesky factorization of the original matrix, without computing it directly:

.. code:: python

    from sksparse.camd import camd
    from sksparse.cholmod import cholesky
    A = ...  # some sparse matrix
    N = A.shape[0]
    p, info = camd(A, return_info=True)
    PAPT_factor = cholesky(A[p][:, p])
    print("nnz in L of A:   ", info.Lnz + N)
    print("nnz in L of PAPT:", PAPT_factor.L().nnz)

Typically, the :class:`CAMDInfo` object is unnecessary, and you can just use
the permutation vector returned by :func:`camd`.


Convenience Methods
-------------------

The CAMD package also provides a convenience function,
:func:`camd_default_control` to get the default control parameters from the
CAMD package. Most users will not need to use this function, as the default
control parameters are used automatically by :func:`camd`.


Error Handling
--------------

Errors raised by the CAMD package are converted into Python exceptions. See
the :ref:`camd-exceptions` for details.

References
----------
* Amestoy, P. R., Davis, T. A., & Duff, I. S. (1996). *An approximate minimum
  degree ordering algorithm*. SIAM Journal on Matrix Analysis and Applications,
  17(4), 886-905. <https://epubs.siam.org/doi/abs/10.1137/S0895479894278952>.
