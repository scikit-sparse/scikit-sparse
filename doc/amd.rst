Approximate Minimum Degree (AMD) Ordering (:mod:`sksparse.amd`)
=======================================================

.. module:: sksparse.amd
   :synopsis: Approximate Minimum Degree (AMD) Ordering

.. versionadded:: 0.5

Overview
--------

This module provides efficient implementations of the `Approximate Minimum
Degree (AMD)`_ ordering algorithm for sparse, square matrices.

It exposes the main function of the `AMD package`_, which computes a symmetric
ordering of a sparse matrix that minimizes the fill-in of the Cholesky
decomposition. The AMD function accepts both real and complex matrices, in any
format supported by :mod:`scipy.sparse` (CSC format is most
efficient).

Quickstart
----------

If :math:`A` is a sparse, square matrix, then the
following code computes the AMD ordering of :math:`A`:

.. code-block:: python
    from sksparse.amd import amd
    p = amd(A)
    PAPT = A[p][:, p]

to give the permuted matrix :math:`PAP^T`, where :math:`P` is the permutation
matrix corresponding to the ordering :math:`p`. If :math:`A` is not symmetric,
then this is the same as AMD computes the ordering of the symbolically
symmetric matrix :math:`A + A^T`.

We can then compute the Cholesky decompositions of the original and permuted
matrix, and compare the number of non-zeros in each:

.. code-block:: python
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

The main function this module provides is :func:`amd`.

.. autofunction:: amd(A, dense_thresh=None, aggressive=None, return_info=False)


:class:`AMDInfo` Objects
------------------------

An :class:`AMDInfo` object is a dataclass returned by the :func:`amd` function
when the `return_info` parameter is set to `True`. It contains information
about the AMD ordering, including the return status, the number of non-zeros in
the Cholesky factorization, and others.

.. autoclass:: AMDInfo

We can use :class:`AMDInfo` objects to compare the number of non-zeros in the
Cholesky factorization of the original matrix, without computing it directly:

.. code-block:: python
    from sksparse.amd import amd
    from sksparse.cholmod import amd
    A = ...  # some sparse matrix
    N = A.shape[0]
    p, info = amd(A, return_info=True)
    PAPT_factor = cholesky(A[p][:, p])
    print("nnz in L of A:   ", info.Lnz + N)
    print("nnz in L of PAPT:", PAPT_factor.L().nnz)

Typically, however, the :class:`AMDInfo` object is unnecessary, and you can
just use the permutation vector returned by :func:`amd`.


Convenience Methods
-------------------

The AMD package also provides a convenience function to print the default
control parameters in the AMD package:

.. autofunction:: print_amd_default_control()


Error Handling
--------------

Errors raised by the AMD package are converted into Python exceptions. The
following exceptions are available:

.. class:: AMDError

    A base class for all exceptions raised by the AMD package.

.. class:: AMDMemoryError
   
    Raised when the AMD package runs out of memory during the ordering process.

.. class:: AMDInvalidMatrixError

    Raised when the input matrix is not valid for the AMD ordering algorithm.
    This error is only raised if the internal format of the input matrix is
    corrupted in some way. The Python wrapper that :mod:`sksparse.amd` provides
    converts its input to :func:`scipy.sparse.csc_array`, so this error should
    not occur in practice.


References
----------
.. _AMD package: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/AMD
    `AMD ordering algorithm`_: Timothy A. Davis. *AMD: Approximate Minimum Degree Ordering*. SuiteSparse project.
.. _Approximate Minimum Degree (AMD): https://people.engr.tamu.edu/davis/publications_files/An_Approximate_Minimum_Degree_Ordering_Algorithm.pdf
    Amestoy, P. R., Davis, T. A., & Duff, I. S. (1996). *An approximate minimum
    degree ordering algorithm*. SIAM Journal on Matrix Analysis and
    Applications, 17(4), 886-905.
