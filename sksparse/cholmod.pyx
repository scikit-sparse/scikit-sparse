# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod.pyx
#  Created: 2025-08-11 14:49
# =============================================================================

"""sksparse.cholmod: Python interface to the CHOLMOD library.

This module provides a Python interface to the CHOLMOD library, which is part
of the SuiteSparse collection by Timothy A. Davis. The main algorithm computes
the Cholesky factorization of a sparse matrix, and solves linear systems.

Interfaces
----------
* `cholesky`: Computes the Cholesky factorization of a sparse matrix.
* `solve`: Solves a linear system using the Cholesky factorization.

This wrapper handles both 32-bit and 64-bit integer types, depending on the
input matrix format.

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse CHOLMOD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD
"""

import numpy as np
cimport numpy as np

from scipy.sparse import csc_array, diags_array
import warnings

from .utils import validate_csc_input


# Define constants for the mode of cholmod_transpose (see cholmod.h)
cdef int CHOLMOD_TRANS_PATTERN = 0    # transpose only the pattern
cdef int CHOLMOD_TRANS_NOCONJ = 1  # numeric (no conjugate)
cdef int CHOLMOD_TRANS_CONJ = 2  # numeric (conjugate transpose)


# -----------------------------------------------------------------------------
#         Define error handling
# -----------------------------------------------------------------------------
class CholmodError(Exception):
    pass


class CholmodNotPositiveDefiniteError(CholmodError):
    pass


class CholmodNotInstalledError(CholmodError):
    pass


class CholmodOutOfMemoryError(CholmodError):
    pass


class CholmodOverflowError(CholmodError):
    pass


class CholmodInvalidInputError(CholmodError):
    pass


class CholmodGpuProblemError(CholmodError):
    pass


class CholmodWarning(Warning):
    pass


class CholmodSmallDiagonalWarning(CholmodWarning):
    pass


cdef _error_handler(int status) except * with gil:
    """Handle CHOLMOD errors by raising Python exceptions or warnings.

    This function should be set as the error handler in the CHOLMOD common
    struct before passing to any CHOLMOD function.

    Parameters
    ----------
    status : int
        The CHOLMOD status code, from the cholmod_common.status field.

    Returns
    -------
    None

    Raises
    ------
    CholmodWarning
        Raises a warning for non-critical issues.
    CholmodError or subclass
        Raises an appropriate Python exception based on the CHOLMOD status code.
    """
    if status == CHOLMOD_OK:
        return

    status_msg = f"(code {status:d})"

    # TODO include informative error messages.
    # Known Errors
    cdef dict error_map = {
        CHOLMOD_NOT_INSTALLED: (
            CholmodNotInstalledError,
            "CHOLMOD library is not installed or not found."
        ),
        CHOLMOD_OUT_OF_MEMORY: (
            CholmodOutOfMemoryError,
            "CHOLMOD ran out of memory."
        ),
        CHOLMOD_TOO_LARGE: (
            CholmodOverflowError,
            "CHOLMOD encountered an integer overflow."
        ),
        CHOLMOD_INVALID: (
            CholmodInvalidInputError,
            "CHOLMOD received invalid input."
        ),
        CHOLMOD_GPU_PROBLEM: (
            CholmodGpuProblemError,
            "CHOLMOD encountered a problem with CUDA."
        ),
        CHOLMOD_NOT_POSDEF: (
            CholmodNotPositiveDefiniteError,
            "Input matrix is not positive definite."
        ),
        CHOLMOD_DSMALL: (
            CholmodSmallDiagonalWarning,
            "A diagonal entry is very small, which may lead to numerical instability."
        ),
    }

    # Fallback to generic error for unknown codes
    exc_class, msg = error_map.get(status, CholmodError)
    full_msg = msg + " " + status_msg

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class)
    else:
        raise exc_class(full_msg)


# -----------------------------------------------------------------------------
#         Data Conversions
# -----------------------------------------------------------------------------
cdef object _cholmod_sparse_from_csc(
    object A_py,
    int stype,
    bint use_int32,
    cholmod_sparse *A_static,
    cholmod_common *cm
):
    """Create a CHOLMOD sparse matrix from a scipy.sparse.csc_array.

    Parameters
    ----------
    A_py : (N, N) csc_array
        The input sparse matrix in Compressed Sparse Column (CSC) format.
    stype : int
        The assumed symmetry type of ``A_py``:
        * -1: lower triangular,
        *  0: unsymmetric (not used here),
        *  1: upper triangular.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for indices and indptr.
    A_static : cholmod_sparse*
        Pointer to a preallocated CHOLMOD sparse matrix structure. Contents
        need not be initialized. Contains the CHOLMOD sparse matrix on output.
    cm : cholmod_common*
        Pointer to a CHOLMOD common structure for configuration and status.

    Returns
    -------
    res : tuple
        A tuple containing a reference to ``A_py`` and the three arrays that
        make it up: ``A.indptr``, ``A.indices``, and ``A.data``. There is no
        use for the output of this function, except to keep the underlying data
        from being garbage collected until the cholmod_sparse object is freed.
    """
    if not isinstance(A_py, csc_array):
        raise ValueError("Input must be a csc_array.")

    cdef supported_dtypes = (
        np.float32,
        np.float64,
        np.complex64,
        np.complex128
    )

    dtype = A_py.dtype

    if dtype not in supported_dtypes:
        raise ValueError(f"Unsupported data type for CHOLMOD: {dtype}")

    # Initialize the CHOLMOD sparse matrix
    cdef cholmod_sparse* A = A_static
    memset(A, 0, sizeof(cholmod_sparse))

    A.nrow = A_py.shape[0]
    A.ncol = A_py.shape[1]
    A.nzmax = A_py.nnz
    A.packed = True
    A.sorted = True
    A.itype = CHOLMOD_INT if use_int32 else CHOLMOD_LONG
    A.stype = -1 if stype < 0 else (0 if stype == 0 else 1)
    A.dtype = (
        CHOLMOD_SINGLE
        if dtype == np.float32 or dtype == np.complex64
        else CHOLMOD_DOUBLE
    )
    A.z = NULL

    # Declare memoryviews for the index and data arrays
    cdef int32_t[::1] Ap_mv_int32, Ai_mv_int32
    cdef int64_t[::1] Ap_mv_int64, Ai_mv_int64

    cdef float32_t[::1] Ax_mv_float32
    cdef float64_t[::1] Ax_mv_float64
    cdef complex64_t[::1] Ax_mv_complex64
    cdef complex128_t[::1] Ax_mv_complex128

    # Declare array references to keep memory alive
    cdef np.ndarray Ap, Ai, Ax

    # Create the index arrays
    itype = np.int32 if use_int32 else np.int64
    Ap = np.ascontiguousarray(A_py.indptr, dtype=itype)
    Ai = np.ascontiguousarray(A_py.indices, dtype=itype)

    if use_int32:
        Ap_mv_int32 = Ap
        Ai_mv_int32 = Ai
        A.p = &Ap_mv_int32[0]
        A.i = &Ai_mv_int32[0]
    else:
        Ap_mv_int64 = Ap
        Ai_mv_int64 = Ai
        A.p = &Ap_mv_int64[0]
        A.i = &Ai_mv_int64[0]

    # Get the numerical values of A
    if dtype == bool:
        A.xtype = CHOLMOD_PATTERN
        A.x = NULL
    else:
        A.xtype = (
            CHOLMOD_COMPLEX
            if np.issubdtype(dtype, np.complexfloating)
            else CHOLMOD_REAL
        )

        Ax = np.ascontiguousarray(A_py.data, dtype=dtype)

        if dtype == np.float32:
            Ax_mv_float32 = Ax
            A.x = &Ax_mv_float32[0]
        elif dtype == np.float64:
            Ax_mv_float64 = Ax
            A.x = &Ax_mv_float64[0]
        elif dtype == np.complex64:
            Ax_mv_complex64 = Ax
            A.x = &Ax_mv_complex64[0]
        elif dtype == np.complex128:
            Ax_mv_complex128 = Ax
            A.x = &Ax_mv_complex128[0]

    return (A_py, Ap, Ai, Ax)


cdef class _CholmodSparseDestructor:
    """A destructor for CHOLMOD sparse matrices.

    This class is used as a base for NumPy arrays that are views on CHOLMOD
    sparse matrices. It ensures that the CHOLMOD sparse matrix is properly
    freed when the NumPy array is no longer in use.

    Attributes
    ----------
    _sparse : cholmod_sparse*
        The CHOLMOD sparse matrix to be freed.
    _common : cholmod_common*
        The CHOLMOD common structure used for memory management.
    """

    cdef cholmod_sparse* _sparse
    cdef cholmod_common* _common

    cdef void init(self, cholmod_sparse* A, cholmod_common* common):
        assert A is not NULL
        assert common is not NULL
        self._sparse = A
        self._common = common

    def __dealloc__(self):
        if self._sparse.itype == CHOLMOD_INT:
            cholmod_free_sparse(&self._sparse, self._common)
        else:
            cholmod_l_free_sparse(&self._sparse, self._common)


# dict[xtype, dtype] -> numpy typenum
cdef dict _np_dtype_from_cholmod = {
    (CHOLMOD_REAL, CHOLMOD_SINGLE): np.NPY_FLOAT32,
    (CHOLMOD_REAL, CHOLMOD_DOUBLE): np.NPY_FLOAT64,
    (CHOLMOD_COMPLEX, CHOLMOD_SINGLE): np.NPY_COMPLEX64,
    (CHOLMOD_COMPLEX, CHOLMOD_DOUBLE): np.NPY_COMPLEX128,
    (CHOLMOD_PATTERN, CHOLMOD_SINGLE): np.NPY_BOOL,
    (CHOLMOD_PATTERN, CHOLMOD_DOUBLE): np.NPY_BOOL,
}


cdef object _csc_from_cholmod_sparse(cholmod_sparse* A, cholmod_common* common):
    """Build a scipy.sparse.csc_array that's a view onto A, with a 'base' with
    appropriate destructor. 'A' must have been allocated by cholmod."""

    # This is a little tricky: We build 3 arrays, views on each part of the
    # cholmod_dense object. They all have the same _CholmodSparseDestructor
    # object as base. So none of them will be deallocated until they have all
    # become unused. Then those are built into a csc_array.

    # init destructor for cholmod data
    cdef _CholmodSparseDestructor base = _CholmodSparseDestructor()
    base.init(A, common)

    cdef int np_itypenum = np.NPY_INT32 if A.itype == CHOLMOD_INT else np.NPY_INT64
    cdef int np_dtypenum = _np_dtype_from_cholmod.get((A.xtype, A.dtype), np.NPY_OBJECT)

    # convert to NumPy arrays
    cdef np.ndarray indptr = np.PyArray_SimpleNewFromData(
        1, [A.ncol + 1], np_itypenum, A.p
    )
    cdef np.ndarray indices = np.PyArray_SimpleNewFromData(
        1, [A.nzmax], np_itypenum, A.i
    )
    cdef np.ndarray data = np.PyArray_SimpleNewFromData(
        1, [A.nzmax], np_dtypenum, A.x
    )

    # set destructor and check if writeable
    for array in (indptr, indices, data):
        np.set_array_base(array, base)
        assert np.PyArray_ISWRITEABLE(array)

    return csc_array((data, indices, indptr), shape=(A.nrow, A.ncol))


cdef np.ndarray _array_from_cholmod_permutation(
    cholmod_factor* L, size_t N, bint use_int32
):
    """Create a NumPy array from the permutation vector in a CHOLMOD factor.

    Parameters
    ----------
    L : cholmod_factor*
        The CHOLMOD factor containing the permutation vector.
    N : size_t
        The size of the permutation vector.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for the permutation indices.

    Returns
    -------
    p : ndarray
        The permutation vector as a NumPy array.
    """
    if L is NULL or L.minor != N:
        raise ValueError("CHOLMOD factorization failed, cannot get permutation.")

    cdef int np_itypenum = np.NPY_INT32 if use_int32 else np.NPY_INT64
    cdef void* data_ptr = L.Perm
    cdef np.ndarray p = np.PyArray_SimpleNewFromData(1, [N], np_itypenum, data_ptr)
    # TODO Instead of returning a copy here, create a view and
    # a _CholmodFactorDestructor object to retain a reference?
    # Return a copy in case L is freed
    return p.copy()


cdef dict _ordering_methods = {
    "default": None,
    "best": None,
    "natural": CHOLMOD_NATURAL,
    "amd": CHOLMOD_AMD,
    "metis": CHOLMOD_METIS,
    "nesdis": CHOLMOD_NESDIS,
    "colamd": CHOLMOD_COLAMD,
    "postordered": CHOLMOD_POSTORDERED,
}


def _cholesky_base(
    A, *, ldl=False, beta=None, lower=False, order=None, remove_zeros=True
):
    """Base function for Cholesky factorization."""
    A, use_int32, out_itype = validate_csc_input(A, require_square=True)

    N = A.shape[0]

    # Check the input ordering method
    if order is not None and order not in _ordering_methods:
        raise ValueError(f"Unknown ordering method: {order}")

    # Empty matrix
    # R = chol2(sparse(0, 0)) -> R: (0, 0) nnz = 0
    # [R, p, q] = chol2(sparse(0, 0)) -> R: (0, 0) nnz = 0, p: 0, q: []
    if N == 0:
        R = csc_array((0, 0), dtype=A.dtype)
        D = diags_array((0,), shape=(0, 0), dtype=A.dtype)
        p = np.array([], dtype=out_itype)
        if ldl:
            return (R, D) if order is None else (R, D, p)
        else:
            return R if order is None else (R, p)

    # Matrix of all zeros
    # R = chol2(sparse(N, N)) -> error not pos def
    # [R, p, q] = chol2(sparse(N, N)) -> R: (0, 10) nnz = 0, p: 1, q: [1:N]
    if A.nnz == 0:
        raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

    # -------------------------------------------------------------------------
    #         Set up Data Structures
    # -------------------------------------------------------------------------
    # Create the CHOLMOD common object
    cdef cholmod_common cm

    if use_int32:
        cholmod_start(&cm)
    else:
        cholmod_l_start(&cm)

    # Convert to packed LL.T when done
    cm.final_asis = False
    cm.final_super = False
    cm.final_ll = not ldl  # LL.T for Cholesky, LDL.T for LDL
    cm.final_pack = True
    cm.final_monotonic = True

    # TODO test if this is needed when we implement chol_update (see ldlchol.c)
    # If we do *not* drop numerically zero entries from the symbolic pattern,
    # we *do* need to drop entries that result from supernodal amalgamation.
    # Otherwise, all zeros are dropped in cholmod_drop, so save the extra step.
    cm.final_resymbol = not remove_zeros

    cm.quick_return_if_not_posdef = True

    if order == "default":
        cm.nmethods = 0
    elif order == "best":
        cm.nmethods = CHOLMOD_MAXMETHODS
    else:
        # CHOLMOD_POSTORDERED is not an input, but an output flag. We treat it
        # as "natural" + postordering, per cholmod.h description.
        ordering = "natural" if order is None or order == "postordered" else order
        cm.nmethods = 1
        cm.method[0].ordering = _ordering_methods[ordering]
        cm.postorder = (order == "postordered" or ordering != "natural")

    # Get the input matrix into CHOLMOD format
    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse *Ac = &Amatrix

    stype = -1 if lower else 1  # use lower or upper triangular part
    # Keep a reference to the input matrix to keep it alive
    cdef object ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix, &cm)

    # Set stype and beta for LDL
    cdef double betac[2]

    if ldl:
        if beta is None:
            Amatrix.stype = -1  # lower triangular
            betac[0] = 0.0
            betac[1] = 0.0
        else:
            if not np.isscalar(beta):
                raise ValueError("beta must be a scalar value.")
            Amatrix.stype = 0  # symmetric, not triangular
            betac[0] = beta
            betac[1] = 0.0

    # -------------------------------------------------------------------------
    #         Analyze and Factorize
    # -------------------------------------------------------------------------
    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_analyze(Ac, &cm)

        if ldl:
            cholmod_factorize_p(Ac, betac, NULL, 0, Lc, &cm)
        else:
            cholmod_factorize(Ac, Lc, &cm)
    else:
        Lc = cholmod_l_analyze(Ac, &cm)

        if ldl:
            cholmod_l_factorize_p(Ac, betac, NULL, 0, Lc, &cm)
        else:
            cholmod_l_factorize(Ac, Lc, &cm)

    # Check for errors
    _error_handler(cm.status)

    # -------------------------------------------------------------------------
    #         Convert to scipy csc_array
    # -------------------------------------------------------------------------
    # NOTE there is no need to keep "minor" here, since we just raise an error
    # if the matrix is not positive definite.
    cdef cholmod_sparse* Lsparse
    cdef cholmod_sparse* Rc

    if use_int32:
        Lsparse = cholmod_factor_to_sparse(Lc, &cm)
    else:
        Lsparse = cholmod_l_factor_to_sparse(Lc, &cm)

    if remove_zeros:
        # drop explicit zeros from Lsparse
        if use_int32:
            cholmod_drop(0, Lsparse, &cm)
        else:
            cholmod_l_drop(0, Lsparse, &cm)

    if lower:
        Rc = Lsparse
    else:
        # Convert to upper triangular (conjugate transpose)
        if use_int32:
            Rc = cholmod_transpose(Lsparse, CHOLMOD_TRANS_CONJ, &cm)
            cholmod_free_sparse(&Lsparse, &cm)
        else:
            Rc = cholmod_l_transpose(Lsparse, CHOLMOD_TRANS_CONJ, &cm)
            cholmod_l_free_sparse(&Lsparse, &cm)

    # -------------------------------------------------------------------------
    #         Create outputs
    # -------------------------------------------------------------------------
    R = _csc_from_cholmod_sparse(Rc, &cm)
    p = _array_from_cholmod_permutation(Lc, N, use_int32)

    # For LDL, we need to extract the diagonal matrix D
    if ldl:
        D = diags_array(R.diagonal())
        R.setdiag(1.0)  # set unit diagonal

    # Free everything else
    # NOTE there is no need to free Ac here, since it is just a pointer to the
    # original input matrix A. The MATLAB interface creates a *new*
    # cholmod_sparse object, so it needs to be freed.
    if use_int32:
        cholmod_free_factor(&Lc, &cm)
        cholmod_finish(&cm)
    else:
        cholmod_l_free_factor(&Lc, &cm)
        cholmod_l_finish(&cm)

    if ldl:
        return (R, D) if order is None else (R, D, p)
    else:
        return R if order is None else (R, p)


def cholesky(A, *, lower=False, order=None, remove_zeros=True):
    """Compute the Cholesky factorization of a sparse matrix.

    This function computes the Cholesky factorization of a symmetric positive
    definite matrix `A`:

    .. math::

        P A P^{\\top} = R^{\\top} R,

    where `R` is an upper triangular matrix. Only the upper triangular part of
    `A` is used. If ``lower`` is True, the lower triangular factor `L` is
    returned instead, such that:

    .. math::

        P A P^{\\top} = L L^{\\top}.

    In this case, only the lower triangular part of `A` is used.

    Parameters
    ----------
    A : (N, N) {array_like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format. Must be symmetric positive definite.
    order : None or str in {"default", "best", "natural", "metis", "nesdis", \
            "amd", "colamd", "postordered"}, optional
        The permutation algorithm to use for the factorization. By default, the
        natural ordering of the input matrix is used. The other options are:

        * ``default``: Use the default method, which first tries AMD, then METIS.
        * ``best``: Automatically select the best ordering based on the input.
        * ``metis``: Use the METIS library for graph partitioning.
        * ``nesdis``: Use the NESDIS library for nested dissection.
        * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
        * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm for the
          symmetric case, or the COLAMD algorithm for the unsymmetric case
          (:math:`A A^{\\top}` or :math:`A^{\\top} A`).
        * ``postordered``: Use natural ordering followed by postordering.

        By default, methods other than ``natural`` will also be postordered.

        .. warning::

            The ordering method ``best`` may be quite slow for large matrices,
            but if the factorization is reused many times, it can be worth it.

    lower : bool, optional
        If True, return the lower triangular factor `L` such that
        :math:`A = L L^{\\top}`. Default is False, returning the upper
        triangular factor `R`.
    remove_zeros : bool, optional
        If False, do not remove explicit zeros from the factor ``L`` or ``R``.
        This flag allows use of the ``chol_update`` function afterwards.
        Default is True, so that the output is in canonical form.

    Returns
    -------
    R : csc_array
        The triangular factor of the Cholesky decomposition. The data type will
        match that of ``A``.
    p : ndarray of int, optional
        The permutation vector used in the factorization. Only returned if the
        ordering is not ``None``.

    Raises
    ------
    ValueError
        If the input matrix is not square, or if an unknown ordering method is
        specified.
    CholmodError
        If the factorization fails for any reason, such as the input matrix not
        being positive definite.
    CholmodWarning
        If the input matrix is not positive definite, but the factorization
        succeeds anyway (*e.g.*, due to a small diagonal entry).

    Notes
    -----
    This function is an interface to the CHOLMOD library, which is part of
    the SuiteSparse collection by Timothy A. Davis. For more details, see the
    documentation in the header file [#cholmod_h]_.

    References
    ----------
    .. [#cholmod_h] ``cholmod.h`` - SuiteSparse CHOLMOD header file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h
    """
    return _cholesky_base(
        A, ldl=False, lower=lower, order=order, remove_zeros=remove_zeros
    )


# TODO refactor docstrings
def ldl(A, beta=None, *, lower=True, order=None, remove_zeros=True):
    """Compute the LDL factorization of a sparse matrix.

    This function computes the LDL factorization of a symmetric matrix `A`:

    .. math::

        L D L^{\\top} = P A P^{\\top},

    where `L` is a lower triangular matrix with unit diagonal, and `D` is
    a diagonal matrix. Only the lower triangular part of `A` is used. If
    ``lower`` is False, the upper triangular factor `R` is returned instead,
    such that:

    .. math::

        R^{\\top} D R = P A P^{\\top}.

    In this case, only the upper triangular part of `A` is used.

    If ``beta`` is a scalar value, compute the factorization of:

    .. math::

        L D L^{\\top} = P A A^{\\top} P^{\\top} + \\beta I,

    where `I` is the identity matrix.

    Parameters
    ----------
    A : (N, N) {array_like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format. Must be symmetric, by may be indefinite.
    beta : float, optional
        The scalar value to add to the diagonal of the symmetrized matrix
        :math:`A A^{\\top}` before factorization. Default is None, which
        computes the factorization of :math:`A` itself.
    order : None or str in {"default", "best", "natural", "metis", "nesdis", \
            "amd", "colamd", "postordered"}, optional
        The permutation algorithm to use for the factorization. By default, the
        natural ordering of the input matrix is used. The other options are:

        * ``default``: Use the default method, which first tries AMD, then METIS.
        * ``best``: Automatically select the best ordering based on the input.
        * ``metis``: Use the METIS library for graph partitioning.
        * ``nesdis``: Use the NESDIS library for nested dissection.
        * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
        * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm for the
          symmetric case, or the COLAMD algorithm for the unsymmetric case
          (:math:`A A^{\\top}` or :math:`A^{\\top} A`).
        * ``postordered``: Use natural ordering followed by postordering.

        By default, methods other than ``natural`` will also be postordered.

        .. warning::

            The ordering method ``best`` may be quite slow for large matrices,
            but if the factorization is reused many times, it can be worth it.

    lower : bool, optional
        If True, return the lower triangular factor `L` such that
        :math:`A = L D L^{\\top}`. Default is False, returning the upper
        triangular factor `R`, such that :math:`A = R^{\\top} D R`.
    remove_zeros : bool, optional
        If False, do not remove explicit zeros from the factor ``L`` or ``R``.
        This flag allows use of the ``chol_update`` function afterwards.
        Default is True, so that the output is in canonical form.

    Returns
    -------
    R : csc_array
        The triangular factor of the Cholesky decomposition. The data type will
        match that of ``A``.
    D : dia_array
        The diagonal matrix `D` of the factorization, in sparse DIA format.
        The data type will match that of ``A``.
    p : ndarray of int, optional
        The permutation vector used in the factorization. Only returned if the
        ordering is not ``None``.

    Raises
    ------
    ValueError
        If the input matrix is not square, or if an unknown ordering method is
        specified.
    CholmodError
        If the factorization fails for any reason, such as the input matrix not
        being positive definite.
    CholmodWarning
        If the input matrix is not positive definite, but the factorization
        succeeds anyway (*e.g.*, due to a small diagonal entry).

    Notes
    -----
    This function is an interface to the CHOLMOD library, which is part of
    the SuiteSparse collection by Timothy A. Davis. For more details, see the
    documentation in the header file [#ldl_h]_.

    References
    ----------
    .. [#ldl_h] ``ldl.h`` - SuiteSparse CHOLMOD header file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h
    """
    return _cholesky_base(
        A, ldl=True, beta=beta, lower=lower, order=order, remove_zeros=remove_zeros
    )
