# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: umfpack.pyx
#  Created: 2025-10-16 11:35
# =============================================================================

"""
===================================================================
Unsymmetric Multifrontal LU Decomposition (:mod:`sksparse.umfpack`)
===================================================================

.. currentmodule:: sksparse.umfpack

.. versionadded:: 0.5.0


An interface to the SuiteSparse `UMFPACK
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/UMFPACK>`_
package, which computes the LU factorization and solves systems of equations
for sparse, possibly non-symmetric, indefinite matrices.
"""

import numpy as np
cimport numpy as np

from .utils import validate_csc_input


cdef bint _is_real_dtype(np.dtype dtype):
    if np.issubdtype(dtype, np.float64):
        return True
    elif np.issubdtype(dtype, np.complex128):
        return False
    else:
        raise TypeError(f"dtype must be float64 or complex128. Got {dtype=}")


cdef class UMFFactor:
    """The main object used for creating and using an LU factorization.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U = P R A Q.

    The numeric factorization is not computed until :meth:`.numeric` is called.
    """

    cdef void *_symbolic
    cdef double _control[UMFPACK_CONTROL]
    cdef double _info[UMFPACK_INFO]
    cdef bint _use_int32
    cdef bint _is_real

    # TODO set up control array
    def __cinit__(self, object A):
        A, use_int32, _ = validate_csc_input(A)

        self._use_int32 = use_int32

        # Compute the symbolic analysis
        cdef int M = A.shape[0]
        cdef int N = A.shape[1]

        cdef np.ndarray indptr = A.indptr
        cdef np.ndarray indices = A.indices
        cdef np.ndarray real_data = A.data.real
        cdef np.ndarray imag_data = A.data.imag

        self._is_real = _is_real_dtype(A.data.dtype)

        cdef int status

        if self._is_real:
            if self._use_int32:
                umfpack_di_defaults(self._control)
                status = umfpack_di_symbolic(
                    M,
                    N,
                    <const int32_t*>indptr.data,
                    <const int32_t*>indices.data,
                    <const double*>real_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )
            else:
                umfpack_dl_defaults(self._control)
                status = umfpack_dl_symbolic(
                    M,
                    N,
                    <const int64_t*>indptr.data,
                    <const int64_t*>indices.data,
                    <const double*>real_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )
        else:
            if self._use_int32:
                umfpack_zi_defaults(self._control)
                status = umfpack_zi_symbolic(
                    M,
                    N,
                    <const int32_t*>indptr.data,
                    <const int32_t*>indices.data,
                    <const double*>real_data.data,
                    <const double*>imag_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )
            else:
                umfpack_zl_defaults(self._control)
                status = umfpack_zl_symbolic(
                    M,
                    N,
                    <const int64_t*>indptr.data,
                    <const int64_t*>indices.data,
                    <const double*>real_data.data,
                    <const double*>imag_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )

        # TODO _handle_errors(status)
        if status != UMFPACK_OK:
            raise RuntimeError(f"UMFPACK symbolic factorization failed with code {status}.")

    def __dealloc__(self):
        if self._symbolic is not NULL:
            if self._is_real:
                if self._use_int32:
                    umfpack_di_free_symbolic(&self._symbolic)
                else:
                    umfpack_dl_free_symbolic(&self._symbolic)
            else:
                if self._use_int32:
                    umfpack_zi_free_symbolic(&self._symbolic)
                else:
                    umfpack_zl_free_symbolic(&self._symbolic)

    # TODO __repr__ and __str__

    # -------------------------------------------------------------------------
    #         Public API
    # -------------------------------------------------------------------------
    # TODO make a python dataclass for control parameters
    def report_control(self, print_level=2):
        """Print a report of the control structure to stdout.

        Parameters
        ----------
        print_level : int, optional
            The verbosity level. Default value is 2.

            Accepted values are:

            * None: use current print level
            * <= 1: no printing
            * 2: print all of control parameters

        """
        pl = print_level if print_level is not None else self._control[UMFPACK_PRL]
        self._control[UMFPACK_PRL] = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_control(self._control)
            else:
                umfpack_dl_report_control(self._control)
        else:
            if self._use_int32:
                umfpack_zi_report_control(self._control)
            else:
                umfpack_zl_report_control(self._control)

        # restore old print level
        if print_level is None:
            self._control[UMFPACK_PRL] = pl

    def report_symbolic(self, print_level=4):
        """Print a report of the symbolic factorization to stdout.

        Parameters
        ----------
        print_level : int, optional
            The verbosity level. Default value is 4.

            Accepted values are:

            * None: use current print level
            * <= 2: no printing
            * 3: fully check input, and print a short summary of its status
            * 4: as 3, but print first few entries of the input
            * 5: as 3, but print all of the input

        """
        pl = print_level if print_level is not None else self._control[UMFPACK_PRL]
        self._control[UMFPACK_PRL] = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_symbolic(self._symbolic, self._control)
            else:
                umfpack_dl_report_symbolic(self._symbolic, self._control)
        else:
            if self._use_int32:
                umfpack_zi_report_symbolic(self._symbolic, self._control)
            else:
                umfpack_zl_report_symbolic(self._symbolic, self._control)

        # restore old print level
        if print_level is None:
            self._control[UMFPACK_PRL] = pl

# =============================================================================
# =============================================================================
