# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Released under the terms of the GNU GPL v2, or, at your option, any
# later version.

# main todo items:
#     figure out best API for retrieving factor matrices
#     add 'control' handling
#     factor out shared code
#     compile/test/docs

import warnings
cimport stdlib
cimport python as py
import numpy as np
cimport numpy as np
from scipy import sparse

from UserDict import DictMixin

np.import_array()

# UMFPACK actually uses void *'s for these, but hey, let's have a little more
# type safety:
cdef struct struct_symbolic:
    pass
ctypedef struct_symbolic * symbolic_t

cdef struct struct_numeric:
    pass
ctypedef struct_numeric * numeric_t

cdef extern from "suitesparse/umfpack.h":
    char * UMFPACK_DATE
    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO

        UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION

        # Return codes:
        UMFPACK_OK

        UMFPACK_WARNING_singular_matrix, UMFPACK_WARNING_determinant_underflow
        UMFPACK_WARNING_determinant_overflow
        
        UMFPACK_ERROR_out_of_memory
        UMFPACK_ERROR_invalid_Numeric_object
        UMFPACK_ERROR_invalid_Symbolic_object
        UMFPACK_ERROR_argument_missing
        UMFPACK_ERROR_n_nonpositive
        UMFPACK_ERROR_invalid_matrix
        UMFPACK_ERROR_different_pattern
        UMFPACK_ERROR_invalid_system
        UMFPACK_ERROR_invalid_permutation
        UMFPACK_ERROR_internal_error
        UMFPACK_ERROR_file_IO
        
        # Control:
        # Printing routines:
        #UMFPACK_PRL
        # umfpack_*_symbolic:
        UMFPACK_DENSE_ROW
        UMFPACK_DENSE_COL
        UMFPACK_BLOCK_SIZE
        UMFPACK_STRATEGY
        UMFPACK_2BY2_TOLERANCE
        UMFPACK_FIXQ
        UMFPACK_AMD_DENSE
        UMFPACK_AGGRESSIVE
        # umfpack_*_numeric:
        UMFPACK_PIVOT_TOLERANCE
        UMFPACK_ALLOC_INIT
        UMFPACK_SYM_PIVOT_TOLERANCE
        UMFPACK_SCALE
        UMFPACK_FRONT_ALLOC_INIT
        UMFPACK_DROPTOL
        # umfpack_*_solve:
        UMFPACK_IRSTEP
        
        # For UMFPACK_STRATEGY:
        UMFPACK_STRATEGY_AUTO
        UMFPACK_STRATEGY_UNSYMMETRIC
        UMFPACK_STRATEGY_2BY2
        UMFPACK_STRATEGY_SYMMETRIC

        # For UMFPACK_SCALE:
        UMFPACK_SCALE_NONE
        UMFPACK_SCALE_SUM
        UMFPACK_SCALE_MAX

    int umfpack_di_symbolic(int n_row, int n_col,
                            int * Ap, int * Ai, double * Ax,
                            symbolic_t * symbolic,
                            double * control, double * info)
    int umfpack_zi_symbolic(int n_row, int n_col,
                            int * Ap, int * Ai, double * Ax, double * Az,
                            symbolic_t * symbolic,
                            double * control, double * info)

    int umfpack_di_qsymbolic(int n_row, int n_col,
                             int * Ap, int * Ai, double * Ax,
                             int * Qinit,
                             symbolic_t * symbolic,
                             double * control, double * info)
    int umfpack_zi_qsymbolic(int n_row, int n_col,
                             int * Ap, int * Ai, double * Ax, double * Az,
                             int * Qinit,
                             symbolic_t * symbolic,
                             double * control, double * info)

    int umfpack_di_numeric(int * Ap, int * Ai, double * Ax,
                           symbolic_t symbolic,
                           numeric_t * numeric,
                           double * control, double * info)
    int umfpack_zi_numeric(int * Ap, int * Ai, double * Ax, double * Az,
                           symbolic_t symbolic,
                           numeric_t * numeric,
                           double * control, double * info)

    cdef enum:
        UMFPACK_A, UMFPACK_At
        UMFPACK_L, UMFPACK_U
        UMFPACK_Lt, UMFPACK_Ut
        UMFPACK_Pt_L, UMFPACK_U_Qt
        UMFPACK_Lt_P, UMFPACK_Q_Ut
        # "at" = "array transpose"... i.e., simple tranpose, not hermitian
        # tranpose:
        UMFPACK_Aat, UMFPACK_Lat_P, UMFPACK_Lat, UMFPACK_Q_Uat, UMFPACK_Uat
    # See UserGuide.pdf page 68 for details of "wsolve" version (basically it
    # just saves some mallocs, but you have to allocate the right amount of
    # space ahead of time). Note that these only take dense vectors, so we
    # have to do sparse->dense conversion, and for multidimensional arrays we
    # have to work column-at-a-time.
    int umfpack_di_wsolve(int sys, int * Ap, int * Ai, double * Ax,
                          double * X, double * B,
                          numeric_t numeric,
                          double * control, double * info,
                          int * Wi, double * W)
    int umfpack_zi_wsolve(int sys, int * Ap, int * Ai, double * Ax, double * Az,
                          double * Xx, double * Xz,
                          double * Bx, double * Bz,
                          numeric_t numeric,
                          double * control, double * info,
                          int * Wi, double * W)

    void umfpack_di_free_symbolic(symbolic_t * symbolic)
    void umfpack_zi_free_symbolic(symbolic_t * symbolic)

    void umfpack_di_free_numeric(numeric_t * numeric)
    void umfpack_zi_free_numeric(numeric_t * numeric)
    
    void umfpack_di_defaults(double * control)
    void umfpack_zi_defaults(double * control)

    int umfpack_di_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, numeric_t numeric)
    int umfpack_zi_get_lunz(int * lnz, int * unz, int * n_row, int * n_col,
                            int * nz_udiag, numeric_t numeric)
    
    int umfpack_di_get_numeric(int * Lp, int * Lj, double * Lx, double * Lz,
                               int * Up, int * Ui, double * Ux, double * Uz,
                               int * P, int * Q, double * Dx, double * Dz,
                               int * do_recip, double * Rs,
                               numeric_t numeric)
    int umfpack_zi_get_numeric(int * Lp, int * Lj, double * Lx,
                               int * Up, int * Ui, double * Ux,
                               int * P, int * Q, double * Dx,
                               int * do_recip, double * Rs,
                               numeric_t numeric)
    # Use non-NULL Ex to get determinant in "scientific notation" (in which
    # case it may return an underflow/overflow warning, but this is irrelevant
    # because the warning is for the actual value, not the value in scientific
    # notation)
    int umfpack_di_get_determinant(double * Mx, double * Ex,
                                   numeric_t numeric, double * info)
    int umfpack_zi_get_determinant(double * Mx, double * Mz, double * Ex,
                                   numeric_t numeric, double * info)
    

umfpack_version = "%s.%s.%s (%s)" % (UMFPACK_MAIN_VERSION,
                                     UMFPACK_SUB_VERSION,
                                     UMFPACK_SUBSUB_VERSION,
                                     UMFPACK_DATE)

class UmfpackError(Exception):
    pass

class UmfpackWarning(UserWarning):
    pass

class UmfpackTypeConversionWarning(UmfpackWarning,
                                   sparse.SparseEffiencyWarning):
    pass

_errors = {
    UMFPACK_ERROR_out_of_memory: "out of memory",
    UMFPACK_ERROR_invalid_Numeric_object: "invalid Numeric factor object",
    UMFPACK_ERROR_invalid_Symbolic_object: "invalid Symbolic factor object",
    UMFPACK_ERROR_argument_missing: "required argument missing",
    UMFPACK_ERROR_n_nonpositive: "matrix has zero or fewer rows or columns",
    UMFPACK_ERROR_invalid_matrix: "matrix is invalid",
    UMFPACK_ERROR_different_pattern: "pattern of non-zeros has changed",
    UMFPACK_ERROR_invalid_system: "invalid system, or non-square A matrix",
    UMFPACK_ERROR_invalid_permutation: "invalid permutation", # can't happen
    UMFPACK_ERROR_internal_error: "bug detected in UMFPACK! Please report!",
    UMFPACK_ERROR_file_IO: "file IO error", # can't happen
    }

cdef _check(int status):
    if status < 0:
        if status in _errors:
            raise UmfpackError, _errors[status]
        else:
            raise (UmfpackError,
                   "encountered unknown UMFPACK error: %s. Please file a bug!"
                   % status)
    elif status > 0:
        if status == UMFPACK_WARNING_singular_matrix:
            # Returned by umfpack_*_numeric: means that your system is
            # singular. Nice to know, but not necessarily something to spam
            # the console about.
            # Returned by umfpack_*_solve: means your system is singular, also
            # signaled by Nan/Inf's in the resulting matrix, though some parts
            # may be valid. Nice to know, but Nan/Inf is just as good a way to
            # signal this a spamming the console.
            #
            # Well, this is a question, actually.
            #   sing = [[1, 0], [0, 0]]
            #   np.linalg.inv(sing) --> LinAlgError
            #   scipy.linalg.inv(sing) --> LinAlgError
            #   scipy.linalg.lu(sing) --> works fine (which makes sense!)
            #   scipy.linalg.lu_factor(sing) --> throws a warning, otherwise works
            #   scipy.linalg.lu_solve(lu_factor) --> silently returns NaN/Inf
            # Perhaps umfpack_*_solve should throw a LinAlgError on singular
            # matrices; alternatively, maybe it should be dependent on the
            # current setting of the numpy 'divide' error handler. (Not that
            # that seems to affect any of the above functions.)
            #
            # Returned by umfpack_*_determinant: means the determinant is
            # zero, which can also be detected by... looking at the
            # determinant.
            # So never mind:
            pass
        elif (status == UMFPACK_WARNING_determinant_underflow
              or status == UMFPACK_WARNING_determinant_overflow):
            # This doesn't mean anything if we use the 'Ex' version of
            # umfpack_*_determinant, which we do, so no need to worry:
            pass
        else:
            warnings.warn(UmfpackError,
                          "encountered unknown UMFPACK warning: %s."
                          "Please file a bug!" % status)
    return status

cdef object _integer_py_dtype = np.dtype(np.int32)
assert sizeof(int) == _integer_py_dtype.itemsize == 4

# XX FIXME: copy-paste code from cholmod.pyx:
cdef _require_1d_integer(a):
    if a.dtype.itemsize != _integer_py_dtype.itemsize:
        warnings.warn("array contains %s bit integers; "
                      "this will be slower than using %s bit integers"
                      % (a.dtype.itemsize * 8,
                         _integer_py_dtype.itemsize * 8),
                      UmfpackTypeConversionWarning)
    a = np.ascontiguousarray(a, dtype=_integer_py_dtype)
    assert a.ndim == 1
    return a

cdef object _real_py_dtype = np.dtype(np.float64)
assert sizeof(double) == _real_py_dtype.itemsize == 8
cdef object _complex_py_dtype = np.dtype(np.complex128)
assert _complex_py_dtype.itemsize == 2 * sizeof(double) == 16

cdef object secret_handshake = object()

# XX FIXME: factor out common code with Cholmod
cdef class UmfpackBaseFactor(object):
    cdef int _complex

    def __init__(self, handshake):
        if handshake is not secret_handshake:
            raise UmfpackError, "may not be constructed directly"

    cdef np.ndarray _cast(self, np.ndarray arr):
        if not issubclass(arr.dtype.type, np.number):
            raise UmfpackError, "non-numeric dtype %s" % (arr.dtype,)
        if self._complex:
            # All numeric types can be upcast to complex:
            return np.asfortranarray(arr, dtype=_complex_py_dtype)
        else:
            # Refuse to downcast complex types to real:
            if issubclass(arr.dtype.type, np.complexfloating):
                raise UmfpackError, "inconsistent use of complex array"
            else:
                return np.asfortranarray(arr, dtype=_real_py_dtype)

    # This returns a Python object which holds the reference count for the
    # sparse matrix; you must ensure that you hold onto a reference to this
    # Python object for as long as you want to use the arrays
    cdef _view_sparse(self, m,
                      int * n_row, int * n_col,
                      int ** Ap, int ** Ai, double ** Ax):
        if not sparse.isspmatrix_csc(m):
            warnings.warn("converting matrix of class %s to CSC format"
                          % (m.__class__.__name__,),
                          UmfpackTypeConversionWarning)
            m = m.tocsc()
        m.sort_indices()
        cdef np.ndarray indptr = _require_1d_integer(m.indptr)
        cdef np.ndarray indices = _require_1d_integer(m.indices)
        cdef np.ndarray data = self._cast(m.data)
        if n_row:
            n_row[0] = m.shape[0]
        if n_col:
            n_col[0] = m.shape[1]
        Ap[0] = indptr.data
        Ai[0] = indices.data
        Ax[0] = data.data
        return (indptr, indices, data)

cdef class UmfpackInfo
cdef class UmfpackNumericFactor

cdef class UmfpackSymbolicFactor(UmfpackBaseFactor):
    cdef symbolic_t _symbolic
    cdef UmfpackInfo info

    def __dealloc__(self):
        if self._is_complex:
            umfpack_zi_free_symbolic(&self._symbolic)
        else:
            umfpack_di_free_symbolic(&self._symbolic)

    def lu(self, A, **control_args):
        # XX FIXME control
        cdef int * Ap, * Ai
        cdef double * Ax
        handle = self._view_sparse(A, NULL, NULL, &Ap, &Ai, &Ax)
        cdef UmfpackNumericFactor n = UmfpackNumericFactor(secret_handshake)
        n._complex = self._complex
        n._work_i = np.empty(A.shape[1], dtype=_integer_py_dtype)
        if self._complex:
            _check(umfpack_zi_numeric(Ap, Ai, Ax, NULL,
                                      self._symbolic, &n._numeric,
                                      NULL, n.info._info))
            n._work_d = np.empty(10 * A.shape[1], dtype=_real_py_dtype)
        else:
            _check(umfpack_di_numeric(Ap, Ai, Ax,
                                      self._symbolic, &n._numeric,
                                      NULL, n.info._info))
            n._work_d = np.empty(5 * A.shape[1], dtype=_real_py_dtype)
        return n
        
def analyze(A, ordering=None, **control_args):
    # XX FIXME control
    cdef UmfpackSymbolicFactor s = UmfpackSymbolicFactor(secret_handshake)
    s._complex = issubclass(A.dtype.type, np.complexfloating):
    cdef int n_row, n_col
    cdef int * Ap, * Ai
    cdef double * Ax
    handle = s._view_sparse(A, &n_row, &n_col, &Ap, &Ai, &Ax)
    cdef int * Qinit
    if ordering is not None:
        ordering_handle = _require_1d_integer(ordering)
        Qinit = <int *>ordering_handle.data
        if s._complex:
            _check(umfpack_zi_qsymbolic(n_row, n_col, Ap, Ai, Ax, NULL,
                                        Qinit,
                                        &s._symbolic, NULL, &s.info._info))
        else:
            _check(umfpack_di_qsymbolic(n_row, n_col, Ap, Ai, Ax,
                                        Qinit,
                                        &s._symbolic, NULL, &s.info._info))
    else:
        if s._complex:
            _check(umfpack_zi_symbolic(n_row, n_col, Ap, Ai, Ax, NULL,
                                       &s._symbolic, NULL, &s.info._info))
        else:
            _check(umfpack_di_symbolic(n_row, n_col, Ap, Ai, Ax,
                                       &s._symbolic, NULL, &s.info._info))
    return s

def lu(A, ordering=None, **control_args):
    return analyze(A, ordering=ordering, **control_args).lu(A, **control_args)

cdef class UmfpackNumericFactor(UmfpackBaseFactor):
    cdef numeric_t _numeric
    cdef UmfpackInfo info
    cdef np.ndarray _work_i, _work_d

    def __dealloc__(self):
        if self._is_complex:
            umfpack_zi_free_numeric(&self._numeric)
        else:
            umfpack_di_free_numeric(&self._numeric)
        
    def solve_A(self, b, A=None, return_info=False, **control_args):
        return self._solve(UMFPACK_A, A, b, return_info, control_args)

    def solve_Ah(self, b, A=None, return_info=False, **control_args):
        return self._solve(UMFPACK_At, A, b, return_info, control_args)

    def solve_At(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Aat, A, b, return_info, control_args)

    def solve_L(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_L, A, b, return_info, control_args)
    
    def solve_U(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_U, A, b, return_info, control_args)
    
    def solve_Lh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Lt, A, b, return_info, control_args)
    
    def solve_Uh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Ut, A, b, return_info, control_args)
    
    def solve_PhL(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Pt_L, A, b, return_info, control_args)
    
    def solve_UQh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_U_Qt, A, b, return_info, control_args)
    
    def solve_LhP(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Lt_P, A, b, return_info, control_args)
    
    def solve_QUh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Q_Ut, A, b, return_info, control_args)
    
    def solve_LtP(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Lat_P, A, b, return_info, control_args)
    
    def solve_Lh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Lat, A, b, return_info, control_args)
    
    def solve_QUh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Q_Uat, A, b, return_info, control_args)
    
    def solve_Uh(self, b, return_info=False, **control_args):
        return self._solve(UMFPACK_Uat, A, b, return_info, control_args)
    
    def _solve(self, int system, A, b, int return_info, control_args):
        # XX FIXME control
        cdef int * Ap = NULL, * Ai = NULL
        cdef double * Ax = NULL
        if A is not None:
            # XX FIXME: force control to enable/disable IRSTEP
            A_handle = self._view(A, NULL, NULL, &Ap, &Ai, &Ax)
        cdef np.ndarray used_b
        if sparse.issparse(b):
            warnings.warn("UMFPACK only supports dense right-hand-sides;"
                          " converting sparse 'b' matrix to dense",
                          UmfpackTypeConversionWarning)
            b = b.todense()
        cdef np.ndarray x
        if self._complex:
            used_b = np.asarray(b, dtype=_complex_py_dtype)
        else:
            used_b = np.asarray(b, dtype=_real_py_dtype)
        original_shape = used_b.shape
        # We handle arbitrary-dimensional arrays in a way compatible with
        # np.dot, i.e., we solve each "column", where "column" is "slice along
        # the second-to-last index". The simplest way to do this is to reshape
        # everything to 3 dimensions, iterate over the 1st and 3rd dimension,
        # and then reshape back:
        if used_b.ndim == 0:
            raise UmfpackError, "b must have at least 1 dimension"
        elif used_b.ndim == 1:
            used_b.resize(1, -1, 1)
        elif used_b.ndim != 3:
            used_b.resize((-1,) + used_b.shape[-2:])
        x = np.empty(used_b.shape, dtype=used_b.dtype)
        cdef np.ndarray current_x, current_b
        assert used_b.ndim == 3
        if return_info:
            info = np.empty((used_b.shape[0], used_b.shape[2]), dtype=object)
        cdef UmfpackInfo current_info
        cdef double * c_current_info = NULL
        for i in used_b.shape[0]:
            for j in used_b.shape[2]:
                current_x = x[i, :, j]
                current_b = used_b[i, :, j]
                if return_info:
                    current_info = UmfpackInfo()
                    info[i, j] = current_info
                    c_current_info = current_info._info
                if self._complex:
                    _check(umfpack_zi_wsolve(system, Ap, Ai, Ax, NULL,
                                             current_x.data, NULL,
                                             current_b.data, NULL,
                                             self._numeric,
                                             NULL, c_current_info,
                                             self._work_i.data,
                                             self._work_d.data))
                else:
                    _check(umfpack_di_wsolve(system, Ap, Ai, Ax,
                                             current_x.data,
                                             current_b.data,
                                             self._numeric,
                                             NULL, c_current_info,
                                             self._work_i.data,
                                             self._work_d.data))
        x.resize(original_shape)
        if return_info:
            assert len(original_shape) > 0
            if len(original_shape) == 1:
                info_shape = ()
            elif len(original_shape) == 2:
                info_shape = (original_shape[1],)
            else:
                info_shape = original_shape[:-2] + original_shape[-1:]
            info.resize(info_shape)
            return (x, info)
        else:
            return x

    def L_U_P_Q_R_recip(self):
        cdef int lnz, unz, n_row, n_col
        if self._complex:
            _check(umfpack_zi_get_lunz(&lnz, &unz, &n_row, &n_col, NULL,
                                       self._numeric))
        else:
            _check(umfpack_di_get_lunz(&lnz, &unz, &n_row, &n_col, NULL,
                                       self._numeric))
        l_shape = (n_row, min(n_row, n_col))
        u_shape = (min(n_row, n_col), n_col)
        cdef np.ndarray Lp, Lj, Lx, Up, Ui, Ux, P, Q, Rs
        cdef int do_recip
        if self._complex:
            data_dtype = _complex_py_dtype
        else:
            data_dtype = _real_py_dtype
        Lp = np.empty(n_row + 1, dtype=_integer_py_dtype)
        Lj = np.empty(lnz, dtype=_integer_py_dtype)
        Lx = np.empty(lnz, dtype=data_dtype)
        Up = np.empty(n_col + 1, dtype=_integer_py_dtype)
        Ui = np.empty(unz, dtype=_integer_py_dtype)
        Ux = np.empty(unz, dtype=data_dtype)
        P = np.empty(n_row, dtype=_integer_py_dtype)
        Q = np.empty(n_col, dtype=_integer_py_dtype)
        R = np.empty(n_row, dtype=_real_py_dtype)
        if self._complex:
            _check(umfpack_zi_get_numeric(Lp.data, Lj.data, Lx.data, NULL,
                                          Up.data, Uj.data, Ux.data, NULL,
                                          P.data, Q.data, NULL, NULL,
                                          &do_recip, R.data,
                                          self._numeric))
        else:
            _check(umfpack_di_get_numeric(Lp.data, Lj.data, Lx.data,
                                          Up.data, Uj.data, Ux.data,
                                          P.data, Q.data, NULL,
                                          &do_recip, R.data,
                                          self._numeric))
        return (sparse.csr_matrix((Lp, Lj, Lx)),
                sparse.csc_matrix((Up, Uj, Ux)),
                P, Q, R, bool(do_recip))
                                  
    def slogdet(self, return_info=False):
        """Compute the sign and (natural) logarithm of the determinant of an
        array.

        If an array has a very small or very large determinant, then a call to
        :meth:`det` may overflow or underflow. This routine is more robust
        against such issues, because it computes the logarithm of the
        determinant rather than the determinant itself.

        Returns a pair `(sign, logdet)` (or `(sign, logdet, info)` if
        `return_info` is `True`).

        `sign` is a number representing the sign of the determinant. For a
        real matrix, this is 1, 0, or -1. For a complex matrix, this is a
        complex number with absolute value 1 (i.e., it is on the unit circle),
        or else 0.

        `logdet` is always a real number, the natural log of the absolute
        value of the determinant.

        If the determinant is exactly zero, then `sign` will be 0 and `logdet`
        will be -Inf. In all cases, the determinant is equal to `sign *
        np.exp(logdet)`.

        See also :func:`numpy.linalg.slogdet`.
        """
        cdef np.ndarray M
        cdef double Ex
        cdef UmfpackInfo info
        cdef double * c_info = NULL
        if return_info:
            info = UmfpackInfo()
            c_info = info._info
        # Result is in 'scientific notation': determinant is M * 10^Ex:
        if self._complex:
            M = np.empty((), dtype=_complex_py_dtype)
            _check(umfpack_zi_get_determinant(M.data, NULL, &Ex,
                                              self._numeric, c_info))
            sign = M[0] / np.abs(M[0])
        else:
            M = np.empty((), dtype=_real_py_dtype)
            _check(umfpack_di_get_determinant(M.data, &Ex,
                                              self._numeric, c_info))
            sign = np.sign(M)
        logdet = np.log(np.abs(M)) + ex * np.log(10)
        if return_info:
            return (sign, logdet, info)
        else:
            return (sign, logdet)

    def det(self, return_info=False):
        if return_info:
            (sign, logdet, info) = self.slogdet(return_info=True)
            return sign * np.exp(logdet), info
        else:
            (sign, logdet) = self.slogdet(return_info=False)
            return sign * np.exp(logdet)

# Info handling:
cdef extern from *:
    enum:
        UMFPACK_STATUS
        UMFPACK_NROW
        UMFPACK_NCOL
        UMFPACK_NZ
        UMFPACK_SIZE_OF_UNIT
        UMFPACK_SIZE_OF_INT
        UMFPACK_SIZE_OF_LONG
        UMFPACK_SIZE_OF_POINTER
        UMFPACK_SIZE_OF_ENTRY
        UMFPACK_NDENSE_ROW
        UMFPACK_NEMPTY_ROW
        UMFPACK_NDENSE_COL
        UMFPACK_NEMPTY_COL
        UMFPACK_SYMBOLIC_DEFRAG
        UMFPACK_SYMBOLIC_PEAK_MEMORY
        UMFPACK_SYMBOLIC_SIZE
        UMFPACK_SYMBOLIC_TIME
        UMFPACK_SYMBOLIC_WALLTIME
        UMFPACK_STRATEGY_USED
        UMFPACK_ORDERING_USED
        UMFPACK_QFIXED
        UMFPACK_DIAG_PREFERRED
        UMFPACK_PATTERN_SYMMETRY
        UMFPACK_NZ_A_PLUS_AT
        UMFPACK_NZDIAG
        UMFPACK_SYMMETRIC_LUNZ
        UMFPACK_SYMMETRIC_FLOPS
        UMFPACK_SYMMETRIC_NDENSE
        UMFPACK_SYMMETRIC_DMAX
        UMFPACK_2BY2_NWEAK
        UMFPACK_2BY2_UNMATCHED
        UMFPACK_2BY2_PATTERN_SYMMETRY
        UMFPACK_2BY2_NZ_PA_PLUS_PAT
        UMFPACK_2BY2_NZDIAG
        UMFPACK_COL_SINGLETONS
        UMFPACK_ROW_SINGLETONS
        UMFPACK_N2
        UMFPACK_S_SYMMETRIC
        UMFPACK_NUMERIC_SIZE_ESTIMATE
        UMFPACK_PEAK_MEMORY_ESTIMATE
        UMFPACK_FLOPS_ESTIMATE
        UMFPACK_LNZ_ESTIMATE
        UMFPACK_UNZ_ESTIMATE
        UMFPACK_VARIABLE_INIT_ESTIMATE
        UMFPACK_VARIABLE_PEAK_ESTIMATE
        UMFPACK_VARIABLE_FINAL_ESTIMATE
        UMFPACK_MAX_FRONT_SIZE_ESTIMATE
        UMFPACK_MAX_FRONT_NROWS_ESTIMATE
        UMFPACK_MAX_FRONT_NCOLS_ESTIMATE
        UMFPACK_NUMERIC_SIZE
        UMFPACK_PEAK_MEMORY
        UMFPACK_FLOPS
        UMFPACK_LNZ
        UMFPACK_UNZ
        UMFPACK_VARIABLE_INIT
        UMFPACK_VARIABLE_PEAK
        UMFPACK_VARIABLE_FINAL
        UMFPACK_MAX_FRONT_SIZE
        UMFPACK_MAX_FRONT_NROWS
        UMFPACK_MAX_FRONT_NCOLS
        UMFPACK_NUMERIC_DEFRAG
        UMFPACK_NUMERIC_REALLOC
        UMFPACK_NUMERIC_COSTLY_REALLOC
        UMFPACK_COMPRESSED_PATTERN
        UMFPACK_LU_ENTRIES
        UMFPACK_NUMERIC_TIME
        UMFPACK_UDIAG_NZ
        UMFPACK_RCOND
        UMFPACK_WAS_SCALED
        UMFPACK_RSMIN
        UMFPACK_RSMAX
        UMFPACK_UMIN
        UMFPACK_UMAX
        UMFPACK_ALLOC_INIT_USED
        UMFPACK_FORCED_UPDATES
        UMFPACK_NUMERIC_WALLTIME
        UMFPACK_NOFF_DIAG
        UMFPACK_ALL_LNZ
        UMFPACK_ALL_UNZ
        UMFPACK_NZDROPPED
        UMFPACK_IR_TAKEN
        UMFPACK_IR_ATTEMPTED
        UMFPACK_OMEGA1
        UMFPACK_OMEGA2
        UMFPACK_SOLVE_FLOPS
        UMFPACK_SOLVE_TIME
        UMFPACK_SOLVE_WALLTIME

        # For UMFPACK_ORDERING_USED:
        UMFPACK_ORDERING_COLAMD
        UMFPACK_ORDERING_AMD
        UMFPACK_ORDERING_GIVEN
        
    
cdef class UmfpackInfo(DictMixin):
    cdef double _info[UMFPACK_INFO]

    _info_entries = {
        "STATUS": UMFPACK_STATUS,
        "NROW": UMFPACK_NROW,
        "NCOL": UMFPACK_NCOL,
        "NZ": UMFPACK_NZ,
        "SIZE_OF_UNIT": UMFPACK_SIZE_OF_UNIT,
        "SIZE_OF_INT": UMFPACK_SIZE_OF_INT,
        "SIZE_OF_LONG": UMFPACK_SIZE_OF_LONG,
        "SIZE_OF_POINTER": UMFPACK_SIZE_OF_POINTER,
        "SIZE_OF_ENTRY": UMFPACK_SIZE_OF_ENTRY,
        "NDENSE_ROW": UMFPACK_NDENSE_ROW,
        "NEMPTY_ROW": UMFPACK_NEMPTY_ROW,
        "NDENSE_COL": UMFPACK_NDENSE_COL,
        "NEMPTY_COL": UMFPACK_NEMPTY_COL,
        "SYMBOLIC_DEFRAG": UMFPACK_SYMBOLIC_DEFRAG,
        "SYMBOLIC_PEAK_MEMORY": UMFPACK_SYMBOLIC_PEAK_MEMORY,
        "SYMBOLIC_SIZE": UMFPACK_SYMBOLIC_SIZE,
        "SYMBOLIC_TIME": UMFPACK_SYMBOLIC_TIME,
        "SYMBOLIC_WALLTIME": UMFPACK_SYMBOLIC_WALLTIME,
        "STRATEGY_USED": UMFPACK_STRATEGY_USED,
        "ORDERING_USED": UMFPACK_ORDERING_USED,
        "QFIXED": UMFPACK_QFIXED,
        "DIAG_PREFERRED": UMFPACK_DIAG_PREFERRED,
        "PATTERN_SYMMETRY": UMFPACK_PATTERN_SYMMETRY,
        "NZ_A_PLUS_AT": UMFPACK_NZ_A_PLUS_AT,
        "NZDIAG": UMFPACK_NZDIAG,
        "SYMMETRIC_LUNZ": UMFPACK_SYMMETRIC_LUNZ,
        "SYMMETRIC_FLOPS": UMFPACK_SYMMETRIC_FLOPS,
        "SYMMETRIC_NDENSE": UMFPACK_SYMMETRIC_NDENSE,
        "SYMMETRIC_DMAX": UMFPACK_SYMMETRIC_DMAX,
        "2BY2_NWEAK": UMFPACK_2BY2_NWEAK,
        "2BY2_UNMATCHED": UMFPACK_2BY2_UNMATCHED,
        "2BY2_PATTERN_SYMMETRY": UMFPACK_2BY2_PATTERN_SYMMETRY,
        "2BY2_NZ_PA_PLUS_PAT": UMFPACK_2BY2_NZ_PA_PLUS_PAT,
        "2BY2_NZDIAG": UMFPACK_2BY2_NZDIAG,
        "COL_SINGLETONS": UMFPACK_COL_SINGLETONS,
        "ROW_SINGLETONS": UMFPACK_ROW_SINGLETONS,
        "N2": UMFPACK_N2,
        "S_SYMMETRIC": UMFPACK_S_SYMMETRIC,
        "NUMERIC_SIZE_ESTIMATE": UMFPACK_NUMERIC_SIZE_ESTIMATE,
        "PEAK_MEMORY_ESTIMATE": UMFPACK_PEAK_MEMORY_ESTIMATE,
        "FLOPS_ESTIMATE": UMFPACK_FLOPS_ESTIMATE,
        "LNZ_ESTIMATE": UMFPACK_LNZ_ESTIMATE,
        "UNZ_ESTIMATE": UMFPACK_UNZ_ESTIMATE,
        "VARIABLE_INIT_ESTIMATE": UMFPACK_VARIABLE_INIT_ESTIMATE,
        "VARIABLE_PEAK_ESTIMATE": UMFPACK_VARIABLE_PEAK_ESTIMATE,
        "VARIABLE_FINAL_ESTIMATE": UMFPACK_VARIABLE_FINAL_ESTIMATE,
        "MAX_FRONT_SIZE_ESTIMATE": UMFPACK_MAX_FRONT_SIZE_ESTIMATE,
        "MAX_FRONT_NROWS_ESTIMATE": UMFPACK_MAX_FRONT_NROWS_ESTIMATE,
        "MAX_FRONT_NCOLS_ESTIMATE": UMFPACK_MAX_FRONT_NCOLS_ESTIMATE,
        "NUMERIC_SIZE": UMFPACK_NUMERIC_SIZE,
        "PEAK_MEMORY": UMFPACK_PEAK_MEMORY,
        "FLOPS": UMFPACK_FLOPS,
        "LNZ": UMFPACK_LNZ,
        "UNZ": UMFPACK_UNZ,
        "VARIABLE_INIT": UMFPACK_VARIABLE_INIT,
        "VARIABLE_PEAK": UMFPACK_VARIABLE_PEAK,
        "VARIABLE_FINAL": UMFPACK_VARIABLE_FINAL,
        "MAX_FRONT_SIZE": UMFPACK_MAX_FRONT_SIZE,
        "MAX_FRONT_NROWS": UMFPACK_MAX_FRONT_NROWS,
        "MAX_FRONT_NCOLS": UMFPACK_MAX_FRONT_NCOLS,
        "NUMERIC_DEFRAG": UMFPACK_NUMERIC_DEFRAG,
        "NUMERIC_REALLOC": UMFPACK_NUMERIC_REALLOC,
        "NUMERIC_COSTLY_REALLOC": UMFPACK_NUMERIC_COSTLY_REALLOC,
        "COMPRESSED_PATTERN": UMFPACK_COMPRESSED_PATTERN,
        "LU_ENTRIES": UMFPACK_LU_ENTRIES,
        "NUMERIC_TIME": UMFPACK_NUMERIC_TIME,
        "UDIAG_NZ": UMFPACK_UDIAG_NZ,
        "RCOND": UMFPACK_RCOND,
        "WAS_SCALED": UMFPACK_WAS_SCALED,
        "RSMIN": UMFPACK_RSMIN,
        "RSMAX": UMFPACK_RSMAX,
        "UMIN": UMFPACK_UMIN,
        "UMAX": UMFPACK_UMAX,
        "ALLOC_INIT_USED": UMFPACK_ALLOC_INIT_USED,
        "FORCED_UPDATES": UMFPACK_FORCED_UPDATES,
        "NUMERIC_WALLTIME": UMFPACK_NUMERIC_WALLTIME,
        "NOFF_DIAG": UMFPACK_NOFF_DIAG,
        "ALL_LNZ": UMFPACK_ALL_LNZ,
        "ALL_UNZ": UMFPACK_ALL_UNZ,
        "NZDROPPED": UMFPACK_NZDROPPED,
        "IR_TAKEN": UMFPACK_IR_TAKEN,
        "IR_ATTEMPTED": UMFPACK_IR_ATTEMPTED,
        "OMEGA1": UMFPACK_OMEGA1,
        "OMEGA2": UMFPACK_OMEGA2,
        "SOLVE_FLOPS": UMFPACK_SOLVE_FLOPS,
        "SOLVE_TIME": UMFPACK_SOLVE_TIME,
        "SOLVE_WALLTIME": UMFPACK_SOLVE_WALLTIME,
    }    

    _orderings = {
        UMFPACK_ORDERING_COLAMD: "COLAMD",
        UMFPACK_ORDERING_AMD: "AMD", 
        UMFPACK_ORDERING_GIVEN: "user",
        }
    
    def __getitem__(self, name):
        name = name.upper()
        # May raise KeyError, which we let propagate:
        index = self._info_entries[name]
        value = self._info[index]
        if index == UMFPACK_ORDERING_USED:
            return self._orderings[value]
        else:
            return value

    def keys(self):
        return self._info_entries.keys()
