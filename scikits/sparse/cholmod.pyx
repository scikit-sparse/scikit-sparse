# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Released under the terms of the GNU GPL v2, or, at your option, any
# later version.

import warnings
cimport stdlib
cimport python as py
import numpy as np
cimport numpy as np
from scipy import sparse

np.import_array()

cdef extern from "numpy/arrayobject.h":
    # Cython 0.12 complains about PyTypeObject being an "incomplete type" on
    # this line:
    #py.PyTypeObject PyArray_Type
    # So use a hack:
    struct MyHackReallyPyTypeObject:
        pass
    MyHackReallyPyTypeObject PyArray_Type
    object PyArray_NewFromDescr(MyHackReallyPyTypeObject * subtype,
                                np.dtype descr,
                                int nd,
                                np.npy_intp * dims,
                                np.npy_intp * strides,
                                void * data,
                                int flags,
                                object obj)
    # This is ridiculous: the description of PyArrayObject in numpy.pxd does
    # not mention the 'base' member, so we need a separate wrapper just to
    # expose it:
    ctypedef struct ndarray_with_base "PyArrayObject":
        void * base

cdef inline np.ndarray set_base(np.ndarray arr, object base):
    cdef ndarray_with_base * hack = <ndarray_with_base *> arr
    py.Py_INCREF(base)
    hack.base = <void *> base
    return arr

cdef extern from "suitesparse/cholmod.h":
    cdef enum:
        CHOLMOD_INT
        CHOLMOD_PATTERN, CHOLMOD_REAL, CHOLMOD_COMPLEX
        CHOLMOD_DOUBLE
        CHOLMOD_AUTO, CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL
        CHOLMOD_OK, CHOLMOD_NOT_POSDEF
        CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD, CHOLMOD_DLt, CHOLMOD_L
        CHOLMOD_Lt, CHOLMOD_D, CHOLMOD_P, CHOLMOD_Pt

    ctypedef struct cholmod_common:
        int supernodal
        int status
        int print_ "print"
        void (*error_handler)(int status, char * file, int line, char * msg)
        
    int cholmod_start(cholmod_common *) except? 0
    int cholmod_finish(cholmod_common *) except? 0
    int cholmod_check_common(cholmod_common *) except? 0
    int cholmod_print_common(char *, cholmod_common *) except? 0

    ctypedef struct cholmod_sparse:
        size_t nrow, ncol, nzmax
        void * p # column pointers
        void * i # row indices
        void * x
        int stype # 0 = regular, >0 = upper triangular, <0 = lower triangular
        int itype # type of p, i, nz
        int xtype
        int dtype
        int sorted
        int packed

    int cholmod_free_sparse(cholmod_sparse **, cholmod_common *) except? 0
    int cholmod_check_sparse(cholmod_sparse *, cholmod_common *) except? 0
    int cholmod_print_sparse(cholmod_sparse *, char *, cholmod_common *) except? 0

    ctypedef struct cholmod_dense:
        size_t nrow, ncol, nzmax
        size_t d
        void * x
        int xtype, dtype

    int cholmod_free_dense(cholmod_dense **, cholmod_common *) except? 0
    int cholmod_check_dense(cholmod_dense *, cholmod_common *) except? 0
    int cholmod_print_dense(cholmod_dense *, char *, cholmod_common *) except? 0

    ctypedef struct cholmod_factor:
        size_t n
        void * Perm
        int itype
        int xtype
        int is_ll, is_super, is_monotonic
    int cholmod_free_factor(cholmod_factor **, cholmod_common *) except? 0
    cholmod_factor * cholmod_copy_factor(cholmod_factor *, cholmod_common *) except? NULL

    cholmod_factor * cholmod_analyze(cholmod_sparse *, cholmod_common *) except? NULL
    int cholmod_factorize_p(cholmod_sparse *, double beta[2],
                              int * fset, size_t fsize,
                              cholmod_factor *,
                              cholmod_common *) except? 0

    cholmod_sparse * cholmod_submatrix(cholmod_sparse *,
                                         int * rset, int rsize,
                                         int * cset, int csize,
                                         int values, int sorted,
                                         cholmod_common *) except? NULL
    int cholmod_updown(int update, cholmod_sparse *, cholmod_factor *,
                         cholmod_common *) except? 0
    
    cholmod_dense * cholmod_solve(int, cholmod_factor *,
                                    cholmod_dense *, cholmod_common *) except? NULL
    cholmod_sparse * cholmod_spsolve(int, cholmod_factor *,
                                       cholmod_sparse *, cholmod_common *) except? NULL
    
    int cholmod_change_factor(int to_xtype, int to_ll, int to_super,
                                int to_packed, int to_monotonic,
                                cholmod_factor *, cholmod_common *) except? 0
    cholmod_sparse * cholmod_factor_to_sparse(cholmod_factor *,
                                                cholmod_common *) except? NULL
    
cdef class Common
cdef class Factor

class CholmodError(Exception):
    pass

class CholmodWarning(UserWarning):
    pass

class CholmodTypeConversionWarning(CholmodWarning):
    pass

cdef object _integer_py_dtype = np.dtype(np.int32)
assert sizeof(int) == _integer_py_dtype.itemsize == 4

cdef _require_1d_integer(a):
    if a.dtype.itemsize != _integer_py_dtype.itemsize:
        warnings.warn("array contains %s bit integers; "
                      "this will be slower than using %s bit integers"
                      % (a.dtype.itemsize * 8,
                         _integer_py_dtype.itemsize * 8),
                      CholmodTypeConversionWarning)
    a = np.ascontiguousarray(a, dtype=_integer_py_dtype)
    assert a.ndim == 1
    return a

cdef object _real_py_dtype = np.dtype(np.float64)
assert sizeof(double) == _real_py_dtype.itemsize == 8
cdef object _complex_py_dtype = np.dtype(np.complex128)
assert _complex_py_dtype.itemsize == 2 * sizeof(double) == 16

##########
# Cholmod -> Python conversion:
##########

cdef np.dtype _np_dtype_for(int xtype):
    if xtype == CHOLMOD_COMPLEX:
        py.Py_INCREF(_complex_py_dtype)
        return _complex_py_dtype
    elif xtype == CHOLMOD_REAL:
        py.Py_INCREF(_real_py_dtype)
        return _real_py_dtype
    else:
        raise CholmodError, "cholmod->numpy type conversion failed"

cdef class _SparseCleanup(object):
    cdef cholmod_sparse * _sparse
    cdef Common _common
    def __dealloc__(self):
        cholmod_free_sparse(&self._sparse, &self._common._common)

cdef _py_sparse(cholmod_sparse * m, Common common):
    """Build a scipy.sparse.csc_matrix that's a view onto m, with a 'base' with
    appropriate destructor. 'm' must have been allocated by cholmod."""

    # This is a little tricky -- we build 3 arrays, views on each part of the
    # cholmod_dense object -- and they all have the same _SparseCleanup object
    # as base. So none of them will be deallocated until they have all become
    # unused. Then those are built into a csc_matrix.

    # Construct cleaner first, so even if we later raise an exception we still
    # fulfill the contract that we will take care of cleanup:
    assert m is not NULL
    assert common is not None
    cdef _SparseCleanup cleaner = _SparseCleanup()
    cleaner._sparse = m
    cleaner._common = common
    shape = (m.nrow, m.ncol)
    assert m.itype == CHOLMOD_INT
    py.Py_INCREF(_integer_py_dtype)
    cdef np.npy_intp ncol_plus_1 = m.ncol + 1
    indptr = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                           _integer_py_dtype, 1,
                                           &ncol_plus_1,
                                           NULL,
                                           m.p,
                                           np.NPY_F_CONTIGUOUS, None),
                      cleaner)
    py.Py_INCREF(_integer_py_dtype)
    cdef np.npy_intp nzmax = m.nzmax
    indices = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                            _integer_py_dtype, 1,
                                            &nzmax,
                                            NULL,
                                            m.i,
                                            np.NPY_F_CONTIGUOUS, None),
                       cleaner)
    data_dtype = _np_dtype_for(m.xtype)
    data = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                         data_dtype, 1,
                                         &nzmax,
                                         NULL,
                                         m.x,
                                         np.NPY_F_CONTIGUOUS, None),
                    cleaner)
    return sparse.csc_matrix((data, indices, indptr), shape=shape)

cdef class _DenseCleanup(object):
    cdef cholmod_dense * _dense
    cdef Common _common
    def __dealloc__(self):
        cholmod_free_dense(&self._dense, &self._common._common)

cdef _py_dense(cholmod_dense * m, Common common):
    """Build an ndarray that's a view onto m, with a 'base' with appropriate
    destructor. 'm' must have been allocated by cholmod."""

    assert m is not NULL
    assert common is not None
    # Construct cleaner first, so even if we later raise an exception we still
    # fulfill the contract that we will take care of cleanup:
    cdef _DenseCleanup cleaner = _DenseCleanup()
    cleaner._dense = m
    cleaner._common = common
    cdef np.dtype dtype = _np_dtype_for(m.xtype)
    cdef np.npy_intp dims[2]
    dims[0] = m.nrow
    dims[1] = m.ncol
    cdef np.ndarray out
    return set_base(PyArray_NewFromDescr(&PyArray_Type, dtype, 2, dims, NULL,
                                         m.x, np.NPY_F_CONTIGUOUS, None),
                    cleaner)

cdef void _error_handler(int status, char * file, int line, char * msg) except * with gil:
    if status == CHOLMOD_OK:
        return
    full_msg = "%s:%s: %s (code %s)" % (file, line, msg, status)
    if status > 0:
        # Warning:
        warnings.warn(full_msg, CholmodWarning)
    else:
        raise CholmodError, full_msg

cdef class Common(object):
    cdef cholmod_common _common
    cdef int _complex
    cdef int _xtype

    def __cinit__(self, _complex):
        self._complex = _complex
        if self._complex:
            self._xtype = CHOLMOD_COMPLEX
        else:
            self._xtype = CHOLMOD_REAL
        cholmod_start(&self._common)
        self._common.print_ = 0
        self._common.error_handler = <void (*)(int, char *, int, char *)>_error_handler

    def __dealloc__(self):
        cholmod_finish(&self._common)

    # Debugging:
    def _print(self):
        print cholmod_check_common(&self._common)
        name = repr(self)
        return cholmod_print_common(name, &self._common)
        
    def _print_sparse(self, name, symmetric, matrix):
        cdef cholmod_sparse * m
        ref = self._view_sparse(matrix, symmetric, &m)
        print cholmod_check_sparse(m, &self._common)
        return cholmod_print_sparse(m, name, &self._common)

    def _print_dense(self, name, matrix):
        cdef cholmod_dense * m
        ref = self._view_dense(matrix, &m)
        print cholmod_check_dense(m, &self._common)
        return cholmod_print_dense(m, name, &self._common)

    ##########
    # Python -> Cholmod conversion:
    ##########
    cdef np.ndarray _cast(self, np.ndarray arr):
        if not issubclass(arr.dtype.type, np.number):
            raise CholmodError, "non-numeric dtype %s" % (arr.dtype,)
        if self._complex:
            # All numeric types can be upcast to complex:
            return np.asfortranarray(arr, dtype=_complex_py_dtype)
        else:
            # Refuse to downcast complex types to real:
            if issubclass(arr.dtype.type, np.complexfloating):
                raise CholmodError, "inconsistent use of complex array"
            else:
                return np.asfortranarray(arr, dtype=_real_py_dtype)

    # This returns a Python object which holds the reference count for the
    # sparse matrix; you must ensure that you hold onto a reference to this
    # Python object for as long as you want to use the cholmod_sparse*
    cdef _view_sparse(self, m, symmetric, cholmod_sparse **outp):
        if not sparse.isspmatrix_csc(m):
            warnings.warn("converting matrix of class %s to CSC format"
                          % (m.__class__.__name__,),
                          CholmodTypeConversionWarning)
            m = m.tocsc()
        if symmetric and m.shape[0] != m.shape[1]:
            raise CholmodError, "supposedly symmetric matrix is not square!"
        m.sort_indices()
        cdef np.ndarray indptr = _require_1d_integer(m.indptr)
        cdef np.ndarray indices = _require_1d_integer(m.indices)
        cdef np.ndarray data = self._cast(m.data)
        cdef cholmod_sparse * out = <cholmod_sparse *> stdlib.malloc(sizeof(cholmod_sparse))
        try:
            out.nrow, out.ncol = m.shape
            out.nzmax = m.nnz
            out.p = indptr.data
            out.i = indices.data
            out.x = data.data
            if symmetric:
                out.stype = -1
            else:
                out.stype = 0
            out.itype = CHOLMOD_INT
            out.dtype = CHOLMOD_DOUBLE
            out.xtype = self._xtype
            out.sorted = 1
            out.packed = 1
            outp[0] = out
            return (indptr, indices, data)
        except:
            stdlib.free(out)
            raise

    # This returns a Python object which holds the reference count for the
    # dense matrix; you must ensure that you hold onto a reference to this
    # Python object for as long as you want to use the cholmod_dense*
    cdef _view_dense(self, np.ndarray m, cholmod_dense **outp):
        if m.ndim != 2:
            raise CholmodError, "array has %s dimensions (expected 2)" % m.ndim
        m = self._cast(m)
        cdef cholmod_dense * out = <cholmod_dense *> stdlib.malloc(sizeof(cholmod_dense))
        try:
            out.nrow = m.shape[0]
            out.ncol = m.shape[1]
            out.nzmax = m.size
            out.d = m.strides[1] // m.itemsize
            out.x = m.data
            out.dtype = CHOLMOD_DOUBLE
            out.xtype = self._xtype
            outp[0] = out
            return m
        except:
            stdlib.free(out)
            raise

cdef object factor_secret_handshake = object()

cdef class Factor(object):
    """This class represents a Cholesky decomposition with a particular
    fill-reducing permutation. It cannot be instantiated directly; see
    :func:`analyze` and :func:`cholesky`, both of which return objects of type
    Factor."""

    cdef readonly Common _common
    cdef cholmod_factor * _factor

    def __init__(self, handshake):
        if handshake is not factor_secret_handshake:
            raise CholmodError, "Factor may not be constructed directly; use analyze()"

    def __dealloc__(self):
        cholmod_free_factor(&self._factor, &self._common._common)

    def cholesky_inplace(self, A, beta=0):
        """Updates this Factor so that it represents the Cholesky
        decomposition of :math:`A + \\beta I`, rather than whatever it
        contained before.

        :math:`A` must have the same pattern of non-zeros as the matrix used
        to create this factor originally."""
        return self._cholesky_inplace(A, True, beta=beta)

    def cholesky_AAt_inplace(self, A, beta=0):
        """The same as :meth:`cholesky_inplace`, except it factors :math:`AA'
        + \\beta I` instead of :math:`A + \\beta I`."""
        return self._cholesky_inplace(A, False, beta=beta)

    def _cholesky_inplace(self, A, symmetric, beta=0, **kwargs):
        cdef cholmod_sparse * c_A
        c_A_ref = self._common._view_sparse(A, symmetric, &c_A)
        cdef double c_beta[2]
        c_beta[0] = beta
        c_beta[1] = 0
        cholmod_factorize_p(c_A, c_beta, NULL, 0,
                            self._factor, &self._common._common)
        if self._common._common.status == CHOLMOD_NOT_POSDEF:
            raise CholmodError, "Matrix is not positive definite"

    def _clone(self):
        cdef cholmod_factor * c_clone = cholmod_copy_factor(self._factor,
                                                            &self._common._common)
        assert c_clone
        cdef Factor clone = Factor(factor_secret_handshake)
        clone._common = self._common
        clone._factor = c_clone
        return clone

    def cholesky(self, A, beta=0):
        """The same as :meth:`cholesky_inplace` except that it first creates
        a copy of the current :class:`Factor` and modifes the copy.

        :returns: The new :class:`Factor` object."""
        clone = self._clone()
        clone.cholesky_inplace(A, beta=beta)
        return clone

    def cholesky_AAt(self, A, beta=0):
        """The same as :meth:`cholesky_AAt_inplace` except that it first
        creates a copy of the current :class:`Factor` and modifes the copy.

        :returns: The new :class:`Factor` object."""
        clone = self._clone()
        clone.cholesky_AAt_inplace(A, beta=beta)
        return clone

    def update_inplace(self, C, int subtract=False):
        """Incremental building of :math:`AA'` decompositions.

        Updates this factor so that instead of representing the decomposition
        of :math:`AA'`, it instead represents the decomposition of :math:`AA'
        + CC'` (for ``subtract=False``, the default), or :math:`AA' - CC'` (for
        ``subtract=True``). This method does not require that the
        :class:`Factor` was created with :func:`cholesky_AAt`, though that
        is the common case.

        The usual use for this is to factor AA' when A has a large number of
        columns, or those columns become available incrementally. Instead of
        loading all of A into memory, one can load in 'strips' of columns and
        pass them to this method one at a time.

        Note that no fill-reduction analysis is done; whatever permutation was
        chosen by the initial call to :func:`analyze` will be used regardless
        of the pattern of non-zeros in C."""
        # permute C
        cdef cholmod_sparse * c_C
        c_C_ref = self._common._view_sparse(C, False, &c_C)
        cdef cholmod_sparse * C_perm
        C_perm = cholmod_submatrix(c_C,
                                     <int *> self._factor.Perm,
                                     self._factor.n,
                                     NULL, -1, True, True,
                                     &self._common._common)
        assert C_perm
        try:
            cholmod_updown(not subtract, C_perm, self._factor,
                           &self._common._common)
        except:
            cholmod_free_sparse(&C_perm, &self._common._common)
            raise

    # Everything below here will fail for matrices that were only analyzed,
    # not factorized.
    def P(self):
        """Returns the fill-reducing permutation P, as a vector of indices.

        The decomposition :math:`LL'` or :math:`LDL'` is of::

          A[P[:, np.newaxis], P[np.newaxis, :]]

        (or similar for AA')."""
        if self._factor.Perm is NULL:
            raise CholmodError, "you must analyze a matrix first"
        assert self._factor.itype == CHOLMOD_INT
        py.Py_INCREF(_integer_py_dtype)
        cdef np.npy_intp n = self._factor.n
        return set_base(PyArray_NewFromDescr(&PyArray_Type,
                                             _integer_py_dtype, 1, &n,
                                             NULL, self._factor.Perm,
                                             0, None),
                        self)

    def _ensure_L_or_LD_inplace(self, want_L):
        # In CHOLMOD, supernodal factorizations are always LL'. If we request
        # to change to a supernodal LDL' factorization, cholmod_change_factor
        # will silently do nothing! So we can only stay supernodal when LL' is
        # requested:
        want_super = self._factor.is_super and want_L
        cholmod_change_factor(self._factor.xtype,
                              want_L, # to_ll
                              want_super,
                              True, # to_packed
                              self._factor.is_monotonic,
                              self._factor,
                              &self._common._common)
        assert bool(self._factor.is_ll) == want_L

    def _L_or_LD(self, want_L):
        cdef Factor f = self._clone()
        cdef cholmod_sparse * l
        f._ensure_L_or_LD_inplace(want_L)
        l = cholmod_factor_to_sparse(f._factor,
                                     &f._common._common)
        assert l
        return _py_sparse(l, self._common)

    def D(self):
        """If necessary, converts this factorization to the style

          .. math:: LDL' = PAP'

        or

          .. math:: LDL' = PAA'P'

        and then returns the diagonal matrix D *as a 1d vector*.
        """
        # XX FIXME: extract diagonal quickly, without converting to LDL,
        # copying the whole matrix, etc...
        # (or just have a .det() method?)
        return self.LD().diagonal()

    def L(self):
        """If necessary, converts this factorization to the style

          .. math:: LL' = PAP'

        or

          .. math:: LL' = PAA'P'

        and then returns the sparse lower-triangular matrix L.

        .. warning:: The L matrix returned by this method and the one returned
           by :meth:`L_D` are different!
        """
        return self._L_or_LD(True)

    def LD(self):
        """If necessary, converts this factorization to the style

          .. math:: LDL' = PAP'

        or

          .. math:: LDL' = PAA'P'

        and then returns a sparse lower-triangular matrix "LD", which contains
        the D matrix on its diagonal, plus the below-diagonal part of L (the
        actual diagonal of L is all-ones).

        See :meth:`L_D` for a more convenient interface."""
        return self._L_or_LD(False)

    def L_D(self):
        """If necessary, converts this factorization to the style

          .. math:: LDL' = PAP'

        or

          .. math:: LDL' = PAA'P'

        and then returns the pair (L, D) where L is a sparse lower-triangular
        matrix and D is a sparse diagonal matrix.

        .. warning:: The L matrix returned by this method and the one returned
           by :meth:`L` are different!
        """
        ld = self.LD()
        l = sparse.tril(ld, -1) + sparse.eye(*ld.shape)
        d = sparse.dia_matrix((ld.diagonal(), [0]), shape=ld.shape)
        return (l, d)

    def solve_A(self, b):
        """Returns :math:`x`, where :math:`Ax = b` (or :math:`AA'x = b`, if
        you used :func:`cholesky_AAt`).

        :meth:`__call__` is an alias for this function, i.e., you can simply
        call the :class:`Factor` object like a function to solve :math:`Ax =
        b`."""
        return self._solve(b, CHOLMOD_A)

    def __call__(self, b):
        "Alias for :meth:`solve_A`."
        return self.solve_A(b)

    def solve_LDLt(self, b):
        """Returns :math:`x`, where :math:`LDL'x = b`.

        (This is different from :meth:`solve_A` because it does not correct
        for the fill-reducing permutation.)"""
        return self._solve(b, CHOLMOD_LDLt)

    def solve_LD(self, b):
        "Returns :math:`x`, where :math:`LDx = b`."
        self._ensure_L_or_LD_inplace(False)
        return self._solve(b, CHOLMOD_LD)

    def solve_DLt(self, b):
        "Returns :math:`x`, where :math:`DL'x = b`."
        self._ensure_L_or_LD_inplace(False)
        return self._solve(b, CHOLMOD_DLt)

    def solve_L(self, b):
        "Returns :math:`x`, where :math:`Lx = b`."
        self._ensure_L_or_LD_inplace(False)
        return self._solve(b, CHOLMOD_L)

    def solve_Lt(self, b):
        "Returns :math:`x`, where :math:`L'x = b`."
        self._ensure_L_or_LD_inplace(False)
        return self._solve(b, CHOLMOD_Lt)

    def solve_D(self, b):
        "Returns :math:`x`, where :math:`Dx = b`."
        return self._solve(b, CHOLMOD_D)

    def solve_P(self, b):
        "Returns :math:`x`, where :math:`x = Pb`."
        return self._solve(b, CHOLMOD_P)

    def solve_Pt(self, b):
        "Returns :math:`x`, where :math:`x = P'b`."
        return self._solve(b, CHOLMOD_Pt)

    def _solve(self, b, system):
        if sparse.issparse(b):
            return self._solve_sparse(b, system)
        else:
            return self._solve_dense(b, system)

    def _solve_sparse(self, b, system):
        cdef cholmod_sparse * c_b
        b_ref = self._common._view_sparse(b, False, &c_b)
        cdef cholmod_sparse * out
        out = cholmod_spsolve(system, self._factor, c_b,
                              &self._common._common)
        return _py_sparse(out, self._common)

    def _solve_dense(self, b, system):
        b = np.asarray(b)
        if b.ndim == 1:
            b = b[:, np.newaxis]
        cdef cholmod_dense * c_b
        b_ref = self._common._view_dense(b, &c_b)
        cdef cholmod_dense * out
        out = cholmod_solve(system, self._factor, c_b,
                            &self._common._common)
        return _py_dense(out, self._common)
        
def analyze(A, mode="auto"):
    """Computes the optimal fill-reducing permutation for the symmetric matrix
    A, but does *not* factor it (i.e., it performs a "symbolic Cholesky
    decomposition"). This function ignores the actual contents of the matrix
    A. All it cares about are (1) which entries are non-zero, and (2) whether
    A has real or complex type.

    :param A: The matrix to be analyzed.

    :param auto: Specifies which algorithm should be used to (eventually)
      compute the Cholesky decomposition -- one of "simplicial", "supernodal",
      or "auto". See the CHOLMOD documentation for details on how "auto" chooses
      the algorithm to be used.

    :returns: A :class:`Factor` object representing the analysis. Many
      operations on this object will fail, because it does not yet hold a full
      decomposition. Use :meth:`Factor.cholesky_inplace` (or similar) to
      actually factor a matrix.
    """
    return _analyze(A, True, mode=mode)

def analyze_AAt(A, mode="auto"):
    """Computes the optimal fill-reducing permutation for the symmetric matrix
    :math:`AA'`, but does *not* factor it (i.e., it performs a "symbolic
    Cholesky decomposition"). This function ignores the actual contents of the
    matrix A. All it cares about are (1) which entries are non-zero, and (2)
    whether A has real or complex type.

    :param A: The matrix to be analyzed.

    :param auto: Specifies which algorithm should be used to (eventually)
      compute the Cholesky decomposition -- one of "simplicial", "supernodal",
      or "auto". See the CHOLMOD documentation for details on how "auto" chooses
      the algorithm to be used.

    :returns: A :class:`Factor` object representing the analysis. Many
      operations on this object will fail, because it does not yet hold a full
      decomposition. Use :meth:`Factor.cholesky_AAt_inplace` (or similar) to
      actually factor a matrix.
    """
    return _analyze(A, False, mode=mode)

_modes = {
    "simplicial": CHOLMOD_SIMPLICIAL,
    "supernodal": CHOLMOD_SUPERNODAL,
    "auto": CHOLMOD_AUTO,
    }
def _analyze(A, symmetric, mode):
    cdef Common common = Common(issubclass(A.dtype.type, np.complexfloating))
    cdef cholmod_sparse * c_A
    c_A_ref = common._view_sparse(A, symmetric, &c_A)
    if mode in _modes:
        common._common.supernodal = _modes[mode]
    else:
        raise CholmodError, ("Unknown mode '%s', must be one of %s"
                             % (mode, ", ".join(_modes.keys())))
    cdef cholmod_factor * c_f
    c_f = cholmod_analyze(c_A, &common._common)
    if c_f is NULL:
        raise CholmodError, "Error in cholmod_analyze"
    cdef Factor f = Factor(factor_secret_handshake)
    f._common = common
    f._factor = c_f
    return f

def cholesky(A, beta=0, mode="auto"):
    """Computes the fill-reducing Cholesky decomposition of

      .. math:: A + \\beta I

    where ``A`` is a sparse, symmetric, positive-definite matrix, preferably
    in CSC format, and ``beta`` is any real scalar (usually 0 or 1). (And
    :math:`I` denotes the identity matrix.)

    ``mode`` is passed to :func:`analyze`.

    :returns: A :class:`Factor` object represented the decomposition.
    """
    return _cholesky(A, True, beta=beta, mode=mode)

def cholesky_AAt(A, beta=0, mode="auto"):
    """Computes the fill-reducing Cholesky decomposition of

      .. math:: AA' + \\beta I

    where ``A`` is a sparse matrix, preferably in CSC format, and ``beta`` is
    any real scalar (usually 0 or 1). (And :math:`I` denotes the identity
    matrix.)

    Note that if you are solving a conventional least-squares problem, you
    will need to transpose your matrix before calling this function, and
    therefore it will be somewhat more efficient to construct your matrix in
    CSR format (so that its transpose will be in CSC format).

    ``mode`` is passed to :func:`analyze_AAt`.

    :returns: A :class:`Factor` object represented the decomposition.
    """
    return _cholesky(A, False, beta=beta, mode=mode)

def _cholesky(A, symmetric, beta, mode):
    f = _analyze(A, symmetric, mode=mode)
    f._cholesky_inplace(A, symmetric, beta=beta)
    return f

__all__ = ["analyze", "analyze_AAt", "cholesky", "cholesky_AAt"]
