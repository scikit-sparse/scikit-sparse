# Test code for the scikits.sparse CHOLMOD wrapper.

# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import os.path
import warnings
from nose.tools import assert_raises
import numpy as np
from scipy import sparse
from scikits.sparse.cholmod import (cholesky, cholesky_AAt,
                                    analyze, analyze_AAt,
                                    CholmodError)

# At time of writing (scipy 0.7.0), scipy.sparse.csc.csc_matrix explicitly
# uses 32-bit integers for everything (even on 64-bit machines), and therefore
# we wrap the 32-bit version of the cholmod API.  If this test ever fails,
# then it means that scipy.sparse has started using 64-bit integers. That
# wouldn't actually make our code become incorrect in any way -- so if this
# test fails, don't panic! -- but it would make it become *inefficient*, so if
# you see this test fail, then please let us know, and we'll see about
# wrapping the 64-bit API to cholmod.
def test_integer_size():
    m = sparse.eye(10, 10).tocsc()
    assert m.indices.dtype.itemsize == 4
    assert m.indptr.dtype.itemsize == 4

def test_cholesky_smoke_test():
    f = cholesky(sparse.eye(10, 10) * 1.)
    d = np.arange(20).reshape(10, 2)
    print "dense"
    assert np.allclose(f(d), d)
    print "sparse"
    s_csc = sparse.csc_matrix(np.eye(10)[:, :2] * 1.)
    assert sparse.issparse(f(s_csc))
    assert np.allclose(f(s_csc).todense(), s_csc.todense())
    print "csr"
    s_csr = s_csc.tocsr()
    assert sparse.issparse(f(s_csr))
    assert np.allclose(f(s_csr).todense(), s_csr.todense())
    print "extract"
    assert np.all(f.P() == np.arange(10))

def real_matrix():
    return sparse.csc_matrix([[10, 0, 3, 0],
                              [0, 5, 0, -2],
                              [3, 0, 5, 0],
                              [0, -2, 0, 2]])

def complex_matrix():
    return sparse.csc_matrix([[10, 0, 3 - 1j, 0],
                              [0, 5, 0, -2],
                              [3 + 1j, 0, 5, 0],
                              [0, -2, 0, 2]])
    
def factor_of(factor, matrix):
    return np.allclose((factor.L() * factor.L().H).todense(),
                       matrix.todense()[factor.P()[:, np.newaxis],
                                        factor.P()[np.newaxis, :]])

def test_complex():
    c = complex_matrix()
    fc = cholesky(c)
    r = real_matrix()
    fr = cholesky(r)
    
    assert factor_of(fc, c)

    assert np.allclose(fc(np.arange(4)),
                       (c.todense().I * np.arange(4)[:, np.newaxis]).ravel())
    assert np.allclose(fc(np.arange(4) * 1j),
                       (c.todense().I * (np.arange(4) * 1j)[:, np.newaxis]).ravel())
    assert np.allclose(fr(np.arange(4)),
                       (r.todense().I * np.arange(4)[:, np.newaxis]).ravel())
    # If we did a real factorization, we can't do solves on complex arrays:
    assert_raises(CholmodError, fr, np.arange(4) * 1j)

def test_beta():
    for matrix in [real_matrix(), complex_matrix()]:
        for beta in [0, 1, 3.4]:
            matrix_plus_beta = matrix + beta * sparse.eye(*matrix.shape)
            for mode in ["auto", "supernodal", "simplicial"]:
                L = cholesky(matrix, beta=beta).L()
                assert factor_of(cholesky(matrix, beta=beta),
                                 matrix_plus_beta)

def test_update_downdate():
    m = real_matrix()
    f = cholesky(m)
    L = f.L()[f.P(), :]
    assert factor_of(f, m)
    f.update_inplace(L)
    assert factor_of(f, 2 * m)
    f.update_inplace(L)
    assert factor_of(f, 3 * m)
    f.update_inplace(L, subtract=True)
    assert factor_of(f, 2 * m)
    f.update_inplace(L, subtract=True)
    assert factor_of(f, m)

def test_solve_edge_cases():
    m = real_matrix()
    f = cholesky(m)
    # sparse matrices give a sparse back:
    assert sparse.issparse(f(sparse.eye(*m.shape).tocsc()))
    # dense matrices give a dense back:
    assert not sparse.issparse(f(np.eye(*m.shape)))
    # 1d dense matrices are accepted and a 1d vector is returned (this matches
    # the behavior of np.dot):
    assert f(np.arange(m.shape[0])).shape == (m.shape[0],)
    # 2d dense matrices are also accepted:
    assert f(np.arange(m.shape[0])[:, np.newaxis]).shape == (m.shape[0], 1)
    # But not if the dimensions are wrong...:
    assert_raises(CholmodError, f, np.arange(m.shape[0] + 1)[:, np.newaxis])
    assert_raises(CholmodError, f, np.arange(m.shape[0])[np.newaxis, :])
    assert_raises(CholmodError, f, np.arange(m.shape[0])[:, np.newaxis, np.newaxis])
    # And ditto for the sparse version:
    assert_raises(CholmodError, f, sparse.eye(m.shape[0] + 1, m.shape[1]).tocsc())

def mm_matrix(name):
    from scipy.io import mmread
    # Supposedly, it is better to use resource_stream and pass the resulting
    # open file object to mmread()... but for some reason this fails?
    from pkg_resources import resource_filename
    filename = resource_filename(__name__, "test_data/%s.mtx.gz" % name)
    matrix = mmread(filename)
    if sparse.issparse(matrix):
        matrix = matrix.tocsc()
    return matrix

def test_cholesky_matrix_market():
    for problem in ("well1033", "illc1033", "well1850", "illc1850"):
        X = mm_matrix(problem)
        y = mm_matrix(problem + "_rhs1")
        answer = np.linalg.lstsq(X.todense(), y)[0]
        XtX = (X.T * X).tocsc()
        Xty = X.T * y
        for mode in ("auto", "simplicial", "supernodal"):
            assert np.allclose(cholesky(XtX, mode=mode)(Xty), answer)
            assert np.allclose(cholesky_AAt(X.T, mode=mode)(Xty), answer)
            assert np.allclose(cholesky(XtX, mode=mode).solve_A(Xty), answer)
            assert np.allclose(cholesky_AAt(X.T, mode=mode).solve_A(Xty), answer)

            f1 = analyze(XtX, mode=mode)
            f2 = f1.cholesky(XtX)
            assert np.allclose(f2(Xty), answer)
            assert_raises(CholmodError, f1, Xty)
            assert_raises(CholmodError, f1.solve_A, Xty)
            assert_raises(CholmodError, f1.solve_LDLt, Xty)
            assert_raises(CholmodError, f1.solve_LD, Xty)
            assert_raises(CholmodError, f1.solve_DLt, Xty)
            assert_raises(CholmodError, f1.solve_L, Xty)
            assert_raises(CholmodError, f1.solve_D, Xty)
            assert_raises(CholmodError, f1.apply_P, Xty)
            assert_raises(CholmodError, f1.apply_Pt, Xty)
            f1.P()
            assert_raises(CholmodError, f1.L)
            assert_raises(CholmodError, f1.LD)
            assert_raises(CholmodError, f1.L_D)
            assert_raises(CholmodError, f1.L_D)
            f1.cholesky_inplace(XtX)
            assert np.allclose(f1(Xty), answer)

            f3 = analyze_AAt(X.T, mode=mode)
            f4 = f3.cholesky(XtX)
            assert np.allclose(f4(Xty), answer)
            assert_raises(CholmodError, f3, Xty)
            f3.cholesky_AAt_inplace(X.T)
            assert np.allclose(f3(Xty), answer)

            print problem, mode
            for f in (f1, f2, f3, f4):
                pXtX = XtX.todense()[f.P()[:, np.newaxis],
                                     f.P()[np.newaxis, :]]
                assert np.allclose(np.prod(f.D()),
                                   np.linalg.det(XtX.todense()))
                assert np.allclose((f.L() * f.L().T).todense(),
                                   pXtX)
                L, D = f.L_D()
                assert np.allclose((L * D * L.T).todense(),
                                   pXtX)

                b = np.arange(XtX.shape[0])[:, np.newaxis]
                assert np.allclose(f.solve_A(b),
                                   np.dot(XtX.todense().I, b))
                assert np.allclose(f(b),
                                   np.dot(XtX.todense().I, b))
                assert np.allclose(f.solve_LDLt(b),
                                   np.dot((L * D * L.T).todense().I, b))
                assert np.allclose(f.solve_LD(b),
                                   np.dot((L * D).todense().I, b))
                assert np.allclose(f.solve_DLt(b),
                                   np.dot((D * L.T).todense().I, b))
                assert np.allclose(f.solve_L(b),
                                   np.dot(L.todense().I, b))
                assert np.allclose(f.solve_Lt(b),
                                   np.dot(L.T.todense().I, b))
                assert np.allclose(f.solve_D(b),
                                   np.dot(D.todense().I, b))

                assert np.allclose(f.apply_P(b), b[f.P(), :])
                assert np.allclose(f.solve_P(b), b[f.P(), :])
                # Pt is the inverse of P, and argsort inverts permutation
                # vectors:
                assert np.allclose(f.apply_Pt(b), b[np.argsort(f.P()), :])
                assert np.allclose(f.solve_Pt(b), b[np.argsort(f.P()), :])

def test_deprecation():
    f = cholesky(sparse.eye(5, 5))
    b = np.ones(5)
    for dep_method in "solve_P", "solve_Pt":
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            getattr(f, dep_method)(b)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "deprecated" in str(w[-1].message)

def test_convenience():
    A_dense_seed = np.array([[10, 0, 3, 0],
                             [0, 5, 0, -2],
                             [3, 0, 5, 0],
                             [0, -2, 0, 2]])
    for mode in ("simplicial", "supernodal"):
        for dtype in (float, complex):
            A_dense = np.array(A_dense_seed, dtype=dtype)
            A_sp = sparse.csc_matrix(A_dense)
            f = cholesky(A_sp, mode=mode)
            assert np.allclose(f.det(), np.linalg.det(A_dense))
            assert np.allclose(f.logdet(), np.log(np.linalg.det(A_dense)))
            assert np.allclose(f.slogdet(), [1, np.log(np.linalg.det(A_dense))])
            assert np.allclose((f.inv() * A_sp).todense(), np.eye(4))
