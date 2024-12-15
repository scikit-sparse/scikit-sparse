# Test code for the scikits.sparse CHOLMOD wrapper.

# Copyright (C) 2008-2017 The scikit-sparse developers:
#
# 2008        David Cournapeau        <cournape@gmail.com>
# 2009-2015   Nathaniel Smith         <njs@pobox.com>
# 2010        Dag Sverre Seljebotn    <dagss@student.matnat.uio.no>
# 2014        Leon Barrett            <lbarrett@climate.com>
# 2015        Yuri                    <yuri@tsoft.com>
# 2016-2017   Antony Lee              <anntzer.lee@gmail.com>
# 2016        Alex Grigorievskiy      <alex.grigorievskiy@gmail.com>
# 2016-2017   Joscha Reimer           <jor@informatik.uni-kiel.de>
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

from functools import partial
import os.path

from pytest import raises as assert_raises
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sksparse.cholmod import (
    cholesky,
    cholesky_AAt,
    analyze,
    analyze_AAt,
    CholmodError,
    CholmodNotPositiveDefiniteError,
    _modes,
    _ordering_methods,
)

modes = tuple(_modes.keys())
ordering_methods = tuple(_ordering_methods.keys())

# Match defaults of np.allclose, which were used before (and are needed).
assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-8)


def test_cholesky_smoke_test():
    f = cholesky(sparse.eye(10, 10))
    d = np.arange(20).reshape(10, 2)
    print("dense")
    assert_allclose(f(d), d)
    print("sparse")
    s_csc = sparse.csc_matrix(np.eye(10)[:, :2])
    assert sparse.issparse(f(s_csc))
    assert_allclose(f(s_csc).todense(), s_csc.todense())
    print("csc_array")
    sa_csc = sparse.csc_array(s_csc)
    assert sparse.issparse(f(sa_csc))
    assert_allclose(f(sa_csc).todense(), sa_csc.todense())
    print("csr")
    s_csr = s_csc.tocsr()
    assert sparse.issparse(f(s_csr))
    assert_allclose(f(s_csr).todense(), s_csr.todense())
    print("extract")
    assert np.all(f.P() == np.arange(10))


def test_writeability():
    t = cholesky(sparse.eye(10, 10))(np.arange(10))
    assert t.flags["WRITEABLE"]


def real_matrix():
    return sparse.csc_matrix([[10, 0, 3, 0], [0, 5, 0, -2], [3, 0, 5, 0], [0, -2, 0, 2]])


def complex_matrix():
    return sparse.csc_matrix([[10, 0, 3 - 1j, 0], [0, 5, 0, -2], [3 + 1j, 0, 5, 0], [0, -2, 0, 2]])


def factor_of(factor, matrix):
    return np.allclose(
        (factor.L() * factor.L().T.conjugate()).todense(), matrix.todense()[factor.P()[:, np.newaxis], factor.P()[np.newaxis, :]]
    )


def convert_matrix_indices_to_long_indices(matrix):
    matrix.indices = np.asarray(matrix.indices, dtype=np.int64)
    matrix.indptr = np.asarray(matrix.indptr, dtype=np.int64)
    return matrix


def test_complex():
    c = complex_matrix()
    fc = cholesky(c)
    r = real_matrix()
    fr = cholesky(r)

    assert factor_of(fc, c)

    assert_allclose(fc(np.arange(4))[:, None], c.todense().I * np.arange(4)[:, None])
    assert_allclose(fc(np.arange(4) * 1j)[:, None], c.todense().I * (np.arange(4) * 1j)[:, None])
    assert_allclose(fr(np.arange(4))[:, None], r.todense().I * np.arange(4)[:, None])
    # If we did a real factorization, we can't do solves on complex arrays:
    assert_raises(CholmodError, fr, np.arange(4) * 1j)


def test_beta():
    for matrix in [real_matrix(), complex_matrix()]:
        for beta in [0, 1, 3.4]:
            matrix_plus_beta = matrix + beta * sparse.eye(*matrix.shape)
            for use_long in [False, True]:
                if use_long:
                    matrix_plus_beta = convert_matrix_indices_to_long_indices(matrix_plus_beta)
                for ordering_method in ordering_methods:
                    for mode in modes:
                        f = cholesky(matrix, beta=beta, mode=mode, ordering_method=ordering_method)
                        L = f.L()
                        assert factor_of(f, matrix_plus_beta)


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
        for mode in modes:
            assert_allclose(cholesky(XtX, mode=mode)(Xty), answer)
            assert_allclose(cholesky_AAt(X.T, mode=mode)(Xty), answer)
            assert_allclose(cholesky(XtX, mode=mode).solve_A(Xty), answer)
            assert_allclose(cholesky_AAt(X.T, mode=mode).solve_A(Xty), answer)

            f1 = analyze(XtX, mode=mode)
            f2 = f1.cholesky(XtX)
            assert_allclose(f2(Xty), answer)
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
            assert_allclose(f1(Xty), answer)

            f3 = analyze_AAt(X.T, mode=mode)
            f4 = f3.cholesky(XtX)
            assert_allclose(f4(Xty), answer)
            assert_raises(CholmodError, f3, Xty)
            f3.cholesky_AAt_inplace(X.T)
            assert_allclose(f3(Xty), answer)

            print(problem, mode)
            for f in (f1, f2, f3, f4):
                pXtX = XtX.todense()[f.P()[:, np.newaxis], f.P()[np.newaxis, :]]
                assert_allclose(np.prod(f.D()), np.linalg.det(XtX.todense()))
                assert_allclose((f.L() * f.L().T).todense(), pXtX)
                L, D = f.L_D()
                assert_allclose((L * D * L.T).todense(), pXtX)

                b = np.arange(XtX.shape[0])[:, np.newaxis]
                assert_allclose(f.solve_A(b), np.dot(XtX.todense().I, b))
                assert_allclose(f(b), np.dot(XtX.todense().I, b))
                assert_allclose(f.solve_LDLt(b), np.dot((L * D * L.T).todense().I, b))
                assert_allclose(f.solve_LD(b), np.dot((L * D).todense().I, b))
                assert_allclose(f.solve_DLt(b), np.dot((D * L.T).todense().I, b))
                assert_allclose(f.solve_L(b), np.dot(L.todense().I, b))
                assert_allclose(f.solve_Lt(b), np.dot(L.T.todense().I, b))
                assert_allclose(f.solve_D(b), np.dot(D.todense().I, b))

                assert_allclose(f.apply_P(b), b[f.P(), :])
                assert_allclose(f.apply_P(b), b[f.P(), :])
                # Pt is the inverse of P, and argsort inverts permutation
                # vectors:
                assert_allclose(f.apply_Pt(b), b[np.argsort(f.P()), :])
                assert_allclose(f.apply_Pt(b), b[np.argsort(f.P()), :])


def test_convenience():
    A_dense_seed = np.array([[10, 0, 3, 0], [0, 5, 0, -2], [3, 0, 5, 0], [0, -2, 0, 2]])
    for dtype in (float, complex):
        A_dense = np.array(A_dense_seed, dtype=dtype)
        A_sp = sparse.csc_matrix(A_dense)
        for use_long in [False, True]:
            if use_long:
                A_sp = convert_matrix_indices_to_long_indices(A_sp)
            for ordering_method in ordering_methods:
                for mode in modes:
                    print("----")
                    print(dtype)
                    print(A_sp.indices.dtype)
                    print(use_long)
                    print(ordering_method)
                    print(mode)
                    print("----")
                    f = cholesky(A_sp, mode=mode, ordering_method=ordering_method)
                    print(f.D())
                    assert_allclose(f.det(), np.linalg.det(A_dense))
                    assert_allclose(f.logdet(), np.log(np.linalg.det(A_dense)))
                    assert_allclose(f.slogdet(), [1, np.log(np.linalg.det(A_dense))])
                    assert_allclose((f.inv() * A_sp).todense(), np.eye(4))


def test_CholmodNotPositiveDefiniteError():
    A = -sparse.eye(4).tocsc()
    f = cholesky(A)
    assert_raises(CholmodNotPositiveDefiniteError, f.L)


def test_natural_ordering_method():
    A = real_matrix()
    f = cholesky(A, ordering_method="natural")
    p = f.P()
    assert_array_equal(p, np.arange(len(p)))
