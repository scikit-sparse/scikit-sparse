# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Released under the terms of the GNU GPL v2, or, at your option, any
# later version.

import os.path
from nose.tools import assert_raises
import numpy as np
from scipy import sparse
from cholmod import cholesky, cholesky_AAt, analyze, analyze_AAt, CholmodError

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
                       c.todense().I * np.arange(4)[:, np.newaxis])
    assert np.allclose(fc(np.arange(4) * 1j),
                       c.todense().I * (np.arange(4) * 1j)[:, np.newaxis])
    assert np.allclose(fr(np.arange(4)),
                       r.todense().I * np.arange(4)[:, np.newaxis])
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
    # 1d dense matrices are accepted and treated as column vectors:
    assert f(np.arange(m.shape[0])).shape == (m.shape[0], 1)
    # 2d dense matrices are also accepted:
    assert f(np.arange(m.shape[0])[:, np.newaxis]).shape == (m.shape[0], 1)
    # But not if the dimensions are wrong...:
    assert_raises(CholmodError, f, np.arange(m.shape[0] + 1)[:, np.newaxis])
    assert_raises(CholmodError, f, np.arange(m.shape[0])[np.newaxis, :])
    assert_raises(CholmodError, f, np.arange(m.shape[0])[:, np.newaxis, np.newaxis])
    # And ditto for the sparse version:
    assert_raises(CholmodError, f, sparse.eye(m.shape[0] + 1, m.shape[1]).tocsc())

def test_cholesky_matrix_market():
    from scipy.io import mmread
    data_dir = os.path.join(os.path.split(__file__)[0], "test_data")
    for problem in ("well1033", "illc1033", "well1850", "illc1850"):
        X = mmread(os.path.join(data_dir, problem + ".mtx.gz")).tocsc()
        y = mmread(os.path.join(data_dir, problem + "_rhs1.mtx.gz"))
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
            assert_raises(CholmodError, f1.solve_P, Xty)
            assert_raises(CholmodError, f1.solve_Pt, Xty)
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
                assert np.allclose(f.solve_P(b), b[f.P(), :])
                # Pt is the inverse of P, and argsort inverts permutation
                # vectors:
                assert np.allclose(f.solve_Pt(b), b[np.argsort(f.P()), :])