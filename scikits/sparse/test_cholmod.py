# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Released under the terms of the GNU GPL v2, or, at your option, any
# later version.

import numpy as np
from scipy import sparse
from cholmod import cholesky

def test_cholesky():
    f = cholesky(sparse.eye(10, 10) * 1.)
    d = np.arange(20).reshape(10, 2)
    print "dense"
    assert np.allclose(f(d), d)
    print "sparse"
    s_csc = sparse.csc_matrix(np.eye(10)[:, :2] * 1.)
    assert sparse.issparse(f(s_csc))
    assert np.allclose(f(s_csc).todense(), s_csc.todense())
    print "csr"
    s_csr = sparse.csr_matrix(np.eye(10)[:, :2] * 1.)
    assert sparse.issparse(f(s_csr))
    assert np.allclose(f(s_csr).todense(), s_csr.todense())
    print "extract"
    assert np.all(f.P() == np.arange(10))
