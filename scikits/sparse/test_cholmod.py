# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Released under the terms of the GNU GPL v2, or, at your option, any
# later version.

import numpy as np
from scipy import sparse
from cholmod import cholesky

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
