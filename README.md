![Python27](https://img.shields.io/badge/python-2.7-blue.svg)
![Python35](https://img.shields.io/badge/python-3.5-blue.svg)
![Python36](https://img.shields.io/badge/python-3.6-blue.svg)
[![Documentation Status](https://readthedocs.org/projects/scikit-sparse/badge/?version=latest)](http://scikit-sparse.readthedocs.io/en/latest/?badge=latest)
[![Travis](https://travis-ci.org/scikit-sparse/scikit-sparse.svg?branch=master)](https://travis-ci.org/scikit-sparse/scikit-sparse)

This is scikit-sparse, a companion to the scipy.sparse library for
sparse matrix manipulation in Python. It provides routines that are
not suitable for inclusion in scipy.sparse proper, usually because
they are GPL'ed.

NOTE:  This library is solid and works well, but no longer actively
developed. Please let us know if you wish to take over development.

So far it just contains a wrapper for the CHOLMOD library for sparse
Cholesky decomposition. Further contributions are welcome!

For more details, including dependencies and installation
instructions, see the [docs](https://scikit-sparse.readthedocs.org).

License
-------

The wrapper code contained in this package is released under a
2-clause BSD license, as per below. However, this applies only to the
original code contained in this package, and NOT to the libraries
(e.g., CHOLMOD) which it uses. These libraries are generally
licensed under less permissive licenses, such as the GNU GPL or LGPL,
and users of this package are responsible for determining what
requirements these licenses impose on their usage. (The intent here is
that if you, for example, buy a license to use CHOLMOD in a commercial
product, then you can also go ahead and use our wrapper code with your
commercial license.)

Copyright (c) 2009-2017, the [scikit-sparse developers](https://scikit-sparse.readthedocs.io/en/latest/overview.html#developers)

    scikits-sparse
    Copyright (c) 2009-2017, the scikit-sparse developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials
      provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
