[![GitHub release (latest by date)](https://img.shields.io/github/v/release/scikit-sparse/scikit-sparse)](https://github.com/scikit-sparse/scikit-sparse/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/scikit-sparse)](https://pypi.org/project/scikit-sparse/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-sparse.svg)](https://anaconda.org/conda-forge/scikit-sparse)
[![GitHub Workflow Status (event)](https://img.shields.io/github/workflow/status/scikit-sparse/scikit-sparse/CI%20targets?label=CI%20Tests)](https://github.com/scikit-sparse/scikit-sparse/actions/workflows/ci_test.yml) 
[![Python Versions](https://img.shields.io/badge/python-3.6%2C%203.7%2C%203.8%2C%203.9-blue.svg)]()
[![GitHub license](https://img.shields.io/github/license/scikit-sparse/scikit-sparse)](https://github.com/scikit-sparse/scikit-sparse/blob/master/LICENSE.txt)

# scikit-sparse

This `scikit-sparse` a companion to the scipy.sparse library for
sparse matrix manipulation in Python. It provides routines that are
not suitable for inclusion in scipy.sparse proper, usually because
they are GPL'ed.

For more details on usage see the [docs](https://scikit-sparse.readthedocs.org).

## Installation

### With `pip`

For pip installs  of `scikit-sparse` depend on the suite-sparse library which can be installed via:
```bash
# mac
brew install suite-sparse

# debian
sudo apt-get install libsuitesparse-dev
```

Then, `scikit-sparse` can be installed via pip:
```bash
pip install scikit-sparse
```

### With `conda`
The `conda` package comes with `suite-sparse` packaged as a dependency so all you need to do is:

```bash
conda install -c conda-forge scikit-sparse
```


## License

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
