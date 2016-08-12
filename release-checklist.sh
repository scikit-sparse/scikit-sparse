#!/bin/sh
set -e
# Cleanup.
git clean -xdi
NAME=$(python -c 'from setup import *; print(DISTNAME + "-" + versioneer.get_version())')
# Test docs.
(python setup.py build_ext -i &&
    cd doc &&
    make html)
# Test.
python setup.py sdist
(cd dist &&
    tar -xvzf "$NAME.tar.gz" &&
    cd "$NAME" &&
    python setup.py build_ext -i && py.test &&
    python2 setup.py build_ext -i && py.test2 &&
    cd .. && rm -rf "$NAME")
echo 'Run `twine upload dist/*`'
