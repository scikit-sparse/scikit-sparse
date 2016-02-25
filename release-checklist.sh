#!/bin/sh
set -e
# Test sdist.
NAME=$(python -c 'from setup import *; print(DISTNAME + "-" + VERSION)')
rm -rf dist
python setup.py sdist
(cd dist && \
 tar -xvzf "$NAME.tar.gz" && \
 cd "$NAME" && \
 python setup.py build_ext -i && py.test && \
 python2 setup.py build_ext -i && py.test2 && \
 cd .. && rm -rf "$NAME")
# Build docs.
(python setup.py build_ext -i && \
 cd doc && \
 make html && \
 cd _build/html && \
 zip -r /tmp/doc.zip .)
echo 'Run `twine upload dist/*`'
echo 'Upload /tmp/doc.zip to pypi.'
