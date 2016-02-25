1) update doc/changes.rst
2) make sure docs are up to date
3) update version in doc/changes.rst, setup.py
4)
```
    git clean -xdn  # to check
    git clean -xdf  # to do it
```
5)
```
    ./release-checklist.sh
```
6) tag
7) announce release on:
        web page
        mailing list
        scipy-dev?
        pypi
8) update version in setup.py again (add "+dev")
