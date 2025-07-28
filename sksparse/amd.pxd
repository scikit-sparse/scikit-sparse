# AMD wrapper for sksparse

# distutils: language = c
# cython: language_level=3

cdef extern from "amd.h":
    int AMD_CONTROL
    int AMD_INFO

    int AMD_DENSE
    int AMD_AGGRESSIVE

    int AMD_OUT_OF_MEMORY
    int AMD_INVALID

    # 32-bit AMD interface
    int amd_order(
        int n,
        int* Ap,
        int* Ai,
        int* P,
        double* Control,
        double* Info
    )
    void amd_defaults(double* Control)

    # 64-bit AMD interface
    int amd_l_order(
        long long n,
        long long* Ap,
        long long* Ai,
        long long* P,
        double* Control,
        double* Info
    )
    void amd_l_defaults(double* Control)
    void amd_l_control(double* Control)
