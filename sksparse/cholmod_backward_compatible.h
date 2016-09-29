#include "cholmod.h"

// CHOLMOD_GPU_PROBLEM is only defined since version 2.0 of cholmod,
// so we need to define it here for backward compatibility
#ifndef CHOLMOD_GPU_PROBLEM
    #define CHOLMOD_GPU_PROBLEM -5
#endif

// SuiteSparse_long is only defined since version 4.0 of cholmod,
// previously its name was UF_long,
// so we need to define it here for backward compatibility
#ifndef SuiteSparse_long
    #ifdef UF_long
        #define SuiteSparse_long UF_long
    #else
        #ifdef _WIN64
            #define SuiteSparse_long __int64
        #else
            #define SuiteSparse_long long
        #endif
    #endif
#endif
