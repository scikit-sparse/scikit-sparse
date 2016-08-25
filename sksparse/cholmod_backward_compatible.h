#include "cholmod.h"

#if CHOLMOD_MAIN_VERSION < 2
#define CHOLMOD_GPU_PROBLEM -5
#endif