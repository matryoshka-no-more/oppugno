#!python
#cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp cimport bool
from cpython cimport array
from libc.stdlib cimport malloc, free


cdef extern from ".cuda/cuda.h":
    cdef bool saxpy_cuda(int N, float alpha, float* xarray, float* yarray, float* resultarray)


def saxpy(N, alpha, xarray, yarray, resultarray):
    cdef array.array xarr = array.array("f", xarray)
    cdef array.array yarr = array.array("f", yarray)
    cdef float* rarr = <float *> malloc(N * sizeof(float))
    if not rarr:
        raise MemoryError()
    try:
        rc = saxpy_cuda(N, alpha, xarr.data.as_floats, yarr.data.as_floats, rarr)
        for i in range(N):
            resultarray[i] = rarr[i]
        return rc
    finally:
        free(rarr)
    
    

def say_hello_to(name):
    print("Hello {}!".format(name))