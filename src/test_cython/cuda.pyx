#!python
#cython: language_level=3

from libcpp cimport bool
from cpython cimport array

cdef extern from ".cuda/cuda.h":
    cdef bool saxpy_cuda(int N, float alpha, float* xarray, float* yarray, float* resultarray)

def saxpy(N, alpha, xarray, yarray, resultarray):
    cdef array.array xarr = array.array("f", xarray)
    cdef array.array yarr = array.array("f", yarray)
    cdef array.array rarr = array.array("f", resultarray)
    rc = saxpy_cuda(N, alpha, xarr.data.as_floats, yarr.data.as_floats, rarr.data.as_floats)
    for i in range(N):
        resultarray[i] = rarr.data.as_floats[i]
    return rc
    
def say_hello_to(name):
    print("Hello {}!".format(name))