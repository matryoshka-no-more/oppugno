#!python
#cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp cimport bool
from cpython cimport array
from libc.stdlib cimport malloc, free


cdef extern from ".cuda/cuda.h":
    cdef bool saxpy_cuda(int N, float alpha, float* xarray, float* yarray, float* resultarray)
    cdef bool matmul_cuda(int rows, int cols, int inner, float* xarray, float* yarray, float* resultarray)

# def saxpy(N, alpha, xarray, yarray, resultarray):
#     cdef np.ndarray[np.float32_t, ndim=1, mode="c"] xarr = np.array(xarray, dtype=np.float32)
#     cdef np.ndarray[np.float32_t, ndim=1, mode="c"] yarr = np.array(yarray, dtype=np.float32)
#     cdef float* rarr = <float *> malloc(N * sizeof(float))
#     if not rarr:
#         raise MemoryError()
#     try:
#         rc = saxpy_cuda(N, alpha, &xarr[0], &yarr[0], rarr)
#         for i in range(N):
#             resultarray[i] = rarr[i]
#         return rc
#     finally:
#         free(rarr)


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


def matmul(rows, cols, inner, xarray, yarray, resultarray):
    cdef np.float32_t [:] xarr = np.array(xarray, dtype=np.float32).flatten()
    cdef np.float32_t [:] yarr = np.array(yarray, dtype=np.float32).flatten()
    cdef float* rarr = <float *> malloc(rows * cols * sizeof(float))
    if not rarr:
        raise MemoryError()
    try:
        rc = matmul_cuda(rows, cols, inner, &xarr[0], &yarr[0], rarr)
        for i in range(rows):
            for j in range(cols):
                resultarray[i][j] = rarr[i * cols + j]
        return rc
    finally:
        free(rarr)