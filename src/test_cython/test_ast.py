#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from time import time
from random import random
from typing import List

PATH_CURR = os.path.dirname(os.path.realpath(__file__))
PATH_BASE = os.path.dirname(PATH_CURR)
sys.path.append(PATH_BASE)

import oppugno as oppu

ALPHA = 2.0
DEFAULT_ARR_SIZE = 1 << 10


def time_func(method):
    def func(*args, **kw):
        start = time()
        result = method(*args, **kw)
        stop = time()
        duration = stop - start
        print("{}: {:.3f}ms".format(method.__name__, duration * 1000))
        return result

    return func


def get_data(size=DEFAULT_ARR_SIZE):
    data = [0] * size
    for i in range(size):
        data[i] = random()
    return data


def get_matrix(rows=DEFAULT_ARR_SIZE, cols=DEFAULT_ARR_SIZE, init=True):
    if init:
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    else:
        data = [[random() for _ in range(cols)] for _ in range(rows)]
    return data


@time_func
@oppu.cuda
def saxpy(size, data_x, data_y, alpha):
    result = [0.0] * size
    for i in range(size):
        result[i] = alpha * data_x[i] + data_y[i]
    return result


@time_func
@oppu.cuda
def matmul(mat_x: oppu.cu_input(dim=2), mat_y: oppu.cu_input(dim=2)):
    rows: oppu.cu_dim = len(mat_x)
    cols: oppu.cu_dim = len(mat_y[0])
    inner: oppu.cu_dim = len(mat_y)
    result: oppu.cu_output(dim=2) = get_matrix(rows, cols, init=True)
    for i in range(rows):
        for j in range(cols):
            for k in range(inner):
                prod_sum = 0.0
                prod_sum += mat_x[i][k] * mat_y[k][j]
                result[i][j] = prod_sum
    return result


def test_matmul(size=DEFAULT_ARR_SIZE):
    mat_x = get_matrix(size, size)
    mat_y = get_matrix(size, size)
    result = matmul(mat_x, mat_y)
    return result


def test_saxpy(size=DEFAULT_ARR_SIZE):
    size = 1 << 2
    alpha = 2.0
    data_x = get_data(size)
    data_y = get_data(size)
    result = saxpy(size, data_x, data_y, alpha)
    return result


def main():
    size = 1 << 2
    # result = test_saxpy(size)
    result = test_matmul(size)
    print(result)


if __name__ == "__main__":
    main()