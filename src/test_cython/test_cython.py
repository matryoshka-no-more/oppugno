#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import ast
import shlex
import inspect
import subprocess
from random import random
from time import time
from prettytable import PrettyTable
from oppugno_cuda import saxpy, matmul

PATH_CURR = os.path.dirname(os.path.realpath(__file__))
PATH_BASE = os.path.dirname(PATH_CURR)
sys.path.append(PATH_BASE)

from oppugno.cuda import Cuda

TABLE_HEADERS = ["Array Size", "CPU Time", "GPU Time", "GPU speedup"]
DEFAULT_ARR_SIZE = 1 << 10
ALPHA = 2.0
execname = os.path.join(os.getcwd(), "main")
filename = os.path.join(os.getcwd(), "main.cu")


def time_func(method):
    def func(*args, **kw):
        start = time()
        result = method(*args, **kw)
        stop = time()
        duration = stop - start
        print("{}: {:.3f}ms".format(method.__name__, duration * 1000))
        return duration, result

    return func


def get_data(size=DEFAULT_ARR_SIZE, init=True):
    if init:
        data = [0.0 for _ in range(size)]
    else:
        data = [random() for _ in range(size)]
    return data


def get_matrix(rows=DEFAULT_ARR_SIZE, cols=DEFAULT_ARR_SIZE, init=True):
    if init:
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    else:
        data = [[random() for _ in range(cols)] for _ in range(rows)]
    return data


@time_func
def saxpy_cpu(data_x, data_y, alpha=ALPHA):
    size = len(data_x)
    result = [0.0] * size
    for i in range(size):
        for _ in range(size):
            result[i] = alpha * data_x[i] + data_y[i]
    return result


@time_func
def saxpy_gpu(data_x, data_y, alpha=ALPHA):
    size = len(data_x)
    result = [0.0] * size
    saxpy(size, alpha, data_x, data_y, result)
    return result


@time_func
def matmul_gpu(mat_x, mat_y):
    rows = len(mat_x)
    cols = 0 if not mat_y else len(mat_y[0])
    inner = len(mat_y)
    result = get_matrix(rows, cols, init=True)
    matmul(rows, cols, inner, mat_x, mat_y, result)
    return result


@time_func
def matmul_cpu(mat_x, mat_y):
    rows = len(mat_x)
    cols = 0 if not mat_y else len(mat_y[0])
    inner = len(mat_y)
    result = get_matrix(rows, cols, init=True)
    for i in range(rows):
        for j in range(cols):
            # oppu.kernel
            for k in range(inner):
                result[i][j] += mat_x[i][k] * mat_y[k][j]
    return result


def compare_saxpy(N=DEFAULT_ARR_SIZE):
    print("========== compare array size: {} ==========".format(N))
    data_x = get_data(N)
    data_y = get_data(N)
    time_cpu, result_cpu = saxpy_cpu(data_x, data_y)
    time_gpu, result_gpu = saxpy_gpu(data_x, data_y)

    for i in range(len(result_cpu)):
        if result_cpu[i] - result_gpu[i] > 0.000001:
            print("Failed: [{}]\t(cpu){:4f} != {:4f}(gpu) diff = {}".format(
                i, result_cpu[i], result_gpu[i],
                result_cpu[i] - result_gpu[i]))
            return time_cpu, time_gpu
    print("Success: the results from cpu and gpu are consistent")
    return time_cpu, time_gpu


def compare_matmul(N=DEFAULT_ARR_SIZE):
    print("========== compare array size: {} ==========".format(N))
    mat_x = get_matrix(N, N)
    mat_y = get_matrix(N, N)
    time_cpu, result_cpu = matmul_cpu(mat_x, mat_y)
    time_gpu, result_gpu = matmul_gpu(mat_x, mat_y)

    for i in range(len(result_cpu)):
        for j in range(len(result_cpu[0])):
            if result_cpu[i][j] - result_gpu[i][j] > 0.000001:
                print(
                    "Failed: [{}]\t(cpu){:4f} != {:4f}(gpu) diff = {}".format(
                        i, result_cpu[i][j], result_gpu[i][j],
                        result_cpu[i][j] - result_gpu[i][j]))
                return time_cpu, time_gpu
    print("Success: the results from cpu and gpu are consistent")
    return time_cpu, time_gpu


def main():
    t = PrettyTable(TABLE_HEADERS)
    t.align = "r"
    t.align[TABLE_HEADERS[0]] = "l"
    for i in range(1, 11):
        N = 1 << i
        time_cpu, time_gpu = compare_matmul(N)
        t.add_row([N, time_cpu, time_gpu, "{:4f}".format(time_cpu / time_gpu)])

    print("\n========== Summary ==========\n")
    print(t)


if __name__ == "__main__":
    main()