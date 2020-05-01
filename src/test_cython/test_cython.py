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
from oppugno_cuda import saxpy

PATH_CURR = os.path.dirname(os.path.realpath(__file__))
PATH_BASE = os.path.dirname(PATH_CURR)
sys.path.append(PATH_BASE)

from oppugno.cuda import Cuda

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


def get_data(size=DEFAULT_ARR_SIZE):
    data = [0] * size
    for i in range(size):
        data[i] = random()
    return data


@time_func
def saxpy_cpu(data_x, data_y, alpha=ALPHA):
    size = len(data_x)
    result = [0.0] * size
    for i in range(size):
        result[i] = alpha * data_x[i] + data_y[i]
    return result


@time_func
def saxpy_gpu(data_x, data_y, alpha=ALPHA):
    size = len(data_x)
    result = [0.0] * size
    saxpy(size, alpha, data_x, data_y, result)
    return result


def compare(N=DEFAULT_ARR_SIZE):
    print("========== compare array size: {} ==========".format(N))
    data_x = get_data(N)
    data_y = get_data(N)
    time_cpu, result_cpu = saxpy_cpu(data_x, data_y)
    time_gpu, result_gpu = saxpy_gpu(data_x, data_y)

    for i in range(len(result_cpu)):
        if result_cpu[i] != result_gpu[i]:
            print("Failed: [{}]\t(cpu){:4f} != {:4f}(gpu) diff = {}".format(
                i, result_cpu[i], result_gpu[i],
                result_cpu[i] - result_gpu[i]))
            return time_cpu, time_gpu
    print("Success: the results from cpu and gpu are consistent")
    return time_cpu, time_gpu


def main():
    t = PrettyTable(["Array Size", "GPU speedup"])
    t.align["Array Size"] = "l"
    for i in range(2, 21):
        N = 1 << i
        time_cpu, time_gpu = compare(N)
        t.add_row([N, "{:4f}".format(time_cpu / time_gpu)])

    print("\n========== Summary ==========\n")
    print(t)


if __name__ == "__main__":
    main()